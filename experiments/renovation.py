"""
E5, the renovation of the multifunction hall as a hold-out test (paper Fig. 8 and Table 4; docs/experiment-plan.md).

    uv run python experiments/renovation.py [--seeds 5] [--rays-per-m3 100] [--data ../data]

The model of the hall has no seats (280 in the room), and with the documented inputs it overestimates the
measured T30 before the renovation by 17-107 % (E1). The question is what such a model can still say about
a treatment. Three ways of predicting the hall after the renovation are compared. Each is fixed here, uses
the measurement before the renovation at most, and never the measurement after it:

    documented   the documented inputs with the renovation materials; nothing is corrected
    anchored     the simulation supplies only the change, the measurement before the renovation the level:
                 1 / T_after = 1 / T_before,measured + (1 / T_after,sim - 1 / T_before,sim), per band. In a
                 diffuse field 1 / T is proportional to the absorption area, so absorption that is missing
                 in both simulated states (the seats) cancels
    calibrated   per band, one absorption area is added on the floor until the simulation reproduces the
                 measurement before the renovation; the renovation materials are then applied to that model

One measured T30 per band identifies one unknown per band, whatever is missing (seats, glazing at low
frequencies, scattering): the calibration therefore lumps everything into the floor and does not claim to
find the seats. calibration.csv lists the absorption area that is missing and the coefficient it would
mean on the seating area, next to upholstered seats from the literature, as a plausibility check.

The treated area is taken in two ways: by layer, as in the thesis (all perforated and upholstered panels,
configs/mf_hall_renovated.toml), and as the paper lists it (walls A, C and D; the panels on the glazed
wall B keep their old coefficients). The ceiling is treated in both.

The predictions are compared with both after-renovation data sets, whose difference is not resolved
(data/README.md): the room mean of the paper and the B&K export of 20 December 2022 (5 valid positions).
The measured values of the hall are 1/3-octave values at the octave centre frequencies.

Writes renovation.csv, calibration.csv, renovation.png and run_info.json to --output.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.air import speed_of_sound
from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config
from gasim.geometry import load_room

from _common import (
    BAND_LABELS,
    COLOURS,
    cell,
    material_spec,
    measured_statistics,
    read_receiver_table,
    room_value,
    run_seeds,
    with_materials,
    write_csv,
    write_run_info,
)

BEFORE, AFTER = 'configs/mf_hall_paper.toml', 'configs/mf_hall_renovated.toml'
FLOOR_LAYER = 'Floor'
PANEL_LAYERS = ('Wood', 'Woolen')
UNTREATED_SURFACE = 'wall_B'  # the glazed wall, not among the treated surfaces of the paper
SEATING_AREA = 168.0  # m2, 280 seats with a 0.5 m edge strip (as in experiments/scenarios.py)
CLOTH_CHAIRS = [0.44, 0.60, 0.77, 0.89, 0.82, 0.70]  # Vorlaender 2020, Annex p. 336: empty upholstered chairs
MID_BAND_LIMITS = (0.9, 1.2)  # s at 500 Hz - 1 kHz for this volume, GB 50356-2005 (thesis Sect. 5.1)
CALIBRATION_TOLERANCE = 0.01  # relative deviation of T30 from the measurement before the renovation
CALIBRATION_STEPS = 6
TREATMENTS = {'by layer (thesis)': False, 'walls A, C, D (paper)': True}
PROTOCOLS = ('documented', 'anchored', 'calibrated')


def sabine_constant(config):
    """
    :return: 24 ln(10) / c, so that A = constant x V / T, in s/m
    """
    return 24 * math.log(10) / float(speed_of_sound(config.air.temperature))


def anchored(before_measured, before_simulated, after_simulated):
    """
    Prediction that takes the level from a measurement and only the change from the simulation
    :return: 1 / (1 / before_measured + 1 / after_simulated - 1 / before_simulated); nan where the simulated
             change would remove more absorption than the measured room has
    """
    rate = 1 / np.asarray(before_measured) + 1 / np.asarray(after_simulated) - 1 / np.asarray(before_simulated)
    return np.where(rate > 0, 1 / np.where(rate > 0, rate, 1.0), np.nan)


def missing_area(measured, simulated, volume: float, constant: float):
    """
    :return: Absorption area that a diffuse field would need in addition to turn the simulated into the
             measured reverberation time, m2 (negative: the model absorbs too much)
    """
    return constant * volume * (1 / np.asarray(measured) - 1 / np.asarray(simulated))


def calibration_step(alpha, measured, simulated, volume: float, area: float, constant: float):
    """
    One update of the lumped absorption coefficient
    :param alpha: (B,) current coefficient on the calibration layer
    :param area: Area of the calibration layer, m2
    :return: (B,) new coefficient, limited to [0, 0.99]
    """
    return np.clip(np.asarray(alpha) + missing_area(measured, simulated, volume, constant) / area, 0.0, 0.99)


def renovated(before, after, room, exclude_untreated: bool):
    """
    Materials after the renovation on the model 'before' (which may carry a calibrated floor)
    :param before, after: Configs before and after the renovation (same model and layers)
    :param exclude_untreated: The panels on UNTREATED_SURFACE keep their coefficients: each panel layer gets
                              the area-weighted mean of old and new
    :return: Config
    """
    old, new = material_spec(before), material_spec(after)
    spec = {**old}
    surface = room.surface_names.index(UNTREATED_SURFACE)
    for layer in new:
        if layer == FLOOR_LAYER or new[layer]['absorption'] == material_spec(load_config(BEFORE))[layer]['absorption']:
            continue  # not treated; keeps the (possibly calibrated) coefficients of 'before'
        share = 0.0
        if exclude_untreated and layer in PANEL_LAYERS:
            kept = sum(area for area, name, s in zip(room.poly_area, room.poly_layer, room.poly_surface)
                       if name == layer and s == surface)
            share = kept / room.area_by_material()[layer]
        mixed = np.asarray(new[layer]['absorption']) + share * (np.asarray(old[layer]['absorption'])
                                                               - np.asarray(new[layer]['absorption']))
        spec[layer] = {**old[layer], 'absorption': mixed.tolist()}
    return with_materials(before, spec)


def calibrate(config, measured, room, seeds: int, rays_per_m3: float):
    """
    Lump the missing absorption into the floor until T30 before the renovation matches the measurement
    :return: Calibrated config, its T30 (B,), list of steps (alpha (B,), T30 (B,))
    """
    constant, area = sabine_constant(config), room.area_by_material()[FLOOR_LAYER]
    alpha = np.asarray(material_spec(config)[FLOOR_LAYER]['absorption'], dtype=float)
    steps = []
    for step in range(CALIBRATION_STEPS + 1):
        spec = material_spec(config)
        current = with_materials(config, {**spec, FLOOR_LAYER: {**spec[FLOOR_LAYER], 'absorption': alpha.tolist()}})
        t30 = t30_of(current, seeds, rays_per_m3, f'calibration step {step}')
        steps.append((alpha.copy(), t30))
        if np.all(np.abs(t30 / measured - 1) <= CALIBRATION_TOLERANCE) or step == CALIBRATION_STEPS:
            return current, t30, steps
        alpha = calibration_step(alpha, measured, t30, room.volume, area, constant)


def t30_of(config, seeds: int, rays_per_m3: float, label: str):
    """
    :return: (B,) T30, median over receivers, mean over seeds
    """
    return room_value(run_seeds(config, seeds, rays_per_m3, label)['t30']).mean(axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/ and thesis_simulated/')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--output', default='output/experiments/renovation')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    before, after = load_config(BEFORE), load_config(AFTER)
    if not before.model.exists():
        print("skip: private model not available")
        return 0
    room, _ = load_room(before)
    measured_before, ci_before = measured_statistics(read_receiver_table(data / 'measured' / 'mf_hall.csv'))
    references = {
        'paper': (read_receiver_table(data / 'measured' / 'mf_hall_renovated.csv')[0], None),
        'bk': measured_statistics(read_receiver_table(data / 'measured' / 'mf_hall_renovated_bk.csv')),
    }
    thesis = read_receiver_table(data / 'thesis_simulated' / 'mf_hall_renovated.csv')

    # Before the renovation: documented inputs, then the calibration
    before_documented = t30_of(before, args.seeds, args.rays_per_m3, 'before, documented')
    calibrated, before_calibrated, steps = calibrate(before, measured_before, room, args.seeds, args.rays_per_m3)
    constant, floor_area = sabine_constant(before), room.area_by_material()[FLOOR_LAYER]
    floor_documented = steps[0][0]
    calibration_rows = []
    for step, (alpha, t30) in enumerate(steps):
        for b, band in enumerate(OCTAVE_BANDS):
            added = (alpha[b] - floor_documented[b]) * floor_area
            calibration_rows.append({
                'step': step, 'band': int(band), 'floor_alpha': float(alpha[b]), 't30_before': float(t30[b]),
                'measured_before': float(measured_before[b]), 'error_pct': float(100 * (t30[b] / measured_before[b] - 1)),
                'added_area_m2': float(added), 'alpha_on_seating_area': float(floor_documented[b] + added / SEATING_AREA),
                'cloth_chairs_literature': CLOTH_CHAIRS[b],
                'missing_area_diffuse_m2': float(missing_area(measured_before, steps[0][1], room.volume, constant)[b])})
    write_csv(output / 'calibration.csv', calibration_rows)

    # After the renovation
    rows, curves = [], {}
    for treatment, exclude in TREATMENTS.items():
        after_documented = t30_of(renovated(before, after, room, exclude), args.seeds, args.rays_per_m3,
                                  f'after, documented, {treatment}')
        predictions = {
            'documented': after_documented,
            'anchored': anchored(measured_before, before_documented, after_documented),
            'calibrated': t30_of(renovated(calibrated, after, room, exclude), args.seeds, args.rays_per_m3,
                                 f'after, calibrated, {treatment}'),
        }
        curves[treatment] = predictions
        for protocol, t30 in predictions.items():
            for b, band in enumerate(OCTAVE_BANDS):
                row = {'treatment': treatment, 'protocol': protocol, 'band': int(band),
                       't30_before_measured': float(measured_before[b]),
                       't30_before_model': float(before_calibrated[b] if protocol == 'calibrated' else before_documented[b]),
                       't30_after_predicted': float(t30[b])}
                for name, (mean, ci) in references.items():
                    row[f'measured_{name}'] = float(mean[b])
                    if ci is not None:
                        row[f'measured_{name}_ci95'] = float(ci[b])
                    row[f'error_{name}_pct'] = float(100 * (t30[b] / mean[b] - 1))
                rows.append(row)

    write_csv(output / 'renovation.csv', rows)
    plot(measured_before, ci_before, before_documented, before_calibrated, curves, references, thesis,
         output / 'renovation.png')
    write_run_info(output, started, configs=[BEFORE, AFTER], seeds=args.seeds, rays_per_m3=args.rays_per_m3,
                   calibration={'layer': FLOOR_LAYER, 'tolerance': CALIBRATION_TOLERANCE, 'steps': len(steps) - 1,
                                'seating_area_m2': SEATING_AREA},
                   untreated_surface=UNTREATED_SURFACE, treatments=list(TREATMENTS))

    print(f"\nBefore: documented inputs {np.round(100 * (before_documented / measured_before - 1)).astype(int)} %, "
          f"calibrated in {len(steps) - 1} steps {np.round(100 * (before_calibrated / measured_before - 1), 1)} %")
    print(f"\n{'treatment':<24}{'protocol':<12}{'vs':<7}" + ''.join(f'{label:>7}' for label in BAND_LABELS)
          + '   (error of the predicted T30 after the renovation, %)')
    for treatment in TREATMENTS:
        for protocol in PROTOCOLS:
            for name in references:
                selected = [row for row in rows if row['treatment'] == treatment and row['protocol'] == protocol]
                print(f"{treatment:<24}{protocol:<12}{name:<7}"
                      + ''.join(cell(row, f'error_{name}_pct', 7, '+.0f') for row in selected))
    print(f"\nWrote {output}")
    return 0


def plot(measured_before, ci_before, before_documented, before_calibrated, curves, references, thesis, path: Path):
    treatments = list(curves)
    fig, axes = plt.subplots(1, 1 + len(treatments), figsize=(4.6 * (1 + len(treatments)), 4.2), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    ax = axes[0][0]
    ax.errorbar(x, measured_before, yerr=ci_before, fmt='o-', color=COLOURS['measured'], capsize=3,
                label='Measured (95% CI)')
    ax.plot(x, before_documented, '^-', color=COLOURS['new'], label='gasim, documented inputs')
    ax.plot(x, before_calibrated, 's--', color='#d35400', ms=4, label='gasim, floor calibrated')
    ax.set_title('(a) Before the renovation', fontsize=10, loc='left')
    ax.set_ylabel('T30, room value (s)')
    styles = {'documented': ('^-', COLOURS['new']), 'anchored': ('D-', '#8e44ad'), 'calibrated': ('s--', '#d35400')}
    for ax, treatment, letter in zip(axes[0][1:], treatments, 'bcd'):
        mean, ci = references['bk']
        ax.errorbar(x, mean, yerr=ci, fmt='o-', color=COLOURS['measured'], capsize=3,
                    label='Measured, B&K export (95% CI)')
        ax.plot(x, references['paper'][0], 'o:', color=COLOURS['measured'], mfc='white', label='Measured, paper Table 4')
        for protocol, (style, colour) in styles.items():
            ax.plot(x, curves[treatment][protocol], style, color=colour, ms=4, label=f'Predicted, {protocol}')
        if thesis is not None:
            ax.plot(x, np.median(thesis, axis=0), 'x-', color=COLOURS['thesis'], lw=0.8, ms=4, label='Thesis code')
        ax.plot([2, 3], [MID_BAND_LIMITS[1]] * 2, color='grey', lw=3, alpha=0.4, label='Upper limit, GB 50356')
        ax.set_title(f'({letter}) After, treated {treatment}', fontsize=10, loc='left')
    for ax in axes[0]:
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
