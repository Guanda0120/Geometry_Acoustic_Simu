"""
E4, scenarios for inputs that were not documented (seating results and Discussion; docs/experiment-plan.md).

    uv run python experiments/scenarios.py [--seeds 5] [--rays-per-m3 100] [--cases mf_hall ...]

Each scenario changes the documented inputs of one room (configs/<room>_paper.toml) by an assumption that
is stated in SCENARIOS, with literature values from LITERATURE. Seats are merged into the floor layer by
area weighting (equal absorption area, no seat geometry): on a layer of area S with absorption alpha, a
seating area A with absorption alpha_seat gives alpha + (A / S)(alpha_seat - alpha). All runs of a room use
the same seeds, so the differences between scenarios are paired. The results show how far each plausible
missing input moves the prediction; they are not used to tune the model.

Writes scenarios.csv, scenarios.png and run_info.json to --output.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config
from gasim.geometry import load_room

from _common import (
    BAND_LABELS,
    COLOURS,
    PAPER_ROOMS,
    cell,
    material_spec,
    measured_statistics,
    read_receiver_table,
    room_value,
    run_seeds,
    uniform_scattering,
    with_materials,
    write_csv,
    write_run_info,
)

# Absorption coefficients, 125 Hz - 4 kHz
LITERATURE = {
    # Vorlaender, Auralization (2020), Annex p. 335, glazing
    'double_glazing_wide_gap': ([0.15, 0.05, 0.03, 0.03, 0.02, 0.02], 'Vorlaender 2020, Annex p. 335: double glazing, 2-3 mm glass, >30 mm gap'),
    'lead_glazing': ([0.30, 0.20, 0.14, 0.10, 0.05, 0.05], 'Vorlaender 2020, Annex p. 335: lead glazing (as in the manuscript calibration)'),
    # Vorlaender, Auralization (2020), Annex p. 336, seating, per m2 of seating at two seats per m2
    'wooden_chairs': ([0.05, 0.08, 0.10, 0.12, 0.12, 0.12], 'Vorlaender 2020, Annex p. 336: wooden chairs without cushion'),
    'cloth_chairs': ([0.44, 0.60, 0.77, 0.89, 0.82, 0.70], 'Vorlaender 2020, Annex p. 336: empty chairs, upholstered with cloth cover'),
    # Thesis Table 4.5 / paper Table A2 (source not documented)
    'thesis_seats': ([0.07, 0.14, 0.21, 0.24, 0.19, 0.27], 'thesis Table 4.5: audience seating'),
    # Paper Table A2
    'timber': ([0.16, 0.15, 0.10, 0.10, 0.10, 0.10], 'paper Table A2: timber'),
}

# Seating areas (m2 at two seats per m2, edge strip included) - assumptions to confirm
SEATING = {
    'classroom': 24.0,  # about 48 chairs; desks and chairs were not documented
    'lecture_hall': 145.0,  # 15.9 x 9.1 m from the rear wall to the first receiver row (manuscript assumption)
    'mf_hall': 168.0,  # 280 seats including a 0.5 m edge strip (Kuttruff 2016, p. 149; manuscript)
}

# Scenario: list of changes (kind, argument); kinds: seats (literature key), material (layer, literature key),
# scattering (uniform value)
SCENARIOS = {
    'classroom': {
        'wooden chairs': [('seats', 'wooden_chairs')],
        'glazing, low-frequency': [('material', ('Glass', 'double_glazing_wide_gap'))],
        's = 0.2': [('scattering', 0.2)],
        'chairs + glazing + s = 0.2': [('seats', 'wooden_chairs'), ('material', ('Glass', 'double_glazing_wide_gap')),
                                       ('scattering', 0.2)],
    },
    'lecture_hall': {
        'S1: thesis seat coefficient': [('seats', 'thesis_seats')],
        'S2: cloth-upholstered seats': [('seats', 'cloth_chairs')],
        'side walls (Absorption_Plane) timber': [('material', ('Absorption_Plane', 'timber'))],
        'S2 + s = 0.3': [('seats', 'cloth_chairs'), ('scattering', 0.3)],
    },
    'mf_hall': {
        'cloth-upholstered seats': [('seats', 'cloth_chairs')],
        'glazing, low-frequency': [('material', ('Glass', 'double_glazing_wide_gap'))],
        'glazing, lead (manuscript)': [('material', ('Glass', 'lead_glazing'))],
        'seats + glazing': [('seats', 'cloth_chairs'), ('material', ('Glass', 'double_glazing_wide_gap'))],
        'seats + glazing + s = 0.2': [('seats', 'cloth_chairs'), ('material', ('Glass', 'double_glazing_wide_gap')),
                                      ('scattering', 0.2)],
    },
}
FLOOR_LAYER = 'Floor'


def apply(config, case: str, changes):
    """
    :return: Config of one scenario
    """
    spec = material_spec(config)
    floor_area = load_room(config)[0].area_by_material()[FLOOR_LAYER]
    scattering = None
    for kind, argument in changes:
        if kind == 'seats':
            seat = np.asarray(LITERATURE[argument][0])
            floor = np.asarray(spec[FLOOR_LAYER]['absorption'])
            share = SEATING[case] / floor_area
            spec[FLOOR_LAYER] = {**spec[FLOOR_LAYER], 'absorption': (floor + share * (seat - floor)).tolist()}
        elif kind == 'material':
            layer, key = argument
            spec[layer] = {**spec[layer], 'absorption': LITERATURE[key][0]}
        elif kind == 'scattering':
            scattering = argument
        else:
            raise ValueError(f"unknown change {kind!r}")
    config = with_materials(config, spec)
    return uniform_scattering(config, scattering) if scattering is not None else config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--cases', nargs='+', choices=list(PAPER_ROOMS), default=list(PAPER_ROOMS))
    parser.add_argument('--output', default='output/experiments/scenarios')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows, results = [], {}
    for case in args.cases:
        title, config_path, stem = PAPER_ROOMS[case]
        config = load_config(config_path)
        if not config.model.exists():
            print(f"skip {case}: private model not available")
            continue
        measured, ci = measured_statistics(read_receiver_table(data / 'measured' / f'{stem}.csv'))
        curves = {}
        for name, changes in {'documented inputs': [], **SCENARIOS[case]}.items():
            run = run_seeds(apply(config, case, changes), args.seeds, args.rays_per_m3, f'{case}: {name}')
            values = {key: room_value(run[key]) for key in ('t30', 't20', 'edt', 'c50')}
            curves[name] = values['t30'].mean(axis=0)
            for b, band in enumerate(OCTAVE_BANDS):
                row = {'case': case, 'scenario': name, 'band': int(band),
                       **{f'{key}_mean': value[:, b].mean() for key, value in values.items()},
                       't30_sd': values['t30'][:, b].std(ddof=1) if args.seeds > 1 else float('nan')}
                if measured is not None:
                    row.update({'measured': measured[b], 'measured_ci95': ci[b],
                                'error_t30_pct': 100 * (row['t30_mean'] / measured[b] - 1)})
                rows.append(row)
        results[case] = (title, curves, measured, ci)

    write_csv(output / 'scenarios.csv', rows)
    plot(results, output / 'scenarios.png')
    write_run_info(output, started, cases={c: PAPER_ROOMS[c][1] for c in args.cases}, seeds=args.seeds,
                   rays_per_m3=args.rays_per_m3, seating_m2=SEATING,
                   scenarios={c: SCENARIOS[c] for c in args.cases},
                   literature={key: source for key, (_, source) in LITERATURE.items()})
    print(f"\n{'case':<13}{'scenario':<40}" + ''.join(f'{label:>8}' for label in BAND_LABELS) + '   (T30 error %)')
    for case in results:
        for name in {'documented inputs': None, **SCENARIOS[case]}:
            errors = [row for row in rows if row['case'] == case and row['scenario'] == name]
            print(f"{case:<13}{name[:39]:<40}" + ''.join(cell(row, 'error_t30_pct', 8, '+.0f') for row in errors))
    print(f"\nWrote {output}")
    return 0


def plot(results, path: Path):
    n = max(len(results), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    for ax, (title, curves, measured, ci) in zip(axes[0], results.values()):
        if measured is not None:
            ax.errorbar(x, measured, yerr=ci, fmt='o-', color=COLOURS['measured'], capsize=3, label='Measured (95% CI)')
        for i, (name, t30) in enumerate(curves.items()):
            style = {'color': COLOURS['new'], 'lw': 2} if i == 0 else {'lw': 1, 'alpha': 0.85}
            ax.plot(x, t30, '^-' if i == 0 else '.--', ms=4, label=name, **style)
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel('T30, median over receivers (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
