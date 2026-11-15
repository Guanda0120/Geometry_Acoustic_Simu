"""
E3, error budget of the inputs (new figure and the prediction band of Fig. 6; docs/experiment-plan.md).

    uv run python experiments/input_sensitivity.py [--seeds 5] [--rays-per-m3 100] [--cases classroom ...]

Starting from the documented inputs of each room (configs/<room>_paper.toml), one input at a time is moved
to both ends of a plausible range and the room is rerun with the same seeds, so that every difference is
paired and largely free of Monte Carlo noise:

- the absorption of each material, by +/- max(DELTA_ABSOLUTE, DELTA_RELATIVE x alpha) in every band;
- the scattering coefficient, uniform at both ends of SCATTERING_RANGE;
- the air, at the four corners of AIR_RANGE (temperature, humidity); the extremes count as one factor.

For every factor and band the changes of T30 (median over receivers, mean over seeds) that shorten it and
those that lengthen it are kept apart: a factor whose base value lies at one end of its range moves T30 in
one direction only. The downward changes of the materials, of the scattering and of the air are combined
as a root sum of squares, and so are the upward changes; this gives a prediction band with two sides,
which is compared with the measured mean and its 95 % confidence interval. Volume is not perturbed: the models agree with the drawings within 3 %, and RT scales
linearly with the size of a room (about 1 % for a 3 % volume error).

Writes factors.csv, budget.csv, input_sensitivity.png and run_info.json to --output.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.bands import OCTAVE_BANDS
from gasim.config import AirConditions, load_config

from _common import (
    BAND_LABELS,
    COLOURS,
    PAPER_ROOMS,
    SCATTERING_RANGE,
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

DELTA_ABSOLUTE = 0.05  # absorption perturbation: the larger of 0.05 and 25 % of the value
DELTA_RELATIVE = 0.25
AIR_RANGE = ((10.0, 40.0), (10.0, 80.0), (30.0, 40.0), (30.0, 80.0))  # (degC, % RH); not recorded on site
PLOT_BAND = 1000


def absorption_perturbations(config):
    """
    :return: [(factor name, config with lower absorption, config with higher absorption)] per material
    """
    perturbed = []
    for name, entry in material_spec(config).items():
        alpha = np.asarray(entry['absorption'])
        delta = np.maximum(DELTA_ABSOLUTE, DELTA_RELATIVE * alpha)
        low, high = np.clip(alpha - delta, 0.0, 1.0), np.clip(alpha + delta, 0.0, 0.99)
        perturbed.append((f'alpha {name}',
                          with_materials(config, {**material_spec(config), name: {**entry, 'absorption': low.tolist()}}),
                          with_materials(config, {**material_spec(config), name: {**entry, 'absorption': high.tolist()}})))
    return perturbed


def run_case(case: str, seeds: int, rays_per_m3: float):
    """
    :return: Base T30 (B,), {factor: (change of the low end, change of the high end) in %, arrays (B,)}
    """
    config = load_config(PAPER_ROOMS[case][1])

    def t30(variant, label):
        return room_value(run_seeds(variant, seeds, rays_per_m3, f'{case} {label}')['t30']).mean(axis=0)

    base = t30(config, 'base')
    changes = {}
    for name, low, high in absorption_perturbations(config):
        changes[name] = (100 * (t30(low, f'{name} -') / base - 1), 100 * (t30(high, f'{name} +') / base - 1))
    low, high = SCATTERING_RANGE[case]
    changes['scattering'] = (100 * (t30(uniform_scattering(config, low), f's = {low}') / base - 1),
                             100 * (t30(uniform_scattering(config, high), f's = {high}') / base - 1))
    air = np.array([100 * (t30(dataclasses.replace(config, air=AirConditions(temperature, humidity,
                                                                              config.air.pressure)),
                               f'air {temperature:g} degC {humidity:g} %') / base - 1)
                    for temperature, humidity in AIR_RANGE])
    changes['air'] = (air.min(axis=0), air.max(axis=0))
    return base, changes


def budget(changes):
    """
    :param changes: {factor: (change at the low end, change at the high end) of T30 in %, arrays (B,)}
    :return: {group: (down, up)}, (B,) one-sided uncertainties in % (both positive), for the materials, the
             scattering, the air and their total
    """
    sides = {name: (np.maximum(0.0, -np.minimum(low, high)), np.maximum(0.0, np.maximum(low, high)))
             for name, (low, high) in changes.items()}
    groups = {}
    for side in (0, 1):
        materials = np.sqrt(sum(value[side] ** 2 for name, value in sides.items() if name.startswith('alpha ')))
        total = np.sqrt(materials ** 2 + sides['scattering'][side] ** 2 + sides['air'][side] ** 2)
        for group, value in (('materials', materials), ('scattering', sides['scattering'][side]),
                             ('air', sides['air'][side]), ('total', total)):
            groups.setdefault(group, []).append(value)
    return {group: tuple(values) for group, values in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--cases', nargs='+', choices=list(PAPER_ROOMS), default=list(PAPER_ROOMS))
    parser.add_argument('--output', default='output/experiments/input_sensitivity')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    factor_rows, budget_rows, results = [], [], {}
    for case in args.cases:
        title, config_path, stem = PAPER_ROOMS[case]
        if not load_config(config_path).model.exists():
            print(f"skip {case}: private model not available")
            continue
        base, changes = run_case(case, args.seeds, args.rays_per_m3)
        groups = budget(changes)
        measured, ci = measured_statistics(read_receiver_table(data / 'measured' / f'{stem}.csv'))
        results[case] = (title, base, changes, groups, measured, ci)
        for name, (low, high) in changes.items():
            for b, band in enumerate(OCTAVE_BANDS):
                factor_rows.append({'case': case, 'factor': name, 'band': int(band), 'change_low_pct': low[b],
                                    'change_high_pct': high[b]})
        for b, band in enumerate(OCTAVE_BANDS):
            row = {'case': case, 'band': int(band), 't30_base': base[b],
                   **{f'u_{group}_{side}_pct': values[i][b] for group, values in groups.items()
                      for i, side in enumerate(('down', 'up'))}}
            if measured is not None:
                down, up = groups['total']
                lower, upper = base[b] * (1 - down[b] / 100), base[b] * (1 + up[b] / 100)
                row.update({'measured': measured[b], 'measured_ci95': ci[b],
                            'error_pct': 100 * (base[b] / measured[b] - 1), 'band_low': lower, 'band_high': upper,
                            'within_band': bool(lower - ci[b] <= measured[b] <= upper + ci[b])})
            budget_rows.append(row)

    write_csv(output / 'factors.csv', factor_rows)
    write_csv(output / 'budget.csv', budget_rows)
    plot(results, output / 'input_sensitivity.png')
    write_run_info(output, started, cases={c: PAPER_ROOMS[c][1] for c in args.cases}, seeds=args.seeds,
                   rays_per_m3=args.rays_per_m3, delta=(DELTA_ABSOLUTE, DELTA_RELATIVE), air_range=AIR_RANGE,
                   scattering_range={c: SCATTERING_RANGE[c] for c in args.cases})
    print(f"\n{'case':<13}{'band':>6}{'T30':>7}{'meas':>7}{'err %':>8}{'mat -/+':>12}{'scat -/+':>12}{'air -/+':>12}"
          f"{'total -/+':>12}  within")
    for row in budget_rows:
        print(f"{row['case']:<13}{row['band']:>6}{cell(row, 't30_base', 7)}{cell(row, 'measured', 7)}"
              f"{cell(row, 'error_pct', 8, '+.1f')}"
              + ''.join(f"{cell(row, f'u_{group}_down_pct', 7, '.0f')}/{cell(row, f'u_{group}_up_pct', 4, '.0f')}"
                        for group in ('materials', 'scattering', 'air', 'total'))
              + f"  {row.get('within_band', '-')}")
    print(f"\nWrote {output}")
    return 0


def plot(results, path: Path):
    n = max(len(results), 1)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.2), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    b = [int(band) for band in OCTAVE_BANDS].index(PLOT_BAND)
    for column, (title, base, changes, groups, measured, ci) in enumerate(results.values()):
        ax = axes[0][column]
        for group, alpha in (('total', 0.15), ('materials', 0.3)):
            down, up = groups[group]
            ax.fill_between(x, base * (1 - down / 100), base * (1 + up / 100), color=COLOURS['new'],
                            alpha=alpha, lw=0, label=f'{group} uncertainty (−/+)')
        ax.plot(x, base, '^-', color=COLOURS['new'], label='gasim T30, documented inputs')
        if measured is not None:
            ax.errorbar(x, measured, yerr=ci, fmt='o-', color=COLOURS['measured'], capsize=3, label='Measured (95% CI)')
        ax.set_xticks(x, BAND_LABELS)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax = axes[1][column]
        order = sorted(changes, key=lambda name: max(abs(changes[name][0][b]), abs(changes[name][1][b])))
        y = np.arange(len(order))
        ax.barh(y, [changes[name][0][b] for name in order], color=COLOURS['measured'], alpha=0.6, label='low end')
        ax.barh(y, [changes[name][1][b] for name in order], color=COLOURS['new'], alpha=0.6, label='high end')
        ax.set_yticks(y, [name.removeprefix('alpha ') for name in order], fontsize=8)
        ax.axvline(0, color='black', lw=0.6)
        ax.set_xlabel(f'Change of T30 at {PLOT_BAND} Hz (%)')
        ax.grid(axis='x', alpha=0.3)
    axes[0][0].set_ylabel('T30 (s)')
    axes[0][0].legend(fontsize=7)
    axes[1][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
