"""
Material sources: the three validation rooms with the materials of the thesis run configs and of the paper appendix.

    uv run python experiments/material_sources.py [--seeds 3] [--rays-per-m3 100] [--data ../data]

The drive configs (configs/<room>.toml) and the paper's Tables A1-A3 (configs/<room>_paper.toml) give different
absorption coefficients for the same rooms. Each room is run with both, at the same ray density, and compared with the
measurements and the thesis engine; the multifunction hall after renovation (configs/mf_hall_renovated.toml, paper
materials) is compared with both after-renovation data sets (see data/README.md). Eyring with the same inputs is
listed for every case: where it misses the measurements too, the materials are the cause, not the propagation.
Writes material_sources.csv and material_sources.png to --output.
"""
import argparse
import math
import sys
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
    cell,
    decay_over_seeds,
    eyring_of,
    read_receiver_table,
    seed_spread,
    with_settings,
    write_csv,
)

# case: (title, data file stem, {source: config})
CASES = {
    'classroom': ('Classroom', 'classroom', {'code': 'configs/classroom.toml', 'paper': 'configs/classroom_paper.toml'}),
    'lecture_hall': ('Lecture hall', 'lecture_hall', {'code': 'configs/lecture_hall.toml',
                                                      'paper': 'configs/lecture_hall_paper.toml'}),
    'mf_hall': ('Multifunction hall', 'mf_hall', {'code': 'configs/mf_hall.toml', 'paper': 'configs/mf_hall_paper.toml'}),
    'mf_hall_renovated': ('Multifunction hall, renovated', 'mf_hall_renovated',
                          {'paper': 'configs/mf_hall_renovated.toml'}),
}
SOURCE_COLOURS = {'code': COLOURS['new'], 'paper': '#d35400'}
SOURCE_LABELS = {'code': 'Run-config materials', 'paper': 'Paper materials'}


def run_source(path: str, seeds: int, rays_per_m3: float, label: str):
    """
    :return: dict with t30 (S, R, B), eyring (B,), rays, volume, max_time
    """
    config = load_config(path)
    room, _ = load_room(config)
    reference = eyring_of(config)
    # Trace far enough for a 50 dB decay of the slowest band with room for a non-diffuse tail
    max_time = min(12.0, max(config.simulation.max_time, 1.5 * float(np.max(reference)) * 70 / 60))
    rays = math.ceil(rays_per_m3 * room.volume)
    t30 = decay_over_seeds(with_settings(config, rays=rays, max_time=max_time), seeds, label)['t30']
    return {'t30': t30, 'eyring': reference, 'rays': rays, 'volume': room.volume, 'max_time': max_time}


def relative_error(simulated, measured):
    return 100 * (simulated - measured) / measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/ and thesis_simulated/')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--output', default='output/experiments/material_sources')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows, results = [], {}
    for case in args.cases:
        title, stem, sources = CASES[case]
        if not load_config(next(iter(sources.values()))).model.exists():
            print(f"skip {case}: private model not available")
            continue
        measured = {'measured': read_receiver_table(data / 'measured' / f'{stem}.csv'),
                    'measured_bk': read_receiver_table(data / 'measured' / f'{stem}_bk.csv')}
        measured = {key: np.nanmean(table, axis=0) for key, table in measured.items() if table is not None}
        thesis = read_receiver_table(data / 'thesis_simulated' / f'{stem}.csv')
        thesis = thesis.mean(axis=0) if thesis is not None else None
        runs = {source: run_source(path, args.seeds, args.rays_per_m3, f'{case} {source}')
                for source, path in sources.items()}
        results[case] = (title, measured, thesis, runs)

        for source, run in runs.items():
            mean, spread = seed_spread(np.nanmean(run['t30'], axis=1))  # room mean per seed, then over seeds
            for b, band in enumerate(OCTAVE_BANDS):
                row = {'case': case, 'materials': source, 'band': int(band), 'rays': run['rays'],
                       'max_time': run['max_time'], 'new_t30': mean[b], 'new_seed_std': spread[b],
                       'eyring': run['eyring'][b]}
                if thesis is not None:
                    row['thesis'] = thesis[b]
                for key, values in measured.items():
                    row[key] = values[b]
                    row[f'error_new_{key}_pct'] = relative_error(mean[b], values[b])
                    row[f'error_eyring_{key}_pct'] = relative_error(run['eyring'][b], values[b])
                rows.append(row)

    write_csv(output / 'material_sources.csv', rows)
    plot(results, output / 'material_sources.png')
    print_summary(rows)
    print(f"\nWrote {output / 'material_sources.csv'} and {output / 'material_sources.png'}")
    return 0


def print_summary(rows):
    print(f"\n{'case':<19}{'materials':<10}{'band':>6}{'measured':>10}{'thesis':>8}{'new T30':>9}{'±seed':>7}"
          f"{'Eyring':>8}{'err new':>9}{'err Eyr':>9}{'B&K':>7}{'err B&K':>9}")
    for row in rows:
        print(f"{row['case']:<19}{row['materials']:<10}{row['band']:>6}{cell(row, 'measured', 10)}"
              f"{cell(row, 'thesis', 8)}{cell(row, 'new_t30', 9)}{cell(row, 'new_seed_std', 7, '.3f')}"
              f"{cell(row, 'eyring', 8)}{cell(row, 'error_new_measured_pct', 8, '+.1f')}%"
              f"{cell(row, 'error_eyring_measured_pct', 8, '+.1f')}%{cell(row, 'measured_bk', 7)}"
              f"{cell(row, 'error_new_measured_bk_pct', 8, '+.1f')}%")


def plot(results, path: Path):
    n = max(len(results), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    for ax, (case, (title, measured, thesis, runs)) in zip(axes[0], results.items()):
        if 'measured' in measured:
            ax.plot(x, measured['measured'], 'o-', color=COLOURS['measured'], label='Measured')
        if 'measured_bk' in measured:
            ax.plot(x, measured['measured_bk'], 'x--', color=COLOURS['measured'], label='Measured, B&K export')
        if thesis is not None:
            ax.plot(x, thesis, 's--', color=COLOURS['thesis'], ms=4, label='Thesis engine')
        for source, run in runs.items():
            mean, spread = seed_spread(np.nanmean(run['t30'], axis=1))
            ax.errorbar(x, mean, yerr=spread if len(run['t30']) > 1 else None, fmt='^-', ms=4, capsize=2,
                        color=SOURCE_COLOURS[source], label=f'New engine, {SOURCE_LABELS[source].lower()}')
            ax.plot(x, run['eyring'], ':', color=SOURCE_COLOURS[source], label=f'Eyring, {SOURCE_LABELS[source].lower()}')
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('T30, room mean (s)')
    handles = {label: handle for ax in axes[0] for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
