"""
E2, convergence with the number of rays (paper convergence section and Table S16; docs/experiment-plan.md).

    uv run python experiments/ray_count.py [--configs configs/classroom_paper.toml ...] [--seeds 10]

Every room is run at several ray densities with --seeds seeds each. The room value of a run is the median
over receivers; for every density the mean over seeds shows whether the result drifts with the number of
rays, and the standard deviation over seeds shows the Monte Carlo noise, which should fall as 1/sqrt(N).
The lecture hall, whose late decay is carried by a few long-lived paths, is run up to 300 rays/m3. Thesis
values (classroom, Table 4.11) and measured means are shown where ../data has them. Configs whose model
is not available are skipped; without any, the public test room is run. Writes ray_count_<room>.csv/png
and run_info.json to --output.
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

from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config
from gasim.geometry import load_room

from _common import (
    BAND_LABELS,
    COLOURS,
    cell,
    decay_over_seeds,
    read_receiver_table,
    room_value,
    seed_spread,
    with_decay_time,
    with_settings,
    write_csv,
    write_run_info,
)

CONFIGS = ['configs/classroom_paper.toml', 'configs/lecture_hall_paper.toml', 'configs/mf_hall_paper.toml']
RAYS_PER_M3 = [1, 2, 5, 10, 20, 50, 100]
EXTRA_RAYS_PER_M3 = {'lecture_hall_paper': [200, 300]}


def data_stem(name: str):
    """
    :return: Stem of the data files of a config: classroom_paper -> classroom
    """
    return name.removesuffix('_paper')


def read_thesis_ray_count(path: Path):
    """
    :return: {rays per m3: (B,) thesis RT}, empty if the file does not exist
    """
    table = read_receiver_table(path)
    return {} if table is None else {int(row[0]): row[1:] for row in np.column_stack(
        [np.loadtxt(path, delimiter=',', skiprows=1, usecols=0), table])}


def run_room(config, densities, seeds: int):
    """
    :return: Rows for the CSV, and {parameter: (means, spreads)} with arrays (D, B) for the plot
    """
    name = Path(config.path).stem
    room, _ = load_room(config)
    config = with_decay_time(config)
    rows, curves = [], {key: ([], []) for key in ('t30', 't20', 'edt')}
    for density in densities:
        rays = max(1, math.ceil(density * room.volume))
        result = decay_over_seeds(with_settings(config, rays=rays), seeds, f'{name} {density:g} rays/m3')
        values = {key: seed_spread(room_value(result[key])) for key in curves}
        for key, (mean, spread) in values.items():
            curves[key][0].append(mean)
            curves[key][1].append(spread)
        for b, band in enumerate(OCTAVE_BANDS):
            rows.append({'room': name, 'rays_per_m3': density, 'rays': rays, 'band': int(band),
                         **{f'{key}_mean': values[key][0][b] for key in values},
                         **{f'{key}_sd': values[key][1][b] for key in values}})
    return rows, {key: (np.array(means), np.array(spreads)) for key, (means, spreads) in curves.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='+', default=CONFIGS)
    parser.add_argument('--data', default='../data')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--densities', type=float, nargs='+', help=f'rays per m3, default {RAYS_PER_M3} '
                                                                   'plus the extra densities of the lecture hall')
    parser.add_argument('--output', default='output/experiments/ray_count')
    args = parser.parse_args(argv)
    output, data = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    configs = [load_config(path) for path in args.configs]
    available = [config for config in configs if config.model.exists()]
    for config in configs:
        if not config.model.exists():
            print(f"skip {config.path}: {config.model} not available")
    if not available:
        print("no private model available, using configs/test_room.toml")
        available = [load_config('configs/test_room.toml')]

    used = {}
    for config in available:
        name = Path(config.path).stem
        densities = args.densities or RAYS_PER_M3 + EXTRA_RAYS_PER_M3.get(name, [])
        used[name] = densities
        rows, curves = run_room(config, densities, args.seeds)
        stem = data_stem(name)
        thesis = read_thesis_ray_count(data / 'thesis_simulated' / f'ray_count_{stem}.csv')
        measured = read_receiver_table(data / 'measured' / f'{stem}.csv')
        measured_mean = np.nanmean(measured, axis=0) if measured is not None else None
        for row in rows:
            b = [int(band) for band in OCTAVE_BANDS].index(row['band'])
            density = row['rays_per_m3']
            if density == int(density) and int(density) in thesis:
                row['thesis'] = thesis[int(density)][b]
            if measured_mean is not None:
                row['measured'] = measured_mean[b]
        write_csv(output / f'ray_count_{name}.csv', rows)
        room, _ = load_room(config)
        plot(name, room.volume, densities, curves, thesis, measured_mean, output / f'ray_count_{name}.png')
        print(f"\n{name}, {room.volume:.1f} m3, {args.seeds} seeds per density (median over receivers)")
        print(f"{'rays/m3':>8}{'band':>6}{'T30':>8}{'±seed':>8}{'T20':>8}{'EDT':>8}{'thesis':>8}{'measured':>10}")
        for row in rows:
            print(f"{row['rays_per_m3']:>8g}{row['band']:>6}{cell(row, 't30_mean', 8)}{cell(row, 't30_sd', 8, '.3f')}"
                  f"{cell(row, 't20_mean', 8)}{cell(row, 'edt_mean', 8)}{cell(row, 'thesis', 8)}"
                  f"{cell(row, 'measured', 10)}")

    write_run_info(output, started, configs=[str(c.path) for c in available], seeds=args.seeds, densities=used)
    print(f"\nWrote {output}")
    return 0


def plot(name, volume, densities, curves, thesis, measured_mean, path: Path):
    means, spreads = curves['t30']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for b, ax in enumerate(axes.ravel()):
        ax.fill_between(densities, means[:, b] - spreads[:, b], means[:, b] + spreads[:, b], color=COLOURS['new'],
                        alpha=0.2, lw=0)
        ax.plot(densities, means[:, b], '^-', color=COLOURS['new'], ms=4, label='gasim T30 (mean ± 1 sd)')
        ax.plot(densities, curves['edt'][0][:, b], 'v:', color=COLOURS['new'], ms=3, label='gasim EDT')
        known = [d for d in densities if d == int(d) and int(d) in thesis]
        if known:
            ax.plot(known, [thesis[int(d)][b] for d in known], 's--', color=COLOURS['thesis'], ms=4,
                    label='Thesis code (one run each)')
        if measured_mean is not None:
            ax.axhline(measured_mean[b], color=COLOURS['measured'], ls='--', lw=1, label='Measured mean')
        ax.set_xscale('log')
        ax.set_title(f'{BAND_LABELS[b]} Hz', fontsize=10, loc='left')
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Rays per m$^3$')
    for ax in axes[:, 0]:
        ax.set_ylabel('Room median (s)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{name}, {volume:.0f} m$^3$', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
