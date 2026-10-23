"""
Ray-count sweep: reverberation time against rays per m3, several seeds each (paper Fig. 5).

    uv run python experiments/ray_count.py [--config configs/classroom.toml] [--seeds 10] [--data ../data]

The thesis ran one seed per setting in the classroom; here every setting gets --seeds runs, so the spread
between seeds shows the Monte Carlo noise and the mean shows whether the result drifts with the ray count.
Falls back to the test room when the classroom model is not available. Thesis values (Table 4.11) and the
measured mean are read from data/ when the config is the classroom. Writes ray_count.csv/png to --output.
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
    read_receiver_table,
    seed_spread,
    with_settings,
    write_csv,
)

RAYS_PER_M3 = [1, 2, 5, 10, 20, 50, 100]


def read_thesis_ray_count(path: Path):
    """
    :return: {rays per m3: (B,) thesis RT}, empty if the file does not exist
    """
    table = read_receiver_table(path)
    return {} if table is None else {int(row[0]): row[1:] for row in np.column_stack(
        [np.loadtxt(path, delimiter=',', skiprows=1, usecols=0), table])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/classroom.toml')
    parser.add_argument('--data', default='../data')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--densities', type=float, nargs='+', default=RAYS_PER_M3, help='rays per m3')
    parser.add_argument('--output', default='output/experiments/ray_count')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if not config.model.exists():
        print(f"{config.model} not available, using configs/test_room.toml")
        config = load_config('configs/test_room.toml')
    name = Path(config.path).stem
    room, _ = load_room(config)
    data = Path(args.data)
    thesis = read_thesis_ray_count(data / 'thesis_simulated' / f'ray_count_{name}.csv')
    measured = read_receiver_table(data / 'measured' / f'{name}.csv')
    measured_mean = measured.mean(axis=0) if measured is not None else None

    rows, means, spreads = [], [], []
    for density in args.densities:
        rays = max(1, math.ceil(density * room.volume))
        t30 = decay_over_seeds(with_settings(config, rays=rays), args.seeds, f'{density:g} rays/m3')['t30']
        mean, spread = seed_spread(np.nanmean(t30, axis=1))  # room mean of each seed, then over seeds
        means.append(mean)
        spreads.append(spread)
        for b, band in enumerate(OCTAVE_BANDS):
            row = {'rays_per_m3': density, 'rays': rays, 'band': int(band), 'new_t30': mean[b],
                   'new_seed_std': spread[b]}
            if int(density) in thesis and density == int(density):
                row['thesis'] = thesis[int(density)][b]
            if measured_mean is not None:
                row['measured'] = measured_mean[b]
            rows.append(row)

    write_csv(output / 'ray_count.csv', rows)
    plot(name, room.volume, args.densities, np.array(means), np.array(spreads), thesis, measured_mean,
         output / 'ray_count.png')
    print(f"\n{name}, {room.volume:.1f} m3, {args.seeds} seeds per setting")
    print(f"{'rays/m3':>8}{'band':>6}{'thesis':>8}{'new T30':>9}{'±seed':>8}{'measured':>10}")
    for row in rows:
        print(f"{row['rays_per_m3']:>8g}{row['band']:>6}{cell(row, 'thesis', 8)}{cell(row, 'new_t30', 9)}"
              f"{cell(row, 'new_seed_std', 8, '.3f')}{cell(row, 'measured', 10)}")
    print(f"\nWrote {output / 'ray_count.csv'} and {output / 'ray_count.png'}")
    return 0


def plot(name, volume, densities, means, spreads, thesis, measured_mean, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for b, ax in enumerate(axes.ravel()):
        ax.fill_between(densities, means[:, b] - spreads[:, b], means[:, b] + spreads[:, b], color=COLOURS['new'],
                        alpha=0.2, lw=0)
        ax.plot(densities, means[:, b], '^-', color=COLOURS['new'], ms=4, label='New engine (mean ± 1 s.d.)')
        known = [d for d in densities if d == int(d) and int(d) in thesis]
        if known:
            ax.plot(known, [thesis[int(d)][b] for d in known], 's--', color=COLOURS['thesis'], ms=4,
                    label='Thesis engine (one run each)')
        if measured_mean is not None:
            ax.axhline(measured_mean[b], color=COLOURS['measured'], ls='--', lw=1, label='Measured mean')
        ax.set_xscale('log')
        ax.set_title(f'{BAND_LABELS[b]} Hz', fontsize=10, loc='left')
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Rays per m$^3$')
    for ax in axes[:, 0]:
        ax.set_ylabel('T30, room mean (s)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{name}, {volume:.0f} m$^3$', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
