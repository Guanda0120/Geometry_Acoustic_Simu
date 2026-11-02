"""
Reflections and energy per surface of the multifunction hall (paper Fig. 9, with energy weighting).

    uv run python experiments/surface_statistics.py [--config configs/mf_hall_paper.toml] [--rays-per-m3 100]

Counts as in the thesis (image-source reflection points of orders 1 .. ism_order summed over the
receivers, ray hits) and the same divided by area, next to two energy measures: the share of the wall
absorption that each surface takes and the share of the received energy that arrives with the surface
as the last reflection. Surfaces come from the [surfaces] table of the config. Writes
surface_statistics.csv (all bands) and surface_statistics.png (--band) to --output.
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
from gasim.simulate import simulate

from _common import COLOURS, with_settings, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/mf_hall_paper.toml')
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--band', type=int, default=1000, help='octave band of the figure, Hz')
    parser.add_argument('--output', default='output/experiments/surfaces')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if not config.model.exists():
        print(f"skip: {config.model} not available")
        return 0
    room, _ = load_room(config)
    result = simulate(with_settings(config, rays=math.ceil(args.rays_per_m3 * room.volume)))
    stats = result.surfaces
    ism = stats.ism_reflections.sum(axis=(0, 1))  # all receivers, orders 1 .. N

    rows = []
    for b, band in enumerate(OCTAVE_BANDS):
        for s, name in enumerate(stats.names):
            rows.append({'band': int(band), 'surface': name, 'area': stats.area[s], 'ism_reflections': int(ism[s]),
                         'ism_per_m2_rel': stats.relative_density(ism)[s], 'ray_hits': int(stats.ray_hits[b, s]),
                         'ray_hits_per_m2_rel': stats.relative_density(stats.ray_hits[b])[s],
                         'incident_per_m2_rel': stats.relative_density(stats.incident[b])[s],
                         'absorbed_share': stats.absorbed_share()[b, s],
                         'received_share': stats.received_share()[:, b, s].mean()})
    write_csv(output / 'surface_statistics.csv', rows)

    b = [int(band) for band in OCTAVE_BANDS].index(args.band)
    panels = [(f'Image-source reflection points\n(orders 1-{stats.ism_reflections.shape[1]}, all receivers)', ism),
              ('Ray hits', stats.ray_hits[b]),
              ('Image-source points per m$^2$\n(relative to room mean)', stats.relative_density(ism)),
              ('Ray hits per m$^2$\n(relative to room mean)', stats.relative_density(stats.ray_hits[b])),
              (f'Share of wall absorption, {args.band} Hz (%)', 100 * stats.absorbed_share()[b]),
              (f'Share of received energy via last\nreflection, {args.band} Hz (%)',
               100 * stats.received_share()[:, b].mean(axis=0))]
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    x = np.arange(len(stats.names))
    for ax, (title, values) in zip(axes.ravel(), panels):
        ax.bar(x, values, color=COLOURS['new'])
        ax.set_xticks(x, stats.names, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9, loc='left')
        ax.grid(axis='y', alpha=0.3)
    for ax in axes[1]:
        ax.axhline(1.0, color='grey', lw=0.8, ls=':')
    fig.suptitle(f'{config.path.stem}: {room.volume:.0f} m$^3$, {result.metadata["rays"]} rays', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'surface_statistics.png', dpi=200)
    plt.close(fig)

    print(f"\n{config.path.stem}, {args.band} Hz")
    print(f"{'surface':<10}{'area':>8}{'ISM':>7}{'ISM/m2':>8}{'hits/m2':>9}{'absorbed%':>11}{'received%':>11}")
    for row in rows:
        if row['band'] == args.band:
            print(f"{row['surface']:<10}{row['area']:>8.1f}{row['ism_reflections']:>7}{row['ism_per_m2_rel']:>8.2f}"
                  f"{row['ray_hits_per_m2_rel']:>9.2f}{100 * row['absorbed_share']:>11.1f}"
                  f"{100 * row['received_share']:>11.1f}")
    print(f"\nWrote {output / 'surface_statistics.csv'} and {output / 'surface_statistics.png'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
