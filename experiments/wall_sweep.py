"""
E7, wall absorption and scattering sweeps in the 9 x 7 x 5 m test room (supplement, Tables S6-S8;
docs/experiment-plan.md).

    uv run python experiments/wall_sweep.py [absorption scattering] [--seeds 10] [--data ../data]

Inputs as in the thesis (Tables 4.12 and 4.14): carpet floor, gypsum ceiling, the four walls varied; s = 0.10
except the walls in the scattering sweep. CATT and thesis-engine values at receiver 1 are read from
data/catt/<kind>_sweep.csv if present; they are shown for comparison and are not a reference (the CATT values
at 125 Hz are not consistent with the documented inputs). Writes <kind>_sweep.csv, <kind>_sweep.png and
run_info.json to --output.
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

from _common import (
    BAND_LABELS,
    COLOURS,
    SWEEP_VALUES,
    TEST_ROOM_CEILING,
    TEST_ROOM_FLOOR,
    TEST_ROOM_SCATTERING_WALLS_ABSORPTION,
    cell,
    decay_over_seeds,
    eyring_of,
    read_sweep_table,
    seed_spread,
    with_materials,
    with_settings,
    write_csv,
    write_run_info,
)


def sweep_config(kind: str, value: float, rays: int):
    """
    Test-room config for one wall value
    :param kind: 'absorption' or 'scattering'
    :param value: Wall absorption or scattering coefficient, all bands
    """
    walls = ({'absorption': value, 'scattering': 0.10} if kind == 'absorption'
             else {'absorption': TEST_ROOM_SCATTERING_WALLS_ABSORPTION, 'scattering': value})
    config = with_materials(load_config('configs/test_room.toml'),
                            {'Mat': {'absorption': TEST_ROOM_FLOOR, 'scattering': 0.10},
                             'Plaster': {'absorption': TEST_ROOM_CEILING, 'scattering': 0.10},
                             'Wood': walls})
    reference = eyring_of(config)
    # Trace far enough for a 70 dB decay of the slowest band, with room for a non-diffuse tail
    max_time = min(12.0, max(1.0, 1.5 * float(np.max(reference)) * 70 / 60))
    return with_settings(config, rays=rays, ism_order=5, max_time=max_time), reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kinds', nargs='*', metavar='{absorption,scattering}', help='default: both')
    parser.add_argument('--data', default='../data', help='folder with catt/<kind>_sweep.csv')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--rays', type=int, default=31_500, help='rays per run (31 500 = 100 per m3)')
    parser.add_argument('--values', type=float, nargs='+', default=SWEEP_VALUES, help='wall coefficients')
    parser.add_argument('--output', default='output/experiments/sweeps')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    args.kinds = args.kinds or ['absorption', 'scattering']
    if set(args.kinds) - {'absorption', 'scattering'}:  # argparse cannot combine choices with an empty list
        parser.error(f"unknown sweep: {' '.join(args.kinds)}")
    for kind in args.kinds:
        sweep(kind, args, output)
    write_run_info(output, started, config='configs/test_room.toml', kinds=args.kinds, seeds=args.seeds,
                   rays=args.rays, values=args.values)
    return 0


def sweep(kind: str, args, output: Path):
    args = argparse.Namespace(**{**vars(args), 'kind': kind})
    reference_table = read_sweep_table(Path(args.data) / 'catt' / f'{args.kind}_sweep.csv')

    rows, curves = [], {'receiver_1': [], 'receiver_1_std': [], 'room_mean': [], 'eyring': []}
    for value in args.values:
        config, reference = sweep_config(args.kind, value, args.rays)
        t30 = decay_over_seeds(config, args.seeds, f'{args.kind} {value}')['t30']  # (S, R, B)
        receiver_1, receiver_1_std = seed_spread(t30[:, 0])
        room_mean, _ = seed_spread(t30.mean(axis=1))
        for key, values in zip(curves, (receiver_1, receiver_1_std, room_mean, reference)):
            curves[key].append(values)
        for b, band in enumerate(OCTAVE_BANDS):
            row = {'wall_value': value, 'band': int(band), 'new_t30_r1': receiver_1[b], 'new_seed_std_r1': receiver_1_std[b],
                   'new_t30_room_mean': room_mean[b], 'eyring': reference[b], 'max_time': config.simulation.max_time}
            for source, key in (('catt', 'catt_r1'), ('thesis_engine', 'thesis_r1')):
                if value in reference_table.get(source, {}):
                    row[key] = reference_table[source][value][b]
            if 'catt_r1' in row:
                row['new_minus_catt'] = row['new_t30_r1'] - row['catt_r1']
                if 'thesis_r1' in row:
                    row['thesis_minus_catt'] = row['thesis_r1'] - row['catt_r1']
            rows.append(row)

    write_csv(output / f'{args.kind}_sweep.csv', rows)
    plot(args.kind, args.values, {key: np.array(value) for key, value in curves.items()}, reference_table,
         output / f'{args.kind}_sweep.png')
    print_summary(args.kind, rows)
    print(f"\nWrote {output / f'{args.kind}_sweep.csv'} and {output / f'{args.kind}_sweep.png'}")


def print_summary(kind: str, rows):
    symbol = 'alpha' if kind == 'absorption' else 's'
    print(f"\n{symbol:>6}{'band':>6}{'CATT':>7}{'thesis':>8}{'new R1':>8}{'±seed':>7}{'Eyring':>8}"
          f"{'thesis-CATT':>13}{'new-CATT':>10}")
    for row in rows:
        print(f"{row['wall_value']:>6.2f}{row['band']:>6}{cell(row, 'catt_r1', 7)}{cell(row, 'thesis_r1', 8)}"
              f"{cell(row, 'new_t30_r1', 8)}{cell(row, 'new_seed_std_r1', 7, '.3f')}{cell(row, 'eyring', 8)}"
              f"{cell(row, 'thesis_minus_catt', 13, '+.2f')}{cell(row, 'new_minus_catt', 10, '+.2f')}")


def plot(kind: str, values, curves, reference_table, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for b, ax in enumerate(axes.ravel()):
        for source, style, label, colour in (('catt', 'o-', 'CATT-Acoustic', COLOURS['catt']),
                                             ('thesis_engine', 's--', 'Thesis engine', COLOURS['thesis'])):
            table = reference_table.get(source, {})
            known = [v for v in values if v in table]
            if known:
                ax.plot(known, [table[v][b] for v in known], style, color=colour, label=label, ms=4)
        ax.errorbar(values, curves['receiver_1'][:, b], yerr=curves['receiver_1_std'][:, b], fmt='^-',
                    color=COLOURS['new'], capsize=2, ms=4, label='New engine')
        ax.plot(values, curves['eyring'][:, b], ':', color=COLOURS['eyring'], label='Eyring')
        ax.set_xscale('log')
        ax.set_title(f'{BAND_LABELS[b]} Hz', fontsize=10, loc='left')
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(f"Wall {'absorption' if kind == 'absorption' else 'scattering'} coefficient")
    for ax in axes[:, 0]:
        ax.set_ylabel('T30 at receiver 1 (s)')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
