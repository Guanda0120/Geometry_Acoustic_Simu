"""
Validation rooms: new engine against the measurements and the thesis engine (paper Fig. 4 and Table 2).

    uv run python experiments/validation_rooms.py [--data ../data] [--seeds 5] [--rays-per-m3 100]

Needs the private models in models/private/ and the measured data outside the repository
(data/measured/<room>.csv, data/thesis_simulated/<room>.csv, see data/README.md there). Materials are
those of configs/<room>.toml. Writes summary.csv, receivers.csv and validation.png to --output.
"""
import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.air import energy_attenuation, speed_of_sound
from gasim.analysis.statistical import eyring
from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config
from gasim.geometry import load_room
from gasim.simulate import simulate

ROOMS = {'classroom': 'Classroom', 'lecture_hall': 'Lecture hall', 'mf_hall': 'Multifunction hall'}
JND = 5.0  # %, just-noticeable difference of reverberance (ISO 3382-1)
BAND_LABELS = ['125', '250', '500', '1k', '2k', '4k']


def read_table(path: Path):
    """
    :param path: CSV with a receiver column and one column per octave band
    :return: (R, B) array, or None if the file does not exist
    """
    if not path.exists():
        return None
    with path.open() as file:
        rows = list(csv.reader(file))[1:]
    return np.array([[float(value) for value in row[1:]] for row in rows])


def run_room(name: str, seeds: int, rays_per_m3: float):
    """
    Simulate one room with several seeds
    :return: dict with T30/T20/EDT arrays (S, R, B), Eyring (B,), rays and volume
    """
    config = load_config(f'configs/{name}.toml')
    room, _ = load_room(config)
    rays = math.ceil(rays_per_m3 * room.volume)
    parameters = {'t30': [], 't20': [], 'edt': []}
    for seed in range(1, seeds + 1):
        settings = dataclasses.replace(config.simulation, rays=rays, seed=seed)
        decay = simulate(dataclasses.replace(config, simulation=settings)).decay()
        for key, values in parameters.items():
            values.append(getattr(decay, key))
        print(f"  {name}: seed {seed}/{seeds} done", flush=True)
    air = config.air
    reference = eyring(room, speed_of_sound(air.temperature), energy_attenuation(air.temperature, air.humidity,
                                                                                  air.pressure))
    return {**{key: np.array(value) for key, value in parameters.items()}, 'eyring': reference, 'rays': rays,
            'volume': room.volume, 'ism_order': config.simulation.ism_order}


def relative_error(simulated, measured):
    return 100 * (simulated - measured) / measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/ and thesis_simulated/')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--output', default='output/experiments/validation')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary, receivers, results = [], [], {}
    for name, title in ROOMS.items():
        if not load_config(f'configs/{name}.toml').model.exists():
            print(f"skip {name}: private model not available")
            continue
        measured = read_table(data / 'measured' / f'{name}.csv')
        thesis = read_table(data / 'thesis_simulated' / f'{name}.csv')
        run = run_room(name, args.seeds, args.rays_per_m3)
        results[name] = (title, measured, thesis, run)

        t30 = run['t30'].mean(axis=0)  # (R, B) mean over seeds
        room_mean_per_seed = run['t30'].mean(axis=1)  # (S, B)
        for b, band in enumerate(OCTAVE_BANDS):
            row = {'room': name, 'band': int(band), 'rays': run['rays'], 'volume': round(run['volume'], 1),
                   'new_t30': t30[:, b].mean(), 'new_seed_std': room_mean_per_seed[:, b].std(ddof=1)
                   if args.seeds > 1 else float('nan'),
                   'new_t20': run['t20'].mean(axis=(0, 1))[b], 'new_edt': run['edt'].mean(axis=(0, 1))[b],
                   'eyring': run['eyring'][b]}
            if measured is not None:
                row['measured'] = measured[:, b].mean()
                row['error_new_pct'] = relative_error(row['new_t30'], row['measured'])
                if len(measured) == len(t30):
                    row['max_receiver_error_new_pct'] = np.abs(relative_error(t30[:, b], measured[:, b])).max()
            if thesis is not None:
                row['thesis'] = thesis[:, b].mean()
                if measured is not None:
                    row['error_thesis_pct'] = relative_error(row['thesis'], row['measured'])
            summary.append(row)
            for r in range(len(t30)):
                receivers.append({'room': name, 'receiver': r + 1, 'band': int(band), 'new_t30': t30[r, b],
                                  'measured': measured[r, b] if measured is not None and r < len(measured) else '',
                                  'thesis': thesis[r, b] if thesis is not None and r < len(thesis) else ''})

    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'receivers.csv', receivers)
    plot(results, output / 'validation.png')
    print_summary(summary)
    print(f"\nWrote {output / 'summary.csv'}, {output / 'receivers.csv'}, {output / 'validation.png'}")
    return 0


def write_csv(path: Path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f'{value:.4f}' if isinstance(value, float) else value) for key, value in row.items()})


def cell(row, key, width, fmt='.2f'):
    """
    One right-aligned table cell, '-' for missing values
    """
    value = row.get(key)
    if isinstance(value, float) and not math.isnan(value):
        return format(value, fmt).rjust(width)
    return '-'.rjust(width)


def print_summary(summary):
    print(f"\n{'room':<13}{'band':>6}{'measured':>10}{'thesis':>8}{'new T30':>9}{'±seed':>7}{'Eyring':>8}"
          f"{'err thesis':>12}{'err new':>9}{'JND new':>9}")
    for row in summary:
        jnd = abs(row['error_new_pct']) / JND if 'error_new_pct' in row else float('nan')
        print(f"{row['room']:<13}{row['band']:>6}{cell(row, 'measured', 10)}{cell(row, 'thesis', 8)}"
              f"{cell(row, 'new_t30', 9)}{cell(row, 'new_seed_std', 7, '.3f')}{cell(row, 'eyring', 8)}"
              f"{cell(row, 'error_thesis_pct', 11, '+.1f')}%{cell(row, 'error_new_pct', 8, '+.1f')}%{jnd:>9.1f}")


def plot(results, path: Path):
    fig, axes = plt.subplots(1, max(len(results), 1), figsize=(4.2 * max(len(results), 1), 3.6), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    for ax, (name, (title, measured, thesis, run)) in zip(axes[0], results.items()):
        if measured is not None:
            ax.fill_between(x, measured.min(axis=0), measured.max(axis=0), color='#1f3b73', alpha=0.15, lw=0)
            ax.plot(x, measured.mean(axis=0), 'o-', color='#1f3b73', label='Measured')
        if thesis is not None:
            ax.plot(x, thesis.mean(axis=0), 's--', color='#c0392b', label='Thesis engine')
        t30 = run['t30'].mean(axis=1)  # (S, B) room mean per seed
        ax.errorbar(x, t30.mean(axis=0), yerr=t30.std(axis=0, ddof=1) if len(t30) > 1 else None, fmt='^-',
                    color='#1e8449', capsize=3, label='New engine')
        ax.plot(x, run['eyring'], ':', color='grey', label='Eyring')
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_title(f"{title}, {run['volume']:.0f} m$^3$", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('T30 (s)')
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
