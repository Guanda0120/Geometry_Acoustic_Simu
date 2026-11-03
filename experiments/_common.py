"""Helpers shared by the experiment scripts."""
import csv
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from gasim import __version__
from gasim.air import energy_attenuation, speed_of_sound
from gasim.analysis.statistical import eyring, sabine
from gasim.config import Config
from gasim.geometry import load_room
from gasim.materials import material_table
from gasim.simulate import simulate

BAND_LABELS = ['125', '250', '500', '1k', '2k', '4k']
COLOURS = {'measured': '#1f3b73', 'catt': '#1f3b73', 'thesis': '#c0392b', 'new': '#1e8449', 'eyring': 'grey'}

# Test room of the CATT comparison (thesis Tables 4.12 and 4.14, paper Table A4)
TEST_ROOM_FLOOR = [0.15, 0.30, 0.40, 0.75, 0.85, 0.70]  # carpet, layer Mat
TEST_ROOM_CEILING = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]  # gypsum board, layer Plaster
TEST_ROOM_SCATTERING_WALLS_ABSORPTION = [0.10, 0.20, 0.10, 0.20, 0.30, 0.40]  # walls in the scattering sweep
SWEEP_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.90, 0.99]

# The three measured rooms with their documented inputs: case -> (title, config, stem of the data files)
PAPER_ROOMS = {'classroom': ('Classroom', 'configs/classroom_paper.toml', 'classroom'),
               'lecture_hall': ('Lecture hall', 'configs/lecture_hall_paper.toml', 'lecture_hall'),
               'mf_hall': ('Multifunction hall', 'configs/mf_hall_paper.toml', 'mf_hall')}
# Uniform scattering coefficients bounding the undocumented scattering (docs/experiment-plan.md, Sect. 3)
SCATTERING_RANGE = {'classroom': (0.05, 0.40), 'lecture_hall': (0.05, 0.40), 'mf_hall': (0.01, 0.40)}
JND = 0.05  # T30, T20 and EDT (ISO 3382-1)


def read_receiver_table(path: Path):
    """
    :param path: CSV with a receiver column and one column per octave band
    :return: (R, B) array, or None if the file does not exist
    """
    if not path.exists():
        return None
    with path.open() as file:
        rows = list(csv.reader(file))[1:]
    return np.array([[float(value) for value in row[1:]] for row in rows])


def read_sweep_table(path: Path):
    """
    :param path: CSV with columns wall_value, source, then one per octave band
    :return: {source: {wall value: (B,) array}}, empty if the file does not exist
    """
    table = {}
    if path.exists():
        with path.open() as file:
            for row in list(csv.reader(file))[1:]:
                table.setdefault(row[1], {})[float(row[0])] = np.array([float(v) for v in row[2:]])
    return table


def write_csv(path: Path, rows):
    """
    Write a list of dicts; floats with 4 decimals
    """
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


def with_settings(config: Config, **changes):
    """
    :return: Config with some simulation settings changed
    """
    return dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, **changes))


def material_spec(config: Config):
    """
    :return: The [materials] table of a config as a dict, to be changed and passed to with_materials
    """
    materials = config.materials
    return {name: {'absorption': materials.absorption[m].tolist(), 'scattering': materials.scattering[m].tolist()}
            for m, name in enumerate(materials.names)}


def uniform_scattering(config: Config, scattering: float):
    """
    :return: Config with the same absorption and one scattering coefficient on every surface
    """
    return with_materials(config, {name: {**entry, 'scattering': scattering}
                                   for name, entry in material_spec(config).items()})


def with_materials(config: Config, spec: dict):
    """
    :param spec: [materials] table as in a TOML config
    :return: Config with these materials
    """
    return dataclasses.replace(config, materials=material_table(spec))


def eyring_of(config: Config):
    """
    :return: (B,) Eyring reverberation time of the config's room, air absorption included
    """
    room, _ = load_room(config)
    air = config.air
    return eyring(room, speed_of_sound(air.temperature), energy_attenuation(air.temperature, air.humidity, air.pressure))


def sabine_of(config: Config):
    """
    :return: (B,) Sabine reverberation time of the config's room, air absorption included
    """
    room, _ = load_room(config)
    air = config.air
    return sabine(room, speed_of_sound(air.temperature), energy_attenuation(air.temperature, air.humidity, air.pressure))


def write_run_info(output: Path, started: float, **extra):
    """
    Write run_info.json next to the results: command line, gasim version, git commit, run time
    :param started: time.time() at the start of the run
    :param extra: Further JSON-serialisable entries, e.g. configs and seeds
    """
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'src'], capture_output=True, text=True, check=False).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    info = {'command': sys.argv, 'gasim_version': __version__, 'git_commit': commit, 'src_modified': dirty,
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - started, 1), **extra}
    (output / 'run_info.json').write_text(json.dumps(info, indent=2, default=str))


def with_decay_time(config: Config):
    """
    :return: Config traced long enough for a 70 dB decay of the slowest band, with room for a non-diffuse
             tail (1.5 x Eyring), at least as long as the config asks and at most 12 s
    """
    longest = float(np.max(eyring_of(config)))
    return with_settings(config, max_time=min(12.0, max(config.simulation.max_time, 1.5 * longest * 70 / 60)))


def run_seeds(config: Config, seeds: int, rays_per_m3: float, label: str = ''):
    """
    Run a config with seeds 1 .. seeds at a ray density, traced long enough (with_decay_time); the same
    seeds for every config give paired runs whose differences are free of most of the Monte Carlo noise
    :return: dict with t30, t20, edt, c50 arrays (S, R, B), the rays and the trace length used
    """
    room, _ = load_room(config)
    config = with_settings(with_decay_time(config), rays=math.ceil(rays_per_m3 * room.volume))
    result = decay_over_seeds(config, seeds, label)
    return {**result, 'rays': config.simulation.rays, 'max_time': config.simulation.max_time, 'volume': room.volume}


def measured_statistics(table):
    """
    :param table: (R, B) measured values per position, or None
    :return: Mean (B,), half-width of the 95 % confidence interval (B,); None, None if there are no data
    """
    if table is None:
        return None, None
    n = np.sum(np.isfinite(table), axis=0)
    mean = np.nanmean(table, axis=0)
    spread = np.nanstd(table, axis=0, ddof=1) if len(table) > 1 else np.full_like(mean, np.nan)
    return mean, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * spread / np.sqrt(n)


def room_value(values: np.ndarray):
    """
    :param values: (S, R, B) one parameter per seed, receiver and band
    :return: (S, B) median over receivers of each seed
    """
    return np.nanmedian(values, axis=1)


def decay_over_seeds(config: Config, seeds: int, label: str = ''):
    """
    Run a config with seeds 1 .. seeds
    :return: dict with t30, t20, edt, c50 arrays of shape (seeds, R, B)
    """
    parameters = {'t30': [], 't20': [], 'edt': [], 'c50': []}
    for seed in range(1, seeds + 1):
        decay = simulate(with_settings(config, seed=seed)).decay()
        for key, values in parameters.items():
            values.append(getattr(decay, key))
        if label:
            print(f"  {label}: seed {seed}/{seeds} done", flush=True)
    return {key: np.array(values) for key, values in parameters.items()}


def seed_spread(values: np.ndarray):
    """
    :param values: (S, ...) one row per seed
    :return: Mean and sample standard deviation over seeds (nan for a single seed)
    """
    mean = np.nanmean(values, axis=0)
    spread = np.nanstd(values, axis=0, ddof=1) if len(values) > 1 else np.full_like(mean, np.nan)
    return mean, spread
