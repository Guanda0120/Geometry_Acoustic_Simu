"""Helpers shared by the experiment scripts."""
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

from gasim.air import energy_attenuation, speed_of_sound
from gasim.analysis.statistical import eyring
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


def decay_over_seeds(config: Config, seeds: int, label: str = ''):
    """
    Run a config with seeds 1 .. seeds
    :return: dict with t30, t20, edt arrays of shape (seeds, R, B)
    """
    parameters = {'t30': [], 't20': [], 'edt': []}
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
