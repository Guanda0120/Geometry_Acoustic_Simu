"""Run configuration, read from TOML files such as configs/test_room.toml."""
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from gasim.materials import MaterialTable, material_table
from gasim.surfaces import surface_rules


@dataclass(frozen=True)
class SimulationSettings:
    """
    Numerical settings of a run
    """
    rays: int = 10000
    ism_order: int = 3
    receiver_radius: float = 0.2  # m
    time_step: float = 1e-3  # s, histogram bin width
    max_time: float = 3.0  # s
    seed: int = 1

    def __post_init__(self):
        if self.rays < 1:
            raise ValueError("simulation.rays must be at least 1")
        if self.ism_order < 0:
            raise ValueError("simulation.ism_order must not be negative")
        if self.receiver_radius <= 0 or self.time_step <= 0:
            raise ValueError("simulation.receiver_radius and simulation.time_step must be positive")
        if self.max_time <= self.time_step:
            raise ValueError("simulation.max_time must be larger than simulation.time_step")


@dataclass(frozen=True)
class AirConditions:
    """
    Air conditions for the speed of sound and the air absorption
    """
    temperature: float = 20.0  # degC
    humidity: float = 50.0  # %
    pressure: float = 101.325  # kPa

    def __post_init__(self):
        if not 0 <= self.humidity <= 100:
            raise ValueError("air.humidity must lie in [0, 100] %")
        if self.pressure <= 0:
            raise ValueError("air.pressure must be positive")


@dataclass(frozen=True)
class Config:
    """
    A complete run configuration
    """
    model: Path
    materials: MaterialTable
    simulation: SimulationSettings
    air: AirConditions
    path: Path  # the config file itself
    surfaces: tuple = ()  # SurfaceRule per [surfaces] entry; empty: one surface per layer


def _section(cls, values: dict, name: str):
    """
    Build a settings dataclass from one TOML table, rejecting unknown keys
    :param cls: Dataclass type
    :param values: TOML table
    :param name: Table name, for error messages
    :return: cls instance
    """
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path):
    """
    Read and check a TOML config; relative paths are taken from the working directory (the repository root)
    :param path: Config file
    :return: Config
    """
    path = Path(path)
    with path.open('rb') as file:
        data = tomllib.load(file)

    unknown = set(data) - {'room', 'materials', 'surfaces', 'simulation', 'air'}
    if unknown:
        raise ValueError(f"Unknown section(s) in {path}: {', '.join(sorted(unknown))}")
    room = data.get('room', {})
    if set(room) != {'model'}:
        raise ValueError("[room] needs exactly one key: model")

    return Config(model=Path(room['model']),
                  materials=material_table(data.get('materials', {})),
                  simulation=_section(SimulationSettings, data.get('simulation', {}), 'simulation'),
                  air=_section(AirConditions, data.get('air', {}), 'air'),
                  path=path,
                  surfaces=surface_rules(data.get('surfaces', {})))
