"""Octave-band absorption and scattering coefficients of the room surfaces."""
from dataclasses import dataclass

import numpy as np

from gasim.bands import N_BANDS


@dataclass(frozen=True)
class MaterialTable:
    """
    Coefficients of all materials, one row per material and one column per octave band
    """
    names: tuple
    absorption: np.ndarray
    scattering: np.ndarray

    def index(self, name: str):
        """
        :param name: Material name
        :return: Row of the material in the table
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"Material {name!r} is not defined") from None


def band_values(value, quantity: str, material: str):
    """
    Turn one value or one value per band into a checked band array
    :param value: Number or list of N_BANDS numbers
    :param quantity: 'absorption' or 'scattering', for error messages
    :param material: Material name, for error messages
    :return: (N_BANDS,) array within [0, 1]
    """
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(N_BANDS, float(values))
    if values.shape != (N_BANDS,):
        raise ValueError(f"{quantity} of {material!r} needs 1 or {N_BANDS} values, got {values.size}")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{quantity} of {material!r} must lie in [0, 1], got {values.tolist()}")
    return values


def material_table(spec: dict):
    """
    Build the material table from the [materials] section of a config
    :param spec: {name: {'absorption': ..., 'scattering': ...}}
    :return: MaterialTable
    """
    names, absorption, scattering = [], [], []
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Material {name!r} must be a table with absorption and scattering")
        unknown = set(entry) - {'absorption', 'scattering'}
        if unknown:
            raise ValueError(f"Unknown key(s) for material {name!r}: {', '.join(sorted(unknown))}")
        if 'absorption' not in entry:
            raise ValueError(f"Material {name!r} has no absorption")
        names.append(name)
        absorption.append(band_values(entry['absorption'], 'absorption', name))
        scattering.append(band_values(entry.get('scattering', 0.0), 'scattering', name))
    if not names:
        raise ValueError("No materials defined")
    return MaterialTable(tuple(names), np.array(absorption), np.array(scattering))
