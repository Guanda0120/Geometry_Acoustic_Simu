"""
Statistics of a run per surface (gasim.surfaces): where reflections happen and where the energy goes.

Counts reproduce the reflection analysis of the thesis; the energy measures weight every reflection by
the energy it carries:

- image-source reflection points per surface and order (specular paths up to ism_order);
- ray hits per surface; in a diffuse field the hits per m2 are the same on every surface;
- energy arriving at and absorbed by each surface, as fractions of the emitted energy;
- energy reaching each receiver with the surface as the last reflection (image sources, rays and
  diffuse rain), as a share of the received energy including the direct sound.
"""
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class SurfaceStatistics:
    names: tuple  # (S,) surface names
    area: np.ndarray  # (S,) m2
    ism_reflections: np.ndarray  # (R, N, S) image-source reflection points per receiver and order 1 .. N
    ray_hits: np.ndarray  # (B, S) ray hits, by the rays that carried each band
    incident: np.ndarray  # (B, S) energy arriving at each surface / emitted energy
    absorbed: np.ndarray  # (B, S) energy absorbed by each surface / emitted energy
    received: np.ndarray  # (R, B, S) received energy by last reflecting surface, 1/m2 per unit source energy
    direct: np.ndarray  # (R, B) direct sound, 1/m2 per unit source energy

    def per_area(self, values: np.ndarray):
        """
        :param values: (..., S) array
        :return: values / area, nan for surfaces without area
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.area > 0, values / self.area, np.nan)

    def relative_density(self, values: np.ndarray):
        """
        :param values: (..., S) counts or energies per surface
        :return: Value per m2 divided by the room average per m2: 1 everywhere in a diffuse field
        """
        total = values.sum(axis=-1, keepdims=True) / self.area.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.per_area(values) / total

    def absorbed_share(self):
        """
        :return: (B, S) share of the energy absorbed by the walls that each surface absorbs
        """
        return self.absorbed / self.absorbed.sum(axis=-1, keepdims=True)

    def received_share(self):
        """
        :return: (R, B, S) share of the received energy (direct sound included) via each last surface
        """
        return self.received / (self.received.sum(axis=-1) + self.direct)[..., None]


def statistics_arrays(statistics: SurfaceStatistics):
    """
    :return: {name: array} for saving, surface names as a string array
    """
    arrays = {f.name: getattr(statistics, f.name) for f in fields(SurfaceStatistics)}
    arrays['names'] = np.array(statistics.names, dtype=str)
    return arrays


def statistics_from_arrays(arrays: dict):
    """
    :param arrays: {name: array} as written by statistics_arrays
    :return: SurfaceStatistics
    """
    values = {f.name: arrays[f.name] for f in fields(SurfaceStatistics)}
    values['names'] = tuple(str(name) for name in values['names'])
    return SurfaceStatistics(**values)
