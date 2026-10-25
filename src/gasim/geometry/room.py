"""The room as flat arrays, ready for the propagation kernels."""
from dataclasses import dataclass

import numpy as np

from gasim.materials import MaterialTable


@dataclass(frozen=True)
class Room:
    """
    Boundary polygons, their triangles and materials, source and receivers.

    Normals point into the room and each polygon is wound counter-clockwise about its normal,
    so a triangle (a, b, c) has (b - a) x (c - a) along the polygon normal.
    """
    poly_vertices: tuple  # P arrays of shape (n_i, 3)
    poly_layer: tuple  # P layer names
    poly_normal: np.ndarray  # (P, 3) unit, inward
    poly_d: np.ndarray  # (P,) plane n . x + d = 0
    poly_area: np.ndarray  # (P,) m2
    poly_material: np.ndarray  # (P,) row in materials
    poly_flipped: np.ndarray  # (P,) True if the drawn orientation was reversed on import
    tri_vertices: np.ndarray  # (T, 3, 3)
    tri_polygon: np.ndarray  # (T,) polygon of each triangle
    materials: MaterialTable
    source: np.ndarray  # (3,)
    receivers: np.ndarray  # (R, 3)
    surface_names: tuple = None  # S surface names (gasim.surfaces); default: the material names
    poly_surface: np.ndarray = None  # (P,) surface of each polygon; default: its material

    def __post_init__(self):
        if self.poly_surface is None:
            object.__setattr__(self, 'surface_names', tuple(self.materials.names))
            object.__setattr__(self, 'poly_surface', np.asarray(self.poly_material, dtype=np.int64))

    @property
    def n_surfaces(self):
        return len(self.surface_names)

    @property
    def n_polygons(self):
        return len(self.poly_vertices)

    @property
    def surface_area(self):
        return float(self.poly_area.sum())

    @property
    def volume(self):
        """
        Enclosed volume by the divergence theorem; exact for a closed, consistently oriented mesh
        """
        a, b, c = self.tri_vertices[:, 0], self.tri_vertices[:, 1], self.tri_vertices[:, 2]
        # Inward normals, so the signed sum is negative
        return float(-np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)

    def area_by_material(self):
        """
        :return: {material name: total area, m2}
        """
        areas = np.bincount(self.poly_material, weights=self.poly_area, minlength=len(self.materials.names))
        return dict(zip(self.materials.names, areas.tolist()))

    def area_by_surface(self):
        """
        :return: (S,) total area of each surface, m2
        """
        return np.bincount(self.poly_surface, weights=self.poly_area, minlength=self.n_surfaces)

    def absorption_area(self):
        """
        :return: (N_BANDS,) equivalent absorption area sum(S_i alpha_i), m2
        """
        return self.poly_area @ self.materials.absorption[self.poly_material]
