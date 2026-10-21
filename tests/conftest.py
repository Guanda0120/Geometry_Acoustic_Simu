import os

import numpy as np
import pytest

from gasim.geometry.build import build_room
from gasim.materials import material_table

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    """Config paths are relative to the repository root, so run every test from there."""
    monkeypatch.chdir(REPO_ROOT)


def box_polygons(lx=4.0, ly=3.0, lz=2.5, seed=0):
    """
    Six faces of an lx x ly x lz box on layer 'Wall', each wound in a random direction
    """
    corners = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]
    rng = np.random.default_rng(seed)
    return [('Wall', corners[f][::-1] if rng.random() < 0.5 else corners[f]) for f in faces]


@pytest.fixture
def box_materials():
    return material_table({'Wall': {'absorption': 0.1}, 'Window': {'absorption': 0.05}})


@pytest.fixture
def make_box_room(box_materials):
    """
    Factory for box rooms with the source at (1, 1, 1.2) and one receiver at (lx - 1, ly - 1, 1.2)
    """
    def make(lx=4.0, ly=3.0, lz=2.5, extra=(), drop_last=False):
        polygons = box_polygons(lx, ly, lz)
        polygons = (polygons[:-1] if drop_last else polygons) + list(extra)
        return build_room(polygons, box_materials, [1.0, 1.0, 1.2], [[lx - 1.0, ly - 1.0, 1.2]])
    return make
