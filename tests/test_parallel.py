"""Parallel execution: results must not depend on the number of threads (T-repro)."""
import numba
import numpy as np
import pytest
from conftest import box_polygons

from gasim.config import SimulationSettings
from gasim.geometry.build import build_room
from gasim.materials import material_table
from gasim.propagation.image_sources import find_image_sources
from gasim.propagation.rays import trace_rays


@pytest.fixture
def room():
    materials = material_table({'Wall': {'absorption': [0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
                                         'scattering': [0.1, 0.1, 0.3, 0.3, 0.5, 0.5]}})
    return build_room(box_polygons(5.0, 4.0, 3.0), materials, [1.13, 1.31, 1.17],
                      [[3.71, 2.93, 1.57], [4.23, 0.97, 2.11]])


def with_threads(n, function, *args):
    previous = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        return function(*args)
    finally:
        numba.set_num_threads(previous)


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2, reason='needs more than one thread')
def test_rays_do_not_depend_on_the_thread_count(room):
    settings = SimulationSettings(rays=3000, receiver_radius=0.3, max_time=0.5, seed=4)
    one = with_threads(1, trace_rays, room, settings, 343.0, np.zeros(6))
    many = with_threads(numba.config.NUMBA_NUM_THREADS, trace_rays, room, settings, 343.0, np.zeros(6))
    np.testing.assert_array_equal(one.energy, many.energy)
    np.testing.assert_array_equal(one.absorbed, many.absorbed)
    assert one.reflections == many.reflections


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2, reason='needs more than one thread')
def test_image_sources_do_not_depend_on_the_thread_count(room):
    one = with_threads(1, find_image_sources, room, 4, np.zeros(6))
    many = with_threads(numba.config.NUMBA_NUM_THREADS, find_image_sources, room, 4, np.zeros(6))
    np.testing.assert_array_equal(one.points, many.points)
    np.testing.assert_array_equal(one.energy, many.energy)
