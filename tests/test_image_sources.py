"""Image sources and the hybrid (docs/rewrite-plan.md, Sect. 5: T-ism, T-hybrid)."""
import numpy as np
import pytest
import rhino3dm
from conftest import box_polygons

from gasim.analysis.decay import decay_parameters
from gasim.cli import main
from gasim.config import SimulationSettings
from gasim.geometry.build import build_room
from gasim.geometry.checks import check_room
from gasim.io.results import Result, load_result, save_result
from gasim.materials import material_table
from gasim.propagation.hybrid import hybrid
from gasim.propagation.image_sources import find_image_sources
from gasim.propagation.rays import trace_rays

C = 343.0
NO_AIR = np.zeros(6)
SIZE = np.array([5.0, 4.0, 3.0])
# Generic positions: no path of order <= 4 touches an edge of the box
SOURCE = [1.13, 1.31, 1.17]
RECEIVERS = [[3.71, 2.93, 1.57], [4.23, 0.97, 2.11]]


def shoebox(absorption=0.2, scattering=0.1, receivers=RECEIVERS, size=SIZE):
    materials = material_table({'Wall': {'absorption': absorption, 'scattering': scattering}})
    return build_room(box_polygons(*size), materials, SOURCE, receivers)


def lattice_lengths(receiver, max_order, size=SIZE, source=SOURCE):
    """
    Path lengths of all images of a rectangular room up to max_order (Allen & Berkley 1979)
    """
    lengths, orders = [], []
    span = range(-max_order, max_order + 1)
    for index in ((i, j, k) for i in span for j in span for k in span):
        order = sum(abs(i) for i in index)
        if order > max_order:
            continue
        image = [i * length + s if i % 2 == 0 else (i + 1) * length - s for i, length, s in zip(index, size, source)]
        lengths.append(np.linalg.norm(np.array(image) - receiver))
        orders.append(order)
    return np.array(lengths), np.array(orders)


def settings(**values):
    defaults = {'rays': 2000, 'ism_order': 3, 'receiver_radius': 0.3, 'time_step': 1e-3, 'max_time': 1.0,
                'seed': 1}
    return SimulationSettings(**{**defaults, **values})


def test_shoebox_matches_the_image_lattice():
    room = shoebox()
    paths = find_image_sources(room, 4, NO_AIR)
    for index, receiver in enumerate(room.receivers):
        np.testing.assert_array_equal(paths.counts(index), [1, 6, 18, 38, 66])  # 4 k2 + 2 images of order k
        lengths, _ = lattice_lengths(receiver, 4)
        np.testing.assert_allclose(np.sort(paths.length[paths.receiver == index]), np.sort(lengths), atol=1e-9)


def test_path_energy():
    attenuation = np.linspace(0.001, 0.006, 6)
    paths = find_image_sources(shoebox(absorption=0.2, scattering=0.1), 3, attenuation)
    expected = (0.8 * 0.9) ** paths.order[:, None] * np.exp(-np.outer(paths.length, attenuation)) \
        / (4 * np.pi * paths.length[:, None] ** 2)
    np.testing.assert_allclose(paths.energy, expected, rtol=1e-12)


def test_reflection_points_obey_the_specular_law():
    room = shoebox()
    paths = find_image_sources(room, 3, NO_AIR)
    for order, planes, points in zip(paths.order, paths.planes, paths.points):
        for j in range(order):
            normal, d = room.poly_normal[planes[j]], room.poly_d[planes[j]]
            point = points[j + 1]
            assert abs(normal @ point + d) < 1e-9
            incoming = (point - points[j]) / np.linalg.norm(point - points[j])
            outgoing = (points[j + 2] - point) / np.linalg.norm(points[j + 2] - point)
            # Mirror law: the difference of the directions is along the normal
            np.testing.assert_allclose(np.cross(outgoing - incoming, normal), 0, atol=1e-9)


def test_order_zero_is_the_direct_sound():
    room = shoebox()
    paths = find_image_sources(room, 0, NO_AIR)
    assert list(paths.order) == [0, 0]
    np.testing.assert_allclose(paths.length, np.linalg.norm(room.receivers - room.source, axis=1))


def l_shaped_room():
    """6 x 6 x 3 m with the 3 x 3 m corner x > 3, y > 3 cut out; source and receiver in different arms"""
    outline = np.array([[0, 0], [6, 0], [6, 3], [3, 3], [3, 6], [0, 6]], dtype=float)
    height = 3.0
    polygons = [('Wall', np.column_stack([outline, np.zeros(6)])), ('Wall', np.column_stack([outline, np.full(6, height)]))]
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        polygons.append(('Wall', np.array([[*a, 0], [*b, 0], [*b, height], [*a, height]])))
    materials = material_table({'Wall': {'absorption': 0.2, 'scattering': 0.2}})
    return build_room(polygons, materials, source=[1.0, 5.2, 1.5], receivers=[[5.1, 1.0, 1.5]])


def test_walls_block_paths_in_an_l_shaped_room():
    room = l_shaped_room()
    assert check_room(room, n_rays=500).ok
    paths = find_image_sources(room, 2, NO_AIR)
    assert 0 not in paths.order  # the inner corner hides the direct sound
    output = hybrid(room, settings(rays=3000, ism_order=2), C, NO_AIR)
    assert output.energy.sum() > 0


def test_hybrid_direct_sound_is_exact():
    room = shoebox(receivers=[RECEIVERS[0]])
    output = hybrid(room, settings(), C, NO_AIR)
    distance = np.linalg.norm(room.receivers[0] - room.source)
    lengths, _ = lattice_lengths(room.receivers[0], 1)
    assert np.sort(lengths)[1] - distance > 2 * C * 1e-3  # first reflection lands two bins later
    k = int(distance / (C * 1e-3))
    np.testing.assert_allclose(output.energy[0, :, k], 1 / (4 * np.pi * distance ** 2), rtol=1e-12)


def test_hybrid_equals_rays_only_in_expectation():
    room = shoebox(absorption=0.2, scattering=0.3, receivers=[RECEIVERS[0]])
    early = slice(0, 60)
    combined = hybrid(room, settings(rays=200_000, max_time=0.06), C, NO_AIR).energy[0, 0, early].sum()
    rays_only = trace_rays(room, settings(rays=200_000, max_time=0.06), C, NO_AIR).energy[0, 0, early].sum()
    assert combined == pytest.approx(rays_only, rel=0.04)

    t30_hybrid = decay_parameters(hybrid(room, settings(rays=20000), C, NO_AIR).energy, 1e-3).t30[0, 0]
    t30_rays = decay_parameters(trace_rays(room, settings(rays=20000), C, NO_AIR).energy, 1e-3).t30[0, 0]
    assert t30_hybrid == pytest.approx(t30_rays, rel=0.02)


def test_result_with_paths_round_trip_and_export(tmp_path):
    room = shoebox()
    output = hybrid(room, settings(rays=500), C, NO_AIR)
    result = Result(energy=output.energy, time_step=1e-3, bands=np.arange(6.0), source=room.source,
                    receivers=room.receivers, metadata={'layers': list(room.poly_layer)}, paths=output.paths)
    save_result(result, tmp_path / 'run.npz')
    loaded = load_result(tmp_path / 'run.npz')
    np.testing.assert_array_equal(loaded.paths.points, output.paths.points)
    np.testing.assert_array_equal(loaded.paths.energy, output.paths.energy)

    assert main(['paths', str(tmp_path / 'run.npz'), str(tmp_path / 'paths.3dm'), '--receiver', '2']) == 0
    model = rhino3dm.File3dm.Read(str(tmp_path / 'paths.3dm'))
    assert len(model.Objects) == output.paths.counts(1).sum()
    assert sorted(layer.Name for layer in model.Layers) == ['Direct', 'Order 1', 'Order 2', 'Order 3']
