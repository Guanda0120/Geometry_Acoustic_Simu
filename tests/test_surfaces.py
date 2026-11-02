"""Surfaces: grouping rules and the statistics per surface (gasim.surfaces, gasim.analysis.surfaces)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import box_polygons

from gasim.air import energy_attenuation
from gasim.cli import main
from gasim.config import SimulationSettings, load_config
from gasim.geometry import load_room
from gasim.geometry.build import build_room
from gasim.io.results import load_result, save_result
from gasim.materials import material_table
from gasim.propagation.hybrid import hybrid
from gasim.propagation.image_sources import find_image_sources, reflections_by_surface
from gasim.propagation.rays import trace_rays
from gasim.simulate import simulate
from gasim.surfaces import OTHER, assign_surfaces, surface_rules

C = 343.0
NO_AIR = np.zeros(6)
BOX = (4.0, 3.0, 2.5)
FACES = {'floor': {'normal': [0, 0, 1]}, 'ceiling': {'normal': [0, 0, -1]}, 'x0': {'x': [0, 0]},
         'x4': {'x': [4, 4]}, 'y0': {'y': [0, 0]}, 'y3': {'y': [3, 3]}}
FACE_AREAS = [12.0, 12.0, 7.5, 7.5, 10.0, 10.0]


def box_room(absorption=0.1, scattering=0.0, rules=FACES):
    materials = material_table({'Wall': {'absorption': absorption, 'scattering': scattering}})
    room = build_room(box_polygons(*BOX), materials, [1.1, 1.3, 1.2], [[2.7, 1.9, 1.4], [3.1, 0.8, 1.6]])
    names, index = assign_surfaces(room.poly_vertices, room.poly_layer, room.poly_normal, surface_rules(rules),
                                   materials.names)
    return dataclasses.replace(room, surface_names=names, poly_surface=index)


def settings(**values):
    defaults = {'rays': 3000, 'ism_order': 2, 'receiver_radius': 0.3, 'time_step': 1e-3, 'max_time': 1.0,
                'seed': 1}
    return SimulationSettings(**{**defaults, **values})


def test_rules_split_the_box_into_its_faces():
    room = box_room()
    assert room.surface_names == tuple(FACES)
    np.testing.assert_allclose(room.area_by_surface(), FACE_AREAS)


def test_without_rules_every_layer_is_a_surface():
    materials = material_table({'Wall': {'absorption': 0.1}, 'Window': {'absorption': 0.05}})
    room = build_room(box_polygons(*BOX), materials, [1.1, 1.3, 1.2], [[2.7, 1.9, 1.4]])
    assert room.surface_names == ('Wall', 'Window')
    np.testing.assert_array_equal(room.poly_surface, room.poly_material)
    np.testing.assert_allclose(room.area_by_surface(), [room.surface_area, 0.0])


def test_first_matching_rule_wins_and_the_rest_is_other():
    room = box_room(rules={'floor': {'normal': [0, 0, 1], 'max_angle': 10}, 'side': {'layers': ['Glass']}})
    assert room.surface_names == ('floor', 'side', OTHER)
    np.testing.assert_allclose(room.area_by_surface(), [12.0, 0.0, room.surface_area - 12.0])
    room = box_room(rules={'everything': {}, 'floor': {'normal': [0, 0, 1]}})
    np.testing.assert_allclose(room.area_by_surface(), [room.surface_area, 0.0])


@pytest.mark.parametrize('rules, message', [
    ({'a': {'x': [1, 0]}}, r'\[min, max\]'),
    ({'a': {'colour': 'red'}}, 'Unknown key'),
    ({OTHER: {}}, 'reserved'),
    ({'a': {'max_angle': 30}}, 'needs a normal'),
    ({'a': {'normal': [0, 0, 0]}}, 'non-zero'),
    ({'a': {'layers': 'Wall'}}, 'list of layer names'),
])
def test_invalid_rules_are_rejected(rules, message):
    with pytest.raises(ValueError, match=message):
        surface_rules(rules)


def test_config_surfaces_reach_the_room():
    room, _ = load_room(load_config('configs/test_room.toml'))
    assert room.surface_names == ('floor', 'ceiling', 'wall_x0', 'wall_x9', 'wall_y0', 'wall_y7')
    np.testing.assert_allclose(room.area_by_surface(), [63, 63, 35, 35, 45, 45])


def test_ray_statistics_add_up():
    alpha = np.array([0.1, 0.2, 0.3, 0.3, 0.4, 0.5])
    rays = trace_rays(box_room(absorption=alpha, scattering=0.4), settings(), C, NO_AIR)
    assert rays.band_groups == 1 and rays.surface_hits[0].sum() == rays.reflections
    np.testing.assert_allclose(rays.surface_absorbed.sum(axis=-1), rays.absorbed, rtol=1e-9)
    np.testing.assert_allclose(rays.surface_absorbed, rays.surface_incident * alpha[:, None], rtol=1e-12)


def test_received_energy_adds_up_to_the_histogram():
    room = box_room(absorption=0.2, scattering=[0.1, 0.1, 0.3, 0.3, 0.6, 0.9])
    output = hybrid(room, settings(), C, energy_attenuation(20.0, 50.0, 101.325))
    np.testing.assert_allclose(output.received.sum(axis=-1) + output.direct, output.energy.sum(axis=-1),
                               rtol=1e-10)
    assert np.all(output.direct > 0) and np.all(output.received > 0)


def test_hits_per_area_are_uniform_in_a_diffuse_field():
    room = box_room(absorption=0.05, scattering=1.0)
    hits = trace_rays(room, settings(rays=4000), C, NO_AIR).surface_hits[0]
    density = (hits / room.area_by_surface()) / (hits.sum() / room.surface_area)
    np.testing.assert_allclose(density, 1.0, atol=0.03)


def test_image_source_reflections_per_surface():
    room = box_room()
    paths = find_image_sources(room, 2, NO_AIR)
    counts = reflections_by_surface(paths, room.poly_surface, len(room.receivers), room.n_surfaces)
    np.testing.assert_array_equal(counts[:, 0], 1)  # one first-order reflection on every face
    np.testing.assert_array_equal(counts[:, 1].sum(axis=-1), [2 * paths.counts(r)[2] for r in range(2)])


def test_statistics_are_saved_and_read_back(tmp_path):
    config = load_config('configs/test_room.toml')
    config = dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, rays=500))
    result = simulate(config)
    save_result(result, tmp_path / 'result.npz')
    stats, loaded = result.surfaces, load_result(tmp_path / 'result.npz').surfaces
    assert loaded.names == stats.names
    for name in ('area', 'ism_reflections', 'ray_hits', 'incident', 'absorbed', 'received', 'direct'):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(stats, name))
    np.testing.assert_allclose(loaded.received_share().sum(axis=-1) + loaded.direct / (
        loaded.received.sum(axis=-1) + loaded.direct), 1.0)


def test_cli_prints_the_surface_table(tmp_path, capsys):
    output = tmp_path / 'result.npz'
    assert main(['run', 'configs/test_room.toml', '--output', str(output)]) == 0
    capsys.readouterr()
    assert main(['surfaces', str(output), '--band', '500']) == 0
    text = capsys.readouterr().out
    assert 'wall_y7' in text and 'direct sound' in text and '500 Hz' in text


@pytest.mark.skipif(not Path('models/private/Real_Work.3dm').exists(), reason='needs the private hall model')
def test_hall_walls_match_the_areas_of_the_paper():
    room, _ = load_room(load_config('configs/mf_hall_paper.toml'))
    areas = dict(zip(room.surface_names, room.area_by_surface()))
    # Paper Fig. 9: ceiling 594, floor 552, wall A 236, wall B 274, wall C 163, wall D 160 m2
    expected = {'ceiling': 594, 'wall_A': 236, 'wall_B': 274, 'wall_C': 163, 'wall_D': 160}
    for name, area in expected.items():
        assert abs(areas[name] - area) < 1.5, name
    assert abs(areas['floor'] - 552) < 4  # the model's floor (stage and its riser included) is 555.7 m2
    assert OTHER not in areas
