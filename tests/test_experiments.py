"""Smoke tests of the experiment scripts; they need the private models and the data outside the repository."""
import sys
from pathlib import Path

import numpy as np
import pytest

NEEDS_DATA = pytest.mark.skipif(not Path('models/private/Real_ClassRoom.3dm').exists()
                                or not Path('../data/measured/classroom.csv').exists(),
                                reason='needs the private models and ../data/measured')


@NEEDS_DATA
def test_validation_rooms_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from validation_rooms import main
    assert main(['--seeds', '2', '--rays-per-m3', '2', '--no-scattering-band', '--cases', 'classroom',
                 '--output', str(tmp_path)]) == 0
    for name in ('summary.csv', 'receivers.csv', 'validation.png', 'run_info.json'):
        assert (tmp_path / name).exists()
    sys.modules.pop('validation_rooms')


def test_wall_sweep_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from wall_sweep import main
    for kind in ('absorption', 'scattering'):
        assert main([kind, '--seeds', '1', '--rays', '300', '--values', '0.1', '0.5', '--output', str(tmp_path)]) == 0
        assert (tmp_path / f'{kind}_sweep.csv').exists() and (tmp_path / f'{kind}_sweep.png').exists()
    sys.modules.pop('wall_sweep')


def test_ray_count_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from ray_count import main
    assert main(['--configs', 'configs/test_room.toml', '--seeds', '2', '--densities', '1', '3',
                 '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'ray_count_test_room.csv').exists() and (tmp_path / 'run_info.json').exists()
    sys.modules.pop('ray_count')


def test_verification_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from verification import main
    main(['--quick', '--output', str(tmp_path)])  # quick runs may miss a stochastic criterion
    assert (tmp_path / 'verification.csv').exists() and (tmp_path / 'verification.png').exists()
    sys.modules.pop('verification')


@pytest.mark.parametrize('room', ['classroom', 'lecture_hall', 'mf_hall'])
def test_paper_configs_assign_every_layer_of_the_run_configs(room):
    from gasim.config import load_config
    code, paper = load_config(f'configs/{room}.toml'), load_config(f'configs/{room}_paper.toml')
    assert paper.model == code.model
    assert sorted(paper.materials.names) == sorted(code.materials.names)


def test_renovation_changes_only_the_treated_layers():
    from gasim.config import load_config
    before, after = load_config('configs/mf_hall_paper.toml'), load_config('configs/mf_hall_renovated.toml')
    assert after.model == before.model and after.materials.names == before.materials.names
    changed = {name for m, name in enumerate(before.materials.names)
               if (before.materials.absorption[m] != after.materials.absorption[m]).any()}
    assert changed == {'Wood', 'Woolen', 'White'}


@NEEDS_DATA
def test_material_sources_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from material_sources import main
    assert main(['--cases', 'classroom', '--seeds', '1', '--rays-per-m3', '2', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'material_sources.csv').exists() and (tmp_path / 'material_sources.png').exists()
    sys.modules.pop('material_sources')


@pytest.mark.skipif(not Path('models/private/Real_Work.3dm').exists(), reason='needs the private hall model')
def test_surface_statistics_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from surface_statistics import main
    assert main(['--rays-per-m3', '2', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'surface_statistics.csv').exists() and (tmp_path / 'surface_statistics.png').exists()
    sys.modules.pop('surface_statistics')


def test_budget_adds_independent_uncertainties_in_quadrature(monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from input_sensitivity import budget
    ones = np.ones(6)
    changes = {'alpha A': (-3 * ones, 2 * ones), 'alpha B': (4 * ones, -1 * ones),
               'scattering': (-12 * ones, 0 * ones), 'air': (-1 * ones, 2 * ones)}
    groups = budget(changes)
    np.testing.assert_allclose(groups['materials'], 5.0)  # sqrt(3^2 + 4^2)
    np.testing.assert_allclose(groups['total'], np.sqrt(5 ** 2 + 12 ** 2 + 2 ** 2))
    sys.modules.pop('input_sensitivity')


@NEEDS_DATA
def test_input_sensitivity_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from input_sensitivity import main
    assert main(['--cases', 'classroom', '--seeds', '1', '--rays-per-m3', '1', '--output', str(tmp_path)]) == 0
    for name in ('factors.csv', 'budget.csv', 'input_sensitivity.png', 'run_info.json'):
        assert (tmp_path / name).exists()
    sys.modules.pop('input_sensitivity')


@NEEDS_DATA
def test_seats_are_merged_into_the_floor_by_area(monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from scenarios import LITERATURE, SEATING, apply

    from gasim.config import load_config
    from gasim.geometry import load_room
    config = load_config('configs/classroom_paper.toml')
    floor = config.materials.names.index('Floor')
    area = load_room(config)[0].area_by_material()['Floor']
    merged = apply(config, 'classroom', [('seats', 'wooden_chairs')])
    seat = np.asarray(LITERATURE['wooden_chairs'][0])
    expected = config.materials.absorption[floor] + SEATING['classroom'] / area * (seat - config.materials.absorption[floor])
    np.testing.assert_allclose(merged.materials.absorption[floor], expected)
    # The absorption area grows by exactly the seating area times the difference of the coefficients
    np.testing.assert_allclose(area * (merged.materials.absorption[floor] - config.materials.absorption[floor]),
                               SEATING['classroom'] * (seat - config.materials.absorption[floor]))
    sys.modules.pop('scenarios')


@NEEDS_DATA
def test_scenarios_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from scenarios import main
    assert main(['--cases', 'classroom', '--seeds', '1', '--rays-per-m3', '1', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'scenarios.csv').exists() and (tmp_path / 'scenarios.png').exists()
    sys.modules.pop('scenarios')
