"""Smoke tests of the experiment scripts; they need the private models and the data outside the repository."""
import sys
from pathlib import Path

import pytest

NEEDS_DATA = pytest.mark.skipif(not Path('models/private/Real_ClassRoom.3dm').exists()
                                or not Path('../data/measured/classroom.csv').exists(),
                                reason='needs the private models and ../data/measured')


@NEEDS_DATA
def test_validation_rooms_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    from validation_rooms import main
    assert main(['--seeds', '1', '--rays-per-m3', '2', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'summary.csv').exists() and (tmp_path / 'validation.png').exists()
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
    assert main(['--config', 'configs/test_room.toml', '--seeds', '2', '--densities', '1', '3',
                 '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'ray_count.csv').exists()
    sys.modules.pop('ray_count')


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
