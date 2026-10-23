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
