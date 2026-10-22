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
