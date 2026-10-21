import numpy as np
import pytest

from gasim.air import energy_attenuation, speed_of_sound
from gasim.analysis.statistical import eyring, sabine
from gasim.config import load_config
from gasim.geometry import load_room


@pytest.fixture(scope='module')
def test_room():
    room, _ = load_room(load_config('configs/test_room.toml'))
    return room


def test_eyring_agrees_with_legacy_engine(test_room):
    # The legacy code used T = 0.161 V / A and no air absorption; its values are pinned in tests/test_legacy.py
    legacy_speed = 24 * np.log(10) / 0.161
    np.testing.assert_allclose(eyring(test_room, legacy_speed), [1.855, 1.015, 1.242, 0.668, 0.497, 0.403],
                               atol=2e-3)


def test_sabine_exceeds_eyring(test_room):
    c = speed_of_sound()
    assert np.all(sabine(test_room, c) > eyring(test_room, c))


def test_air_absorption_shortens_high_frequencies(test_room):
    c = speed_of_sound()
    change = eyring(test_room, c, energy_attenuation()) / eyring(test_room, c) - 1
    assert np.all(change < 0)
    assert change[-1] < change[0]


def test_sabine_formula_for_a_box(make_box_room):
    # 10 x 10 x 10 m, alpha = 0.1 everywhere: T = 24 ln(10) V / (c S alpha)
    room = make_box_room(lx=10.0, ly=10.0, lz=10.0)
    expected = 24 * np.log(10) * 1000 / (343.0 * 600 * 0.1)
    np.testing.assert_allclose(sabine(room, 343.0), expected)
