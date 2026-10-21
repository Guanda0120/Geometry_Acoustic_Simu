import numpy as np
import pytest

from gasim.air import energy_attenuation, speed_of_sound

FREQUENCIES = [500, 1000, 2000, 3000, 4000, 6000, 8000]
# Kuttruff, Room Acoustics, 6th ed., Table 6.1 (after Bass et al. 1995): m in 1e-3 1/m at 20 degC, 101.325 kPa
KUTTRUFF_TABLE_6_1 = {
    40: [0.60, 1.07, 2.58, 5.03, 8.40, 17.71, 30.00],
    50: [0.63, 1.08, 2.28, 4.20, 6.84, 14.26, 24.29],
    60: [0.64, 1.11, 2.14, 3.72, 5.91, 12.08, 20.52],
    70: [0.64, 1.15, 2.08, 3.45, 5.32, 10.62, 17.91],
}


@pytest.mark.parametrize('humidity', sorted(KUTTRUFF_TABLE_6_1))
def test_air_attenuation_matches_reference_table(humidity):
    m = energy_attenuation(20.0, humidity, frequency=FREQUENCIES) * 1e3
    np.testing.assert_allclose(m, KUTTRUFF_TABLE_6_1[humidity], rtol=0.02)


def test_air_attenuation_grows_with_frequency():
    assert np.all(np.diff(energy_attenuation()) > 0)


def test_speed_of_sound():
    assert speed_of_sound(20.0) == pytest.approx(343.2, abs=0.1)
    assert speed_of_sound(0.0) == pytest.approx(331.3)
