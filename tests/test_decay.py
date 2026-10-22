import numpy as np
import pytest

from gasim.analysis.decay import DYNAMIC_RANGE_MARGIN, decay_parameters, fit_decay, schroeder_curve

DT = 1e-3


def exponential(reverberation_time, depth_db=100.0, onset=0.0):
    """
    Energy histogram of a single-slope decay
    :param reverberation_time: s
    :param depth_db: Decay depth before the histogram ends, dB
    :param onset: Empty time before the first arrival, s
    """
    k = np.arange(int(depth_db / 60 * reverberation_time / DT))
    energy = np.exp(-np.log(1e6) * k * DT / reverberation_time)
    return np.concatenate([np.zeros(round(onset / DT)), energy])


def test_schroeder_curve_of_an_exponential_is_a_line():
    # 100 dB deep: over the first 30 dB the truncated tail changes the slope by far less than 1e-6
    level = schroeder_curve(exponential(1.0))
    slope = np.diff(level[:500]) / DT
    np.testing.assert_allclose(slope, -60.0, rtol=1e-6)


@pytest.mark.parametrize('reverberation_time', [0.3, 1.2, 2.5])
def test_single_slope_parameters_are_exact(reverberation_time):
    parameters = decay_parameters(exponential(reverberation_time), DT)
    for value in (parameters.edt, parameters.t20, parameters.t30):
        assert value == pytest.approx(reverberation_time, rel=1e-6)
    assert parameters.xi_t30 == pytest.approx(0, abs=1e-6)
    assert parameters.curvature == pytest.approx(0, abs=1e-4)


def test_onset_delay_is_ignored():
    with_delay = decay_parameters(exponential(1.2, onset=0.03), DT)
    assert with_delay.edt == pytest.approx(1.2, rel=1e-6)
    assert with_delay.t30 == pytest.approx(1.2, rel=1e-6)


def test_double_slope_is_flagged():
    # Fast early decay (T = 0.8 s) handing over to a slow tail (T = 2.0 s) 15 dB down
    k = np.arange(8000)
    energy = np.exp(-np.log(1e6) * k * DT / 0.8) + 10 ** -1.5 * np.exp(-np.log(1e6) * k * DT / 2.0)
    parameters = decay_parameters(energy, DT)
    assert parameters.edt < parameters.t20 < parameters.t30
    assert parameters.curvature > 10
    assert parameters.xi_t30 > 10


def test_insufficient_dynamic_range_gives_nan():
    # 39 dB deep: enough for EDT (10 dB + margin), not for T20 (25 dB + margin) or T30
    energy = exponential(1.0, depth_db=39.0)
    parameters = decay_parameters(energy, DT)
    assert parameters.edt == pytest.approx(1.0, rel=1e-3)
    assert np.isnan(parameters.t20) and np.isnan(parameters.t30)
    assert 'deep' in fit_decay(energy, DT, -5, -25).reason
    # A flat 10-bin response: its Schroeder curve ends at -10 dB
    assert 'does not reach' in fit_decay(np.ones(10), DT, -5, -35).reason


@pytest.mark.parametrize('depth_db', [35 + DYNAMIC_RANGE_MARGIN, 60.0, 100.0])
def test_truncation_bias_is_small_once_evaluated(depth_db):
    assert decay_parameters(exponential(1.0, depth_db=depth_db + 0.1), DT).t30 == pytest.approx(1.0, rel=5e-3)


def test_empty_histogram_gives_nan():
    parameters = decay_parameters(np.zeros(100), DT)
    assert np.isnan(parameters.edt) and np.isnan(parameters.t30)


def test_shapes_follow_the_histogram():
    k = np.arange(2000)
    curves = np.stack([np.exp(-np.log(1e6) * k * DT / t) for t in (0.5, 0.6, 0.7)])
    energy = np.stack([curves] * 4)
    parameters = decay_parameters(energy, DT)
    assert parameters.t20.shape == (4, 3)
    np.testing.assert_allclose(parameters.t20[0], [0.5, 0.6, 0.7], rtol=1e-6)
