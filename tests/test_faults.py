"""The fault switches of gasim.faults (experiment E6): each one changes what it names, and nothing else."""
import sys
from pathlib import Path

import numpy as np
import pytest

from gasim import faults
from gasim.air import attenuation_db, energy_attenuation
from gasim.analysis.decay import decay_parameters
from gasim.faults import FAULTS, reinserted
from gasim.propagation.image_sources import find_image_sources
from gasim.propagation.rays import rain_fraction, trace_rays
from gasim.verification import NO_AIR, C, _box, _settings, exponential_decay


def box(scattering=0.2, absorption=0.2):
    return _box(5.0, 4.0, 3.0, absorption=absorption, scattering=scattering, receivers=[[3.7, 2.9, 1.6]])


def test_no_fault_is_on_by_default_and_blocks_restore_the_state():
    assert faults.active() == frozenset()
    with reinserted('F1', 'F6'):
        assert faults.on('F1') and faults.on('F6') and not faults.on('F2')
        with reinserted('F2'):
            assert faults.active() == {'F1', 'F2', 'F6'}
        assert faults.active() == {'F1', 'F6'}
    assert faults.active() == frozenset()


def test_state_is_restored_after_an_error():
    with pytest.raises(RuntimeError), reinserted('F3'):
        raise RuntimeError
    assert not faults.on('F3')


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError, match='F99'), reinserted('F99'):
        pass


def test_empty_block_leaves_the_engine_unchanged():
    room, settings = box(), _settings(rays=500, ism_order=2)
    plain = trace_rays(room, settings, C, NO_AIR).energy
    with reinserted():
        np.testing.assert_array_equal(trace_rays(room, settings, C, NO_AIR).energy, plain)


def test_f1_makes_the_received_energy_depend_on_the_number_of_rays():
    room = box()
    totals = {}
    with reinserted('F1'):
        for rays in (500, 4000):
            totals[rays] = trace_rays(room, _settings(rays=rays), C, NO_AIR).energy.sum()
    correct = trace_rays(room, _settings(rays=4000), C, NO_AIR).energy.sum()
    # N / (4 pi L2) per hit instead of 1 / (pi r2): the sum grows with N
    assert totals[4000] / totals[500] > 4
    assert abs(totals[4000] / correct - 1) > 0.5


def test_f2_uses_r_over_d_as_the_cosine_of_the_cone():
    with reinserted('F2'):
        assert rain_fraction(0.5, 0.2, 2.0) == pytest.approx(2 * 0.5 * (1 - 0.1))
    assert rain_fraction(0.5, 0.2, 2.0) == pytest.approx(2 * 0.5 * (1 - np.sqrt(1 - 0.01)))


def test_f3_replaces_t30_by_the_crossing_times_and_leaves_the_rest():
    histogram = exponential_decay(1.0, depth_db=60.0)
    correct = decay_parameters(histogram, 1e-3)
    with reinserted('F3'):
        thesis = decay_parameters(histogram, 1e-3)
    # The Schroeder curve of a response truncated at -60 dB bends down and crosses -65 dB early
    assert thesis.t30 < 0.95
    assert correct.t30 == pytest.approx(1.0, abs=0.005)
    assert thesis.edt == correct.edt and thesis.t20 == correct.t20 and thesis.c50 == correct.c50
    # On a deep decay the crossings are 60 dB apart
    with reinserted('F3'):
        assert decay_parameters(exponential_decay(1.0, depth_db=140.0), 1e-3).t30 == pytest.approx(1.0, abs=0.005)


def test_f6_keeps_the_scattered_share_on_the_image_sources():
    room = box(scattering=0.3)
    correct = find_image_sources(room, 2, NO_AIR)
    with reinserted('F6'):
        thesis = find_image_sources(room, 2, NO_AIR)
    np.testing.assert_array_equal(thesis.length, correct.length)
    np.testing.assert_allclose(thesis.energy[:, 0] / correct.energy[:, 0], 0.7 ** -correct.order.astype(float))


def test_f9_is_the_classical_absorption_of_the_thesis_code():
    with reinserted('F9'):
        np.testing.assert_allclose(attenuation_db([125, 4000], humidity=30.0), [1.5625e-6, 1.6e-3])
        thesis = energy_attenuation()
    assert np.all(thesis < energy_attenuation())
    assert (energy_attenuation() / thesis).max() > 100  # 125 Hz


def test_f16_changes_path_lengths_from_the_second_order():
    room = box()
    correct = find_image_sources(room, 3, NO_AIR)
    with reinserted('F16'):
        thesis = find_image_sources(room, 3, NO_AIR)
    low = correct.order <= 1
    np.testing.assert_allclose(thesis.length[low], correct.length[low])
    assert np.mean(np.abs(thesis.length[~low] - correct.length[~low]) > 1e-3) > 0.9


def test_f17_stops_every_ray_at_one_reflection_order():
    room, settings = box(absorption=0.05), _settings(rays=400, max_time=3.0)
    correct = trace_rays(room, settings, C, NO_AIR)
    with reinserted('F17'):
        thesis = trace_rays(room, settings, C, NO_AIR)
    assert correct.stop_order == -1
    assert thesis.stop_order > 1
    assert thesis.reflections <= settings.rays * thesis.stop_order
    assert thesis.reflections < correct.reflections
    # The early response is the same; the late response is cut
    np.testing.assert_allclose(thesis.energy[..., :20], correct.energy[..., :20])
    assert thesis.energy[..., -500:].sum() < correct.energy[..., -500:].sum()


def test_thesis_stop_order():
    assert faults.thesis_stop_order(np.array([1.0, 1e-3, 1e-7, 1e-9])) == 3
    assert faults.thesis_stop_order(np.array([1.0, 1e-3])) == -1
    assert faults.thesis_stop_order(np.zeros(0)) == -1


def test_every_fault_has_a_description():
    assert set(FAULTS) == {'F1', 'F2', 'F3', 'F6', 'F9', 'F16', 'F17'}


# ---------------------------------------------------------------- criteria of experiments/fault_reinsertion.py

@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend('experiments')
    import fault_reinsertion
    yield fault_reinsertion
    sys.modules.pop('fault_reinsertion')


def test_effect_needs_more_than_a_jnd_and_twice_the_seed_noise(script):
    baseline = np.array([[1.00, 2.0], [1.02, 2.4], [0.98, 1.6]])  # sigma 0.02 and 0.4
    mutant = baseline + np.array([0.06, 0.5])
    result = script.effect(baseline, mutant, 't30')
    np.testing.assert_allclose(result['difference'], [0.06, 0.5])
    np.testing.assert_allclose(result['threshold_1'], [0.05, 0.8])  # 1 JND = 5 %; 2 sigma = 0.8
    np.testing.assert_allclose(result['threshold_2'], [0.10, 0.8])
    assert result['exceeds_1'].tolist() == [True, False] and result['exceeds_2'].tolist() == [False, False]


def test_c50_uses_decibels_and_nan_counts_as_a_difference(script):
    baseline = np.zeros((3, 2))
    result = script.effect(baseline, baseline + np.array([1.5, np.nan]), 'c50')
    np.testing.assert_allclose(result['threshold_1'], 1.0)
    assert result['exceeds_1'].tolist() == [True, True] and result['exceeds_2'].tolist() == [False, True]


def test_measured_comparison_needs_three_of_the_four_upper_bands(script):
    measured = np.ones(6)
    assert script.passes_measured(np.array([2.0, 2.0, 1.09, 0.91, 1.0, 1.5]), measured)
    assert not script.passes_measured(np.array([1.0, 1.0, 1.0, 1.0, 1.2, 0.8]), measured)
    assert not script.passes_measured(np.array([1.0, 1.0, np.nan, np.nan, 1.0, 1.0]), measured)
    assert script.measured_verdict(False, False) == 'not informative'
    assert script.measured_verdict(True, False) == 'detected'
    assert script.measured_verdict(True, True) == 'not detected'


def test_known_answer_verdict(script):
    assert script.known_answer_verdict((3, 0, 0.1), (3, 1, 40.0)) == 'detected'
    assert script.known_answer_verdict((3, 0, 0.1), (3, 0, 0.2)) == 'not detected'
    assert script.known_answer_verdict((3, 1, 9.0), (3, 2, 40.0)) == 'not informative'


@pytest.mark.skipif(not Path('models/private/Real_ClassRoom.3dm').exists()
                    or not Path('../data/measured/classroom.csv').exists(),
                    reason='needs the private models and ../data/measured')
def test_fault_reinsertion_runs(tmp_path, script):
    assert script.main(['--cases', 'classroom', '--faults', 'F1', 'F6', '--seeds', '2', '--rays-per-m3', '2',
                        '--skip-checks', '--output', str(tmp_path)]) == 0
    for name in ('effects.csv', 'matrix.csv', 'fault_reinsertion.png', 'run_info.json'):
        assert (tmp_path / name).exists()
