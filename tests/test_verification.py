"""The known-answer checks of gasim.verification (experiment E0)."""
import numpy as np
import pytest

from gasim.propagation.rays import rain_fraction
from gasim.verification import check_air, check_decay_estimator, check_image_sources, run_checks


@pytest.mark.parametrize('check', [check_image_sources, check_decay_estimator, check_air])
def test_deterministic_checks_pass(check):
    results = check()
    assert results and all(result.passed for result in results)


def test_rain_fraction_is_the_lambert_cap():
    # Small sphere: 2 cos(theta) (1 - cos(gamma)) -> cos(theta) (r / d)^2
    assert rain_fraction(0.5, 0.01, 10.0) == pytest.approx(0.5 * 1e-6, rel=1e-4)
    assert rain_fraction(0.0, 0.2, 1.0) == 0.0


def test_quick_run_covers_every_check():
    results = run_checks(quick=True)
    assert {result.id for result in results} == {f'V{i}' for i in range(1, 9)}
    # The Monte Carlo checks hold even with the small quick samples
    assert all(result.passed for result in results if result.id == 'V6')
    assert all(np.isfinite(result.value) for result in results)
