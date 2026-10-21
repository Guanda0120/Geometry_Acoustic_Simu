"""Diffuse-field absorption of a locally reacting surface with real normal impedance."""
import numpy as np
import pytest

from gasim.legacy.Acoustic.Material import Material


def paris_integral(normal_alpha: float):
    """
    Diffuse-field absorption from Paris's integral, alpha_d = int alpha(theta) sin(2 theta) d theta
    :param normal_alpha: Normal-incidence absorption, real impedance assumed
    :return: Diffuse-field absorption
    """
    ref = np.sqrt(1 - normal_alpha)
    ksi = (1 + ref) / (1 - ref)
    theta = np.linspace(0, np.pi / 2, 200001)
    ref_theta = (ksi * np.cos(theta) - 1) / (ksi * np.cos(theta) + 1)
    return np.trapezoid((1 - ref_theta ** 2) * np.sin(2 * theta), theta)


@pytest.mark.parametrize('normal_alpha', [0.01, 0.1, 0.3, 0.6, 0.9])
def test_closed_form_matches_paris_integral(normal_alpha):
    closed_form = Material.normal2diffuse_res(normal_alpha, 0.0)
    assert closed_form == pytest.approx(paris_integral(normal_alpha), abs=1e-6)


@pytest.mark.parametrize('diffuse_alpha', [0.01, 0.05, 0.2, 0.5, 0.9])
def test_inversion_reproduces_input(diffuse_alpha):
    material = Material('flat', np.full(6, diffuse_alpha), np.full(6, 0.1),
                        plane_wave_model='JCA', opt_max_res=0.2)
    assert np.max(np.abs(material.normal_res)) < 1e-10


def test_diffuse_alpha_above_real_impedance_limit_has_no_solution():
    # A real impedance cannot give alpha_d above ~0.951, so alpha_d = 0.99 is not reproduced
    material = Material('hard_limit', np.full(6, 0.99), np.full(6, 0.1),
                        plane_wave_model='JCA', opt_max_res=0.2)
    assert np.max(np.abs(material.normal_res)) > 0.03
