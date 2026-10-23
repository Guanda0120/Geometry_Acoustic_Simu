"""Validation of the ray tracer against exact and statistical references (docs/rewrite-plan.md, Sect. 5)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import box_polygons

from gasim.analysis.decay import decay_parameters
from gasim.analysis.statistical import eyring, kuttruff
from gasim.cli import main
from gasim.config import SimulationSettings, load_config
from gasim.geometry.build import build_room
from gasim.materials import material_table
from gasim.propagation.rays import trace_rays
from gasim.simulate import simulate

C = 343.0
NO_AIR = np.zeros(6)


def box(lx=4.0, ly=3.0, lz=2.5, absorption=0.1, scattering=0.0, receivers=None, drop_last=False):
    materials = material_table({'Wall': {'absorption': absorption, 'scattering': scattering}})
    polygons = box_polygons(lx, ly, lz)
    if drop_last:
        polygons = polygons[:-1]
    receivers = receivers if receivers is not None else [[lx - 1.3, ly - 1.1, 1.4]]
    return build_room(polygons, materials, [1.1, 1.3, 1.2], receivers)


def settings(**values):
    defaults = {'rays': 2000, 'ism_order': 0, 'receiver_radius': 0.3, 'time_step': 1e-3, 'max_time': 1.0,
                'seed': 1}
    return SimulationSettings(**{**defaults, **values})


@pytest.mark.parametrize('scattering, groups', [(0.4, 1), ([0.1, 0.1, 0.3, 0.3, 0.6, 0.9], 4)])
def test_energy_balance_is_exact(scattering, groups):
    rays = trace_rays(box(absorption=[0.1, 0.2, 0.3, 0.3, 0.4, 0.5], scattering=scattering), settings(), C, NO_AIR)
    np.testing.assert_allclose(rays.absorbed + rays.remaining + rays.escaped, 1.0, rtol=1e-9)
    assert rays.escaped_rays == 0
    assert rays.band_groups == groups


def test_escaped_rays_are_counted():
    rays = trace_rays(box(drop_last=True), settings(rays=500), C, NO_AIR)
    assert rays.escaped_rays > 0
    np.testing.assert_allclose(rays.absorbed + rays.remaining + rays.escaped, 1.0, rtol=1e-9)


@pytest.mark.parametrize('scattering', [0.0, 0.5, 1.0, [0.0, 0.2, 0.5, 0.8, 1.0, 0.5]],
                         ids=['specular', 'half', 'diffuse', 'per-band'])
def test_lossless_box_receives_c_dt_over_v(scattering):
    # In a closed lossless room the energy density settles at E/V, so every bin of the histogram receives
    # c dt / V per unit source energy. This checks the detector weight 1/(pi r2), the diffuse rain
    # (solid angle, Lambert factor, detection skip) and the band reweighting together.
    room = box(absorption=0.0, scattering=scattering)
    rays = trace_rays(room, settings(rays=4000, max_time=0.6), C, NO_AIR)
    steady = rays.energy[0, :, 100:600].mean(axis=1)
    np.testing.assert_allclose(steady, C * 1e-3 / room.volume, rtol=0.04)


@pytest.mark.parametrize('absorption', [0.1, 0.3])
def test_diffuse_box_follows_kuttruff(absorption):
    # Diffuse walls: Eyring corrected for the free-path spread (Kuttruff Eq. 5.34); gamma2 = 0.36 for a
    # 1 : 1.5 : 2 room, between 1:1:2 (0.356) and 1:2:2 (0.363) in Kuttruff's Table 5.1. Eyring alone is
    # 2 % (alpha = 0.1) and 7 % (alpha = 0.3) too short.
    room = box(8.0, 6.0, 4.0, absorption=absorption, scattering=1.0,
               receivers=[[6.5, 4.6, 1.5], [2.5, 4.0, 2.8]])
    expected = kuttruff(room, C, relative_variance=0.36)[0]
    assert expected > eyring(room, C)[0]
    rays = trace_rays(room, settings(rays=20000, max_time=1.4 * expected * 70 / 60), C, NO_AIR)
    t30 = decay_parameters(rays.energy, 1e-3).t30
    np.testing.assert_allclose(t30, expected, rtol=0.02)


def test_reverberation_time_does_not_drift_with_ray_count():
    room = box(8.0, 6.0, 4.0, absorption=0.15, scattering=0.2)

    def mean_t30(n_rays):
        values = [decay_parameters(trace_rays(room, settings(rays=n_rays, seed=seed, max_time=1.5), C,
                                              NO_AIR).energy, 1e-3).t30[0].mean() for seed in range(1, 5)]
        return np.mean(values), np.std(values) / np.sqrt(len(values))

    few, few_error = mean_t30(1500)
    many, many_error = mean_t30(6000)
    assert abs(few - many) < max(3 * np.hypot(few_error, many_error), 0.01 * many)


def test_direct_sound_energy():
    # Large room and a short window: only the direct sound arrives
    # Source at (1.1, 1.3, 1.2), receiver 2.9 m away (8.6 ms); the first reflection arrives after 12.4 ms
    room = box(10.0, 10.0, 10.0, receivers=[[3.5, 2.8, 2.0]])
    distance = np.linalg.norm(room.receivers[0] - room.source)
    rays = trace_rays(room, settings(rays=200_000, receiver_radius=0.5, max_time=0.011), C, NO_AIR)
    np.testing.assert_allclose(rays.energy[0].sum(axis=1), 1 / (4 * np.pi * distance ** 2), rtol=0.1)


def test_air_absorption_sets_the_decay_of_a_lossless_room():
    attenuation = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    rays = trace_rays(box(absorption=0.0, scattering=1.0), settings(rays=1000, max_time=5.0), C, attenuation)
    t30 = decay_parameters(rays.energy, 1e-3).t30[0]
    np.testing.assert_allclose(t30, 60 / (10 * np.log10(np.e) * attenuation * C), rtol=0.02)


def test_same_seed_same_result():
    room = box(scattering=0.3)
    first = trace_rays(room, settings(rays=300, seed=5), C, NO_AIR).energy
    np.testing.assert_array_equal(first, trace_rays(room, settings(rays=300, seed=5), C, NO_AIR).energy)
    assert not np.array_equal(first, trace_rays(room, settings(rays=300, seed=6), C, NO_AIR).energy)


def test_simulate_test_room():
    config = load_config('configs/test_room.toml')
    config = dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, rays=3000))
    result = simulate(config)
    assert result.energy.shape == (4, 6, 3000)
    assert result.metadata['escaped_rays'] == 0
    assert '[room]' in result.metadata['config']
    assert np.all(np.isfinite(result.decay().t30))


def test_cli_run(tmp_path, capsys):
    text = Path('configs/test_room.toml').read_text().replace('rays = 31500', 'rays = 2000')
    config = tmp_path / 'small.toml'
    config.write_text(text)
    output = tmp_path / 'run.npz'
    assert main(['run', str(config), '--output', str(output)]) == 0
    assert output.exists()
    assert 'T30 (s)' in capsys.readouterr().out


def lattice_histogram(size, source, receiver, wall, floor, ceiling, max_time, time_step=1e-3):
    """
    Exact energy response of a shoebox with specular walls: sum over the image lattice (Allen & Berkley 1979)
    :param wall: Absorption of the four side walls; floor (z = 0) and ceiling (z = size[2]) separately
    """
    n_bins = int(max_time / time_step)
    histogram = np.zeros(n_bins)
    n = [int(np.ceil(C * max_time / length)) + 1 for length in size]
    ii, jj = np.arange(-n[0], n[0] + 1), np.arange(-n[1], n[1] + 1)
    x = np.where(ii % 2 == 0, ii * size[0] + source[0], (ii + 1) * size[0] - source[0]) - receiver[0]
    y = np.where(jj % 2 == 0, jj * size[1] + source[1], (jj + 1) * size[1] - source[1]) - receiver[1]
    xy2, side_reflections = x[:, None] ** 2 + y[None, :] ** 2, np.abs(ii)[:, None] + np.abs(jj)[None, :]
    for k in range(-n[2], n[2] + 1):
        z = (k * size[2] + source[2] if k % 2 == 0 else (k + 1) * size[2] - source[2]) - receiver[2]
        ceiling_hits, floor_hits = (-(-k // 2), k // 2) if k >= 0 else ((-k) // 2, -(k // 2))
        distance = np.sqrt(xy2 + z * z)
        bins = (distance / (C * time_step)).astype(np.int64)
        keep = bins < n_bins
        energy = ((1 - wall) ** side_reflections * (1 - floor) ** floor_hits * (1 - ceiling) ** ceiling_hits
                  / (4 * np.pi * distance ** 2))
        np.add.at(histogram, bins[keep], energy[keep])
    return histogram


def test_specular_non_diffuse_room_matches_the_image_lattice():
    # 9 x 7 x 5 m, absorptive floor, hard walls and ceiling, no scattering: the decay is far from diffuse
    # (Eyring 0.64 s, exact T30 1.9 s); the tracer must follow the exact lattice, not Eyring
    size, source, receiver = np.array([9.0, 7.0, 5.0]), [2.5, 3.5, 1.5], [7.0, 5.0, 1.5]
    corners = box_polygons(*size)
    materials = material_table({'Wall': {'absorption': 0.1, 'scattering': 0.0}, 'Floor': {'absorption': 0.75},
                                'Ceiling': {'absorption': 0.1}})
    polygons = [('Floor' if np.allclose(v[:, 2], 0) else 'Ceiling' if np.allclose(v[:, 2], size[2]) else 'Wall', v)
                for _, v in corners]
    room = build_room(polygons, materials, source, [receiver])
    exact = decay_parameters(lattice_histogram(size, source, receiver, 0.1, 0.75, 0.1, 2.5), 1e-3)
    rays = trace_rays(room, settings(rays=50_000, receiver_radius=0.2, max_time=2.5), C, NO_AIR)
    traced = decay_parameters(rays.energy[0, 0], 1e-3)
    assert float(exact.t30) > 2.5 * eyring(room, C)[0]
    assert float(traced.t30) == pytest.approx(float(exact.t30), rel=0.06)
    assert float(traced.t20) == pytest.approx(float(exact.t20), rel=0.06)
