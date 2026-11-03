"""
Known-answer checks of the engine: problems whose answer does not depend on reading the code.

Each check builds a room or a signal with an independent reference (an analytical solution, an exact sum
over the image lattice, a formula from the literature or a Monte Carlo sample drawn without the engine)
and compares the engine with it under a fixed criterion. run_checks() runs them all; the test suite
covers the same properties with smaller cases. The references are also used by the tests.

    V1  image sources of a shoebox: counts, path lengths and first-order energies (Allen & Berkley 1979)
    V2  lossless box: every time bin receives c dt / V per unit source energy
    V3  fully diffuse box: T30 equals Eyring corrected for the free-path spread (Kuttruff Eq. 5.34)
    V4  specular box with an absorptive floor: T20/T30 equal the exact image-lattice response
    V5  received energy does not depend on the number of rays
    V6  scattering-rain solid angle equals a Monte Carlo sample of Lambert directions
    V7  decay estimator: T20/T30/EDT of synthetic exponential decays
    V8  air absorption equals the reference values of ISO 9613-1 (Kuttruff Table 6.1)
"""
from dataclasses import dataclass

import numpy as np

from gasim.air import energy_attenuation
from gasim.analysis.decay import decay_parameters
from gasim.analysis.statistical import eyring, kuttruff
from gasim.config import SimulationSettings
from gasim.geometry.build import build_room
from gasim.materials import material_table
from gasim.propagation.image_sources import find_image_sources
from gasim.propagation.rays import rain_fraction, trace_rays

C = 343.0
NO_AIR = np.zeros(6)


@dataclass(frozen=True)
class Check:
    """
    One comparison with a reference
    """
    id: str
    case: str
    quantity: str
    reference: float
    value: float
    criterion: str
    passed: bool

    @property
    def deviation(self):
        """
        :return: Relative deviation from the reference, nan if the reference is 0
        """
        return (self.value - self.reference) / self.reference if self.reference else float('nan')


# ---------------------------------------------------------------- references

def box_polygons(lx: float, ly: float, lz: float, layers=('Floor', 'Ceiling', 'Wall')):
    """
    Six faces of an lx x ly x lz box: floor (z = 0), ceiling (z = lz) and four walls
    :param layers: Layer names of floor, ceiling and walls
    :return: List of (layer, (4, 3) vertices)
    """
    corners = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]], dtype=float)
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]
    names = [layers[0], layers[1]] + [layers[2]] * 4
    return [(name, corners[face]) for name, face in zip(names, faces)]


def image_lattice(size, source, receiver, max_order: int):
    """
    Path lengths of all images of a rectangular room up to max_order (Allen & Berkley 1979)
    :return: Lengths (m) and reflection orders of the images
    """
    lengths, orders = [], []
    span = range(-max_order, max_order + 1)
    for index in ((i, j, k) for i in span for j in span for k in span):
        order = sum(abs(i) for i in index)
        if order > max_order:
            continue
        image = [i * length + s if i % 2 == 0 else (i + 1) * length - s for i, length, s in zip(index, size, source)]
        lengths.append(np.linalg.norm(np.array(image) - receiver))
        orders.append(order)
    return np.array(lengths), np.array(orders)


def image_lattice_histogram(size, source, receiver, wall, floor, ceiling, max_time, time_step=1e-3, c=C):
    """
    Exact energy response of a shoebox with specular walls: sum over the image lattice
    :param wall: Absorption of the four side walls; floor (z = 0) and ceiling (z = size[2]) separately
    :return: (K,) energy per bin, 1/m2 per unit source energy
    """
    n_bins = int(max_time / time_step)
    histogram = np.zeros(n_bins)
    n = [int(np.ceil(c * max_time / length)) + 1 for length in size]
    ii, jj = np.arange(-n[0], n[0] + 1), np.arange(-n[1], n[1] + 1)
    x = np.where(ii % 2 == 0, ii * size[0] + source[0], (ii + 1) * size[0] - source[0]) - receiver[0]
    y = np.where(jj % 2 == 0, jj * size[1] + source[1], (jj + 1) * size[1] - source[1]) - receiver[1]
    xy2, side_reflections = x[:, None] ** 2 + y[None, :] ** 2, np.abs(ii)[:, None] + np.abs(jj)[None, :]
    for k in range(-n[2], n[2] + 1):
        z = (k * size[2] + source[2] if k % 2 == 0 else (k + 1) * size[2] - source[2]) - receiver[2]
        ceiling_hits, floor_hits = (-(-k // 2), k // 2) if k >= 0 else ((-k) // 2, -(k // 2))
        distance = np.sqrt(xy2 + z * z)
        bins = (distance / (c * time_step)).astype(np.int64)
        keep = bins < n_bins
        energy = ((1 - wall) ** side_reflections * (1 - floor) ** floor_hits * (1 - ceiling) ** ceiling_hits
                  / (4 * np.pi * distance ** 2))
        np.add.at(histogram, bins[keep], energy[keep])
    return histogram


def exponential_decay(reverberation_time: float, depth_db: float = 100.0, time_step: float = 1e-3,
                      onset: float = 0.0):
    """
    Energy histogram of a single-slope decay
    :param depth_db: Decay depth before the histogram ends, dB
    :param onset: Empty time before the first arrival, s
    """
    k = np.arange(int(depth_db / 60 * reverberation_time / time_step))
    energy = np.exp(-np.log(1e6) * k * time_step / reverberation_time)
    return np.concatenate([np.zeros(round(onset / time_step)), energy])


def lambert_fraction_monte_carlo(cos_theta: float, radius: float, distance: float, samples: int, seed: int = 1):
    """
    Share of Lambert-distributed directions about the normal (0, 0, 1) that pass through a sphere, drawn
    with NumPy independently of the engine
    :return: Share, its standard error
    """
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(samples), rng.random(samples)
    sin_t, cos_t, phi = np.sqrt(u1), np.sqrt(1 - u1), 2 * np.pi * u2
    centre = np.array([np.sqrt(1 - cos_theta ** 2), 0.0, cos_theta])
    along = sin_t * np.cos(phi) * centre[0] + cos_t * centre[2]
    hit = along >= np.sqrt(1 - (radius / distance) ** 2)
    share = hit.mean()
    return share, np.sqrt(share * (1 - share) / samples)


def _box(lx, ly, lz, absorption, scattering, receivers, source=(1.1, 1.3, 1.2), floor=None, ceiling=None):
    spec = {'Wall': {'absorption': absorption, 'scattering': scattering},
            'Floor': {'absorption': absorption if floor is None else floor, 'scattering': scattering},
            'Ceiling': {'absorption': absorption if ceiling is None else ceiling, 'scattering': scattering}}
    return build_room(box_polygons(lx, ly, lz), material_table(spec), list(source), receivers)


def _settings(**values):
    defaults = {'rays': 2000, 'ism_order': 0, 'receiver_radius': 0.3, 'time_step': 1e-3, 'max_time': 1.0, 'seed': 1}
    return SimulationSettings(**{**defaults, **values})


# ---------------------------------------------------------------- checks

def check_image_sources():
    size, source = np.array([5.0, 4.0, 3.0]), [1.13, 1.31, 1.17]
    receivers = [[3.71, 2.93, 1.57], [4.23, 0.97, 2.11]]
    alpha, s = 0.2, 0.1
    room = _box(*size, absorption=alpha, scattering=s, receivers=receivers, source=source)
    paths = find_image_sources(room, 4, NO_AIR)
    counts_ok, length_error, energy_error = True, 0.0, 0.0
    for index, receiver in enumerate(room.receivers):
        counts_ok &= np.array_equal(paths.counts(index), [1, 6, 18, 38, 66])
        lengths, _ = image_lattice(size, source, receiver, 4)
        mine = paths.receiver == index
        length_error = max(length_error, np.abs(np.sort(paths.length[mine]) - np.sort(lengths)).max())
        first = mine & (paths.order == 1)
        expected = (1 - alpha) * (1 - s) / (4 * np.pi * paths.length[first] ** 2)
        energy_error = max(energy_error, np.abs(paths.energy[first, 0] / expected - 1).max())
    return [Check('V1', '5 x 4 x 3 m shoebox, orders 0-4, 2 receivers', 'image sources per order (1/6/18/38/66)',
                  1.0, float(counts_ok), 'exact', counts_ok),
            Check('V1', 'same', 'largest path-length error (m)', 0.0, length_error, '< 1e-9 m', length_error < 1e-9),
            Check('V1', 'same', 'first-order energy / (1 - a)(1 - s)/(4 pi d2)', 1.0, 1.0 + energy_error,
                  'within 1e-12', energy_error < 1e-12)]


def check_lossless_box(rays=4000):
    checks = []
    for s in (0.0, 0.5, 1.0):
        room = _box(4.0, 3.0, 2.5, absorption=0.0, scattering=s, receivers=[[2.7, 1.9, 1.4]])
        energy = trace_rays(room, _settings(rays=rays, max_time=0.6), C, NO_AIR).energy
        steady = float(energy[0, :, 100:600].mean())
        reference = C * 1e-3 / room.volume
        checks.append(Check('V2', f'lossless 4 x 3 x 2.5 m box, s = {s}', 'energy per 1 ms bin (1/m2)', reference,
                            steady, 'within 4%', abs(steady / reference - 1) < 0.04))
    return checks


def check_diffuse_box(rays=20000):
    checks = []
    for alpha in (0.1, 0.3):
        room = _box(8.0, 6.0, 4.0, absorption=alpha, scattering=1.0, receivers=[[6.5, 4.6, 1.5], [2.5, 4.0, 2.8]])
        reference = float(kuttruff(room, C, relative_variance=0.36)[0])
        t30 = decay_parameters(trace_rays(room, _settings(rays=rays, max_time=1.4 * reference * 70 / 60), C,
                                          NO_AIR).energy, 1e-3).t30
        value = float(np.mean(t30))
        checks.append(Check('V3', f'8 x 6 x 4 m box, s = 1, alpha = {alpha} (Eyring {eyring(room, C)[0]:.3f} s)',
                            'T30 (s) vs Kuttruff Eq. 5.34', reference, value, 'within 2%',
                            abs(value / reference - 1) < 0.02))
    return checks


def check_specular_box(rays=50_000):
    size, source, receiver = np.array([9.0, 7.0, 5.0]), [2.5, 3.5, 1.5], [7.0, 5.0, 1.5]
    room = _box(*size, absorption=0.1, scattering=0.0, receivers=[receiver], source=source, floor=0.75)
    exact = decay_parameters(image_lattice_histogram(size, source, receiver, 0.1, 0.75, 0.1, 2.5), 1e-3)
    traced = decay_parameters(trace_rays(room, _settings(rays=rays, receiver_radius=0.2, max_time=2.5), C,
                                         NO_AIR).energy[0, 0], 1e-3)
    case = f'9 x 7 x 5 m box, absorptive floor, s = 0 (Eyring {eyring(room, C)[0]:.2f} s)'
    return [Check('V4', case, f'{name.upper()} (s) vs exact image lattice', float(getattr(exact, name)),
                  float(getattr(traced, name)), 'within 6%',
                  abs(float(getattr(traced, name)) / float(getattr(exact, name)) - 1) < 0.06)
            for name in ('t30', 't20')]


def check_ray_count(counts=(250, 1000, 4000), seeds=4):
    room = _box(8.0, 6.0, 4.0, absorption=0.15, scattering=0.2, receivers=[[6.5, 4.6, 1.5]])
    totals = {n: [trace_rays(room, _settings(rays=n, seed=seed, max_time=1.5), C, NO_AIR).energy.sum()
                  for seed in range(1, seeds + 1)] for n in counts}
    reference = np.mean(totals[counts[-1]])
    reference_error = np.std(totals[counts[-1]], ddof=1) / np.sqrt(seeds)
    checks = []
    for n in counts[:-1]:
        value, error = np.mean(totals[n]), np.std(totals[n], ddof=1) / np.sqrt(seeds)
        tolerance = max(3 * np.hypot(error, reference_error), 0.03 * reference)
        checks.append(Check('V5', f'8 x 6 x 4 m box, {n} vs {counts[-1]} rays, {seeds} seeds',
                            'received energy (sum over bins and bands)', float(reference), float(value),
                            'within 3 standard errors or 3%', abs(value - reference) <= tolerance))
    return checks


def check_rain_solid_angle(samples=4_000_000, radius=0.2):
    checks = []
    for distance in (1.0, 2.0, 5.0, 10.0):
        for angle in (0.0, 45.0, 70.0):
            cos_theta = np.cos(np.radians(angle))
            share, error = lambert_fraction_monte_carlo(cos_theta, radius, distance, samples)
            value = rain_fraction(cos_theta, radius, distance)
            tolerance = max(3 * error, 0.03 * share)
            checks.append(Check('V6', f'sphere r = {radius} m at {distance:g} m, {angle:g} deg from the normal',
                                'scattered share reaching the sphere vs Monte Carlo', float(share), float(value),
                                'within 3 standard errors or 3%', abs(value - share) <= tolerance))
    return checks


def check_decay_estimator():
    checks = []
    for reverberation_time in (0.5, 1.0, 2.0):
        for depth in (60.0, 80.0, 120.0):
            parameters = decay_parameters(exponential_decay(reverberation_time, depth), 1e-3)
            for name in ('t30', 't20', 'edt'):
                value = float(getattr(parameters, name))
                checks.append(Check('V7', f'exponential, T = {reverberation_time} s, {depth:g} dB deep',
                                    f'{name.upper()} (s)', reverberation_time, value, 'within 1%',
                                    abs(value / reverberation_time - 1) < 0.01))
    return checks


def check_air():
    frequencies = [500, 1000, 2000, 4000]
    reference = {50: [0.63, 1.08, 2.28, 6.84], 70: [0.64, 1.15, 2.08, 5.32]}  # Kuttruff Table 6.1, 1e-3 1/m
    checks = []
    for humidity, values in reference.items():
        m = energy_attenuation(20.0, humidity, frequency=frequencies) * 1e3
        for f, mine, theirs in zip(frequencies, m, values):
            checks.append(Check('V8', f'20 degC, {humidity}% RH, {f} Hz', 'air attenuation m (1e-3 1/m)', theirs,
                                float(mine), 'within 2%', abs(mine / theirs - 1) < 0.02))
    return checks


CHECKS = (check_image_sources, check_lossless_box, check_diffuse_box, check_specular_box, check_ray_count,
          check_rain_solid_angle, check_decay_estimator, check_air)


def run_checks(quick: bool = False):
    """
    Run all known-answer checks
    :param quick: Smaller ray counts and Monte Carlo samples, for smoke tests (criteria unchanged)
    :return: List of Check
    """
    if not quick:
        return [check for function in CHECKS for check in function()]
    return (check_image_sources() + check_lossless_box(rays=1000) + check_diffuse_box(rays=4000)
            + check_specular_box(rays=10_000) + check_ray_count(counts=(250, 1000), seeds=3)
            + check_rain_solid_angle(samples=200_000) + check_decay_estimator() + check_air())
