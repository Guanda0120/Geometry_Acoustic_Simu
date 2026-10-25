"""
Stochastic ray tracing with energy particles and diffuse rain.

Each of the N rays carries 1/N of the source energy in every band. A ray that crosses a receiver sphere
of radius r deposits energy / (pi r2), so the histogram holds energy per unit area per unit source
energy (Vorlaender 2020, Sect. 11.3.3). At every wall hit the energy is multiplied by (1 - alpha) and the
ray continues specularly or in a Lambert direction. Diffuse rain adds, at every wall hit, the expected
energy the diffuse share sends straight to each visible receiver (Vorlaender 2020, Eq. 11.7; Schroeder
2011, Eq. 5.20); sphere detection is therefore skipped on a segment that starts with a diffuse
reflection, so that every path is counted once in expectation.

Per surface (gasim.surfaces) the tracer also counts the ray hits and sums the energy that arrives
at and is absorbed by the surface, and the energy each receiver gets with the surface as the last
reflection (diffuse rain and sphere crossings).

Bands whose scattering coefficients differ are traced separately, each group with its own rays: one
direction must serve all bands of a ray, and reweighting bands by s_b / p would be unbiased but its
weights multiply over hundreds of reflections and degenerate.
"""
import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from gasim.config import SimulationSettings
from gasim.geometry.kernels import first_hit, kernel_arrays
from gasim.geometry.room import Room

ENERGY_THRESHOLD = 1e-7  # tracing stops when a ray (air absorption included) is 70 dB below its start
# Rays are split into this many chunks, each with its own histogram, traced in parallel and summed in a
# fixed order: results do not depend on the number of threads. Memory: CHUNKS x R x B x K x 8 bytes.
CHUNKS = 32

# splitmix64: a small counter-based generator, so every ray has its own reproducible stream
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_SHIFT_30, _SHIFT_27, _SHIFT_31, _SHIFT_11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_UNIT = 1.0 / 9007199254740992.0  # 2 ** -53


@njit(cache=True)
def _mix(z):
    z = (z ^ (z >> _SHIFT_30)) * _MIX_1
    z = (z ^ (z >> _SHIFT_27)) * _MIX_2
    return z ^ (z >> _SHIFT_31)


@njit(cache=True)
def _uniform(state):
    """
    :return: New state, uniform number in [0, 1)
    """
    state = state + _GOLDEN
    return state, (_mix(state) >> _SHIFT_11) * _UNIT


@njit(cache=True)
def _lambert(nx, ny, nz, u1, u2):
    """
    Cosine-weighted direction about the unit normal n
    """
    # Orthonormal basis (t1, t2, n)
    if abs(nx) < 0.9:
        ax, ay, az = 1.0, 0.0, 0.0
    else:
        ax, ay, az = 0.0, 1.0, 0.0
    t1x, t1y, t1z = ny * az - nz * ay, nz * ax - nx * az, nx * ay - ny * ax
    norm = math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    t1x, t1y, t1z = t1x / norm, t1y / norm, t1z / norm
    t2x, t2y, t2z = ny * t1z - nz * t1y, nz * t1x - nx * t1z, nx * t1y - ny * t1x
    sin_theta = math.sqrt(u1)
    cos_theta = math.sqrt(1.0 - u1)
    phi = 2.0 * math.pi * u2
    a, b = sin_theta * math.cos(phi), sin_theta * math.sin(phi)
    return (a * t1x + b * t2x + cos_theta * nx,
            a * t1y + b * t2y + cos_theta * ny,
            a * t1z + b * t2z + cos_theta * nz)


@njit(cache=True)
def _deposit(histogram, received, receiver, k, energy, factor, path, attenuation, surface):
    """
    Add energy * factor * exp(-m path) of every band to bin k of a receiver, and to the energy received
    via the last reflecting surface (none for the direct sound, surface < 0)
    """
    if k < histogram.shape[2]:
        for b in range(energy.shape[0]):
            value = energy[b] * factor * math.exp(-attenuation[b] * path)
            histogram[receiver, b, k] += value
            if surface >= 0:
                received[receiver, b, surface] += value


@njit(cache=True)
def _trace_range(first, last, n_rays, seed, stream, source, receivers, radius, v0, e1, e2, tri_polygon,
                 poly_normal, poly_material, poly_surface, absorption, diffuse_probability, attenuation, speed,
                 time_step, max_length, exclude_order, histogram, absorbed, remaining, escaped, counters,
                 surface_hits, surface_incident, surface_absorbed, surface_received):
    """
    Trace rays first .. last - 1 of one group of bands that share their scattering coefficients
    :param diffuse_probability: (M,) scattering coefficient of each material in this group of bands
    :param histogram: (R, B, K) output, added to
    :param absorbed, remaining, escaped: (B,) energy outputs, added to
    :param counters: (2,) output: escaped rays, wall hits
    :param surface_hits: (S,) output, ray hits per surface
    :param surface_incident, surface_absorbed: (B, S) output, energy arriving at / absorbed by each surface
    :param surface_received: (R, B, S) output, energy deposited at the receivers via each last surface
    """
    n_receivers, n_bands, n_bins = histogram.shape
    escaped_rays = 0
    reflections = 0
    inv_area = 1.0 / (math.pi * radius * radius)
    bin_length = speed * time_step
    energy = np.empty(n_bands)
    base = _mix(np.uint64(seed) + _GOLDEN * np.uint64(stream + 1))

    for ray in range(first, last):
        state = _mix(base + np.uint64(ray) * _GOLDEN)
        state, u1 = _uniform(state)
        state, u2 = _uniform(state)
        dz = 1.0 - 2.0 * u1
        rho = math.sqrt(max(0.0, 1.0 - dz * dz))
        dx, dy = rho * math.cos(2.0 * math.pi * u2), rho * math.sin(2.0 * math.pi * u2)
        ox, oy, oz = source[0], source[1], source[2]
        energy[:] = 1.0 / n_rays
        length = 0.0
        order = 0
        specular_only = True
        after_diffuse = False
        last_surface = -1

        while True:
            t_hit, tri = first_hit(ox, oy, oz, dx, dy, dz, v0, e1, e2)
            if tri < 0:
                escaped_rays += 1
                escaped += energy
                break

            # Receivers crossed on this segment
            if not after_diffuse and not (specular_only and order <= exclude_order):
                for r in range(n_receivers):
                    wx, wy, wz = receivers[r, 0] - ox, receivers[r, 1] - oy, receivers[r, 2] - oz
                    along = wx * dx + wy * dy + wz * dz
                    if along < 0.0 or along > t_hit:
                        continue
                    if wx * wx + wy * wy + wz * wz - along * along > radius * radius:
                        continue
                    path = length + along
                    _deposit(histogram, surface_received, r, int(path / bin_length), energy, inv_area, path,
                             attenuation, last_surface)

            length += t_hit
            if length > max_length:
                remaining += energy
                break
            ox, oy, oz = ox + t_hit * dx, oy + t_hit * dy, oz + t_hit * dz
            order += 1
            reflections += 1
            polygon = tri_polygon[tri]
            nx, ny, nz = poly_normal[polygon, 0], poly_normal[polygon, 1], poly_normal[polygon, 2]
            m = poly_material[polygon]
            surface = poly_surface[polygon]
            surface_hits[surface] += 1
            for b in range(n_bands):
                arriving = energy[b] * math.exp(-attenuation[b] * length)
                surface_incident[b, surface] += arriving
                surface_absorbed[b, surface] += arriving * absorption[m, b]

            # Diffuse rain to every visible receiver
            for r in range(n_receivers):
                vx, vy, vz = receivers[r, 0] - ox, receivers[r, 1] - oy, receivers[r, 2] - oz
                distance = math.sqrt(vx * vx + vy * vy + vz * vz)
                if distance <= radius:
                    continue
                cos_theta = (nx * vx + ny * vy + nz * vz) / distance
                if cos_theta <= 0.0:
                    continue
                t_block, _ = first_hit(ox, oy, oz, vx / distance, vy / distance, vz / distance, v0, e1, e2)
                if t_block < distance:
                    continue
                cos_gamma = math.sqrt(1.0 - (radius / distance) ** 2)
                path = length + distance
                k = int(path / bin_length)
                if k < n_bins:
                    geometric = 2.0 * cos_theta * (1.0 - cos_gamma) * inv_area
                    for b in range(n_bands):
                        value = (energy[b] * (1.0 - absorption[m, b]) * diffuse_probability[m] * geometric
                                 * math.exp(-attenuation[b] * path))
                        histogram[r, b, k] += value
                        surface_received[r, b, surface] += value

            # Wall absorption, then a specular or diffuse direction
            for b in range(n_bands):
                absorbed[b] += energy[b] * absorption[m, b]
                energy[b] *= 1.0 - absorption[m, b]
            state, u = _uniform(state)
            if u < diffuse_probability[m]:
                state, u1 = _uniform(state)
                state, u2 = _uniform(state)
                dx, dy, dz = _lambert(nx, ny, nz, u1, u2)
                after_diffuse = True
                specular_only = False
            else:
                along_normal = dx * nx + dy * ny + dz * nz
                dx, dy, dz = dx - 2.0 * along_normal * nx, dy - 2.0 * along_normal * ny, dz - 2.0 * along_normal * nz
                after_diffuse = False
            last_surface = surface

            peak = 0.0
            for b in range(n_bands):
                peak = max(peak, energy[b] * math.exp(-attenuation[b] * length))
            if peak * n_rays < ENERGY_THRESHOLD:
                remaining += energy
                break

    counters[0] += escaped_rays
    counters[1] += reflections


@njit(cache=True, parallel=True)
def _trace(n_rays, seed, stream, source, receivers, radius, v0, e1, e2, tri_polygon, poly_normal, poly_material,
           poly_surface, n_surfaces, absorption, diffuse_probability, attenuation, speed, time_step, n_bins,
           max_length, exclude_order):
    """
    Trace all rays of one group of bands in CHUNKS parallel chunks
    :return: Per-chunk histograms (C, R, B, K), energies (C, 3, B), counters (C, 2), surface hits (C, S),
             surface incident and absorbed energies (C, 2, B, S) and received energies (C, R, B, S)
    """
    n_receivers, n_bands = receivers.shape[0], absorption.shape[1]
    histograms = np.zeros((CHUNKS, n_receivers, n_bands, n_bins))
    energies = np.zeros((CHUNKS, 3, n_bands))
    counters = np.zeros((CHUNKS, 2), np.int64)
    hits = np.zeros((CHUNKS, n_surfaces), np.int64)
    surface_energies = np.zeros((CHUNKS, 2, n_bands, n_surfaces))
    received = np.zeros((CHUNKS, n_receivers, n_bands, n_surfaces))
    for chunk in prange(CHUNKS):
        _trace_range(chunk * n_rays // CHUNKS, (chunk + 1) * n_rays // CHUNKS, n_rays, seed, stream, source,
                     receivers, radius, v0, e1, e2, tri_polygon, poly_normal, poly_material, poly_surface, absorption,
                     diffuse_probability, attenuation, speed, time_step, max_length, exclude_order,
                     histograms[chunk], energies[chunk, 0], energies[chunk, 1], energies[chunk, 2], counters[chunk],
                     hits[chunk], surface_energies[chunk, 0], surface_energies[chunk, 1], received[chunk])
    return histograms, energies, counters, hits, surface_energies, received


@dataclass(frozen=True)
class RayTracing:
    """
    Output of trace_rays; energy fractions are of the energy emitted by the source, per band
    """
    energy: np.ndarray  # (R, B, K) histogram, 1/m2 per unit source energy
    absorbed: np.ndarray  # (B,) absorbed by the walls
    remaining: np.ndarray  # (B,) still in the rays when tracing stopped (air absorption not included)
    escaped: np.ndarray  # (B,) carried out by rays that found no wall
    escaped_rays: int  # summed over the band groups
    reflections: int  # wall hits of all rays, summed over the band groups
    band_groups: int  # number of separately traced groups of bands
    surface_hits: np.ndarray  # (B, S) ray hits per surface, by the rays that carried each band
    surface_incident: np.ndarray  # (B, S) energy arriving at each surface (air absorption included)
    surface_absorbed: np.ndarray  # (B, S) energy absorbed by each surface (air absorption included)
    surface_received: np.ndarray  # (R, B, S) energy at the receivers (histogram summed over time), 1/m2,
    # by the surface of the last reflection; the direct sound is not included


def scattering_groups(scattering: np.ndarray):
    """
    Group the bands that have the same scattering coefficient in every material
    :param scattering: (M, B) scattering coefficients
    :return: List of band index arrays
    """
    _, group_of_band = np.unique(scattering.T, axis=0, return_inverse=True)
    return [np.flatnonzero(group_of_band.ravel() == g) for g in range(group_of_band.max() + 1)]


def trace_rays(room: Room, settings: SimulationSettings, speed_of_sound: float, air_attenuation,
               exclude_specular_order: int = -1):
    """
    Trace rays from the source and collect energy histograms at the receivers
    :param room: Room
    :param settings: rays, receiver_radius, time_step, max_time and seed are used
    :param speed_of_sound: m/s
    :param air_attenuation: (B,) intensity attenuation constant m per band, 1/m
    :param exclude_specular_order: Purely specular paths up to this order (0 = direct sound) are not
                                   recorded, because the image sources hold them; -1 records everything
    :return: RayTracing
    """
    v0, e1, e2 = kernel_arrays(room)
    n_bins = math.ceil(settings.max_time / settings.time_step)
    absorption, scattering = room.materials.absorption, room.materials.scattering
    attenuation = np.asarray(air_attenuation, dtype=float)

    n_receivers, n_bands, n_surfaces = len(room.receivers), absorption.shape[1], room.n_surfaces
    histogram = np.zeros((n_receivers, n_bands, n_bins))
    absorbed, remaining, escaped = np.zeros(n_bands), np.zeros(n_bands), np.zeros(n_bands)
    surface_hits = np.zeros((n_bands, n_surfaces), np.int64)
    surface_incident, surface_absorbed = np.zeros((n_bands, n_surfaces)), np.zeros((n_bands, n_surfaces))
    surface_received = np.zeros((n_receivers, n_bands, n_surfaces))
    escaped_rays = reflections = 0
    groups = scattering_groups(scattering)
    for stream, bands in enumerate(groups):
        result = _trace(
            settings.rays, settings.seed, stream, room.source.astype(float), room.receivers.astype(float),
            float(settings.receiver_radius), v0, e1, e2, room.tri_polygon.astype(np.int64),
            np.ascontiguousarray(room.poly_normal), room.poly_material.astype(np.int64),
            room.poly_surface.astype(np.int64), n_surfaces,
            np.ascontiguousarray(absorption[:, bands]), np.ascontiguousarray(scattering[:, bands[0]]),
            np.ascontiguousarray(attenuation[bands]), float(speed_of_sound), float(settings.time_step), n_bins,
            float(settings.max_time * speed_of_sound), int(exclude_specular_order))
        histograms, energies, counters, hits, surface_energies, received = result
        # Sum the chunks in a fixed order, so the result does not depend on the thread count
        histogram[:, bands] = histograms.sum(axis=0)
        absorbed[bands], remaining[bands], escaped[bands] = energies.sum(axis=0)
        escaped_rays += int(counters[:, 0].sum())
        reflections += int(counters[:, 1].sum())
        surface_hits[bands] = hits.sum(axis=0)
        surface_incident[bands], surface_absorbed[bands] = surface_energies.sum(axis=0)
        surface_received[:, bands] = received.sum(axis=0)
    return RayTracing(histogram, absorbed, remaining, escaped, escaped_rays, reflections, len(groups),
                      surface_hits, surface_incident, surface_absorbed, surface_received)
