"""
Stochastic ray tracing with energy particles and diffuse rain.

Each of the N rays carries 1/N of the source energy in every band. A ray that crosses a receiver sphere
of radius r deposits energy / (pi r2), so the histogram holds energy per unit area per unit source
energy (Vorlaender 2020, Sect. 11.3.3). At every wall hit the energy is multiplied by (1 - alpha) and the
ray continues specularly or in a Lambert direction. Diffuse rain adds, at every wall hit, the expected
energy the diffuse share sends straight to each visible receiver (Vorlaender 2020, Eq. 11.7; Schroeder
2011, Eq. 5.20); sphere detection is therefore skipped on a segment that starts with a diffuse
reflection, so that every path is counted once in expectation.

Bands whose scattering coefficients differ are traced separately, each group with its own rays: one
direction must serve all bands of a ray, and reweighting bands by s_b / p would be unbiased but its
weights multiply over hundreds of reflections and degenerate.
"""
import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from gasim.config import SimulationSettings
from gasim.geometry.room import Room

ENERGY_THRESHOLD = 1e-7  # tracing stops when a ray (air absorption included) is 70 dB below its start
EPSILON = 1e-7  # m, hits closer than this to the ray start are ignored (self-intersection)

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
def _first_hit(ox, oy, oz, dx, dy, dz, v0, e1, e2):
    """
    Nearest triangle along a ray (Moeller-Trumbore, two-sided)
    :return: Distance, triangle index (-1 if none)
    """
    best_t = np.inf
    best = -1
    for k in range(v0.shape[0]):
        px = dy * e2[k, 2] - dz * e2[k, 1]
        py = dz * e2[k, 0] - dx * e2[k, 2]
        pz = dx * e2[k, 1] - dy * e2[k, 0]
        det = e1[k, 0] * px + e1[k, 1] * py + e1[k, 2] * pz
        if abs(det) < 1e-14:
            continue
        inv = 1.0 / det
        sx, sy, sz = ox - v0[k, 0], oy - v0[k, 1], oz - v0[k, 2]
        u = (sx * px + sy * py + sz * pz) * inv
        if u < 0.0 or u > 1.0:
            continue
        qx = sy * e1[k, 2] - sz * e1[k, 1]
        qy = sz * e1[k, 0] - sx * e1[k, 2]
        qz = sx * e1[k, 1] - sy * e1[k, 0]
        v = (dx * qx + dy * qy + dz * qz) * inv
        if v < 0.0 or u + v > 1.0:
            continue
        t = (e2[k, 0] * qx + e2[k, 1] * qy + e2[k, 2] * qz) * inv
        if EPSILON < t < best_t:
            best_t = t
            best = k
    return best_t, best


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
def _deposit(histogram, receiver, k, energy, factor, path, attenuation):
    """
    Add energy * factor * exp(-m path) of every band to bin k of a receiver
    """
    if k < histogram.shape[2]:
        for b in range(energy.shape[0]):
            histogram[receiver, b, k] += energy[b] * factor * math.exp(-attenuation[b] * path)


@njit(cache=True)
def _trace(n_rays, seed, stream, source, receivers, radius, v0, e1, e2, tri_polygon, poly_normal, poly_material,
           absorption, diffuse_probability, attenuation, speed, time_step, n_bins, max_length, exclude_order):
    """
    Trace one group of bands that share their scattering coefficients
    :param diffuse_probability: (M,) scattering coefficient of each material in this group of bands
    """
    n_receivers, n_bands = receivers.shape[0], absorption.shape[1]
    histogram = np.zeros((n_receivers, n_bands, n_bins))
    absorbed = np.zeros(n_bands)
    remaining = np.zeros(n_bands)
    escaped = np.zeros(n_bands)
    escaped_rays = 0
    reflections = 0
    inv_area = 1.0 / (math.pi * radius * radius)
    bin_length = speed * time_step
    energy = np.empty(n_bands)
    base = _mix(np.uint64(seed) + _GOLDEN * np.uint64(stream + 1))

    for ray in range(n_rays):
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

        while True:
            t_hit, tri = _first_hit(ox, oy, oz, dx, dy, dz, v0, e1, e2)
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
                    _deposit(histogram, r, int(path / bin_length), energy, inv_area, path, attenuation)

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

            # Diffuse rain to every visible receiver
            for r in range(n_receivers):
                vx, vy, vz = receivers[r, 0] - ox, receivers[r, 1] - oy, receivers[r, 2] - oz
                distance = math.sqrt(vx * vx + vy * vy + vz * vz)
                if distance <= radius:
                    continue
                cos_theta = (nx * vx + ny * vy + nz * vz) / distance
                if cos_theta <= 0.0:
                    continue
                t_block, _ = _first_hit(ox, oy, oz, vx / distance, vy / distance, vz / distance, v0, e1, e2)
                if t_block < distance:
                    continue
                cos_gamma = math.sqrt(1.0 - (radius / distance) ** 2)
                path = length + distance
                k = int(path / bin_length)
                if k < n_bins:
                    geometric = 2.0 * cos_theta * (1.0 - cos_gamma) * inv_area
                    for b in range(n_bands):
                        histogram[r, b, k] += (energy[b] * (1.0 - absorption[m, b]) * diffuse_probability[m]
                                               * geometric * math.exp(-attenuation[b] * path))

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

            peak = 0.0
            for b in range(n_bands):
                peak = max(peak, energy[b] * math.exp(-attenuation[b] * length))
            if peak * n_rays < ENERGY_THRESHOLD:
                remaining += energy
                break

    return histogram, absorbed, remaining, escaped, escaped_rays, reflections


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
    triangles = room.tri_vertices
    v0 = np.ascontiguousarray(triangles[:, 0])
    e1 = np.ascontiguousarray(triangles[:, 1] - triangles[:, 0])
    e2 = np.ascontiguousarray(triangles[:, 2] - triangles[:, 0])
    n_bins = math.ceil(settings.max_time / settings.time_step)
    absorption, scattering = room.materials.absorption, room.materials.scattering
    attenuation = np.asarray(air_attenuation, dtype=float)

    n_bands = absorption.shape[1]
    histogram = np.zeros((len(room.receivers), n_bands, n_bins))
    absorbed, remaining, escaped = np.zeros(n_bands), np.zeros(n_bands), np.zeros(n_bands)
    escaped_rays = reflections = 0
    groups = scattering_groups(scattering)
    for stream, bands in enumerate(groups):
        result = _trace(
            settings.rays, settings.seed, stream, room.source.astype(float), room.receivers.astype(float),
            float(settings.receiver_radius), v0, e1, e2, room.tri_polygon.astype(np.int64),
            np.ascontiguousarray(room.poly_normal), room.poly_material.astype(np.int64),
            np.ascontiguousarray(absorption[:, bands]), np.ascontiguousarray(scattering[:, bands[0]]),
            np.ascontiguousarray(attenuation[bands]), float(speed_of_sound), float(settings.time_step), n_bins,
            float(settings.max_time * speed_of_sound), int(exclude_specular_order))
        histogram[:, bands], absorbed[bands], remaining[bands], escaped[bands] = result[:4]
        escaped_rays += int(result[4])
        reflections += int(result[5])
    return RayTracing(histogram, absorbed, remaining, escaped, escaped_rays, reflections, len(groups))
