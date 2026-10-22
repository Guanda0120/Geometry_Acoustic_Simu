"""Numba kernels on the room triangles, shared by the ray tracer and the image sources."""
import numpy as np
from numba import njit

from gasim.geometry.room import Room

EPSILON = 1e-7  # m, hits closer than this to the ray start are ignored (self-intersection)


def kernel_arrays(room: Room):
    """
    Triangle data in the form the kernels use
    :param room: Room
    :return: v0, e1 = v1 - v0, e2 = v2 - v0, each (T, 3) contiguous
    """
    triangles = room.tri_vertices
    return (np.ascontiguousarray(triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 1] - triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 2] - triangles[:, 0]))


@njit(cache=True)
def first_hit(ox, oy, oz, dx, dy, dz, v0, e1, e2):
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
def segment_blocked(ax, ay, az, bx, by, bz, v0, e1, e2):
    """
    Whether a wall lies between two points (hits at the end points themselves do not count)
    """
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    t, _ = first_hit(ax, ay, az, dx / length, dy / length, dz / length, v0, e1, e2)
    return t < length - 1e-6


@njit(cache=True)
def in_triangles(px, py, pz, first, count, v0, e1, e2):
    """
    Whether a point in the plane of a polygon lies on one of its triangles first .. first + count - 1
    """
    for k in range(first, first + count):
        wx, wy, wz = px - v0[k, 0], py - v0[k, 1], pz - v0[k, 2]
        d00 = e1[k, 0] * e1[k, 0] + e1[k, 1] * e1[k, 1] + e1[k, 2] * e1[k, 2]
        d01 = e1[k, 0] * e2[k, 0] + e1[k, 1] * e2[k, 1] + e1[k, 2] * e2[k, 2]
        d11 = e2[k, 0] * e2[k, 0] + e2[k, 1] * e2[k, 1] + e2[k, 2] * e2[k, 2]
        d20 = wx * e1[k, 0] + wy * e1[k, 1] + wz * e1[k, 2]
        d21 = wx * e2[k, 0] + wy * e2[k, 1] + wz * e2[k, 2]
        denominator = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / denominator
        v = (d00 * d21 - d01 * d20) / denominator
        if u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12:
            return True
    return False
