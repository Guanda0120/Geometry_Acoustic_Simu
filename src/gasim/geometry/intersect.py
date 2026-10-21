"""
Ray-triangle intersection with NumPy (Moeller-Trumbore), vectorised over rays and triangles.

Used for model checks; the propagation kernels get their own Numba versions.
"""
import numpy as np

# Oblique, irrational-looking directions: parity rays along them almost never graze an edge
_PARITY_DIRECTIONS = np.array([[0.5383, 0.6271, 0.5627],
                               [-0.7213, 0.2867, 0.6305],
                               [0.1839, -0.8121, 0.5537]])
_PARITY_DIRECTIONS /= np.linalg.norm(_PARITY_DIRECTIONS, axis=1, keepdims=True)


def ray_triangle_distances(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray, t_min: float = 1e-9):
    """
    Distance along each ray to each triangle, two-sided
    :param origins: (R, 3) ray start points
    :param directions: (R, 3) unit ray directions
    :param triangles: (T, 3, 3) triangle corners
    :param t_min: Hits closer than this are ignored (self-intersection)
    :return: (R, T) distances, inf where the ray misses
    """
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    p = np.cross(directions[:, None, :], e2[None, :, :])
    det = np.einsum('tk,rtk->rt', e1, p)
    parallel = np.abs(det) < 1e-12
    inv_det = 1.0 / np.where(parallel, 1.0, det)
    s = origins[:, None, :] - v0[None, :, :]
    u = np.einsum('rtk,rtk->rt', s, p) * inv_det
    q = np.cross(s, e1[None, :, :])
    v = np.einsum('rk,rtk->rt', directions, q) * inv_det
    t = np.einsum('tk,rtk->rt', e2, q) * inv_det
    hit = ~parallel & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min)
    return np.where(hit, t, np.inf)


def first_hit(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray, t_min: float = 1e-9,
              chunk: int = 512):
    """
    Nearest triangle hit by each ray
    :param origins: (R, 3) ray start points
    :param directions: (R, 3) unit ray directions
    :param triangles: (T, 3, 3) triangle corners
    :param t_min: Hits closer than this are ignored
    :param chunk: Rays per batch, bounds the memory use
    :return: (R,) distances (inf for no hit), (R,) triangle indices (-1 for no hit)
    """
    distance = np.full(len(origins), np.inf)
    index = np.full(len(origins), -1)
    for start in range(0, len(origins), chunk):
        t = ray_triangle_distances(origins[start:start + chunk], directions[start:start + chunk], triangles, t_min)
        if t.shape[1] == 0:
            continue
        nearest = np.argmin(t, axis=1)
        nearest_t = t[np.arange(len(t)), nearest]
        distance[start:start + chunk] = nearest_t
        index[start:start + chunk] = np.where(np.isfinite(nearest_t), nearest, -1)
    return distance, index


def points_inside(points: np.ndarray, triangles: np.ndarray, tolerance: float = 1e-7):
    """
    Whether points lie inside a closed triangle mesh, by majority vote of three parity rays
    :param points: (K, 3) points
    :param triangles: (T, 3, 3) closed mesh
    :param tolerance: Hits closer together than this along a ray count as one crossing, m
    :return: (K,) bool
    """
    points = np.atleast_2d(points)
    votes = np.zeros(len(points), dtype=int)
    for direction in _PARITY_DIRECTIONS:
        directions = np.broadcast_to(direction, points.shape)
        t = np.sort(ray_triangle_distances(points, directions, triangles, t_min=0.0), axis=1)
        # Hits at the same distance are one crossing: a shared edge, or a window drawn on a wall
        with np.errstate(invalid='ignore'):  # inf - inf between misses
            gaps = np.diff(t, axis=1)
        new_distance = np.concatenate([np.ones((len(t), 1), dtype=bool), gaps > tolerance], axis=1)
        crossings = (np.isfinite(t) & new_distance).sum(axis=1)
        votes += crossings % 2
    return votes >= 2
