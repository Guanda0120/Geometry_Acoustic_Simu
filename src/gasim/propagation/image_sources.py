"""
Image sources: exact specular paths up to a given reflection order.

The source is mirrored recursively in the boundary planes; an image is only kept if its parent lies in
front of the mirror plane (Borish 1984). A path is valid for a receiver if, traced back from the
receiver, every reflection point lies inside its polygon and no segment is blocked by another wall.
Its energy is prod (1 - alpha_i)(1 - s_i) exp(-m d) / (4 pi d2): the specular share only, because the
rays carry every path that scatters at least once (docs/rewrite-plan.md, Sect. 4.4).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np
from numba import njit

from gasim.geometry.kernels import in_triangles, kernel_arrays, segment_blocked
from gasim.geometry.room import Room


@njit(cache=True)
def _path_is_valid(level, sequence, images, receiver, source, poly_normal, poly_d, tri_first, tri_count,
                   v0, e1, e2, reflection):
    """
    Back-trace a candidate path from the receiver; fill the reflection points if it is valid
    :param level: Number of reflections
    :param sequence: Planes of the reflections in order from the source
    :param images: images[j] is the image after j reflections, images[0] the source
    :param reflection: (max_order, 3) output, reflection points in order from the source
    """
    xx, xy, xz = receiver[0], receiver[1], receiver[2]
    for j in range(level - 1, -1, -1):
        p = sequence[j]
        nx, ny, nz = poly_normal[p, 0], poly_normal[p, 1], poly_normal[p, 2]
        side_point = nx * xx + ny * xy + nz * xz + poly_d[p]
        ix, iy, iz = images[j + 1, 0], images[j + 1, 1], images[j + 1, 2]
        side_image = nx * ix + ny * iy + nz * iz + poly_d[p]
        if side_point <= 1e-9 or side_image >= -1e-9:
            return False
        t = side_point / (side_point - side_image)
        qx, qy, qz = xx + t * (ix - xx), xy + t * (iy - xy), xz + t * (iz - xz)
        if not in_triangles(qx, qy, qz, tri_first[p], tri_count[p], v0, e1, e2):
            return False
        reflection[j, 0], reflection[j, 1], reflection[j, 2] = qx, qy, qz
        xx, xy, xz = qx, qy, qz

    # Visibility of every segment: receiver -> last reflection -> ... -> first reflection -> source
    ax, ay, az = receiver[0], receiver[1], receiver[2]
    for j in range(level - 1, -1, -1):
        if segment_blocked(ax, ay, az, reflection[j, 0], reflection[j, 1], reflection[j, 2], v0, e1, e2):
            return False
        ax, ay, az = reflection[j, 0], reflection[j, 1], reflection[j, 2]
    return not segment_blocked(ax, ay, az, source[0], source[1], source[2], v0, e1, e2)


@njit(cache=True, nogil=True)
def _find_paths(max_order, source, receiver, poly_normal, poly_d, tri_first, tri_count, v0, e1, e2, capacity,
                first_plane, with_direct):
    """
    Depth-first search over plane sequences up to max_order
    :param first_plane: Only sequences starting with this plane (-1: all), so subtrees can run in parallel
    :param with_direct: Also test the direct sound
    :return: orders, planes, points of the valid paths, their number, and whether capacity was exceeded
    """
    width = max(max_order, 1)
    orders = np.zeros(capacity, np.int64)
    planes = np.full((capacity, width), -1, np.int64)
    points = np.full((capacity, max_order + 2, 3), np.nan)
    count = 0
    reflection = np.empty((width, 3))

    # Direct sound
    if with_direct and not segment_blocked(source[0], source[1], source[2], receiver[0], receiver[1], receiver[2],
                                           v0, e1, e2):
        points[0, 0] = source
        points[0, 1] = receiver
        count = 1

    sequence = np.empty(width, np.int64)
    images = np.empty((max_order + 1, 3))
    images[0] = source
    choice = np.full(max_order + 1, -1, np.int64)
    n_planes = poly_normal.shape[0]
    level = 1
    while 1 <= level <= max_order:
        choice[level] += 1
        p = choice[level]
        if level == 1 and first_plane >= 0:
            if p > first_plane:
                break
            p = choice[1] = first_plane
        if p >= n_planes:
            choice[level] = -1
            level -= 1
            continue
        if level > 1 and p == sequence[level - 2]:
            continue
        parent = images[level - 1]
        side = poly_normal[p, 0] * parent[0] + poly_normal[p, 1] * parent[1] + poly_normal[p, 2] * parent[2] \
            + poly_d[p]
        if side <= 1e-9:
            continue  # parent behind or on the plane: the image cannot be seen through this polygon
        images[level, 0] = parent[0] - 2 * side * poly_normal[p, 0]
        images[level, 1] = parent[1] - 2 * side * poly_normal[p, 1]
        images[level, 2] = parent[2] - 2 * side * poly_normal[p, 2]
        sequence[level - 1] = p
        if _path_is_valid(level, sequence, images, receiver, source, poly_normal, poly_d, tri_first, tri_count,
                          v0, e1, e2, reflection):
            if count == capacity:
                return orders, planes, points, count, True
            orders[count] = level
            planes[count, :level] = sequence[:level]
            points[count, 0] = source
            points[count, 1:level + 1] = reflection[:level]
            points[count, level + 1] = receiver
            count += 1
        if level < max_order:
            level += 1
    return orders, planes, points, count, False


@dataclass(frozen=True)
class ImageSourcePaths:
    """
    Valid specular paths of all receivers, one row per path
    """
    receiver: np.ndarray  # (P,) receiver index
    order: np.ndarray  # (P,) number of reflections, 0 = direct sound
    planes: np.ndarray  # (P, N) polygon of each reflection, -1 after the last
    points: np.ndarray  # (P, N + 2, 3) source, reflection points, receiver; nan after the receiver
    length: np.ndarray  # (P,) m
    energy: np.ndarray  # (P, B) per unit area per unit source energy, 1/m2

    def counts(self, receiver: int):
        """
        :return: Number of paths per order (0 .. N) for one receiver
        """
        return np.bincount(self.order[self.receiver == receiver], minlength=self.planes.shape[1] + 1)


def find_image_sources(room: Room, max_order: int, air_attenuation):
    """
    All valid specular paths up to max_order for every receiver
    :param room: Room
    :param max_order: Highest reflection order (0: direct sound only)
    :param air_attenuation: (B,) intensity attenuation constant m per band, 1/m
    :return: ImageSourcePaths
    """
    v0, e1, e2 = kernel_arrays(room)
    n_polygons = room.n_polygons
    tri_count = np.bincount(room.tri_polygon, minlength=n_polygons).astype(np.int64)
    tri_first = np.concatenate([[0], np.cumsum(tri_count)[:-1]]).astype(np.int64)
    if np.any(np.diff(room.tri_polygon) < 0):
        raise ValueError("Triangles must be grouped by polygon")

    def search(task):
        index, first_plane = task
        capacity = 1024
        while True:
            orders, planes, points, count, overflow = _find_paths(
                max_order, room.source.astype(float), room.receivers[index].astype(float),
                np.ascontiguousarray(room.poly_normal), room.poly_d.astype(float), tri_first, tri_count, v0, e1, e2,
                capacity, first_plane, first_plane <= 0)
            if not overflow:
                return np.full(count, index), orders[:count], planes[:count, :max_order], points[:count]
            capacity *= 4

    # One task per receiver and first reflecting plane; the GIL is released inside the search
    first_planes = range(n_polygons) if max_order > 0 else [-1]
    tasks = [(index, plane) for index in range(len(room.receivers)) for plane in first_planes]
    with ThreadPoolExecutor(max_workers=numba.get_num_threads()) as pool:
        found = list(pool.map(search, tasks))
    columns = {name: [f[i] for f in found] for i, name in enumerate(('receiver', 'order', 'planes', 'points'))}
    receiver, order = np.concatenate(columns['receiver']), np.concatenate(columns['order'])
    planes, points = np.concatenate(columns['planes']), np.concatenate(columns['points'])
    length = np.array([np.linalg.norm(np.diff(p[:o + 2], axis=0), axis=1).sum() for p, o in zip(points, order)])

    # Specular share of every reflection, per band
    materials = room.materials
    keep = np.ones((len(order), materials.absorption.shape[1]))
    for column in range(max_order):
        reflected = planes[:, column] >= 0
        material = room.poly_material[planes[reflected, column]]
        keep[reflected] *= (1 - materials.absorption[material]) * (1 - materials.scattering[material])
    energy = keep * np.exp(-np.outer(length, air_attenuation)) / (4 * np.pi * length[:, None] ** 2)
    return ImageSourcePaths(receiver.astype(np.int64), order, planes, points, length, energy)


def add_to_histogram(paths: ImageSourcePaths, histogram: np.ndarray, time_step: float, speed_of_sound: float):
    """
    Add the path energies to the bins of their arrival times
    :param paths: ImageSourcePaths
    :param histogram: (R, B, K) energy histogram, changed in place
    :param time_step: Bin width, s
    :param speed_of_sound: m/s
    """
    bins = (paths.length / (speed_of_sound * time_step)).astype(np.int64)
    inside = bins < histogram.shape[2]
    for receiver, k, energy in zip(paths.receiver[inside], bins[inside], paths.energy[inside]):
        histogram[receiver, :, k] += energy
