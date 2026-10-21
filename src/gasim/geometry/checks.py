"""Checks that a room model is closed and unambiguous before it is simulated."""
from dataclasses import dataclass

import numpy as np

from gasim.geometry.intersect import first_hit, points_inside
from gasim.geometry.room import Room


@dataclass(frozen=True)
class RoomReport:
    """
    Results of check_room
    """
    source_inside: bool
    receivers_inside: np.ndarray  # (R,) bool
    escaped_rays: int  # rays from the source that left the room
    traced_rays: int
    overlaps: list  # (polygon i, polygon j) coplanar polygons drawn on top of each other

    @property
    def ok(self):
        return self.source_inside and bool(self.receivers_inside.all()) and self.escaped_rays == 0 \
            and not self.overlaps


def leak_test(room: Room, n_rays: int = 2000, reflections: int = 30, seed: int = 0):
    """
    Trace specular rays from the source and count those that find no wall
    :param room: Room
    :param n_rays: Number of rays
    :param reflections: Reflections per ray
    :param seed: Random seed of the ray directions
    :return: Number of escaped rays
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_rays, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.broadcast_to(room.source, directions.shape).copy()
    normals = room.poly_normal[room.tri_polygon]
    escaped = 0
    for _ in range(reflections):
        distance, triangle = first_hit(origins, directions, room.tri_vertices, t_min=1e-7)
        lost = triangle < 0
        escaped += int(lost.sum())
        origins, directions, triangle, distance = origins[~lost], directions[~lost], triangle[~lost], distance[~lost]
        if len(origins) == 0:
            break
        origins = origins + distance[:, None] * directions
        n = normals[triangle]
        directions = directions - 2 * np.einsum('ij,ij->i', directions, n)[:, None] * n
    return escaped


def coplanar_overlaps(room: Room, tolerance: float = 1e-4):
    """
    Pairs of polygons in the same plane that cover each other (e.g. a window drawn on a wall)
    :param room: Room
    :param tolerance: Plane distance tolerance, m
    :return: List of (i, j) polygon index pairs, i < j
    """
    pairs = []
    for i in range(room.n_polygons):
        for j in range(i + 1, room.n_polygons):
            alignment = room.poly_normal[i] @ room.poly_normal[j]
            if abs(alignment) < 1 - 1e-9:
                continue
            offset = room.poly_d[j] - np.sign(alignment) * room.poly_d[i]
            if abs(offset) > tolerance:
                continue
            if _covers(room, i, j) or _covers(room, j, i):
                pairs.append((i, j))
    return pairs


def _covers(room: Room, outer: int, inner: int):
    """
    Whether a triangle centroid of polygon `inner` lies inside polygon `outer` (same plane)
    """
    normal = room.poly_normal[outer]
    for centroid in room.tri_vertices[room.tri_polygon == inner].mean(axis=1):
        for a, b, c in room.tri_vertices[room.tri_polygon == outer]:
            # Barycentric sign test in the plane of `outer`
            if (np.cross(b - a, centroid - a) @ normal > 0 and np.cross(c - b, centroid - b) @ normal > 0
                    and np.cross(a - c, centroid - c) @ normal > 0):
                return True
    return False


def check_room(room: Room, n_rays: int = 2000):
    """
    Run all model checks
    :param room: Room
    :param n_rays: Rays for the leak test
    :return: RoomReport
    """
    inside = points_inside(np.vstack([room.source, room.receivers]), room.tri_vertices)
    return RoomReport(source_inside=bool(inside[0]),
                      receivers_inside=inside[1:],
                      escaped_rays=leak_test(room, n_rays),
                      traced_rays=n_rays,
                      overlaps=coplanar_overlaps(room))
