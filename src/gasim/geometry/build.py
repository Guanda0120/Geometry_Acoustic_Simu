"""Turn drawn polygons into a Room: clean up, triangulate and orient them."""
import numpy as np

from gasim.geometry.intersect import points_inside
from gasim.geometry.room import Room
from gasim.materials import MaterialTable

PLANARITY_TOLERANCE = 1e-6  # m
INWARD_OFFSET = 1e-4  # m, offset of the probe point used to orient a polygon


def newell_normal(vertices: np.ndarray):
    """
    Normal of a polygon from its winding (Newell's method), robust for concave polygons
    :param vertices: (n, 3) ordered vertices
    :return: Unit normal, counter-clockwise winding about it
    """
    current, following = vertices, np.roll(vertices, -1, axis=0)
    normal = np.array([np.sum((current[:, 1] - following[:, 1]) * (current[:, 2] + following[:, 2])),
                       np.sum((current[:, 2] - following[:, 2]) * (current[:, 0] + following[:, 0])),
                       np.sum((current[:, 0] - following[:, 0]) * (current[:, 1] + following[:, 1]))])
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ValueError("Degenerate polygon (zero area)")
    return normal / length


def clean_polygon(vertices: np.ndarray, tolerance: float = 1e-9):
    """
    Remove repeated closing points, duplicate and collinear vertices
    :param vertices: (n, 3) ordered vertices
    :param tolerance: Relative tolerance on lengths
    :return: (m, 3) vertices, m >= 3
    """
    vertices = np.asarray(vertices, dtype=float)
    scale = max(np.ptp(vertices, axis=0).max(), 1e-12)
    while len(vertices) > 3:
        previous, following = np.roll(vertices, 1, axis=0), np.roll(vertices, -1, axis=0)
        turn = np.linalg.norm(np.cross(vertices - previous, following - vertices), axis=1)
        redundant = np.flatnonzero(turn < tolerance * scale ** 2)
        if redundant.size == 0:
            break
        vertices = np.delete(vertices, redundant[0], axis=0)
    if len(vertices) < 3:
        raise ValueError("Polygon has fewer than three distinct vertices")
    return vertices


def triangulate(vertices: np.ndarray, normal: np.ndarray):
    """
    Ear-clipping triangulation of a planar, simple polygon
    :param vertices: (n, 3) vertices wound counter-clockwise about normal
    :param normal: Unit polygon normal
    :return: (n - 2, 3) vertex indices, each triangle counter-clockwise about normal
    """
    # 2D coordinates in the polygon plane, with (u, w, normal) right-handed
    u = vertices[1] - vertices[0]
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    points = np.column_stack([(vertices - vertices[0]) @ u, (vertices - vertices[0]) @ w])

    def cross_2d(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    remaining = list(range(len(vertices)))
    triangles = []
    while len(remaining) > 3:
        for k in range(len(remaining)):
            i, j, m = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = points[i], points[j], points[m]
            if cross_2d(a, b, c) <= 1e-14:
                continue  # reflex or degenerate corner
            others = [p for p in remaining if p not in (i, j, m)]
            inside = any(cross_2d(a, b, points[p]) >= 0 and cross_2d(b, c, points[p]) >= 0
                         and cross_2d(c, a, points[p]) >= 0 for p in others)
            if not inside:
                triangles.append((i, j, m))
                del remaining[k]
                break
        else:
            raise ValueError("Polygon could not be triangulated (self-intersecting?)")
    triangles.append(tuple(remaining))
    return np.array(triangles)


def build_room(polygons, materials: MaterialTable, source, receivers):
    """
    Build a Room from drawn polygons
    :param polygons: Iterable of (layer name, (n, 3) vertices); the layer name is the material
    :param materials: MaterialTable with every layer used
    :param source: (3,) source position
    :param receivers: (R, 3) receiver positions
    :return: Room with inward normals
    """
    layers, vertex_list, normals, material_index = [], [], [], []
    for layer, vertices in polygons:
        vertices = clean_polygon(vertices)
        normal = newell_normal(vertices)
        distance = np.abs((vertices - vertices[0]) @ normal).max()
        if distance > PLANARITY_TOLERANCE:
            raise ValueError(f"Polygon on layer {layer!r} is not planar ({distance:.2e} m off its plane)")
        layers.append(layer)
        vertex_list.append(vertices)
        normals.append(normal)
        material_index.append(materials.index(layer))

    # Triangulate with the drawn winding, then flip whole polygons whose normal points outward
    triangles = [vertices[triangulate(vertices, normal)] for vertices, normal in zip(vertex_list, normals)]
    tri_vertices = np.concatenate(triangles)
    probes = np.array([_inner_point(tris) + INWARD_OFFSET * normal for tris, normal in zip(triangles, normals)])
    flipped = ~points_inside(probes, tri_vertices)

    tri_polygon = []
    for p, (vertices, tris) in enumerate(zip(vertex_list, triangles)):
        if flipped[p]:
            vertex_list[p] = vertices[::-1].copy()
            normals[p] = -normals[p]
            triangles[p] = tris[:, ::-1]
        tri_polygon.extend([p] * len(tris))

    normals = np.array(normals)
    areas = np.array([0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
                      for t in triangles])
    return Room(poly_vertices=tuple(vertex_list),
                poly_layer=tuple(layers),
                poly_normal=normals,
                poly_d=-np.einsum('ij,ij->i', normals, np.array([v[0] for v in vertex_list])),
                poly_area=areas,
                poly_material=np.array(material_index),
                poly_flipped=flipped,
                tri_vertices=np.concatenate(triangles),
                tri_polygon=np.array(tri_polygon),
                materials=materials,
                source=np.asarray(source, dtype=float),
                receivers=np.atleast_2d(np.asarray(receivers, dtype=float)))


def _inner_point(triangles: np.ndarray):
    """
    A point strictly inside a polygon: the centroid of its largest triangle
    :param triangles: (k, 3, 3) triangles of the polygon
    :return: (3,) point
    """
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return triangles[np.argmax(areas)].mean(axis=0)
