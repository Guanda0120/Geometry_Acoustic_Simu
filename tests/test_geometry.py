import numpy as np
import pytest
from conftest import box_polygons

from gasim.geometry.build import build_room, clean_polygon, newell_normal, triangulate
from gasim.geometry.checks import check_room, coplanar_overlaps, leak_test
from gasim.geometry.intersect import points_inside


def test_box_volume_area_and_inward_normals(make_box_room):
    room = make_box_room()
    assert room.volume == pytest.approx(4 * 3 * 2.5)
    assert room.surface_area == pytest.approx(2 * (4 * 3 + 4 * 2.5 + 3 * 2.5))
    centre = np.array([2.0, 1.5, 1.25])
    for vertices, normal in zip(room.poly_vertices, room.poly_normal):
        assert normal @ (centre - vertices.mean(axis=0)) > 0


def test_triangles_follow_polygon_normal(make_box_room):
    tri = make_box_room().tri_vertices
    room = make_box_room()
    tri_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum('ij,ij->i', tri_normals, room.poly_normal[room.tri_polygon]) > 0)


def test_concave_polygon_triangulation():
    # L-shaped floor of 3 x 3 m with a 1.5 x 1.5 m corner cut out
    vertices = np.array([[0, 0, 0], [3, 0, 0], [3, 1.5, 0], [1.5, 1.5, 0], [1.5, 3, 0], [0, 3, 0]], dtype=float)
    normal = newell_normal(vertices)
    triangles = vertices[triangulate(vertices, normal)]
    areas = 0.5 * np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) @ normal
    assert len(triangles) == 4
    assert np.all(areas > 0)
    assert areas.sum() == pytest.approx(9 - 1.5 ** 2)


def test_clean_polygon_drops_duplicate_and_collinear_points():
    square = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 0], [2, 2, 0], [0, 2, 0], [0, 0, 0]], dtype=float)
    assert len(clean_polygon(square)) == 4


def test_non_planar_polygon_is_rejected(box_materials):
    polygons = box_polygons()
    layer, vertices = polygons[0]
    vertices = vertices.astype(float)
    vertices[0, 2] += 0.01
    polygons[0] = (layer, vertices)
    with pytest.raises(ValueError, match='not planar'):
        build_room(polygons, box_materials, [1, 1, 1], [[2, 2, 1]])


def test_points_inside_box(make_box_room):
    room = make_box_room()
    inside = points_inside(np.array([[2, 1.5, 1.25], [0.01, 0.01, 0.01], [5, 1, 1], [2, 1.5, -0.1]]),
                           room.tri_vertices)
    np.testing.assert_array_equal(inside, [True, True, False, False])


def test_closed_box_passes_all_checks(make_box_room):
    assert check_room(make_box_room()).ok


def test_missing_wall_leaks(make_box_room):
    assert leak_test(make_box_room(drop_last=True), n_rays=500) > 0


def test_window_drawn_on_wall_is_reported_and_oriented(make_box_room):
    window = np.array([[1, 0, 0.8], [2, 0, 0.8], [2, 0, 1.8], [1, 0, 1.8]], dtype=float)  # on the wall y = 0
    room = make_box_room(extra=[('Window', window)])
    assert len(coplanar_overlaps(room)) == 1
    np.testing.assert_allclose(room.poly_normal[-1], [0, 1, 0])
