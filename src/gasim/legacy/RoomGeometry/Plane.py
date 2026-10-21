import numpy as np
import rhino3dm as rhino3dm
from gasim.legacy.Graphic.BaseGeometry import BaseGeometry
from gasim.legacy.Acoustic.Material import Material


class Plane:
    """
    This is the object to Construct room
    """

    def __init__(self, rhino_plane: rhino3dm.Brep, material: Material):
        """
        Init plane
        :param rhino_plane: A rhino Brep object
        :param material: The material object of this plane
        """
        self.rhino_plane = rhino_plane
        self.material = material
        self.material_name = material.name
        self.diffuse_abs_coe = material.diffuse_abs
        self.normal_abs_coe = material.normal_abs
        self.ref_coe = material.ref_0
        self.sca_coe = material.scatter
        self.array_plane = []

        # Change the rhino data to Ndarray with shape (n,3)
        for idx in range(len(rhino_plane.Geometry.Edges)):
            # TODO BUG is Here: The point is not in the order
            # No closed polygon
            pt_1 = rhino_plane.Geometry.Edges[idx].PointAtStart
            tmp_pt1 = np.array([pt_1.X, pt_1.Y, pt_1.Z])
            pt_2 = rhino_plane.Geometry.Edges[idx].PointAtEnd
            tmp_pt2 = np.array([pt_2.X, pt_2.Y, pt_2.Z])
            tmp_pt = np.array([tmp_pt1, tmp_pt2])
            # Get the start point and end point, append it into array_plane
            self.array_plane.append(tmp_pt)
        # print(f"Array Plane: {self.array_plane}")
        # Switch the right order of the first item
        # if (self.array_plane[0][0,:] == self.array_plane[1][0,:]).all():
        #     self.array_plane[0] = np.array([self.array_plane[0][1,:],self.array_plane[0][0,:]])
        # elif (self.array_plane[0][0,:] == self.array_plane[1][1,:]).all():
        #     self.array_plane[0] = np.array([self.array_plane[0][1, :], self.array_plane[0][0, :]])
        tmp_plane = [self.array_plane[0]]
        rest_plane = self.array_plane[1:]
        while len(rest_plane) != 0:
            compare_pt = tmp_plane[-1][1, :]
            for idx in range(len(rest_plane)):
                if np.allclose(compare_pt, rest_plane[idx][0, :]):
                    tmp_plane.append(rest_plane[idx])
                    del rest_plane[idx]
                    break
                elif np.allclose(compare_pt, rest_plane[idx][1, :]):
                    tmp_plane.append(np.array([rest_plane[idx][1, :], rest_plane[idx][0, :]]))
                    del rest_plane[idx]
                    break
        self.array_plane = tmp_plane
        # print(f"Array Plane: {self.array_plane}")

        # polygon is the list contain the unique point
        polygon = []
        for i in range(len(self.array_plane)):
            polygon.append(self.array_plane[i][0,:])

        # Struct the polygon
        polygon = np.array(polygon)
        self.polygon = polygon

        self.array_plane = np.array(self.array_plane)
        self.planar_state = BaseGeometry.is_planar(self.polygon)
        self.normal_direc, self.cen_pt, self.intercept_d = BaseGeometry.normal_direction(self.polygon)
        self.area = BaseGeometry.area(polygon, self.cen_pt)

    def __str__(self):
        return f"The plane is {self.polygon}. The material is {self.material_name}"
