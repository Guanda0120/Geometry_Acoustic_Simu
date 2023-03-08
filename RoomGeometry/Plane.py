import numpy as np
import rhino3dm as rhino3dm
from Graphic.BaseGeometry import BaseGeometry
from Acoustic.Material import Material


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
                if (compare_pt == rest_plane[idx][0, :]).all():
                    tmp_plane.append(rest_plane[idx])
                    del rest_plane[idx]
                    break
                elif (compare_pt == rest_plane[idx][1, :]).all():
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

    def __str__(self):
        return f"The plane is {self.polygon}. The material is {self.material_name}"


if __name__ == "__main__":
    from FileIO.ReadRhino import Read3dm
    path = "/Users/liguanda/Desktop/MasterReasearch/MasterFinal/code/Simu/Test_File/Test_Theater.3dm"
    rhino_file = Read3dm(path)
    plane_dict, _, _ = rhino_file.convert_file()
    plane_list = []
    for key in plane_dict.keys():
        for rh_plane in plane_dict[key]:
            tmp_plane = Plane(rh_plane,np.asarray([0.1,0.1,0.1,0.1,0.1]),np.asarray([0.1,0.1,0.1,0.1,0.1]),key)
            plane_list.append(tmp_plane)
            print(f"Array: {tmp_plane.polygon}, is planar: {tmp_plane.planar_state}, "
                  f"normal direction is: {tmp_plane.normal_direc}")
    print(f"Total {len(plane_list)} planes")