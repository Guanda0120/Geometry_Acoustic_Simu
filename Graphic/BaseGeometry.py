import numpy as np


class BaseGeometry:

    @staticmethod
    def is_planar(polygon: np.ndarray):
        """
        Check it is planer polygon
        Ref: https://www.zhihu.com/question/304358033
        :param polygon: Ndarray represent polygon
        :return: True for planar, False for non-planar
        """
        # Check is legal input
        assert polygon.shape[1] == 3
        # concatenate with ones and compute determine it
        # ｜ x1  y1  z1  1 ｜
        # ｜ x2  y2  z2  1 ｜
        # ｜ x3  y3  z3  1 ｜
        # ｜ x4  y4  z4  1 ｜
        # dimension check
        if polygon.shape[0] != polygon.shape[1]:
            # Normal case, more than 3 pts to test
            insert_matrix = np.ones((polygon.shape[0], polygon.shape[0] - polygon.shape[1]))
            test_matrix = np.concatenate((polygon, insert_matrix), axis=1)
        else:
            # Special case ,3 pts to test
            test_matrix = polygon
        # np.det == 0 for planar, others for non-planar
        if np.linalg.det(test_matrix) == 0:
            return True
        else:
            return False

    @staticmethod
    def normal_direction(polygon: np.ndarray):
        """
        Compute the normal direction of a polygon
        :param polygon: Ndarray represent polygon
        :return: a vector with 3 elements
        """
        # Compute the center point
        center_pt = np.sum(polygon, axis=0) / polygon.shape[0]

        # Two vector of the polygon
        vec_1 = polygon[2, :] - polygon[1, :]
        vec_2 = polygon[0, :] - polygon[1, :]
        direc = np.cross(vec_1, vec_2)

        # Normalize the vector
        dist = np.linalg.norm(direc)
        normal_direc = direc / dist

        # Intercept of plane
        # A plane can be interpreted as following equation: Ax+By+Cz+D = 0
        # Where we can say that [A,B,C] is the normal direction coefficient
        intercept_d = -np.dot(polygon[0], normal_direc)

        return normal_direc, center_pt, intercept_d

    @staticmethod
    def random_orthogonal(direc_vec: np.ndarray):
        """
        Random generate a vector orthogonal with the direc_vec
        :param direc_vec: A ndarray represent direction vector
        :return: A ndarray represent orthogonal vector
        """
        orthogonal_vector = np.random.random(3)
        # Prevent the case of direc vec has 0 inside

        if direc_vec[2] != 0:
            orthogonal_vector[2] = (0 - (
                    orthogonal_vector[0] * direc_vec[0] + orthogonal_vector[1] * direc_vec[1])) / direc_vec[2]
        elif direc_vec[1] != 0:
            orthogonal_vector[1] = (0 - (
                    orthogonal_vector[0] * direc_vec[0] + orthogonal_vector[2] * direc_vec[2])) / direc_vec[1]
        else:
            orthogonal_vector[0] = (0 - (
                    orthogonal_vector[1] * direc_vec[1] + orthogonal_vector[2] * direc_vec[2])) / direc_vec[0]

        # Normalized Vec
        orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

        return orthogonal_vector

    @staticmethod
    def reflect(normal_direc: np.ndarray, insert_ray: np.ndarray):
        """
        Compute the reflection direction of the insert ray, Normalized vec
        :param normal_direc: Normal direction of the plane
        :param insert_ray: insert ray direction of the ray
        :return: The reflection direction
        """
        assert normal_direc.shape == insert_ray.shape
        # The reflection vec can represent as r = d-2(d·n)n
        reflect_direc = insert_ray - (2 * np.dot(insert_ray, normal_direc) * normal_direc)
        # Normalize the vec
        reflect_direc = reflect_direc / np.linalg.norm(reflect_direc)

        return reflect_direc

    @staticmethod
    def ray_sphere_inter(ray_start: np.ndarray, ray_direction: np.ndarray, sphere_cen: np.ndarray, radius: float):
        """
        Check the intersection within the radius oor not
        :param ray_start: The start location of ray
        :param ray_direction: The direction of the ray
        :param sphere_cen: The sphere center point location
        :param radius: The radius of the sphere
        :return: None for no intersection, inter point for have intersection
        """
        assert ray_start.shape == ray_direction.shape == sphere_cen.shape == (3,)

        # vec_start is for the vector of ray start point to sphere center
        vec_start = sphere_cen - ray_start
        # proj_dist is for project distance
        proj_dist = np.dot(vec_start, ray_direction)
        # inter_pt is for orthogonal intersection point location coordinate
        inter_pt = proj_dist * ray_direction + ray_start
        # dist is for the sphere center to the inter_pt
        dist = np.linalg.norm(inter_pt - sphere_cen)

        # If the dist is larger than radius, or need to back propagate, then no intersection
        if dist > radius or proj_dist < 0:
            inter_pt = None

        return inter_pt

    @staticmethod
    def lambert_diffuse(rec_loca: np.ndarray, rec_radius: float, normal_direc: np.ndarray, insert_loca: np.ndarray):
        """
        Get the cosine of normal direction and diffuse angle and solid angel
        :param rec_loca: ndarray represent receiver location
        :param rec_radius: the receiver location
        :param normal_direc: normal direction of a polygon
        :param insert_loca: ndarray represent diffusion start point
        :return: cos_theta for normal direction and diffuse angle
                cos_gamma for solid angel
        """
        # Check legal input
        assert rec_loca.shape == normal_direc.shape == insert_loca.shape == (3,)
        # diffuse_ray is for diffuse vector
        diffuse_ray = rec_loca - insert_loca
        # diffuse_dist is for the distance between receiver and insert point
        diffuse_dist = np.linalg.norm(diffuse_ray)
        # diffuse_ray_norm Normalized the vector of diffuse ray
        diffuse_ray_norm = diffuse_ray / diffuse_dist
        # cos_theta for normal direction and diffuse angle
        cos_theta = np.dot(normal_direc, diffuse_ray_norm)
        # cos_gamma for solid angel
        cos_gamma = rec_radius / diffuse_dist

        return diffuse_ray_norm, diffuse_dist, cos_theta, cos_gamma

    @staticmethod
    def mirror_pt(pt: np.ndarray, normal_direc: np.ndarray, intercept_d: float):
        """
        Find the mirror point of a plane
        :param pt: The point to be mirrored, (3, ) ndarray represent point
        :param normal_direc: The normal direction of specific plane, (3, ) ndarray represent point
        :param intercept_d: ax+by+cz+d=0, intercept_d is the "d" in the equation
        :return: mirror_pt, (3, ) ndarray represent point
        """
        assert pt.shape == normal_direc.shape == (3,)
        # Compute the scaler
        scale = (-intercept_d - np.dot(pt, normal_direc)) / np.dot(normal_direc, normal_direc)
        # Compute the mirror point
        mir_pt = 2 * scale * normal_direc + pt

        return mir_pt

    @staticmethod
    def diff_side(pt_1: np.ndarray, pt_2: np.ndarray, normal_direc: np.ndarray, plane_pt: np.ndarray,
                  intercept_d: float):
        """
        This function is to check the whether two side is on the same side of a plane or not
        :param pt_1: Point 1 to be checked
        :param pt_2: Point 2 to be checked
        :param normal_direc: The normal direction of a plane
        :param plane_pt: A point on the plane
        :param intercept_d: Ax+By+Cz+d = 0, the intercept_d is the d in equation
        :return: Return a ndarray point for different sides and None for the same side
        """
        assert pt_1.shape == pt_2.shape == normal_direc.shape == plane_pt.shape == (3,)

        # Compare value for the pt_1 and pt_2
        compare_1 = np.dot(normal_direc, (plane_pt - pt_1))
        compare_2 = np.dot(normal_direc, (plane_pt - pt_2))
        # Multiply two value
        compare = compare_1 * compare_2

        # Negative for different side, Positive for same side
        if compare < 0:
            # TODO Bug is here, check the inter_pt is inside or outside
            scale = (-intercept_d - np.dot(pt_1, normal_direc)) / np.dot((pt_2 - pt_1), normal_direc)
            inter_pt = scale * (pt_2 - pt_1) + pt_1

            return inter_pt
        else:
            return None

    @staticmethod
    def pt_in_polygon(pt: np.ndarray, polygon: np.ndarray, normal_direc: np.ndarray, intercept_d: float):
        """
        Check the point is in the 3D polygon or not
        :param pt: A (3,) ndarray represent point
        :param polygon: A multi points represent polygon (3,n) dimension
        :param normal_direc: The normal direction of the plane
        :param intercept_d: ax+by+cz+d=0, intercept_d
        :return: True for in polygon, False for not in polygon
        """
        # Check the point is on the plane
        assert pt.shape == normal_direc.shape == (3,)
        inside_flag = False
        # Check the point is on the same plane, if in the same plane may have
        if np.abs(np.dot(pt, normal_direc) + intercept_d) < 1e-6:

            # Generate Random orthogonal
            orthogonal_vector = np.random.random(3)
            # Prevent the case of direc vec has 0 inside
            if normal_direc[2] != 0:
                orthogonal_vector[2] = (0 - (
                        orthogonal_vector[0] * normal_direc[0] + orthogonal_vector[1] * normal_direc[1])) / \
                                       normal_direc[2]
            elif normal_direc[1] != 0:
                orthogonal_vector[1] = (0 - (
                        orthogonal_vector[0] * normal_direc[0] + orthogonal_vector[2] * normal_direc[2])) / \
                                       normal_direc[1]
            else:
                orthogonal_vector[0] = (0 - (
                        orthogonal_vector[1] * normal_direc[1] + orthogonal_vector[2] * normal_direc[2])) / \
                                       normal_direc[0]
            # Normalized Vec
            orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

            # Point is not on the corner of the polygon
            # Side direction
            rever_polygon = np.vstack([polygon[-1, :], polygon[0:-1, :]])
            side_vec = rever_polygon - polygon

            # Check intersection of each side
            onside_time = 0  # onside_time ray start point is on side time
            cross_time = 0  # cross_time is ray intersect with the side time

            for i in range(side_vec.shape[0]):
                tmp_pl_start = polygon[i]
                tmp_pl_direc = side_vec[i]
                # Generate a random orthogonal vector
                # Check Line Ray intersection
                # coe_1 is for polygon start to intersect
                insert_vec = np.array([1, 1, 1])

                coe_1_numerator = np.linalg.det(np.vstack([insert_vec, (pt - tmp_pl_start), orthogonal_vector]))
                coe_2_numerator = np.linalg.det(
                    np.vstack([insert_vec, (tmp_pl_start - pt), tmp_pl_direc]))

                coe_1_denominator = np.linalg.det(np.vstack([insert_vec, tmp_pl_direc, orthogonal_vector]))
                coe_2_denominator = np.linalg.det(np.vstack([insert_vec, orthogonal_vector, tmp_pl_direc]))

                # coe_1 is for polygon start to intersect
                coe_1 = coe_1_numerator / coe_1_denominator
                # coe_2 is for ray start to intersect
                coe_2 = coe_2_numerator / coe_2_denominator

                # Start point is not on the polygon side, and intersect is with in edge of polygon
                if 0 < coe_1 < 1 and coe_2 > 0:
                    cross_time += 1

                # Start point is on the polygon side, and intersect is with in edge of polygon
                elif 0 < coe_1 < 1 and coe_2 == 0:
                    onside_time += 1

            # Check cross time to verify inside or not
            # print(f"onside_time: {onside_time}, cross_time: {cross_time}")
            state_time = onside_time * 2 + cross_time
            if state_time % 2 == 1:
                inside_flag = True

        return inside_flag

    @staticmethod
    def area(polygon: np.ndarray, cen_pt: np.ndarray):
        """
        Compute the area of a 3d polygon
        :param polygon: the point represent polygon
        :param cen_pt: the center point of the polygon
        :return:
        """
        if (polygon[0, :] != polygon[-1, :]).any():
            polygon = np.vstack((polygon, np.array([polygon[0]])))

        area = 0
        for i in range(polygon.shape[0] - 1):
            vec_1 = polygon[i] - cen_pt
            vec_2 = polygon[i + 1] - cen_pt
            tmp_tri = 0.5 * np.linalg.norm(np.cross(vec_1, vec_2))
            area += tmp_tri
        area = np.abs(area)
        return area

    def cross_polygon(self, pt_1: np.ndarray, pt_2: np.ndarray, plane_normal: np.ndarray, plane_polygon: np.ndarray):
        """
        Check the finite line construct with pt_1 and pt_2 cross a plane or not.
        This function use in image source method to check image source and receiver line cross a plane or not
        :param pt_1: Point 1 to be checked
        :param pt_2: Point 2 to be checked
        :param plane_normal: The normal direction of a plane
        :param plane_polygon: The boundary point list
        :return: False for not cross the polygon, and True for cross polygon
        """
        assert pt_1.shape == pt_2.shape == plane_normal.shape == (3,)
        # This ray direction is none unit vector
        ray_direction = pt_2 - pt_1

        # STEP ONE
        # Check the intersection with infinite plane
        # ray_scaler is for the t in eq: p = s+tl
        # This is prevented divide by zero

        denominator = np.dot(ray_direction, plane_normal)
        tolerance = 1e-6
        state_cross = False

        if np.abs(denominator) >= tolerance:
            # Divide by zero, means the ray and direction is parallel, keep inter_pt as None
            ray_scaler = np.dot((plane_polygon[0] - pt_1), plane_normal) / denominator
            # The inter_pt is on the infinite plane
            inter_pt = pt_1 + ray_direction * ray_scaler

            # TODO Buffer Can be Modified
            if 0 < ray_scaler < 1:
                # The ray Propagate along the positive side
                # STEP TWO
                # Check the inter_pt is in the plane
                # Check the point is in polygon
                if not any(np.equal(plane_polygon, inter_pt).all(1)):
                    # Point is not on the corner of the polygon
                    # Side direction
                    rever_polygon = np.vstack([plane_polygon[-1, :], plane_polygon[0:-1, :]])
                    side_vec = rever_polygon - plane_polygon

                    # Check intersection of each side
                    onside_time = 0  # onside_time ray start point is on side time
                    cross_time = 0  # cross_time is ray intersect with the side time
                    check_vec = self.random_orthogonal(plane_normal)

                    for i in range(side_vec.shape[0]):
                        tmp_pl_start = plane_polygon[i]
                        tmp_pl_direc = side_vec[i]
                        # Generate a random orthogonal vector
                        # Check Line Ray intersection
                        # coe_1 is for polygon start to intersect
                        insert_vec = np.array([1, 1, 1])

                        coe_1_numerator = np.linalg.det(np.vstack([insert_vec, (inter_pt - tmp_pl_start), check_vec]))
                        coe_2_numerator = np.linalg.det(
                            np.vstack([insert_vec, (tmp_pl_start - inter_pt), tmp_pl_direc]))

                        coe_1_denominator = np.linalg.det(np.vstack([insert_vec, tmp_pl_direc, check_vec]))
                        coe_2_denominator = np.linalg.det(np.vstack([insert_vec, check_vec, tmp_pl_direc]))

                        # coe_1 is for polygon start to intersect
                        coe_1 = coe_1_numerator / coe_1_denominator
                        # coe_2 is for ray start to intersect
                        coe_2 = coe_2_numerator / coe_2_denominator

                        # Start point is not on the polygon side, and intersect is with in edge of polygon
                        if 0 < coe_1 < 1 and coe_2 > 0:
                            cross_time += 1

                        # Start point is on the polygon side, and intersect is with in edge of polygon
                        elif 0 < coe_1 < 1 and coe_2 == 0:
                            onside_time += 1

                    # Check cross time to verify inside or not
                    # print(f"onside_time: {onside_time}, cross_time: {cross_time}")
                    state_time = onside_time * 2 + cross_time
                    if state_time % 2 == 1:
                        state_cross = True

        return state_cross

    def ray_plane_inter(self, ray_start: np.ndarray, ray_direction: np.ndarray,
                        plane_polygon: np.ndarray, plane_normal: np.ndarray):
        """
        Check a ray have intersection with polygon
        :param ray_start: The start point, (3,) ndarray
        :param ray_direction: The direction vector, (3,) ndarray
        :param plane_polygon: The ndarray represent polygon
        :param plane_normal: The normal direction of the plane
        :return: state: True for have intercept, False for no intercept
                 inter_pt: (3,) shape ndarray for have intersection, None for no inter
        """
        # Input assert
        assert ray_start.shape == ray_direction.shape == plane_normal.shape
        assert len(plane_polygon.shape) == 2

        # STEP ONE
        # Check the intersection with infinite plane
        # ray_scaler is for the t in eq: p = s+tl
        # This is prevented divide by zero

        denominator = np.dot(ray_direction, plane_normal)
        tolerance = 1e-6
        inter_pt = None

        if np.abs(denominator) >= tolerance:
            # Divide by zero, means the ray and direction is parallel, keep inter_pt as None
            ray_scaler = np.dot((plane_polygon[0] - ray_start), plane_normal) / denominator
            # print(f"ray_scaler: {ray_scaler}")
            # The inter_pt is on the infinite plane
            inter_pt = ray_start + ray_direction * ray_scaler
            # print(f"Intersection Point: {inter_pt}")
            # TODO Buffer Can be Modify
            if ray_scaler > 1e-6:
                # The ray Propagate along the positive side
                # STEP TWO
                # Check the inter_pt is in the plane
                # Check the point is in polygon
                if not any(np.equal(plane_polygon, inter_pt).all(1)):
                    # Point is not on the corner of the polygon
                    # Side direction
                    rever_polygon = np.vstack([plane_polygon[-1, :], plane_polygon[0:-1, :]])
                    # print(f"Rever_Polygon: {rever_polygon}")
                    side_vec = rever_polygon - plane_polygon

                    # print(f"Side Vector is {side_vec}")
                    # Check intersection of each side
                    onside_time = 0  # onside_time ray start point is on side time
                    cross_time = 0  # cross_time is ray intersect with the side time
                    check_vec = self.random_orthogonal(plane_normal)
                    # print(f"Check Vector: {check_vec}")

                    for i in range(side_vec.shape[0]):
                        tmp_pl_start = plane_polygon[i]
                        tmp_pl_direc = side_vec[i]
                        # Generate a random orthogonal vector
                        # Check Line Ray intersection
                        # coe_1 is for polygon start to intersect
                        insert_vec = np.array([1, 1, 1])

                        coe_1_numerator = np.linalg.det(np.vstack([insert_vec, (inter_pt - tmp_pl_start), check_vec]))
                        coe_2_numerator = np.linalg.det(
                            np.vstack([insert_vec, (tmp_pl_start - inter_pt), tmp_pl_direc]))
                        # demoniator = np.linalg.det(np.vstack([insert_vec, tmp_pl_direc, check_vec]))
                        coe_1_denominator = np.linalg.det(np.vstack([insert_vec, tmp_pl_direc, check_vec]))
                        coe_2_denominator = np.linalg.det(np.vstack([insert_vec, check_vec, tmp_pl_direc]))

                        # coe_1 is for polygon start to intersect
                        coe_1 = coe_1_numerator / coe_1_denominator
                        # coe_2 is for ray start to intersect
                        coe_2 = coe_2_numerator / coe_2_denominator
                        # print(f"coe_1: {coe_1}, coe_2: {coe_2}")
                        # print(f"Ray Line intersection: {tmp_pl_start+coe_1*tmp_pl_direc}")

                        # Start point is not on the polygon side, and intersect is with in edge of polygon
                        if 0 < coe_1 < 1 and coe_2 > 0:
                            cross_time += 1

                        # Start point is on the polygon side, and intersect is with in edge of polygon
                        elif 0 < coe_1 < 1 and coe_2 == 0:
                            onside_time += 1

                    # Check cross time to verify inside or not
                    # print(f"onside_time: {onside_time}, cross_time: {cross_time}")
                    state_time = onside_time * 2 + cross_time
                    if state_time % 2 == 0:
                        inter_pt = None

                else:
                    # The ray start is on the corner of the polygon
                    inter_pt = ray_start

            else:
                inter_pt = None

        return inter_pt


if __name__ == "__main__":
    from FileIO.ReadRhino import Read3dm
    from RoomGeometry.FieldGeometry import FieldGeometry
    from RoomGeometry.Plane import Plane
    from CONFIG import absorptionDict, scatterDict, fileDir

    # Read the rhino file, and convert it
    rh_model = Read3dm(fileDir)
    # Get the rh_plane, receiver and source
    rh_plane, receiver, source = rh_model.convert_file()

    start_pt = np.array([0, 0, 0])
    ray_direc = np.array([1, 0, 0])
    sphere_cen = np.array([-5.1, 0.2, 0])
    inter_pt = BaseGeometry.ray_sphere_inter(start_pt, ray_direc, sphere_cen, 0.1)
    print(inter_pt)

    a = BaseGeometry.mirror_pt(np.array([1, 3, 4]), np.array([2, -1, 1]), 3)
    print(a)

    # Two side or not
    print(BaseGeometry.diff_side())
