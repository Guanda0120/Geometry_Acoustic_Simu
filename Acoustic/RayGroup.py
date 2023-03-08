import numpy as np
from typing import List
from Acoustic.AcousticRay import AcousticRay
from Acoustic.BaseAcoustic import BaseAcoustic
from Acoustic.CONST import acoustic_speed
from Graphic.BaseGeometry import BaseGeometry
from RoomGeometry.FieldGeometry import FieldGeometry
import multiprocessing


class RayGroup:
    """
    The RayGroup is to contain the A amount of AcousticRay
    """

    def __init__(self, num_ray: int, init_swl: float, init_location: np.ndarray, sound_field: FieldGeometry):
        """
        Init a group of acoustic ray.
        :param num_ray: The amount of the acoustic ray.
        :param init_swl: The init sound power level of acoustic ray
        :param init_location: The init location of the ray
        """
        self.num_ray = num_ray
        self.init_swl = init_swl
        self.init_location = init_location
        # Assign the sound field to the ray group
        self.sound_field = sound_field
        # Restrict the object type of container
        self.ray_container: List[AcousticRay] = []

        # Uniform distribution of a sphere
        # TODO
        theta = np.random.random(self.num_ray) * np.pi * 2
        phi = np.arccos(1 - 2 * np.random.random(self.num_ray))
        x = np.multiply(np.sin(phi), np.cos(theta))
        y = np.multiply(np.sin(phi), np.sin(theta))
        z = np.cos(phi)
        self.ray_direc = np.hstack((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]))

        # Loop over and init the ray
        for i in range(self.num_ray):
            self.ray_container.append(AcousticRay(self.init_swl, self.init_location, self.ray_direc[i, :]))

    def __str__(self):
        return f"The group has {len(self.ray_container)} rays, init Power Level is {self.init_swl}"

    def reflect_all(self):
        """
        Check every acoustic ray intersect with which plane, renew the acoustic ray info
        :return:
        """
        tmp_spl = []
        for ray in self.ray_container:

            # Record the inter plane index
            tmp_plane_idx = []
            # Record the inter point
            tmp_interpt = []

            for plane_idx in range(len(self.sound_field.plane)):
                # plane is current plane
                plane = self.sound_field.plane[plane_idx]
                # Check intersection for each plane
                pt = BaseGeometry().ray_plane_inter(ray.start_point[-1], ray.ray_direc[-1],
                                                    plane.polygon, plane.normal_direc)
                # If pt is not None, record it
                if pt is not None:
                    tmp_plane_idx.append(plane_idx)
                    tmp_interpt.append(pt)

            # For the current ray, select the minime dist point
            if len(tmp_plane_idx) > 1:
                num_pt = len(tmp_interpt)
                tmp_start_matrix = np.tile(ray.start_point[-1], (num_pt, 1))
                dist = np.linalg.norm(tmp_start_matrix - np.array(tmp_interpt), axis=1)
                need_idx = np.argmin(dist)
                tmp_plane_idx = [tmp_plane_idx[need_idx]]
                tmp_interpt = [tmp_interpt[need_idx]]

            # Compute the reflection direction of the current ray
            # TODO BUG is here
            tmp_normal_direc = self.sound_field.plane[tmp_plane_idx[0]].normal_direc
            tmp_reflex = BaseGeometry.reflect(tmp_normal_direc, ray.ray_direc[-1])

            # insert ray cos theta
            tmp_cos = np.abs(np.dot(tmp_normal_direc, ray.ray_direc[-1]))
            # Renew the ray info
            # Record insert point
            ray.start_point.append(tmp_interpt[0])
            # Record insert plane_idx
            ray.reflect_plane_idx.append(tmp_plane_idx[0])
            # Record the reflex ray
            ray.ray_direc.append(tmp_reflex)
            # Record the absorption coe and scatter coe
            ray.ref_coe.append(self.sound_field.plane[tmp_plane_idx[0]].material.ref_0)
            ray.sca_coe.append(self.sound_field.plane[tmp_plane_idx[0]].material.scatter)
            # Record dist
            ray.dist.append(np.linalg.norm(ray.start_point[-1] - ray.start_point[-2]))
            # Record time
            ray.time.append(np.sum(ray.dist) / acoustic_speed)
            # Record cos
            ray.insert_cos.append(tmp_cos)
            # Record the end sound pressure level
            end_pressure = BaseAcoustic.source2pressure(ray.init_power, np.sum(ray.dist),
                                                        np.array(ray.ref_coe), np.array(ray.sca_coe), ray.insert_cos)
            end_pressure = BaseAcoustic.air_atten(np.sum(ray.dist), end_pressure)
            end_spl = BaseAcoustic.pressure2spl(end_pressure)
            tmp_spl.append(end_spl)

        # Select the max of the spl, then return
        max_spl = np.max(tmp_spl)
        return max_spl


if __name__ == "__main__":
    g = RayGroup(20000, 94, np.array([1, 1, 1]))
    print(g.ray_container)
