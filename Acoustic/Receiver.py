import os
import numpy as np
from typing import List
import copy
from Acoustic.AcousticRay import AcousticRay
from Acoustic.ImageSourceRay import ImageSourceRay
from Acoustic.RayGroup import RayGroup
from Acoustic.BaseAcoustic import BaseAcoustic
from Acoustic.Source import Source
from Acoustic.CONST import acoustic_speed
from RoomGeometry.FieldGeometry import FieldGeometry
from Graphic.BaseGeometry import BaseGeometry
from AdaptiveDSP.RT_Process import RT_Process


class Receiver:

    # A single receiver object

    def __init__(self, rec_location: np.ndarray, src: Source, sound_field: FieldGeometry, rec_radius: float = 0.1):
        """
        Init the sphere receiver
        :param rec_location: A ndarray represent 3D point coordinate
        :param src: A Source object
        :param sound_field: A FieldGeometry object
        :param rec_radius: float represent sphere radius, Default radius is 0.1m
        """
        assert rec_location.shape == (3,)
        assert rec_radius > 0

        self.rec_location = rec_location
        self.src = src
        self.sound_field = sound_field
        self.plane_idx = list(range(len(sound_field.plane)))
        self.radius = rec_radius

        # None reflect ray insert spl
        self.direc_pressure = []
        # None reflect ray insert time
        self.direc_time = None

        # im_ray list records the image source acoustic ray insert into the sphere receiver
        self.im_ray: List[ImageSourceRay] = []
        # im_pressure records the end pressure insert into the sphere
        self.im_pressure = []
        # im_time records the insert time
        self.im_time: List[float] = []
        # im_order_list, all the exist order list, which list contain plane index
        self.im_order_list = []

        # stochastic_ray list records the stochastic acoustic ray insert into the sphere receiver
        self.stochastic_ray: List[AcousticRay] = []
        # stochastic_pressure records the end pressure insert into the sphere
        self.stochastic_pressure = []
        # stochastic_time records the insert time
        self.stochastic_time: List[float] = []

        # diffuse_ray_idx is ray index of the ray_group
        self.diffuse_ray_idx: List[int] = []
        # diffuse_ray_order records after how much reflection can diffuse ray intersect with the sphere
        self.diffuse_ray_order: List[int] = []
        # diffuse_ray_time records the insert time
        self.diffuse_ray_time: List[float] = []
        # diffuse_ray_pressure record the pressure info
        self.diffuse_ray_pressure = []

        # all_pressure records the all the pressure from image source, stochastic method and diffuse rain
        self.all_pressure = []
        # all_time records the all the time from image source, stochastic method and diffuse
        self.all_time = []
        # all_index records the index, 0 for direc sound, 1 for image source, 2 for stochastic ray, 3 for diffuse rain
        self.all_index = []

        # fs_pressure is the sound pressure match the fs time
        self.fs_pressure = []
        # fs_time is the time match the fs time
        self.fs_time = []

        # Smooth DATA Linear Regression
        self.smooth_response_lr = None
        # RT Time Linear Regression
        self.rt_time_lr = None
        self.smooth_time_lr = None

        # Smooth DATA Back Integral
        self.smooth_response_bi = None
        self.smooth_time_bi = None
        # RT Time Back Integral
        self.rt_time_bi = None

    def direc_insert(self):
        """
        Renew self.direc_spl and self.direc_time
        :return: Renew self.direc_spl and self.direc_time
        """
        cross_state = False

        for tmp_polygon in self.sound_field.plane:
            tmp_state = BaseGeometry().cross_polygon(self.rec_location, self.src.loc, tmp_polygon.normal_direc, tmp_polygon.polygon)
            if tmp_state:
                cross_state = True
                break

        if not cross_state:
            # Get the distance between rec and src
            dist = np.linalg.norm(self.rec_location - self.src.loc)

            direc_pressure = BaseAcoustic.direct_pressure(self.src.init_power, dist)
            direc_pressure = BaseAcoustic.air_atten(dist, direc_pressure)
            self.direc_pressure.append(direc_pressure)
            self.direc_time = dist / acoustic_speed

    def image_source(self, max_order: int):
        """
        :param: max_order: The max order of image source ray
        :return: renew self.im_ray, self.im_pressure, self.time list
        """
        total_order_list = []
        # pre_order_list = []
        # pre_source_pt = []

        # tmp_order_list cache the list of exist reflex order
        tmp_order_list = [[]]
        # tmp_source_list cache the list of
        tmp_source_list = np.array([self.src.loc])
        tmp_order = 1

        while tmp_order <= max_order:
            # Iter and make tmp_order get to the tmp_order
            # Store this order list
            cache_order_list = []
            # Store image source point
            cache_mir_pt = []
            for pre_idx in range(len(tmp_order_list)):
                # pre_order is the plane index list
                pre_order = tmp_order_list[pre_idx]
                # pre_source is the source point about plane
                pre_source = tmp_source_list[pre_idx]
                # Debug
                # print(f"pre_order: {pre_order}")
                # print(f"pre_source: {pre_source}")
                # Loop over all the
                for idx in self.plane_idx:
                    # Check this index should append after this list
                    # The plane to added
                    added_plane = self.sound_field.plane[idx]
                    # print(f"added_plane: {added_plane}")
                    # The order is lager than 1
                    if len(pre_order) > 0 and idx != pre_order[-1]:
                        # Mirror the pre source
                        mir_pt = BaseGeometry.mirror_pt(pre_source, added_plane.normal_direc, added_plane.intercept_d)
                        # Then check different side or not
                        im_pt = BaseGeometry.diff_side(mir_pt, self.rec_location, added_plane.normal_direc,
                                                       added_plane.polygon[0], added_plane.intercept_d)
                        # print(f"im_pt: {im_pt}")
                        if im_pt is not None:
                            # Exist index, record it
                            # Seperate a new memory space
                            tmp_pre = copy.deepcopy(pre_order)
                            # append it
                            tmp_pre.append(idx)
                            # print(f"ImageSourceModule.py, tmp_pre: {tmp_pre}")
                            # Store order list into cache_order_list
                            cache_order_list.append(tmp_pre)
                            # Store image source point into cache_mir_pt
                            # TODO 0108 modify append mir point not im point
                            # cache_mir_pt.append(im_pt)
                            # Above is pre code
                            cache_mir_pt.append(mir_pt)
                            ray = ImageSourceRay(self.src, self.rec_location, self.sound_field)
                            ray.n_order_im(tmp_pre)
                            if ray.exist_state:
                                # Renew the self.im_ray, im_pressure, im_time
                                self.im_ray.append(ray)
                                self.im_pressure.append(ray.rec_pressure)
                                self.im_time.append(ray.rec_time)
                                self.im_order_list.append(tmp_pre)

                    # The order is 1
                    elif len(pre_order) == 0:
                        # Mirror the pre source
                        mir_pt = BaseGeometry.mirror_pt(pre_source, added_plane.normal_direc, added_plane.intercept_d)
                        # Then check different side or not
                        im_pt = BaseGeometry.diff_side(mir_pt, self.rec_location, added_plane.normal_direc,
                                                       added_plane.polygon[0], added_plane.intercept_d)
                        if im_pt is not None:
                            # Exist index, record it
                            # Seperate a new memory space
                            tmp_pre = copy.deepcopy(pre_order)
                            # append it
                            tmp_pre.append(idx)

                            # Store order list into cache_order_list
                            cache_order_list.append(tmp_pre)
                            # Store image source point into cache_mir_pt
                            # TODO 0108 modify append mir point not im point
                            # cache_mir_pt.append(im_pt)
                            # Above is pre code
                            cache_mir_pt.append(mir_pt)
                            ray = ImageSourceRay(self.src, self.rec_location, self.sound_field)
                            ray.n_order_im(tmp_pre)
                            if ray.exist_state:
                                # Renew the self.im_ray, im_pressure, im_time
                                self.im_ray.append(ray)
                                self.im_pressure.append(ray.rec_pressure)
                                self.im_time.append(ray.rec_time)
                                self.im_order_list.append(tmp_pre)

            # Debuger
            # print(f"In order {tmp_order}, the cache order list is {cache_order_list}")
            # print(f"In order {tmp_order}, the image ray num is {len(total_ray)}")
            # Store the Result
            total_order_list.extend(cache_order_list)
            # Renew var for the next loop iter
            tmp_order_list = cache_order_list
            tmp_source_list = cache_mir_pt
            tmp_order += 1

    def image_diffuse(self):
        """
        Use this method after image source method to predict image source path diffusion
        :return: Renew the diffusion list
        """
        # TODO Something here is wrong
        # Loop over all the ray
        for ray in self.im_ray:
            # Check whether the image ray have diffuse
            if len(ray.plane_idx) > 1 and ray.exist_state:
                # Loop Over all the index and except the last one
                for idx in range(len(ray.plane_idx) - 1):
                    # Get the plane object
                    tmp_plane = self.sound_field.plane[ray.plane_idx[idx]]
                    # Get the temp start point
                    tmp_start = ray.ref_pt[idx]
                    # Compute the cos of diffuse vec
                    diffuse_ray_norm, diffuse_dist, cos_theta, cos_gamma = \
                        BaseGeometry.lambert_diffuse(self.rec_location, self.radius, tmp_plane.normal_direc,
                                                     ray.ref_pt[idx])
                    # Check whether the cos could diffuse
                    if cos_theta > 0:
                        inter_dist_list = []
                        for polygon in self.sound_field.plane:
                            # Get the intersection point
                            tmp_inter_pt = BaseGeometry().ray_plane_inter(tmp_start, diffuse_ray_norm,
                                                                          polygon.polygon, polygon.normal_direc)
                            if tmp_inter_pt is not None:
                                # If the inter point is not None, record the distance
                                inter_dist_list.append(np.linalg.norm(tmp_inter_pt - tmp_start))

                        if inter_dist_list != []:
                            if np.min(inter_dist_list) > diffuse_dist:
                                # Intersect with receiver first
                                # Record the info
                                self.diffuse_ray_idx.append(-1)
                                self.diffuse_ray_order.append(idx + 1)
                                # Compute the total distance of the ray propagate
                                pt_matrix = np.vstack((np.asarray([self.src.loc]), ray.ref_pt[:(idx + 1), :]))
                                sta_pt = pt_matrix[0:-1, :]
                                end_pt = pt_matrix[1:, :]
                                total_dist = np.sum(np.linalg.norm((end_pt - sta_pt), axis=1)) + diffuse_dist

                                # Get the right abs_list and sca_list
                                if idx != 0:
                                    tmp_ref = np.array(ray.ref_coe_list[:(idx + 1), :])
                                else:
                                    tmp_ref = np.ones((6, 1)).T
                                tmp_sca = np.array(ray.sca_coe_list[:(idx + 1), :])
                                tmp_cos = ray.insert_cos[:(idx + 1)]

                                # Compute the non_attenuate sound pressure
                                tmp_pressure = BaseAcoustic.source_diffuse_pressure(self.src.init_power, total_dist,
                                                                                    tmp_ref, tmp_sca, tmp_cos,
                                                                                    cos_theta, cos_gamma)
                                # Compute the air absorption
                                atten_pressure = BaseAcoustic.air_atten(total_dist, tmp_pressure)
                                # Record the pressure and time
                                self.diffuse_ray_pressure.append(atten_pressure)
                                self.diffuse_ray_time.append(total_dist / acoustic_speed)

    def stochastic_insert(self, ray_group: RayGroup):
        """
        Check the ray sphere intersection, if have inter_section record it, if None, then pass
        Use this function before RayGroup.reflect_all()
        :param ray_group: RayGroup object
        :return: Nothing, just renew the
        """

        for ray in ray_group.ray_container:
            # Check have intersection with sphere or not
            inter_pt = BaseGeometry.ray_sphere_inter(ray.start_point[-1], ray.ray_direc[-1],
                                                     self.rec_location, self.radius)
            # If the inter_pt is a ndarray
            if inter_pt is not None:
                # print(f"Intersection Point is {inter_pt}")
                # For safety, deepcopy the ray object.
                self.stochastic_ray.append(copy.deepcopy(ray))
                # dist is for the last start point to the
                dist = np.linalg.norm(ray.start_point[-1] - inter_pt)
                # Sum over the all the distance
                total_dist = np.sum(ray.dist) + dist
                # If is not direct, then record it
                if len(ray.ref_coe) != 0 and len(ray.sca_coe) != 0:
                    total_ref = np.array(ray.ref_coe)
                    total_sca = np.array(ray.sca_coe)
                else:
                    # If is direct, then get the absorption and scatter all zero.
                    total_ref = np.ones((6, 1)).T
                    total_sca = np.zeros((6, 1)).T

                # Compute total time when ray insert in to the rec
                time = total_dist / acoustic_speed

                # Compute the pressure as the sphere wave, and attenuate pressure.
                pressure = BaseAcoustic.source2pressure(self.src.init_power, total_dist, total_ref, total_sca,
                                                        ray.insert_cos)
                atten_pressure = BaseAcoustic.air_atten(total_dist, pressure)

                # Record into the rec obj
                self.stochastic_ray.append(ray)
                self.stochastic_time.append(time)
                self.stochastic_pressure.append(atten_pressure)

    def lambert_diffuse(self, ray_group: RayGroup):
        """
        This function is used to compute the diffuse sound pressure
        :param ray_group: Ray group object contain ray
        :return: Renew the ray info
        """
        diffuse_num = 0
        for ray_idx in range(len(ray_group.ray_container)):
            ray = ray_group.ray_container[ray_idx]
            # Get the insert point and ray direction
            tmp_direc = ray.ray_direc[-1]
            tmp_start = ray.start_point[-1]
            # Get the cosine of theta and cosine of gamma
            diffuse_ray_norm, diffuse_dist, cos_theta, cos_gamma = \
                BaseGeometry.lambert_diffuse(self.rec_location, self.radius, tmp_direc, tmp_start)
            # Check whether bigger than 0
            if cos_theta > 0:
                # On the right half sphere
                # Check have intersection with other polygon before insert into sphere receiver
                # Loop over all the polygon and get intersection
                inter_dist_list = []
                for polygon in self.sound_field.plane:
                    # Get the intersection point
                    tmp_inter_pt = BaseGeometry().ray_plane_inter(tmp_start, diffuse_ray_norm,
                                                                  polygon.polygon, polygon.normal_direc)
                    if tmp_inter_pt is not None:
                        # If the inter point is not None, record the distance
                        inter_dist_list.append(np.linalg.norm(tmp_inter_pt - tmp_start))

                if np.min(inter_dist_list) > diffuse_dist:
                    # Intersect with receiver first
                    # Record the info
                    self.diffuse_ray_idx.append(ray_idx)
                    self.diffuse_ray_order.append(len(ray.ray_direc))
                    # Compute the total distance of the ray propagate
                    total_dist = np.sum(ray.dist) + diffuse_dist
                    # Compute the non_attenuate sound pressure
                    tmp_ref = np.array(ray.ref_coe)
                    tmp_sca = np.array(ray.sca_coe)
                    tmp_pressure = BaseAcoustic.source_diffuse_pressure(self.src.init_power, total_dist,
                                                                        tmp_ref, tmp_sca, ray.insert_cos,
                                                                        cos_theta, cos_gamma)
                    # Compute the air absorption
                    atten_pressure = BaseAcoustic.air_atten(total_dist, tmp_pressure)
                    # Record the pressure and time
                    self.diffuse_ray_pressure.append(atten_pressure)
                    self.diffuse_ray_time.append(total_dist / acoustic_speed)
                    diffuse_num += 1

        return diffuse_num

    def stochastic_2_image(self, max_order: int):
        """
        Get stochastic to image_source, A Huge Innovation
        :param max_order: Max image source order
        :return: Renew the image_source info
        """
        for sto_ray in self.stochastic_ray:
            # Loop Over all the stochastic ray
            print(f"Plane Index is {sto_ray.reflect_plane_idx}")
            if len(sto_ray.reflect_plane_idx) > max_order:
                print(f"This Branch")
                # Init an Image Source Ray
                tmp_im_ray = ImageSourceRay(self.src, self.rec_location, self.sound_field)
                # Check the index path is an exist path
                tmp_im_ray.n_order_im(sto_ray.reflect_plane_idx)
                if tmp_im_ray.exist_state:
                    # Do exist
                    print(f"Init and append")
                    self.im_ray.append(tmp_im_ray)
                    self.im_pressure.append(tmp_im_ray.rec_pressure)
                    self.im_time.append(tmp_im_ray.rec_time)
                    self.im_order_list.append(sto_ray.reflect_plane_idx)

    def merge_together(self):
        """
        Merge three list of sound pressure, time, index, sort it as time.
        Use after all the diffuse, stochastic and im finish
                            :return: Renew the all list
        """
        # Merge into one array
        all_pressure = []
        all_pressure.extend(self.im_pressure)
        all_pressure.extend(self.stochastic_pressure)
        all_pressure.extend(self.diffuse_ray_pressure)
        if self.direc_time is not None:
            all_pressure.extend(self.direc_pressure)
        all_pressure = np.array(all_pressure, dtype=complex)

        all_time = []
        all_time.extend(self.im_time)
        all_time.extend(self.stochastic_time)
        all_time.extend(self.diffuse_ray_time)
        if self.direc_time is not None:
            all_time.append(self.direc_time)
        all_time = np.array(all_time)

        all_index = []
        all_index.extend(np.ones(len(self.im_time)))
        all_index.extend(np.ones(len(self.stochastic_time)) * 2)
        all_index.extend(np.ones(len(self.diffuse_ray_time)) * 3)
        all_index.append(0)
        all_index = np.array(all_index)

        # Sort the time and get the index
        sorted_pressure = np.zeros(all_pressure.shape, dtype=np.complex128)
        sorted_pointer = np.zeros(all_index.shape)
        sorted_time = np.sort(all_time)
        index_time = np.argsort(np.argsort(all_time))

        # Sort the index and pressure
        for i in range(index_time.shape[0]):
            sorted_pressure[index_time[i], :] = all_pressure[i, :]
            sorted_pointer[index_time[i]] = all_index[i]

        self.all_pressure = np.array(sorted_pressure)
        self.all_time = np.array(sorted_time)
        self.all_index = all_index

    def match_fs(self, fs: int):
        """

        :param fs:
        :return:
        """
        # A Sample time duration
        delta_t = 1 / fs
        mod = np.mod(self.all_time, delta_t)
        time_index = np.floor_divide(self.all_time, delta_t)

        # If the mod is 0,
        time_index = np.where(mod != 0, time_index, time_index - 1)
        time_index = time_index.astype('int64')
        # Shift the sound pressure
        shift_t = delta_t - mod
        shift_pressure = BaseAcoustic.pressure_shift(np.array(self.all_pressure), shift_t)

        pressure_merge = np.zeros((int(max(time_index)) + 1, 6), dtype=complex)

        # Create a dictionary
        pressure_dict = {}
        for i in range(max(time_index) + 1):
            pressure_dict[str(i)] = []

        for t_idx in range(len(time_index)):
            pressure_dict[str(time_index[t_idx])].append(shift_pressure[t_idx])

        for key in pressure_dict.keys():
            tmp_array = np.array(pressure_dict[key])
            tmp_array = BaseAcoustic.pressure_addition(tmp_array, add_method='rms')
            pressure_merge[int(key)] = tmp_array

        self.fs_time = np.array(range(len(pressure_merge) + 1)) * delta_t
        pressure_merge = np.concatenate((np.zeros((1, 6), dtype=complex), pressure_merge))
        self.fs_pressure = pressure_merge

    def lr_process(self):
        """
        Use linear regression method to estimate the rt time
        :return: Renew the object
        """
        # To ndarray
        tmp_pressure = np.array(self.fs_pressure)
        tmp_time = np.array(self.fs_time)

        # Remove 0 item
        tmp_lr_time = []
        tmp_lr_res = []
        for idx in range(tmp_pressure.shape[0]):
            if np.abs(tmp_pressure[idx, 0]) != 0:
                tmp_lr_time.append(tmp_time[idx])
                tmp_lr_res.append(tmp_pressure[idx])
        tmp_lr_res = np.array(tmp_lr_res)
        tmp_lr_time = np.array(tmp_lr_time)

        # Use linear regression method to estimate RT
        lr_rt, lr_response, lr_time = RT_Process.linear_regression(tmp_lr_res, tmp_lr_time)
        # Contain Data
        self.smooth_response_lr = lr_response

        self.smooth_time_lr = lr_time
        self.rt_time_lr = lr_rt

    def back_integral_process(self, rt_method: str):
        """
        Use back integral method to estimate the rt time
        :return: Renew the object
        """
        # Check the right input rt_method
        assert rt_method in ["T20", "T30", "T60"]
        # Use linear regression method to estimate RT
        # bi_rt, bi_response, bi_time = RT_Process.bi_method(tmp_pressure, tmp_time, rt_method)
        bi_rt, bi_response, bi_time = RT_Process.back_integral(self.all_pressure, self.all_time, rt_method)
        # Contain Data
        self.smooth_response_bi = bi_response
        self.smooth_time_bi = bi_time
        self.rt_time_bi = bi_rt

    def to_json(self):
        rec_dict = {}
        for key in self.__dict__:
            if type(self.__dict__[key]) is np.ndarray:
                # This list is numpy ndarray, check the dtpye
                # Copy the array
                tmp_array = self.__dict__[key]
                if self.__dict__[key].dtype == np.complex128:
                    # Deconstruct the dictionary as real and image part
                    key_real = key + '_real'
                    key_imag = key + '_imag'
                    # Check the length of the shape
                    if len(tmp_array.shape) == 2:
                        # 2D Array
                        real_list = []
                        imag_list = []
                        for item in tmp_array:
                            real_list.append(np.real(item).tolist())
                            imag_list.append(np.imag(item).tolist())
                        rec_dict[key_real] = real_list
                        rec_dict[key_imag] = imag_list
                    else:
                        # 1D Array
                        # Change the real and image part to the list
                        rec_dict[key_real] = np.real(tmp_array).tolist()
                        rec_dict[key_imag] = np.imag(tmp_array).tolist()

                else:
                    # Normal dtype, no need to change the image and real part
                    # Need to check the dimension
                    if len(tmp_array.shape) == 2:
                        # A 2D Array
                        tmp_list = [item.tolist() for item in tmp_array]
                        rec_dict[key] = tmp_list
                    else:
                        # A 1D Array
                        rec_dict[key] = tmp_array.tolist()

            elif type(self.__dict__[key]) is list and type(self.__dict__[key][0]) in [ImageSourceRay, AcousticRay]:
                pass
                '''
                tmp_list = []
                for item in self.__dict__[key]:
                    tmp_list.append(item.to_json())
                rec_dict[key] = tmp_list
                '''

            elif type(self.__dict__[key]) is Source:
                pass
                # rec_dict[key] = self.__dict__[key].to_json()

            elif type(self.__dict__[key]) is not FieldGeometry:
                rec_dict[key] = self.__dict__[key]

        for key in rec_dict.keys():
            print(f"Key is {key}, Type is {type(rec_dict[key])}")

        return rec_dict
