import numpy as np
from copy import deepcopy
from Graphic.BaseGeometry import BaseGeometry
from Acoustic.BaseAcoustic import BaseAcoustic
from Acoustic.Source import Source
from RoomGeometry.FieldGeometry import FieldGeometry
from Acoustic.CONST import acoustic_speed


class ImageSourceRay:
    """

    """

    def __init__(self, src: Source, rec_loc: np.ndarray, sound_field: FieldGeometry):
        """
        For a ImageSourceRay object, need Source Object, Receiver Object,
        :param src: A Source Object
        :param rec_loc: A ndarray represent rec_loc
        :param sound_field: The Sound Field Object
        """
        assert rec_loc.shape == (3,)

        # Pass para into object
        self.src = src
        # self.rec = rec
        self.sound_field = sound_field
        # self.init_power is transfer from sound power level to sound power
        self.init_power = src.init_power
        # self.init_location is for Source Location.
        self.src_loc = src.loc
        # self.rec_loc is for Source Location.
        self.rec_loc = rec_loc

        # self.plane_idx is every reflex plane index
        self.plane_idx = None
        # self.ref_pt is the reflex point on specific plane
        self.ref_pt = None
        # self.abs_list is the reflex point on which plane, the plane absorption coe
        self.ref_coe_list = None
        # self.sca_list is the reflex point on which plane, the plane scatter coe
        self.sca_coe_list = None
        # self.insert_cos is the insert angle cos
        self.insert_cos = None
        # self.rec_spl is the sound pressure level of receiver
        self.rec_pressure = None
        # The time when image source ray cross the receiver
        self.rec_time = None
        # The Image Source Ray is Valid or not
        self.exist_state = True

    def n_order_im(self, plane_idx: list):
        """
        Check whether the plane index is an exist image source model
        :param plane_idx: List of int represent plane index
        :return: Renew the state para and other para
        """
        self.plane_idx = plane_idx

        # Record the process var
        im_src_list = []
        ref_pt_list = []
        ref_coe_list = []
        sca_coe_list = []
        cos_list = []
        tmp_src_loc = self.src_loc

        # Loop over index and compute image source

        for idx in self.plane_idx:
            tmp_plane = self.sound_field.plane[idx]
            # Mirror the point to the specific plane
            tmp_im_pt = BaseGeometry.mirror_pt(tmp_src_loc, tmp_plane.normal_direc, tmp_plane.intercept_d)
            # Record the image source point
            im_src_list.append(tmp_im_pt)
            # Change the next tmp source
            tmp_src_loc = tmp_im_pt

        # Reverse the list
        im_src_list.reverse()
        plane_idx.reverse()
        tmp_rec_loc = self.rec_loc
        next_flag = True

        # Check every segment of ray is diff two sides, and none intersect with each plane
        for im_idx in range(len(im_src_list)):
            # Construct the var to loop
            tmp_im_pt = im_src_list[im_idx]
            tmp_plane_id = plane_idx[im_idx]
            tmp_plane = self.sound_field.plane[tmp_plane_id]
            # Get the reflex point on the specific plane
            tmp_ref_pt = BaseGeometry.diff_side(tmp_rec_loc, tmp_im_pt, tmp_plane.normal_direc,
                                                tmp_plane.polygon[0], tmp_plane.intercept_d)

            # Check the tmp_ref_pt is inside tmp_plane or not
            inside_state = False
            if tmp_ref_pt is not None:
                # print(f"tmp_ref_pt: {tmp_ref_pt}")
                inside_state = BaseGeometry.pt_in_polygon(tmp_ref_pt, tmp_plane.polygon, tmp_plane.normal_direc,
                                                          tmp_plane.intercept_d)

            if tmp_ref_pt is not None and inside_state is True:

                # Check the path have intersection with other plane
                inter_state = False
                # Remove self plane
                all_plane_idx = list(range(len(self.sound_field.plane)))
                all_plane_idx.remove(tmp_plane_id)

                # Check each polygon exclude self, and compute the intersection
                for check_idx in all_plane_idx:
                    check_plane = self.sound_field.plane[check_idx]
                    # Have intersection with it, need to exit
                    if BaseGeometry().cross_polygon(tmp_rec_loc, tmp_ref_pt, check_plane.normal_direc,
                                                    check_plane.polygon):
                        inter_state = True
                        break

                if not inter_state:
                    # TODO Compute Cosine Here
                    tmp_insert_vec = (tmp_ref_pt - tmp_im_pt) / np.linalg.norm((tmp_ref_pt - tmp_im_pt))
                    tmp_cos = np.abs(np.dot(tmp_insert_vec, tmp_plane.normal_direc))
                    # Record the point and abs and sca
                    ref_pt_list.append(tmp_ref_pt)
                    ref_coe_list.append(tmp_plane.ref_coe)
                    sca_coe_list.append(tmp_plane.sca_coe)
                    cos_list.append(tmp_cos)

                else:
                    next_flag = False

            else:
                next_flag = False

            # Get the next var to loop
            tmp_rec_loc = tmp_ref_pt
            if not next_flag:
                break

        if next_flag:
            # Valid Image Source Ray, Renew the ref_pt, abs_list, sca_list
            self.exist_state = True
            # Need to Reverse the List
            self.ref_pt = np.array(ref_pt_list)[::-1]
            self.ref_coe_list = np.array(ref_coe_list)[::-1]
            self.sca_coe_list = np.array(sca_coe_list)[::-1]
            self.insert_cos = np.array(cos_list)[::-1]
            # print(self.insert_cos)
            # Compute the receiver SPL
            # Compute propagate distance
            start_pt = np.vstack((self.rec_loc[np.newaxis, :], self.ref_pt))
            end_pt = np.vstack((self.ref_pt, self.src.loc[np.newaxis, :]))
            dist = np.sum(np.linalg.norm((end_pt - start_pt), axis=1))

            # Compute sound pressure reduce
            rec_pressure = BaseAcoustic.source2pressure(self.init_power, dist, self.ref_coe_list, self.sca_coe_list,
                                                        self.insert_cos)
            # Air attenuate
            rec_pressure = BaseAcoustic.air_atten(dist, rec_pressure)
            self.rec_pressure = rec_pressure
            self.rec_time = dist / acoustic_speed

        else:
            # Invalid Image Source Ray
            self.exist_state = False

        plane_idx.reverse()

    def to_json(self):
        """
        Turn this obj to a dict
        :return: a dict
        """
        # Construct a blank dictionary
        ray_dict = {}
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
                        ray_dict[key_real] = real_list
                        ray_dict[key_imag] = imag_list
                    else:
                        # 1D Array
                        # Change the real and image part to the list
                        ray_dict[key_real] = np.real(tmp_array).tolist()
                        ray_dict[key_imag] = np.imag(tmp_array).tolist()

                else:
                    # Normal dtype, no need to change the image and real part
                    # Need to check the dimension
                    if len(tmp_array.shape) == 2:
                        # A 2D Array
                        tmp_list = [item.tolist() for item in tmp_array]
                        ray_dict[key] = tmp_list
                    else:
                        # A 1D Array
                        ray_dict[key] = tmp_array.tolist()

            elif type(self.__dict__[key]) not in [Source, FieldGeometry]:
                # Something need to del
                ray_dict[key] = self.__dict__[key]

        return ray_dict
