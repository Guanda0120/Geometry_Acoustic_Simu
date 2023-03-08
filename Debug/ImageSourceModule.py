import numpy as np
import copy
from typing import List
from Graphic.BaseGeometry import BaseGeometry
from RoomGeometry.FieldGeometry import FieldGeometry
from RoomGeometry.Plane import Plane
from Acoustic.ImageSourceRay import ImageSourceRay
from Acoustic.Source import Source
from Acoustic.Receiver import Receiver


class IM_Module:

    def __init__(self, sound_field: FieldGeometry, src: Source, rec: Receiver, max_order: int):
        """
        Init an Image source module
        :param sound_field: A FieldGeometry Object
        :param src: A Source Object
        :param rec: A Receiver Object
        :param max_order: The max image source reflex time
        """
        # Following is for result var
        self.sound_field = sound_field
        self.src = src
        self.rec = rec
        self.max_order = max_order
        self.plane_idx = list(range(len(sound_field.plane)))
        # Result Container
        self.im_ray: List[ImageSourceRay] = []

    def get_path(self):
        """
        Permutation and Combination the plane index, and verify whether exist or not.
        :return: plane index list, and exist
        """
        total_order_list = []
        total_ray: List[ImageSourceRay] = []
        # pre_order_list = []
        # pre_source_pt = []

        # tmp_order_list cache the list of exist reflex order
        tmp_order_list = [[]]
        # tmp_source_list cache the list of
        tmp_source_list = np.array([self.src.loc])
        tmp_order = 1

        while tmp_order <= self.max_order:
            # Iter and make tmp_order get to the tmp_order
            # Store this order list
            cache_order_list = []
            # Store image source point
            cache_image_pt = []

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
                        im_pt = BaseGeometry.diff_side(mir_pt, self.rec.rec_location, added_plane.normal_direc,
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
                            # Store image source point into cache_image_pt
                            cache_image_pt.append(im_pt)
                            im_ray = ImageSourceRay(self.src, self.rec, self.sound_field)
                            im_ray.n_order_im(tmp_pre)
                            if im_ray.exist_state:
                                total_ray.append(im_ray)

                    # The order is 1
                    elif len(pre_order) == 0:
                        # Mirror the pre source
                        mir_pt = BaseGeometry.mirror_pt(pre_source, added_plane.normal_direc, added_plane.intercept_d)
                        # Then check different side or not
                        im_pt = BaseGeometry.diff_side(mir_pt, self.rec.rec_location, added_plane.normal_direc,
                                                       added_plane.polygon[0], added_plane.intercept_d)
                        # print(f"im_pt: {im_pt}")
                        if im_pt is not None:
                            # Exist index, record it
                            # Seperate a new memory space
                            tmp_pre = copy.deepcopy(pre_order)
                            # append it
                            tmp_pre.append(idx)
                            # Store order list into cache_order_list
                            cache_order_list.append(tmp_pre)
                            # Store image source point into cache_image_pt
                            cache_image_pt.append(im_pt)
                            im_ray = ImageSourceRay(self.src, self.rec, self.sound_field)
                            im_ray.n_order_im(tmp_pre)
                            if im_ray.exist_state:
                                total_ray.append(im_ray)

            # Debuger
            # print(f"In order {tmp_order}, the cache order list is {cache_order_list}")
            # print(f"In order {tmp_order}, the image ray num is {len(total_ray)}")
            # Store the Result
            total_order_list.extend(cache_order_list)
            # Renew var for the next loop iter
            tmp_order_list = cache_order_list
            tmp_source_list = cache_image_pt
            tmp_order += 1

        return total_order_list, total_ray


if __name__ == '__main__':
    import time
    from CONFIG import fileDir, init_swl, receiver_radius, absorptionDict, scatterDict
    from FileIO.ReadRhino import Read3dm
    from Acoustic.ReceiverGroup import ReceiverGroup
    from Acoustic.ImageSourceRay import ImageSourceRay

    rh_model = Read3dm(fileDir)
    # Get the rh_plane, receiver and source
    rh_plane, source_location, receiver_location = rh_model.convert_file()

    # Init a Source and Receiver
    source = Source(source_location, init_swl)
    receiver_group = ReceiverGroup(receiver_location, receiver_radius)
    rec_1 = receiver_group.receiver_container[0]
    # Init a sound field
    sound_field = FieldGeometry(rh_plane, absorptionDict, scatterDict)

    im_rec1 = IM_Module(sound_field, source, rec_1, max_order=5)
    start_time = time.time()
    im_list, ray_list = im_rec1.get_path()
    end_start = time.time()
    print(f"Time Duration: {end_start - start_time}")
    for ray in ray_list:
        print(f"Receiver SPL: {ray.rec_spl}")
    # print(im_list)
