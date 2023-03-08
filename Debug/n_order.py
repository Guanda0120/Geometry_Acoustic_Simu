import numpy as np
from RoomGeometry.FieldGeometry import FieldGeometry
from Acoustic.AcousticRay import AcousticRay
from Acoustic.BaseAcoustic import BaseAcoustic
from Graphic.BaseGeometry import BaseGeometry
from Acoustic.Receiver import Receiver
from Acoustic.Source import Source


def image_source(plane_id_list: list, room: FieldGeometry, rec: Receiver, src: Source):
    """

    :param plane_id_list: From the Source to Every index plane to Receiver
    :param room: FieldGeometry Object
    :param rec: Receiver Object
    :param src: Source Object
    :return: AcousticRay Object for the image method exist, None for the image method do not exist
    """
    # Record the process var
    im_src_list = []
    ref_pt_list = []
    abs_list = []
    sca_list = []
    tmp_src_loc = src.loc

    for idx in plane_id_list:
        tmp_plane = room.plane[idx]
        # Mirror the point to the specific plane
        tmp_im_pt = BaseGeometry.mirror_pt(tmp_src_loc, tmp_plane.normal_direc, tmp_plane.intercept_d)
        # Record the image source point
        im_src_list.append(tmp_im_pt)
        # Change the next tmp source
        # TODO BUG is Here
        tmp_src_loc = tmp_im_pt

    print(f"Image Source: {im_src_list}")
    print(f"Plane ID：{plane_id_list}")
    # Reverse the list
    im_src_list.reverse()
    plane_id_list.reverse()
    tmp_rec_loc = rec.rec_location
    next_flag = True

    # Check every segment of ray is diff two sides, and none intersect with each plane
    for im_idx in range(len(im_src_list)):
        # Construct the var to loop
        tmp_im_pt = im_src_list[im_idx]
        tmp_plane_id = plane_id_list[im_idx]
        tmp_plane = room.plane[tmp_plane_id]
        # Get the reflex point on the specific plane
        tmp_ref_pt = BaseGeometry.diff_side(tmp_rec_loc, tmp_im_pt, tmp_plane.normal_direc,
                                            tmp_plane.polygon[0], tmp_plane.intercept_d)
        # Check the tmp_ref_pt is inside tmp_plane or not
        inside_state = BaseGeometry.pt_in_polygon(tmp_ref_pt, tmp_plane.polygon, tmp_plane.normal_direc,
                                                  tmp_plane.intercept_d)

        if tmp_ref_pt is not None and inside_state is True:

            # Check the path have intersection with other plane
            inter_state = False
            # Remove self plane
            all_plane_idx = list(range(len(room.plane)))
            all_plane_idx.remove(tmp_plane_id)

            # Check each polygon exclude self, and compute the intersection
            for check_idx in all_plane_idx:
                check_plane = room.plane[check_idx]
                # Have intersection with it, need to exit
                if BaseGeometry().cross_polygon(tmp_rec_loc, tmp_ref_pt, check_plane.normal_direc, check_plane.polygon):
                    inter_state = True
                    break

            if not inter_state:
                # Record the point and abs and sca
                ref_pt_list.append(tmp_ref_pt)
                abs_list.append(tmp_plane.absorption_coe)
                sca_list.append(tmp_plane.scatter_coe)

            else:
                next_flag = False

        else:
            next_flag = False

        # Get the next var to loop
        tmp_rec_loc = tmp_ref_pt
        if not next_flag:
            break

    if next_flag:
        # Init an acoustic ray
        abs_list = np.array(abs_list)
        sca_list = np.array(sca_list)
        ref_pt_list = np.array(ref_pt_list)
        ray = {"abs": abs_list,
               "sca": sca_list,
               "ref": ref_pt_list}

        # ray = AcousticRay()
    else:
        ray = None

    return ray


if __name__ == '__main__':
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
    # Init a sound field
    sound_field = FieldGeometry(rh_plane, absorptionDict, scatterDict)
    print(f"Receiver Location: {receiver_group.receiver_container[0].rec_location}")
    print(f"Source Location: {source.loc}")
    print(f"Plane Normal Direction: {sound_field.plane[1].normal_direc}")
    plane_idx = [4, 2, 3]
    ray_info = image_source(plane_idx, sound_field, receiver_group.receiver_container[0], source)
    im_ray = ImageSourceRay(init_swl, source.loc, receiver_group.receiver_container[0].rec_location)
    im_ray.spl_compute(np.array(plane_idx),ray_info["ref"],ray_info["abs"],ray_info["sca"])
    print(f"Sound pressure level is {im_ray.rec_spl} dB")
