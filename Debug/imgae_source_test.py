import numpy as np
import time
from FileIO.ReadRhino import Read3dm
from Graphic.BaseGeometry import BaseGeometry
from RoomGeometry.FieldGeometry import FieldGeometry
from Acoustic.RayGroup import RayGroup
from Acoustic.AcousticRay import AcousticRay
from Acoustic.ReceiverGroup import ReceiverGroup
from Acoustic.Receiver import Receiver
from Acoustic.Source import Source
from CONFIG import absorptionDict, scatterDict, fileDir, init_swl, num_ray, rt_method, receiver_radius, end_rt_spl
from FileIO.ResponsePlot import plot_response
from Acoustic.BaseAcoustic import BaseAcoustic


def order_1_im(rec:Receiver, src:Source, sound_field: FieldGeometry):
    ray_container = []
    abs_container = []
    sca_container = []
    print(f"Length of the polygon list: {len(sound_field.plane)}")
    for pl_idx in range(len(sound_field.plane)):
        polygon = sound_field.plane[pl_idx]
        print(f"Polygon numpy array is {polygon.polygon}")
        rest_polygon = [pl for pl in sound_field.plane]
        # print(f"sound_field Memory ID is {id(sound_field.plane)}")
        # print(f"rest_polygon Memory ID is {id(rest_polygon)}")
        rest_polygon.pop(pl_idx)
        # print(f"Length of the {len(sound_field.plane)}")
        im_source = BaseGeometry.mirror_pt(src.loc, polygon.normal_direc, polygon.intercept_d)
        inter_pt = BaseGeometry.diff_side(im_source, rec.rec_location, polygon.normal_direc, polygon.polygon[0],polygon.intercept_d)

        # Check intersect point inside polygon

        if inter_pt is not None and BaseGeometry.pt_in_polygon(inter_pt, polygon.polygon, polygon.normal_direc, polygon.intercept_d):
            print(f"point is {inter_pt}. Need Check have inter with other.")
            cross_flag = False
            for pl_check in rest_polygon:
                cross_bool = BaseGeometry().cross_polygon(im_source,rec.rec_location,pl_check.normal_direc,pl_check.polygon)
                if cross_bool:
                    cross_flag = True
                    # break
            if cross_flag is False:
                ray_container.append(inter_pt)
                abs_container.append(polygon.absorption_coe)
                sca_container.append(polygon.scatter_coe)
                print("Yes")
            else:
                print("No")

        else:
            print("Drop Out")
        print("========================================")

    # Acoustic Compute
    ray_container = np.array(ray_container)
    abs_container = np.array(abs_container)
    sca_container = np.array(sca_container)
    src_loc = np.tile(src.loc,(ray_container.shape[0],1))
    rec_loc = np.tile(rec.rec_location, (ray_container.shape[0],1))

    dist = np.linalg.norm((src_loc-ray_container),axis=1)+np.linalg.norm((ray_container-rec_loc),axis=1)
    print(f"dist: {dist}")
    print(f"Insert point list is {np.array(ray_container)}")
    spl_container = []
    for d_prime in dist:
        sound_pressure = BaseAcoustic.source2pressure(source.init_power, d_prime, abs_container, sca_container)
        spl = BaseAcoustic.pressure2spl(sound_pressure)
        spl_container.append(spl)

    spl_container = np.array(spl_container)
    return spl_container


if __name__ == "__main__":

    # Read the rhino file, and convert it
    rh_model = Read3dm(fileDir)
    # Get the rh_plane, receiver and source
    rh_plane, source_location, receiver_location = rh_model.convert_file()

    # Init a Source and Receiver
    source = Source(source_location, init_swl)
    receiver_group = ReceiverGroup(receiver_location, receiver_radius)
    # Init a sound field
    sound_field = FieldGeometry(rh_plane, absorptionDict, scatterDict)
    print(f"Source Location is {source.loc}")
    print(f"Receiver Location is {receiver_group.receiver_container[0].rec_location}")
    spl_all = order_1_im(receiver_group.receiver_container[0], source, sound_field)
    print(f"spl_all: {spl_all}")