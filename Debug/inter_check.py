import numpy as np
from FileIO.ReadRhino import Read3dm
from CONFIG import absorptionDict, scatterDict, fileDir
from RoomGeometry.FieldGeometry import FieldGeometry
from Graphic.BaseGeometry import BaseGeometry


# Read the rhino file, and convert it
rh_model = Read3dm(fileDir)
# Get the rh_plane, receiver and source
rh_plane, receiver, source = rh_model.convert_file()

start_pt = np.array([3.07607917, 6.58435732, 8.5])
ray_direc = np.array([0.66702204, 0.19928198, -0.71789156])
sound_field = FieldGeometry(rh_plane, absorptionDict, scatterDict)

inter_list = []
plane_list = []
for idx in range(len(sound_field.plane)):
    polygon = sound_field.plane[idx]
    inter_pt = BaseGeometry().ray_plane_inter(start_pt, ray_direc, polygon.polygon, polygon.normal_direc)
    if inter_pt is not None:
        inter_list.append(inter_pt)
        plane_list.append(idx)
        print(f"The polygon {polygon.polygon}, Have intersection point on {inter_pt}")
