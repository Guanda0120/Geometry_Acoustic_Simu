import numpy as np
import rhino3dm
import pickle
import os
from Acoustic.ImageSourceRay import ImageSourceRay
from Acoustic.Receiver import Receiver


def im2rhino(pickle_dir: str, out_sir: str):
    """

    :param pickle_dir:
    :param out_sir:
    :return:
    """
    assert os.path.exists(pickle_dir), ValueError("Pickle dir is not exist!")

    with open(pickle_dir, 'rb') as f:
        rec = pickle.load(f)

        # Create a Rhino file
        rh_file = rhino3dm.File3dm()
        # Layer setting
        layer_name = ['1st_order', '2nd_order', '3rd_order', '4th_order', '5th_order']
        layer_color = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (0, 255, 255, 255), (255, 0, 255, 255)]

        for idx in range(len(layer_name)):
            cur_layer = rhino3dm.Layer()
            cur_layer.Name = layer_name[idx]
            cur_layer.Visible = True
            cur_layer.Color = layer_color[idx]
            rh_file.Layers.Add(cur_layer)

        # Get layer index
        layer_index = [each.Index for each in rh_file.Layers]

        # Construct Building Object
        id_list = []
        layer_id = []

        if isinstance(rec, Receiver):

            for imray in rec.im_ray:
                # Point Container
                point_container = []
                src_point = rhino3dm.Point3d(imray.src_loc[0], imray.src_loc[1], imray.src_loc[2])
                point_container.append(src_point)
                for pt in imray.ref_pt:
                    tmp_pt = rhino3dm.Point3d(pt[0], pt[1], pt[2])
                    point_container.append(tmp_pt)

                rec_point = rhino3dm.Point3d(imray.rec_loc[0], imray.rec_loc[1], imray.rec_loc[2])
                point_container.append(rec_point)

                polyline_rh = rhino3dm.Polyline(point_container)
                polycur = rhino3dm.Curve.CreateControlPointCurve(polyline_rh, degree=1)
                id = rh_file.Objects.AddCurve(polycur)
                # Add Layer
                cur_obj = rh_file.Objects.FindId(str(id))
                cur_obj.Attributes.LayerIndex = layer_index[len(point_container)-3]

            rh_file.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Meters
            rh_file.Write(out_sir, 6)


if __name__ == '__main__':
    pickle_dir = r"D:\\shape\\L2B\\rec_1.pkl"
    rhino_dir = r'C:\Users\12748\Desktop\L2B_rec1.3dm'
    im2rhino(pickle_dir, rhino_dir)

