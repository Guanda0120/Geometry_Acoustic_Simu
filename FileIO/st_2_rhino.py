import numpy as np
import rhino3dm
import pickle
import os
from Acoustic.ImageSourceRay import ImageSourceRay
from Acoustic.Receiver import Receiver


def st2rhino(pickle_dir: str, out_sir: str):
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
        cur_layer = rhino3dm.Layer()
        cur_layer.Name = 'Stochastic_Ray'
        cur_layer.Visible = True
        cur_layer.Color = (255, 128, 128, 255)
        rh_file.Layers.Add(cur_layer)


        # Get layer index
        layer_index = [each.Index for each in rh_file.Layers]

        if isinstance(rec, Receiver):

            for st_ray in rec.stochastic_ray:
                # Point Container
                point_container = []
                src_point = rhino3dm.Point3d(st_ray.init_location[0], st_ray.init_location[1], st_ray.init_location[2])
                point_container.append(src_point)
                for pt in st_ray.start_point:
                    tmp_pt = rhino3dm.Point3d(pt[0], pt[1], pt[2])
                    point_container.append(tmp_pt)

                polyline_rh = rhino3dm.Polyline(point_container)
                polycur = rhino3dm.Curve.CreateControlPointCurve(polyline_rh, degree=1)
                id = rh_file.Objects.AddCurve(polycur)
                # Add Layer
                cur_obj = rh_file.Objects.FindId(str(id))
                cur_obj.Attributes.LayerIndex = layer_index[0]

            rh_file.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Meters
            rh_file.Write(out_sir, 6)


if __name__ == '__main__':
    pickle_dir = r"D:\\PlanningHall\\rec_0.pkl"
    rhino_dir = r'C:\Users\12748\Desktop\Real_rec0.3dm'
    st2rhino(pickle_dir, rhino_dir)

