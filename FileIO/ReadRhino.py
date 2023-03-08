import rhino3dm as rhino3dm
import numpy as np


class Read3dm:
    """
    This module is for read rhino3dm file
    Read as layer, some specific layer is below
    receiver: rhino3dm.Point3d
    source: rhino3dm.Point3d
    all the other layer is sort as material
    """

    def __init__(self, file_dir: str):
        """
        Get the direction of the rhino file
        :param file_dir: absolute direction of rhino
        """
        self.file_dir = file_dir

    def convert_file(self):
        """
        Convert the rhino file into structure data
        :return: A dict have material info, receiver ndarray, source ndarray
        """
        # Read the Rhino File
        model = rhino3dm.File3dm.Read(self.file_dir)

        # Creat a Dictionary match the layer name and object
        receiver = []
        tmp_polygon_cen = []
        tmp_direc_line = []
        source = None
        model_dict = {}
        for lay in model.Layers:
            if lay.Name not in ["Receiver", "Source", "Center_pt", "Direc_line"]:
                model_dict[lay.Name] = []

        # Match the Geometry object to the dictionarygras
        for rh_obj in model.Objects:
            lay_idx = rh_obj.Attributes.LayerIndex
            dict_key = model.Layers[lay_idx].Name

            # Add material to dict
            if dict_key not in ["Receiver", "Source", "Center_pt", "Direc_line"]:
                model_dict[dict_key].append(rh_obj)

            # Add receiver point in receiver list
            elif dict_key == "Receiver":
                receiver.append(np.array([rh_obj.Geometry.Location.X, rh_obj.Geometry.Location.Y, rh_obj.Geometry.Location.Z]))

            # Change the source
            elif dict_key == "Source":
                source = np.array([rh_obj.Geometry.Location.X, rh_obj.Geometry.Location.Y, rh_obj.Geometry.Location.Z])

            # Plane Center Point
            elif dict_key == "Center_pt":
                cen_pt = np.array([rh_obj.Geometry.Location.X, rh_obj.Geometry.Location.Y, rh_obj.Geometry.Location.Z])
                tmp_polygon_cen.append(cen_pt)

            # Direction Line
            # TODO Analytical Solution of this
            elif dict_key == "Direc_line":
                # Transform Curve into Polyline
                tmp_polyline = rh_obj.Geometry.TryGetPolyline()
                # Cache the Start Point and End Point
                pt_list = []
                # Get the Rhino start point and end point
                rh_start_pt = tmp_polyline.PointAt(0)
                rh_end_pt = tmp_polyline.PointAt(1)
                # Numpy array point
                start_pt = np.asarray([rh_start_pt.X, rh_start_pt.Y, rh_start_pt.Z])
                end_pt = np.asarray([rh_end_pt.X, rh_end_pt.Y, rh_end_pt.Z])
                # 2D Array
                tmp_line = np.asarray([start_pt, end_pt])
                tmp_direc_line.append(tmp_line)

        assert source is not None
        assert len(receiver) != 0
        receiver = np.asarray(receiver)
        tmp_polygon_cen = np.asarray(tmp_polygon_cen)
        tmp_direc_line = np.asarray(tmp_direc_line)

        # Correct the normal direction
        direc_line = []
        for cen_pt in tmp_polygon_cen:
            for tmp_line in tmp_direc_line:
                if (cen_pt == tmp_line[0]).all():
                    direc_line.append(tmp_line)
                    break
                elif (cen_pt == tmp_line[1]).all():
                    direc_line.append(np.asarray([tmp_line[1],tmp_line[0]]))
                    break
        direc_line = np.asarray(direc_line)

        return model_dict, source, receiver, direc_line


if __name__ == "__main__":
    path = r"C:\\Users\\李冠达\\Desktop\\Simulation\\Test_File\\Simple_Test.3dm"
    model = Read3dm(path)
    a,b,c,d = model.convert_file()
    print(f"Direct line is {d}")
