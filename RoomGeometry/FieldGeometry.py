from typing import List
import numpy as np
from RoomGeometry.Plane import Plane
from Acoustic.MaterialDict import MaterialDict


class FieldGeometry:
    """
    Contain the plane object
    """

    def __init__(self, plane_dict: dict, material_dict: MaterialDict, correct_line: np.ndarray):
        """
        Init a FieldGeometry object, and init all the plane
        :param plane_dict: Plane Dict from readrhino
        :param absorption_dict: Absorption Coe
        :param scatter_dict: Scatter Coe
        :param correct_line: The direction to be Correct
        """

        self.plane_dict = plane_dict
        self.material_dict = material_dict
        # Restrict the list can only contain the Plane object
        self.plane: List[Plane] = []

        # Loop over all the key in dictionary
        for key in self.plane_dict.keys():
            # Loop over all the rhino brep
            for rh_plane in plane_dict[key]:
                # Init a plane
                tmp_plane = Plane(rh_plane, material_dict.material_dict[key])
                self.plane.append(tmp_plane)

        # Correct the normal direction
        tmp_direc = correct_line[:, 1, :] - correct_line[:, 0, :]
        for tmp_plane in self.plane:
            for direc_idx in range(len(correct_line)):
                # Get the center point
                cen_pt = correct_line[direc_idx, 0, :]
                if np.linalg.norm(cen_pt - tmp_plane.cen_pt) < 1e-4:
                    # Check if the wrong direction
                    if np.linalg.norm(tmp_direc[direc_idx] + tmp_plane.normal_direc) < 1e-4:
                        tmp_plane.normal_direc = tmp_direc[direc_idx]/np.linalg.norm(tmp_direc[direc_idx])
                        d = -tmp_plane.intercept_d
                        tmp_plane.intercept_d = d
                    break
        # TODO Volume of the sound field

    def __str__(self):
        return f"The Sound Field Have {len(self.plane)} planes. Use {len(self.material_dict.material_dict)} Kinds of Material."


if __name__ == "__main__":
    from CONFIG import absorptionDict, scatterDict, fileDir
    from FileIO.ReadRhino import Read3dm

    rh_model = Read3dm(fileDir)
    rh_plane, receiver, source, correct_direc = rh_model.convert_file()
    sound_field = FieldGeometry(rh_plane, absorptionDict, scatterDict,correct_direc)
    direc_line = [tmp_plane.normal_direc for tmp_plane in sound_field.plane]
    print(np.asarray(direc_line))
