import numpy as np
from Acoustic.BaseAcoustic import BaseAcoustic


class AcousticRay:
    """
    Acoustic Ray is the object to record every reflect and attenuation
    """
    def __init__(self, init_swl: float, init_location: np.ndarray, init_direc: np.ndarray):
        """
        Init an acoustic ray, contain every reflection info
        :param init_swl: The init sound power level span 125Hz to 4kHz
        :param init_location: The location of the start point
        :param init_direc: The direction of the start ray
        """
        self.init_swl = init_swl * np.ones(6)
        # self.init_power is transfer from sound power level to sound power
        self.init_power = BaseAcoustic.swl2power(self.init_swl) + np.zeros(6, dtype=complex)
        # self.init_location is for Source Location.
        self.init_location = init_location
        # self.init_direc is for initial direction of the ray
        self.init_direc = init_direc
        # self.reflect_plane_idx is for which plane the ray insert on.
        self.reflect_plane_idx = []
        # self.reflect_point the point coordinate
        self.start_point = [self.init_location]
        # self.ray_direc is the direction of every ray
        self.ray_direc = [self.init_direc]
        # The reflection coefficient of the plane
        self.ref_coe = []
        # The scatter coefficient of the plane
        self.sca_coe = []
        # self.insert_cos is insert ray incidence theta cos
        self.insert_cos = []
        # The time
        self.time = []
        # The wave propagate distance
        self.dist = []

    def to_json(self):
        """
        Turn this obj to a dict
        :return: a dict
        """
        # Construct a blank dict
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
            else:
                ray_dict[key] = self.__dict__[key]

        return ray_dict


if __name__ == "__main__":
    ray = AcousticRay(94,np.array([1,1,1]),np.array([1,1,1]))
    print(f"Sound power level: {ray.init_swl}")
    print(f"Sound power level: {ray.init_power}")
    print(f"Sound direction: {ray.ray_direc}")


