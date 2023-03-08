import numpy as np
from Acoustic.BaseAcoustic import BaseAcoustic


class Source:
    # Acoustic Source object
    def __init__(self, loc: np.ndarray, init_swl: float):
        """
        Init a source object
        :param loc: A ndarray represent location
        :param init_swl: Sound power level
        """
        assert loc.shape == (3,)
        self.loc = loc
        self.init_swl = init_swl
        self.init_power = BaseAcoustic.swl2power(np.ones(6) * init_swl) + np.zeros(6, dtype=complex)

    def to_json(self):
        """
        Turn this obj to a dict
        :return: a dict
        """
        # Construct a blank dict
        src_dict = {}
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
                        src_dict[key_real] = real_list
                        src_dict[key_imag] = imag_list
                    else:
                        # 1D Array
                        # Change the real and image part to the list
                        src_dict[key_real] = np.real(tmp_array).tolist()
                        src_dict[key_imag] = np.imag(tmp_array).tolist()

                else:
                    # Normal dtype, no need to change the image and real part
                    # Need to check the dimension
                    if len(tmp_array.shape) == 2:
                        # A 2D Array
                        tmp_list = [item.tolist() for item in tmp_array]
                        src_dict[key] = tmp_list
                    else:
                        # A 1D Array
                        src_dict[key] = tmp_array.tolist()

        return src_dict

    def poisson_distribute(self):
        """

        :return:
        """
        # TODO As refer says Poisson Distribution Dirac Signal
        pass


if __name__ == '__main__':
    s_1 = Source(np.array([1, 1, 1]), 94)
    print(f"Power Array{s_1.init_power}")
