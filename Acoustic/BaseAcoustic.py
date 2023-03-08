import numpy as np
from Acoustic.CONST import air_rho, acoustic_speed, attenuation_coe


class BaseAcoustic:
    """
    Some transfer of acoustic physical quantity
    """

    @staticmethod
    def spl2pressure(spl: float or np.ndarray):
        """
        Input a sound pressure level, transform it into sound pressure
        :param spl: Sound pressure level, dB. Ndarray or float
        :return: Sound pressure, pa. Ndarray or float
        """
        p_0 = 2e-5
        sound_pressure = 10 ** (spl / 20) * p_0
        return sound_pressure

    @staticmethod
    def pressure2spl(sound_pressure: float or np.ndarray):
        """
        Input a sound pressure, transform it into sound pressure level
        :param sound_pressure: Sound pressure, pa. Ndarray or float
        :return: Sound pressure level, dB. Ndarray or float
        """
        p_0 = 2e-5
        spl = 20 * np.log10((np.abs(sound_pressure) / p_0))
        return spl

    @staticmethod
    def power2swl(sound_power: float or np.ndarray):
        """
        Input a sound power level, transform it into sound power
        :param sound_power: Sound power, W. Ndarray or float
        :return: Sound power level, dB. Ndarray or float
        """
        w_0 = 1e-12
        swl = 10 * np.log10((sound_power / w_0))
        return swl

    @staticmethod
    def swl2power(swl: float or np.ndarray):
        """
        Input a sound power level, transform it into sound power
        :param swl: Sound power level, dB. Ndarray or float
        :return: Sound power, W. Ndarray or float
        """
        w_0 = 1e-12
        sound_power = 10 ** (swl / 10) * w_0
        return sound_power

    @staticmethod
    def direct_pressure(init_power: np.ndarray, dist: float):
        """
        Compute the direct sound propagate into the receiver
        :param init_power: 125Hz to 4kHz source init power, W
        :param dist: Total distance propagate to this point, m
        :return: The pressure of the point. Hint no distance attenuate pressure.
        """
        # Compute the direct pressure of the source
        pressure = np.sqrt((init_power * air_rho * acoustic_speed) / (4 * np.pi * (dist ** 2)))

        return pressure

    @staticmethod
    def source2pressure(init_power: np.ndarray, dist: float,
                        ref_list: np.ndarray, scatter_list: np.ndarray, incidence_cos: np.ndarray):
        """
        Get the pressure of the reflection point
        :param init_power: 125Hz to 4kHz source init power, W
        :param dist: Total distance propagate to this point, m
        :param ref_list: Every plane reflection coe, complex or float, (-)
        :param scatter_list: Every plane scatter coe, (-)
        :param incidence_cos: Insert incident cos, (-)
        :return: The pressure of the point. Hint no distance attenuate pressure.
        """
        # Check legal input or not
        assert init_power.shape == (6,)
        assert ref_list.shape[1] == scatter_list.shape[1] == 6
        # Get direct reflection attenuate coe
        # Reshape the cos of the incidence coe
        incidence_cos = np.tile(incidence_cos, (6, 1)).T
        # Get the normal-incidence reflection coefficient
        ksi_0 = (1 + ref_list) / (1 - ref_list)
        # Get the Special incidence of the reflection ray
        reflection_coe = (np.multiply(ksi_0, incidence_cos) - 1) / (np.multiply(ksi_0, incidence_cos) + 1)
        # Get the reduce of the power
        ref_prod = np.prod(reflection_coe, axis=0)
        # sca_prod = np.prod((1 - scatter_list), axis=0)
        '''
        Original Code without the cos consideration
        coe = np.prod(np.multiply(np.sqrt(1-absorption_list), (1 - scatter_list)), axis=0)
        '''
        # Suppose it is sphere wave, get the pressure
        # TODO Bug Is Here, The coe is for pressure, not for energy.
        pressure = ref_prod * np.sqrt((init_power * acoustic_speed * air_rho) / (4 * np.pi * (dist ** 2)))

        return pressure

    # TODO Consider The cos of the incidence
    @staticmethod
    def source_diffuse_pressure(init_power: np.ndarray, dist: float, ref_list: np.ndarray,
                                scatter_list: np.ndarray, incidence_cos: np.ndarray, diffuse_cos_theta: float,
                                diffuse_cos_gamma: float):
        """
        This method uses for the last reflection is diffusion, get the insert sound pressure
        :param init_power: 125Hz to 4kHz source init sound power, W
        :param dist: Total distance propagate to this point, m
        :param ref_list: Every plane reflection coe, complex or float, (-)
        :param scatter_list: Every plane scatter coe, (-)
        :param incidence_cos: Each insert incident, (-)
        :param diffuse_cos_theta: The diffuse ray direction and plane normal direction angle, (-)
        :param diffuse_cos_gamma: solid angle, (-)
        :return: The diffuse pressure.
        """
        # Check legal input or not
        assert init_power.shape == (6,)
        assert ref_list.shape[1] == scatter_list.shape[1] == 6

        # Get direct reflection atten coe
        # TODO Bug Is Here
        if ref_list.shape[0] != 1:
            # Reshape the cos of the incidence coe
            incidence_cos = np.tile(incidence_cos[:-1], (6, 1)).T
            # Get the normal-incidence reflection coefficient
            ksi_0 = (1 + ref_list[:-1]) / (1 - ref_list[:-1])
            # Get the Special incidence of the reflection ray, pressure reflection coe
            pre_reflection_coe = (np.multiply(ksi_0, incidence_cos) - 1) / (np.multiply(ksi_0, incidence_cos) + 1)
            pre_reflection_coe = np.prod(pre_reflection_coe, axis=0)
            # Get the pre scatter coe of the ray
            pre_scatter_coe = np.prod((1 - scatter_list[:-1]), axis=0)
        else:
            pre_reflection_coe = np.array([1, 1, 1, 1, 1, 1])
            pre_scatter_coe = np.array([1, 1, 1, 1, 1, 1])

        # The final scatter coe
        tmp_coe = np.multiply(ref_list[-1] ** 2, scatter_list[-1]) * (
                1 - diffuse_cos_gamma) * diffuse_cos_theta * 2
        scatter_energy_coe = tmp_coe * pre_scatter_coe
        # TODO Bug Is Here, The coe is for pressure, not for energy.
        pressure = pre_reflection_coe * np.sqrt(
            np.multiply(scatter_energy_coe, init_power) * acoustic_speed * air_rho / (4 * np.pi * (dist ** 2)))

        return pressure

    @staticmethod
    def air_atten(dist: float, un_atten_pressure: np.ndarray):
        """
        Compute the air absorption in pressure level
        :param dist: Sound propagate distance
        :param un_atten_pressure: The sound pressure without air absorption
        :return: The sound pressure level with air absorption
        """
        # SNR
        atten_pressure = un_atten_pressure / (10 ** ((dist * attenuation_coe) / 20))

        return atten_pressure

    @staticmethod
    def pressure_shift(pressure: np.ndarray, delay_t: np.ndarray):
        """
        Use to move the sound pressure to the 44100 sample ratio postion
        :param pressure: The complex sound pressure of the propagate insert
        :param delay_t: The time delay
        :return: shift sound pressure
        """
        assert pressure.shape[0] == delay_t.shape[0]
        # Match time delay vector to the pressure array
        delay_t = np.tile(delay_t, (6, 1)).T
        # Match the frequency vector to pressure matrix
        freq = np.tile(np.array([125, 250, 500, 1000, 2000, 4000]), (pressure.shape[0], 1))
        # The angular frequency
        omega = 2 * np.pi * freq
        # The shift sound pressure
        shift_pressure = pressure * np.exp(1j * (omega * delay_t))

        return shift_pressure

    @staticmethod
    def pressure_addition(pressure_array: np.ndarray, **kwargs):
        """
        Add multiple pressure together
        :param pressure_array: 2D array, axis_0 is the dimension to added, axis_1 is the frequency interest
        :param kwargs: add_method['linear'] simple add; add_method['rms'] root mean square
        :return: The pressure array
        """
        # Default add method is linear
        add_method = 'max'
        if kwargs.__contains__('add_method'):
            add_method = kwargs['add_method']
        assert add_method in ['linear', 'rms', 'max']

        if add_method == 'linear':
            # Keep Phase
            add_pressure = np.sum(pressure_array, axis=0)

        elif add_method == 'max' and pressure_array != []:
            # Keep Phase
            add_pressure = np.max(np.abs(pressure_array), axis=0)

        else:
            # Loss Phase
            add_pressure = np.sqrt(np.sum(np.abs(pressure_array) ** 2, axis=0))

        return add_pressure


if __name__ == "__main__":
    pressure = np.ones((10, 6)) * (0.5 + 0.3j)
    delta_t = np.linspace(0, 0.01, 10)

    shift_pressure = BaseAcoustic.pressure_shift(pressure, delta_t)
    print(shift_pressure)
    add_shift = BaseAcoustic.pressure_addition(shift_pressure, add_method='linear')
    print(add_shift)
