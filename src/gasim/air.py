"""Speed of sound and atmospheric absorption (ISO 9613-1)."""
import numpy as np

from gasim.bands import OCTAVE_BANDS

REFERENCE_PRESSURE = 101.325  # kPa
REFERENCE_TEMPERATURE = 293.15  # K
TRIPLE_POINT = 273.16  # K, triple point of water


def speed_of_sound(temperature: float = 20.0):
    """
    Speed of sound in air
    :param temperature: Air temperature, degC
    :return: Speed of sound, m/s
    """
    return 331.3 * np.sqrt(1 + temperature / 273.15)


def attenuation_db(frequency, temperature: float = 20.0, humidity: float = 50.0, pressure: float = REFERENCE_PRESSURE):
    """
    Pure-tone atmospheric absorption coefficient, ISO 9613-1:1993 Eq. (3)-(5)
    :param frequency: Frequency, Hz, scalar or array
    :param temperature: Air temperature, degC
    :param humidity: Relative humidity, %
    :param pressure: Ambient pressure, kPa
    :return: Attenuation coefficient, dB/m
    """
    f = np.asarray(frequency, dtype=float)
    t = temperature + 273.15
    t_rel = t / REFERENCE_TEMPERATURE
    p_rel = pressure / REFERENCE_PRESSURE
    # Molar concentration of water vapour, %
    p_sat_rel = 10 ** (-6.8346 * (TRIPLE_POINT / t) ** 1.261 + 4.6151)
    h = humidity * p_sat_rel / p_rel
    # Relaxation frequencies of oxygen and nitrogen, Hz
    fr_o = p_rel * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = p_rel * t_rel ** -0.5 * (9 + 280 * h * np.exp(-4.170 * (t_rel ** (-1 / 3) - 1)))
    return 8.686 * f ** 2 * (
        1.84e-11 / p_rel * t_rel ** 0.5
        + t_rel ** -2.5 * (0.01275 * np.exp(-2239.1 / t) / (fr_o + f ** 2 / fr_o)
                           + 0.1068 * np.exp(-3352.0 / t) / (fr_n + f ** 2 / fr_n)))


def energy_attenuation(temperature: float = 20.0, humidity: float = 50.0, pressure: float = REFERENCE_PRESSURE,
                       frequency=OCTAVE_BANDS):
    """
    Intensity attenuation constant m, so that intensity decays as exp(-m x)
    :param temperature: Air temperature, degC
    :param humidity: Relative humidity, %
    :param pressure: Ambient pressure, kPa
    :param frequency: Frequencies, Hz, default the octave bands
    :return: m per frequency, 1/m
    """
    return attenuation_db(frequency, temperature, humidity, pressure) / (10 * np.log10(np.e))
