"""Statistical reverberation time: Sabine and Eyring, with air absorption."""
import numpy as np

from gasim.geometry.room import Room


def sabine(room: Room, speed_of_sound: float, air_attenuation=0.0):
    """
    Sabine reverberation time T = 24 ln(10) V / (c (A + 4 m V))
    :param room: Room
    :param speed_of_sound: m/s
    :param air_attenuation: Intensity attenuation constant m per band, 1/m
    :return: (N_BANDS,) reverberation time, s
    """
    volume = room.volume
    return 24 * np.log(10) * volume / (speed_of_sound * (room.absorption_area() + 4 * air_attenuation * volume))


def eyring(room: Room, speed_of_sound: float, air_attenuation=0.0):
    """
    Eyring reverberation time T = 24 ln(10) V / (c (-S ln(1 - mean alpha) + 4 m V))
    :param room: Room
    :param speed_of_sound: m/s
    :param air_attenuation: Intensity attenuation constant m per band, 1/m
    :return: (N_BANDS,) reverberation time, s
    """
    volume, surface = room.volume, room.surface_area
    mean_alpha = room.absorption_area() / surface
    return 24 * np.log(10) * volume / (speed_of_sound * (-surface * np.log(1 - mean_alpha)
                                                         + 4 * air_attenuation * volume))


def kuttruff(room: Room, speed_of_sound: float, air_attenuation=0.0, relative_variance: float = 0.4):
    """
    Eyring corrected for the spread of free path lengths, Kuttruff, Room Acoustics, 6th ed., Eq. (5.33)-(5.34):
    a'' = -ln(1 - mean alpha) (1 + gamma2 / 2 ln(1 - mean alpha)); valid for diffusely reflecting walls
    :param room: Room
    :param speed_of_sound: m/s
    :param air_attenuation: Intensity attenuation constant m per band, 1/m
    :param relative_variance: gamma2 of the free path lengths, about 0.34-0.6 for rectangular rooms (Table 5.1)
    :return: (N_BANDS,) reverberation time, s
    """
    volume, surface = room.volume, room.surface_area
    log_reflection = np.log(1 - room.absorption_area() / surface)
    exponent = -log_reflection * (1 + relative_variance / 2 * log_reflection)
    return 24 * np.log(10) * volume / (speed_of_sound * (surface * exponent + 4 * air_attenuation * volume))
