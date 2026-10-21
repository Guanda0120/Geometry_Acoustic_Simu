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
