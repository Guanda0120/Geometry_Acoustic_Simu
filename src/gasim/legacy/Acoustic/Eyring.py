import numpy as np
from gasim.legacy.RoomGeometry.FieldGeometry import FieldGeometry


def eyring(sound_field: FieldGeometry, volume: float):
    """
    Compute Reverb Time use Eyring and Sabine equation
    :param sound_field: The Sound Field Object
    :param volume: Room volume, m3
    :return: Eyring reverb time, Sabine reverb time, area-weighted mean absorption coefficient
    """
    alpha_avg = 0
    area_sum = 0

    for polygon in sound_field.plane:
        area_sum += polygon.area
        alpha_avg += polygon.diffuse_abs_coe * polygon.area

    alpha_avg = alpha_avg/area_sum
    eyring_reverb = 0.161*volume/(-area_sum*np.log(1-alpha_avg))
    sabine_reverb = 0.161*volume/(area_sum*alpha_avg)

    return eyring_reverb, sabine_reverb, alpha_avg
