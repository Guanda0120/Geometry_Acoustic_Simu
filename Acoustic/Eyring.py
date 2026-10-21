import numpy as np
from RoomGeometry.FieldGeometry import FieldGeometry


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


if __name__ == '__main__':
    # Usage: python -m Acoustic.Eyring configs/<name>.py <volume in m3>
    import sys
    from Simulation import load_config, repo_path
    from Acoustic.MaterialDict import MaterialDict
    from FileIO.ReadRhino import Read3dm

    Config = load_config(sys.argv[1])
    volume = float(sys.argv[2])

    material_dict = MaterialDict(Config['absorptionDict'], Config['scatterDict'],
                                 opt_max_res=Config['opt_max_res'], model_name=Config['model_name'])
    # Read the rhino file, and convert it
    rh_model = Read3dm(repo_path(Config['fileDir']))
    # Get the rh_plane, receiver and source
    rh_plane, source_location, receiver_location, correct_line = rh_model.convert_file()
    # Init a sound field
    sound_field = FieldGeometry(rh_plane, material_dict, correct_line)

    # Compute Eyring
    e_rt, s_rt, alpha_avg = eyring(sound_field, volume)
    print(f"Eyring Equation Reverb Time is {e_rt}")
    print(f"Sabine Equation Reverb Time is {s_rt}")
    print(f"The Average Absorption Coe is {alpha_avg}")
