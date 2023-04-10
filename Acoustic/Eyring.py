import numpy as np
from RoomGeometry.FieldGeometry import FieldGeometry


def eyring(sound_field: FieldGeometry, volume: float):
    """
    Compute Reverb Time use Eyling equation
    :param sound_field: The Sound Field Object
    :param volume: 
    :return: 
    """
    alpha_avg = 0
    area_sum = 0

    for polygon in sound_field.plane:
        area_sum += polygon.area
        alpha_avg += polygon.diffuse_abs_coe * polygon.area

    alpha_avg = alpha_avg/area_sum
    eyring_reverb = 0.161*volume/(-area_sum*np.log(1-alpha_avg))

    return eyring_reverb

if __name__ == '__main__':

    # from CONFIG import Config
    from Real_Config import Config
    from Acoustic.MaterialDict import MaterialDict
    from FileIO.ReadRhino import Read3dm
    volume = 3695
    absorptionDict = Config['absorptionDict']
    scatterDict = Config['scatterDict']
    fileDir = Config['fileDir']
    init_swl = Config['init_swl']
    num_ray = Config['num_ray']
    rt_method = Config['rt_method']
    receiver_radius = Config['receiver_radius']
    end_rt_spl = Config['end_rt_spl']
    im_order = Config['im_order']
    output_folder = Config['output_folder']
    sample_freq = Config['sample_freq']
    opt_max_res = Config['opt_max_res']
    model_name = Config['model_name']

    material_dict = MaterialDict(absorptionDict, scatterDict, opt_max_res=opt_max_res, model_name=model_name)
    # Read the rhino file, and convert it
    rh_model = Read3dm(fileDir)
    # Get the rh_plane, receiver and source
    rh_plane, source_location, receiver_location, correct_line = rh_model.convert_file()
    # Init a sound field
    sound_field = FieldGeometry(rh_plane, material_dict, correct_line)

    # Compute Eyring
    rt = eyring(sound_field, volume)
    print(f"Eyring Equation Reverb Time is {rt}")
