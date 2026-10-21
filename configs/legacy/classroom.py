# Run settings ported from the thesis code (TestDict/Shape/RealClassRoom/Real_ClassRoom_Config.py).
# Paths are relative to the repository root.
import numpy as np

Config = {
    'absorptionDict': {
        "Glass": np.array([0.03, 0.03, 0.02, 0.02, 0.01, 0.01]),
        "Wood": np.array([0.17, 0.12, 0.11, 0.09, 0.08, 0.10]),
        "BlackBoard": np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]),
        "White": np.array([0.02, 0.02, 0.03, 0.02, 0.03, 0.04]),
        "Roof": np.array([0.11, 0.25, 0.18, 0.25, 0.19, 0.14]),
        "Floor": np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
    },

    'scatterDict': { 
        "Glass": np.ones(6)*0.10,
        "Wood": np.ones(6)*0.10,
        "BlackBoard": np.ones(6)*0.10,
        "White": np.ones(6)*0.10,
        "Roof": np.ones(6)*0.10,
        "Floor": np.ones(6)*0.10
    },
    # ====================================== Could Change Para ==========================================
    'fileDir': 'models/private/Real_ClassRoom.3dm',
    'num_ray': 9160,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 5,
    'init_swl': 94,
    'end_rt_spl': 60,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': 'output/classroom'
}
