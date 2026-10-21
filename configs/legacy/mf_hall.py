# Run settings ported from the thesis code (Real_Config.py).
# Paths are relative to the repository root.
import numpy as np

Config = {
    'absorptionDict': {
        "Door": np.array([0.18, 0.15, 0.10, 0.10, 0.10, 0.10]),
        "Floor": np.array([0.02, 0.01, 0.02, 0.02, 0.02, 0.03]),
        "Wood": np.array([0.27, 0.42, 0.85, 0.42, 0.48, 0.50]),
        "Glass": np.array([0.04, 0.03, 0.04, 0.04, 0.05, 0.05]),
        "Led": np.array([0.03, 0.05, 0.04, 0.04, 0.04, 0.04]),
        "Woolen": np.array([0.35, 0.42, 0.85, 0.39, 0.52, 0.50]),
        "Col": np.array([0.02, 0.01, 0.02, 0.02, 0.02, 0.03]),
        "White": np.array([0.23, 0.15, 0.30, 0.21, 0.32, 0.24]),
    },

    'scatterDict': {
        "Door": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Floor": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Wood": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Glass": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Led": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Woolen": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Col": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "White": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
    },
    # ====================================== Could Change Para ==========================================
    'fileDir': 'models/private/Real_Work.3dm',
    'num_ray': 20000,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 3,
    'init_swl': 94,
    'end_rt_spl': 50,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': 'output/mf_hall'
}
