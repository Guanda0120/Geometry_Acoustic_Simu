import numpy as np


Config = {
    'absorptionDict': {
        "Woolen": np.array([0.15, 0.20, 0.30, 0.45, 0.52, 0.40]),
        "Mat": np.array([0.15, 0.30, 0.40, 0.75, 0.85, 0.70]),
        "Plaster": np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20]),
        "Wood": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    },

    'scatterDict': {
        "Woolen": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Mat": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Plaster": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Wood": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    },
    # ====================================== Could Change Para ==========================================
    'fileDir': r"TestDict/Absorption/Simple_Test.3dm",
    'num_ray': 31500,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 5,
    'init_swl': 94,
    'end_rt_spl': 65,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': r'TestDict/Absorption/Abs_001',
    'volume': 315
}
