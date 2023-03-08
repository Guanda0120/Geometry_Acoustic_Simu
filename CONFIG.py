import numpy as np


Config = {
    'absorptionDict': {
        "Woolen": np.array([0.15, 0.20, 0.30, 0.45, 0.52, 0.40]),
        "Mat": np.array([0.15, 0.20, 0.30, 0.45, 0.45, 0.40]),
        "Plaster": np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20]),
        "Wood": np.array([0.10, 0.20, 0.10, 0.20, 0.30, 0.40])
    },

    'scatterDict': {
        "Woolen": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Mat": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Plaster": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        "Wood": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    },
    # ====================================== Could Change Para ==========================================
    'fileDir': r"C:\\Users\\12748\\Desktop\\Simulation\\Test_File\\Real_Work.3dm",
    'num_ray': 315,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 3,
    'init_swl': 94,
    'end_rt_spl': 65,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': r'C:\\Users\\12748\\Desktop\\Simulation\\TestDict\\Shape\\RealWork'
}
