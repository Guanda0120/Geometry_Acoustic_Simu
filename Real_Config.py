import numpy as np

Config = {
    'absorptionDict': {
        "Door": np.array([0.15, 0.20, 0.09, 0.08, 0.17, 0.16]),
        "Floor": np.array([0.12, 0.18, 0.31, 0.35, 0.38, 0.36]),
        "Wood": np.array([0.10, 0.57, 0.73, 0.76, 0.70, 0.68]),
        "Glass": np.array([0.07, 0.08, 0.04, 0.03, 0.06, 0.05]),
        "Led": np.array([0.10, 0.12, 0.07, 0.05, 0.06, 0.06]),
        "Woolen": np.array([0.15, 0.20, 0.30, 0.45, 0.52, 0.40]),
        "Col": np.array([0.03, 0.04, 0.04, 0.03, 0.03, 0.01]),
        "White": np.array([0.15, 0.15, 0.09, 0.09, 0.17, 0.16]),
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
    'fileDir': r"C:\\Users\\12748\\Desktop\\Simulation\\Test_File\\Real_Work.3dm",
    'num_ray': 10000,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 3,
    'init_swl': 94,
    'end_rt_spl': 60,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': r'C:\\Users\\12748\\Desktop\\Simulation\\TestDict\\Shape\\RealWork'
}
