# Run settings ported from the thesis code (JianZuRenew_Config.py).
# Paths are relative to the repository root.
import numpy as np

Config = {
    'absorptionDict': {
        "Glass": np.array([0.03, 0.03, 0.02, 0.02, 0.01, 0.01]),
        "Absorber": np.array([0.30, 0.45, 0.35, 0.55, 0.55, 0.80]),
        "Door": np.array([0.12, 0.12, 0.11, 0.09, 0.08, 0.10]),
        "Ceil": np.array([0.05, 0.18, 0.30, 0.25, 0.25, 0.60]),
        "Floor": np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]),
        "Front": np.array([0.05, 0.20, 0.30, 0.25, 0.25, 0.60]),
        "Absorption_Plane": np.array([0.35, 0.45, 0.40, 0.55, 0.55, 0.90]),
    },

    'scatterDict': {
        "Glass": np.ones(6) * 0.10,
        "Absorber": np.ones(6) * 0.10,
        "Door": np.ones(6) * 0.10,
        "Ceil": np.ones(6) * 0.10,
        "Floor": np.ones(6) * 0.10,
        "Front": np.ones(6) * 0.10,
        "Absorption_Plane": np.ones(6) * 0.10
    },
    # ====================================== Could Change Para ==========================================
    'fileDir': 'models/private/JianZuTest.3dm',
    'num_ray': 20000,
    'rt_method': "T60",
    # ====================================== Default Para ==============================================
    'im_order': 4,
    'init_swl': 94,
    'end_rt_spl': 60,
    'receiver_radius': 0.2,
    'sample_freq': 44100,
    'opt_max_res': 0.20,
    'model_name': 'JCA',
    # ====================================== Export Direction ==========================================
    'output_folder': 'output/lecture_hall_jianzu_renew'
}
