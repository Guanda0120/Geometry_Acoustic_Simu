import pickle
import numpy as np
from Acoustic.Receiver import Receiver


with open('D:\\L2B\\rec_0.pkl', 'rb') as f:
    data = pickle.load(f)

    if isinstance(data, Receiver):
        print(f"Image Source Ray {len(data.im_ray)}")
        print(f"Stochastic Ray {len(data.stochastic_ray)}")
        print(f"Diffuse Ray {len(data.diffuse_ray_pressure)}")

with open('D:\\L2B\\rec_1.pkl', 'rb') as f:
    data = pickle.load(f)

    if isinstance(data, Receiver):
        print(f"Image Source Ray {len(data.im_ray)}")
        print(f"Stochastic Ray {len(data.stochastic_ray)}")
        print(f"Diffuse Ray {len(data.diffuse_ray_pressure)}")