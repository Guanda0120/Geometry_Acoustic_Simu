import numpy as np
import scipy
from Acoustic.BaseAcoustic import BaseAcoustic
import matplotlib.pyplot as plt

import matplotlib.pyplot as plot
import matplotlib as mpl


# Plot the signal read from wav file
import pickle
import time

with open(r'C:\Users\12748\Desktop\Simulation\TestDict\Absorption\Abs_099\rec_1.pkl', 'rb') as f:
    data = pickle.load(f)
import matplotlib.pyplot as plt
import matplotlib
st = time.time()
plt.plot(data.diffuse_ray_time,data.diffuse_ray_pressure, linewidth=0.1)
matplotlib.use('Agg')
# mpl.rcParams['path.simplify'] = True
#
# mpl.rcParams['path.simplify_threshold'] = 1.0
plt.savefig('1.png')
end=time.time()
print(end-st)


'''
fs_ratio = 2000
t = np.linspace(0, 1, fs_ratio)
spl = -60 * t + 60
noise = np.random.random(fs_ratio) * 10
spl += noise
pressure = BaseAcoustic.spl2pressure(spl)
h_t = (pressure ** 2)[::-1]
# spl_smooth = BaseAcoustic.pressure2spl(np.sqrt(np.cumsum(h_t)[::-1]))
spl_smooth = BaseAcoustic.pressure2spl(np.cumsum(h_t)[::-1])

plt.plot(t, spl, 'm', linewidth=0.1)
plt.plot(t, spl_smooth, 'c', linewidth=1.0)
plt.ylabel('SPL(dB)')
plt.xlabel('Time(s)')
plt.grid()
plt.show()
'''
