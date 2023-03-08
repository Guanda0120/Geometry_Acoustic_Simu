import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from Acoustic.BaseAcoustic import BaseAcoustic
from Acoustic.CONST import acoustic_speed, air_rho
'''
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

alpha = np.arange(0, 1, 0.001)
cos_theta = np.arange(0, 1, 0.001)
alpha, cos_theta = np.meshgrid(alpha, cos_theta)
r_0 = np.sqrt(1-alpha)
ksi_0 = (1+r_0)/(1-r_0)
r = (ksi_0 * cos_theta - 1) / (ksi_0 * cos_theta + 1)

surf = ax.plot_surface(alpha, cos_theta, r, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1, 1)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel(r'Absorption Coefficient $\alpha$', fontsize=10)
ax.set_ylabel('Cos Theta ', fontsize=10)
ax.set_zlabel(r'Reflection Coefficient r', fontsize=10)

plt.show()
'''


'''
dist = np.linspace(0, 100, 1000)
dist = dist[1:]
time = dist / 343
ang_freq = np.pi * 2 * 125
phase = np.exp(complex(0, 1) * (ang_freq * time - dist * ang_freq / 343))

source_swl = 94
source_power = BaseAcoustic.swl2power(source_swl)
intensity = source_power / (4 * np.pi * (dist ** 2)) * phase
pressure = np.sqrt(intensity * air_rho * acoustic_speed)

# spl = BaseAcoustic.pressure2spl(pressure)
dist /= 343
plt.plot(dist, np.real(pressure), 'c', linewidth=1)
plt.xscale('log')
plt.grid(True, which='both', linestyle='--')
plt.show()
'''
np.random.uniform()