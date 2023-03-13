import numpy as np
import matplotlib.pyplot as plt

# wood = np.asarray([0.20, 0.32, 0.36, 0.21, 0.39, 0.27])
# wool = np.asarray([0.10, 0.25, 0.48, 0.46, 0.60, 0.64])
wood = np.asarray([0.21, 0.56, 0.71, 0.53, 0.63, 0.73])
agg = np.asarray([0.19, 0.18, 0.28, 0.25, 0.38, 0.36])

x = np.linspace(1, 6, 6)
freq = [125, 250, 500, '1k', '2k', '4k']
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.xlabel(r'频率（Hz）', fontsize=12)
plt.ylabel(r'吸声系数$\alpha$（-）', fontsize=12)
plt.plot(x, wood, '-o', color='firebrick', linewidth=1, label=r'吸声软包', markersize=4)
plt.plot(x, agg, '-d', color='grey', linewidth=1, label=r'聚砂喷涂', markersize=4)
# plt.plot(x, wool, '-d', color='grey', linewidth=0.7, label=r'人造皮软包', markersize=4)
plt.yticks(fontsize=12)
plt.xticks(x, freq, fontsize=12)
plt.grid()
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.show()
