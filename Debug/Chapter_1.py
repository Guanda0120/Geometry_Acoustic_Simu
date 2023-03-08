import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# The Real Reverb Time of the sound field
reverb_time = np.asarray(
    [1.97, 1.74, 1.55, 1.81, 1.99, 2.28, 2.46, 2.26, 2.49, 2.50, 2.14, 1.96, 1.55, 1.40, 1.35, 1.30])
x = np.linspace(0, reverb_time.shape[0], reverb_time.shape[0])
# Upper Limit of Reverb Time
upper_limit = np.asarray(
    [1.56, 1.56, 1.38, 1.38, 1.38, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20])
# Lower Limit of Reverb Time
lower_limit = np.asarray([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.81, 0.81, 0.81, 0.72, 0.72])

plt.xlabel(r'频率（Hz）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.plot(x, reverb_time, '-o', color='firebrick', linewidth=0.7, label=r'实测结果', markersize=4)
plt.plot(x, upper_limit, ':d', color='grey', linewidth=0.7, label=r'标准上限', markersize=2)
plt.plot(x, lower_limit, ':s', color='grey', linewidth=0.7, label=r'标准下限', markersize=2)
plt.fill_between(x, upper_limit, lower_limit, facecolor='grey', alpha=0.2)
plt.yticks(fontsize = 12)
plt.xticks(x, ['125', '', '', '250', '', '', '500', '', '', '1k', '', '', '2k', '', '', '4k'], fontsize=12)
plt.grid()
plt.ylim(0, 3)
plt.legend(fontsize=12)
plt.show()
