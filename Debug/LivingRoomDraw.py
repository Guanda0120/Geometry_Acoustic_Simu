import numpy as np
import matplotlib.pyplot as plt


arr = np.loadtxt(r"C:\\Users\\12748\\Desktop\\MasterFinal\\FigureInPaper\\Chapter4\\LivingRoom\\Livingroom.csv", delimiter=',', dtype=float)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
freq = np.asarray([0, 1, 2, 3, 4, 5])
plt.plot(freq, arr[3, :], '-.s', color='rosybrown',  linewidth=0.7, label=r'$R_{2}$ CATT')
plt.plot(freq, arr[4, :], '-o', color='firebrick', linewidth=0.7, label=r'$R_{2}$ 本文算法')
plt.plot(freq, arr[5, :], ':o', color='grey', linewidth=1, label=r'$R_{2}$ 误差')
plt.fill_between(freq, arr[3, :], arr[4, :], facecolor='grey', alpha=0.2)

plt.xlabel(r'频率（Hz）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.ylim(0,1)
plt.yticks(fontsize=12)
plt.xticks(freq, ['125', '250', '500', '1k', '2k', '4k'], fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()
