import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# arr = np.loadtxt(r"C:\\Users\\12748\\Desktop\\MasterFinal\\FigureInPaper\\Chapter4\\Abs_Res.csv", delimiter=",", dtype=float)
arr = np.loadtxt(r"C:\\Users\12748\Desktop\MasterFinal\FigureInPaper\Chapter4\Sca_Res.csv", delimiter=",", dtype=float)
# arr = arr[:, 1:].astype(np.float64)
arr.astype(np.float64)
catt_simu = []
mine_simu = []
res = []
for i in range(int(arr.shape[0] / 3)):
    catt_simu.append(arr[i * 3, :])
    mine_simu.append(arr[i * 3 + 1, :])
    res.append(arr[i * 3 + 2, :])

catt_simu = np.asarray(catt_simu)
mine_simu = np.asarray(mine_simu)
res = np.asarray(res)
print(np.average(res,axis=0))
freq_idx = 5
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
abs_coe = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
freq = np.asarray([0, 1, 2, 3, 4, 5])
plt.plot(abs_coe, catt_simu[::-1, freq_idx], '-', linewidth=1, label='CATT算法')
plt.plot(abs_coe, mine_simu[::-1, freq_idx], '-', linewidth=1, label='模拟算法')
plt.fill_between(abs_coe, catt_simu[::-1, freq_idx], mine_simu[::-1, freq_idx], facecolor='grey', alpha=0.2)
plt.xlabel(r'吸声系数$\alpha$(-)', fontsize=12)
# plt.xlabel(r'散射系数s（-）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.ylim(0,1.2)
plt.yticks(fontsize=12)
plt.xticks(abs_coe, ['0.01', '0.02', '0.05', '0.10', '0.20', '0.50', '0.90', '0.99'], fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()
plt.close()

'''
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(abs_coe, freq, res, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel(r'Absorption Coefficient $\alpha$', fontsize=10)
ax.set_ylabel('Cos Theta ', fontsize=10)
ax.set_zlabel(r'Reflection Coefficient r', fontsize=10)

plt.show()
'''

'''
plt.plot(x, arr[0], ':', linewidth=1, label='1倍体积')
plt.plot(x, arr[1], ':', linewidth=1, label='2倍体积')
plt.plot(x, arr[2], ':', linewidth=1, label='5倍体积')
plt.plot(x, arr[3], '--', linewidth=1, label='10倍体积')
plt.plot(x, arr[4], '--', linewidth=1, label='20倍体积')
plt.plot(x, arr[5], '-', linewidth=1, label='50倍体积')
plt.plot(x, arr[6], '-', linewidth=1, label='100倍体积')
plt.grid()
plt.fill_between(x, min_arr, max_arr,  facecolor='grey', alpha=0.2)
plt.xticks([0, 1, 2, 3, 4, 5], ['125', '250', '500', '1k', '2k', '4k'], fontsize=12)
plt.xlabel('频率（Hz）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()
'''
