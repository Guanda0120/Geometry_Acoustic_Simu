import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

excel_frame = pd.read_excel(r'C:\Users\12748\Desktop\MasterFinal\FigureInPaper\Chapter5\Test_2_Simu.xlsx',index_col=0, header=0)
data = excel_frame.to_numpy()

x = np.linspace(1, 6, 6)
freq = [125, 250, 500, '1k', '2k', '4k']
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.plot(x, data[0, :], '-o', color='grey', linewidth=0.7, label=r'实测数据', markersize=5)
plt.plot(x, data[7, :], '-d', color='firebrick', linewidth=0.7, label=r'算法模拟', markersize=5)
plt.fill_between(x, np.max(data[1:-2, :], axis=0), np.min(data[1:-2, :], axis=0), facecolor='grey', alpha=0.2)
plt.xlabel(r'频率（Hz）', fontsize=12)
plt.ylabel(r'混响时间（s）', fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(x, freq, fontsize=12)
plt.grid()
plt.ylim(0, 3)
plt.legend(fontsize=12)
plt.show()
