import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


excel_dir = r"C:\\Users\\12748\\Desktop\\MasterFinal\\FigureInPaper\\Chapter4\\1075_Theater\\Result.xlsx"
excel_frame = pd.read_excel(excel_dir, index_col=0, header=None)
data = excel_frame.to_numpy()

simu = []
catt = []
err = []
err_percent = []

for i in range(int(data.shape[0]/4)):
    catt.append(data[i*4, :])
    simu.append(data[i*4+1, :])
    err.append(data[i*4+2, :])
    err_percent.append(data[i*4+3, :])

catt = np.asarray(catt)
simu = np.asarray(simu)
err = np.asarray(err)
err_percent = np.asarray(err_percent)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

index = 5
freq = np.asarray([0, 1, 2, 3, 4, 5])
plt.plot(freq, catt[index, :], '-s', color='rosybrown',  linewidth=0.7, label=f'$R_{index+1}$ CATT')
plt.plot(freq, simu[index, :], '-o', color='firebrick', linewidth=0.7, label=f'$R_{index+1}$ 本文算法')
plt.plot(freq, err[index, :], ':d', color='grey', linewidth=1, label=f'$R_{index+1}$ 误差')
plt.fill_between(freq, simu[index, :], catt[index, :], facecolor='grey', alpha=0.2)

plt.xlabel(r'频率（Hz）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.ylim(0,2.0)
plt.yticks(fontsize=12)
plt.xticks(freq, ['125', '250', '500', '1k', '2k', '4k'], fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()