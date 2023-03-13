import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def saibin_eq(volume, surface, rt):
    return 1/(rt/(0.161*volume)*surface)


if __name__ == '__main__':
    rt = np.asarray([1.38, 1.40, 1.83, 1.25, 1.10, 0.94])
    print(saibin_eq(4389, 1981, rt))

    excel_frame = pd.read_excel(r'C:\Users\12748\Desktop\MasterFinal\FigureInPaper\Chapter5\After_Simu.xlsx',
                                index_col=0, header=0)
    data = excel_frame.to_numpy()
    data = data[:, :6]
    x = np.linspace(1, 6, 6)
    freq = [125, 250, 500, '1k', '2k', '4k']
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # plt.plot(x, data[0, :], '-o', color='', linewidth=0.7, label=r'实测数据', markersize=5)
    plt.plot(x, data[11, :], '-d', color='chocolate', linewidth=0.7, label=r'改造后实测', markersize=5)
    plt.plot(x, data[6, :], '-d', color='firebrick', linewidth=0.7, label=r'改造后模拟', markersize=5)

    # plt.plot(x, data[8, :], '-o', color='chocolate', linewidth=0.7, label=r'改造前实测', markersize=5)
    plt.plot(x, data[9, :], '--', color='grey', linewidth=0.7, label=r'标准上限', markersize=5)
    plt.plot(x, data[10, :], '--', color='grey', linewidth=0.7, label=r'标准下限', markersize=5)
    plt.plot(x, data[12, :], '-o', color='grey', linewidth=0.7, label=r'模拟误差', markersize=5)
    # plt.fill_between(x, np.max(data[:6, :], axis=0), np.min(data[:6, :], axis=0), facecolor='red', alpha=0.2)
    plt.fill_between(x, data[9, :], data[10, :], facecolor='grey', alpha=0.2)
    plt.xlabel(r'频率（Hz）', fontsize=12)
    plt.ylabel(r'混响时间（s）', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(x, freq, fontsize=12)
    plt.grid()
    plt.ylim(0, 2)
    plt.legend(fontsize=12)
    plt.show()