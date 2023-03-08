import numpy as np
import matplotlib.pyplot as plt

# Use Chinese Require
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


arr = np.loadtxt(r"C:\\Users\12748\Desktop\MasterFinal\FigureInPaper\Chapter4\100NumRay.csv",
                 delimiter=",", dtype=str)
print(arr)
arr = arr[1:, 2:].astype(np.float64)

simu_arr = []
catt_arr = []
err_arr = []
print(arr)
for i in range(int(arr.shape[0]/3)):
    simu_arr.append(arr[3*i])
    catt_arr.append(arr[3*i+1])
    err_arr.append(arr[3*i+2])

simu_arr = np.array(simu_arr)
catt_arr = np.array(catt_arr)
x = np.array([0,1,2,3,4,5])

mean_simu = np.mean(simu_arr, axis=0)
mean_catt = np.mean(catt_arr, axis=0)
max_simu = np.max(simu_arr, axis=0)
min_simu = np.min(simu_arr, axis=0)
max_catt = np.max(catt_arr, axis=0)
min_catt = np.min(catt_arr, axis=0)
max_err = np.max(err_arr, axis=0)
min_err = np.min(err_arr, axis=0)


plt.fill_between(x, min_simu, max_simu, facecolor='r', alpha=0.3)
plt.fill_between(x, min_catt, max_catt,  facecolor='grey', alpha=0.3)

plt.plot(x, mean_simu, 'r', linewidth=1, label='本文算法')
plt.plot(x, mean_catt, 'grey', linewidth=1, label='CATT算法')

plt.xlabel('频率（Hz）', fontsize=12)
plt.ylabel('混响时间（s）', fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['125', '250', '500', '1k', '2k', '4k'],fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend()
plt.show()
plt.close()

plt.plot(x, np.mean(err_arr,axis=0), 'r', linewidth=1)
plt.fill_between(x, min_err, max_err,  facecolor='grey', alpha=0.2)
plt.xlabel('频率（Hz）', fontsize=12)
plt.ylabel('误差（s）', fontsize=12)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5], ['125', '250', '500', '1k', '2k', '4k'], fontsize=12)
plt.grid()
plt.show()
plt.close()

arr = np.loadtxt(r"C:\\Users\12748\Desktop\MasterFinal\FigureInPaper\Chapter4\NumRayCompare.csv", delimiter=",", dtype=str)
arr = arr[1:, 1:].astype(np.float64)
x = np.array([0, 1, 2, 3, 4, 5])
max_arr = np.max(arr, axis=0)
min_arr = np.min(arr, axis=0)

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
