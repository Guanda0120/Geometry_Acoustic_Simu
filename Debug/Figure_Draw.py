import random
import numpy as np
import matplotlib.pyplot as plt
from Acoustic.BaseAcoustic import BaseAcoustic
from AdaptiveDSP.RT_Process import RT_Process


def xyzSphere(n: int):
    """
    Plot n points onto the sphere. Using x, y, z coordinate.
    :param n: Number of point
    :return: Figure
    """
    print("XYZ Method!")
    # Generate Data
    rnd_pt = np.random.uniform(-1, 1, (n, 3))
    divide_scaler = np.tile(np.linalg.norm(rnd_pt, axis=1), (3, 1)).T
    rnd_pt = rnd_pt / divide_scaler

    # Plot Data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    marker_size = np.ones((n))
    ax.scatter(rnd_pt[:, 0], rnd_pt[:, 1], rnd_pt[:, 2], s=marker_size)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def polarSphere(n: int):
    """
    Plot n points onto the sphere. Using theta phi coordinate.
    :param n: Number of point
    :return: Figure
    """
    print("Solar Method!")
    random.seed(42)
    theta = np.random.uniform(0, 1, n) * np.pi * 2
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n))
    x = np.multiply(np.sin(phi), np.cos(theta))
    y = np.multiply(np.sin(phi), np.sin(theta))
    z = np.cos(phi)

    # Plot Data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    marker_size = np.ones((n))
    ax.scatter(x, y, z, s=marker_size)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

    return theta, phi


def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
    """

    :param x:
    :param y:
    :return:
    """
    marker_size = np.ones((x.shape))
    plt.scatter(x, y, s=marker_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.array([0, 0.5, 1, 1.5, 2]) * np.pi, ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    plt.yticks(np.array([0, 0.25, 0.5, 0.75, 1]) * np.pi, ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    plt.show()


def lsq_signal_gnerator(size: int):
    """

    :return:
    """
    # Use Chinese Require
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # Generate Signal
    t = np.linspace(0, 1, size)
    spl = -60 * t + 60 + np.random.random(size) * 10
    plt.plot(t, spl, linewidth=0.5, color='r')

    # LSQ Smooth
    '''
    pressure = BaseAcoustic.spl2pressure(spl)
    I_1 = np.sum(spl)
    I_2 = np.sum(np.multiply(spl, t))

    delta_1 = t[-1] - t[0]
    delta_2 = (t[-1] ** 2 - t[0] ** 2) / 2
    delta_3 = (t[-1] ** 3 - t[0] ** 3) / 3
    slope = (delta_1 * I_2 - delta_2 * I_1) / (delta_1 * delta_3 - delta_2 ** 2)
    lsq_smooth = t * slope+60
    plt.plot(t, lsq_smooth, linewidth=1, color='m')
    '''

    # Back Integral
    pressure = BaseAcoustic.spl2pressure(spl) ** 2

    plt.xlabel("时间t(s)")
    plt.ylabel("声压级SPL(dB)")
    plt.grid()
    plt.show()


def linear_reg():
    """

    :param res:
    :param time:
    :return:
    """
    shape = 1000
    time = np.linspace(0, 1, shape)
    res = -60 * time + 60 + np.random.random(shape) * 10

    bias = np.ones((shape, 1))
    # Add dim to the time
    time = time[:, np.newaxis]
    # Construct the X
    time_bias = np.hstack([time, bias])
    # Compute the coe matrix (X^{T}X)^{-1}X^{T}
    coe_matrix = np.dot(np.linalg.inv(np.dot(time_bias.T, time_bias)), time_bias.T)
    smooth = time * coe_matrix[0] + coe_matrix[1]
    omega = np.dot(coe_matrix, res)
    # The smooth decay curve
    smooth = np.dot(time_bias, omega)

    # Use Chinese Require
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.plot(time, res, linewidth=0.3, label='脉冲响应')
    plt.plot(time, smooth, linewidth=2, label='最小二乘法')
    plt.legend()
    plt.grid()
    plt.xlabel('时间（s）')
    plt.ylabel('声压级（dB）')
    plt.show()


def bi_method():
    def sigmoid(x):
        return 1.0 / (1 + np.exp((-x)))

    # Generate signal
    shape = 1000
    time = np.linspace(0, 3, shape)
    res = (1-sigmoid(time))*60 + np.random.random(shape)*5

    # Linreg
    bias = np.ones((shape, 1))
    # Add dim to the time
    time = time[:, np.newaxis]
    # Construct the X
    time_bias = np.hstack([time, bias])
    # Compute the coe matrix (X^{T}X)^{-1}X^{T}
    coe_matrix = np.dot(np.linalg.inv(np.dot(time_bias.T, time_bias)), time_bias.T)
    omega = np.dot(coe_matrix, res)
    # The smooth decay curve
    lin_smooth = np.dot(time_bias, omega)

    # BI
    pres_res = BaseAcoustic.spl2pressure(res)
    tmp_power_res = pres_res[::-1]**2
    # Get the smoothed sound pressure
    smooth_response = np.cumsum(tmp_power_res)
    # print(f"Back Integral: After Cumsum: {smooth_response}")
    smooth_response = np.sqrt(smooth_response[::-1])
    # smooth_response = np.dot(tmp_response[:, np.newaxis].T, (p_init * conv_m))
    # smooth_response = smooth_response.reshape(-1)
    # Turn the pressure to the sound pressure level
    bi_smooth = BaseAcoustic.pressure2spl(smooth_response)-20

    # Use Chinese Require
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.plot(time, res, linewidth=0.3, label='脉冲响应')
    plt.plot(time, lin_smooth, linewidth=2, label='最小二乘法')
    plt.plot(time, bi_smooth, linewidth=2, label='向后积分法')
    plt.legend()
    plt.grid()
    plt.xlabel('时间（s）')
    plt.xlim(0,2.5)
    plt.ylim(0,40)
    plt.ylabel('声压级（dB）')
    plt.show()

if __name__ == '__main__':
    bi_method()
    # lsq_signal_gnerator(400)
    """
    theta, phi = polarSphere(10000)
    plot_scatter(theta, phi, r"$\theta$", r"$\phi$")
    plot_scatter(np.random.uniform(0, 2 * np.pi, 10000), np.random.uniform(0, np.pi, 10000), r"$\theta$", r"$\phi$")
    xyzSphere(10000)
    """
