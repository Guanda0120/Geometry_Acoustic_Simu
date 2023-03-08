from timeit import timeit
import numpy as np
import scipy


def normal2diffuse(normal_alpha: float, *args):
    """
    The object function use to solve the nonlinear equation.
    :param normal_alpha: The normal incidence absorption, the param to optimize.
    :param args:
    :return:
    """
    reflection = np.sqrt(1 - normal_alpha)
    diffuse_alpha = args[0]
    return (2 / (1 - reflection) - (1 - reflection) / 2 + 2 * np.log((1 - reflection) / 2)) * 8 * (
            (1 - reflection) / (1 + reflection)) ** 2 - diffuse_alpha


def fsolve_opt(diffuse_alpha: float):
    """

    :param diffuse_alpha:
    :return:
    """
    normal_alpha = scipy.optimize.fsolve(normal2diffuse, diffuse_alpha, args=diffuse_alpha)
    res = normal2diffuse(normal_alpha[0], diffuse_alpha)
    return normal_alpha, res


def de_opt(diffuse_alpha: np.ndarray):
    """

    :param diffuse_alpha:
    :return:
    """
    # bounds = scipy.optimize.Bounds([(0, 1)])
    bounds = [(0, 1)]
    normal_alpha = scipy.optimize.differential_evolution(normal2diffuse, bounds, args=(diffuse_alpha,))

    return normal_alpha.x, normal_alpha


def brentq_opt(diffuse_alpha: float):
    """

    :param diffuse_alpha:
    :return:
    """
    normal_alpha = scipy.optimize.brentq(normal2diffuse, 0, 1, args=(diffuse_alpha,))
    return normal_alpha


if __name__ == '__main__':
    # Fsolve Method
    x = fsolve_opt(0.5)
    print(x)
    print("================================")
    x = de_opt(0.5)
    print(x)
    print("================================")
    x = brentq_opt(0.5)
    print(x)
