import numpy as np
import pydub


def sinc_match(nw: int, n: np.ndarray, epsilon: float):
    """

    :param nw:
    :param n:
    :param epsilon:
    :return:
    """
    container = []
    for n_item in n:
        if -nw / 2 < n_item < nw / 2:
            tmp = (np.cos(2 * np.pi * (n_item - epsilon) / nw) + 1) / 2 * np.sinc((n_item - epsilon))
            container.append(tmp)
        else:
            container.append(0)

    return np.asarray(container)


def wav_convert(input_dir: str, out_dir: str):
    sound = pydub.AudioSegment.from_mp3(input_dir)
    sound.export(out_dir, format='wav')


if __name__ == '__main__':
    '''
    import matplotlib.pyplot as plt
    a = np.linspace(0, 1, 10000)
    fs = 44100
    tao = 4e-4
    nw = 10
    dirac = sinc_match(nw, a, tao*fs)

    plt.plot(a, dirac)
    plt.grid()                        
    plt.show()
    '''
    '''
    from Acoustic.BaseAcoustic import BaseAcoustic
    import matplotlib.pyplot as plt
    t = np.linspace(0,0.1,1000)
    c = 343
    dist = c*t
    init_power_level = 100
    init_power = BaseAcoustic.swl2power(init_power_level)
    pressure = np.sqrt((init_power*c*1.224)/(4*np.pi*(dist**2)))
    spl = BaseAcoustic.pressure2spl(pressure)
    plt.plot(dist, spl)
    plt.xlabel('Distance(m)')
    plt.ylabel('SPL(dB)')
    plt.grid()
    plt.show()
    '''
    a = 0.01
    print(10*np.log10(a**2))
    print(20*np.log10(a))
