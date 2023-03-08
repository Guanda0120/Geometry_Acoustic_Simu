import numpy as np
import matplotlib.pyplot as plt
from Acoustic.Receiver import Receiver
from Acoustic.BaseAcoustic import BaseAcoustic


def plot_res(rec: Receiver, output_folder: str, rec_id: int):
    """
    Plot Response data
    :param rec: A receiver Object
    :param output_folder: The output folder direction
    :param rec_id: the id
    :return: Save the figure
    """
    file_name = f"\\rec_{int(rec_id)}.png"
    export_dir = output_folder+file_name

    # Load Data
    direc_res = BaseAcoustic.pressure2spl(np.array(rec.direc_pressure))
    direc_time = rec.direc_time

    im_res = BaseAcoustic.pressure2spl(np.array(rec.im_pressure))
    im_time = rec.im_time

    stochas_res = BaseAcoustic.pressure2spl(np.array(rec.stochastic_pressure))
    stochas_time = rec.stochastic_time

    diffuse_res = BaseAcoustic.pressure2spl(np.array(rec.diffuse_ray_pressure))
    diffuse_time = rec.diffuse_ray_time

    lr_res = rec.smooth_response_lr
    lr_time = rec.smooth_time_lr
    lr_rt = rec.rt_time_lr

    bi_res = rec.smooth_response_bi
    bi_time = rec.smooth_time_bi
    bi_rt = rec.rt_time_bi

    fs_res = rec.fs_pressure
    fs_time = rec.fs_time

    # Plot Data
    fig = plt.figure(figsize=(20, 15), dpi=300)

    # Plot 125 Hz
    plt.subplot(3, 2, 1)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 0], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 0], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 0], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 0], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 0], lr_res[:, 0], 'c', linewidth=1)
    plt.plot(bi_time[0], bi_res[0], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 0], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'125 Hz Decay Curve, RT: {np.round(lr_rt[0], 2)}, {np.round(bi_rt[0], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    # Plot 250 Hz
    plt.subplot(3, 2, 2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 1], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 1], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 1], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 1], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 1], lr_res[:, 1], 'c', linewidth=1)
    plt.plot(bi_time[1], bi_res[1], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 1], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'250 Hz Decay Curve, RT: {np.round(lr_rt[1], 2)}, {np.round(bi_rt[1], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    # Plot 500 Hz
    plt.subplot(3, 2, 3)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 2], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 2], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 2], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 2], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 2], lr_res[:, 2], 'c', linewidth=1)
    plt.plot(bi_time[2], bi_res[2], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 2], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'500 Hz Decay Curve, RT: {np.round(lr_rt[2], 2)}, {np.round(bi_rt[2], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    # Plot 1k Hz
    plt.subplot(3, 2, 4)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 3], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 3], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 3], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 3], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 3], lr_res[:, 3], 'c', linewidth=1)
    plt.plot(bi_time[3], bi_res[3], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 3], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'1k Hz Decay Curve, RT: {np.round(lr_rt[3], 2)}, {np.round(bi_rt[3], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    # Plot 2k Hz
    plt.subplot(3, 2, 5)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 4], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 4], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 4], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 4], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 4], lr_res[:, 4], 'c', linewidth=1)
    plt.plot(bi_time[4], bi_res[4], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 4], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'2k Hz Decay Curve, RT: {np.round(lr_rt[4], 2)}, {np.round(bi_rt[4], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    # Plot 4k Hz
    plt.subplot(3, 2, 6)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 5], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.1)

    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 5], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.2)
    plt.setp(markerline, markersize=0.4)

    markerline, stemlines, baseline = plt.stem(im_time, im_res[:, 5], linefmt='m')
    plt.setp(stemlines, 'linewidth', 0.3)
    plt.setp(markerline, markersize=0.6)

    markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 5], linefmt='r')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=2.2)

    plt.plot(lr_time[:, 5], lr_res[:, 5], 'c', linewidth=1)
    plt.plot(bi_time[5], bi_res[5], 'r', linewidth=1)
    plt.plot(fs_time, fs_res[:, 5], 'b', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'4k Hz Decay Curve, RT: {np.round(lr_rt[5], 2)}, {np.round(bi_rt[5], 2)}', fontsize=10)
    plt.ylim(0, 120)
    plt.xlim(0, 1.2)
    plt.grid()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    # plt.show()
    plt.savefig(export_dir)
    print(f"File {export_dir} exports successfully!")

def plot_response(im_res: list, stochas_res: list, diffuse_res: list,
                  im_time: list, stochas_time: list, diffuse_time: list,
                  lr_res: list, lr_time: list, lr_rt,
                  bi_res: list, bi_time: list, bi_rt):
    """

    :param im_res:
    :param stochas_res:
    :param diffuse_res:
    :param im_time:
    :param stochas_time:
    :param diffuse_time:
    :param lr_res:
    :param lr_time:
    :param lr_rt:
    :param bi_res:
    :param bi_time:
    :param bi_rt:
    :return:
    """
    fig = plt.figure(figsize=(10, 15), dpi=300)

    stochas_res = np.array(stochas_res)
    diffuse_res = np.array(diffuse_res)

    # Plot 125 Hz
    plt.subplot(3, 2, 1)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 0], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 0], linefmt='g')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 0], linefmt='grey')
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 0], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 0], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'125 Hz Decay Curve, RT: {np.round(lr_rt[0], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    # Plot 250 Hz
    plt.subplot(3, 2, 2)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 1])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 1])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 1], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 1], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'250 Hz Decay Curve, RT: {np.round(lr_rt[1], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    # Plot 500 Hz
    plt.subplot(3, 2, 3)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 2])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 2])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 2], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 2], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'500 Hz Decay Curve, RT: {np.round(lr_rt[2], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    # Plot 1k Hz
    plt.subplot(3, 2, 4)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 3])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 3])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 3], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 3], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'1k Hz Decay Curve, RT: {np.round(lr_rt[3], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    # Plot 2k Hz
    plt.subplot(3, 2, 5)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 4])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 4])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 4], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 4], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'2k Hz Decay Curve, RT:{np.round(lr_rt[4], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    # Plot 4k Hz
    plt.subplot(3, 2, 6)
    markerline, stemlines, baseline = plt.stem(stochas_time, stochas_res[:, 5])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    markerline, stemlines, baseline = plt.stem(diffuse_time, diffuse_res[:, 5])
    plt.setp(stemlines, 'linewidth', 0.1)
    plt.setp(markerline, markersize=0.2)
    plt.plot(lr_time, lr_res[:, 5], 'c', linewidth=1)
    plt.plot(bi_time, bi_res[:, 5], 'r', linewidth=1)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Sound Pressure Level (dB)', fontsize=10)
    plt.title(f'2k Hz Decay Curve, RT: {np.round(lr_rt[5], 2)}', fontsize=10)
    plt.ylim(-20, 90)
    plt.grid()

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    plt.savefig('1.png')
