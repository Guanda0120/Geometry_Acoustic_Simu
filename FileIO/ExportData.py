import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab
import scipy
from Acoustic.ReceiverGroup import ReceiverGroup
from Acoustic.Receiver import Receiver
from Acoustic.BaseAcoustic import BaseAcoustic


class ExportData:
    def __init__(self, rec_group: ReceiverGroup, export_folder: str):
        """

        :param rec_group:
        :param export_folder:
        """
        self.rec_group = rec_group
        if os.path.exists(export_folder):
            self.export_folder = export_folder
        else:
            ValueError(f'The output folder {export_folder} do not exist')

    def _match_fs(self, rec: Receiver):
        # A Sample time duration
        fs = 44100
        # A Sample time duration
        delta_t = 1 / fs
        mod = np.mod(rec.diffuse_ray_time, delta_t)
        time_index = np.floor_divide(rec.diffuse_ray_time, delta_t)

        # If the mod is 0,
        time_index = np.where(mod != 0, time_index, time_index - 1)
        time_index = time_index.astype('int64')
        # Shift the sound pressure
        shift_t = delta_t - mod
        shift_pressure = BaseAcoustic.pressure_shift(np.array(rec.diffuse_ray_pressure), shift_t)

        pressure_merge = np.zeros((int(max(time_index)) + 1, 6), dtype=complex)

        # Create a dictionary
        pressure_dict = {}
        for i in range(max(time_index) + 1):
            pressure_dict[str(i)] = []

        for t_idx in range(len(time_index)):
            pressure_dict[str(time_index[t_idx])].append(shift_pressure[t_idx])

        for key in pressure_dict.keys():
            tmp_array = np.array(pressure_dict[key])
            tmp_array = BaseAcoustic.pressure_addition(tmp_array, add_method='rms')
            pressure_merge[int(key)] = tmp_array

        time_merge = np.array(range(len(pressure_merge) + 1)) * delta_t
        pressure_merge = np.concatenate((np.zeros((1, 6), dtype=complex), pressure_merge))

        return_time = []
        return_pressure =[]
        for idx in range(pressure_merge.shape[0]):
            if np.abs(pressure_merge[idx][0]) != 0:
                return_time.append(time_merge[idx])
                return_pressure.append(pressure_merge[idx,:])

        return_time = np.asarray(return_time)
        return_pressure = np.asarray(return_pressure)

        return return_pressure, return_time

    def _plot_res(self, rec: Receiver, rec_id: int):
        """
        Hidden plot method, plot a png to the folder
        :param rec: a Receiver object
        :param rec_id: The rec id
        :return: A png file
        """
        matplotlib.use('Agg')
        file_name = f"\\rec_{int(rec_id)}.png"
        export_dir = self.export_folder + file_name

        # Load Data
        direc_res = BaseAcoustic.pressure2spl(np.array(rec.direc_pressure))
        direc_time = rec.direc_time

        im_res = BaseAcoustic.pressure2spl(np.array(rec.im_pressure))
        im_time = rec.im_time

        stochas_res = BaseAcoustic.pressure2spl(np.array(rec.stochastic_pressure))
        stochas_time = rec.stochastic_time

        diffuse_res, diffuse_time = self._match_fs(rec)
        diffuse_res = BaseAcoustic.pressure2spl(diffuse_res)

        lr_res = rec.smooth_response_lr
        lr_time = rec.smooth_time_lr
        lr_rt = rec.rt_time_lr

        bi_res = rec.smooth_response_bi
        bi_time = rec.smooth_time_bi
        bi_rt = rec.rt_time_bi

        fs_res = BaseAcoustic.pressure2spl(rec.fs_pressure)
        fs_time = rec.fs_time

        # Plot the Impulse Response
        # Chinese font set
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 0], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 0], lr_res[:, 0], 'c', linewidth=1)
        plt.plot(bi_time[0], bi_res[0], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 0], 'b', linewidth=0.2)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'125Hz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 1], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 1], lr_res[:, 1], 'c', linewidth=1)
        plt.plot(bi_time[1], bi_res[1], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 1], 'b', linewidth=0.1)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'250Hz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 2], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 2], lr_res[:, 2], 'c', linewidth=1)
        plt.plot(bi_time[2], bi_res[2], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 2], 'b', linewidth=0.1)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'500Hz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 3], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 3], lr_res[:, 3], 'c', linewidth=1)
        plt.plot(bi_time[3], bi_res[3], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 3], 'b', linewidth=0.1)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'1kHz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 4], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 4], lr_res[:, 4], 'c', linewidth=1)
        plt.plot(bi_time[4], bi_res[4], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 4], 'b', linewidth=0.1)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'2kHz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
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

        if rec.direc_time is not None:
            markerline, stemlines, baseline = plt.stem(direc_time, direc_res[:, 5], linefmt='r')
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(markerline, markersize=2.2)

        plt.plot(lr_time[:, 5], lr_res[:, 5], 'c', linewidth=1)
        plt.plot(bi_time[5], bi_res[5], 'r', linewidth=1)
        # plt.plot(fs_time, fs_res[:, 5], 'b', linewidth=0.1)
        plt.yticks(size=15)
        plt.xticks(size=15)
        plt.xlabel('时间（s）', fontsize=15)
        plt.ylabel('声压级（dB）', fontsize=15)
        plt.title(f'4kHz衰变曲线', fontsize=15)
        plt.ylim(0, 100)
        plt.xlim(0, 1)
        plt.grid()

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # plt.show()
        plt.savefig(export_dir)
        print(f"File {export_dir} exports successfully!")

    def _record_data(self, rec: Receiver, rec_id: int):
        """
        Export a pkl file in the folder
        :param rec: A Receiver object
        :param rec_id: The index of the receiver
        :return: A pkl file inside the folder
        """
        # Create the file direction
        file_name = f"\\rec_{str(rec_id)}.pkl"
        export_dir = self.export_folder + file_name
        # Open the file
        data_output = open(export_dir, 'wb')
        # Remove Rhino3dm object
        rec.sound_field = None
        for i in rec.im_ray:
            i.sound_field = None
        # Write the object
        pickle.dump(rec, data_output)
        # Close the file
        data_output.close()
        print(f"File {export_dir} exports successfully!")

    def save_plot(self):
        """

        :return:
        """
        for i in range(len(self.rec_group.receiver_container)):
            self._plot_res(self.rec_group.receiver_container[i], i)

    def save_pkl(self):
        """

        :return:
        """
        for i in range(len(self.rec_group.receiver_container)):
            self._record_data(self.rec_group.receiver_container[i], i)

    def _export_ir(self, rec_id: int):
        """
        Save impulse response as a wav file and spectrum of ir
        :param rec_id: The receiver index in receiver group
        :return: A wav file, and spectrum of ir
        """

        if rec_id > len(self.rec_group.receiver_container):
            ValueError('The index is out of receiver container number.')
        file_name = f"\\rec_{str(rec_id)}_res.wav"
        export_dir = self.export_folder + file_name

        # Keep the real part of the impulse response, transform the Data type as np.float32
        sample = np.real(np.sum(self.rec_group.receiver_container[rec_id].fs_pressure, axis=1)).astype(np.float32)
        # Save the Impulse response
        scipy.io.wavfile.write(export_dir, 44100, sample)
        print(f"File {export_dir} exports successfully!")
        matplotlib.use('Agg')
        # Plot the Impulse Response
        # Chinese font set
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.subplot(211)
        plt.title('脉冲响应', fontsize=25)
        plt.plot(sample, linewidth=0.3)
        plt.ylabel('响应幅值', fontsize=25)
        plt.grid()
        plt.yticks(size=25)
        plt.xticks(size=25)
        vmin = 20 * np.log10(np.max(sample)) - 100
        plt.subplot(212)
        pxx, freq, t, cax = plt.specgram(sample, Fs=44100, cmap='magma', vmin=vmin, mode='magnitude')
        plt.xlabel('时间（s）', fontsize=25)
        plt.ylabel('频率（Hz）', fontsize=25)
        # plt.yscale('log')
        plt.ylim((0, 20000))
        plt.yticks(size=25)
        plt.xticks(size=25)
        plt.colorbar(cax).set_label('强度 (dB)')

        ir_file_name = f"\\rec_{str(rec_id)}_res.png"
        ir_export_dir = self.export_folder + ir_file_name
        plt.savefig(ir_export_dir)
        plt.close()
        print(f"File {ir_export_dir} exports successfully!")

    def save_ir(self):
        """

        :return:
        """
        for i in range(len(self.rec_group.receiver_container)):
            self._export_ir(i)

    def conv_audio(self, ori_audio_dir: str, rec_id: int):
        """

        :param ori_audio_dir:
        :param rec_id:
        :return:
        """
        if not os.path.exists(ori_audio_dir):
            ValueError('The original file is not exist.')

        if rec_id > len(self.rec_group.receiver_container):
            ValueError('The index is out of receiver container number.')

        # Read the audio file
        ori_fs, ori_wave = scipy.io.wavfile.read(ori_audio_dir)

        # Check the audio fs is 44100
        assert ori_fs == 44100
        # If the wave is two chanel, merge it
        if len(ori_wave.shape) != 1:
            ori_wave = np.average(ori_wave, axis=1)

        # Keep the real part of the impulse response
        sample = np.real(np.sum(self.rec_group.receiver_container[rec_id].fs_pressure, axis=1))
        # Convolve 1D, transform the Data type as np.float32
        wave_conv = np.convolve(ori_wave, sample, mode='full').astype(np.float32)

        # Save the file
        file_name = f"\\rec_{str(rec_id)}_conv.wav"
        output_file_dir = self.export_folder + file_name
        scipy.io.wavfile.write(output_file_dir, ori_fs, wave_conv)
