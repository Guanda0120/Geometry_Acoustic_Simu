import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io.wavfile
from gasim.legacy.Acoustic.ReceiverGroup import ReceiverGroup
from gasim.legacy.Acoustic.Receiver import Receiver
from gasim.legacy.Acoustic.BaseAcoustic import BaseAcoustic


class ExportData:
    def __init__(self, rec_group: ReceiverGroup, export_folder: str):
        """

        :param rec_group:
        :param export_folder:
        """
        self.rec_group = rec_group
        os.makedirs(export_folder, exist_ok=True)
        self.export_folder = export_folder

    def _match_fs(self, rec: Receiver):
        # A Sample time duration
        fs = 44100
        # A Sample time duration
        delta_t = 1 / fs
        if len(rec.diffuse_ray_time) == 0:
            return np.zeros((0, 6), dtype=complex), np.zeros(0)
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
        file_name = f"rec_{int(rec_id)}.png"
        export_dir = os.path.join(self.export_folder, file_name)

        def spl_array(pressure):
            # Arrivals as (n, 6) sound pressure level, n may be 0
            return BaseAcoustic.pressure2spl(np.asarray(pressure).reshape(-1, 6))

        diffuse_res, diffuse_time = self._match_fs(rec)
        # (time, level, line colour, line width, marker size) from the weakest to the strongest component
        series = [(diffuse_time, spl_array(diffuse_res), 'grey', 0.1, 0.1),
                  (rec.stochastic_time, spl_array(rec.stochastic_pressure), 'g', 0.2, 0.4),
                  (rec.im_time, spl_array(rec.im_pressure), 'm', 0.3, 0.6)]
        if rec.direc_time is not None:
            series.append((rec.direc_time, spl_array(rec.direc_pressure), 'r', 2, 2.2))

        # Plot the Impulse Response
        # Chinese font set
        plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # Plot Data
        fig = plt.figure(figsize=(20, 15), dpi=300)
        for band, band_name in enumerate(['125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz']):
            plt.subplot(3, 2, band + 1)
            for tmp_time, tmp_res, colour, line_width, marker_size in series:
                if tmp_res.shape[0] == 0:
                    continue
                markerline, stemlines, baseline = plt.stem(np.atleast_1d(tmp_time), tmp_res[:, band], linefmt=colour)
                plt.setp(stemlines, 'linewidth', line_width)
                plt.setp(markerline, markersize=marker_size)

            plt.plot(rec.smooth_time_lr[:, band], rec.smooth_response_lr[:, band], 'c', linewidth=1)
            plt.plot(rec.smooth_time_bi[band], rec.smooth_response_bi[band], 'r', linewidth=1)
            plt.yticks(size=15)
            plt.xticks(size=15)
            plt.xlabel('时间（s）', fontsize=15)
            plt.ylabel('声压级（dB）', fontsize=15)
            plt.title(f'{band_name}衰变曲线', fontsize=15)
            plt.ylim(0, 100)
            plt.xlim(0, 1)
            plt.grid()

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.savefig(export_dir)
        plt.close(fig)
        print(f"File {export_dir} exports successfully!")

    def _record_data(self, rec: Receiver, rec_id: int):
        """
        Export a pkl file in the folder
        :param rec: A Receiver object
        :param rec_id: The index of the receiver
        :return: A pkl file inside the folder
        """
        # Create the file direction
        file_name = f"rec_{str(rec_id)}.pkl"
        export_dir = os.path.join(self.export_folder, file_name)
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
            raise ValueError('The index is out of receiver container number.')
        file_name = f"rec_{str(rec_id)}_res.wav"
        export_dir = os.path.join(self.export_folder, file_name)

        # Keep the real part of the impulse response, transform the Data type as np.float32
        sample = np.real(np.sum(self.rec_group.receiver_container[rec_id].fs_pressure, axis=1)).astype(np.float32)
        # Save the Impulse response
        scipy.io.wavfile.write(export_dir, 44100, sample)
        print(f"File {export_dir} exports successfully!")
        # Plot the Impulse Response
        # Chinese font set
        plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
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

        ir_file_name = f"rec_{str(rec_id)}_res.png"
        ir_export_dir = os.path.join(self.export_folder, ir_file_name)
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
            raise ValueError('The original file is not exist.')

        if rec_id > len(self.rec_group.receiver_container):
            raise ValueError('The index is out of receiver container number.')

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
        file_name = f"rec_{str(rec_id)}_conv.wav"
        output_file_dir = os.path.join(self.export_folder, file_name)
        scipy.io.wavfile.write(output_file_dir, ori_fs, wave_conv)
