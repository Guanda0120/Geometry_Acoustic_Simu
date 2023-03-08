import scipy
import numpy as np
from Acoustic.Receiver import Receiver


def conv_audio(rec: Receiver, read_file_dir: str, output_file_dir: str):
    # Read the audio file
    fs, wave = scipy.io.wavfile.read(read_file_dir)
    # Check the audio fs is 44100
    assert fs == 44100
    # If the wave is two chanel, merge it
    if len(wave.shape) != 1:
        wave = np.average(wave, axis=1)
    # Keep the real part of the impulse response
    sample = np.real(np.sum(rec.fs_pressure, axis=1))
    # Convolve 1D, transform the Data type as np.float32
    wave_conv = np.convolve(wave, sample, mode='full').astype(np.float32)
    # Save the file
    scipy.io.wavfile.write(output_file_dir, fs, wave_conv)


def save_ir(rec: Receiver, out_file_dir: str):
    # Keep the real part of the impulse response, transform the Data type as np.float32
    sample = np.real(np.sum(rec.fs_pressure, axis=1)).astype(np.float32)
    # Save the Impulse response
    scipy.io.wavfile.write(out_file_dir, 44100, sample)
