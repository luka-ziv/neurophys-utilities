from sonpy import lib as son
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sps

filepath = "C:/Users/lukaz/OneDrive/Research/DBS/Python_DBS/Sample Data/2450-s2-240-std-plast.smr"
file = son.SonFile(filepath, True)
channel = 1    # Manually set because I know where the wave data is in this file.

read_max_time = file.ChannelMaxTime(channel) * file.GetTimeBase()
period = file.ChannelDivide(channel) * file.GetTimeBase()
num_points = math.floor(read_max_time / period)
fs = 1/period
times = np.arange(0, num_points * period, period)
signal = file.ReadFloats(channel, num_points, 0)

threshold = 0.75 * np.max(signal)
min_peak_samples = round(3/1000 * fs)
peak_indices = sps.find_peaks(signal, height=threshold, distance=min_peak_samples)[0]

window_size = 5    # This is samples, but input should be ms converted to samples.


# Method 1: subtract average wave

def subtract_stims(magnitudes, peak_indices, size):
    empty_signal = np.empty(0)
    magnitudes = np.append(empty_signal, magnitudes)
    total_sampled = np.zeros((1, 2*size))
    for i in peak_indices:
        window = magnitudes[i - size: size + i]
        total_sampled = np.append(total_sampled, window.reshape(1, 2*size), axis=0)
    # Delete first columns of zeros used for initialization.
    total_sampled = np.delete(total_sampled, 0, axis=0)
    mean_wave = np.mean(total_sampled, axis=0)
    for i in peak_indices:
        magnitudes[i - size: i + size] -= mean_wave
        plt.plot(times[i - size: size + i], mean_wave, 'g')
    return magnitudes


plt.figure(1)
plt.plot(times, signal, 'b')
new_signal = subtract_stims(signal, peak_indices, size=window_size)

plt.plot(times, new_signal, 'r')
plt.title('Mean Subtracted Stim Artifact')
plt.show()


# Method 2: linearize removed window

def linearize_stims(magnitudes, times, peak_indices, size):
    empty_signal = np.empty(0)
    magnitudes = np.append(empty_signal, magnitudes)
    for i in peak_indices:
        magnitudes_win = magnitudes[i - size: i + size]
        times_win = times[i - size: i + size]
        slope = (magnitudes_win[-1] - magnitudes_win[0]) / (times_win[-1] - times_win[0])
        linearized = [slope * j + magnitudes_win[0] for j in times[: 2*size]]
        magnitudes[i - size: i + size] = linearized
    return magnitudes


plt.figure(2)
plt.plot(times, signal, 'b')

new_signal = linearize_stims(signal, times, peak_indices, size=window_size)

plt.plot(times, new_signal, 'r')
plt.title('Linearized Stim Artifact')
plt.show()


# Method 3: Gaussian noise interpolation

def gaussian_noise_stims(magnitudes, times, peak_indices, size):
    empty_signal = np.empty(0)
    magnitudes = np.append(empty_signal, magnitudes)
    for i in peak_indices:
        pre_stim_window = magnitudes[i - size - 50: i - size]
        mean = np.mean(pre_stim_window)
        sd = np.std(pre_stim_window)
        gauss_noise = np.random.normal(loc=mean, scale=sd, size=2*size)
        magnitudes[i - size: i + size] = gauss_noise
    return magnitudes


plt.figure(3)
plt.plot(times, signal, 'b')

new_signal = gaussian_noise_stims(signal, times, peak_indices, window_size)

plt.plot(times, new_signal, 'r')
plt.title('Gaussian Noise Stim Artifact')
plt.show()