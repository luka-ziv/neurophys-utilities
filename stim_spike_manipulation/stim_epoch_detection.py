from sonpy import lib as son
import numpy as np
import scipy.signal as sps
import math
import matplotlib.pyplot as plt

filepath = r"C:\Users\lukaz\OneDrive\Research\DBS\Python_DBS\Sample Data\2450-s2-240-std-plast.smr"
file = son.SonFile(filepath, True)
channel = 1    # Manually set because I know where the wave data is in this file.

read_max_time = file.ChannelMaxTime(channel) * file.GetTimeBase()
period = file.ChannelDivide(channel) * file.GetTimeBase()
num_points = math.floor(read_max_time / period)
fs = 1/period
time_points = np.arange(0, num_points * period, period)
signal = file.ReadFloats(channel, num_points, 0)

maximum = np.max(signal)
threshold = 0.9*maximum

peak_indices = sps.find_peaks(signal, height=threshold)[0]

freq_1 = 1    # Units: Hz
freq_2 = 100    # Units: Hz

epoch_indices = {}
epoch_id = 0
colour_labels = []

for index, i in enumerate(peak_indices):
    current = time_points[i]
    try:
        next_point = time_points[peak_indices[index + 1]]
    except IndexError:
        # no next_point defined, so triggers else condition, acts as final point.
        pass
    
    if next_point >= current + 0.95/freq_1 and next_point <= current + 1.05/freq_1:
        try:    # Accounts for peaks in middle of stim train.
            epoch_indices[epoch_id].append(i)
        # if no epoch defined (gives KeyError), means we are starting new one:
        except KeyError:    # Accounts for first peak in stim.
            epoch_indices[epoch_id] = [i]
            # Only take colours at start of valid epochs.
            colour_labels.append(freq_1)
    elif next_point >= current + 0.95/freq_2 and next_point <= current + 1.05/freq_2:
        try:
            epoch_indices[epoch_id].append(i)
        except KeyError:
            epoch_indices[epoch_id] = [i]
            colour_labels.append(freq_2)
    else:
        try:    # Accounts for last peaks in stim.
            epoch_indices[epoch_id].append(i)
        except KeyError:
            epoch_indices[epoch_id] = [i]
        epoch_id += 1


invalid_keys = [key for key, value in epoch_indices.items() if len(value) <= 1]
for i in invalid_keys:
    del epoch_indices[i]

all_indices = []
for key, value in epoch_indices.items():
    all_indices.append(value)

plt.plot(time_points, signal, 'cornflowerblue')
labels_height = 1.05*maximum
for i in range(len(all_indices)):
    epoch_duration = [np.min(time_points[all_indices[i]]), np.max(time_points[all_indices[i]])]
    if colour_labels[i] == 1:
        plt.plot(time_points[all_indices[i]], signal[all_indices[i]], 'r.')
        plt.plot([np.min(time_points[all_indices[i]]), np.max(time_points[all_indices[i]])], [labels_height, labels_height], 'k')
        midpoint = np.sum(epoch_duration)/2
        plt.annotate(f'E{i}', (midpoint, labels_height), textcoords='offset points', xytext=(0,3), ha='center')
    elif colour_labels[i] == 100:
        plt.plot(time_points[all_indices[i]], signal[all_indices[i]], 'g.')
        plt.plot([np.min(time_points[all_indices[i]]), np.max(time_points[all_indices[i]])], [labels_height, labels_height], 'k')
        midpoint = np.sum(epoch_duration)/2
        plt.annotate(f'E{i}', (midpoint, labels_height), textcoords='offset points', xytext=(0,3), ha='center')
    
plt.show()

