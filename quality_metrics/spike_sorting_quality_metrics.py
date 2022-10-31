from sonpy import lib as son
import numpy as np
import math
import scipy.signal as sps
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

filepath = 'C:/Users/lukaz/OneDrive/Research/DBS/Python_DBS/Sample Data/clear_mua.smr'

file = son.SonFile(filepath, True)
channel = 1    # Channels are zero-indexed with Sonpy

read_max_time = file.ChannelMaxTime(channel) * file.GetTimeBase()
period = file.ChannelDivide(channel) * file.GetTimeBase()
num_points = math.floor(read_max_time / period)
fs = 1/period
times = np.arange(0, num_points * period, period)
raw_signal = file.ReadFloats(channel, num_points, 0)

b, a = sps.iirfilter(4, [300, 3000], btype='bandpass', ftype='butter', fs=fs)
filt_signal = sps.filtfilt(b, a, np.array(raw_signal))


## Spike sorting

threshold = 10 * np.median(np.abs(filt_signal))/0.6745
event_indices = sps.find_peaks(filt_signal, height=threshold)[0]
window_size = int(10 ** -3 * fs)
spike_waveforms = np.empty((1, 2*window_size))
event_indices = np.delete(event_indices, 0)    # Had to delete first spike because index was too close to 0
for i in event_indices:
    spike_buffer = filt_signal[int(i - window_size):int(i + window_size)]
    # Need to reshape, or else will flatten arrays before use.
    spike_waveforms = np.append(spike_waveforms, spike_buffer.reshape(1, 2*window_size), axis=0)
# Specify axis because its 2D array.
spike_waveforms = np.delete(spike_waveforms, 0, axis=0)
pca = PCA(n_components=2)
# Features will contain rows corresponding to principal components for each spike.
features = pca.fit_transform(spike_waveforms)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features)
cluster_labels = kmeans.labels_


## Amplitude Cutoff

def amplitude_cutoff(amplitudes,
                     num_histogram_bins = 500,
                     histogram_smoothing_value = 3):
    h,b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h,histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:])*bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing

# Will be applying this method to individual clusters to measure miss rate.
times_cluster_1 = times[event_indices[np.where(cluster_labels==0)[0]]]
times_cluster_2 = times[event_indices[np.where(cluster_labels==1)[0]]]
amplitudes_cluster_1 = filt_signal[event_indices[np.where(cluster_labels==0)[0]]]
amplitudes_cluster_2 = filt_signal[event_indices[np.where(cluster_labels==1)[0]]]
print('Miss rate cluster 1 (green):', amplitude_cutoff(amplitudes_cluster_1))
print('Miss rate cluster 2 (red):', amplitude_cutoff(amplitudes_cluster_2))

plt.figure(1)
plt.plot(times, filt_signal, 'cornflowerblue')
plt.plot(times_cluster_1, amplitudes_cluster_1, 'g.')
plt.plot(times_cluster_2, amplitudes_cluster_2, 'r.')
plt.show()


## Isolation Distance

def isolation_distance(all_pcs, all_labels, this_unit_id):
    pcs_for_this_unit = all_pcs[all_labels == this_unit_id,:]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit,0),0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError: # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value,
                       pcs_for_other_units,
                       'mahalanobis', VI = VI)[0])

    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]]) # number of spikes

    if n >= 2:

        isolation_distance = pow(mahalanobis_other[n-1],2)

    else:
        isolation_distance = np.nan

    return isolation_distance

iso_dist_cluster_1 = isolation_distance(
    all_pcs=features,
    all_labels=cluster_labels,
    this_unit_id=0
    )
iso_dist_cluster_2 = isolation_distance(
    all_pcs=features,
    all_labels=cluster_labels,
    this_unit_id=1
    )
print('Isolation distance cluster 1 (green):', iso_dist_cluster_1)
print('Isolation distance cluster 1 (red):', iso_dist_cluster_2)

pc_1_cluster_1 = features[np.where(cluster_labels == 0), 0]
pc_2_cluster_1 = features[np.where(cluster_labels == 0), 1]
pc_1_cluster_2 = features[np.where(cluster_labels == 1), 0]
pc_2_cluster_2 = features[np.where(cluster_labels == 1), 1]

plt.figure(2)
plt.plot(pc_1_cluster_1, pc_2_cluster_1, 'g.')
plt.plot(pc_1_cluster_2, pc_2_cluster_2, 'r.')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()