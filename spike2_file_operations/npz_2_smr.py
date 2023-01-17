from sonpy import lib as sp
import numpy as np


# ---------- Open numpy file ----------

read_filepath = ""    # Absolute path of numpy array file (*.npz) that will be read.

data = np.load(read_filepath)

spiketrain = data['spiketrain']    # Indexed by sample.
spikes = data['spikes']
raw_data = data['raw_data']
fs = data['fs']
t_stop = data['t_stop']
t_start = data['t_start']
channel_id = data['channel_id']


# ---------- Save to Spike2 datafile (.smr) ----------

save_filepath = ""    # Absolute path of new file that will be saved (include .smr extension).
new = sp.SonFile(save_filepath)


# ---------- Assigning new file parameters ----------

wave_offset = 0
wave_scale = 1
wave_time_base = 5*10**-6    # Units: s/tick
wave_Y_low = -5
wave_Y_high = 5
wave_units = 'V'
wave_title = f'E{channel_id}'
wave_channel_type = sp.DataType.Adc
wave_sample_ticks = int(1/(fs * wave_time_base))
time_start = t_start
wave_Fs = fs


# ---------- Setting new file parameters ----------
# Wave channel

new.SetTimeBase(wave_time_base)
new_wave_channel = channel_id - 1

new.SetWaveChannel(new_wave_channel, wave_sample_ticks, wave_channel_type, wave_Fs)
new.SetChannelTitle(new_wave_channel, wave_title)
new.SetChannelUnits(new_wave_channel, wave_units)
new.SetChannelScale(new_wave_channel, wave_scale)
new.SetChannelOffset(new_wave_channel, wave_offset)
new.SetChannelYRange(new_wave_channel, wave_Y_low, wave_Y_high)


# Spike events channel

spike_channel = 2
max_event_rate = 1/(wave_time_base * wave_sample_ticks)
spike_channel_type = sp.DataType.EventFall

new.SetEventChannel(spike_channel, max_event_rate, spike_channel_type)
new.SetChannelTitle(spike_channel, 'Spikes')
new.SetChannelUnits(spike_channel, wave_units)
new.SetChannelScale(spike_channel, wave_scale)
new.SetChannelOffset(spike_channel, wave_offset)
new.SetChannelYRange(spike_channel, wave_Y_low, wave_Y_high)
del new

# Write wave
reload = sp.SonFile(save_filepath, False)
raw_data = np.array(6553.6 * raw_data, dtype=int)
write_wave = reload.WriteInts(new_wave_channel, raw_data, time_start)
if write_wave < 0:
    print(f'Unable to write waveform.\nError code: {write_wave}.')
    print(sp.GetErrorString(write_wave))
    quit()
else:
    print('Waveform written successfully.')

# Write spikes
spike_ticks = spiketrain/(fs * wave_time_base)   # Units: ticks
spike_ticks_formatted = np.array(spike_ticks, dtype=np.int64)
write_train = reload.WriteEvents(spike_channel, spike_ticks_formatted)
if write_train < 0:
    print(f'Unable to write spike train.\nError Code: {write_train}.')
    print(sp.GetErrorString(write_train))
    quit()
else:
    print('Spike train written successfully.')

del reload