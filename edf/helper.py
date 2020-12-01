import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import minmax_scale

from dataset.constants import EEG_SIGNAL_NUMBER, DATASET_SAMPLE_RATE
from edf.constants import CHANNELS_NAMES, CUT_OFF_LOWER, CUT_OFF_HIGHER


def normalize(input_signal):
    return (input_signal - input_signal.mean(axis=1).reshape((input_signal.shape[0], 1))) / input_signal.std(
        axis=1).reshape(
        (input_signal.shape[0], 1))


def filter_data(input_signal):
    from frequency_splitter import butter_bandpass_filter
    filtered = butter_bandpass_filter(input_signal, CUT_OFF_LOWER, CUT_OFF_HIGHER, DATASET_SAMPLE_RATE)
    return filtered


def get_channel_conf(eeg_file):
    channel_labels = eeg_file.getSignalLabels()
    if "LE" in channel_labels[0]:
        return "LE"
    if "REF" in channel_labels[0]:
        return "REF"
    return "DIFF"


def verify_integrity(eeg_file):
    channel_labels = eeg_file.getSignalLabels()
    channels_indexes = list()
    for chn in CHANNELS_NAMES:
        if chn in channel_labels:
            channels_indexes.append(channel_labels.index(chn))

    if len(channels_indexes) < 18:
        raise OSError
    return channels_indexes


def read_data(eeg_file, channels_indexes):
    n = EEG_SIGNAL_NUMBER
    number_of_seconds = int(eeg_file.getNSamples()[0]) / DATASET_SAMPLE_RATE
    intended_number_of_samples = int(number_of_seconds * DATASET_SAMPLE_RATE)

    result = np.zeros((n, intended_number_of_samples), dtype=np.float64)

    output_index = 0
    for i in channels_indexes:
        original_signal = eeg_file.readSignal(i)
        result[output_index, :] = original_signal
        output_index += 1
    return result


def min_max_scale(data):
    return minmax_scale(data, feature_range=(-1, 1))


def guassian_smooth(data, order=1):
    return gaussian_filter1d(data, order)
