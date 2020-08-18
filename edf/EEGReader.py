import numpy as np
import pyedflib

from edf.constants import CHANNELS_NAMES, le_channel_transformer, ref_channel_transformer
from edf.helper import verify_integrity, read_data, filter_data, normalize, min_max_scale, guassian_smooth, \
    get_channel_conf
from plotter.drawing import plot_signal


class EEGReader:

    def read_file(self, file_path, start=None, end=None):
        print("Reading file: {}".format(file_path))

        eeg_file = pyedflib.EdfReader(file_path)
        channel_conf = get_channel_conf(eeg_file)

        result = np.zeros((len(CHANNELS_NAMES), eeg_file.getNSamples()[0]), dtype=np.float64)

        if channel_conf == "DIFF":
            channels_indexes = verify_integrity(eeg_file)

            result = read_data(eeg_file, channels_indexes)

        if channel_conf == "LE":
            for index, diff in enumerate(CHANNELS_NAMES):
                first, second = diff.split("-")
                labels = eeg_file.getSignalLabels()
                try:
                    channel_first = labels.index(le_channel_transformer(first))
                except ValueError:
                    channel_first = -1
                try:
                    channel_second = labels.index(le_channel_transformer(second))
                except ValueError:
                    channel_second = -1

                if channel_first == -1 and channel_second == -1:
                    continue

                if channel_first == -1 and channel_second != -1:
                    total_channel = eeg_file.readSignal(channel_second)
                if channel_second == -1 and channel_first != -1:
                    total_channel = eeg_file.readSignal(channel_first)

                if channel_first != -1 and channel_second != -1:
                    total_channel = eeg_file.readSignal(channel_first) - eeg_file.readSignal(channel_second)

                result[index, :] = total_channel
        if channel_conf == "REF":
            for index, diff in enumerate(CHANNELS_NAMES):
                first, second = diff.split("-")
                labels = eeg_file.getSignalLabels()
                try:
                    channel_first = labels.index(ref_channel_transformer(first))
                except ValueError:
                    channel_first = -1
                try:
                    channel_second = labels.index(ref_channel_transformer(second))
                except ValueError:
                    channel_second = -1

                if channel_first == -1 and channel_second == -1:
                    raise OSError

                if channel_first == -1:
                    total_channel = eeg_file.readSignal(channel_second)
                if channel_second == -1:
                    total_channel = eeg_file.readSignal(channel_first)

                if channel_first != -1 and channel_second != -1:
                    total_channel = eeg_file.readSignal(channel_first) - eeg_file.readSignal(channel_second)
                result[index, :] = total_channel
        # plot_signal(result[0], highlight=(start, end))

        # smoothed = guassian_smooth(result, 1)
        # plot_signal(smoothed[0], highlight=(start, end))

        # filtered_result = filter_data(result)
        # plot_signal(filtered_result[0], highlight=(start, end))

        normalized_result = normalize(result)
        # plot_signal(normalized_result[0], highlight=(start, end))
        # where_are_NaNs = np.isnan(normalized_result)
        # normalized_result[where_are_NaNs] = 0
        return normalized_result.astype(dtype=np.float32)

    def read_file_in_interval(self, file_path, start, end):
        data = self.read_file(file_path, start, end)
        if end:
            interest_signal = data[:, start: end]
        else:
            interest_signal = data[:, start:]
        if end is None:
            end = len(data[0])
        return interest_signal
