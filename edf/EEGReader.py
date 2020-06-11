import numpy as np
import pyedflib

from constants import OLD_SAMPLE_RATE, EEG_SIGNAL_NUMBER


class EEGReader:
    ORIGINAL_SAMPLE_RATE = OLD_SAMPLE_RATE

    channels_names = ['FP1-F7',
                      'F7-T7',
                      'T7-P7',
                      'P7-O1',
                      'FP1-F3',
                      'F3-C3',
                      'C3-P3',
                      'P3-O1',
                      'FP2-F4',
                      'F4-C4',
                      'C4-P4',
                      'P4-O2',
                      'FP2-F8',
                      'F8-T8',
                      'T8-P8',
                      'P8-O2',
                      'FZ-CZ',
                      'CZ-PZ']

    def read_file(self, file_path):
        print("Reading file: {}".format(file_path))
        eeg_file = pyedflib.EdfReader(file_path)
        n = EEG_SIGNAL_NUMBER
        number_of_seconds = int(eeg_file.getNSamples()[0]) / self.ORIGINAL_SAMPLE_RATE
        intended_number_of_samples = int(number_of_seconds * self.ORIGINAL_SAMPLE_RATE)
        result = np.zeros((n, intended_number_of_samples), dtype=np.float64)

        channel_labels = eeg_file.getSignalLabels()

        channels_indexes = list()
        for chn in self.channels_names:
            if chn in channel_labels:
                channels_indexes.append(channel_labels.index(chn))

        output_index = 0

        if len(channels_indexes) < 18:
            raise OSError
        for i in channels_indexes:
            original_signal = eeg_file.readSignal(i)
            result[output_index, :] = original_signal
            output_index += 1

        normed = result
        del (result)
        normed = (normed - normed.mean(axis=1).reshape((normed.shape[0], 1))) / normed.std(axis=1).reshape(
            (normed.shape[0], 1))
        return normed.astype(dtype=np.float16)

    def read_file_in_interval(self, file_path, start, end):
        read_data = self.read_file(file_path)
        return read_data[:, start: end]
