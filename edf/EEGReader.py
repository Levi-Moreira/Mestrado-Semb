import numpy as np
import pyedflib

from edf.helper import verify_integrity, read_data, filter_data, normalize


class EEGReader:

    def read_file(self, file_path):
        print("Reading file: {}".format(file_path))
        eeg_file = pyedflib.EdfReader(file_path)
        channels_indexes = verify_integrity(eeg_file)
        result = read_data(eeg_file, channels_indexes)
        filtered_result = filter_data(result)
        normalized_result = normalize(filtered_result)
        return normalized_result.astype(dtype=np.float32)

    def read_file_in_interval(self, file_path, start, end):
        data = self.read_file(file_path)
        if end:
            interest_signal = data[:, start: end]
        else:
            interest_signal = data[:, start:]
        if end is None:
            end = len(data[0])
        return interest_signal
