import os

from constants import POSITIVE_FOLDER_NAME, NEGATIVE_FOLDER_NAME, WINDOW_IN_SECONDS
from interface.BaseDatabaseGenerator import BaseDatabaseGenerator


class PositiveEEGDatasetGenerator(BaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)
        self.positive_files = self.explorer.positive_edf_files
        self.__generate_data_chunks(subject)

    def __generate_data_chunks(self, subject):
        for file in self.positive_files:
            file_path = os.path.join(os.getcwd(), *["data", "chb-mit-scalp-eeg-database-1.0.0", subject, file])
            if self.summary[file]["seizure_info"]:
                previous_end_time_sample = 0
                for info in self.summary[file]["seizure_info"]:
                    start_sample = info["start_time"] * self.reader.ORIGINAL_SAMPLE_RATE
                    end_sample = info["end_time"] * self.reader.ORIGINAL_SAMPLE_RATE
                    try:
                        data = self.reader.read_file_in_interval(file_path, start_sample, end_sample)
                        negative_data = self.reader.read_file_in_interval(file_path, previous_end_time_sample,
                                                                          start_sample)
                    except OSError:
                        print("Dropped file: {}".format(file))
                        continue
                    finally:
                        previous_end_time_sample = end_sample
                    cc = self.__get_positive_chunks_from_data(data)
                    nn = self.__get_negative_chunks_from_data(negative_data)
                    self.chunks.extend(cc)
                    self.extra_chunks.extend(nn)
            self.save_chunks(POSITIVE_FOLDER_NAME)
            self.save_extra_chunks(NEGATIVE_FOLDER_NAME)
            self.total_chuncks += len(self.chunks)
            self.total_extra_chunks += len(self.extra_chunks)
            self.chunks.clear()
            self.extra_chunks.clear()

    def __get_positive_chunks_from_data(self, data):
        chuncks = []
        window_start = 0
        while (window_start + self.CHUNCK_SIZE) < data.shape[1]:
            chuncks.append(data[:, window_start:window_start + self.CHUNCK_SIZE])
            # window_start += int(self.SHIFT_WINDOW * self.reader.ORIGINAL_SAMPLE_RATE)
            window_start += self.SHIFT_WINDOW
        return chuncks

    # data[:,0:1280
    # data[:,25,1280 + 25

    def __get_negative_chunks_from_data(self, data):
        chunks = []
        window_start = 0
        while (window_start + self.CHUNCK_SIZE) < data.shape[1]:
            chunks.append(data[:, window_start:window_start + self.CHUNCK_SIZE])
            window_start += int(WINDOW_IN_SECONDS * self.reader.ORIGINAL_SAMPLE_RATE)
        return chunks