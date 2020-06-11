import os

from constants import WINDOW_IN_SECONDS, NEGATIVE_FOLDER_NAME
from interface.BaseDatabaseGenerator import BaseDatabaseGenerator


class NegativeEEGDatasetGenerator(BaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)
        self.negative_files = self.explorer.negative_edf_files
        self.__generate_data_chuncks(subject)

    def __generate_data_chuncks(self, subject):
        for file in self.negative_files:
            file_path = os.path.join(os.getcwd(), *["data", "chb-mit-scalp-eeg-database-1.0.0", subject, file])
            try:
                data = self.reader.read_file(file_path)
            except OSError:
                print("Dropped file: {}".format(file))
                continue

            window_start = 0
            while (window_start + self.CHUNCK_SIZE) < data.shape[1]:
                self.chunks.append(data[:, window_start:window_start + self.CHUNCK_SIZE])
                window_start += int(WINDOW_IN_SECONDS * self.reader.ORIGINAL_SAMPLE_RATE)

            self.save_chunks(NEGATIVE_FOLDER_NAME)
            self.total_chuncks += len(self.chunks)
            self.chunks.clear()
