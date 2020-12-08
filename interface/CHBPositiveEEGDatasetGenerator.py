import os

from dataset.constants import DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, ORIGINAL_DATASET_LOCATION, \
    POSITIVE_FOLDER_NAME, NEGATIVE_FOLDER_NAME, DATASET_SAMPLE_RATE
from interface.CHBBaseDatabaseGenerator import CHBBaseDatabaseGenerator
from interface.constants import SEIZURE_INFO_KEY, START_TIME_KEY, \
    END_TIME_KEY
from interface.helper import get_positive_chunks_from_data, get_negative_chunks_from_data


class CHBPositiveEEGDatasetGenerator(CHBBaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)

    def generate_data_chunks(self, subject):
        for file in self.explorer.positive_edf_files:
            file_path = os.path.join(ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, subject,
                                     file)
            try:
                data = self.reader.read_file(file_path)
            except OSError as e:
                print("Dropped file: {}".format(file))
                continue

            # self.chunks = get_positive_chunks_from_data(data)
            if self.summary[file][SEIZURE_INFO_KEY]:
                previous_end_time_sample = 0
                for index, info in enumerate(self.summary[file][SEIZURE_INFO_KEY]):
                    start_sample = info[START_TIME_KEY] * DATASET_SAMPLE_RATE
                    end_sample = info[END_TIME_KEY] * DATASET_SAMPLE_RATE
                    try:
                        data = self.reader.read_file_in_interval(file_path, start_sample, end_sample)
                        negative_data = self.reader.read_file_in_interval(file_path, previous_end_time_sample,
                                                                          start_sample)
                    except OSError as e:
                        print("Dropped file: {}".format(file))
                        continue
                    finally:
                        previous_end_time_sample = end_sample
                    cc = get_positive_chunks_from_data(data)
                    nn = get_negative_chunks_from_data(negative_data)

                    # last seizure, retrieve negatives from after it
                    if index == len(self.summary[file][SEIZURE_INFO_KEY]) - 1:
                        negative_data = self.reader.read_file_in_interval(file_path, previous_end_time_sample,
                                                                          None)
                        cn = get_negative_chunks_from_data(negative_data)
                        self.extra_chunks.extend(cn)

                    self.chunks.extend(cc)
                    self.extra_chunks.extend(nn)
            self.save_chunks(POSITIVE_FOLDER_NAME, file)
            self.save_extra_chunks(NEGATIVE_FOLDER_NAME, file)
            self.total_chunks += len(self.chunks)
            self.total_extra_chunks += len(self.extra_chunks)
            self.chunks.clear()
            self.extra_chunks.clear()
