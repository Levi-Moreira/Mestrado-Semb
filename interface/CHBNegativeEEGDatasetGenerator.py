import os

from dataset.constants import ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, \
    NEGATIVE_FOLDER_NAME
from interface.CHBBaseDatabaseGenerator import CHBBaseDatabaseGenerator
from interface.helper import get_negative_chunks_from_data


class CHBNegativeEEGDatasetGenerator(CHBBaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)

    def generate_data_chunks(self, subject):
        for file in self.explorer.negative_edf_files:
            file_path = os.path.join(ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, subject,
                                     file)
            try:
                data = self.reader.read_file(file_path)
            except OSError as e:
                print("Dropped file: {}".format(file))
                continue

            self.chunks = get_negative_chunks_from_data(data)

            self.save_chunks(NEGATIVE_FOLDER_NAME, file)
            self.total_chunks += len(self.chunks)
            self.chunks.clear()
