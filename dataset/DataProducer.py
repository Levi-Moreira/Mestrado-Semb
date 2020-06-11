import numpy as np

from constants import WINDOW_SIZE, POSITIVE_FOLDER_NAME
from interface.NegativeEEGDatasetGenerator import NegativeEEGDatasetGenerator
from interface.PositiveEEGDatasetGenerator import PositiveEEGDatasetGenerator
from models.generator import DataGenerator


class DataProducer:
    PATIENT_CODE = None

    def data_file_creation(self, channels, patient):
        self.channels = channels
        self.PATIENT_CODE = patient
        positive_dataset_generator = PositiveEEGDatasetGenerator(self.PATIENT_CODE)
        print("Created positive chuncks {}".format(positive_dataset_generator.total_chuncks))
        print("Created negative chuncks {}".format(positive_dataset_generator.total_extra_chunks))
        del (positive_dataset_generator)

        negative_dataset_generator = NegativeEEGDatasetGenerator(self.PATIENT_CODE)
        print("Created negative chuncks {}".format(negative_dataset_generator.total_chuncks))
        del (negative_dataset_generator)

    def load_data_with_channels(self, path, channels):
        X = np.empty((1, 1, WINDOW_SIZE, channels))

        # Generate data
        for i, path in enumerate([path]):
            # Store sample
            X[i,] = DataGenerator._load_data(path, channels, (1, WINDOW_SIZE, channels))

        return X

    def get_files_split(self):
        train_data = [line.rstrip('\n') for line in open('train.txt')]
        val_data = [line.rstrip('\n') for line in open('val.txt')]

        return train_data, val_data

    def get_test_files(self):
        return [line.rstrip('\n') for line in open('test.txt')]

    def build_labels(self, files):
        y = np.empty((len(files), 1), dtype=int)

        # Generate data
        for i, path in enumerate(files):
            # Store sample
            if POSITIVE_FOLDER_NAME in path:
                y[i,] = 1
            else:
                y[i,] = 0
        return y
