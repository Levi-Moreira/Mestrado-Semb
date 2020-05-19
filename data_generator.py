import os
import random

import numpy as np
from keras.utils import Sequence
from sklearn import preprocessing

from constants import NEGATIVE_FOLDER_NAME, POSITIVE_FOLDER_NAME, POSITIVE_PATH, NEGATIVE_PATH, TRAIN_SPLIT, \
    WINDOW_SIZE, BATCH_SIZE, CLASSES, PATIENT_CODE
from edf_interfacer import NegativeEEGDatasetGenerator, PositiveEEGDatasetGenerator
from frequency_splitter import butter_bandpass_filter


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, paths, channels,
                 to_fit=True, batch_size=BATCH_SIZE, dim=None,
                 n_channels=1, n_classes=CLASSES, shuffle=True):
        'Initialization'
        self.channels = channels
        self.paths = paths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (1, WINDOW_SIZE, self.channels)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.paths[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def _generate_X(self, paths):
        'Generates data containing batch_size images'
        # Initialization
        frequency_bands = [(4, 7), (7, 12), (12, 19), (19, 30), (30, 40)]
        X = np.empty((self.batch_size, 5, *self.dim), dtype=np.float64)

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            for index, band in enumerate(frequency_bands):
                low, high = band
                filtered = butter_bandpass_filter(DataGenerator._load_data(path, self.channels, self.dim), low, high,
                                                  256)
                # filtered = self.scale(filtered)
                X[i, index,] = filtered

        return X

    @staticmethod
    def _load_data(path, channels, dim):
        data = np.load(path).astype('float64')
        data = data[:channels, :WINDOW_SIZE]
        data = DataGenerator.scale(data, dim)
        return data

    @staticmethod
    def scale(data, dim):
        # data = preprocessing.scale(data)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
        data = min_max_scaler.fit_transform(data)

        return data.reshape(dim)

    def _generate_y(self, paths):
        'Generates data containing batch_size masks'
        y = np.empty((self.batch_size, 1), dtype=int)

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            if POSITIVE_FOLDER_NAME in path:
                y[i,] = 1
            else:
                y[i,] = 0
        return y


class DataProducer:
    def data_file_creation(self, channels):
        self.channels = channels
        negative_dataset_generator = NegativeEEGDatasetGenerator(PATIENT_CODE)
        negative_dataset_generator.save_chunks(NEGATIVE_FOLDER_NAME)
        print(len(negative_dataset_generator.chunks))

        positive_dataset_generator = PositiveEEGDatasetGenerator(PATIENT_CODE)
        positive_dataset_generator.save_chunks(POSITIVE_FOLDER_NAME)
        print(len(positive_dataset_generator.chunks))

    def load_data_with_channels(self, path, channels):
        # data = np.loadtxt(path, delimiter=",")
        # data = data[:channels, :WINDOW_SIZE]
        # data = preprocessing.scale(data)
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
        # data = min_max_scaler.fit_transform(data)
        # return data.reshape((1, WINDOW_SIZE, channels))
        frequency_bands = [(4, 7), (7, 12), (12, 19), (19, 30), (30, 40)]
        X = np.empty((1, 5, 1, WINDOW_SIZE, channels))

        # Generate data
        for i, path in enumerate([path]):
            # Store sample
            for index, band in enumerate(frequency_bands):
                low, high = band

                # Preprocessing
                filtered = butter_bandpass_filter(
                    DataGenerator._load_data(path, channels, (1, WINDOW_SIZE, channels)), low, high, 256)

                X[i, index,] = filtered

        return X

    @staticmethod
    def __apply_path(file):
        if POSITIVE_FOLDER_NAME in file:
            return os.path.join(POSITIVE_PATH, file)
        else:
            return os.path.join(NEGATIVE_PATH, file)

    def __get_data_list(self):
        BALACING_FACTOR = 5480
        data = os.listdir(POSITIVE_PATH)[:BALACING_FACTOR] + os.listdir(NEGATIVE_PATH)[:BALACING_FACTOR]
        return list(map(self.__apply_path, data))

    def generate_files_split(self):
        full_path_data = self.__get_data_list()
        random.shuffle(full_path_data)
        data_count = len(full_path_data)
        split_point_for_test = int(data_count * 0.9)

        all_data = full_path_data[0:split_point_for_test]
        data_count = len(all_data)
        test_data = full_path_data[split_point_for_test:]
        split_point_val = int(data_count * TRAIN_SPLIT)

        train_data = all_data[0:split_point_val]
        val_data = all_data[split_point_val:]

        with open('train.txt', 'w') as f:
            for item in train_data:
                f.write("%s\n" % item)

        with open('val.txt', 'w') as f:
            for item in val_data:
                f.write("%s\n" % item)

        with open('test.txt', 'w') as f:
            for item in test_data:
                f.write("%s\n" % item)

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


def generate_max_splits():
    patients_to_train = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
                         "chb11", "chb12", "chb13", "chb14", "chb16", "chb17", "chb18", "chb19", "chb20",
                         "chb21", "chb22", "chb23", "chb24"]
    maxs = [5488, 2117, 4949, 4826, 7186, 1395, 4180, 10805, 3452, 5556,
            10660, 7924, 2881, 1746, 395, 3749, 3870, 2980, 3428,
            2415, 2549, 5246, 5815]

    patients_to_test = ["chb15"]
    maxs_to_test = [18001]

    def __apply_path(file):
        if POSITIVE_FOLDER_NAME in file:
            return os.path.join(POSITIVE_PATH, file)
        else:
            return os.path.join(NEGATIVE_PATH, file)

    full_path_data = []
    for index, patient in enumerate(patients_to_train):
        POSITIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", "chb-mit-scalp-eeg-database-1.0.0", patient,
                                       POSITIVE_FOLDER_NAME])
        NEGATIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", "chb-mit-scalp-eeg-database-1.0.0", patient,
                                       NEGATIVE_FOLDER_NAME])
        pos_dir = os.listdir(POSITIVE_PATH)[:int(maxs[index])]
        neg_dir = os.listdir(NEGATIVE_PATH)[:int(maxs[index])]
        data = set(pos_dir + neg_dir)
        full_path_data += list(map(__apply_path, data))

    random.shuffle(full_path_data)
    data_count = len(full_path_data)
    split_point_val = int(data_count * TRAIN_SPLIT)
    train_data = full_path_data[0:split_point_val]
    val_data = full_path_data[split_point_val:]

    with open('train.txt', 'w') as f:
        for item in train_data:
            f.write("%s\n" % item)

    with open('val.txt', 'w') as f:
        for item in val_data:
            f.write("%s\n" % item)

    full_path_test_data = []
    for index, patient in enumerate(patients_to_test):
        POSITIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", "chb-mit-scalp-eeg-database-1.0.0", patient,
                                       POSITIVE_FOLDER_NAME])
        NEGATIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", "chb-mit-scalp-eeg-database-1.0.0", patient,
                                       NEGATIVE_FOLDER_NAME])
        pos_dir = os.listdir(POSITIVE_PATH)[:int(maxs_to_test[index] / 2)]
        neg_dir = os.listdir(NEGATIVE_PATH)[:int(maxs_to_test[index] / 2)]
        data = set(pos_dir + neg_dir)
        full_path_test_data += list(map(__apply_path, data))

    test_data = full_path_test_data
    with open('test.txt', 'w') as f:
        for item in test_data:
            f.write("%s\n" % item)
