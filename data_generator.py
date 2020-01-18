import os
import random

import numpy as np
from keras.utils import Sequence
from sklearn import preprocessing

from constants import NEGATIVE_FOLDER_NAME, POSITIVE_FOLDER_NAME, POSITIVE_PATH, NEGATIVE_PATH, TRAIN_SPLIT, \
    WINDOW_SIZE, CHANNELS, BATCH_SIZE, CLASSES, PATIENT_CODE
from edf_interfacer import NegativeEEGDatasetGenerator, PositiveEEGDatasetGenerator


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, paths,
                 to_fit=True, batch_size=BATCH_SIZE, dim=(1, WINDOW_SIZE, CHANNELS),
                 n_channels=1, n_classes=CLASSES, shuffle=True):
        'Initialization'
        self.paths = paths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
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
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            X[i,] = self._load_data(path)

        return X

    def _load_data(self, path):
        data = np.loadtxt(path, delimiter=",")
        data = data[:CHANNELS, :WINDOW_SIZE]
        data = preprocessing.scale(data)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
        data = min_max_scaler.fit_transform(data)
        return data.reshape(self.dim)

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
    def data_file_creation(self):
        negative_dataset_generator = NegativeEEGDatasetGenerator(PATIENT_CODE)
        negative_dataset_generator.save_chunks(NEGATIVE_FOLDER_NAME)
        print(len(negative_dataset_generator.chunks))

        positive_dataset_generator = PositiveEEGDatasetGenerator(PATIENT_CODE)
        positive_dataset_generator.save_chunks(POSITIVE_FOLDER_NAME)
        print(len(positive_dataset_generator.chunks))

    def load_data_with_channels(self, path, channels):
        data = np.loadtxt(path, delimiter=",")
        data = data[:channels, :WINDOW_SIZE]
        data = preprocessing.scale(data)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
        data = min_max_scaler.fit_transform(data)
        return data.reshape((1, WINDOW_SIZE, channels))

    def __apply_path(self, file):
        if POSITIVE_FOLDER_NAME in file:
            return os.path.join(POSITIVE_PATH, file)
        else:
            return os.path.join(NEGATIVE_PATH, file)

    def __get_data_list(self):
        data = os.listdir(POSITIVE_PATH) + os.listdir(NEGATIVE_PATH)
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
