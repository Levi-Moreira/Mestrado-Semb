import numpy as np
from keras.utils import Sequence

from constants import BATCH_SIZE, CLASSES, WINDOW_SIZE, POSITIVE_FOLDER_NAME


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
        X = np.empty((self.batch_size, *self.dim), dtype=np.float64)

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            X[i,] = DataGenerator._load_data(path, self.channels, self.dim)
            # for index, band in enumerate(frequency_bands):
            #     low, high = band
            #     filtered = butter_bandpass_filter(DataGenerator._load_data(path, self.channels, self.dim), low, high,
            #                                       256)
            #     # filtered = self.scale(filtered)
            #     X[i, index,] = filtered

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
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
        # data = min_max_scaler.fit_transform(data)

        return data.reshape(dim)

    def _generate_y(self, paths):
        'Generates data containing batch_size masks'
        y = np.empty((self.batch_size, 2), dtype=int)

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            if POSITIVE_FOLDER_NAME in path:
                y[i, 1] = 1.0
                y[i, 0] = 0.0
            else:
                y[i, 1] = 0.0
                y[i, 0] = 1.0
        return y