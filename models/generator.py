import numpy as np
from keras.utils import Sequence

from dataset.constants import POSITIVE_FOLDER_NAME
from interface.constants import WINDOW_SIZE
from models.constants import BATCH_SIZE, CLASSES
from keras.preprocessing.image import load_img, img_to_array


def _load_data(path, channels, dim):
    data = np.load(path, allow_pickle=True).astype('float32')
    data = data[:channels, :WINDOW_SIZE]
    data = data.reshape(dim)
    return data


def load_image(path):
    img = load_img(path, target_size=(224, 224))
    return img_to_array(img)/255


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
        self.dim = (224,224,3)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_i_ds_temp = [self.paths[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_i_ds_temp)

        if self.to_fit:
            y = self._generate_y(list_i_ds_temp)
            return X, y
        else:
            return X

    def _generate_X(self, paths):
        X = np.empty((self.batch_size, *self.dim), dtype=np.float64)

        # Generate data
        for i, path in enumerate(paths):
            # Store sample
            X[i,] = load_image(path)
        return X

    def _generate_y(self, paths):
        y = np.empty((self.batch_size, 1), dtype=int)

        for i, path in enumerate(paths):
            if POSITIVE_FOLDER_NAME in path:
                y[i] = 1.0
                # y[i] = 0.0
            else:
                y[i] = 0.0
                # y[i, 0] = 1.0
        return y
