import datetime
import random

import numpy as np
import os

from constants import WORKING_DIR
from dataset.constants import POSITIVE_FOLDER_NAME, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, NEGATIVE_FOLDER_NAME
from dataset.splitter import __apply_path
from interface.constants import WINDOW_SIZE
from imblearn.over_sampling import SMOTE


def _load_data(path, channels, dim):
    data = np.load(path).astype('float32')
    data = data[:channels, :WINDOW_SIZE]
    data = data.reshape(dim)
    return data


patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
            "chb11", "chb12", "chb13", "chb14", "chb16", "chb17", "chb18", "chb19", "chb20",
            "chb21", "chb22", "chb23", "chb24"]

for index, patient in enumerate(patients):
    print("Adding patient {}".format(patient))
    positive_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 POSITIVE_FOLDER_NAME)
    negative_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 NEGATIVE_FOLDER_NAME)

    positive_files = os.listdir(positive_path)
    negative_files = os.listdir(negative_path)
    random.shuffle(negative_files)

    all = positive_files + negative_files[:2 * len(positive_files)]
    all = list(map(lambda p: __apply_path(p, positive_path, negative_path), all))

    X = np.empty((len(all), 18 * WINDOW_SIZE), dtype=np.float32)
    y = np.empty((len(all)), dtype=np.float32)
    for i, path in enumerate(all):
        all_c = _load_data(path, 18, (18 * WINDOW_SIZE))
        X[i,] = all_c
        if POSITIVE_FOLDER_NAME in path:
            y[i,] = 1.0
        else:
            y[i,] = 0.0

    oversample = SMOTE()
    Xs, ys = oversample.fit_resample(X, y)
    Xs = Xs.reshape((Xs.shape[0], 18, WINDOW_SIZE, 1))
    file_path = os.path.join(WORKING_DIR, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient, POSITIVE_FOLDER_NAME)

    print("Created {} samples".format(len(ys) / 2))
    for i, c in enumerate(ys):
        if c == 1.0:
            np.save(
                os.path.join(file_path,
                             "smote_{}_{}_{}".format(POSITIVE_FOLDER_NAME, i, datetime.datetime.now().timestamp())),
                Xs[i])
