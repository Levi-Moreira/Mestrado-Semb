import os
import numpy as np
import random
from sklearn.svm import SVC

from models.generator import _load_data


def load(X, Y, data):
    for index, file in enumerate(data):
        X[index,] = np.load(file).astype('float16')
        if "positive" in file:
            Y[index] = 1.0
        else:
            Y[index] = 0.0


def map_full_path(path, is_train):
    initial_path = "/home/levi/PycharmProjects/Mestrado-Semb/data/svm/"
    if is_train:
        secondary = "train/"
    else:
        secondary = "test/"

    if "positive" in path:
        final = "positive/"
    else:
        final = "negative/"

    return initial_path + secondary + final + path


positive = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/train/positive")
negative = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/train/negative")

all_data = [map_full_path(p, True) for p in positive + negative]
random.shuffle(all_data)
X = np.empty((len(positive) + len(negative), 4240), dtype=np.float16)
Y = np.empty((len(positive) + len(negative)), dtype=np.float16)
load(X, Y, all_data)

svclassifier = SVC(kernel='rbf', verbose=True)
svclassifier.fit(X, Y)

positive_test = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/positive")
negative_test = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/negative")

all_test_data = [map_full_path(p, False) for p in positive_test + negative_test]
random.shuffle(all_test_data)
X_t = np.empty((len(all_test_data), 4240), dtype=np.float16)
Y_t = np.empty((len(all_test_data)), dtype=np.float16)
load(X_t, Y_t, all_test_data)
y_pred = svclassifier.predict(X_t)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_t, y_pred))
print(classification_report(Y_t, y_pred))
