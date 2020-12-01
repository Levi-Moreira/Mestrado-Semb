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
    if "chb15" not in path:
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

positive_test = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/positive")
negative_test = os.listdir("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/negative")

# post_neg = positive + negative + positive_test + negative_test
# random.shuffle(post_neg)

# split = int(len(post_neg) * 0.7)
all_data = positive + negative
all_data = [map_full_path(p, True) for p in all_data]
random.shuffle(all_data)
# all_data = all_data[:int(len(all_data))]

X = np.empty((len(all_data), 896), dtype=np.float16)
Y = np.empty((len(all_data)), dtype=np.float16)
load(X, Y, all_data)

svclassifier = SVC(kernel='rbf', verbose=True)
svclassifier.fit(X, Y)

all_test_data = positive_test + negative_test
all_test_data = [map_full_path(p, False) for p in all_test_data]
random.shuffle(all_test_data)
X_t = np.empty((len(all_test_data), 896), dtype=np.float16)
Y_t = np.empty((len(all_test_data)), dtype=np.float16)
load(X_t, Y_t, all_test_data)
y_pred = svclassifier.predict(X_t)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_t, y_pred))
print(classification_report(Y_t, y_pred))
