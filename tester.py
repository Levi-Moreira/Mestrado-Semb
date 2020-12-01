import numpy as np

from dataset.constants import POSITIVE_FOLDER_NAME
from interface.constants import WINDOW_SIZE
from models.generator import DataGenerator
from models.seizenet import SeizeNet


class Evaluator:
    file_per_channels = {
        2: "best_model_2.h5",
        13: "best_model_13.h5",
        18: "best_model_18.h5"
    }

    def __init__(self, channels, filename=None):
        self.channels = channels
        self.seize_net = SeizeNet(channels)
        if filename:
            self.filename = filename
        else:
            self.filename = self.file_per_channels[channels]
        self.seize_net.load_model(self.filename)
        self.confusion_matrix = None
        self.test_data_path = get_test_files()
        self.test_data_labels = build_labels(self.test_data_path)
        self.outputs = []

    def get_confusion_matrix(self):
        confusion_matrix = np.zeros((2, 2))
        for index, path in enumerate(self.test_data_path[:1000]):

            try:
                data = load_data_with_channels(path, self.channels)
            except Exception as e:
                raise e
            # print("Loading data #{}".format(index + 1))
            class_label = self.seize_net.model.predict(data)
            class_label = np.argmax(class_label)
            actual_label = self.test_data_labels[index].item()
            confusion_matrix[actual_label, class_label] += 1
            print(path)
            print("ACC: {}".format((confusion_matrix[0][0] + confusion_matrix[1][1]) * 100 / (index + 1)))

            self.outputs.append(self.seize_net.model.predict(data))
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def save_stats(self):
        np.savetxt("stats_for_{}.txt".format(self.filename), self.confusion_matrix,
                   delimiter=",")


for epoch in range(1, 41):
    filename = "best_model_18_{:02d}.h5".format(epoch)
    evaluator = Evaluator(18, filename)
    evaluator.get_confusion_matrix()
    evaluator.save_stats()


def load_data_with_channels(path, channels):
    X = np.empty((1, channels, WINDOW_SIZE, 1))

    # Generate data
    for i, path in enumerate([path]):
        # Store sample
        X[i,] = _load_data(channels, (channels, WINDOW_SIZE, 1))

    return X


def get_test_files():
    return [line.rstrip('\n') for line in open('test.txt')]


def build_labels(files):
    y = np.empty((len(files), 1), dtype=int)

    # Generate data
    for i, path in enumerate(files):
        # Store sample
        if POSITIVE_FOLDER_NAME in path:
            y[i,] = 1
        else:
            y[i,] = 0
    return y
