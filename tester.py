import numpy as np

from data_generator import DataProducer
from forward_pass import forward_pass
from models.seizenet import SeizeNet


class Evaluator:
    file_per_channels = {
        2: "best_model_2.h5",
        18: "best_model_18.h5",
        31: "best_model_38.h5"
    }

    def __init__(self, channels):
        self.channels = channels
        self.seize_net = SeizeNet()
        self.seize_net.load_model(self.file_per_channels[channels])
        self.confusion_matrix = None
        self.data_producer = DataProducer()
        self.test_data_path = self.data_producer.get_test_files()
        self.test_data_labels = self.data_producer.build_labels(self.test_data_path)
        self.outputs = []

    def get_confusion_matrix(self):
        confusion_matrix = np.zeros((2, 2))
        for index, path in enumerate(self.test_data_path):
            data = self.data_producer.load_data_with_channels(path, self.channels)
            print("Loading data #{}".format(index + 1))
            class_label = self.seize_net.model.predict_classes(
                data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
            actual_label = self.test_data_labels[index]
            confusion_matrix[actual_label, class_label] += 1

            self.outputs.append(self.seize_net.model.predict(
                data.reshape(1, data.shape[0], data.shape[1], data.shape[2])))
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def save_stats(self):
        np.savetxt("stats_for_{}.txt".format(self.file_per_channels[self.channels]), self.confusion_matrix,
                   delimiter=",")


class EvaluatorPy:
    file_per_channels = {
        2: "best_model_2.h5",
        18: "best_model_18.h5",
        31: "best_model_38.h5"
    }

    def __init__(self, channels):
        self.confusion_matrix = None
        self.channels = channels

    def get_confusion_matrix(self):
        data_producer = DataProducer()
        test_data_path = data_producer.get_test_files()
        test_data_labels = data_producer.build_labels(test_data_path)

        confusion_matrix = np.zeros((2, 2))
        for index, file in enumerate(test_data_path):
            print("Reading file #{}".format(index + 1))
            Y = int(test_data_labels[index].item())
            Y_hat = forward_pass(file, self.channels)
            confusion_matrix[Y, Y_hat] += 1
        self.confusion_matrix = confusion_matrix

    def save_stats(self):
        np.savetxt("py_stats_for_{}.txt".format(self.file_per_channels[self.channels]), self.confusion_matrix,
                   delimiter=",")


class ModelWeightsExtractor:

    def __init__(self, model_file):
        self.model_file_name = model_file
        self.seize_net = SeizeNet()
        self.seize_net.load_model(model_file)

    def export_weights(self):
        for layer in self.seize_net.model.layers:
            if "dense" in layer.name or "conv" in layer.name:
                layer_name = layer.name
                weights, biases = layer.get_weights()
                self.save_file("{}_{}_weights.npy".format(self.model_file_name, layer_name), weights.flatten())
                self.save_file("{}_{}_bias.npy".format(self.model_file_name, layer_name), biases.flatten())

            if "batch" in layer.name:
                layer_name = layer.name
                gamma, beta = layer.gamma.numpy(), layer.beta.numpy()
                mean, variance = layer.moving_mean.numpy(), layer.moving_variance.numpy()
                self.save_file("{}_{}_beta.npy".format(self.model_file_name, layer_name), beta.flatten())
                self.save_file("{}_{}_gamma.npy".format(self.model_file_name, layer_name), gamma.flatten())
                self.save_file("{}_{}_mean.npy".format(self.model_file_name, layer_name), mean.flatten())
                self.save_file("{}_{}_variance.npy".format(self.model_file_name, layer_name), variance.flatten())

    def save_file(self, name, data):
        np.savetxt(name, data)


# extractor = ModelWeightsExtractor("best_model_38.h5")
# extractor.export_weights()
evaluator = EvaluatorPy(31)
evaluator.get_confusion_matrix()
evaluator.save_stats()
