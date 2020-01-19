from shutil import copyfile

from data_generator import DataProducer
from forward_pass import forward_pass
from models.seizenet import SeizeNet

import numpy as np
class Evaluator:
    file_per_channels = {
        2: "best_model_2.h5",
        13: "best_model_13.h5",
        23: "best_model_23.h5"
    }

    def __init__(self, channels):
        self.channels = channels
        self.seize_net = SeizeNet(channels)
        self.seize_net.load_model(self.file_per_channels[channels])
        self.confusion_matrix = None
        self.data_producer = DataProducer()
        self.test_data_path = self.data_producer.get_test_files()
        self.test_data_labels = self.data_producer.build_labels(self.test_data_path)
        self.outputs = []

    def get_confusion_matrix(self):
        confusion_matrix = np.zeros((2, 2))
        for index, path in enumerate(self.test_data_path):
            filepath = path.split("/")[-1]
            path = "/Users/levialbuquerque/PycharmProjects/semb/test/" + filepath

            try:
                data = self.data_producer.load_data_with_channels(path, self.channels)
            except:
                continue
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
        np.savetxt("py2_stats_for_{}.txt".format(self.file_per_channels[self.channels]), self.confusion_matrix,
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
evaluator = Evaluator(23)
evaluator.get_confusion_matrix()
evaluator.save_stats()
# file = "/Users/levialbuquerque/PycharmProjects/semb/data/chb15/positive/positive_23027.txt"
# import numpy as np
#
# #
# channels = 18
# data_generator = DataProducer()
# file_list = data_generator.get_test_files()
# index = 0
# for file in file_list:
#     file_name = file.split("/")[-1]
#     input = data_generator.load_data_with_channels(file, channels)
#     if "positive" in file_name:
#         file_name = file_name.replace("positive", "p")
#
#     if "negative" in file_name:
#         file_name = file_name.replace("negative", "n")
#     file_name = file_name.replace(".txt", "")
#     data = list(input.flatten())
#
#     import struct
#
#     s = struct.pack('f' * len(data), *data)
#     f = open('/Users/levialbuquerque/PycharmProjects/semb/test_files_18/{}'.format(file_name), 'wb')
#     f.write(s)
#     f.close()
#
#     # first_half = data[:int(len(data)/2)]
#     # second_half = data[int(len(data)/2):]
#     #
#     # np.savetxt("/Users/levialbuquerque/PycharmProjects/semb/test_files_31/{}.1".format(file_name), first_half,
#     #            fmt='%.3f')
#     # np.savetxt("/Users/levialbuquerque/PycharmProjects/semb/test_files_31/{}.2".format(file_name), second_half,
#     #            fmt='%.3f')
#     print("Saving: {} of {}".format(index, len(file_list)))
#     index += 1
# forward_pass(file, 18)

# import tensorflow as tf
#
# converter = tf.lite.TFLiteConverter.from_keras_model_file('best_model_38.h5')
# tfmodel = converter.convert()
# open("model38.tflite", "wb").write(tfmodel)

# import os
#
# os.mkdir("test")
# data_generator = DataProducer()
# file_list = data_generator.get_test_files()
#
# for file in file_list:
#     filename = file.split("/")[-1]
#     copyfile(file, "test/" + filename)
