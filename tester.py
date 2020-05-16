import numpy as np
from sklearn.metrics import classification_report

from data_generator import DataProducer, DataGenerator
from models.seizenet import SeizeNet


class Evaluator:
    file_per_channels = {
        2: "best_model_2.h5",
        13: "best_model_13.h5",
        23: "best_model_23_07.h5"
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

            try:
                data = self.data_producer.load_data_with_channels(path, self.channels)
            except Exception as e:
                raise e
            # print("Loading data #{}".format(index + 1))
            class_label = self.seize_net.model.predict(data)
            class_label = 0 if class_label <= 0.5 else 1
            actual_label = self.test_data_labels[index].item()
            confusion_matrix[actual_label, class_label] += 1
            print(path)
            print("ACC: {}".format((confusion_matrix[0][0] + confusion_matrix[1][1]) * 100 / (index + 1)))

            self.outputs.append(self.seize_net.model.predict(data))
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def save_stats(self):
        np.savetxt("stats_for_{}.txt".format(self.file_per_channels[self.channels]), self.confusion_matrix,
                   delimiter=",")


# extractor = ModelWeightsExtractor("/Volumes/ELEMENTS/Mestrado/CNN/BALANCED_DATA/CHB15/best_model_2.h5", 2)
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
