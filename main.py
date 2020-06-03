import os
import random



from constants import POSITIVE_FOLDER_NAME, TRAIN_SPLIT, NEGATIVE_FOLDER_NAME
from data_generator import DataGenerator, DataProducer, generate_max_splits
from models.seizenet import SeizeNet

# data_generator = DataGenerator()
#
# train_list, test_list = data_generator.create_files_split()
#


# CHANNELS = 2
# producer = DataProducer()
# producer.generate_files_split()
# train_data, val_data = producer.get_files_split()
# train_data_generator = DataGenerator(train_data, CHANNELS)
# val_data_generator = DataGenerator(val_data, CHANNELS)
# net = SeizeNet(CHANNELS)
# net.get_model_summary()
# net.build()
# net.fit_data(train_data_generator, val_data_generator)
#
# CHANNELS = 13
# producer = DataProducer()
# # producer.generate_files_split()
# train_data, val_data = producer.get_files_split()
# train_data_generator = DataGenerator(train_data, CHANNELS)
# val_data_generator = DataGenerator(val_data, CHANNELS)
# net = SeizeNet(CHANNELS)
# net.get_model_summary()
# net.build()
# net.fit_data(train_data_generator, val_data_generator)
#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
CHANNELS = 18
producer = DataProducer()
# generate_max_splits()
train_data, val_data = producer.get_files_split()
# train_data = train_data[0:int(len(train_data) / 2)]
# val_data = val_data[0:int(len(val_data) / 2)]
train_data_generator = DataGenerator(train_data, CHANNELS)
val_data_generator = DataGenerator(val_data, CHANNELS)
net = SeizeNet(CHANNELS)
net.get_model_summary()
net.build()
net.fit_data(train_data_generator, val_data_generator)



