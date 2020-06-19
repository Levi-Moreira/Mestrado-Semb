from dataset.splitter import get_files_split
from models.generator import DataGenerator

from models.seizenet import SeizeNet

CHANNELS = 18


def device_info():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


def start():
    train_data, val_data = get_files_split()
    train_data_generator = DataGenerator(train_data, CHANNELS)
    val_data_generator = DataGenerator(val_data, CHANNELS)
    net = SeizeNet(CHANNELS)
    net.get_model_summary()
    net.build()
    net.fit_data(train_data_generator, val_data_generator)


device_info()
start()
