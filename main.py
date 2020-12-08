from keras import backend as K
from dataset.splitter import get_files_split, generate_max_splits
from models.generator import DataGenerator

from models.seizenet import SeizeNet

CHANNELS = 18


def device_info():
    K.clear_session()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # tf.compat.v1.disable_eager_execution()


def start():
    patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
                "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20",
                "chb21", "chb22", "chb23", "chb24"]
    for patient in patients:
        all = set(patients)
        all.remove(patient)
        totrain = list(all)
        totest = [patient]
        print(" Running for patient {}".format(patient))
        generate_max_splits(totrain, totest, totest, patient)
        train_data, val_data = get_files_split(patient)
        if len(val_data) < 100:
            val_data = val_data + val_data
        if len(val_data) < 150:
            val_data = val_data + val_data
        train_data_generator = DataGenerator(train_data, CHANNELS)
        val_data_generator = DataGenerator(val_data, CHANNELS)
        net = SeizeNet(CHANNELS, patient)
        net.get_model_summary()
        net.build()
        net.fit_data(train_data_generator, val_data_generator)


device_info()
start()
