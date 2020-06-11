import os
from random import random

from constants import POSITIVE_FOLDER_NAME, MAIN_FOLDER_NAME, NEGATIVE_FOLDER_NAME


def generate_max_splits():
    patients_to_train = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
                         "chb11", "chb13", "chb12", "chb14", "chb16", "chb17", "chb18", "chb19", "chb20",
                         "chb21", "chb22", "chb23", "chb24"]

    patients_to_test = ["chb15"]
    patients_to_validate = ["chb15"]

    def __apply_path(file):
        if POSITIVE_FOLDER_NAME in file:
            return os.path.join(POSITIVE_PATH, file)
        else:
            return os.path.join(NEGATIVE_PATH, file)

    full_path_train_data = []
    for index, patient in enumerate(patients_to_train):
        print("Adding patient {}".format(patient))
        POSITIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       POSITIVE_FOLDER_NAME])
        NEGATIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       NEGATIVE_FOLDER_NAME])
        pos_dir = os.listdir(POSITIVE_PATH)
        neg_dir = os.listdir(NEGATIVE_PATH)
        print("Positive: {}. Negative: {}".format(len(pos_dir), len(neg_dir)))
        random.shuffle(neg_dir)
        random.shuffle(pos_dir)
        neg_dir = neg_dir[:len(pos_dir)]
        data = set(pos_dir + neg_dir)
        print("Total added: {}".format(len(data)))
        print("")
        full_path_train_data += list(map(__apply_path, data))

    random.shuffle(full_path_train_data)
    train_data = full_path_train_data

    pos_count = len(list(filter(lambda x: "positive" in x, train_data)))
    negative_count = len(list(filter(lambda x: "negative" in x, train_data)))

    print("Data split")
    print("Positive: {}".format(pos_count))
    print("Negative: {}".format(negative_count))

    with open('train.txt', 'w') as f:
        for item in train_data:
            f.write("%s\n" % item)

    full_path_val_data = []
    for index, patient in enumerate(patients_to_validate):
        POSITIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       POSITIVE_FOLDER_NAME])
        NEGATIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       NEGATIVE_FOLDER_NAME])
        pos_dir = os.listdir(POSITIVE_PATH)
        neg_dir = os.listdir(NEGATIVE_PATH)
        random.shuffle(neg_dir)
        random.shuffle(pos_dir)
        min_size = min(len(pos_dir), len(neg_dir))
        neg_dir = neg_dir[:min_size]
        pos_dir = pos_dir[:min_size]
        data = set(pos_dir + neg_dir)
        full_path_val_data += list(map(__apply_path, data))

    val_data = full_path_val_data
    with open('val.txt', 'w') as f:
        for item in val_data:
            f.write("%s\n" % item)

    full_path_test_data = []
    for index, patient in enumerate(patients_to_test):
        POSITIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       POSITIVE_FOLDER_NAME])
        NEGATIVE_PATH = os.path.join(os.getcwd(),
                                     *["data", MAIN_FOLDER_NAME, patient,
                                       NEGATIVE_FOLDER_NAME])
        pos_dir = os.listdir(POSITIVE_PATH)
        neg_dir = os.listdir(NEGATIVE_PATH)
        random.shuffle(neg_dir)
        data = set(pos_dir + neg_dir)
        full_path_test_data += list(map(__apply_path, data))

    test_data = full_path_test_data
    with open('test.txt', 'w') as f:
        for item in test_data:
            f.write("%s\n" % item)
