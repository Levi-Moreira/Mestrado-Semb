import os
import random

from constants import WORKING_DIR, PATIENTS_TO_TRAIN, PATIENTS_TO_TEST, PATIENTS_TO_VALIDATE
from dataset.constants import POSITIVE_FOLDER_NAME, SEGMENTED_DATA_FOLDER, NEGATIVE_FOLDER_NAME, DATA_SUBFOLDER_LOCATION

MAX_DATA_SIZE = 500000000


def __apply_path(file, positive_path, negative_path):
    if POSITIVE_FOLDER_NAME in file:
        return os.path.join(positive_path, file)
    else:
        return os.path.join(negative_path, file)


def generate_train_set(patients_to_train):
    split('train.txt', patients_to_train)


def generate_validation_set(patients_to_validate):
    split('val.txt', patients_to_validate)


def generate_test_set(patients_to_test):
    split('test.txt', patients_to_test)


def get_files_split():
    train_data = [line.rstrip('\n') for line in open('train.txt')]
    val_data = [line.rstrip('\n') for line in open('val.txt')]
    return train_data, val_data


def get_test_files():
    return [line.rstrip('\n') for line in open('test.txt')]


def split(final_file, patients):
    print("------------------------------------------------------------------")
    print(final_file)
    full_path_data = []
    for index, patient in enumerate(patients):
        print("Adding patient {}".format(patient))
        positive_path = os.path.join(WORKING_DIR,
                                     DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                     POSITIVE_FOLDER_NAME)
        negative_path = os.path.join(WORKING_DIR,
                                     DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                     NEGATIVE_FOLDER_NAME)
        positive_files = os.listdir(positive_path)
        negative_files = os.listdir(negative_path)
        print("Positive: {}. Negative: {}".format(len(positive_files), len(negative_files)))
        random.shuffle(negative_files)
        random.shuffle(positive_files)

        min_size = min(len(positive_files), len(negative_files), MAX_DATA_SIZE)
        negative_files = negative_files[:min_size]
        positive_files = positive_files[:min_size]

        data = set(positive_files + negative_files)
        print("Total added: {}".format(len(data)))
        print("")
        full_path_data += list(map(lambda p: __apply_path(p, positive_path, negative_path), data))

    random.shuffle(full_path_data)

    positive_count = len(list(filter(lambda x: POSITIVE_FOLDER_NAME in x, full_path_data)))
    negative_count = len(list(filter(lambda x: NEGATIVE_FOLDER_NAME in x, full_path_data)))

    print("Data split")
    print("Positive: {}".format(positive_count))
    print("Negative: {}".format(negative_count))
    print("")

    with open(final_file, 'w') as f:
        for item in full_path_data:
            f.write("%s\n" % item)


def generate_max_splits():
    generate_train_set(PATIENTS_TO_TRAIN)
    generate_validation_set(PATIENTS_TO_VALIDATE)
    generate_test_set(PATIENTS_TO_TEST)
