import os

from constants import WORKING_DIR
from dataset.constants import POSITIVE_FOLDER_NAME, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, NEGATIVE_FOLDER_NAME
from dataset.splitter import __apply_path

patients = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10",
            "chb15", ]

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
    all = positive_files + negative_files
    all = list(map(lambda p: __apply_path(p, positive_path, negative_path), all))
    all_smote = list(filter(lambda x: "png" in x, all))

    for smote in all_smote:
        os.remove(smote)
