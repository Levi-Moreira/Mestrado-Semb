import os
import shutil

from constants import WORKING_DIR
from dataset.constants import DATA_SUBFOLDER_LOCATION, POSITIVE_FOLDER_NAME, SEGMENTED_DATA_FOLDER, NEGATIVE_FOLDER_NAME

patient = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06",
           "chb07", "chb08", "chb09", "chb10", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16",
           "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23", "chb24"]
for patient in patient:
    positive_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 POSITIVE_FOLDER_NAME)
    negative_path = os.path.join(WORKING_DIR,
                                 DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, patient,
                                 NEGATIVE_FOLDER_NAME)

    if os.path.exists(positive_path):
        shutil.rmtree(positive_path)

    if os.path.exists(negative_path):
        shutil.rmtree(negative_path)
    os.makedirs(positive_path)
    os.makedirs(negative_path)
