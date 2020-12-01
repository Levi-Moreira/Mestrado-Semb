import os

from dataset.constants import ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, EDF_FILE_MARKER, \
    SEIZURE_FILE_MARKER


class CHBFolderExplorer:
    def __init__(self, subject_folder):
        all_files = os.listdir(
            os.path.join(ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, subject_folder))
        edf_files = list(filter(lambda file: EDF_FILE_MARKER in file, all_files))
        seizure_edf_files = list(filter(lambda file: SEIZURE_FILE_MARKER in file, edf_files))
        self.positive_edf_files = list(map(lambda file: file.strip(SEIZURE_FILE_MARKER), seizure_edf_files))

        self.negative_edf_files = list(set(map(lambda file: file.strip(SEIZURE_FILE_MARKER), edf_files)))
        self.negative_edf_files = list(set(self.negative_edf_files) - set(self.positive_edf_files))
