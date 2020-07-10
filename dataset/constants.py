from constants import WORKING_DIR

POSITIVE_FOLDER_NAME = "positive"
NEGATIVE_FOLDER_NAME = "negative"

SEGMENTED_DATA_FOLDER = "chb-mit-scalp-eeg-database-1.0.0-segmented"
DATASET_FOLDER_NAME = "chb-mit-scalp-eeg-database-1.0.0"
EEG_SIGNAL_NUMBER = 18

DATASET_SAMPLE_RATE = 256
ON_EXTERNAL_HARD_DISK = True

if ON_EXTERNAL_HARD_DISK:
    ORIGINAL_DATASET_LOCATION = "/media/levi/ELEMENTS/Mestrado"
else:
    ORIGINAL_DATASET_LOCATION = WORKING_DIR

DATA_SUBFOLDER_LOCATION = "data"

EDF_FILE_MARKER = ".edf"
SEIZURE_FILE_MARKER = ".seizures"

TUH_SEIZURE_FILES_LIST = "/media/levi/ELEMENTS/tuh_eeg_seizure/v1.5.1/_DOCS/05_files_with_seizures.list"
HARD_DISK_LOCATION = "/media/levi/ELEMENTS/"