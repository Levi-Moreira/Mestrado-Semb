import os

POSITIVE_FOLDER_NAME = "positive"
NEGATIVE_FOLDER_NAME = "negative"
PATIENT_CODE = "chb20"
POSITIVE_PATH = os.path.join(os.getcwd(),
                             *["data", "chb-mit-scalp-eeg-database-1.0.0", PATIENT_CODE, POSITIVE_FOLDER_NAME])
NEGATIVE_PATH = os.path.join(os.getcwd(),
                             *["data", "chb-mit-scalp-eeg-database-1.0.0", PATIENT_CODE, NEGATIVE_FOLDER_NAME])
EEG_SIGNAL_NUMBER = 18
TRAIN_SPLIT = 0.7

OLD_SAMPLE_RATE = 256
WINDOW_IN_SECONDS = 5
POSITIVE_SHIFT_WINDOW = 0.1
WINDOW_SIZE = OLD_SAMPLE_RATE * WINDOW_IN_SECONDS
BATCH_SIZE = 256
CLASSES = 2
# Review 20, 24