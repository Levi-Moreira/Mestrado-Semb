import os

POSITIVE_FOLDER_NAME = "positive"
NEGATIVE_FOLDER_NAME = "negative"
PATIENT_CODE = "chb20"
MAIN_FOLDER_NAME = "chb-mit-scalp-eeg-database-1.0.0-2xshift1"
POSITIVE_PATH = os.path.join(os.getcwd(),
                             *["data", MAIN_FOLDER_NAME, PATIENT_CODE, POSITIVE_FOLDER_NAME])
NEGATIVE_PATH = os.path.join(os.getcwd(),
                             *["data", MAIN_FOLDER_NAME, PATIENT_CODE, NEGATIVE_FOLDER_NAME])
EEG_SIGNAL_NUMBER = 18
TRAIN_SPLIT = 0.7

OLD_SAMPLE_RATE = 256
WINDOW_IN_SECONDS = 2
POSITIVE_SHIFT_WINDOW = 0.075
WINDOW_SIZE = OLD_SAMPLE_RATE * WINDOW_IN_SECONDS
BATCH_SIZE = 256
CLASSES = 2
# "chb-mit-scalp-eeg-database-1.0.0" = 5 seconds and 0.1 shift
# "chb-mit-scalp-eeg-database-1.0.0-5xshift0075" = 5 seconds and 0.075 shift

# "chb-mit-scalp-eeg-database-1.0.0-5xshift0075" = 5 seconds and 1 shift
# "chb-mit-scalp-eeg-database-1.0.0-5xshift0075" = 2 seconds and 1 shift
# "chb-mit-scalp-eeg-database-1.0.0-5xshift0075" = 2 seconds and 4 samples de shift
