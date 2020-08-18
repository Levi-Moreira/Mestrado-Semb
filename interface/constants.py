from dataset.constants import DATASET_SAMPLE_RATE

WINDOW_IN_SECONDS = 2
POSITIVE_SHIFT_WINDOW_SAMPLE_SIZE = int(WINDOW_IN_SECONDS * DATASET_SAMPLE_RATE)
WINDOW_SIZE = DATASET_SAMPLE_RATE * WINDOW_IN_SECONDS

SEIZURE_INFO_KEY = "seizure_info"
START_TIME_KEY = "start_time"
END_TIME_KEY = "end_time"
