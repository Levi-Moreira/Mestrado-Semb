from dataset.constants import HARD_DISK_LOCATION, TUH_SEIZURE_FILES_LIST


class TUHFolderExplorer:
    def __init__(self):
        self.positive_edf_files = [HARD_DISK_LOCATION + line.rstrip('\n') for line in open(TUH_SEIZURE_FILES_LIST)]
        self.positive_edf_files = list(filter(lambda x: "tcp_le" in x, self.positive_edf_files))
