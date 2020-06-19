import os
from abc import abstractmethod, ABC

import numpy as np

from constants import WORKING_DIR
from dataset.CHBFolderExplorer import CHBFolderExplorer
from dataset.constants import ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, \
    SEGMENTED_DATA_FOLDER
from edf.EEGReader import EEGReader
from interface.constants import SEIZURE_INFO_KEY, START_TIME_KEY, END_TIME_KEY


class BaseDatabaseGenerator(ABC):
    LIMIT_FILES = 2
    FILE_NAME_POSITION = 0
    NUMBER_OF_SEIZURE_POSITION = 3

    def __init__(self, subject):
        self.subject = subject
        self.__summary_builder()
        self.reader = EEGReader()
        self.chunks = []
        self.extra_chunks = []
        self.total_chunks = 0
        self.total_extra_chunks = 0
        self.explorer = CHBFolderExplorer(subject)
        self.generate_data_chunks(subject)

    def __summary_builder(self):

        file = open(os.path.join(ORIGINAL_DATASET_LOCATION, DATA_SUBFOLDER_LOCATION, DATASET_FOLDER_NAME, self.subject,
                                 "{}-summary.txt".format(self.subject)), "r")
        file = str(file.read())
        file = file.split("\n\n")
        infos = {}
        for file_info in file:
            lines = file_info.split("\n")
            if 20 > len(lines) > 2:
                info = {}
                file_name = lines[self.FILE_NAME_POSITION].split(": ")[1]
                info["file_name"] = file_name
                seizures = int(lines[self.NUMBER_OF_SEIZURE_POSITION].split(": ")[1])
                info["number_of_seizures"] = seizures
                if seizures > 0:
                    info[SEIZURE_INFO_KEY] = []
                    lines = list(filter(lambda x: x is not "", lines))
                    for n in range(4, len(lines), 2):
                        seizure_info = {}
                        start_time = lines[n].split(":")[1]
                        seizure_info[START_TIME_KEY] = int(start_time.split(" ")[1])
                        end_time = lines[n + 1].split(":")[1]
                        seizure_info[END_TIME_KEY] = int(end_time.split(" ")[1])
                        info[SEIZURE_INFO_KEY].append(seizure_info)
                infos[file_name] = info
        self.summary = infos

    def save_chunks(self, folder_name, original_file_name):
        file_path = os.path.join(WORKING_DIR, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, self.subject, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for index, chunk in enumerate(self.chunks):
            np.save(os.path.join(file_path, "{}_{}_{}".format(folder_name, original_file_name, index)), chunk)

    def save_extra_chunks(self, folder_name, original_file_name):
        file_path = os.path.join(WORKING_DIR, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, self.subject, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for index, chunk in enumerate(self.extra_chunks):
            np.save(os.path.join(file_path, "{}_ex_{}_{}".format(folder_name, original_file_name, index)), chunk)

    @abstractmethod
    def generate_data_chunks(self, subject):
        pass
