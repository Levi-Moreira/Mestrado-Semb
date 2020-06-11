import os
from datetime import datetime

import numpy as np

from constants import WINDOW_SIZE, MAIN_FOLDER_NAME
from dataset.CHBFolderExporer import CHBFolderExporer
from edf.EEGReader import EEGReader


class BaseDatabaseGenerator:
    CHUNCK_SIZE = WINDOW_SIZE
    # SHIFT_WINDOW = 0.075
    SHIFT_WINDOW = 10
    # SHIFT_WINDOW = 1
    LIMIT_FILES = 2
    FILE_NAME_POSITION = 0
    NUMBER_OF_SEIZURE_POSITION = 3

    def __init__(self, subject):
        self.subject = subject
        self.__summary_builder()
        self.reader = EEGReader()
        self.chunks = []
        self.extra_chunks = []
        self.total_chuncks = 0
        self.total_extra_chunks = 0
        self.explorer = CHBFolderExporer(subject)

    def __summary_builder(self):
        file = open(os.path.join(os.getcwd(), *["data", "chb-mit-scalp-eeg-database-1.0.0", self.subject,
                                                "{}-summary.txt".format(self.subject)]), "r")
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
                    info["seizure_info"] = []
                    lines = list(filter(lambda x: x is not "", lines))
                    for n in range(4, len(lines), 2):
                        seizure_info = {}
                        start_time = lines[n].split(":")[1]
                        seizure_info["start_time"] = int(start_time.split(" ")[1])
                        end_time = lines[n + 1].split(":")[1]
                        seizure_info["end_time"] = int(end_time.split(" ")[1])
                        info["seizure_info"].append(seizure_info)
                infos[file_name] = info
        self.summary = infos

    def save_chunks(self, folder_name):
        file_path = os.path.join(os.getcwd(), *["data", MAIN_FOLDER_NAME, self.subject, folder_name])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for index, chunck in enumerate(self.chunks):
            np.save(os.path.join(file_path, "{}_{}".format(folder_name, datetime.timestamp(datetime.now()))), chunck)

    def save_extra_chunks(self, folder_name):
        file_path = os.path.join(os.getcwd(), *["data", MAIN_FOLDER_NAME, self.subject, folder_name])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for index, chunck in enumerate(self.extra_chunks):
            np.save(os.path.join(file_path, "{}_ex_{}".format(folder_name, datetime.timestamp(datetime.now()))), chunck)
