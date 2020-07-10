import os
from abc import abstractmethod, ABC
from pathlib import Path
import numpy as np
import pyedflib

from constants import WORKING_DIR
from dataset.TUHFolderExplorer import TUHFolderExplorer
from dataset.constants import DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER
from edf.EEGReader import EEGReader


class TUHBaseDatabaseGenerator(ABC):
    LIMIT_FILES = 2
    FILE_NAME_POSITION = 0
    NUMBER_OF_SEIZURE_POSITION = 3

    def __init__(self):
        self.summary = {}
        self.explorer = TUHFolderExplorer()
        self.__summary_builder()
        self.reader = EEGReader()
        self.chunks = []
        self.total_chunks = 0
        self.generate_data_chunks()

    def __summary_builder(self):
        for positive_file in self.explorer.positive_edf_files[:150]:
            positive_file_path = Path(positive_file)
            information_file = positive_file_path.parent.joinpath(positive_file_path.stem + '.tse_bi')
            eeg_file = pyedflib.EdfReader(positive_file)
            seizure_info = []
            bckg_info = []
            for line in open(information_file):
                if "seiz" in line:
                    parts = line.split(" ")
                    seizure_start = parts[0]
                    seizure_end = parts[1]
                    seizure_info.append({
                        "start": int(float(seizure_start) * eeg_file.getSampleFrequencies()[0]),
                        "end": int(float(seizure_end) * eeg_file.getSampleFrequencies()[0])
                    })

                if "bckg" in line:
                    parts = line.split(" ")
                    b_start = parts[0]
                    b_end = parts[1]
                    bckg_info.append({
                        "start": int(float(b_start) * eeg_file.getSampleFrequencies()[0]),
                        "end": int(float(b_end) * eeg_file.getSampleFrequencies()[0])
                    })

            self.summary[positive_file] = {
                "seiz": seizure_info,
                "bckg": bckg_info
            }

    def save_chunks(self, folder_name, s, original_file_name):
        file_path = os.path.join(WORKING_DIR, DATA_SUBFOLDER_LOCATION, SEGMENTED_DATA_FOLDER, s, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for index, chunk in enumerate(self.chunks):
            if np.isnan(chunk).any():
                print("Dropped chunck {}".format(index))
                continue
            np.save(
                os.path.join(file_path, "{}_{}_{}".format(folder_name, os.path.basename(original_file_name), index)),
                chunk)

    @abstractmethod
    def generate_data_chunks(self):
        pass
