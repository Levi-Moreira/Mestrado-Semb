import os

import numpy as np
import pyedflib
from scipy.signal import resample

from constants import CHANNELS, OLD_SAMPLE_RATE, NEW_SAMPLE_RATE, WINDOW_SIZE, EEG_SIGNAL_NUMBER


class EEGReader:
    ORIGINAL_SAMPLE_RATE = OLD_SAMPLE_RATE
    FINAL_SAMPLE_RATE = NEW_SAMPLE_RATE
    CHANNEL_NUMBER = CHANNELS

    def read_file(self, file_path):
        eeg_file = pyedflib.EdfReader(file_path)
        n = EEG_SIGNAL_NUMBER
        number_of_seconds = int(eeg_file.getNSamples()[0]) / self.ORIGINAL_SAMPLE_RATE
        intended_number_of_samples = int(number_of_seconds * self.FINAL_SAMPLE_RATE)
        result = np.zeros((n, intended_number_of_samples), dtype=np.float16)
        for i in np.arange(n):
            original_signal = eeg_file.readSignal(i)
            # resample to 200hz
            result[i, :] = resample(original_signal, intended_number_of_samples)
        return result

    def read_file_in_interval(self, file_path, start, end):
        read_data = self.read_file(file_path)
        return read_data[:, start: end]


class BaseDatabaseGenerator:
    CHUNCK_SIZE = WINDOW_SIZE
    LIMIT_FILES = 2
    FILE_NAME_POSITION = 0
    NUMBER_OF_SEIZURE_POSITION = 3

    def __init__(self, subject):
        self.subject = subject
        self.__summary_builder()
        self.reader = EEGReader()
        self.chunks = []
        self.explorer = CHBFolderExporer(subject)

    def __summary_builder(self):
        file = open(os.path.join(os.getcwd(), *["data", self.subject, "{}-summary.txt".format(self.subject)]), "r")
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
        file_path = os.path.join(os.getcwd(), *["data", self.subject, folder_name])
        for index, chunck in enumerate(self.chunks):
            np.savetxt(os.path.join(file_path, "{}_{}.txt".format(folder_name, index)), chunck, delimiter=",")


class NegativeEEGDatasetGenerator(BaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)
        self.negative_files = self.explorer.negative_edf_files
        self.__generate_data_chuncks(subject)

    def __generate_data_chuncks(self, subject):
        for file in self.negative_files:
            file_path = os.path.join(os.getcwd(), *["data", subject, file])
            data = self.reader.read_file(file_path)
            self.chunks.extend(np.array_split(data, int(data.shape[1] / self.CHUNCK_SIZE), axis=1))


class PositiveEEGDatasetGenerator(BaseDatabaseGenerator):
    def __init__(self, subject):
        super().__init__(subject)
        self.positive_files = self.explorer.positive_edf_files
        self.__generate_data_chunks(subject)

    def __generate_data_chunks(self, subject):
        for file in self.positive_files:
            file_path = os.path.join(os.getcwd(), *["data", subject, file])
            if self.summary[file]["seizure_info"]:
                for info in self.summary[file]["seizure_info"]:
                    start_sample = info["start_time"] * self.reader.FINAL_SAMPLE_RATE
                    end_sample = info["end_time"] * self.reader.FINAL_SAMPLE_RATE
                    data = self.reader.read_file_in_interval(file_path, start_sample, end_sample)
                    cc = self.__get_positive_chunks_from_data(data)
                    self.chunks.extend(cc)

    def __get_positive_chunks_from_data(self, data):
        chuncks = []
        window_start = 0
        while (window_start + self.CHUNCK_SIZE) < data.shape[1]:
            chuncks.append(data[:, window_start:window_start + self.CHUNCK_SIZE])
            window_start += int(0.075 * self.reader.FINAL_SAMPLE_RATE)
        return chuncks


class CHBFolderExporer:
    def __init__(self, subject_folder):
        all_files = os.listdir(os.path.join(os.getcwd(), "data/{}".format(subject_folder)))
        edf_files = list(filter(lambda file: ".edf" in file, all_files))
        seizure_edf_files = list(filter(lambda file: ".seizures" in file, edf_files))
        self.negative_edf_files = list(set(map(lambda file: file.strip(".seizures"), edf_files)))
        self.positive_edf_files = list(map(lambda file: file.strip(".seizures"), seizure_edf_files))
