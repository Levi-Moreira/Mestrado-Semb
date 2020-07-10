import pyedflib

from interface.TUHBaseDatabaseGenerator import TUHBaseDatabaseGenerator
from interface.helper import get_positive_chunks_from_data, get_negative_chunks_from_data


class TUHPositiveEEGDatasetGenerator(TUHBaseDatabaseGenerator):

    def generate_data_chunks(self):
        for file in self.summary:
            for seizure in self.summary[file]["seiz"]:
                try:
                    data = self.reader.read_file_in_interval(file, seizure["start"], seizure["end"])
                except OSError:
                    print("Dropped {}".format(file))
                    continue
                cc = get_positive_chunks_from_data(data)
                self.chunks.extend(cc)
            self.save_chunks("positive", "extra", file)
            self.chunks.clear()


class TUHNegativeEEGDatasetGenerator(TUHBaseDatabaseGenerator):

    def generate_data_chunks(self):
        for file in self.summary:
            for seizure in self.summary[file]["bckg"]:
                try:
                    data = self.reader.read_file_in_interval(file, seizure["start"], seizure["end"])
                except OSError:
                    print("Dropped {}".format(file))
                    continue
                cc = get_negative_chunks_from_data(data)
                self.chunks.extend(cc)
            self.save_chunks("negative", "extra", file)
            self.chunks.clear()
