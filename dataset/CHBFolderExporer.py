import os


class CHBFolderExporer:
    def __init__(self, subject_folder):
        all_files = os.listdir(
            os.path.join(os.getcwd(), "data/chb-mit-scalp-eeg-database-1.0.0/{}".format(subject_folder)))
        edf_files = list(filter(lambda file: ".edf" in file, all_files))
        seizure_edf_files = list(filter(lambda file: ".seizures" in file, edf_files))
        self.positive_edf_files = list(map(lambda file: file.strip(".seizures"), seizure_edf_files))

        self.negative_edf_files = list(set(map(lambda file: file.strip(".seizures"), edf_files)))
        self.negative_edf_files = list(set(self.negative_edf_files) - set(self.positive_edf_files))
        print()
