from interface.NegativeEEGDatasetGenerator import NegativeEEGDatasetGenerator
from interface.PositiveEEGDatasetGenerator import PositiveEEGDatasetGenerator


class DataSegmentProducer:
    def __init__(self, patient):
        self.patient = patient

    def data_file_creation(self):
        positive_dataset_generator = PositiveEEGDatasetGenerator(self.patient)
        print("Created positive chuncks {}".format(positive_dataset_generator.total_chunks))
        print("Created negative chuncks {}".format(positive_dataset_generator.total_extra_chunks))
        del positive_dataset_generator

        negative_dataset_generator = NegativeEEGDatasetGenerator(self.patient)
        print("Created negative chuncks {}".format(negative_dataset_generator.total_chunks))
        del negative_dataset_generator
