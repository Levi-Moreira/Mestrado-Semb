from interface.CHBNegativeEEGDatasetGenerator import CHBNegativeEEGDatasetGenerator
from interface.CHBPositiveEEGDatasetGenerator import CHBPositiveEEGDatasetGenerator


class CHBDataSegmentProducer:
    def __init__(self, patient):
        self.patient = patient

    def data_file_creation(self):
        positive_dataset_generator = CHBPositiveEEGDatasetGenerator(self.patient)
        print("Created positive chuncks {}".format(positive_dataset_generator.total_chunks))
        print("Created negative chuncks {}".format(positive_dataset_generator.total_extra_chunks))
        del positive_dataset_generator

        negative_dataset_generator = CHBNegativeEEGDatasetGenerator(self.patient)
        print("Created negative chuncks {}".format(negative_dataset_generator.total_chunks))
        del negative_dataset_generator
