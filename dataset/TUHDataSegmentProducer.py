from interface.TUHPositiveEEGDatasetGenerator import TUHPositiveEEGDatasetGenerator, TUHNegativeEEGDatasetGenerator


class TUHDataSegmentProducer:

    def data_file_creation(self):
        positive_dataset_generator = TUHPositiveEEGDatasetGenerator()
        print("Created positive chuncks {}".format(positive_dataset_generator.total_chunks))

        negative_dataset_generator = TUHNegativeEEGDatasetGenerator()
        print("Created negative chuncks {}".format(negative_dataset_generator.total_chunks))
