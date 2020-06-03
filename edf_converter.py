from data_generator import DataProducer

patients = [ "chb24"]

for p in patients:
    producer = DataProducer()
    producer.data_file_creation(18, p)
    del (producer)
