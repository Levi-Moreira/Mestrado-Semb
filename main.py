from data_generator import DataGenerator, DataProducer
from models.seizenet import SeizeNet

# data_generator = DataGenerator()
#
# train_list, test_list = data_generator.create_files_split()
#
producer = DataProducer()
producer.generate_files_split()
train_data, val_data = producer.get_files_split()
train_data_generator = DataGenerator(train_data)
val_data_generator = DataGenerator(val_data)

net = SeizeNet()
net.get_model_summary()
net.build()
net.fit_data(train_data_generator, val_data_generator)
