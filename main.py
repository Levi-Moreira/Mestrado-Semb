from data_generator import DataGenerator, DataProducer
from models.seizenet import SeizeNet

# data_generator = DataGenerator()
#
# train_list, test_list = data_generator.create_files_split()
#


# CHANNELS = 2
# producer = DataProducer()
# producer.generate_files_split()
# train_data, val_data = producer.get_files_split()
# train_data_generator = DataGenerator(train_data, CHANNELS)
# val_data_generator = DataGenerator(val_data, CHANNELS)
# net = SeizeNet(CHANNELS)
# net.get_model_summary()
# net.build()
# net.fit_data(train_data_generator, val_data_generator)
#
# CHANNELS = 13
# producer = DataProducer()
# # producer.generate_files_split()
# train_data, val_data = producer.get_files_split()
# train_data_generator = DataGenerator(train_data, CHANNELS)
# val_data_generator = DataGenerator(val_data, CHANNELS)
# net = SeizeNet(CHANNELS)
# net.get_model_summary()
# net.build()
# net.fit_data(train_data_generator, val_data_generator)
#
CHANNELS = 23
producer = DataProducer()
# producer.generate_files_split()
train_data, val_data = producer.get_files_split()
train_data = train_data[0:int(len(train_data) / 2)]
val_data = val_data[0:int(len(val_data) / 2)]
train_data_generator = DataGenerator(train_data, CHANNELS)
val_data_generator = DataGenerator(val_data, CHANNELS)
net = SeizeNet(CHANNELS)
net.get_model_summary()
net.build()
net.fit_data(train_data_generator, val_data_generator)
