from tensorflow.python.keras.models import Model
import numpy as np
from interface.constants import WINDOW_SIZE
from models.generator import _load_data
from models.seizenet import SeizeNet

final_train_data = [line.rstrip('\n') for line in open('train.txt')]
test_data = [line.rstrip('\n') for line in open('test2.txt')]

# test_data = list(filter(lambda x: "chb12" in x, train_data))
# final_train_data = list(filter(lambda x: "chb12" not in x, train_data))

seize_net = SeizeNet(18)
seize_net.load_model("best_model_01.h5")

layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=seize_net.model.input,
                                 outputs=seize_net.model.get_layer(layer_name).output)

count = 0
for train in final_train_data:
    count += 1
    print("Converting {}%".format(count * 100 / len(final_train_data)))
    data = _load_data(train, 18, (1, 1, WINDOW_SIZE, 18))
    intermediate_output = intermediate_layer_model(data)
    final_output = intermediate_output.numpy()
    final_output.reshape((896))
    final_output = final_output.astype('float16')
    t = train.split("/")
    if "positive" in train:
        identity = "positive"
    else:
        identity = "negative"
    np.save("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/train/{}/{}".format(identity, t[-1]), final_output)

count = 0
for test in test_data:
    count += 1
    print("Converting {}%".format(count * 100 / len(test_data)))
    data = _load_data(test, 18, (1, 1, WINDOW_SIZE, 18))
    intermediate_output = intermediate_layer_model(data)
    final_output = intermediate_output.numpy()
    final_output.reshape((896))
    final_output = final_output.astype('float16')
    t = test.split("/")
    if "positive" in test:
        identity = "positive"
    else:
        identity = "negative"
    np.save("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/{}/{}".format(identity, t[-1]), final_output)
