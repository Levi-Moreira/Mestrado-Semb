from tensorflow.python.keras.models import Model
import numpy as np
from interface.constants import WINDOW_SIZE
from models.generator import _load_data
from models.seizenet import SeizeNet

train_data = [line.rstrip('\n') for line in open('val.txt')]

seize_net = SeizeNet(18)
seize_net.load_model("best_model_180.h5")

layer_name = 'flatten'
intermediate_layer_model = Model(inputs=seize_net.model.input,
                                 outputs=seize_net.model.get_layer(layer_name).output)

count = 0
for train in train_data:
    count += 1
    print("Converting {}%".format(count * 100 / len(train_data)))
    data = _load_data(train, 18, (1, 18, WINDOW_SIZE, 1))
    intermediate_output = intermediate_layer_model(data)
    final_output = intermediate_output.numpy()
    final_output.reshape((4240))
    final_output = final_output.astype('float16')
    t = train.split("/")
    if "positive" in train:
        identity = "positive"
    else:
        identity = "negative"
    np.save("/home/levi/PycharmProjects/Mestrado-Semb/data/svm/test/{}/{}".format(identity, t[-1]), final_output)
