import datetime

from keras import Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, AveragePooling2D, \
    GlobalAveragePooling2D, Conv1D, DepthwiseConv2D, Activation, Lambda
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from tensorflow.python import confusion_matrix
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.python.keras.metrics import Precision, AUC, Recall
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.nn_ops import log_softmax_v2

from interface.constants import WINDOW_SIZE
from tensorflow.keras.constraints import max_norm


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class SeizeNet:
    def __init__(self, channels):
        self.channels = channels
        self.mc = ModelCheckpoint('best_model_{epoch:02d}.h5', monitor='val_loss', mode='min',
                                  save_best_only=False)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                           patience=20, min_lr=0.00001)
        self.model = self.build_model()
        print("STARTING CREATING MODEL:")
        print('best_model_{}.h5'.format(self.channels))
        self.plot_model_arch()

    def build_alt_mode(self):
        Chans = 18
        Samples = WINDOW_SIZE
        dropoutRate = 0.5
        nb_classes = 2
        input_main = Input((Chans, Samples, 1))
        block1 = Conv2D(25, (1, 5),
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(25, (Chans, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(dropoutRate)(block1)

        block2 = Conv2D(50, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(dropoutRate)(block2)

        block3 = Conv2D(100, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(dropoutRate)(block3)

        block4 = Conv2D(200, (1, 5),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(dropoutRate)(block4)

        flatten = Flatten()(block4)

        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=input_main, outputs=softmax)

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.channels, WINDOW_SIZE, 1)))

        model.add(Conv2D(25, (1, 10), input_shape=(self.channels, WINDOW_SIZE, 1),
                         kernel_constraint=max_norm(2., axis=(0, 1, 2))))

        model.add(Conv2D(25, (self.channels, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2))))
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5, ))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(0.5))

        model.add(
            Conv2D(50, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2))))
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5, ))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(0.5))

        model.add(
            Conv2D(100, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2))))
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5, ))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(0.5))

        model.add(
            Conv2D(200, (1, 10), kernel_constraint=max_norm(2., axis=(0, 1, 2))))
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5, ))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(2, kernel_constraint=max_norm(0.5)))
        model.add(Activation('softmax'))
        # model.add(Reshape((2, 1, 1)))
        # model.add(Lambda(log))
        # model.add(Lambda(_squeeze_final_output))
        return model

    def plot_model_arch(self):
        plot_model(
            self.model,
            to_file="model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    def get_model_summary(self):
        self.model.summary()

    def build(self):
        optimizer = Adam()
        lr_metric = get_lr_metric(optimizer)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy',
                                    Precision(), AUC(), Recall()
                                    ])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit(train_generator, epochs=100,
                                      validation_data=test_generator,
                                      callbacks=[self.mc, self.reduce_lr, self.tensorboard_callback])

    def load_model(self, file):
        print("Loading model from file{}".format(file))
        self.model.load_weights(file)
        self.model.summary()


def log(x):
    return log_softmax_v2(x, axis=1)


def _squeeze_final_output(x):
    assert x.shape[3] == 1
    x = x[:, :, :, 0]
    if x.shape[2] == 1:
        x = x[:, :, 0]
    return x
