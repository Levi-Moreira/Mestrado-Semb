import datetime
from keras import Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, AveragePooling2D, \
    GlobalAveragePooling2D, Conv1D, DepthwiseConv2D, Activation, Lambda
from keras.optimizers import Adam, SGD
from keras.utils import plot_model

from interface.constants import WINDOW_SIZE

#
# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return optimizer.lr
#
#     return lr


class SeizeNet:
    def __init__(self, channels):
        self.channels = channels
        # self.mc = ModelCheckpoint('best_model_{epoch:02d}.h5', monitor='val_loss', mode='min',
        #                           save_best_only=True)
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                                    patience=20, min_lr=0.00001)
        self.model = self.build_model_functional()
        print("STARTING CREATING MODEL:")
        print('best_model_{}.h5'.format(self.channels))
        # self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(8, (1, 10), activation='relu', input_shape=(1, WINDOW_SIZE, self.channels)))
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(16, (1, 10), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(32, (1, 10), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (1, 10), strides=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(128, (1, 10), strides=2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        return self.model

    def build_model_functional(self):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=(1, 10), activation='elu', input_shape=(self.channels, WINDOW_SIZE, 1)))

        model.add(Conv2D(20, (18, 20), activation='elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((1, 2)))
        model.add(Dropout(0.2))
        model.add(Reshape((20, 242, 1)))

        model.add(Conv2D(40, (20, 10), activation='elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((1, 2)))
        model.add(Dropout(0.2))
        model.add(Reshape((40, 116, 1)))

        model.add(Conv2D(80, (40, 10), activation='elu'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

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
        # lr_metric = get_lr_metric(optimizer)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit(train_generator, epochs=90,
                                      validation_data=test_generator)

    def load_model(self, file):
        print("Loading model from file{}".format(file))
        self.model.load_weights(file)
        self.model.summary()

# def log(x):
#     return log_softmax_v2(x, axis=1)
#
#
# def _squeeze_final_output(x):
#     assert x.shape[3] == 1
#     x = x[:, :, :, 0]
#     if x.shape[2] == 1:
#         x = x[:, :, 0]
#     return x
