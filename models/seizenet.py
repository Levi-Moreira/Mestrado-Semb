from keras import Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from interface.constants import WINDOW_SIZE


class SeizeNet:
    def __init__(self, channels):
        self.channels = channels
        self.mc = ModelCheckpoint('best_model_{epoch:02d}.h5', monitor='val_loss', mode='min',
                                  save_best_only=False)

        self.model = self.build_model()
        print("STARTING CREATING MODEL:")
        print('best_model_{}.h5'.format(self.channels))
        self.plot_model_arch()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.channels, WINDOW_SIZE, 1)))
        model.add(Conv2D(20, kernel_size=(1, 10), activation='relu'))

        model.add(Conv2D(20, (18, 20), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((1, 2)))
        model.add(Dropout(0.2))
        model.add(Reshape((20, 626, 1)))

        model.add(Conv2D(40, (20, 10), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((1, 2)))
        model.add(Dropout(0.2))
        model.add(Reshape((40, 308, 1)))

        model.add(Conv2D(80, (40, 10), activation='relu'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        return model

    def plot_model_arch(self):
        plot_model(
            self.model,
            to_file="model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    def get_model_summary(self):
        self.model.summary()

    def build(self):
        self.model.compile(optimizer=Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit(train_generator, epochs=90,
                                      validation_data=test_generator, callbacks=[self.mc, ])

    def load_model(self, file):
        print("Loading model from file{}".format(file))
        self.model.load_weights(file)
        self.model.summary()
