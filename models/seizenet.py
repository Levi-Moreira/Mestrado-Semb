from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam

from constants import WINDOW_SIZE


class SeizeNet:
    def __init__(self, channels):
        self.model = None
        self.channels = channels
        self.mc = ModelCheckpoint('best_model_{}.h5'.format(self.channels), monitor='val_loss', mode='min',
                                  save_best_only=True)
        self.build_model()

        print("STARTING CREATING MODEL:")
        print('best_model_{}.h5'.format(self.channels))

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

        self.model.add(Conv2D(64, (1, 10), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((1, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model

    def get_model_summary(self):
        self.model.summary()

    def build(self):
        self.model.compile(optimizer=Adam(lr=0.00041),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit_generator(generator=train_generator, epochs=10,
                                                validation_data=test_generator, callbacks=[self.mc, ])

    def load_model(self, file):
        self.model = load_model(file)
        self.model.summary()
