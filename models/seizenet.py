from keras import Input, Model, optimizers
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Lambda, Concatenate
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from tensorflow_core.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from constants import WINDOW_SIZE


class SeizeNet:
    def __init__(self, channels):
        self.channels = channels
        self.mc = ModelCheckpoint('best_model_{}.h5'.format(self.channels), monitor='val_loss', mode='min',
                                  save_best_only=True)

        frequency_bands = [(4, 7), (7, 12), (12, 19), (19, 30), (30, 40)]

        input = Input(shape=(len(frequency_bands), 1, WINDOW_SIZE, self.channels))

        branch_outputs = []
        for index, band in enumerate(frequency_bands):
            out = Lambda(lambda x: x[:, index])(input)
            out = self.build_model(out)
            branch_outputs.append(out)

        out = Concatenate()(branch_outputs)
        out = Dense(len(frequency_bands))(out)
        out = Dense(1, activation='sigmoid')(out)
        self.model = Model(inputs=input, outputs=out)

        plot_model(
            self.model,
            to_file="model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

        print("STARTING CREATING MODEL:")
        print('best_model_{}.h5'.format(self.channels))

    def build_model(self, input_tensor):
        model = Conv2D(8, (1, 10), activation='relu')(input_tensor)
        model = MaxPooling2D((1, 2))(model)
        model = Dropout(0.2)(model)

        model = Conv2D(16, (1, 10), activation='relu')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((1, 2))(model)
        model = Dropout(0.2)(model)

        model = Conv2D(32, (1, 10), activation='relu')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((1, 2))(model)
        model = Dropout(0.2)(model)

        model = Conv2D(64, (1, 10), activation='relu')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((1, 2))(model)
        model = Dropout(0.2)(model)

        model = Flatten()(model)

        model = Dense(10, activation='relu')(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)


        return model

    def get_model_summary(self):
        self.model.summary()

    def build(self):
        self.model.compile(optimizer=Adam(lr=0.00041),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit_generator(generator=train_generator, epochs=20,
                                                validation_data=test_generator, callbacks=[self.mc, ])

    def load_model(self, file):
        self.model.load_weights(file)
        self.model.summary()
