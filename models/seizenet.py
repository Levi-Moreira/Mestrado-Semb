from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Lambda, Concatenate, LSTM, \
    TimeDistributed, GlobalAveragePooling2D, GlobalAveragePooling1D, Reshape, Conv1D
from keras.optimizers import Adam
from keras.utils import plot_model

from constants import WINDOW_SIZE


class SeizeNet:
    def __init__(self, channels):
        self.channels = channels
        self.mc = ModelCheckpoint('best_model_23_{epoch:02d}.h5', monitor='val_loss', mode='min',
                                  save_best_only=False)

        frequency_bands = [(4, 7), (7, 12), (12, 19), (19, 30), (30, 40)]

        input = Input(shape=(1, WINDOW_SIZE, self.channels))
        out = self.build_model(input)
        # branch_outputs = []
        # for index, band in enumerate(frequency_bands):
        #     out = Lambda(lambda x: x[:, index])(input)
        #     out = self.build_model(out)
        #     branch_outputs.append(out)

        # out = Concatenate()(branch_outputs)
        # out = Reshape((5,50))(out)
        # out = GlobalAveragePooling1D()(out)
        # out = Dense(10, activation='relu')(out)
        # out = BatchNormalization()(out)
        # out = Dropout(0.5)(out)
        #
        # out = Dense(len(frequency_bands))(out)
        out = Dense(2, activation='softmax')(out)
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
        model = Conv2D(8, (10, 1), activation='relu', data_format='channels_first')(input_tensor)
        # model = MaxPooling2D((2, 1), data_format='channels_first')(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(16, (8, 23), activation='relu', data_format='channels_first')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((2, 1), data_format='channels_first')(model)
        model = Dropout(0.2)(model)

        model = Reshape((1, 632, 16))(model)

        model = Conv2D(32, (10, 16), activation='relu', data_format='channels_first')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((2, 1), data_format='channels_first')(model)
        model = Dropout(0.2)(model)

        model = Reshape((1, 311, 32))(model)

        model = Conv2D(64, (10, 32), activation='relu', data_format='channels_first')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D((2, 1), data_format='channels_first')(model)
        model = Dropout(0.2)(model)

        model = Flatten()(model)

        # model = Dense(2048, activation='relu')(model)
        # model = BatchNormalization()(model)
        # model = Dropout(0.5)(model)
        # model = Dense(2048, activation='relu')(model)
        # model = BatchNormalization()(model)
        # model = Dropout(0.5)(model)

        return model

    def get_model_summary(self):
        self.model.summary()

    def build(self):
        self.model.compile(optimizer=Adam(lr=0.00041),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_data(self, train_generator, test_generator):
        self.history = self.model.fit_generator(generator=train_generator, epochs=30,
                                                validation_data=test_generator, callbacks=[self.mc, ])

    def load_model(self, file):
        print("Loading model from file{}".format(file))
        self.model.load_weights(file)
        self.model.summary()
