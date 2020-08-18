from keras import applications
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model, Sequential

from models.generator import DataGenerator


def get_files_split():
    train_data = [line.rstrip('\n') for line in open('/home/levi/PycharmProjects/Mestrado-Semb/train.txt')]
    val_data = [line.rstrip('\n') for line in open('/home/levi/PycharmProjects/Mestrado-Semb/val.txt')]
    return train_data, val_data


def create_base_model():
    base_model = applications.VGG16(weights='imagenet',
                                    include_top=False,
                                    input_shape=(224, 224, 3))

    base_model.trainable = False
    return base_model


from keras import backend as K

K.clear_session()

base_model = create_base_model()
custom_model = Sequential([
    base_model,
    Conv2D(32, 3, activation='relu'),
    Dropout(0.2),
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

custom_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
custom_model.summary()
train_data, val_data = get_files_split()
train_data_generator = DataGenerator(train_data, 18)
val_data_generator = DataGenerator(val_data, 18)

custom_model.fit(train_data_generator, validation_data=val_data_generator, epochs=100)
