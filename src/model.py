import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    filters=32,
    kernel_size=(3,3),
    dropout_rate=0.5,
    learning_rate=0.001,
    use_batchnorm=True
):
    model = keras.Sequential()

    model.add(layers.Conv2D(filters, kernel_size,
                            activation='relu',
                            input_shape=(28,28,1)))

    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model