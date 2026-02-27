import numpy as np
from tensorflow import keras


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test