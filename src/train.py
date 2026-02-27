import tensorflow as tf
from model import build_model
from utils import load_data


def train():
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        epochs=15,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_acc)

    model.save("cnn_mnist_model.h5")


if __name__ == "__main__":
    train()