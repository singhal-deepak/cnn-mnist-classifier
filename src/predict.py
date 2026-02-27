import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from utils import load_data
from sklearn.metrics import classification_report, confusion_matrix


def predict():
    print("Loading model...")
    model = keras.models.load_model("cnn_mnist_model.h5")

    print("Loading MNIST test data...")
    _, _, x_test, y_test = load_data()

    print("Running predictions...")
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    # Show first prediction
    print("\nSample Prediction:")
    print("Predicted:", y_pred[0])
    print("Actual:", y_test[0])

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.show()

    # Show 9 sample predictions
    plt.figure(figsize=(8,8))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap="gray")
        plt.title(f"Pred: {y_pred[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict()