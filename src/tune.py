import pandas as pd
from model import build_model
from utils import load_data


def run_experiment(config):
    x_train, y_train, x_test, y_test = load_data()

    model = build_model(**config)

    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_split=0.2,
        verbose=0
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    return acc


if __name__ == "__main__":

    experiments = [
        {"filters": 32, "kernel_size": (3,3), "dropout_rate": 0.5, "learning_rate": 0.001, "use_batchnorm": True},
        {"filters": 64, "kernel_size": (3,3), "dropout_rate": 0.3, "learning_rate": 0.001, "use_batchnorm": True},
        {"filters": 32, "kernel_size": (5,5), "dropout_rate": 0.5, "learning_rate": 0.0005, "use_batchnorm": False},
        {"filters": 64, "kernel_size": (5,5), "dropout_rate": 0.2, "learning_rate": 0.001, "use_batchnorm": True},
    ]

    results = []

    for config in experiments:
        acc = run_experiment(config)
        config["test_accuracy"] = acc
        results.append(config)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("tuning_results.csv", index=False)