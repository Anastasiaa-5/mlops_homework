from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import pickle
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    if "accuracy_min" not in params:
        raise ValueError("Необходимо определить accuracy_min в params.yaml")
    return params


def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)


def load_data():
    return pd.read_csv("data/processed/dataset.csv")


def compute_accuracy(model, data):
    X = data[["total_bill", "size"]]
    y = data["high_tip"]
    params = load_params()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def validate_model():
    params = load_params()
    model = load_model()
    data = load_data()
    accuracy = compute_accuracy(model, data)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Минимально требуемая accuracy: {params['accuracy_min']}")

    if accuracy < params["accuracy_min"]:
        print(f"Валидация не выполнена: accuracy {accuracy:.4f} < {params['accuracy_min']}")
        sys.exit(1)
    else:
        print("Валидация выполнена")
        sys.exit(0)

if __name__ == "__main__":
    validate_model()