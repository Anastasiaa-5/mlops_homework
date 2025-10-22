import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import json
import pickle

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    rows = len(df)
    metrics = {
        "accuracy": float(accuracy),
        "rows": int(rows)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_metrics(metrics)

    mlflow.log_artifact("metrics/metrics.json")

    print(f"Сохранённые метрики: accuracy={accuracy:.4f}, num_rows={rows}")

if __name__ == "__main__":
    with mlflow.start_run():
        evaluate_model()