
from pathlib import Path
import json
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# File paths
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")

TARGET = "target"


def main():

    # 1. Load test dataset
    test_df = pd.read_csv(TEST_PATH)

    # 2. Separate features and target
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # 3. Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 5. Get predicted probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Calculate evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_prob)
        )
    }

    # 7. Save metrics to JSON
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    # 8. Display results
    print("Model Evaluation Completed")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()