
from pathlib import Path
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

# File paths
TRAIN_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("model.pkl")

TARGET = "target"


def main():

    # 1. Load hyperparameters from params.yaml
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)["train"]

    # 2. Load training dataset
    df = pd.read_csv(TRAIN_PATH)

    # 3. Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 4. Create Random Forest model
    model = RandomForestClassifier(**params)

    # 5. Train the model
    model.fit(X, y)

    # 6. Save the trained model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Model training completed successfully")
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()