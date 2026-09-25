
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
DATA_PATH = Path("data/processed/data.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

TARGET = "target"


def main():

    # 1. Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 3. Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 4. Combine features and target for training data
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    # 5. Combine features and target for testing data
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    # 6. Create output directory
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 7. Save train and test datasets
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # 8. Display results
    print("Preprocessing completed successfully")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    print("Train data saved to:", TRAIN_PATH)
    print("Test data saved to:", TEST_PATH)


if __name__ == "__main__":
    main()