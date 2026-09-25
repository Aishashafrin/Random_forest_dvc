from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer(as_frame=True)
df = dataset.frame.copy()
df.to_csv("data/raw/data.csv", index=False)


from pathlib import Path
import pandas as pd

# Define file paths
RAW_PATH = Path("data/raw/data.csv")
PROCESSED_PATH = Path("data/processed/data.csv")


def main():

    # 1. Load raw dataset
    df = pd.read_csv(RAW_PATH)

    # 2. Basic validation
    if df.empty:
        raise ValueError("Dataset is empty")

    if "target" not in df.columns:
        raise ValueError("Target column is missing")

    # 3. Check missing values
    print("Missing values:")
    print(df.isnull().sum().sum())

    # 4. Handle missing values
    df = df.dropna()

    # 5. Save cleaned dataset
    PROCESSED_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(PROCESSED_PATH, index=False)

    print("Data loading completed successfully")
    print("Dataset shape:", df.shape)
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()