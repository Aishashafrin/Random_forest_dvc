
from pathlib import Path
import pickle
import pandas as pd

# File paths
MODEL_PATH = Path("model.pkl")
TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_PATH = Path("reports/top_10_feature_importance.csv")


def main():

    # 1. Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # 2. Load training data
    train_df = pd.read_csv(TRAIN_PATH)

    # 3. Get feature names
    feature_names = train_df.drop(
        columns=["target"]
    ).columns

    # 4. Extract feature importance
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    # 5. Sort by importance
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    # 6. Select top 10 features
    top_10 = importance_df.head(10)

    # 7. Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 8. Save CSV
    top_10.to_csv(OUTPUT_PATH, index=False)

    print("Feature importance completed")
    print(top_10.to_string(index=False))
    print("Saved to:", OUTPUT_PATH)

# 9. Display results
    print("Top 10 Feature Importance:")
    print(top_10.to_string(index=False))

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()