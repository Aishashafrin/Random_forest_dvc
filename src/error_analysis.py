import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LABEL = "target"  # use your actual label column name

def main():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    test = pd.read_csv("data/processed/test.csv")
    X, y = test.drop(columns=[LABEL]), test[LABEL]

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["malignant (0)", "benign (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

    # Save misclassified rows with the model's confidence
    errors = test.copy()
    errors["predicted"] = pred
    errors["prob_benign"] = proba
    errors = errors[errors[LABEL] != errors["predicted"]]
    errors.to_csv("errors.csv", index=False)

    tn, fp, fn, tp = cm.ravel()
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Malignant misclassified: {fp}/{tn+fp} = {fp/(tn+fp):.1%}")
    print(f"Benign misclassified:    {fn}/{fn+tp} = {fn/(fn+tp):.1%}")

if __name__ == "__main__":
    main()