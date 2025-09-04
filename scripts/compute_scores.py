import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def main() -> None:
    csv_path = os.path.join("data", "emails.csv")
    df = pd.read_csv(csv_path)
    X = df["texte"].astype(str)
    y = df["label"].astype(str)

    model = joblib.load("model.pkl")
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label="spam")
    rec = recall_score(y, y_pred, pos_label="spam")
    f1 = f1_score(y, y_pred, pos_label="spam")
    roc = roc_auc_score((y == "spam").astype(int), proba) if proba is not None else float("nan")
    prauc = (
        average_precision_score((y == "spam").astype(int), proba)
        if proba is not None
        else float("nan")
    )

    print(f"accuracy={acc:.4f}")
    print(f"precision_spam={prec:.4f}")
    print(f"recall_spam={rec:.4f}")
    print(f"f1_spam={f1:.4f}")
    print(f"roc_auc={roc:.4f}")
    print(f"pr_auc={prauc:.4f}")


if __name__ == "__main__":
    main()


