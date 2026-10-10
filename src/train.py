"""Train, evaluate, and save the baseline churn model."""
import json
from pathlib import Path

import joblib
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import build_pipeline

RANDOM_STATE = 42


def main():
    df = load_data()
    y = df.pop("Churn").map({"No": 0, "Yes": 1})
    X = df.drop(columns=["customerID"])
    if y.isna().any():
        raise ValueError("Unexpected target labels; expected Yes/No.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(y_test)),
        "threshold": 0.5,
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "average_precision": float(average_precision_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "random_state": RANDOM_STATE,
    }
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(model, "models/churn_pipeline.joblib")
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
