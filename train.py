"""
Train classifiers on the UCI Bank Marketing dataset.

Outputs:
  model/preprocessor.joblib   fitted ColumnTransformer
  model/<name>.joblib         one fitted estimator per algorithm
  model/metrics.json          evaluation metrics per model
  test_data.csv               held-out test split: original features + target
  data/comparison_table.csv   metric table
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "bank-additional" / "bank-additional-full.csv"
MODEL_DIR = ROOT / "model"
TEST_FILE = ROOT / "test_data.csv"
COMPARISON_FILE = ROOT / "data" / "comparison_table.csv"

TARGET = "y"
METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")

DROP_COLUMNS = ["duration"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
}


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, sep=";").drop(columns=DROP_COLUMNS)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = features.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def positive_class_score(estimator, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def evaluate(estimator, features: np.ndarray, labels: np.ndarray) -> dict:
    predictions = estimator.predict(features)
    scores = positive_class_score(estimator, features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, scores),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    data = load_dataset()

    features = data.drop(columns=[TARGET])
    labels = (data[TARGET] == "yes").astype(int).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(x_train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    print(f"train={x_train_t.shape} test={x_test_t.shape}")

    held_out = x_test.copy()
    held_out[TARGET] = np.where(y_test == 1, "yes", "no")
    held_out.to_csv(TEST_FILE, index=False)
    print(f"wrote {TEST_FILE.name} ({len(held_out)} rows)")

    results: dict[str, dict] = {}
    for key, estimator in MODELS.items():
        estimator.fit(x_train_t, y_train)
        joblib.dump(estimator, MODEL_DIR / f"{key}.joblib")
        metrics = evaluate(estimator, x_test_t, y_test)
        results[key] = metrics
        print(
            f"{DISPLAY_NAMES[key]:<24} acc={metrics['Accuracy']:.4f} "
            f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f} mcc={metrics['MCC']:.4f}"
        )

    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    table = pd.DataFrame(
        [
            {"ML Model Name": DISPLAY_NAMES[key],
             **{name: round(results[key][name], 4) for name in METRIC_NAMES}}
            for key in MODELS
        ]
    )
    table.to_csv(COMPARISON_FILE, index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
