"""
Streamlit front end for the Bank Marketing classifiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"

TARGET = "y"
POSITIVE_LABEL = "yes"
METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")

# display name -> (artifact filename, key used in model/metrics.json)
MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.joblib", "logistic_regression"),
    "Decision Tree": ("decision_tree.joblib", "decision_tree"),
    "K-Nearest Neighbors": ("knn.joblib", "knn"),
    "Naive Bayes": ("naive_bayes.joblib", "naive_bayes"),
    "Support Vector Machine": ("svm.joblib", "svm"),
}


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    estimators = {
        name: joblib.load(MODEL_DIR / filename)
        for name, (filename, _) in MODEL_FILES.items()
    }
    return preprocessor, estimators


@st.cache_data
def load_default_test_data() -> pd.DataFrame | None:
    if DEFAULT_TEST_FILE.exists():
        return pd.read_csv(DEFAULT_TEST_FILE)
    return None


@st.cache_data
def load_reference_metrics() -> dict | None:
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def score(estimator, features):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def compute_metrics(estimator, features, labels) -> dict:
    predictions = estimator.predict(features)
    scores = score(estimator, features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, scores),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }, predictions


def main() -> None:
    st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")

    st.title("Bank Marketing Term-Deposit Classifier")
    st.caption(
        "Predicts whether a client subscribes to a term deposit (`y`). "
        "Five classifiers trained on the UCI Bank Marketing dataset."
    )

    preprocessor, estimators = load_artifacts()

    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox("Select a model", list(estimators))

    st.sidebar.subheader("Test data")
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    st.sidebar.caption(
        "The CSV must contain the input features and the target column `y`. "
        "A held-out split is preloaded if you do not upload anything."
    )

    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        source = f"uploaded file ({uploaded.name})"
    else:
        frame = load_default_test_data()
        source = "bundled test_data.csv"

    if frame is None:
        st.error("No data available. Upload a CSV containing the features and `y`.")
        return

    st.info(f"Scoring **{len(frame):,} rows** from {source} with {model_name}.")

    if TARGET not in frame.columns:
        st.error(f"The target column `{TARGET}` is missing from the data.")
        st.write("Columns found:", ", ".join(frame.columns))
        return

    features = frame.drop(columns=[TARGET])
    labels = (frame[TARGET].astype(str).str.strip() == POSITIVE_LABEL).astype(int).to_numpy()

    try:
        transformed = preprocessor.transform(features)
    except Exception as exc:  # noqa: BLE001 - surface the decoding problem to the user
        st.error(f"Could not apply preprocessing to the uploaded data: {exc}")
        st.write("Expected feature columns:", ", ".join(preprocessor.feature_names_in_))
        return

    metrics, predictions = compute_metrics(estimators[model_name], transformed, labels)

    st.subheader("Evaluation metrics")
    columns = st.columns(len(METRIC_NAMES))
    for column, name in zip(columns, METRIC_NAMES):
        column.metric(name, f"{metrics[name]:.4f}")

    reference = load_reference_metrics()
    if reference:
        baseline = reference.get(MODEL_FILES[model_name][1])
        if baseline:
            st.caption(
                "Trained-model baseline — Accuracy "
                f"{baseline['Accuracy']:.4f}, AUC {baseline['AUC']:.4f}, "
                f"F1 {baseline['F1']:.4f}, MCC {baseline['MCC']:.4f}"
            )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(labels, predictions)
        figure, axis = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["no", "yes"],
            yticklabels=["no", "yes"],
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            labels, predictions, target_names=["no", "yes"], output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.subheader("Model comparison")
    rows = []
    for name, estimator in estimators.items():
        other_metrics, _ = compute_metrics(estimator, transformed, labels)
        rows.append(
            {"ML Model Name": name,
             **{m: round(other_metrics[m], 4) for m in METRIC_NAMES}}
        )
    comparison = pd.DataFrame(rows)
    st.dataframe(
        comparison.style.highlight_max(subset=list(METRIC_NAMES), color="#c6efce"),
        use_container_width=True,
    )
    st.caption("Green cells mark the best score for each metric on this data.")

    with st.expander("Preview of scored data"):
        preview = frame.head(25).copy()
        preview["prediction"] = pd.Series(predictions, index=frame.index).map(
            {1: "yes", 0: "no"}
        ).head(25)
        st.dataframe(preview, use_container_width=True)


if __name__ == "__main__":
    main()
