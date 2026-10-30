"""
Train classifiers on the UCI Bank Marketing dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "bank-additional" / "bank-additional-full.csv"

TARGET = "y"

DROP_COLUMNS = ["duration"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


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


def main() -> None:
    data = load_dataset()

    features = data.drop(columns=[TARGET])
    labels = (data[TARGET] == "yes").astype(int).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(x_train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    print(f"train={x_train_t.shape} test={x_test_t.shape}")


if __name__ == "__main__":
    main()
