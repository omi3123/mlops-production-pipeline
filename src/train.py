from __future__ import annotations
import argparse
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_DATA_PATH, MODEL_PATH, MODEL_META_PATH
from src.features import clean_telco, NUMERIC_FEATURES, TARGET_COLUMN, ID_COLUMN

def main(smoke: bool = False) -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_telco(df)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    if smoke:
        df = df.sample(min(1200, len(df)), random_state=42)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    categorical_features = [c for c in X.columns if c not in [ID_COLUMN] + NUMERIC_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    metadata = {
        "model_name": "telco-churn-prod-monitor",
        "version": "1.0.0",
        "algorithm": "LogisticRegression + OneHotEncoder + StandardScaler",
        "task": "binary_classification",
        "threshold": 0.5,
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "average_precision": float(average_precision_score(y_test, probs)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "rows": int(len(df)),
        "smoke_mode": smoke,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
