from __future__ import annotations
import json
import joblib
import pandas as pd
from src.config import MODEL_PATH, MODEL_META_PATH
from src.features import clean_telco, drop_model_extras

_MODEL = None
_METADATA = None

def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL

def load_metadata():
    global _METADATA
    if _METADATA is None:
        _METADATA = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    return _METADATA

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    clean = drop_model_extras(clean_telco(df))
    probs = model.predict_proba(clean)[:, 1]
    result = df.copy()
    result["score"] = probs
    result["predicted_label"] = (probs >= load_metadata()["threshold"]).astype(int)
    result["priority_band"] = pd.cut(
        result["score"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    return result

def score_record(record: dict) -> dict:
    df = pd.DataFrame([record])
    return score_dataframe(df).iloc[0].to_dict()
