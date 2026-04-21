import json
import pandas as pd
from src.config import (
    RAW_DATA_PATH,
    REFERENCE_PATH,
    PRODUCTION_PATH,
    MODEL_META_PATH,
    FEATURE_IMPORTANCE_PATH,
    DRIFT_REPORT_PATH,
    DRIFT_SUMMARY_PATH,
    BATCH_REPORT_PATH,
    SCORED_HOLDOUT_PATH,
)

def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)

def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PATH)

def load_production() -> pd.DataFrame:
    return pd.read_csv(PRODUCTION_PATH)

def load_metadata() -> dict:
    return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))

def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)

def load_drift_report() -> pd.DataFrame:
    return pd.read_csv(DRIFT_REPORT_PATH)

def load_drift_summary() -> dict:
    return json.loads(DRIFT_SUMMARY_PATH.read_text(encoding="utf-8"))

def load_batch_monitoring() -> pd.DataFrame:
    return pd.read_csv(BATCH_REPORT_PATH)

def load_scored_holdout() -> pd.DataFrame:
    return pd.read_csv(SCORED_HOLDOUT_PATH)
