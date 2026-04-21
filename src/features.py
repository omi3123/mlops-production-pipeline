from __future__ import annotations
import pandas as pd

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned

def drop_model_extras(df: pd.DataFrame) -> pd.DataFrame:
    extras = [
        "score", "predicted_label", "priority_band", "case_status",
        "analyst_owner", "sla_hours_remaining", "batch_date", "event_id",
        "actual_churn", "target",
    ]
    return df.drop(columns=[c for c in extras if c in df.columns], errors="ignore")
