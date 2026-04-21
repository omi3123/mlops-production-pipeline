from __future__ import annotations
import pandas as pd
from src.data_loader import load_reference, load_production
from src.drift import psi_numeric, psi_categorical, classify_psi
from src.features import NUMERIC_FEATURES

DEFAULT_CAT_FEATURES = ["Contract", "InternetService", "PaymentMethod"]

def build_live_drift_report() -> pd.DataFrame:
    reference = load_reference()
    production = load_production()
    current_batch = production["batch_date"].max()
    current = production[production["batch_date"] == current_batch]
    rows = []
    for col in NUMERIC_FEATURES:
        psi = psi_numeric(reference[col], current[col])
        rows.append({"feature": col, "type": "numeric", "psi": psi, "severity": classify_psi(psi)})
    for col in DEFAULT_CAT_FEATURES:
        psi = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi, "severity": classify_psi(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
