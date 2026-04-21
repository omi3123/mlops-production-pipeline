from __future__ import annotations
import math
import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH, PRODUCTION_PATH
from src.features import clean_telco
from src.predict import score_dataframe

def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_telco(df)
    df = df.drop(columns=["Churn"], errors="ignore").copy().reset_index(drop=True)

    rng = np.random.default_rng(42)
    n = min(1800, len(df))
    prod = df.sample(n, random_state=42).reset_index(drop=True)
    prod["event_id"] = [f"evt_{i:06d}" for i in range(len(prod))]
    days = pd.date_range("2026-01-01", periods=12, freq="7D")
    batch_size = math.ceil(len(prod) / len(days))
    prod["batch_date"] = np.repeat(days.astype(str), batch_size)[: len(prod)]

    late_mask = prod["batch_date"] >= str(days[-4].date())
    prod.loc[late_mask, "MonthlyCharges"] = (
        pd.to_numeric(prod.loc[late_mask, "MonthlyCharges"]) * rng.normal(1.12, 0.06, late_mask.sum())
    ).round(2)
    prod.loc[late_mask, "Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        p=[0.7, 0.2, 0.1],
        size=late_mask.sum(),
    )

    scored = score_dataframe(prod)
    scored["case_status"] = rng.choice(
        ["New", "Queued", "Investigating", "Resolved"],
        p=[0.42, 0.25, 0.21, 0.12],
        size=len(scored),
    )
    scored["analyst_owner"] = rng.choice(
        ["A. Khan", "M. Ali", "S. Fatima", "R. Ahmed", "Unassigned"],
        p=[0.18, 0.18, 0.18, 0.16, 0.30],
        size=len(scored),
    )
    scored["sla_hours_remaining"] = rng.integers(2, 48, size=len(scored))
    PRODUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PRODUCTION_PATH, index=False)
    print(f"Saved simulated production data to {PRODUCTION_PATH}")

if __name__ == "__main__":
    main()
