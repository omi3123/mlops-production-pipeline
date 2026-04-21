from __future__ import annotations
import numpy as np
import pandas as pd

def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        return 0.0
    ref_bins = pd.cut(ref, bins=quantiles, include_lowest=True)
    cur_bins = pd.cut(cur, bins=quantiles, include_lowest=True)
    ref_dist = ref_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    cur_dist = cur_bins.value_counts(normalize=True, sort=False).reindex(ref_dist.index).fillna(1e-6).replace(0, 1e-6)
    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())

def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("MISSING").astype(str)
    cur = current.fillna("MISSING").astype(str)
    cats = sorted(set(ref.unique()).union(set(cur.unique())))
    ref_dist = ref.value_counts(normalize=True).reindex(cats).fillna(1e-6).replace(0, 1e-6)
    cur_dist = cur.value_counts(normalize=True).reindex(cats).fillna(1e-6).replace(0, 1e-6)
    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())

def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "Stable"
    if psi < 0.25:
        return "Watch"
    return "Drift"
