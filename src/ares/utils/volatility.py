from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_iqr(values: pd.Series) -> float:
    """IQR on log-space values with safe numeric coercion."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return 0.0
    return float(np.percentile(clean_values, 75) - np.percentile(clean_values, 25))


def shrink_to_global(
    local_value: float,
    n_listings: float,
    global_value: float,
    k_smoothing: float,
    force_full_weight: bool = False,
) -> float:
    """Empirical Bayes-style shrinkage toward global value."""
    baseline = float(global_value)
    local = float(local_value) if np.isfinite(local_value) else baseline

    if force_full_weight:
        return local

    n_value = max(float(n_listings), 0.0)
    if k_smoothing <= 0:
        weight = 1.0
    else:
        weight = n_value / (n_value + float(k_smoothing))
    return float((weight * local) + ((1.0 - weight) * baseline))


def derive_volatility_thresholds(values: pd.Series) -> tuple[float, float]:
    """Returns stable Q25/Q75 thresholds from a series of log-IQR values."""
    clean_values = pd.to_numeric(values, errors="coerce")
    clean_values = clean_values[np.isfinite(clean_values)]

    if clean_values.empty:
        return 0.0, 1e-6

    q25 = float(clean_values.quantile(0.25))
    q75 = float(clean_values.quantile(0.75))
    if q75 <= q25:
        q75 = q25 + 1e-6
    return q25, q75


def classify_volatility_tier(log_iqr: float, q25: float, q75: float) -> str:
    """Maps log-IQR to calibrated tier labels."""
    if log_iqr <= q25:
        return "Stable"
    if log_iqr <= q75:
        return "Moderate"
    return "Volatile"


def log_iqr_to_relative_pct(log_iqr: float) -> float:
    """Converts log-IQR to relative IQR percentage in price space."""
    return float((np.exp(float(log_iqr)) - 1.0) * 100.0)
