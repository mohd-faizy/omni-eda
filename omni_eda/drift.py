"""Data Drift Detection Module.

Calculates drift metrics (PSI, KS Test) between two datasets
(e.g., train vs test, or time period A vs time period B).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger

logger = get_logger()


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two numeric arrays.
    
    Rule of thumb for PSI:
    < 0.1: No significant population change
    0.1 - 0.2: Moderate population change
    > 0.2: Significant population change
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bins using quantiles of the expected distribution
    bins = np.quantile(expected, np.linspace(0, 1, num_bins + 1))
    bins[0] -= 1e-5  # slightly extend boundaries to catch all values
    bins[-1] += 1e-5
    
    # Avoid duplicate bin edges (can happen if data is highly skewed/constant)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    # Avoid zero division
    def substitute_zero(arr):
        return np.where(arr == 0, 0.0001, arr)

    expected_pct = substitute_zero(expected_pct)
    actual_pct = substitute_zero(actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def calculate_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Calculate PSI for categorical data."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)
    
    all_categories = set(expected_counts.index).union(set(actual_counts.index))
    
    expected_pct = np.array([expected_counts.get(cat, 0.0) for cat in all_categories])
    actual_pct = np.array([actual_counts.get(cat, 0.0) for cat in all_categories])
    
    # Avoid zero division
    def substitute_zero(arr):
        return np.where(arr == 0, 0.0001, arr)
        
    expected_pct = substitute_zero(expected_pct)
    actual_pct = substitute_zero(actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def compare_datasets(
    df_reference: pd.DataFrame, 
    df_current: pd.DataFrame, 
    profiles: dict[str, ColumnProfile],
    config: EDAConfig
) -> dict[str, Any]:
    """Compare two datasets and detect feature drift."""
    results = []
    
    common_cols = set(df_reference.columns).intersection(set(df_current.columns))
    
    for col in common_cols:
        profile = profiles.get(col)
        if not profile or config.is_ignored(col):
            continue
            
        ref_series = df_reference[col].dropna()
        cur_series = df_current[col].dropna()
        
        if len(ref_series) < 10 or len(cur_series) < 10:
            continue
            
        drift_data = {
            "column": col,
            "type": profile.base_type,
            "psi": 0.0,
            "drift_status": "No Drift",
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
        }
        
        if profile.is_numeric and not profile.is_constant:
            ref_vals = pd.to_numeric(ref_series, errors="coerce").dropna().values
            cur_vals = pd.to_numeric(cur_series, errors="coerce").dropna().values
            
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                # Calculate PSI
                psi = calculate_psi(ref_vals, cur_vals)
                drift_data["psi"] = psi
                
                # Calculate KS Test
                try:
                    ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
                    drift_data["ks_stat"] = float(ks_stat)
                    drift_data["ks_pvalue"] = float(ks_pval)
                except Exception:
                    pass
                
                if psi > 0.2:
                    drift_data["drift_status"] = "Severe Drift"
                elif psi > 0.1:
                    drift_data["drift_status"] = "Moderate Drift"
                    
        elif profile.is_categorical or profile.is_boolean:
            psi = calculate_categorical_psi(ref_series, cur_series)
            drift_data["psi"] = psi
            if psi > 0.2:
                drift_data["drift_status"] = "Severe Drift"
            elif psi > 0.1:
                drift_data["drift_status"] = "Moderate Drift"
                
        if drift_data["psi"] > 0:
            results.append(drift_data)
            
    # Sort by PSI descending
    results = sorted(results, key=lambda x: x["psi"], reverse=True)
    
    return {
        "reference_rows": len(df_reference),
        "current_rows": len(df_current),
        "n_drifted": sum(1 for r in results if r["psi"] >= 0.1),
        "details": results
    }
