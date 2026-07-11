"""Automated A/B Testing and Hypothesis Testing module.

Performs statistical comparisons between two groups (Control vs. Variant)
across multiple continuous metrics, calculating statistical significance
and practical effect sizes (Cohen's d).
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


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_sd == 0:
        return 0.0
        
    return float((np.mean(group1) - np.mean(group2)) / pooled_sd)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d according to standard rules of thumb."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def run_ab_test(
    df: pd.DataFrame, 
    profiles: dict[str, ColumnProfile], 
    config: EDAConfig
) -> dict[str, Any] | None:
    """Run A/B testing on metrics grouped by the treatment column."""
    treatment_col = config.treatment_column
    if not treatment_col or treatment_col not in df.columns:
        return None

    # The treatment column must have exactly 2 unique values to run standard A/B
    treatment_series = df[treatment_col].dropna()
    unique_groups = treatment_series.unique()
    
    if len(unique_groups) != 2:
        logger.warning(
            "A/B testing skipped: Treatment column '%s' has %d unique values (must be exactly 2).", 
            treatment_col, len(unique_groups)
        )
        return None

    group_a_name, group_b_name = str(unique_groups[0]), str(unique_groups[1])
    group_a_mask = df[treatment_col] == unique_groups[0]
    group_b_mask = df[treatment_col] == unique_groups[1]
    
    # If explicit metrics are provided, use them. Otherwise, test all numeric columns.
    metric_cols = config.ab_metric_columns
    if not metric_cols:
        metric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant and c != treatment_col]

    results = []
    
    for metric in metric_cols:
        if metric not in df.columns:
            continue
            
        series_a = pd.to_numeric(df.loc[group_a_mask, metric], errors="coerce").dropna().values
        series_b = pd.to_numeric(df.loc[group_b_mask, metric], errors="coerce").dropna().values
        
        if len(series_a) < 5 or len(series_b) < 5:
            continue

        mean_a, mean_b = float(np.mean(series_a)), float(np.mean(series_b))
        
        # T-Test (Welch's unequal variances)
        try:
            t_stat, p_val_t = stats.ttest_ind(series_a, series_b, equal_var=False)
        except Exception:
            t_stat, p_val_t = np.nan, np.nan
            
        # Mann-Whitney U (Non-parametric)
        try:
            u_stat, p_val_u = stats.mannwhitneyu(series_a, series_b, alternative='two-sided')
        except Exception:
            u_stat, p_val_u = np.nan, np.nan
            
        # Effect Size
        d = cohens_d(series_b, series_a) # B relative to A
        
        # Determine if significant (alpha = 0.05)
        is_significant = bool(p_val_t < 0.05) if not np.isnan(p_val_t) else False
        
        results.append({
            "metric": metric,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "diff": mean_b - mean_a,
            "diff_pct": ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else np.nan,
            "t_stat": float(t_stat),
            "p_value_t": float(p_val_t),
            "p_value_u": float(p_val_u),
            "cohens_d": d,
            "effect_size": interpret_effect_size(d),
            "significant": is_significant
        })
        
    return {
        "treatment_column": treatment_col,
        "group_a": group_a_name,
        "group_b": group_b_name,
        "group_a_count": int(group_a_mask.sum()),
        "group_b_count": int(group_b_mask.sum()),
        "results": sorted(results, key=lambda x: (not x["significant"], abs(x["p_value_t"])))
    }
