"""Univariate and multivariate outlier detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.logger import get_logger
from omni_eda.utils import sample_df


@dataclass
class OutlierResult:
    method: str
    column: str | None  # None for multivariate methods
    mask: pd.Series  # boolean Series aligned to the (possibly sampled) frame's index
    n_outliers: int
    pct_outliers: float
    bounds: dict[str, float] = field(default_factory=dict)


def zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean, std = s.mean(), s.std(ddof=0)
    if not std:
        return pd.Series(False, index=series.index)
    z = (s - mean) / std
    return z.abs() > threshold


def modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median()
    mad = (s - median).abs().median()
    if not mad:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (s - median) / mad
    return modified_z.abs() > threshold


def iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (s < lower) | (s > upper)


def detect_univariate_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    methods: list[str],
    config: EDAConfig | None = None,
) -> dict[str, dict[str, OutlierResult]]:
    """Returns {column: {method: OutlierResult}} for every requested univariate method."""
    cfg = config or EDAConfig()
    fns = {
        "zscore": lambda s: zscore_outliers(s, cfg.zscore_threshold),
        "modified_zscore": lambda s: modified_zscore_outliers(s, cfg.zscore_threshold),
        "iqr": lambda s: iqr_outliers(s, cfg.iqr_multiplier),
    }
    results: dict[str, dict[str, OutlierResult]] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dropna().empty:
            continue
        col_results: dict[str, OutlierResult] = {}
        for method in methods:
            if method not in fns:
                continue
            mask = fns[method](series).fillna(False)
            n = int(mask.sum())
            col_results[method] = OutlierResult(
                method=method, column=col, mask=mask, n_outliers=n, pct_outliers=100 * n / len(series) if len(series) else 0.0
            )
        if col_results:
            results[col] = col_results
    return results


def detect_multivariate_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    config: EDAConfig | None = None,
) -> dict[str, OutlierResult]:
    """Isolation Forest / LOF / Elliptic Envelope over all numeric columns jointly."""
    cfg = config or EDAConfig()
    logger = get_logger()
    results: dict[str, OutlierResult] = {}

    if not cfg.enable_model_based_outliers or len(numeric_cols) < 2:
        return results

    working = sample_df(df, cfg.max_rows_for_expensive_ops, cfg.random_state)
    matrix = working[numeric_cols].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.shape[1] < 2 or matrix.shape[0] < 10:
        return results

    contamination = "auto"

    if "isolation_forest" in cfg.outlier_methods:
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(random_state=cfg.random_state, contamination=contamination, n_jobs=cfg.n_jobs)
            preds = model.fit_predict(matrix)
            mask = pd.Series(preds == -1, index=matrix.index)
            n = int(mask.sum())
            results["isolation_forest"] = OutlierResult("isolation_forest", None, mask, n, 100 * n / len(mask))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isolation Forest outlier detection skipped: %s", exc)

    if "lof" in cfg.outlier_methods:
        try:
            from sklearn.neighbors import LocalOutlierFactor

            n_neighbors = min(20, max(2, matrix.shape[0] - 1))
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=cfg.n_jobs)
            preds = model.fit_predict(matrix)
            mask = pd.Series(preds == -1, index=matrix.index)
            n = int(mask.sum())
            results["lof"] = OutlierResult("lof", None, mask, n, 100 * n / len(mask))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LOF outlier detection skipped: %s", exc)

    if "elliptic_envelope" in cfg.outlier_methods and matrix.shape[1] <= 30:
        try:
            from sklearn.covariance import EllipticEnvelope

            model = EllipticEnvelope(random_state=cfg.random_state, contamination=0.05)
            preds = model.fit_predict(matrix)
            mask = pd.Series(preds == -1, index=matrix.index)
            n = int(mask.sum())
            results["elliptic_envelope"] = OutlierResult("elliptic_envelope", None, mask, n, 100 * n / len(mask))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Elliptic Envelope outlier detection skipped: %s", exc)

    if "dbscan" in cfg.outlier_methods:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(matrix)
            labels = DBSCAN(eps=1.5, min_samples=max(5, matrix.shape[1] + 1)).fit_predict(scaled)
            mask = pd.Series(labels == -1, index=matrix.index)
            n = int(mask.sum())
            results["dbscan"] = OutlierResult("dbscan", None, mask, n, 100 * n / len(mask))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DBSCAN outlier detection skipped: %s", exc)

    return results


def summarize_outliers(
    univariate: dict[str, dict[str, OutlierResult]],
    multivariate: dict[str, OutlierResult],
) -> pd.DataFrame:
    rows = []
    for col, methods in univariate.items():
        for method, result in methods.items():
            rows.append(
                {"column": col, "method": method, "n_outliers": result.n_outliers, "pct_outliers": round(result.pct_outliers, 2)}
            )
    for method, result in multivariate.items():
        rows.append(
            {
                "column": "<all numeric columns>",
                "method": method,
                "n_outliers": result.n_outliers,
                "pct_outliers": round(result.pct_outliers, 2),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)
        if rows
        else pd.DataFrame(columns=["column", "method", "n_outliers", "pct_outliers"])
    )

def explain_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    multivariate_results: dict[str, OutlierResult],
    config: EDAConfig | None = None
) -> dict[str, Any]:
    """Provide feature-level explanations for why multivariate outliers were flagged.
    
    Returns a dictionary mapping the outlier algorithm to its top flagged rows
    and the features that deviated the most from the median.
    """
    cfg = config or EDAConfig()
    
    if not multivariate_results:
        return {}
        
    explanations = {}
    
    # Pre-calculate median and MAD for all numeric columns to score deviation
    matrix = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median()
    mads = (matrix - medians).abs().median()
    
    # Avoid zero division
    mads[mads == 0] = 1.0

    for method, result in multivariate_results.items():
        if result.n_outliers == 0:
            continue
            
        # Get up to 10 outlier rows to explain
        outlier_indices = result.mask[result.mask].index[:10]
        method_explanations = []
        
        for idx in outlier_indices:
            row_data = matrix.loc[idx]
            
            # Calculate modified z-scores for this row
            z_scores = (row_data - medians).abs() / mads
            z_scores = z_scores.sort_values(ascending=False)
            
            # Get top 3 features contributing to this being an outlier
            top_features = z_scores.head(3)
            
            reasons = []
            for feat, z in top_features.items():
                if pd.notna(z) and z > 2.0:
                    val = row_data[feat]
                    med = medians[feat]
                    direction = "high" if val > med else "low"
                    reasons.append(f"'{feat}' is unusually {direction} ({val:.2f} vs median {med:.2f})")
                    
            if reasons:
                method_explanations.append({
                    "row_index": str(idx),
                    "reasons": reasons
                })
                
        explanations[method] = method_explanations
        
    return explanations
