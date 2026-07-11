"""Correlation / association analysis across numeric and categorical columns."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import distance as scipy_distance

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.utils import sample_df
from omni_eda.advanced_correlation import AdvancedCorrelation


def numeric_correlation_matrices(df: pd.DataFrame, numeric_cols: list[str], methods: list[str]) -> dict[str, pd.DataFrame]:
    """Pearson / Spearman / Kendall correlation matrices for numeric columns."""
    matrices: dict[str, pd.DataFrame] = {}
    if len(numeric_cols) < 2:
        return matrices
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for method in methods:
        try:
            matrices[method] = numeric_df.corr(method=method, min_periods=2)
        except Exception:  # noqa: BLE001
            continue
    return matrices


def cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    """Bias-corrected Cramer's V association measure for two categorical series."""
    contingency = pd.crosstab(x, y)
    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    chi2 = scipy_stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0.0
    r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def categorical_association_matrix(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Symmetric matrix of Cramer's V between every pair of categorical columns."""
    n = len(categorical_cols)
    matrix = pd.DataFrame(np.eye(n), index=categorical_cols, columns=categorical_cols)
    for a, b in combinations(categorical_cols, 2):
        value = cramers_v(df[a], df[b])
        matrix.loc[a, b] = matrix.loc[b, a] = value if value is not None else np.nan
    return matrix


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Eta / correlation ratio: association between a categorical and a numeric column."""
    valid = pd.DataFrame({"cat": categories, "val": pd.to_numeric(values, errors="coerce")}).dropna()
    if valid.empty or valid["cat"].nunique() < 2:
        return None
    groups = valid.groupby("cat", observed=True)["val"]
    overall_mean = valid["val"].mean()
    ss_between = float(sum(len(g) * (g.mean() - overall_mean) ** 2 for _, g in groups))
    ss_total = float(((valid["val"] - overall_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def categorical_numeric_association_matrix(
    df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
) -> pd.DataFrame:
    """Correlation-ratio matrix (rows=categorical, cols=numeric)."""
    matrix = pd.DataFrame(index=categorical_cols, columns=numeric_cols, dtype=float)
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            matrix.loc[cat_col, num_col] = correlation_ratio(df[cat_col], df[num_col])
    return matrix


def mutual_information_matrix(
    df: pd.DataFrame, columns: list[str], profiles: dict[str, ColumnProfile], n_bins: int = 10
) -> pd.DataFrame:
    """Pairwise mutual information (nats), discretizing numeric columns into bins."""
    from sklearn.feature_selection import mutual_info_regression

    discretized: dict[str, np.ndarray] = {}
    for col in columns:
        profile = profiles.get(col)
        series = df[col]
        if profile and profile.is_numeric:
            values = pd.to_numeric(series, errors="coerce")
            try:
                codes = pd.qcut(values, q=min(n_bins, values.nunique()), duplicates="drop")
                discretized[col] = codes.astype(str).fillna("NA").to_numpy()
            except (ValueError, IndexError):
                discretized[col] = values.fillna(values.median() if values.notna().any() else 0).to_numpy()
        else:
            discretized[col] = series.astype(str).fillna("NA").to_numpy()

    n = len(columns)
    matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)
    from sklearn.preprocessing import LabelEncoder

    encoded = {}
    for col in columns:
        arr = discretized[col]
        if arr.dtype.kind in "fi":
            encoded[col] = arr.reshape(-1, 1)
        else:
            encoded[col] = LabelEncoder().fit_transform(arr).reshape(-1, 1)

    for a, b in combinations(columns, 2):
        try:
            mi = mutual_info_regression(encoded[a], encoded[b].ravel(), random_state=0, discrete_features=True)[0]
        except Exception:  # noqa: BLE001
            mi = np.nan
        matrix.loc[a, b] = matrix.loc[b, a] = mi
    np.fill_diagonal(matrix.values, np.nan)
    return matrix


def distance_correlation(x: pd.Series, y: pd.Series, sample_size: int = 2000) -> float | None:
    """Distance correlation (captures non-linear relationships), sampled for speed."""
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(valid) < 5:
        return None
    if len(valid) > sample_size:
        valid = valid.sample(n=sample_size, random_state=0)
    a = scipy_distance.squareform(scipy_distance.pdist(valid[["x"]].to_numpy()))
    b = scipy_distance.squareform(scipy_distance.pdist(valid[["y"]].to_numpy()))
    a_centered = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    b_centered = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (a_centered * b_centered).mean()
    dvar_x = (a_centered * a_centered).mean()
    dvar_y = (b_centered * b_centered).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0)) / np.sqrt(denom))


def find_highly_correlated_pairs(matrix: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    pairs = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"col_a": a, "col_b": b, "value": float(value)})
    return sorted(pairs, key=lambda d: -abs(d["value"]))


def compute_correlations(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
) -> dict[str, Any]:
    """Compute the full correlation/association suite, guarding against huge column counts."""
    cfg = config or EDAConfig()
    numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant][: cfg.max_cols_for_pairwise]
    categorical_cols = [
        c for c, p in profiles.items() if (p.is_categorical or p.is_boolean) and not p.is_constant and p.n_unique <= 100
    ][: cfg.max_cols_for_pairwise]

    working_df = sample_df(df, cfg.max_rows_for_expensive_ops, cfg.random_state)

    result: dict[str, Any] = {
        "numeric": numeric_correlation_matrices(working_df, numeric_cols, cfg.correlation_methods),
        "categorical_cramers_v": categorical_association_matrix(working_df, categorical_cols)
        if len(categorical_cols) >= 2
        else pd.DataFrame(),
        "categorical_numeric_eta": categorical_numeric_association_matrix(working_df, categorical_cols, numeric_cols)
        if categorical_cols and numeric_cols
        else pd.DataFrame(),
    }
    
    # Advanced nonlinear/partial correlations
    adv_corr = AdvancedCorrelation(working_df)
    result["advanced"] = adv_corr.get_report()

    high_corr_pairs: list[dict[str, Any]] = []
    primary_method = cfg.correlation_methods[0] if cfg.correlation_methods else "pearson"
    if primary_method in result["numeric"]:
        high_corr_pairs.extend(find_highly_correlated_pairs(result["numeric"][primary_method], cfg.high_correlation_threshold))
    if not result["categorical_cramers_v"].empty:
        high_corr_pairs.extend(find_highly_correlated_pairs(result["categorical_cramers_v"], cfg.high_correlation_threshold))

    result["high_correlation_pairs"] = high_corr_pairs

    if cfg.target_column and cfg.target_column in profiles:
        result["target_leakage"] = _target_leakage_candidates(working_df, profiles, cfg)
    else:
        result["target_leakage"] = []

    return result


def _target_leakage_candidates(df: pd.DataFrame, profiles: dict[str, ColumnProfile], cfg: EDAConfig) -> list[dict[str, Any]]:
    target = cfg.target_column
    target_profile = profiles[target]
    findings = []
    for col, profile in profiles.items():
        if col == target or cfg.is_ignored(col) or profile.is_constant:
            continue
        value: float | None = None
        try:
            if profile.is_numeric and target_profile.is_numeric:
                value = df[col].astype(float).corr(df[target].astype(float))
            elif (profile.is_categorical or profile.is_boolean) and (target_profile.is_categorical or target_profile.is_boolean):
                value = cramers_v(df[col], df[target])
            elif profile.is_numeric and (target_profile.is_categorical or target_profile.is_boolean):
                value = correlation_ratio(df[target], df[col])
            elif (profile.is_categorical or profile.is_boolean) and target_profile.is_numeric:
                value = correlation_ratio(df[col], df[target])
        except Exception:  # noqa: BLE001
            value = None
        if value is not None and abs(value) >= cfg.leakage_correlation_threshold:
            findings.append({"column": col, "value": float(value)})
    return sorted(findings, key=lambda d: -abs(d["value"]))
