"""Descriptive statistics for every column type omni_eda recognizes.

v0.2 enhancements
-----------------
- Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
- Geometric / harmonic / trimmed means
- Gini coefficient, monotonicity score
- Simpson diversity, concentration ratio (categorical)
- Language diversity metrics (text)
- Frequency estimation, weekend analysis (datetime)
- Dataset-level summary statistics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.utils import safe_pct


def _entropy(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(scipy_stats.entropy(probs, base=2)) if len(probs) else 0.0


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    return float((2.0 * np.sum(cumulative) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])) if cumulative[-1] > 0 else 0.0


def _monotonicity_score(values: np.ndarray) -> float:
    """Fraction of consecutive pairs that are monotonically increasing or decreasing."""
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    n_pairs = len(diffs)
    if n_pairs == 0:
        return 0.0
    inc = np.sum(diffs > 0)
    dec = np.sum(diffs < 0)
    return float(max(inc, dec) / n_pairs)


def _normality_test(values: np.ndarray) -> dict[str, Any]:
    """Run Shapiro-Wilk (small n) and D'Agostino-Pearson (n >= 20) normality tests."""
    result: dict[str, Any] = {}
    n = len(values)
    if n < 8:
        return result
    try:
        if n <= 5000:
            stat, p = scipy_stats.shapiro(values[:5000])
            result["shapiro_wilk_stat"] = round(float(stat), 6)
            result["shapiro_wilk_p"] = round(float(p), 6)
            result["shapiro_wilk_normal"] = p > 0.05
    except Exception:
        pass
    if n >= 20:
        try:
            stat, p = scipy_stats.normaltest(values)
            result["dagostino_stat"] = round(float(stat), 6)
            result["dagostino_p"] = round(float(p), 6)
            result["dagostino_normal"] = p > 0.05
        except Exception:
            pass
    return result


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    non_null_raw = s.dropna()
    n = len(s)
    n_infinite = int(np.isinf(non_null_raw).sum())
    # Infinite values are reported but excluded from descriptive statistics so a
    # handful of Inf's don't turn every summary statistic into NaN/Inf.
    non_null = non_null_raw[~np.isinf(non_null_raw)]
    result: dict[str, Any] = {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null_raw.shape[0]),
        "missing_pct": safe_pct(n - non_null_raw.shape[0], n),
        "n_infinite": n_infinite,
    }
    if non_null.empty:
        return result

    mean = float(non_null.mean())
    std = float(non_null.std(ddof=1)) if len(non_null) > 1 else 0.0
    median = float(non_null.median())
    mad = float((non_null - median).abs().median())
    q1, q3 = non_null.quantile([0.25, 0.75])
    percentiles = non_null.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()

    # Advanced means
    values_arr = non_null.to_numpy()
    positive_vals = values_arr[values_arr > 0]
    geometric_mean = float(scipy_stats.gmean(positive_vals)) if len(positive_vals) > 0 else None
    harmonic_mean = float(scipy_stats.hmean(positive_vals)) if len(positive_vals) > 0 and np.all(positive_vals > 0) else None
    try:
        trimmed_mean = float(scipy_stats.trim_mean(values_arr, proportiontocut=0.05))
    except Exception:
        trimmed_mean = None
        
    skewness = float(non_null.skew()) if len(non_null) > 2 else None
    
    # Calculate optimal power transform lambda if skewed
    power_transform_lambda = None
    power_transform_method = None
    if skewness and abs(skewness) > 1.0 and len(values_arr) > 10:
        try:
            if float(non_null.min()) > 0:
                _, opt_lambda = scipy_stats.boxcox(values_arr)
                power_transform_lambda = float(opt_lambda)
                power_transform_method = "box-cox"
            else:
                _, opt_lambda = scipy_stats.yeojohnson(values_arr)
                power_transform_lambda = float(opt_lambda)
                power_transform_method = "yeo-johnson"
        except Exception:
            pass

    result.update(
        {
            "mean": mean,
            "median": median,
            "mode": float(non_null.mode().iloc[0]) if not non_null.mode().empty else None,
            "std": std,
            "variance": float(non_null.var(ddof=1)) if len(non_null) > 1 else 0.0,
            "mad": mad,
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "range": float(non_null.max() - non_null.min()),
            "iqr": float(q3 - q1),
            "q1": float(q1),
            "q3": float(q3),
            "percentiles": {f"p{int(k * 100)}": float(v) for k, v in percentiles.items()},
            "cv": float(std / mean) if mean else None,
            "skewness": skewness,
            "kurtosis": float(non_null.kurt()) if len(non_null) > 3 else None,
            "power_transform_lambda": power_transform_lambda,
            "power_transform_method": power_transform_method,
            "zeros_pct": safe_pct((non_null == 0).sum(), n),
            "negative_pct": safe_pct((non_null < 0).sum(), n),
            "positive_pct": safe_pct((non_null > 0).sum(), n),
            "n_unique": int(non_null.nunique()),
            "sum": float(non_null.sum()),
            # --- v0.2 enhancements ---
            "geometric_mean": geometric_mean,
            "harmonic_mean": harmonic_mean,
            "trimmed_mean_5pct": trimmed_mean,
            "gini_coefficient": _gini_coefficient(values_arr) if len(values_arr) > 1 else None,
            "monotonicity": _monotonicity_score(values_arr),
            "n_zeros": int((non_null == 0).sum()),
            "n_negative": int((non_null < 0).sum()),
            "n_positive": int((non_null > 0).sum()),
        }
    )

    # Normality tests
    normality = _normality_test(values_arr)
    if normality:
        result["normality"] = normality

    return result


def categorical_stats(series: pd.Series, rare_threshold: float = 0.01, top_n: int = 10) -> dict[str, Any]:
    non_null = series.dropna()
    n = len(series)
    counts = non_null.value_counts()
    freqs = (counts / counts.sum()) if counts.sum() else counts
    rare = freqs[freqs < rare_threshold]

    # Simpson diversity index: probability that two randomly picked items are different
    simpson_div = 0.0
    if counts.sum() > 1:
        probs = counts / counts.sum()
        simpson_div = float(1.0 - (probs ** 2).sum())

    # Concentration ratio (top 3)
    top3_pct = float(freqs.head(3).sum() * 100) if not freqs.empty else 0.0

    # Imbalance ratio (majority / minority)
    imbalance_ratio = None
    if len(counts) >= 2 and counts.iloc[-1] > 0:
        imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])

    return {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null.shape[0]),
        "missing_pct": safe_pct(n - non_null.shape[0], n),
        "n_unique": int(non_null.nunique()),
        "cardinality_ratio": safe_pct(non_null.nunique(), n) / 100.0,
        "top_values": {str(k): int(v) for k, v in counts.head(top_n).items()},
        "top_value": str(counts.index[0]) if not counts.empty else None,
        "top_value_pct": safe_pct(counts.iloc[0], counts.sum()) if not counts.empty else 0.0,
        "rare_categories": {str(k): float(v) for k, v in rare.items()},
        "n_rare_categories": int(len(rare)),
        "entropy": _entropy(counts),
        # --- v0.2 enhancements ---
        "simpson_diversity": simpson_div,
        "concentration_ratio_top3": top3_pct,
        "imbalance_ratio": imbalance_ratio,
    }


def text_stats(series: pd.Series, top_n: int = 15) -> dict[str, Any]:
    non_null = series.dropna().astype(str)
    n = len(series)
    if non_null.empty:
        return {"count": 0, "missing": n, "missing_pct": 100.0}

    lengths = non_null.str.len()
    word_counts = non_null.str.split().str.len()
    all_words = non_null.str.lower().str.findall(r"[a-zA-Z']+").explode()
    all_chars = non_null.str.cat().replace(" ", "")
    char_freq = pd.Series(list(all_chars)).value_counts().head(top_n)
    word_freq = all_words.value_counts().head(top_n) if not all_words.empty else pd.Series(dtype=int)

    # Language diversity: unique words / total words
    total_words = len(all_words.dropna())
    unique_words = all_words.nunique()
    lexical_diversity = unique_words / total_words if total_words > 0 else 0.0

    # Pattern detection
    has_digits_pct = safe_pct(non_null.str.contains(r"\d", regex=True).sum(), len(non_null))
    has_special_pct = safe_pct(non_null.str.contains(r"[^a-zA-Z0-9\s]", regex=True).sum(), len(non_null))
    has_uppercase_pct = safe_pct(non_null.str.contains(r"[A-Z]", regex=True).sum(), len(non_null))

    return {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null.shape[0]),
        "missing_pct": safe_pct(n - non_null.shape[0], n),
        "avg_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "median_length": float(lengths.median()),
        "avg_word_count": float(word_counts.mean()),
        "top_words": {str(k): int(v) for k, v in word_freq.items()},
        "top_chars": {str(k): int(v) for k, v in char_freq.items()},
        "n_empty_strings": int((non_null.str.strip() == "").sum()),
        "n_whitespace_only": int((non_null != non_null.str.strip()).sum()),
        # --- v0.2 enhancements ---
        "lexical_diversity": round(lexical_diversity, 4),
        "has_digits_pct": round(has_digits_pct, 2),
        "has_special_chars_pct": round(has_special_pct, 2),
        "has_uppercase_pct": round(has_uppercase_pct, 2),
        "total_unique_words": unique_words,
    }


def datetime_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_datetime(series, errors="coerce", format="mixed") if not pd.api.types.is_datetime64_any_dtype(series) else series
    non_null = s.dropna()
    n = len(s)
    if non_null.empty:
        return {"count": 0, "missing": n, "missing_pct": 100.0}

    now = pd.Timestamp.now()
    by_month = non_null.dt.month.value_counts().sort_index()
    by_dow = non_null.dt.dayofweek.value_counts().sort_index()

    # Frequency estimation
    if len(non_null) >= 2:
        sorted_dates = non_null.sort_values()
        diffs = sorted_dates.diff().dropna()
        median_gap_days = float(diffs.dt.total_seconds().median() / 86400) if not diffs.empty else None
        if median_gap_days is not None:
            if median_gap_days < 0.1:
                freq_estimate = "sub-daily"
            elif median_gap_days < 1.5:
                freq_estimate = "daily"
            elif median_gap_days < 8:
                freq_estimate = "weekly"
            elif median_gap_days < 35:
                freq_estimate = "monthly"
            elif median_gap_days < 100:
                freq_estimate = "quarterly"
            else:
                freq_estimate = "yearly_or_irregular"
        else:
            freq_estimate = "unknown"
            median_gap_days = None
    else:
        freq_estimate = "insufficient_data"
        median_gap_days = None

    # Weekend percentage
    weekend_count = int(non_null.dt.dayofweek.isin([5, 6]).sum())
    weekend_pct = safe_pct(weekend_count, len(non_null))

    return {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null.shape[0]),
        "missing_pct": safe_pct(n - non_null.shape[0], n),
        "min_date": non_null.min().isoformat(),
        "max_date": non_null.max().isoformat(),
        "range_days": int((non_null.max() - non_null.min()).days),
        "n_future_dates": int((non_null > now).sum()),
        "most_common_month": int(by_month.idxmax()) if not by_month.empty else None,
        "most_common_weekday": int(by_dow.idxmax()) if not by_dow.empty else None,
        "n_unique_dates": int(non_null.dt.date.nunique()),
        # --- v0.2 enhancements ---
        "median_date": non_null.median().isoformat() if len(non_null) > 0 else None,
        "frequency_estimate": freq_estimate,
        "median_gap_days": round(median_gap_days, 2) if median_gap_days is not None else None,
        "weekend_pct": round(weekend_pct, 2),
    }


def boolean_stats(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    n = len(series)
    counts = non_null.astype(str).str.lower().map({"1": True, "0": False, "true": True, "false": False}).fillna(non_null)
    value_counts = counts.value_counts()
    return {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null.shape[0]),
        "missing_pct": safe_pct(n - non_null.shape[0], n),
        "true_pct": safe_pct(value_counts.get(True, 0), non_null.shape[0]) if non_null.shape[0] else 0.0,
        "false_pct": safe_pct(value_counts.get(False, 0), non_null.shape[0]) if non_null.shape[0] else 0.0,
    }


def compute_column_stats(series: pd.Series, profile: ColumnProfile, config: EDAConfig | None = None) -> dict[str, Any]:
    """Dispatch to the right stats function based on the column's detected base type."""
    cfg = config or EDAConfig()
    if profile.is_boolean:
        return {"type": "boolean", **boolean_stats(series)}
    if profile.is_datetime:
        return {"type": "datetime", **datetime_stats(series)}
    if profile.is_numeric:
        return {"type": "numeric", **numeric_stats(series)}
    if profile.is_text:
        return {"type": "text", **text_stats(series)}
    return {"type": "categorical", **categorical_stats(series, rare_threshold=cfg.rare_category_threshold)}


def compute_all_statistics(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-column descriptive statistics for the whole DataFrame."""
    cfg = config or EDAConfig()
    results: dict[str, dict[str, Any]] = {}
    for col, profile in profiles.items():
        if cfg.is_ignored(col) or profile.is_constant and profile.n_unique == 0:
            continue
        try:
            results[col] = compute_column_stats(df[col], profile, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad column must not sink the whole run
            results[col] = {"type": "error", "error": str(exc)}
    return results


def compute_dataset_summary(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate dataset-level summary statistics (v0.2)."""
    n_rows, n_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())

    # Column type breakdown
    type_counts: dict[str, int] = {}
    for profile in profiles.values():
        type_counts[profile.base_type] = type_counts.get(profile.base_type, 0) + 1

    # Overall missing
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = safe_pct(total_missing, total_cells)

    # Duplicates
    n_dup_rows = int(df.duplicated().sum())

    # Constant columns
    n_constant = sum(1 for p in profiles.values() if p.is_constant)

    # Highly skewed columns
    n_skewed = 0
    for col, stats in statistics.items():
        skew = stats.get("skewness")
        if skew is not None and abs(skew) >= 1.0:
            n_skewed += 1

    # Average completeness per column
    completeness_scores = []
    for profile in profiles.values():
        completeness_scores.append(100.0 - profile.missing_pct)
    avg_completeness = float(np.mean(completeness_scores)) if completeness_scores else 100.0

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "memory_bytes": memory_bytes,
        "type_counts": type_counts,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "overall_missing_pct": round(overall_missing_pct, 2),
        "n_duplicate_rows": n_dup_rows,
        "duplicate_rows_pct": round(safe_pct(n_dup_rows, n_rows), 2),
        "n_constant_columns": n_constant,
        "n_skewed_columns": n_skewed,
        "avg_completeness": round(avg_completeness, 2),
    }


def classify_distribution(stats: dict[str, Any]) -> str:
    """Classify the shape of a numeric distribution."""
    skewness = stats.get("skewness")
    kurtosis = stats.get("kurtosis")
    n_unique = stats.get("n_unique", 0)

    if n_unique is not None and n_unique <= 2:
        return "binary"
    if skewness is None:
        return "unknown"

    shape = "symmetric"
    if abs(skewness) > 2.0:
        shape = "highly skewed right" if skewness > 0 else "highly skewed left"
    elif abs(skewness) > 1.0:
        shape = "skewed right" if skewness > 0 else "skewed left"
    elif abs(skewness) > 0.5:
        shape = "moderately skewed right" if skewness > 0 else "moderately skewed left"

    if kurtosis is not None:
        if kurtosis > 3:
            shape += " (heavy-tailed)"
        elif kurtosis < -1:
            shape += " (light-tailed)"

    # Check normality from tests
    normality = stats.get("normality", {})
    if normality.get("shapiro_wilk_normal") or normality.get("dagostino_normal"):
        if abs(skewness) < 0.5:
            shape = "approximately normal"

    return shape


def build_numeric_summary_table(
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a summary table for all numeric columns, suitable for rendering in the report."""
    rows = []
    for col, profile in profiles.items():
        if not profile.is_numeric or profile.is_constant:
            continue
        stats = statistics.get(col, {})
        rows.append({
            "column": col,
            "count": stats.get("count", 0),
            "missing": stats.get("missing", 0),
            "missing_pct": round(stats.get("missing_pct", 0.0), 1),
            "mean": _safe_round(stats.get("mean")),
            "std": _safe_round(stats.get("std")),
            "min": _safe_round(stats.get("min")),
            "q1": _safe_round(stats.get("q1")),
            "median": _safe_round(stats.get("median")),
            "q3": _safe_round(stats.get("q3")),
            "max": _safe_round(stats.get("max")),
            "skewness": _safe_round(stats.get("skewness")),
            "kurtosis": _safe_round(stats.get("kurtosis")),
            "n_unique": stats.get("n_unique", 0),
            "n_zeros": stats.get("n_zeros", 0),
            "distribution": classify_distribution(stats),
        })
    return rows


def build_categorical_summary_table(
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a summary table for all categorical/boolean columns."""
    rows = []
    for col, profile in profiles.items():
        if not (profile.is_categorical or profile.is_boolean) or profile.is_constant:
            continue
        stats = statistics.get(col, {})
        rows.append({
            "column": col,
            "count": stats.get("count", 0),
            "missing": stats.get("missing", 0),
            "missing_pct": round(stats.get("missing_pct", 0.0), 1),
            "n_unique": stats.get("n_unique", profile.n_unique),
            "top_value": stats.get("top_value", ""),
            "top_value_pct": round(stats.get("top_value_pct", 0.0), 1),
            "entropy": _safe_round(stats.get("entropy")),
            "cardinality_ratio": _safe_round(stats.get("cardinality_ratio")),
            "n_rare_categories": stats.get("n_rare_categories", 0),
            "simpson_diversity": _safe_round(stats.get("simpson_diversity")),
            "imbalance_ratio": _safe_round(stats.get("imbalance_ratio")),
        })
    return rows


def build_quality_scorecard(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a per-column data quality scorecard."""
    n_rows = len(df)
    rows = []
    for col, profile in profiles.items():
        stats = statistics.get(col, {})
        completeness = round(100.0 - profile.missing_pct, 1)

        # Uniqueness score
        uniqueness = round(profile.unique_ratio * 100, 1)

        # Validity: no infinities, no mixed types
        issues = []
        if profile.is_mixed_type:
            issues.append("mixed_types")
        if stats.get("n_infinite", 0) > 0:
            issues.append("infinite_values")
        if profile.is_constant:
            issues.append("constant")
        if profile.is_zero_variance:
            issues.append("zero_variance")
        validity = 100.0 if not issues else max(0.0, 100.0 - len(issues) * 25)

        # Overall quality score for this column
        quality = round((completeness * 0.4 + min(uniqueness, 100) * 0.2 + validity * 0.4), 1)

        rows.append({
            "column": col,
            "type": profile.base_type,
            "completeness": completeness,
            "uniqueness": min(round(uniqueness, 1), 100.0),
            "validity": round(validity, 1),
            "quality_score": quality,
            "issues": issues,
        })
    return sorted(rows, key=lambda r: r["quality_score"])


def _safe_round(value: Any, digits: int = 4) -> Any:
    """Round a value safely, returning None for non-numeric inputs."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value
