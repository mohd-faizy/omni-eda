"""Descriptive statistics for every column type omni_eda recognizes."""

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
            "skewness": float(non_null.skew()) if len(non_null) > 2 else None,
            "kurtosis": float(non_null.kurt()) if len(non_null) > 3 else None,
            "zeros_pct": safe_pct((non_null == 0).sum(), n),
            "negative_pct": safe_pct((non_null < 0).sum(), n),
            "positive_pct": safe_pct((non_null > 0).sum(), n),
            "n_unique": int(non_null.nunique()),
            "sum": float(non_null.sum()),
        }
    )
    return result


def categorical_stats(series: pd.Series, rare_threshold: float = 0.01, top_n: int = 10) -> dict[str, Any]:
    non_null = series.dropna()
    n = len(series)
    counts = non_null.value_counts()
    freqs = (counts / counts.sum()) if counts.sum() else counts
    rare = freqs[freqs < rare_threshold]

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

    return {
        "count": int(non_null.shape[0]),
        "missing": int(n - non_null.shape[0]),
        "missing_pct": safe_pct(n - non_null.shape[0], n),
        "avg_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "avg_word_count": float(word_counts.mean()),
        "top_words": {str(k): int(v) for k, v in word_freq.items()},
        "top_chars": {str(k): int(v) for k, v in char_freq.items()},
        "n_empty_strings": int((non_null.str.strip() == "").sum()),
        "n_whitespace_only": int((non_null != non_null.str.strip()).sum()),
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
