"""Small, dependency-light helpers shared across the package."""

from __future__ import annotations

import functools
import hashlib
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def human_bytes(n: float) -> str:
    """Format a byte count as a human readable string (e.g. ``1.2 GB``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def safe_pct(numerator: float, denominator: float) -> float:
    """Percentage helper that never raises on a zero denominator."""
    if not denominator:
        return 0.0
    return 100.0 * numerator / denominator


def truncate(text: str, max_len: int = 80) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# --------------------------------------------------------------------------- #
# Sampling / memory
# --------------------------------------------------------------------------- #
def sample_df(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Return ``df`` unchanged if small enough, otherwise a reproducible random sample."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype to save memory.

    Returns a new DataFrame; the original is left untouched.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["integer"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def optimize_dtypes(df: pd.DataFrame, category_ratio: float = 0.5) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality object columns to ``category``."""
    out = downcast_numeric(df)
    n = len(out)
    if n == 0:
        return out
    for col in out.select_dtypes(include=["object"]).columns:
        nunique = out[col].nunique(dropna=True)
        if nunique and nunique / n < category_ratio:
            out[col] = out[col].astype("category")
    return out


# --------------------------------------------------------------------------- #
# Caching / timing
# --------------------------------------------------------------------------- #
def hash_dataframe(df: pd.DataFrame, sample_rows: int = 1000) -> str:
    """Cheap, order-sensitive fingerprint of a DataFrame for cache keys.

    Uses a bounded sample so hashing stays fast even for huge frames.
    """
    sample = df.head(sample_rows)
    try:
        values = pd.util.hash_pandas_object(sample, index=True).values
        digest = hashlib.md5(values.tobytes())
    except TypeError:
        digest = hashlib.md5(str(sample.shape).encode() + str(list(df.columns)).encode())
    digest.update(str(df.shape).encode())
    return digest.hexdigest()


def timed(func: F) -> F:
    """Decorator that logs execution time via the package logger at DEBUG level."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from omni_eda.logger import get_logger

        logger = get_logger()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug("%s took %.3fs", func.__name__, time.perf_counter() - start)
        return result

    return wrapper  # type: ignore[return-value]


def memoize(func: F) -> F:
    """Lightweight memoization for pure functions keyed on args' repr.

    Not intended for huge argument objects; used for small, repeated
    lookups (e.g. regex compilation, palette generation).
    """
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper  # type: ignore[return-value]


def find_duplicate_columns(df: pd.DataFrame) -> list:
    """Return names of columns that are exact duplicates of an earlier column.

    Hashes each column independently instead of transposing the whole
    DataFrame, which is dramatically faster (and far less memory-hungry) on
    wide-ish, long DataFrames than the naive ``df.T.duplicated()`` approach.
    """
    seen: dict = {}
    duplicates: list = []
    for col in df.columns:
        series = df[col]
        try:
            key = hashlib.md5(pd.util.hash_pandas_object(series, index=False).values.tobytes()).hexdigest()
        except TypeError:
            key = hashlib.md5(series.astype(str).values.tobytes()).hexdigest()
        if key in seen:
            duplicates.append(col)
        else:
            seen[key] = col
    return duplicates


# --------------------------------------------------------------------------- #
# Misc numeric helpers
# --------------------------------------------------------------------------- #
def is_effectively_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    """True if >= threshold fraction of non-null values can be parsed as numbers."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    return coerced.notna().mean() >= threshold


def safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    try:
        value = a.corr(b, method=method)
        return None if pd.isna(value) else float(value)
    except Exception:
        return None


def clip_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lower, upper = series.quantile([lower_q, upper_q])
    return series.clip(lower=lower, upper=upper)


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into plain Python for JSON export."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return to_serializable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_serializable(obj.to_dict(orient="records"))
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
