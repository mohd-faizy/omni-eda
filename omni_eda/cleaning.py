"""Optional, explicit, auditable cleaning operations.

Nothing in this module mutates data unless the caller asks for it: every
function takes a DataFrame and returns a *new* DataFrame plus a log of what
it did, so cleaning is always inspectable and reversible (the original
object is never touched).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile


@dataclass
class CleaningReport:
    actions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0

    def log(self, message: str) -> None:
        self.actions.append(message)


def remove_duplicate_rows(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates()
    removed = before - len(out)
    if removed:
        report.log(f"Removed {removed} duplicate row(s).")
    return out


def remove_duplicate_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    from omni_eda.utils import find_duplicate_columns

    dup_cols = find_duplicate_columns(df)
    if dup_cols:
        report.log(f"Removed {len(dup_cols)} duplicate column(s): {dup_cols}")
        return df.drop(columns=dup_cols)
    return df


def drop_constant_columns(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: CleaningReport) -> pd.DataFrame:
    const_cols = [c for c, p in profiles.items() if p.is_constant and c in df.columns]
    if const_cols:
        report.log(f"Dropped {len(const_cols)} constant column(s): {const_cols}")
        return df.drop(columns=const_cols)
    return df


def fill_missing_values(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    report: CleaningReport,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    out = df.copy()
    for col, profile in profiles.items():
        if col not in out.columns or profile.n_missing == 0:
            continue
        if profile.is_numeric:
            fill_value = out[col].median() if numeric_strategy == "median" else out[col].mean()
            out[col] = out[col].fillna(fill_value)
            report.log(
                f"Filled {profile.n_missing} missing value(s) in numeric column '{col}' with {numeric_strategy} ({fill_value:.4g})."
            )
        elif profile.is_datetime:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            out[col] = out[col].ffill().bfill()
            report.log(f"Forward/backward filled missing datetime values in '{col}'.")
        elif profile.is_categorical or profile.is_boolean or profile.is_text:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            out[col] = out[col].fillna(fill_value)
            report.log(f"Filled {profile.n_missing} missing value(s) in '{col}' with mode ('{fill_value}').")
    return out


def convert_dtypes(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: CleaningReport) -> pd.DataFrame:
    out = df.copy()
    for col, profile in profiles.items():
        if col not in out.columns:
            continue
        try:
            if profile.is_datetime and not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
                report.log(f"Converted '{col}' to datetime.")
            elif profile.is_numeric and not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")
                report.log(f"Converted '{col}' to numeric.")
            elif profile.is_boolean and not pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
                report.log(f"Converted '{col}' to boolean.")
            elif profile.is_categorical:
                out[col] = out[col].astype("category")
        except Exception:  # noqa: BLE001 - best-effort conversion
            continue
    return out


def strip_whitespace_and_normalize(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: CleaningReport) -> pd.DataFrame:
    out = df.copy()
    text_like_cols = [c for c, p in profiles.items() if (p.is_text or p.is_categorical) and c in out.columns]
    for col in text_like_cols:
        if out[col].dtype == object:
            stripped = out[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            if not stripped.equals(out[col].astype(str)):
                out[col] = stripped
                report.log(f"Stripped whitespace / normalized spacing in '{col}'.")
    return out


def replace_infinities(df: pd.DataFrame, report: CleaningReport, replacement: float = np.nan) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    n_inf = 0
    for col in numeric_cols:
        mask = np.isinf(out[col])
        n_inf += int(mask.sum())
        if mask.any():
            out.loc[mask, col] = replacement
    if n_inf:
        report.log(f"Replaced {n_inf} infinite value(s) across {len(numeric_cols)} numeric column(s).")
    return out


def clip_outliers_iqr(df: pd.DataFrame, columns: Sequence[str], report: CleaningReport, multiplier: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_clipped = int(((out[col] < lower) | (out[col] > upper)).sum())
        if n_clipped:
            out[col] = out[col].clip(lower=lower, upper=upper)
            report.log(f"Clipped {n_clipped} outlier value(s) in '{col}' to [{lower:.4g}, {upper:.4g}].")
    return out


def encode_categoricals(
    df: pd.DataFrame, columns: Sequence[str], report: CleaningReport, max_cardinality: int = 20
) -> pd.DataFrame:
    """One-hot encode low-cardinality categoricals; label-encode the rest."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        nunique = out[col].nunique(dropna=True)
        if nunique <= max_cardinality:
            dummies = pd.get_dummies(out[col], prefix=col, dummy_na=False)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
            report.log(f"One-hot encoded '{col}' ({nunique} categories).")
        else:
            out[col] = out[col].astype("category").cat.codes
            report.log(f"Label-encoded high-cardinality column '{col}' ({nunique} categories).")
    return out


def scale_numeric_columns(
    df: pd.DataFrame, columns: Sequence[str], report: CleaningReport, method: str = "standard"
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col]
        if method == "standard":
            std = series.std(ddof=0)
            out[col] = (series - series.mean()) / std if std else 0.0
        elif method == "minmax":
            rng = series.max() - series.min()
            out[col] = (series - series.min()) / rng if rng else 0.0
        report.log(f"Scaled '{col}' using {method} scaling.")
    return out


def auto_clean(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
    steps: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run a configurable pipeline of the cleaning steps above.

    ``steps`` defaults to a conservative, non-destructive sequence:
    dedup rows/columns -> fix infinities -> convert dtypes -> strip whitespace.
    Steps like filling missing values, dropping constants, encoding and
    scaling are opt-in because they change the data more aggressively.
    """
    config or EDAConfig()
    steps = steps or ["dedup_rows", "dedup_columns", "infinities", "convert_dtypes", "strip_whitespace"]
    report = CleaningReport(rows_before=len(df), columns_before=df.shape[1])

    out = df.copy()
    step_fns = {
        "dedup_rows": lambda d: remove_duplicate_rows(d, report),
        "dedup_columns": lambda d: remove_duplicate_columns(d, report),
        "drop_constant": lambda d: drop_constant_columns(d, profiles, report),
        "fill_missing": lambda d: fill_missing_values(d, profiles, report),
        "convert_dtypes": lambda d: convert_dtypes(d, profiles, report),
        "strip_whitespace": lambda d: strip_whitespace_and_normalize(d, profiles, report),
        "infinities": lambda d: replace_infinities(d, report),
    }
    for step in steps:
        fn = step_fns.get(step)
        if fn:
            out = fn(out)

    report.rows_after = len(out)
    report.columns_after = out.shape[1]
    return out, report
