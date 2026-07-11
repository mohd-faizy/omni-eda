"""Aggregate every data-quality issue omni_eda can detect into one report.

v0.2 enhancements
-----------------
- Dataset Health Score (0-100)
- Memory optimization suggestions
- Data type recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile, detect_hidden_characters
from omni_eda.utils import find_duplicate_columns, human_bytes


@dataclass
class Issue:
    severity: str  # "info" | "warning" | "critical"
    category: str
    column: str | None
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "column": self.column,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class QualityReport:
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, category: str, message: str, column: str | None = None, **detail: Any) -> None:
        self.issues.append(Issue(severity=severity, category=category, column=column, message=message, detail=detail))

    def by_severity(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "issues": [i.to_dict() for i in self.issues]}


_IMPOSSIBLE_NEGATIVE_HINTS = (
    "age",
    "price",
    "cost",
    "count",
    "quantity",
    "salary",
    "revenue",
    "amount",
    "weight",
    "height",
    "duration",
    "population",
)


def _check_impossible_negatives(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: QualityReport) -> None:
    for col, profile in profiles.items():
        if not profile.is_numeric or col not in df.columns:
            continue
        lname = col.lower()
        if any(h in lname for h in _IMPOSSIBLE_NEGATIVE_HINTS):
            series = pd.to_numeric(df[col], errors="coerce")
            n_neg = int((series < 0).sum())
            if n_neg:
                report.add(
                    "warning",
                    "impossible_values",
                    f"Column '{col}' looks like it should be non-negative but has {n_neg} negative value(s).",
                    column=col,
                    n_negative=n_neg,
                )


def _check_future_dates(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: QualityReport) -> None:
    now = pd.Timestamp.now()
    for col, profile in profiles.items():
        if not profile.is_datetime or col not in df.columns:
            continue
        s = pd.to_datetime(df[col], errors="coerce", format="mixed")
        n_future = int((s > now).sum())
        n_invalid = int(df[col].notna().sum() - s.notna().sum())
        if n_future:
            report.add(
                "info", "future_dates", f"Column '{col}' has {n_future} date(s) in the future.", column=col, n_future=n_future
            )
        if n_invalid:
            report.add(
                "warning",
                "invalid_dates",
                f"Column '{col}' has {n_invalid} value(s) that failed to parse as dates.",
                column=col,
                n_invalid=n_invalid,
            )


def _check_empty_and_whitespace_strings(df: pd.DataFrame, profiles: dict[str, ColumnProfile], report: QualityReport) -> None:
    for col, _profile in profiles.items():
        if col not in df.columns or df[col].dtype != object:
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        n_empty = int((s == "").sum())
        n_ws_only = int((s.str.strip() == "").sum()) - n_empty
        if n_empty:
            report.add(
                "warning", "empty_strings", f"Column '{col}' has {n_empty} empty string value(s).", column=col, n_empty=n_empty
            )
        if n_ws_only > 0:
            report.add(
                "warning",
                "whitespace_only",
                f"Column '{col}' has {n_ws_only} whitespace-only value(s).",
                column=col,
                n_whitespace_only=n_ws_only,
            )

        hidden_ratio = detect_hidden_characters(s)
        if hidden_ratio > 0.01:
            report.add(
                "warning",
                "hidden_characters",
                f"Column '{col}' has hidden/non-printable characters in ~{hidden_ratio * 100:.1f}% of sampled values.",
                column=col,
                ratio=hidden_ratio,
            )

        try:
            s.str.encode("utf-8")
        except UnicodeEncodeError:
            report.add("warning", "encoding_issue", f"Column '{col}' contains values that fail UTF-8 encoding.", column=col)


def _check_infinities(df: pd.DataFrame, report: QualityReport) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        n_inf = int(np.isinf(numeric_df[col]).sum())
        if n_inf:
            report.add("critical", "infinite_values", f"Column '{col}' has {n_inf} infinite value(s).", column=col, n_inf=n_inf)


def _check_skew_and_imbalance(
    df: pd.DataFrame, profiles: dict[str, ColumnProfile], config: EDAConfig, report: QualityReport
) -> None:
    for col, profile in profiles.items():
        if col not in df.columns:
            continue
        if profile.is_numeric:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 2:
                skew = float(s.skew())
                if abs(skew) >= config.skew_threshold:
                    report.add(
                        "info",
                        "skewed_distribution",
                        f"Column '{col}' is skewed (skewness={skew:.2f}).",
                        column=col,
                        skewness=skew,
                    )
        elif profile.is_categorical or profile.is_boolean:
            counts = df[col].value_counts(normalize=True)
            if not counts.empty and counts.iloc[0] >= config.class_imbalance_threshold:
                report.add(
                    "warning",
                    "class_imbalance",
                    f"Column '{col}' is highly imbalanced: '{counts.index[0]}' makes up {counts.iloc[0] * 100:.1f}% of values.",
                    column=col,
                    top_value=str(counts.index[0]),
                    top_pct=float(counts.iloc[0] * 100),
                )


def _check_structural_issues(
    df: pd.DataFrame, profiles: dict[str, ColumnProfile], config: EDAConfig, report: QualityReport
) -> None:
    n_rows, n_cols = df.shape

    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows:
        report.add(
            "warning",
            "duplicate_rows",
            f"Found {n_dup_rows} duplicate row(s) ({n_dup_rows / n_rows * 100:.2f}%).",
            n_duplicates=n_dup_rows,
        )

    if n_cols > 1:
        try:
            dup_cols = find_duplicate_columns(df)
            if dup_cols:
                report.add(
                    "warning",
                    "duplicate_columns",
                    f"Found {len(dup_cols)} duplicate column(s): {dup_cols}.",
                    n_duplicates=len(dup_cols),
                )
        except Exception:  # noqa: BLE001
            pass

    for col, profile in profiles.items():
        if profile.is_constant:
            report.add("info", "constant_column", f"Column '{col}' has a single unique value (constant).", column=col)
        if profile.is_high_cardinality:
            report.add(
                "info",
                "high_cardinality",
                f"Column '{col}' has high cardinality ({profile.n_unique} unique values).",
                column=col,
                n_unique=profile.n_unique,
            )
        if profile.is_low_variance and not profile.is_zero_variance:
            report.add("info", "low_variance", f"Column '{col}' has very low variance.", column=col)
        if profile.is_zero_variance:
            report.add("warning", "zero_variance", f"Column '{col}' has zero variance.", column=col)
        if profile.is_mixed_type:
            report.add(
                "warning", "mixed_dtype", f"Column '{col}' contains mixed Python types (e.g. numbers and strings).", column=col
            )
        if profile.missing_pct >= config.missing_drop_threshold * 100:
            report.add(
                "critical",
                "excessive_missing",
                f"Column '{col}' is {profile.missing_pct:.1f}% missing.",
                column=col,
                missing_pct=profile.missing_pct,
            )
        elif profile.missing_pct >= config.missing_warn_threshold * 100:
            report.add(
                "warning",
                "missing_values",
                f"Column '{col}' is {profile.missing_pct:.1f}% missing.",
                column=col,
                missing_pct=profile.missing_pct,
            )

    all_null_cols = [c for c, p in profiles.items() if p.n_missing == p.n and p.n > 0]
    if all_null_cols:
        report.add(
            "critical",
            "all_null_columns",
            f"{len(all_null_cols)} column(s) are entirely missing: {all_null_cols}.",
            n_columns=len(all_null_cols),
        )


# ---------------------------------------------------------------------------
# v0.2: Memory optimization analysis
# ---------------------------------------------------------------------------
def compute_memory_analysis(df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, Any]:
    """Analyze memory usage and suggest optimizations per column."""
    current_mem = int(df.memory_usage(deep=True).sum())
    column_analysis: list[dict[str, Any]] = []
    total_savings = 0

    for col in df.columns:
        current_bytes = int(df[col].memory_usage(deep=True))
        current_dtype = str(df[col].dtype)
        recommended_dtype = current_dtype
        savings = 0

        profile = profiles.get(col)
        if profile is None:
            continue

        # Integer downcast
        if pd.api.types.is_integer_dtype(df[col]):
            s = df[col].dropna()
            if not s.empty:
                mn, mx = s.min(), s.max()
                if mn >= 0 and mx <= 255:
                    recommended_dtype = "uint8"
                elif mn >= -128 and mx <= 127:
                    recommended_dtype = "int8"
                elif mn >= 0 and mx <= 65535:
                    recommended_dtype = "uint16"
                elif mn >= -32768 and mx <= 32767:
                    recommended_dtype = "int16"
                elif mn >= 0 and mx <= 4294967295:
                    recommended_dtype = "uint32"
                elif mn >= -2147483648 and mx <= 2147483647:
                    recommended_dtype = "int32"

        # Float downcast
        elif pd.api.types.is_float_dtype(df[col]):
            if current_dtype == "float64":
                s = df[col].dropna()
                if not s.empty:
                    # Check if all values fit in float32 without significant loss
                    as_f32 = s.astype(np.float32)
                    if np.allclose(s.values, as_f32.values, rtol=1e-5, equal_nan=True):
                        recommended_dtype = "float32"

        # Object -> category for low-cardinality
        elif df[col].dtype == object:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > 0 and n_unique / len(df) < 0.5:
                recommended_dtype = "category"

        # Estimate savings
        if recommended_dtype != current_dtype:
            try:
                new_bytes = int(df[col].astype(recommended_dtype).memory_usage(deep=True))
                savings = current_bytes - new_bytes
                if savings < 0:
                    savings = 0
                    recommended_dtype = current_dtype  # revert if no actual savings
            except Exception:
                recommended_dtype = current_dtype
                savings = 0

        total_savings += savings
        column_analysis.append({
            "column": col,
            "current_dtype": current_dtype,
            "recommended_dtype": recommended_dtype,
            "current_bytes": current_bytes,
            "savings_bytes": savings,
            "can_optimize": recommended_dtype != current_dtype,
        })

    optimized_mem = current_mem - total_savings

    return {
        "current_memory_bytes": current_mem,
        "current_memory_human": human_bytes(current_mem),
        "optimized_memory_bytes": optimized_mem,
        "optimized_memory_human": human_bytes(optimized_mem),
        "total_savings_bytes": total_savings,
        "total_savings_human": human_bytes(total_savings),
        "savings_pct": round(total_savings / current_mem * 100, 1) if current_mem > 0 else 0.0,
        "columns": column_analysis,
        "n_optimizable": sum(1 for c in column_analysis if c["can_optimize"]),
    }


# ---------------------------------------------------------------------------
# v0.2: Dataset Health Score
# ---------------------------------------------------------------------------
def compute_health_score(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    quality_report: QualityReport,
) -> dict[str, Any]:
    """Compute a 0-100 health score for the dataset.

    The score is a weighted combination of several quality dimensions:
    - Completeness (30%): Inverse of overall missing percentage
    - Consistency (20%): Penalty for mixed types, duplicates, encoding issues
    - Validity (20%): Penalty for critical quality issues
    - Uniqueness (15%): Penalty for excessive duplicate rows
    - Accuracy (15%): Penalty for suspicious patterns (impossible values, outliers)
    """
    n_rows, n_cols = df.shape

    # --- Completeness (30%) ---
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    # 0% missing → 100, 50% missing → 50, 100% missing → 0
    completeness = max(0.0, 100.0 - missing_pct)

    # --- Consistency (20%) ---
    n_mixed = sum(1 for p in profiles.values() if p.is_mixed_type)
    n_encoding = len([i for i in quality_report.issues if i.category in ("encoding_issue", "hidden_characters")])
    consistency_penalties = min(100.0, (n_mixed * 15) + (n_encoding * 10))
    consistency = max(0.0, 100.0 - consistency_penalties)

    # --- Validity (20%) ---
    n_critical = len(quality_report.by_severity("critical"))
    n_warning = len(quality_report.by_severity("warning"))
    validity_penalties = min(100.0, (n_critical * 20) + (n_warning * 5))
    validity = max(0.0, 100.0 - validity_penalties)

    # --- Uniqueness (15%) ---
    n_dup_rows = int(df.duplicated().sum())
    dup_pct = (n_dup_rows / n_rows * 100) if n_rows > 0 else 0.0
    uniqueness = max(0.0, 100.0 - dup_pct * 2)  # 50% duplicates → 0

    # --- Accuracy (15%) ---
    n_impossible = len([i for i in quality_report.issues if i.category in ("impossible_values", "infinite_values")])
    n_constant = sum(1 for p in profiles.values() if p.is_constant)
    accuracy_penalties = min(100.0, (n_impossible * 15) + (n_constant * 5))
    accuracy = max(0.0, 100.0 - accuracy_penalties)

    # Weighted total
    score = (
        completeness * 0.30
        + consistency * 0.20
        + validity * 0.20
        + uniqueness * 0.15
        + accuracy * 0.15
    )
    score = round(min(100.0, max(0.0, score)), 1)

    # Grade
    if score >= 90:
        grade, label = "A", "Excellent"
    elif score >= 75:
        grade, label = "B", "Good"
    elif score >= 60:
        grade, label = "C", "Fair"
    elif score >= 40:
        grade, label = "D", "Poor"
    else:
        grade, label = "F", "Critical"

    return {
        "score": score,
        "grade": grade,
        "label": label,
        "dimensions": {
            "completeness": round(completeness, 1),
            "consistency": round(consistency, 1),
            "validity": round(validity, 1),
            "uniqueness": round(uniqueness, 1),
            "accuracy": round(accuracy, 1),
        },
        "weights": {
            "completeness": 0.30,
            "consistency": 0.20,
            "validity": 0.20,
            "uniqueness": 0.15,
            "accuracy": 0.15,
        },
    }


def build_quality_report(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
    correlation_findings: list[dict[str, Any]] | None = None,
    target_leakage_findings: list[dict[str, Any]] | None = None,
) -> QualityReport:
    """Run every quality check and return a consolidated :class:`QualityReport`."""
    cfg = config or EDAConfig()
    report = QualityReport()

    _check_structural_issues(df, profiles, cfg, report)
    _check_impossible_negatives(df, profiles, report)
    _check_future_dates(df, profiles, report)
    _check_empty_and_whitespace_strings(df, profiles, report)
    _check_infinities(df, report)
    _check_skew_and_imbalance(df, profiles, cfg, report)

    if correlation_findings:
        for pair in correlation_findings:
            report.add(
                "warning",
                "high_correlation",
                f"Columns '{pair['col_a']}' and '{pair['col_b']}' are highly correlated ({pair['value']:.2f}).",
                col_a=pair["col_a"],
                col_b=pair["col_b"],
                value=pair["value"],
            )

    if target_leakage_findings:
        for finding in target_leakage_findings:
            report.add(
                "critical",
                "target_leakage",
                f"Column '{finding['column']}' is suspiciously correlated with the target ({finding['value']:.2f}); possible leakage.",
                column=finding["column"],
                value=finding["value"],
            )

    report.summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_issues": len(report.issues),
        "n_critical": len(report.by_severity("critical")),
        "n_warning": len(report.by_severity("warning")),
        "n_info": len(report.by_severity("info")),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }
    return report
