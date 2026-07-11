"""Automated insight engine for omni_eda (v0.2).

Generates human-readable observations and key findings from analysis results.
Each insight carries a severity level (highlight / observation / warning) and
a category so the report template can group and style them appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile


@dataclass
class Insight:
    """A single automated observation about the dataset."""

    severity: str  # "highlight" | "observation" | "warning"
    category: str
    title: str
    description: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "detail": self.detail,
        }


def generate_insights(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    quality_report: Any,
    health_score: dict[str, Any] | None = None,
    missing_analysis: dict[str, Any] | None = None,
    correlation: dict[str, Any] | None = None,
    config: EDAConfig | None = None,
    **kwargs: Any,
) -> list[Insight]:
    """Generate automated insights from analysis results."""
    cfg = config or EDAConfig()
    insights: list[Insight] = []

    n_rows, n_cols = df.shape

    # --- Dataset overview insights ---
    _dataset_overview_insights(df, profiles, statistics, health_score, insights)

    # --- Missing value insights ---
    _missing_insights(df, profiles, missing_analysis, insights)

    # --- Distribution insights ---
    _distribution_insights(profiles, statistics, insights)

    # --- Correlation insights ---
    _correlation_insights(correlation, insights)

    # --- Quality insights ---
    _quality_insights(quality_report, insights)

    # --- Column-level insights ---
    _column_insights(df, profiles, statistics, cfg, insights)

    # --- Statistical test insights ---
    stat_tests = kwargs.get("statistical_tests")
    if stat_tests:
        _statistical_test_insights(stat_tests, insights)

    # --- Outlier insights ---
    outliers_summary = kwargs.get("outliers_summary")
    if outliers_summary is not None:
        _outlier_insights(outliers_summary, insights)

    # --- Recommendation insights ---
    _recommendation_insights(df, profiles, statistics, quality_report, correlation, insights)

    return insights



def _dataset_overview_insights(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    health_score: dict[str, Any] | None,
    insights: list[Insight],
) -> None:
    n_rows, n_cols = df.shape

    # Dataset size insight
    if n_rows > 100_000:
        insights.append(Insight(
            "observation", "overview",
            "Large Dataset",
            f"This dataset has {n_rows:,} rows — consider sampling for exploratory work or using Dask/Spark for production pipelines.",
        ))
    elif n_rows < 100:
        insights.append(Insight(
            "warning", "overview",
            "Small Dataset",
            f"Only {n_rows:,} rows — statistical analysis and ML models may not be reliable with so few observations.",
        ))

    # Health score insight
    if health_score:
        score = health_score["score"]
        grade = health_score["grade"]
        label = health_score["label"]
        if score >= 90:
            insights.append(Insight(
                "highlight", "health",
                f"Excellent Data Quality (Score: {score}/100)",
                f"This dataset scored {grade} ({label}). It has minimal quality issues and is well-suited for analysis.",
            ))
        elif score >= 75:
            insights.append(Insight(
                "observation", "health",
                f"Good Data Quality (Score: {score}/100)",
                f"This dataset scored {grade} ({label}). A few improvements could boost reliability.",
            ))
        elif score < 60:
            insights.append(Insight(
                "warning", "health",
                f"Data Quality Needs Attention (Score: {score}/100)",
                f"This dataset scored {grade} ({label}). Address the critical issues before using this data for modeling.",
            ))

    # Column type balance
    type_counts: dict[str, int] = {}
    for p in profiles.values():
        type_counts[p.base_type] = type_counts.get(p.base_type, 0) + 1
    numeric_cols = type_counts.get("numeric", 0)
    cat_cols = type_counts.get("categorical", 0) + type_counts.get("boolean", 0)
    if numeric_cols > 0 and cat_cols > 0:
        insights.append(Insight(
            "observation", "overview",
            "Mixed Feature Types",
            f"Dataset has {numeric_cols} numeric and {cat_cols} categorical column(s) — suitable for diverse ML approaches.",
        ))

    # Wide dataset
    if n_cols > 50:
        insights.append(Insight(
            "warning", "overview",
            "Wide Dataset",
            f"With {n_cols} columns, consider dimensionality reduction (PCA, feature selection) before modeling.",
        ))


def _missing_insights(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    missing_analysis: dict[str, Any] | None,
    insights: list[Insight],
) -> None:
    cols_with_missing = [p for p in profiles.values() if p.missing_pct > 0]

    if not cols_with_missing:
        insights.append(Insight(
            "highlight", "missing",
            "No Missing Values",
            "This dataset is complete — no missing values detected in any column.",
        ))
        return

    # Severe missing
    heavily_missing = [p for p in cols_with_missing if p.missing_pct > 50]
    if heavily_missing:
        names = ", ".join(f"'{p.name}'" for p in heavily_missing[:5])
        insights.append(Insight(
            "warning", "missing",
            f"{len(heavily_missing)} Column(s) Are Mostly Empty",
            f"Columns {names} are over 50% missing. Consider dropping them unless they carry critical signal.",
            detail={"columns": [p.name for p in heavily_missing]},
        ))

    # Overall missing
    overall_pct = missing_analysis.get("overall_missing_pct", 0.0) if missing_analysis else 0.0
    if overall_pct > 10:
        insights.append(Insight(
            "warning", "missing",
            f"Significant Missing Data ({overall_pct:.1f}%)",
            "Over 10% of all cells are missing. Imputation strategy is critical for reliable analysis.",
        ))
    elif overall_pct > 0:
        insights.append(Insight(
            "observation", "missing",
            f"Moderate Missing Data ({overall_pct:.1f}%)",
            f"{len(cols_with_missing)} column(s) contain missing values affecting {overall_pct:.1f}% of total cells.",
        ))


def _distribution_insights(
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    insights: list[Insight],
) -> None:
    # Highly skewed columns
    skewed = []
    for col, stats in statistics.items():
        skew = stats.get("skewness")
        if skew is not None and abs(skew) > 2.0:
            skewed.append((col, skew))

    if skewed:
        names = ", ".join(f"'{c}' (skew={s:.1f})" for c, s in skewed[:5])
        insights.append(Insight(
            "observation", "distribution",
            f"{len(skewed)} Highly Skewed Column(s)",
            f"Consider log/power transforms for: {names}. Skewed features can bias linear models.",
            detail={"columns": [c for c, _ in skewed]},
        ))

    # Near-normal columns
    normal_count = 0
    for col, stats in statistics.items():
        normality = stats.get("normality", {})
        if normality.get("shapiro_wilk_normal") or normality.get("dagostino_normal"):
            normal_count += 1
    if normal_count > 0:
        insights.append(Insight(
            "observation", "distribution",
            f"{normal_count} Column(s) Pass Normality Tests",
            "These columns follow approximately normal distributions — parametric methods are applicable.",
        ))


def _correlation_insights(
    correlation: dict[str, Any] | None,
    insights: list[Insight],
) -> None:
    if not correlation:
        return

    high_pairs = correlation.get("high_correlation_pairs", [])
    if high_pairs:
        top = high_pairs[0]
        insights.append(Insight(
            "warning", "correlation",
            f"{len(high_pairs)} Highly Correlated Feature Pair(s)",
            f"Strongest: '{top['col_a']}' ↔ '{top['col_b']}' (r={top['value']:.3f}). "
            "Consider dropping redundant features to reduce multicollinearity.",
            detail={"pairs": high_pairs[:5]},
        ))

    leakage = correlation.get("target_leakage", [])
    if leakage:
        cols = ", ".join(f"'{f['column']}'" for f in leakage)
        insights.append(Insight(
            "warning", "correlation",
            "Potential Target Leakage Detected",
            f"Column(s) {cols} are suspiciously correlated with the target. "
            "Investigate whether these contain future information.",
            detail={"columns": [f["column"] for f in leakage]},
        ))


def _quality_insights(
    quality_report: Any,
    insights: list[Insight],
) -> None:
    n_critical = quality_report.summary.get("n_critical", 0)
    n_warning = quality_report.summary.get("n_warning", 0)

    if n_critical == 0 and n_warning == 0:
        insights.append(Insight(
            "highlight", "quality",
            "Clean Dataset",
            "No critical quality issues or warnings detected. The data is in good shape for analysis.",
        ))
    elif n_critical > 0:
        insights.append(Insight(
            "warning", "quality",
            f"{n_critical} Critical Quality Issue(s)",
            "Address these issues before using the data for modeling — they could invalidate results.",
        ))

    # Duplicate rows
    for issue in quality_report.issues:
        if issue.category == "duplicate_rows":
            n_dup = issue.detail.get("n_duplicates", 0)
            if n_dup > 0:
                pct = n_dup / quality_report.summary.get("n_rows", 1) * 100
                insights.append(Insight(
                    "warning" if pct > 5 else "observation", "quality",
                    f"{n_dup:,} Duplicate Rows ({pct:.1f}%)",
                    "Duplicate rows can bias statistical analysis and model training. Consider deduplication.",
                ))
            break


def _column_insights(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    config: EDAConfig,
    insights: list[Insight],
) -> None:
    # ID-like columns
    id_cols = [p.name for p in profiles.values() if p.is_id_like]
    if id_cols:
        insights.append(Insight(
            "observation", "columns",
            f"{len(id_cols)} ID-Like Column(s) Detected",
            f"Columns {', '.join(repr(c) for c in id_cols[:5])} appear to be identifiers. "
            "They carry no predictive signal and should be excluded from modeling.",
        ))

    # Constant columns
    constant_cols = [p.name for p in profiles.values() if p.is_constant]
    if constant_cols:
        insights.append(Insight(
            "observation", "columns",
            f"{len(constant_cols)} Constant Column(s)",
            f"Columns {', '.join(repr(c) for c in constant_cols[:5])} have only one unique value. "
            "They can be safely dropped as they provide no information.",
        ))

    # High cardinality categorical
    high_card = [p for p in profiles.values() if p.is_high_cardinality and p.is_categorical]
    if high_card:
        names = ", ".join(f"'{p.name}' ({p.n_unique} unique)" for p in high_card[:5])
        insights.append(Insight(
            "observation", "columns",
            f"{len(high_card)} High-Cardinality Categorical Column(s)",
            f"Columns {names} have many unique values. Consider frequency encoding, target encoding, "
            "or grouping rare categories.",
        ))

    # Binary features
    binary_cols = [p.name for p in profiles.values() if p.is_binary_encoded and p.is_numeric]
    if binary_cols:
        insights.append(Insight(
            "observation", "columns",
            f"{len(binary_cols)} Binary Feature(s)",
            f"Columns {', '.join(repr(c) for c in binary_cols[:5])} contain only 0/1 values — "
            "these may already be one-hot encoded.",
        ))

    # Boolean columns (with class imbalance info)
    for col, stats in statistics.items():
        profile = profiles.get(col)
        if profile and profile.is_boolean:
            true_pct = stats.get("true_pct", 50.0)
            if true_pct > 95 or true_pct < 5:
                insights.append(Insight(
                    "warning", "column",
                    f"Extreme Imbalance in '{col}'",
                    f"True={true_pct:.1f}% — this boolean column is extremely one-sided.",
                ))


def _statistical_test_insights(
    stat_tests: dict[str, Any],
    insights: list[Insight],
) -> None:
    """Generate insights from automated statistical tests."""
    summary = stat_tests.get("summary", {})
    total = summary.get("total_tests", 0)
    n_sig = summary.get("n_significant", 0)

    if total == 0:
        return

    insights.append(Insight(
        "observation", "statistical_test",
        f"{total} Statistical Tests Executed",
        f"{n_sig} test(s) yielded statistically significant results (α=0.05). "
        f"Review the Statistical Testing section for detailed findings.",
        detail={"total": total, "significant": n_sig},
    ))

    # Highlight the most impactful significant findings
    by_category = stat_tests.get("by_category", {})

    # Significant comparisons
    comparisons = by_category.get("comparison", [])
    sig_comparisons = [r for r in comparisons if r.get("significant")]
    if sig_comparisons:
        # Find the one with largest effect size
        best = max(sig_comparisons, key=lambda r: abs(r.get("effect_size") or 0), default=None)
        if best and best.get("effect_size"):
            label = best.get("effect_size_label", "")
            insights.append(Insight(
                "highlight", "statistical_test",
                f"Strong Group Difference: {best['column_a']} by {best['column_b']}",
                best.get("interpretation", ""),
                detail={"test": best["test_name"], "effect_size": best["effect_size"]},
            ))

    # Significant associations
    associations = by_category.get("association", [])
    sig_assoc = [r for r in associations if r.get("significant")]
    if sig_assoc:
        best = max(sig_assoc, key=lambda r: abs(r.get("effect_size") or 0), default=None)
        if best:
            insights.append(Insight(
                "highlight", "statistical_test",
                f"Significant Association: {best['column_a']} × {best['column_b']}",
                best.get("interpretation", ""),
            ))

    # Normality summary
    normality = by_category.get("normality", [])
    normal_cols = set()
    non_normal_cols = set()
    for r in normality:
        if r.get("test_name") == "Shapiro-Wilk":
            if r.get("significant"):
                non_normal_cols.add(r["column_a"])
            else:
                normal_cols.add(r["column_a"])
    if non_normal_cols:
        names = ", ".join(f"'{c}'" for c in list(non_normal_cols)[:5])
        insights.append(Insight(
            "observation", "statistical_test",
            f"{len(non_normal_cols)} Column(s) Are Non-Normal",
            f"Columns {names} fail normality tests — use non-parametric methods for these features.",
        ))


def _outlier_insights(
    outliers_summary: Any,
    insights: list[Insight],
) -> None:
    """Generate insights from outlier detection results."""
    if not isinstance(outliers_summary, pd.DataFrame) or outliers_summary.empty:
        return

    # Columns with most outliers
    if "column" in outliers_summary.columns and "pct_outliers" in outliers_summary.columns:
        high_outlier = outliers_summary[outliers_summary["pct_outliers"] > 5]
        if not high_outlier.empty:
            # Get unique columns
            cols = high_outlier["column"].unique()[:5]
            names = ", ".join(f"'{c}'" for c in cols)
            insights.append(Insight(
                "warning", "outlier",
                f"{len(cols)} Column(s) Have >5% Outliers",
                f"Columns {names} have a substantial fraction of outlier values. "
                "Investigate whether these are data errors or genuine extreme values.",
                detail={"columns": list(cols)},
            ))

        # Overall
        total_outliers = int(outliers_summary["n_outliers"].sum()) if "n_outliers" in outliers_summary.columns else 0
        if total_outliers > 0:
            insights.append(Insight(
                "observation", "outlier",
                f"{total_outliers:,} Total Outlier Detections",
                "Multiple outlier detection methods were applied. Cross-reference methods to identify "
                "the most robust outliers (those flagged by multiple approaches).",
            ))


def _recommendation_insights(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    quality_report: Any,
    correlation: dict[str, Any] | None,
    insights: list[Insight],
) -> None:
    """Generate actionable recommendation insights."""
    n_rows, n_cols = df.shape
    recommendations = []

    # Data cleaning recommendations
    n_critical = quality_report.summary.get("n_critical", 0) if quality_report else 0
    if n_critical > 0:
        recommendations.append(
            f"Address {n_critical} critical data quality issue(s) before any modeling work."
        )

    # Missing data strategy
    cols_high_missing = [p for p in profiles.values() if p.missing_pct > 50]
    cols_moderate_missing = [p for p in profiles.values() if 5 < p.missing_pct <= 50]
    if cols_high_missing:
        recommendations.append(
            f"Consider dropping {len(cols_high_missing)} column(s) with >50% missing data, "
            "or use advanced imputation (e.g., MICE, KNN imputer) if they contain critical signal."
        )
    if cols_moderate_missing:
        recommendations.append(
            f"Apply imputation to {len(cols_moderate_missing)} column(s) with moderate missing data (5-50%). "
            "Consider multiple imputation to preserve uncertainty."
        )

    # Feature engineering recommendations
    skewed_cols = [col for col, stats in statistics.items()
                   if stats.get("skewness") is not None and abs(stats["skewness"]) > 2.0]
    if skewed_cols:
        recommendations.append(
            f"Apply log or power transforms to {len(skewed_cols)} highly skewed column(s) "
            "to improve linearity assumptions for models like linear regression."
        )

    # Multicollinearity
    if correlation:
        high_pairs = correlation.get("high_correlation_pairs", [])
        if len(high_pairs) > 3:
            recommendations.append(
                f"Consider VIF analysis or PCA to handle {len(high_pairs)} highly correlated feature pairs. "
                "Multicollinearity can inflate coefficient variance in linear models."
            )

    # Scale recommendations
    if n_rows > 100_000:
        recommendations.append(
            "For large-scale modeling, use gradient-boosted trees (XGBoost, LightGBM) which handle "
            "mixed types and missing values natively."
        )
    elif n_rows < 500:
        recommendations.append(
            "With limited data, prefer simpler models (logistic regression, small decision trees) "
            "and use cross-validation to get reliable performance estimates."
        )

    if recommendations:
        for i, rec in enumerate(recommendations[:6]):
            insights.append(Insight(
                "observation", "recommendation",
                f"Recommendation #{i+1}",
                rec,
            ))
