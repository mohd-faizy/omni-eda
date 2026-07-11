"""Builds the final human-readable report from everything the pipeline computed."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from omni_eda.config import EDAConfig
from omni_eda.themes import get_theme
from omni_eda.utils import human_bytes

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportBuilder:
    """Consumes the raw results dict produced by :class:`~omni_eda.analyzer.OmniEDA` and renders reports."""

    def __init__(self, config: EDAConfig | None = None, version: str = "0.1.0"):
        self.config = config or EDAConfig()
        self.version = version
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "j2"]),
        )

    # ------------------------------------------------------------------ #
    # Context assembly
    # ------------------------------------------------------------------ #
    def _build_context(self, results: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        df_shape = results["shape"]
        profiles = results["profiles"]
        quality = results["quality"]
        stats = results["statistics"]
        univariate_plots = results.get("univariate_plots", {})

        type_counts: dict[str, int] = {}
        for profile in profiles.values():
            type_counts[profile.base_type] = type_counts.get(profile.base_type, 0) + 1

        columns = []
        for name, profile in profiles.items():
            col_stats = dict(stats.get(name, {}))
            col_stats.pop("type", None)
            col_stats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in col_stats.items()}
            columns.append(
                {
                    "name": name,
                    "base_type": profile.base_type,
                    "semantic_type": profile.semantic_type,
                    "n_unique": profile.n_unique,
                    "missing_pct": profile.missing_pct,
                    "stats": col_stats,
                    "plots": univariate_plots.get(name, {}),
                }
            )

        correlation = results.get("correlation", {})
        target_analysis = results.get("target_analysis") or {}
        if target_analysis.get("feature_importance") is not None:
            target_analysis = dict(target_analysis)
            target_analysis["feature_importance_table"] = target_analysis["feature_importance"].to_dict(orient="records")

        outliers_df = results.get("outliers_summary")
        outliers = (
            outliers_df.to_dict(orient="records") if isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty else []
        )

        overview = {
            "n_rows": df_shape[0],
            "n_columns": df_shape[1],
            "memory_human": human_bytes(quality.summary.get("memory_usage_bytes", 0)),
            "n_duplicate_rows": next(
                (i.detail.get("n_duplicates", 0) for i in quality.issues if i.category == "duplicate_rows"), 0
            ),
            "missing_pct": results.get("missing", {}).get("overall_missing_pct", 0.0),
            "type_counts": type_counts,
        }

        return {
            "title": cfg.title,
            "version": self.version,
            "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "theme": get_theme(cfg.theme),
            "overview": overview,
            "quality": {
                "n_critical": quality.summary.get("n_critical", 0),
                "n_warning": quality.summary.get("n_warning", 0),
                "n_info": quality.summary.get("n_info", 0),
                "issues": [i.to_dict() for i in quality.issues],
            },
            "missing": results.get("missing", {}),
            "missing_plots": results.get("missing_plots", {}),
            "columns": columns,
            "correlation": {"high_pairs": correlation.get("high_correlation_pairs", [])},
            "outliers": outliers,
            "bivariate_plots": results.get("bivariate_plots", {}),
            "multivariate_plots": results.get("multivariate_plots", {}),
            "timeseries_plots": results.get("timeseries_plots", {}),
            "additional_plots": results.get("additional_plots", {}),
            "target_analysis": target_analysis,
            "health_score": results.get("health_score"),
            "memory_analysis": results.get("memory_analysis"),
            "insights": [i.to_dict() for i in results.get("insights", [])] if results.get("insights") else [],
            "suggestions": [s.to_dict() for s in results.get("suggestions", [])],
            "ab_testing": results.get("ab_testing"),
            "drift": results.get("drift"),
            "hopkins_statistic": results.get("hopkins_statistic"),
            "timeseries_changepoints": results.get("timeseries_changepoints"),
            "outlier_explanations": results.get("outlier_explanations", {}),
            "final_summary": build_final_summary(results),
            # --- v0.3 additions ---
            "dataset_summary": results.get("dataset_summary", {}),
            "numeric_summary_table": results.get("numeric_summary_table", []),
            "categorical_summary_table": results.get("categorical_summary_table", []),
            "quality_scorecard": results.get("quality_scorecard", []),
            "statistical_tests": results.get("statistical_tests"),
            "data_sample_head": _df_to_records_safe(results.get("df_head")),
            "data_sample_tail": _df_to_records_safe(results.get("df_tail")),
            "data_sample_columns": results.get("df_columns", []),
            
            # --- Phase 1 & 2 additions ---
            "fingerprint": results.get("fingerprint"),
            "complexity": results.get("complexity"),
            "information_theory": results.get("information_theory"),
            "distribution_diagnostics": results.get("distribution_diagnostics"),
            "power_analysis": results.get("power_analysis"),
            "synthetic_diagnostics": results.get("synthetic_diagnostics"),
            "advanced": results.get("correlation", {}).get("advanced"),
            "dependencies": results.get("dependencies"),
            "interactions": results.get("interactions"),
            "feature_clustering": results.get("feature_clustering"),
            "sample_clustering": results.get("sample_clustering"),
            "advanced_anomalies": results.get("advanced_anomalies"),
            "causal_discovery": results.get("causal_discovery"),
        }

    # ------------------------------------------------------------------ #
    # Renderers
    # ------------------------------------------------------------------ #
    def render_html(self, results: dict[str, Any]) -> str:
        template = self._env.get_template("report.html.j2")
        return template.render(**self._build_context(results))

    def render_markdown(self, results: dict[str, Any]) -> str:
        return build_markdown_report(results, self.config, self.version)

    def render_console_summary(self, results: dict[str, Any]) -> str:
        return build_console_summary(results)


def build_final_summary(results: dict[str, Any]) -> str:
    quality = results["quality"]
    shape = results["shape"]
    n_crit, n_warn = quality.summary.get("n_critical", 0), quality.summary.get("n_warning", 0)
    parts = [
        f"Analyzed {shape[0]:,} rows and {shape[1]} columns.",
        f"Found {n_crit} critical issue(s) and {n_warn} warning(s).",
    ]
    n_suggestions = len(results.get("suggestions", []))
    if n_suggestions:
        parts.append(f"{n_suggestions} feature-engineering suggestion(s) were generated.")
    target_analysis = results.get("target_analysis")
    if target_analysis and target_analysis.get("curves", {}).get("roc_auc"):
        parts.append(
            f"A baseline classifier on '{target_analysis['target']}' reached ROC AUC = {target_analysis['curves']['roc_auc']:.3f}."
        )
    if n_crit == 0 and n_warn == 0:
        parts.append("Overall, the dataset looks clean and ready for modeling.")
    elif n_crit > 0:
        parts.append("Address the critical issues above before modeling on this data.")
    return " ".join(parts)


def build_console_summary(results: dict[str, Any]) -> str:
    shape = results["shape"]
    quality = results["quality"]
    lines = [
        "=" * 60,
        "OMNI-EDA SUMMARY",
        "=" * 60,
        f"Rows: {shape[0]:,}   Columns: {shape[1]}",
        f"Critical issues: {quality.summary.get('n_critical', 0)}   Warnings: {quality.summary.get('n_warning', 0)}   Info: {quality.summary.get('n_info', 0)}",
        "-" * 60,
    ]
    for issue in quality.issues[:25]:
        lines.append(f"[{issue.severity.upper():8s}] {issue.category:22s} {issue.message}")
    if len(quality.issues) > 25:
        lines.append(f"... and {len(quality.issues) - 25} more (see full report).")
    lines.append("-" * 60)
    lines.append(build_final_summary(results))
    return "\n".join(lines)


def build_markdown_report(results: dict[str, Any], config: EDAConfig | None = None, version: str = "0.1.0") -> str:
    """Comprehensive markdown report matching the HTML report structure."""
    cfg = config or EDAConfig()
    shape = results["shape"]
    quality = results["quality"]
    profiles = results["profiles"]
    stats = results["statistics"]

    lines: list[str] = [
        f"# {cfg.title}",
        "",
        f"_Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by omni_eda v{version}_",
        "",
        "Built by [GitHub](https://github.com/mohd-faizy) | [Mohd Faizy](https://mohdfaizy.vercel.app/)",
        "",
    ]

    # --- 1. Executive Summary ---
    lines += ["## 1. Executive Summary", ""]
    health = results.get("health_score")
    if health:
        lines.append(f"**Dataset Health Score:** {health['score']}/100 (Grade: {health['grade']} — {health['label']})")
        lines.append("")
    lines.append(build_final_summary(results))
    lines.append("")

    # Top insights
    all_insights = results.get("insights", [])
    top_insights = [i for i in all_insights if i.severity in ("highlight", "warning")][:5] if all_insights else []
    if top_insights:
        lines.append("**Key Findings:**")
        for ins in top_insights:
            lines.append(f"- **{ins.title}**: {ins.description}")
        lines.append("")

    # --- 2. Dataset Overview ---
    lines += ["## 2. Dataset Overview", ""]
    lines += [
        f"- Rows: **{shape[0]:,}**",
        f"- Columns: **{shape[1]}**",
        f"- Memory usage: **{human_bytes(quality.summary.get('memory_usage_bytes', 0))}**",
        "",
    ]

    # Type breakdown
    type_counts: dict[str, int] = {}
    for profile in profiles.values():
        type_counts[profile.base_type] = type_counts.get(profile.base_type, 0) + 1
    if type_counts:
        lines.append("**Column Types:**")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {c}")
        lines.append("")

    # --- 3. Data Quality Assessment ---
    lines += ["## 3. Data Quality Assessment", ""]
    lines.append(f"- Critical issues: **{quality.summary.get('n_critical', 0)}**")
    lines.append(f"- Warnings: **{quality.summary.get('n_warning', 0)}**")
    lines.append(f"- Info: **{quality.summary.get('n_info', 0)}**")
    lines.append("")

    if quality.issues:
        lines.append("| Severity | Category | Column | Message |")
        lines.append("|---|---|---|---|")
        for issue in quality.issues:
            lines.append(f"| {issue.severity} | {issue.category} | {issue.column or '-'} | {issue.message} |")
    else:
        lines.append("_No issues detected._")
    lines.append("")

    # --- 4. Descriptive Statistics ---
    lines += ["## 4. Descriptive Statistics", ""]

    # Numeric summary table
    numeric_table = results.get("numeric_summary_table", [])
    if numeric_table:
        lines.append("### Numeric Columns")
        lines.append("")
        lines.append("| Column | Count | Missing% | Mean | Std | Min | Median | Max | Skewness | Distribution |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in numeric_table:
            lines.append(
                f"| {r['column']} | {r['count']} | {r['missing_pct']}% "
                f"| {_fmt(r['mean'])} | {_fmt(r['std'])} | {_fmt(r['min'])} "
                f"| {_fmt(r['median'])} | {_fmt(r['max'])} | {_fmt(r['skewness'])} | {r['distribution']} |"
            )
        lines.append("")

    # Categorical summary table
    cat_table = results.get("categorical_summary_table", [])
    if cat_table:
        lines.append("### Categorical Columns")
        lines.append("")
        lines.append("| Column | Count | Missing% | Unique | Top Value | Top% | Entropy |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in cat_table:
            lines.append(
                f"| {r['column']} | {r['count']} | {r['missing_pct']}% "
                f"| {r['n_unique']} | {r.get('top_value', '-')} | {r.get('top_value_pct', '-')}% | {_fmt(r.get('entropy'))} |"
            )
        lines.append("")

    # --- 5-7. Column Summary ---
    lines += ["## 5. Column Summary", ""]
    lines.append("| Column | Type | Semantic | Unique | Missing % |")
    lines.append("|---|---|---|---|---|")
    for name, profile in profiles.items():
        lines.append(
            f"| {name} | {profile.base_type} | {profile.semantic_type or '-'} | {profile.n_unique} | {profile.missing_pct:.1f}% |"
        )
    lines.append("")

    # --- 8. Correlation Analysis ---
    correlation = results.get("correlation", {})
    high_pairs = correlation.get("high_correlation_pairs", [])
    lines += ["## 8. Correlation Analysis", ""]
    if high_pairs:
        lines.append("### Highly Correlated Pairs")
        lines.append("| Column A | Column B | Value |")
        lines.append("|---|---|---|")
        for p in high_pairs:
            lines.append(f"| {p['col_a']} | {p['col_b']} | {p['value']:.3f} |")
    else:
        lines.append("_None above the configured threshold._")
    lines.append("")

    # --- 9. Statistical Testing ---
    stat_tests = results.get("statistical_tests")
    if stat_tests:
        lines += ["## 9. Statistical Testing", ""]
        summary = stat_tests.get("summary", {})
        lines.append(f"**Total tests run:** {summary.get('total_tests', 0)}")
        lines.append(f"**Significant results:** {summary.get('n_significant', 0)}")
        lines.append("")

        all_results_list = stat_tests.get("results", [])
        sig_results = [r for r in all_results_list if r.get("significant")]
        if sig_results:
            lines.append("### Significant Findings")
            lines.append("")
            lines.append("| Test | Column(s) | Statistic | p-value | Effect Size | Interpretation |")
            lines.append("|---|---|---|---|---|---|")
            for r in sig_results[:20]:
                cols = r['column_a']
                if r.get('column_b'):
                    cols += f" × {r['column_b']}"
                es = f"{r['effect_size']:.3f} ({r['effect_size_label']})" if r.get('effect_size') else "-"
                pv = f"{r['p_value']:.4g}" if r.get('p_value', -1) >= 0 else "-"
                lines.append(f"| {r['test_name']} | {cols} | {r['statistic']:.4f} | {pv} | {es} | {_trunc(r.get('interpretation',''), 80)} |")
            lines.append("")

    # --- 10-11. Outliers & Missing ---
    outliers_df = results.get("outliers_summary")
    if isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty:
        lines += ["## 11. Outlier Analysis", ""]
        lines.append(f"Detected outliers across {len(outliers_df)} column-method combinations.")
        lines.append("")

    missing_data = results.get("missing", {})
    if missing_data.get("total_missing_cells", 0) > 0:
        lines += ["## 12. Missing Data Analysis", ""]
        lines.append(f"- Total missing cells: **{missing_data['total_missing_cells']:,}** ({missing_data.get('overall_missing_pct', 0):.1f}%)")
        lines.append(f"- Rows with any missing: **{missing_data.get('rows_with_any_missing', 0):,}**")
        lines.append(f"- Columns with missing: **{missing_data.get('columns_with_missing', 0)}**")
        lines.append("")

    # --- 13. Key Insights ---
    if all_insights:
        lines += ["## 13. Key Insights", ""]
        for ins in all_insights:
            icon = {"highlight": "\u2b50", "warning": "\u26a0\ufe0f", "observation": "\U0001f4ca"}.get(ins.severity, "\u2022")
            lines.append(f"{icon} **{ins.title}**")
            lines.append(f"   {ins.description}")
            lines.append("")

    # --- 14. Recommendations ---
    suggestions = results.get("suggestions", [])
    lines += ["## 14. Feature Engineering Suggestions", ""]
    if suggestions:
        for s in suggestions:
            col_part = f" (`{s.column}`)" if s.column else ""
            lines.append(f"- **{s.action}**{col_part}: {s.rationale}")
    else:
        lines.append("_None._")
    lines.append("")

    # --- 15. Target Analysis ---
    target_analysis = results.get("target_analysis")
    if target_analysis:
        lines += [f"## 15. Target Analysis: `{target_analysis['target']}`", ""]
        if target_analysis.get("class_imbalance"):
            ci = target_analysis["class_imbalance"]
            lines.append(f"Majority class `{ci['majority_class']}` = {ci['majority_pct']:.1f}% of rows.")
            lines.append("")
        if target_analysis.get("curves", {}).get("roc_auc"):
            lines.append(f"Baseline classifier ROC AUC: **{target_analysis['curves']['roc_auc']:.3f}**")
            lines.append("")

    lines += [
        "---",
        "",
        "> Visualizations are available in the HTML version of this report.",
    ]

    return "\n".join(lines)


def _df_to_records_safe(df: Any) -> list[dict]:
    """Convert a DataFrame to list of dicts, safely handling None."""
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        return df.head(5).fillna("").astype(str).to_dict(orient="records")
    return []


def _fmt(value: Any) -> str:
    """Format a numeric value for markdown tables."""
    if value is None:
        return "-"
    try:
        f = float(value)
        if abs(f) >= 1000:
            return f"{f:,.1f}"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(value)


def _trunc(text: str, max_len: int = 80) -> str:
    """Truncate text for table cells."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "\u2026"
