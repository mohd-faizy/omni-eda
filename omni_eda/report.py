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
            "target_analysis": target_analysis,
            "suggestions": [s.to_dict() for s in results.get("suggestions", [])],
            "final_summary": build_final_summary(results),
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
    cfg = config or EDAConfig()
    shape = results["shape"]
    quality = results["quality"]
    profiles = results["profiles"]
    results["statistics"]

    lines: list[str] = [
        f"# {cfg.title}",
        "",
        f"_Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by omni_eda v{version}_",
        "",
    ]

    lines += [
        "## Overview",
        "",
        f"- Rows: **{shape[0]:,}**",
        f"- Columns: **{shape[1]}**",
        f"- Memory usage: **{human_bytes(quality.summary.get('memory_usage_bytes', 0))}**",
        f"- Critical issues: **{quality.summary.get('n_critical', 0)}**, Warnings: **{quality.summary.get('n_warning', 0)}**",
        "",
    ]

    lines += ["## Data Quality Issues", ""]
    if quality.issues:
        lines.append("| Severity | Category | Column | Message |")
        lines.append("|---|---|---|---|")
        for issue in quality.issues:
            lines.append(f"| {issue.severity} | {issue.category} | {issue.column or '-'} | {issue.message} |")
    else:
        lines.append("_No issues detected._")
    lines.append("")

    lines += ["## Column Summary", ""]
    lines.append("| Column | Type | Semantic | Unique | Missing % |")
    lines.append("|---|---|---|---|---|")
    for name, profile in profiles.items():
        lines.append(
            f"| {name} | {profile.base_type} | {profile.semantic_type or '-'} | {profile.n_unique} | {profile.missing_pct:.1f}% |"
        )
    lines.append("")

    correlation = results.get("correlation", {})
    high_pairs = correlation.get("high_correlation_pairs", [])
    lines += ["## Highly Correlated Pairs", ""]
    if high_pairs:
        lines.append("| Column A | Column B | Value |")
        lines.append("|---|---|---|")
        for p in high_pairs:
            lines.append(f"| {p['col_a']} | {p['col_b']} | {p['value']:.3f} |")
    else:
        lines.append("_None above the configured threshold._")
    lines.append("")

    suggestions = results.get("suggestions", [])
    lines += ["## Feature Engineering Suggestions", ""]
    if suggestions:
        for s in suggestions:
            col_part = f" (`{s.column}`)" if s.column else ""
            lines.append(f"- **{s.action}**{col_part}: {s.rationale}")
    else:
        lines.append("_None._")
    lines.append("")

    target_analysis = results.get("target_analysis")
    if target_analysis:
        lines += [f"## Target Analysis: `{target_analysis['target']}`", ""]
        if target_analysis.get("class_imbalance"):
            ci = target_analysis["class_imbalance"]
            lines.append(f"Majority class `{ci['majority_class']}` = {ci['majority_pct']:.1f}% of rows.")
            lines.append("")
        if target_analysis.get("curves", {}).get("roc_auc"):
            lines.append(f"Baseline classifier ROC AUC: **{target_analysis['curves']['roc_auc']:.3f}**")
            lines.append("")

    lines += [
        "## Summary",
        "",
        build_final_summary(results),
        "",
        "> Visualizations are available in the HTML version of this report.",
    ]

    return "\n".join(lines)
