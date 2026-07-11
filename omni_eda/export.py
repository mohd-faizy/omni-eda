"""Export the analysis results (and/or the report) to every requested format."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.logger import get_logger
from omni_eda.report import ReportBuilder
from omni_eda.utils import to_serializable


def export_html(results: dict[str, Any], path: Path, config: EDAConfig | None = None, version: str = "0.1.0") -> Path:
    builder = ReportBuilder(config, version=version)
    html = builder.render_html(results)
    path.write_text(html, encoding="utf-8")
    return path


def export_markdown(results: dict[str, Any], path: Path, config: EDAConfig | None = None, version: str = "0.1.0") -> Path:
    builder = ReportBuilder(config, version=version)
    md = builder.render_markdown(results)
    path.write_text(md, encoding="utf-8")
    return path


def export_json(results: dict[str, Any], path: Path) -> Path:
    """Export a machine-readable summary (stats, quality issues, correlations, suggestions).

    Large embedded images are intentionally excluded to keep the JSON small
    and genuinely useful as a data interchange format.
    """
    payload = {
        "shape": results.get("shape"),
        "columns": {
            name: {
                "base_type": profile.base_type,
                "semantic_type": profile.semantic_type,
                "n_unique": profile.n_unique,
                "missing_pct": profile.missing_pct,
                "flags": profile.flags,
            }
            for name, profile in results.get("profiles", {}).items()
        },
        "statistics": results.get("statistics", {}),
        "quality": results.get("quality").to_dict() if results.get("quality") else {},
        "correlation_high_pairs": results.get("correlation", {}).get("high_correlation_pairs", []),
        "outliers_summary": results.get("outliers_summary").to_dict(orient="records")
        if isinstance(results.get("outliers_summary"), pd.DataFrame)
        else [],
        "suggestions": [s.to_dict() for s in results.get("suggestions", [])],
        "target_analysis": {
            k: v for k, v in (results.get("target_analysis") or {}).items() if k not in ("feature_importance", "curves")
        },
    }
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    return path


def export_csv_bundle(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Export the main tabular results (stats, quality issues, correlations) as separate CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    stats_rows = []
    for col, col_stats in results.get("statistics", {}).items():
        row = {"column": col}
        row.update({k: v for k, v in col_stats.items() if not isinstance(v, dict)})
        stats_rows.append(row)
    if stats_rows:
        p = output_dir / "statistics.csv"
        pd.DataFrame(stats_rows).to_csv(p, index=False)
        written.append(p)

    quality = results.get("quality")
    if quality and quality.issues:
        p = output_dir / "quality_issues.csv"
        pd.DataFrame([i.to_dict() for i in quality.issues]).to_csv(p, index=False)
        written.append(p)

    high_pairs = results.get("correlation", {}).get("high_correlation_pairs", [])
    if high_pairs:
        p = output_dir / "high_correlation_pairs.csv"
        pd.DataFrame(high_pairs).to_csv(p, index=False)
        written.append(p)

    outliers_df = results.get("outliers_summary")
    if isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty:
        p = output_dir / "outliers_summary.csv"
        outliers_df.to_csv(p, index=False)
        written.append(p)

    suggestions = results.get("suggestions", [])
    if suggestions:
        p = output_dir / "feature_suggestions.csv"
        pd.DataFrame([s.to_dict() for s in suggestions]).to_csv(p, index=False)
        written.append(p)

    return written


def export_excel(results: dict[str, Any], path: Path) -> Path:
    """Export the key result tables as sheets of a single Excel workbook."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1})
        
        def format_sheet(df, sheet_name):
            if df.empty: return
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                col_len = max([len(str(v)) for v in df[df.columns[col_num]]] + [len(str(value))]) + 2
                worksheet.set_column(col_num, col_num, min(col_len, 60))

        q_sum = results.get("quality").summary if results.get("quality") else {}
        overview = pd.DataFrame([{
            "rows": results["shape"][0],
            "columns": results["shape"][1],
            "critical_issues": q_sum.get("n_critical", 0),
            "warnings": q_sum.get("n_warning", 0),
        }])
        format_sheet(overview, "Overview")

        stats_rows = []
        for col, col_stats in results.get("statistics", {}).items():
            row = {"column": col}
            row.update({k: v for k, v in col_stats.items() if not isinstance(v, dict)})
            stats_rows.append(row)
        if stats_rows:
            format_sheet(pd.DataFrame(stats_rows), "Statistics")

        quality = results.get("quality")
        if quality and quality.issues:
            format_sheet(pd.DataFrame([i.to_dict() for i in quality.issues]), "Quality Issues")

        high_pairs = results.get("correlation", {}).get("high_correlation_pairs", [])
        if high_pairs:
            format_sheet(pd.DataFrame(high_pairs), "Correlations")

        outliers_df = results.get("outliers_summary")
        if isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty:
            format_sheet(outliers_df, "Outliers")

        suggestions = results.get("suggestions", [])
        if suggestions:
            format_sheet(pd.DataFrame([s.to_dict() for s in suggestions]), "Suggestions")

        missing_table = results.get("missing", {}).get("summary_table")
        if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
            format_sheet(missing_table, "Missing Values")

    return path


def _iter_base64_images(results: dict[str, Any]):
    for name, img in results.get("missing_plots", {}).items():
        yield f"missing__{name}", img
    for col, plots in results.get("univariate_plots", {}).items():
        for name, img in plots.items():
            yield f"univariate__{col}__{name}", img
    for name, img in results.get("bivariate_plots", {}).items():
        yield f"bivariate__{name}", img
    for name, img in results.get("multivariate_plots", {}).items():
        yield f"multivariate__{name}", img
    for col, plots in results.get("timeseries_plots", {}).items():
        for name, img in plots.items():
            yield f"timeseries__{col}__{name}", img


def export_figures(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Dump every generated figure as an individual PNG/SVG file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data_uri in _iter_base64_images(results):
        try:
            header, encoded = data_uri.split(",", 1)
            ext = "svg" if "svg" in header else "png"
            raw = base64.b64decode(encoded)
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            p = output_dir / f"{safe_name}.{ext}"
            p.write_bytes(raw)
            written.append(p)
        except Exception:  # noqa: BLE001
            continue
    return written


def export_pdf(results: dict[str, Any], path: Path) -> Path | None:
    """Bundle every PNG figure into a single multi-page PDF (SVG figures are skipped)."""
    logger = get_logger()
    from PIL import Image

    images = []
    for _, data_uri in _iter_base64_images(results):
        if "image/png" not in data_uri:
            continue
        try:
            _, encoded = data_uri.split(",", 1)
            raw = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(img)
        except Exception:  # noqa: BLE001
            continue

    if not images:
        logger.warning("No PNG figures available to build a PDF (are figures in SVG format?).")
        return None

    images[0].save(path, save_all=True, append_images=images[1:])
    return path


EXPORTERS = {
    "html": export_html,
    "markdown": export_markdown,
    "json": export_json,
    "excel": export_excel,
    "pdf": export_pdf,
}


def export_all(
    results: dict[str, Any],
    output_dir: Path,
    formats: list[str],
    config: EDAConfig | None = None,
    version: str = "0.1.0",
    basename: str = "report",
) -> dict[str, Path]:
    """Run every requested exporter and return a dict of format -> written path."""
    logger = get_logger()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for fmt in formats:
        try:
            if fmt in ("html", "markdown", "json", "excel"):
                ext = {"html": "html", "markdown": "md", "json": "json", "excel": "xlsx"}[fmt]
                path = output_dir / f"{basename}.{ext}"
                fn = EXPORTERS[fmt]
                if fmt in ("html", "markdown"):
                    fn(results, path, config, version)
                else:
                    fn(results, path)
                written[fmt] = path
            elif fmt == "pdf":
                path = export_pdf(results, output_dir / f"{basename}.pdf")
                if path:
                    written["pdf"] = path
            elif fmt == "csv":
                csv_dir = output_dir / "csv"
                paths = export_csv_bundle(results, csv_dir)
                if paths:
                    written["csv"] = csv_dir
            elif fmt in ("png", "svg", "figures"):
                fig_dir = output_dir / "figures"
                paths = export_figures(results, fig_dir)
                if paths:
                    written["figures"] = fig_dir
            elif fmt == "dashboard":
                from omni_eda.dashboard import build_dashboard

                path = output_dir / f"{basename}_dashboard.html"
                build_dashboard(results, path, config)
                written["dashboard"] = path
            else:
                logger.warning("Unknown export format '%s' - skipped.", fmt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Export format '%s' failed: %s", fmt, exc)

    return written
