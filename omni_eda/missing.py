"""Missing-value analysis: summaries plus matrix/heatmap/dendrogram/bar visualizations."""

from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
from matplotlib.figure import Figure

from omni_eda.config import EDAConfig
from omni_eda.utils import sample_df

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    missing = df.isna().sum()
    out = pd.DataFrame(
        {
            "column": missing.index,
            "n_missing": missing.values,
            "pct_missing": (missing.values / n * 100.0) if n else 0.0,
        }
    ).sort_values("n_missing", ascending=False)
    out["n_present"] = n - out["n_missing"]
    return out.reset_index(drop=True)


def missing_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between columns' *missingness* patterns (nullity correlation)."""
    null_df = df.isna()
    null_df = null_df.loc[:, null_df.nunique() > 1]  # drop columns that are never/always missing
    if null_df.shape[1] < 2:
        return pd.DataFrame()
    return null_df.astype(int).corr()


def plot_missing_bar(df: pd.DataFrame, config: EDAConfig | None = None) -> Figure:
    cfg = config or EDAConfig()
    summary = missing_summary(df)
    summary = summary[summary["n_missing"] > 0]
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(summary)), 5), dpi=cfg.figure_dpi)
    if summary.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.bar(summary["column"], summary["pct_missing"], color="#4C72B0")
    ax.set_ylabel("% missing")
    ax.set_title("Missing values by column")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def plot_missing_matrix(df: pd.DataFrame, config: EDAConfig | None = None, max_rows: int = 500) -> Figure:
    cfg = config or EDAConfig()
    sampled = sample_df(df, max_rows, cfg.random_state)
    null_mask = sampled.isna().to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * df.shape[1]), 6), dpi=cfg.figure_dpi)
    ax.imshow(~null_mask, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title(f"Missing value matrix (white = missing, sampled {len(sampled)} rows)")
    fig.tight_layout()
    return fig


def plot_missing_heatmap(df: pd.DataFrame, config: EDAConfig | None = None) -> Figure | None:
    cfg = config or EDAConfig()
    corr = missing_correlation(df)
    if corr.empty:
        return None
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(corr)), max(4, 0.5 * len(corr))), dpi=cfg.figure_dpi)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(corr) <= 15, fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Missingness correlation (do columns go missing together?)")
    fig.tight_layout()
    return fig


def plot_missing_dendrogram(df: pd.DataFrame, config: EDAConfig | None = None) -> Figure | None:
    cfg = config or EDAConfig()
    null_df = df.isna()
    null_df = null_df.loc[:, null_df.nunique() > 1]
    if null_df.shape[1] < 2:
        return None
    from scipy.cluster.hierarchy import dendrogram, linkage

    linkage_matrix = linkage(null_df.astype(int).T, method="average", metric="hamming")
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * null_df.shape[1]), 5), dpi=cfg.figure_dpi)
    dendrogram(linkage_matrix, labels=null_df.columns.tolist(), ax=ax, leaf_rotation=90)
    ax.set_title("Missingness pattern dendrogram")
    fig.tight_layout()
    return fig


def compute_missing_analysis(df: pd.DataFrame, config: EDAConfig | None = None) -> dict[str, Any]:
    config or EDAConfig()
    summary = missing_summary(df)
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(summary["n_missing"].sum())

    rows_with_missing = int(df.isna().any(axis=1).sum())
    complete_rows = len(df) - rows_with_missing

    return {
        "summary_table": summary,
        "total_missing_cells": total_missing,
        "total_cells": total_cells,
        "overall_missing_pct": (total_missing / total_cells * 100.0) if total_cells else 0.0,
        "rows_with_any_missing": rows_with_missing,
        "complete_rows": complete_rows,
        "columns_with_missing": int((summary["n_missing"] > 0).sum()),
    }
