"""The visualization engine.

Every ``plot_*`` function is a pure function: it takes data (and maybe a
column profile) and returns a :class:`matplotlib.figure.Figure`, or
``None`` when the plot genuinely doesn't apply (e.g. a pie chart for a
column with 300 categories). The :class:`PlotEngine` orchestrator below
decides *which* of these to call for a given dataset and is the only part
of this module that needs to know about column counts, sampling limits,
etc. -- i.e. it is where "intelligently skip impossible plots" lives.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger, progress
from omni_eda.themes import apply_theme
from omni_eda.utils import sample_df


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def fig_to_base64(fig: Figure, fmt: str = "png", dpi: int = 110) -> str:
    """Encode a matplotlib figure as a base64 data URI and close it."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
    finally:
        plt.close(fig)
    mime = "image/svg+xml" if fmt == "svg" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _new_fig(figsize: tuple[float, float] = (7, 4.5), dpi: int = 110) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


# --------------------------------------------------------------------------- #
# Univariate plots
# --------------------------------------------------------------------------- #
def plot_histogram_kde(series: pd.Series, name: str) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.nunique() < 2:
        return None
    import seaborn as sns

    fig, ax = _new_fig()
    sns.histplot(s, kde=True, ax=ax)
    ax.set_title(f"Distribution of {name}")
    ax.set_xlabel(name)
    fig.tight_layout()
    return fig


def plot_boxplot(series: pd.Series, name: str) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    fig, ax = _new_fig(figsize=(4, 5))
    ax.boxplot(s, vert=True, patch_artist=True)
    ax.set_title(f"Boxplot of {name}")
    ax.set_xticklabels([name])
    fig.tight_layout()
    return fig


def plot_violin(series: pd.Series, name: str) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.nunique() < 2:
        return None
    import seaborn as sns

    fig, ax = _new_fig(figsize=(4, 5))
    sns.violinplot(y=s, ax=ax)
    ax.set_title(f"Violin plot of {name}")
    fig.tight_layout()
    return fig


def plot_strip(series: pd.Series, name: str, max_points: int = 2000) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if len(s) > max_points:
        s = s.sample(max_points, random_state=0)
    import seaborn as sns

    fig, ax = _new_fig(figsize=(4, 5))
    sns.stripplot(y=s, ax=ax, alpha=0.5, size=3)
    ax.set_title(f"Strip plot of {name}")
    fig.tight_layout()
    return fig


def plot_swarm(series: pd.Series, name: str, max_points: int = 500) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if len(s) > max_points:
        s = s.sample(max_points, random_state=0)
    import seaborn as sns

    fig, ax = _new_fig(figsize=(4, 5))
    try:
        sns.swarmplot(y=s, ax=ax, size=3)
    except Exception:
        return None
    ax.set_title(f"Swarm plot of {name}")
    fig.tight_layout()
    return fig


def plot_ecdf(series: pd.Series, name: str) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if s.empty:
        return None
    y = np.arange(1, len(s) + 1) / len(s)
    fig, ax = _new_fig()
    ax.plot(s, y, marker=".", linestyle="none", markersize=2)
    ax.set_xlabel(name)
    ax.set_ylabel("ECDF")
    ax.set_title(f"Empirical CDF of {name}")
    fig.tight_layout()
    return fig


def plot_rug(series: pd.Series, name: str, max_points: int = 3000) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if len(s) > max_points:
        s = s.sample(max_points, random_state=0)
    import seaborn as sns

    fig, ax = _new_fig(figsize=(7, 1.5))
    sns.rugplot(s, ax=ax)
    ax.set_title(f"Rug plot of {name}")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_countplot(series: pd.Series, name: str, max_categories: int = 20) -> Figure | None:
    counts = series.value_counts().head(max_categories)
    if counts.empty:
        return None
    fig, ax = _new_fig(figsize=(max(5, 0.4 * len(counts)), 4.5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Counts for {name}" + (f" (top {max_categories})" if series.nunique() > max_categories else ""))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_barplot_aggregate(
    df: pd.DataFrame, category_col: str, value_col: str, agg: str = "mean", max_categories: int = 20
) -> Figure | None:
    grouped = df.groupby(category_col, observed=True)[value_col].agg(agg).sort_values(ascending=False).head(max_categories)
    if grouped.empty:
        return None
    fig, ax = _new_fig(figsize=(max(5, 0.4 * len(grouped)), 4.5))
    ax.bar(grouped.index.astype(str), grouped.values)
    ax.set_title(f"{agg.title()} of {value_col} by {category_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_pie(series: pd.Series, name: str, max_categories: int = 8) -> Figure | None:
    counts = series.value_counts()
    if counts.empty or len(counts) > max_categories:
        return None
    fig, ax = _new_fig(figsize=(5, 5))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Share of {name}")
    fig.tight_layout()
    return fig


def plot_qq(series: pd.Series, name: str) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return None
    from scipy import stats as scipy_stats

    fig, ax = _new_fig()
    scipy_stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot of {name}")
    fig.tight_layout()
    return fig


def plot_rank(series: pd.Series, name: str, max_points: int = 5000) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if s.empty:
        return None
    if len(s) > max_points:
        s = s.iloc[np.linspace(0, len(s) - 1, max_points).astype(int)]
    fig, ax = _new_fig()
    ax.plot(range(len(s)), s.values)
    ax.set_xlabel("Rank")
    ax.set_ylabel(name)
    ax.set_title(f"Rank plot of {name}")
    fig.tight_layout()
    return fig


def plot_lollipop(series: pd.Series, name: str, max_categories: int = 15) -> Figure | None:
    counts = series.value_counts().head(max_categories).sort_values()
    if counts.empty:
        return None
    fig, ax = _new_fig(figsize=(6, max(3, 0.35 * len(counts))))
    ax.hlines(y=counts.index.astype(str), xmin=0, xmax=counts.values, color="#4C72B0")
    ax.plot(counts.values, counts.index.astype(str), "o", color="#4C72B0")
    ax.set_title(f"Lollipop chart of {name}")
    fig.tight_layout()
    return fig


def plot_stem(series: pd.Series, name: str, max_points: int = 200) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if len(s) > max_points:
        s = s.iloc[:max_points]
    fig, ax = _new_fig()
    ax.stem(range(len(s)), s.values)
    ax.set_title(f"Stem plot of {name} (first {len(s)} values)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Bivariate plots
# --------------------------------------------------------------------------- #
def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None, max_points: int = 5000) -> Figure | None:
    data = df[[x, y] + ([hue] if hue else [])].apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != hue else c)
    data = data.dropna(subset=[x, y])
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    fig, ax = _new_fig()
    if hue and hue in data.columns:
        for level, group in data.groupby(hue, observed=True):
            ax.scatter(group[x], group[y], s=10, alpha=0.6, label=str(level))
        ax.legend(fontsize=7, title=hue)
    else:
        ax.scatter(data[x], data[y], s=10, alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    return fig


def plot_hexbin(df: pd.DataFrame, x: str, y: str) -> Figure | None:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        return None
    fig, ax = _new_fig()
    hb = ax.hexbin(data[x], data[y], gridsize=30, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Hexbin: {y} vs {x}")
    fig.tight_layout()
    return fig


def plot_regression(df: pd.DataFrame, x: str, y: str, max_points: int = 5000) -> Figure | None:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    import seaborn as sns

    fig, ax = _new_fig()
    sns.regplot(data=data, x=x, y=y, ax=ax, scatter_kws={"s": 10, "alpha": 0.5}, line_kws={"color": "red"})
    ax.set_title(f"Regression: {y} ~ {x}")
    fig.tight_layout()
    return fig


def plot_residual(df: pd.DataFrame, x: str, y: str, max_points: int = 5000) -> Figure | None:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    coeffs = np.polyfit(data[x], data[y], 1)
    predicted = np.polyval(coeffs, data[x])
    residuals = data[y] - predicted
    fig, ax = _new_fig()
    ax.scatter(predicted, residuals, s=10, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual plot: {y} ~ {x}")
    fig.tight_layout()
    return fig


def plot_joint(df: pd.DataFrame, x: str, y: str, max_points: int = 3000):
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 10:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    import seaborn as sns

    grid = sns.jointplot(data=data, x=x, y=y, kind="scatter", height=5, joint_kws={"s": 10, "alpha": 0.5})
    grid.fig.suptitle(f"Joint distribution: {x} & {y}", y=1.02)
    return grid.fig


def plot_pairplot(df: pd.DataFrame, numeric_cols: list[str], hue: str | None = None, max_cols: int = 6, max_points: int = 1500):
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        return None
    data = df[cols + ([hue] if hue and hue in df.columns else [])].copy()
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=cols)
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    if data.empty:
        return None
    import seaborn as sns

    grid = sns.pairplot(
        data, vars=cols, hue=hue if hue in data.columns else None, diag_kind="kde", plot_kws={"s": 10, "alpha": 0.5}
    )
    grid.fig.suptitle("Pairwise relationships", y=1.02)
    return grid.fig


def plot_line(df: pd.DataFrame, x: str, y: str, max_points: int = 5000) -> Figure | None:
    data = df[[x, y]].copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna().sort_values(x)
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.iloc[np.linspace(0, len(data) - 1, max_points).astype(int)]
    fig, ax = _new_fig()
    ax.plot(data[x], data[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} over {x}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_area(df: pd.DataFrame, x: str, y: str, max_points: int = 5000) -> Figure | None:
    data = df[[x, y]].copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna().sort_values(x)
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.iloc[np.linspace(0, len(data) - 1, max_points).astype(int)]
    fig, ax = _new_fig()
    ax.fill_between(data[x], data[y], alpha=0.4)
    ax.plot(data[x], data[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} over {x} (area)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_grouped_bar(df: pd.DataFrame, cat1: str, cat2: str, max_categories: int = 8) -> Figure | None:
    top1 = df[cat1].value_counts().head(max_categories).index
    top2 = df[cat2].value_counts().head(max_categories).index
    sub = df[df[cat1].isin(top1) & df[cat2].isin(top2)]
    ct = pd.crosstab(sub[cat1], sub[cat2])
    if ct.empty:
        return None
    fig, ax = _new_fig(figsize=(max(6, 0.6 * len(ct)), 5))
    ct.plot(kind="bar", ax=ax)
    ax.set_title(f"{cat1} grouped by {cat2}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_stacked_bar(df: pd.DataFrame, cat1: str, cat2: str, max_categories: int = 10) -> Figure | None:
    top1 = df[cat1].value_counts().head(max_categories).index
    top2 = df[cat2].value_counts().head(max_categories).index
    sub = df[df[cat1].isin(top1) & df[cat2].isin(top2)]
    ct = pd.crosstab(sub[cat1], sub[cat2])
    if ct.empty:
        return None
    fig, ax = _new_fig(figsize=(max(6, 0.6 * len(ct)), 5))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"{cat1} vs {cat2} (stacked)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(matrix: pd.DataFrame, title: str = "Correlation heatmap") -> Figure | None:
    if matrix is None or matrix.empty or matrix.shape[0] < 2:
        return None
    import seaborn as sns

    fig, ax = _new_fig(figsize=(max(5, 0.55 * len(matrix)), max(4, 0.5 * len(matrix))))
    sns.heatmap(matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=len(matrix) <= 15, fmt=".2f", ax=ax, square=True)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mosaic(df: pd.DataFrame, cat1: str, cat2: str, max_categories: int = 6) -> Figure | None:
    top1 = df[cat1].value_counts().head(max_categories).index
    top2 = df[cat2].value_counts().head(max_categories).index
    sub = df[df[cat1].isin(top1) & df[cat2].isin(top2)]
    if sub.empty:
        return None
    try:
        from statsmodels.graphics.mosaicplot import mosaic
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        mosaic(sub, [cat1, cat2], ax=ax, title=f"Mosaic plot: {cat1} x {cat2}")
    except Exception:
        plt.close(fig)
        return None
    fig.tight_layout()
    return fig


def plot_crosstab_heatmap(df: pd.DataFrame, cat1: str, cat2: str, max_categories: int = 15) -> Figure | None:
    top1 = df[cat1].value_counts().head(max_categories).index
    top2 = df[cat2].value_counts().head(max_categories).index
    sub = df[df[cat1].isin(top1) & df[cat2].isin(top2)]
    ct = pd.crosstab(sub[cat1], sub[cat2])
    if ct.empty:
        return None
    import seaborn as sns

    fig, ax = _new_fig(figsize=(max(5, 0.5 * ct.shape[1]), max(4, 0.5 * ct.shape[0])))
    sns.heatmap(ct, annot=ct.size <= 200, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Cross-tab heatmap: {cat1} x {cat2}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Multivariate plots
# --------------------------------------------------------------------------- #
def plot_facetgrid(df: pd.DataFrame, numeric_col: str, facet_col: str, max_facets: int = 6, max_points: int = 3000):
    top = df[facet_col].value_counts().head(max_facets).index
    sub = df[df[facet_col].isin(top)][[numeric_col, facet_col]].copy()
    sub[numeric_col] = pd.to_numeric(sub[numeric_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return None
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    import seaborn as sns

    grid = sns.FacetGrid(sub, col=facet_col, col_wrap=min(4, max_facets), height=2.6)
    grid.map(sns.histplot, numeric_col, kde=True)
    grid.fig.suptitle(f"{numeric_col} distribution by {facet_col}", y=1.03)
    return grid.fig


def plot_clustermap(matrix: pd.DataFrame):
    if matrix is None or matrix.shape[0] < 3 or matrix.isna().all().all():
        return None
    import seaborn as sns

    clean = matrix.fillna(0)
    try:
        grid = sns.clustermap(clean, cmap="coolwarm", center=0, figsize=(max(5, 0.5 * len(clean)), max(5, 0.5 * len(clean))))
    except Exception:
        return None
    grid.fig.suptitle("Clustered correlation map", y=1.02)
    return grid.fig


def plot_parallel_coordinates(
    df: pd.DataFrame, numeric_cols: list[str], class_col: str, max_cols: int = 8, max_points: int = 1000
) -> Figure | None:
    cols = numeric_cols[:max_cols]
    if len(cols) < 2 or class_col not in df.columns:
        return None
    from pandas.plotting import parallel_coordinates

    data = df[cols + [class_col]].copy()
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    for col in cols:  # normalize so differing scales don't dominate the plot
        span = data[col].max() - data[col].min()
        data[col] = (data[col] - data[col].min()) / span if span else 0.0
    fig, ax = _new_fig(figsize=(max(7, len(cols)), 5))
    parallel_coordinates(data, class_col, cols=cols, ax=ax, alpha=0.4)
    ax.set_title("Parallel coordinates (min-max normalized)")
    fig.tight_layout()
    return fig


def plot_andrews_curves(
    df: pd.DataFrame, numeric_cols: list[str], class_col: str, max_cols: int = 8, max_points: int = 500
) -> Figure | None:
    cols = numeric_cols[:max_cols]
    if len(cols) < 2 or class_col not in df.columns:
        return None
    from pandas.plotting import andrews_curves

    data = df[cols + [class_col]].copy()
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    fig, ax = _new_fig()
    andrews_curves(data, class_col, ax=ax)
    ax.set_title("Andrews curves")
    fig.tight_layout()
    return fig


def plot_radar_chart(
    df: pd.DataFrame, numeric_cols: list[str], class_col: str, max_cols: int = 8, max_classes: int = 6
) -> Figure | None:
    cols = numeric_cols[:max_cols]
    if len(cols) < 3 or class_col not in df.columns:
        return None
    data = df[cols + [class_col]].copy()
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    grouped = data.groupby(class_col, observed=True)[cols].mean().head(max_classes)
    if grouped.empty:
        return None
    normalized = (grouped - grouped.min()) / (grouped.max() - grouped.min()).replace(0, 1)
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for idx, row in normalized.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=str(idx))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_title(f"Radar chart: mean of {len(cols)} features by {class_col}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    fig.tight_layout()
    return fig


def plot_bubble_chart(
    df: pd.DataFrame, x: str, y: str, size_col: str, hue: str | None = None, max_points: int = 2000
) -> Figure | None:
    cols = [x, y, size_col] + ([hue] if hue else [])
    data = df[cols].copy()
    for c in (x, y, size_col):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[x, y, size_col])
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    sizes = data[size_col]
    span = sizes.max() - sizes.min()
    scaled_sizes = 20 + 200 * ((sizes - sizes.min()) / span if span else 0.5)
    fig, ax = _new_fig()
    if hue and hue in data.columns:
        for level, group in data.groupby(hue, observed=True):
            ax.scatter(group[x], group[y], s=scaled_sizes.loc[group.index], alpha=0.5, label=str(level))
        ax.legend(fontsize=7, title=hue)
    else:
        ax.scatter(data[x], data[y], s=scaled_sizes, alpha=0.5, color="#4C72B0")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Bubble chart: {y} vs {x} (size = {size_col})")
    fig.tight_layout()
    return fig


def plot_3d_scatter(df: pd.DataFrame, x: str, y: str, z: str, hue: str | None = None, max_points: int = 2000) -> Figure | None:
    cols = [x, y, z] + ([hue] if hue else [])
    data = df[cols].copy()
    for c in (x, y, z):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[x, y, z])
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.sample(max_points, random_state=0)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    if hue and hue in data.columns:
        for level, group in data.groupby(hue, observed=True):
            ax.scatter(group[x], group[y], group[z], s=10, alpha=0.6, label=str(level))
        ax.legend(fontsize=7)
    else:
        ax.scatter(data[x], data[y], data[z], s=10, alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f"3D scatter: {x}, {y}, {z}")
    fig.tight_layout()
    return fig


def plot_pca(df: pd.DataFrame, numeric_cols: list[str], hue: str | None = None, max_points: int = 5000) -> Figure | None:
    if len(numeric_cols) < 2:
        return None
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    hue_series = df[hue] if hue and hue in df.columns else None
    mask = data.notna().all(axis=1)
    data = data[mask]
    if hue_series is not None:
        hue_series = hue_series[mask]
    if len(data) < 5:
        return None
    if len(data) > max_points:
        idx = data.sample(max_points, random_state=0).index
        data = data.loc[idx]
        if hue_series is not None:
            hue_series = hue_series.loc[idx]

    scaled = StandardScaler().fit_transform(data)
    n_components = min(2, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    components = pca.fit_transform(scaled)
    fig, ax = _new_fig()
    if n_components == 1:
        ax.scatter(components[:, 0], np.zeros_like(components[:, 0]), s=10, alpha=0.6)
    elif hue_series is not None:
        for level in pd.unique(hue_series):
            m = (hue_series == level).to_numpy()
            ax.scatter(components[m, 0], components[m, 1], s=10, alpha=0.6, label=str(level))
        ax.legend(fontsize=7, title=hue)
    else:
        ax.scatter(components[:, 0], components[:, 1], s=10, alpha=0.6)
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    if n_components > 1:
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title("PCA projection")
    fig.tight_layout()
    return fig


def plot_tsne(df: pd.DataFrame, numeric_cols: list[str], hue: str | None = None, max_points: int = 2000) -> Figure | None:
    if len(numeric_cols) < 2:
        return None
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    hue_series = df[hue] if hue and hue in df.columns else None
    mask = data.notna().all(axis=1)
    data, hue_series = data[mask], (hue_series[mask] if hue_series is not None else None)
    if len(data) > max_points:
        idx = data.sample(max_points, random_state=0).index
        data = data.loc[idx]
        hue_series = hue_series.loc[idx] if hue_series is not None else None
    if len(data) < 30:
        return None

    scaled = StandardScaler().fit_transform(data)
    try:
        embedding = TSNE(n_components=2, random_state=0, init="pca", perplexity=min(30, max(5, len(data) // 10))).fit_transform(
            scaled
        )
    except Exception:
        return None
    fig, ax = _new_fig()
    if hue_series is not None:
        for level in pd.unique(hue_series):
            m = (hue_series == level).to_numpy()
            ax.scatter(embedding[m, 0], embedding[m, 1], s=10, alpha=0.6, label=str(level))
        ax.legend(fontsize=7, title=hue)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6)
    ax.set_title(f"t-SNE projection (n={len(data)})")
    fig.tight_layout()
    return fig


def plot_umap(df: pd.DataFrame, numeric_cols: list[str], hue: str | None = None, max_points: int = 5000) -> Figure | None:
    try:
        import umap  # type: ignore
    except ImportError:
        return None  # optional dependency not installed; gracefully skipped
    if len(numeric_cols) < 2:
        return None
    from sklearn.preprocessing import StandardScaler

    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    hue_series = df[hue] if hue and hue in df.columns else None
    mask = data.notna().all(axis=1)
    data, hue_series = data[mask], (hue_series[mask] if hue_series is not None else None)
    if len(data) > max_points:
        idx = data.sample(max_points, random_state=0).index
        data = data.loc[idx]
        hue_series = hue_series.loc[idx] if hue_series is not None else None
    if len(data) < 15:
        return None
    scaled = StandardScaler().fit_transform(data)
    embedding = umap.UMAP(random_state=0).fit_transform(scaled)
    fig, ax = _new_fig()
    if hue_series is not None:
        for level in pd.unique(hue_series):
            m = (hue_series == level).to_numpy()
            ax.scatter(embedding[m, 0], embedding[m, 1], s=10, alpha=0.6, label=str(level))
        ax.legend(fontsize=7, title=hue)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6)
    ax.set_title("UMAP projection")
    fig.tight_layout()
    return fig


def plot_correlation_network(matrix: pd.DataFrame, threshold: float = 0.5, max_nodes: int = 40) -> Figure | None:
    if matrix is None or matrix.empty:
        return None
    try:
        import networkx as nx
    except ImportError:
        return None

    cols = list(matrix.columns)[:max_nodes]
    graph = nx.Graph()
    graph.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                graph.add_edge(a, b, weight=float(value))
    if graph.number_of_edges() == 0:
        return None

    fig, ax = _new_fig(figsize=(7, 7))
    pos = nx.spring_layout(graph, seed=0, k=1.2 / max(1, np.sqrt(graph.number_of_nodes())))
    weights = [abs(graph[u][v]["weight"]) * 3 for u, v in graph.edges()]
    colors = ["#C44E52" if graph[u][v]["weight"] < 0 else "#4C72B0" for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=350, node_color="#DDDDDD", edgecolors="#555555")
    nx.draw_networkx_edges(graph, pos, ax=ax, width=weights, edge_color=colors, alpha=0.6)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)
    ax.set_title(f"Correlation network (|r| >= {threshold})")
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Time series plots
# --------------------------------------------------------------------------- #
def plot_trend(df: pd.DataFrame, date_col: str, value_col: str, max_points: int = 5000) -> Figure | None:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", format="mixed")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if data.empty:
        return None
    if len(data) > max_points:
        data = data.iloc[np.linspace(0, len(data) - 1, max_points).astype(int)]
    fig, ax = _new_fig()
    ax.plot(data[date_col], data[value_col], linewidth=1)
    ax.set_title(f"Trend of {value_col} over {date_col}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_seasonality(df: pd.DataFrame, date_col: str, value_col: str) -> Figure | None:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", format="mixed")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return None
    monthly = data.groupby(data[date_col].dt.month)[value_col].mean()
    fig, ax = _new_fig()
    ax.bar(monthly.index, monthly.values)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Mean {value_col}")
    ax.set_title(f"Seasonality of {value_col} by month")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    return fig


def plot_rolling(df: pd.DataFrame, date_col: str, value_col: str, window: int = 7) -> Figure | None:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", format="mixed")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if len(data) < window * 2:
        return None
    data = data.set_index(date_col)
    rolling_mean = data[value_col].rolling(window).mean()
    rolling_std = data[value_col].rolling(window).std()
    fig, ax = _new_fig()
    ax.plot(data.index, data[value_col], alpha=0.35, label="raw", linewidth=0.8)
    ax.plot(rolling_mean.index, rolling_mean, label=f"rolling mean ({window})", color="#C44E52")
    ax.fill_between(
        data.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.15, color="#C44E52", label="±1 std"
    )
    ax.legend(fontsize=7)
    ax.set_title(f"Rolling mean/std of {value_col} (window={window})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_lag(series: pd.Series, name: str, lag: int = 1) -> Figure | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= lag + 1:
        return None
    fig, ax = _new_fig(figsize=(5, 5))
    ax.scatter(s[:-lag], s[lag:], s=8, alpha=0.5)
    ax.set_xlabel(f"{name}(t)")
    ax.set_ylabel(f"{name}(t+{lag})")
    ax.set_title(f"Lag plot of {name} (lag={lag})")
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, name: str, lags: int = 30):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < lags * 2:
        return None
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        plot_acf(s, lags=min(lags, len(s) // 2 - 1), ax=axes[0])
        plot_pacf(s, lags=min(lags, len(s) // 2 - 1), ax=axes[1], method="ywm")
    except Exception:
        plt.close(fig)
        return None
    axes[0].set_title(f"ACF: {name}")
    axes[1].set_title(f"PACF: {name}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class PlotEngine:
    """Decides which plots make sense for a dataset and generates them.

    All figures are returned already base64-encoded (and closed), so the
    engine can run over datasets with hundreds of columns without holding
    hundreds of open matplotlib Figure objects in memory at once.
    """

    def __init__(self, config: EDAConfig | None = None):
        self.config = config or EDAConfig()
        self.logger = get_logger()
        apply_theme(self.config.theme)

    def _encode(self, fig: Figure | None) -> str | None:
        if fig is None:
            return None
        return fig_to_base64(fig, fmt=self.config.figure_format, dpi=self.config.figure_dpi)

    def univariate_plots(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, dict[str, str]]:
        cfg = self.config
        out: dict[str, dict[str, str]] = {}
        working = sample_df(df, cfg.sample_for_plots, cfg.random_state)

        numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant and not cfg.is_ignored(c)]
        categorical_cols = [
            c for c, p in profiles.items() if (p.is_categorical or p.is_boolean) and not p.is_constant and not cfg.is_ignored(c)
        ]

        for col in progress(numeric_cols, desc="Univariate (numeric)", enabled=cfg.verbose):
            plots = {
                "histogram_kde": plot_histogram_kde(working[col], col),
                "boxplot": plot_boxplot(working[col], col),
                "violin": plot_violin(working[col], col),
                "ecdf": plot_ecdf(working[col], col),
                "qq": plot_qq(working[col], col),
                "rank": plot_rank(working[col], col),
            }
            encoded = {k: self._encode(v) for k, v in plots.items() if v is not None}
            if encoded:
                out[col] = encoded

        for col in progress(categorical_cols, desc="Univariate (categorical)", enabled=cfg.verbose):
            plots = {
                "countplot": plot_countplot(working[col], col, cfg.max_categories_for_plot),
                "pie": plot_pie(working[col], col),
                "lollipop": plot_lollipop(working[col], col),
            }
            encoded = {k: self._encode(v) for k, v in plots.items() if v is not None}
            if encoded:
                out[col] = encoded

        return out

    def bivariate_plots(
        self,
        df: pd.DataFrame,
        profiles: dict[str, ColumnProfile],
        correlations: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        cfg = self.config
        out: dict[str, str] = {}
        working = sample_df(df, cfg.sample_for_plots, cfg.random_state)

        [c for c, p in profiles.items() if p.is_numeric and not p.is_constant][: cfg.max_cols_for_pairwise]
        categorical_cols = [
            c for c, p in profiles.items() if (p.is_categorical or p.is_boolean) and not p.is_constant and p.n_unique <= 30
        ][: cfg.max_cols_for_pairwise]

        if correlations and "numeric" in correlations:
            primary = cfg.correlation_methods[0] if cfg.correlation_methods else "pearson"
            matrix = correlations["numeric"].get(primary)
            fig = self._encode(plot_correlation_heatmap(matrix, title=f"{primary.title()} correlation heatmap"))
            if fig:
                out["correlation_heatmap"] = fig
            if correlations.get("categorical_cramers_v") is not None and not correlations["categorical_cramers_v"].empty:
                fig = self._encode(
                    plot_correlation_heatmap(correlations["categorical_cramers_v"], title="Cramer's V association heatmap")
                )
                if fig:
                    out["cramers_v_heatmap"] = fig

        # top correlated numeric pair -> scatter + regression + hexbin + residual
        top_pairs = (correlations or {}).get("high_correlation_pairs", [])[:3]
        for _i, pair in enumerate(top_pairs):
            a, b = pair["col_a"], pair["col_b"]
            if (
                profiles.get(a, ColumnProfile(a, "", 0, 0, 0, 0, 0)).is_numeric
                and profiles.get(b, ColumnProfile(b, "", 0, 0, 0, 0, 0)).is_numeric
            ):
                for plot_name, fn in (
                    ("scatter", plot_scatter),
                    ("regression", plot_regression),
                    ("hexbin", plot_hexbin),
                    ("residual", plot_residual),
                ):
                    fig = self._encode(fn(working, a, b))
                    if fig:
                        out[f"{plot_name}_{a}_vs_{b}"] = fig

        if len(categorical_cols) >= 2:
            c1, c2 = categorical_cols[0], categorical_cols[1]
            for plot_name, fn in (
                ("grouped_bar", plot_grouped_bar),
                ("stacked_bar", plot_stacked_bar),
                ("crosstab_heatmap", plot_crosstab_heatmap),
                ("mosaic", plot_mosaic),
            ):
                fig = self._encode(fn(working, c1, c2))
                if fig:
                    out[f"{plot_name}_{c1}_vs_{c2}"] = fig

        return out

    def multivariate_plots(
        self,
        df: pd.DataFrame,
        profiles: dict[str, ColumnProfile],
        correlations: dict[str, Any] | None = None,
        target: str | None = None,
    ) -> dict[str, str]:
        cfg = self.config
        out: dict[str, str] = {}
        working = sample_df(df, cfg.sample_for_plots, cfg.random_state)
        numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant]
        categorical_cols = [
            c for c, p in profiles.items() if (p.is_categorical or p.is_boolean) and not p.is_constant and p.n_unique <= 12
        ]

        hue = (
            target
            if target and (profiles.get(target) and (profiles[target].is_categorical or profiles[target].is_boolean))
            else (categorical_cols[0] if categorical_cols else None)
        )

        if cfg.generate_pairplot and len(numeric_cols) >= 2:
            fig = self._encode(plot_pairplot(working, numeric_cols, hue=hue))
            if fig:
                out["pairplot"] = fig

        if len(numeric_cols) >= 2 and categorical_cols:
            fig = self._encode(plot_facetgrid(working, numeric_cols[0], categorical_cols[0]))
            if fig:
                out["facetgrid"] = fig

        if correlations and correlations.get("numeric"):
            primary = cfg.correlation_methods[0] if cfg.correlation_methods else "pearson"
            matrix = correlations["numeric"].get(primary)
            fig = self._encode(plot_clustermap(matrix))
            if fig:
                out["clustermap"] = fig
            if cfg.generate_correlation_network:
                fig = self._encode(plot_correlation_network(matrix, threshold=cfg.high_correlation_threshold * 0.6))
                if fig:
                    out["correlation_network"] = fig

        if len(numeric_cols) >= 2 and categorical_cols:
            class_col = categorical_cols[0]
            fig = self._encode(plot_parallel_coordinates(working, numeric_cols, class_col))
            if fig:
                out["parallel_coordinates"] = fig
            fig = self._encode(plot_andrews_curves(working, numeric_cols, class_col))
            if fig:
                out["andrews_curves"] = fig
            if len(numeric_cols) >= 3:
                fig = self._encode(plot_radar_chart(working, numeric_cols, class_col))
                if fig:
                    out["radar_chart"] = fig

        if len(numeric_cols) >= 3:
            fig = self._encode(plot_bubble_chart(working, numeric_cols[0], numeric_cols[1], numeric_cols[2], hue=hue))
            if fig:
                out["bubble_chart"] = fig
            fig = self._encode(plot_3d_scatter(working, numeric_cols[0], numeric_cols[1], numeric_cols[2], hue=hue))
            if fig:
                out["scatter_3d"] = fig

        if cfg.generate_pca and len(numeric_cols) >= 2:
            fig = self._encode(plot_pca(working, numeric_cols, hue=hue))
            if fig:
                out["pca"] = fig
            if len(working) <= cfg.max_rows_for_expensive_ops:
                fig = self._encode(plot_tsne(working, numeric_cols, hue=hue))
                if fig:
                    out["tsne"] = fig
            fig = self._encode(plot_umap(working, numeric_cols, hue=hue))  # no-op if umap isn't installed
            if fig:
                out["umap"] = fig

        return out

    def timeseries_plots(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, dict[str, str]]:
        cfg = self.config
        out: dict[str, dict[str, str]] = {}
        datetime_cols = [c for c, p in profiles.items() if p.is_datetime]
        numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant]
        if not datetime_cols or not numeric_cols:
            return out

        working = sample_df(df, cfg.sample_for_plots, cfg.random_state)
        date_col = datetime_cols[0]
        for value_col in numeric_cols[:5]:
            plots = {
                "trend": plot_trend(working, date_col, value_col),
                "seasonality": plot_seasonality(working, date_col, value_col),
                "rolling": plot_rolling(working, date_col, value_col),
                "lag": plot_lag(working[value_col], value_col),
                "acf_pacf": plot_acf_pacf(working[value_col], value_col),
            }
            encoded = {k: self._encode(v) for k, v in plots.items() if v is not None}
            if encoded:
                out[value_col] = encoded
        return out

    def missing_plots(self, df: pd.DataFrame) -> dict[str, str]:
        from omni_eda import missing as missing_mod

        out: dict[str, str] = {}
        fig = self._encode(missing_mod.plot_missing_bar(df, self.config))
        if fig:
            out["bar"] = fig
        fig = self._encode(missing_mod.plot_missing_matrix(df, self.config))
        if fig:
            out["matrix"] = fig
        fig = self._encode(missing_mod.plot_missing_heatmap(df, self.config))
        if fig:
            out["heatmap"] = fig
        fig = self._encode(missing_mod.plot_missing_dendrogram(df, self.config))
        if fig:
            out["dendrogram"] = fig
        return out
