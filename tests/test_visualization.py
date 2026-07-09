from __future__ import annotations

import pandas as pd

from omni_eda.detection import ColumnTypeDetector
from omni_eda.visualization import (
    PlotEngine,
    plot_boxplot,
    plot_countplot,
    plot_histogram_kde,
    plot_pie,
    plot_scatter,
)


def test_plot_histogram_returns_figure():
    s = pd.Series(range(100))
    fig = plot_histogram_kde(s, "x")
    assert fig is not None


def test_plot_histogram_none_for_constant_series():
    s = pd.Series([5] * 50)
    fig = plot_histogram_kde(s, "x")
    assert fig is None  # a constant column can't have a meaningful histogram/KDE


def test_plot_boxplot_returns_figure():
    s = pd.Series(range(50))
    fig = plot_boxplot(s, "x")
    assert fig is not None


def test_plot_countplot_returns_figure():
    s = pd.Series(["a", "b", "a", "c"] * 10)
    fig = plot_countplot(s, "x")
    assert fig is not None


def test_plot_pie_skips_high_cardinality():
    s = pd.Series([f"cat_{i}" for i in range(50)])
    fig = plot_pie(s, "x", max_categories=8)
    assert fig is None


def test_plot_scatter_returns_figure():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    fig = plot_scatter(df, "x", "y")
    assert fig is not None


def test_plot_engine_univariate(basic_df, basic_profiles, default_config):
    engine = PlotEngine(default_config)
    plots = engine.univariate_plots(basic_df, basic_profiles)
    assert "age" in plots
    assert all(v.startswith("data:image") for v in plots["age"].values())


def test_plot_engine_missing_plots(default_config):
    df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": range(5)})
    engine = PlotEngine(default_config)
    plots = engine.missing_plots(df)
    assert "bar" in plots
    assert plots["bar"].startswith("data:image")


def test_plot_engine_handles_all_constant_dataframe(default_config):
    df = pd.DataFrame({"a": ["x"] * 20, "b": [1] * 20})
    engine = PlotEngine(default_config)
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    # should not raise even though there's nothing meaningful to plot
    plots = engine.univariate_plots(df, profiles)
    assert isinstance(plots, dict)
