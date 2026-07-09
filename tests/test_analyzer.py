from __future__ import annotations

import pandas as pd
import pytest

from omni_eda import EDAConfig, OmniEDA


def _fast_config(tmp_path, **overrides) -> EDAConfig:
    defaults = dict(
        verbose=False,
        output_dir=str(tmp_path / "out"),
        sample_for_plots=300,
        max_rows_for_expensive_ops=500,
        generate_pairplot=True,
        generate_pca=True,
    )
    defaults.update(overrides)
    return EDAConfig(**defaults)


def test_full_pipeline_runs(basic_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(basic_df)
    assert results["shape"] == basic_df.shape
    assert len(results["profiles"]) == basic_df.shape[1]
    assert results["quality"].summary["n_rows"] == basic_df.shape[0]
    assert "univariate_plots" in results
    assert "correlation" in results


def test_pipeline_with_target(target_df, tmp_path):
    cfg = _fast_config(tmp_path, target_column="target")
    eda = OmniEDA(config=cfg)
    results = eda.run(target_df)
    assert results["target_analysis"] is not None
    assert results["target_analysis"]["is_classification"] is True


def test_pipeline_with_timeseries(timeseries_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(timeseries_df)
    assert results["timeseries_plots"]  # at least one numeric column x datetime column


def test_generate_html_report(basic_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    eda.run(basic_df)
    path = eda.generate_report(tmp_path / "report.html")
    assert path.exists()
    content = path.read_text()
    assert "{{" not in content  # no unrendered Jinja
    assert "<html" in content.lower()


def test_generate_markdown_report(basic_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    eda.run(basic_df)
    path = eda.generate_report(tmp_path / "report.md", fmt="markdown")
    assert path.exists()
    assert "# " in path.read_text()


def test_export_multiple_formats(basic_df, tmp_path):
    cfg = _fast_config(tmp_path, export_formats=["html", "json", "csv"])
    eda = OmniEDA(config=cfg)
    eda.run(basic_df)
    written = eda.export()
    assert "html" in written
    assert "json" in written
    assert "csv" in written
    assert written["html"].exists()


def test_summary_returns_text(basic_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    eda.run(basic_df)
    text = eda.summary()
    assert "OMNI-EDA SUMMARY" in text


def test_clean_method(messy_df, tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    eda.run(messy_df)
    cleaned = eda.clean()
    assert len(cleaned) <= len(messy_df)


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #
def test_empty_dataframe_zero_rows(tmp_path):
    df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(df)
    assert results["shape"] == (0, 2)


def test_no_columns_raises(tmp_path):
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    with pytest.raises(ValueError):
        eda.run(pd.DataFrame())


def test_single_row(tmp_path):
    df = pd.DataFrame({"a": [1], "b": ["x"], "c": [pd.Timestamp("2021-01-01")]})
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(df)
    assert results["shape"] == (1, 3)


def test_single_column(tmp_path):
    df = pd.DataFrame({"a": range(100)})
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(df)
    assert results["shape"] == (100, 1)


def test_all_null_column(tmp_path):
    df = pd.DataFrame({"a": range(50), "allnull": [None] * 50})
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(config=cfg)
    results = eda.run(df)
    categories = {i.category for i in results["quality"].issues}
    assert "all_null_columns" in categories


def test_run_without_data_raises():
    eda = OmniEDA(config=EDAConfig(verbose=False))
    with pytest.raises(ValueError):
        eda.run()


def test_construct_from_csv_path(tmp_path, basic_df):
    csv_path = tmp_path / "data.csv"
    basic_df.to_csv(csv_path, index=False)
    cfg = _fast_config(tmp_path)
    eda = OmniEDA(csv_path, config=cfg)
    results = eda.run()
    assert results["shape"][0] == len(basic_df)


def test_wide_dataframe_many_columns(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    n_cols = 60
    df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, 200) for i in range(n_cols)})
    cfg = _fast_config(tmp_path, max_cols_for_pairwise=20)
    eda = OmniEDA(config=cfg)
    results = eda.run(df)
    assert results["shape"][1] == n_cols
