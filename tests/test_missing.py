from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda.missing import compute_missing_analysis, missing_correlation, missing_summary


def test_missing_summary_counts():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    summary = missing_summary(df)
    row_a = summary[summary["column"] == "a"].iloc[0]
    assert row_a["n_missing"] == 2
    assert row_a["pct_missing"] == 50.0


def test_missing_correlation_detects_joint_missingness():
    n = 200
    rng = np.random.default_rng(0)
    base_missing = rng.random(n) < 0.3
    df = pd.DataFrame(
        {"a": np.where(base_missing, np.nan, 1.0), "b": np.where(base_missing, np.nan, 2.0), "c": rng.normal(0, 1, n)}
    )
    corr = missing_correlation(df)
    assert corr.loc["a", "b"] > 0.9


def test_compute_missing_analysis(default_config):
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    result = compute_missing_analysis(df, default_config)
    assert result["columns_with_missing"] == 2
    assert result["total_missing_cells"] == 3
    assert 0 < result["overall_missing_pct"] < 100


def test_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3]})
    summary = missing_summary(df)
    assert (summary["n_missing"] == 0).all()
