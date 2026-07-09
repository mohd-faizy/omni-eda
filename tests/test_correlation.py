from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda.correlation import compute_correlations, cramers_v, find_highly_correlated_pairs
from omni_eda.detection import ColumnTypeDetector


def test_pearson_correlation_perfect_linear():
    df = pd.DataFrame({"a": range(100)})
    df["b"] = df["a"] * 2 + 1
    df["c"] = np.random.default_rng(0).normal(0, 1, 100)
    from omni_eda.correlation import numeric_correlation_matrices

    matrices = numeric_correlation_matrices(df, ["a", "b", "c"], ["pearson"])
    assert abs(matrices["pearson"].loc["a", "b"] - 1.0) < 1e-6


def test_cramers_v_independent_vs_dependent():
    rng = np.random.default_rng(0)
    n = 2000
    # independent
    x_indep = pd.Series(rng.choice(["a", "b", "c"], n))
    y_indep = pd.Series(rng.choice(["x", "y"], n))
    v_indep = cramers_v(x_indep, y_indep)

    # fully dependent
    x_dep = pd.Series(rng.choice(["a", "b"], n))
    y_dep = x_dep.map({"a": "x", "b": "y"})
    v_dep = cramers_v(x_dep, y_dep)

    assert v_indep < 0.15
    assert v_dep > 0.9


def test_find_highly_correlated_pairs():
    matrix = pd.DataFrame({"a": [1.0, 0.9, 0.1], "b": [0.9, 1.0, 0.2], "c": [0.1, 0.2, 1.0]}, index=["a", "b", "c"])
    pairs = find_highly_correlated_pairs(matrix, threshold=0.8)
    assert len(pairs) == 1
    assert {pairs[0]["col_a"], pairs[0]["col_b"]} == {"a", "b"}


def test_compute_correlations_end_to_end(target_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(target_df)
    result = compute_correlations(target_df, profiles, default_config)
    assert "numeric" in result
    assert "pearson" in result["numeric"]
    assert isinstance(result["high_correlation_pairs"], list)


def test_target_leakage_detection(default_config):
    n = 500
    df = pd.DataFrame({"feature": range(n)})
    df["leaky_target"] = df["feature"] * 2  # perfectly correlated -> leakage
    default_config.target_column = "leaky_target"
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    result = compute_correlations(df, profiles, default_config)
    assert len(result["target_leakage"]) >= 1
    assert result["target_leakage"][0]["column"] == "feature"
