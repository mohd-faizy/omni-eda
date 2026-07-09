from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda.outliers import (
    detect_multivariate_outliers,
    detect_univariate_outliers,
    iqr_outliers,
    modified_zscore_outliers,
    zscore_outliers,
)


def _series_with_outliers():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 300)
    values = np.append(values, [50, -50, 45])  # obvious outliers
    return pd.Series(values)


def test_zscore_detects_obvious_outliers():
    s = _series_with_outliers()
    mask = zscore_outliers(s, threshold=3.0)
    assert mask.iloc[-3:].all()


def test_iqr_detects_obvious_outliers():
    s = _series_with_outliers()
    mask = iqr_outliers(s)
    assert mask.iloc[-3:].all()


def test_modified_zscore_detects_obvious_outliers():
    s = _series_with_outliers()
    mask = modified_zscore_outliers(s)
    assert mask.iloc[-3:].all()


def test_constant_series_no_outliers():
    s = pd.Series([5.0] * 50)
    assert not zscore_outliers(s).any()
    assert not iqr_outliers(s).any()


def test_detect_univariate_outliers_multi_column(default_config):
    df = pd.DataFrame({"a": _series_with_outliers(), "b": np.random.default_rng(1).normal(0, 1, 303)})
    result = detect_univariate_outliers(df, ["a", "b"], default_config.outlier_methods, default_config)
    assert "a" in result
    assert result["a"]["iqr"].n_outliers >= 3


def test_multivariate_outliers_isolation_forest(default_config):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
    df = pd.concat([df, pd.DataFrame({"a": [20], "b": [20]})], ignore_index=True)
    result = detect_multivariate_outliers(df, ["a", "b"], default_config)
    assert "isolation_forest" in result
    assert result["isolation_forest"].n_outliers >= 1
