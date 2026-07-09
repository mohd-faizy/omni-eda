from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda.statistics import (
    boolean_stats,
    categorical_stats,
    compute_all_statistics,
    datetime_stats,
    numeric_stats,
    text_stats,
)


def test_numeric_stats_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = numeric_stats(s)
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["missing"] == 0


def test_numeric_stats_handles_infinities():
    s = pd.Series([1.0, 2.0, np.inf, 4.0])
    stats = numeric_stats(s)
    assert stats["n_infinite"] == 1
    assert np.isfinite(stats["mean"])  # infinities must not poison the mean


def test_numeric_stats_empty():
    s = pd.Series([np.nan, np.nan], dtype=float)
    stats = numeric_stats(s)
    assert stats["count"] == 0
    assert stats["missing_pct"] == 100.0


def test_categorical_stats():
    s = pd.Series(["a", "a", "a", "b", "c"])
    stats = categorical_stats(s)
    assert stats["n_unique"] == 3
    assert stats["top_value"] == "a"
    assert stats["top_value_pct"] == 60.0


def test_text_stats():
    s = pd.Series(["hello world", "foo bar baz", ""])
    stats = text_stats(s)
    assert stats["count"] == 3
    assert stats["n_empty_strings"] == 1
    assert stats["avg_length"] > 0


def test_datetime_stats():
    s = pd.Series(pd.date_range("2020-01-01", periods=10, freq="D"))
    stats = datetime_stats(s)
    assert stats["count"] == 10
    assert stats["range_days"] == 9


def test_boolean_stats():
    s = pd.Series([True, False, True, True])
    stats = boolean_stats(s)
    assert stats["true_pct"] == 75.0


def test_compute_all_statistics_covers_every_column(basic_df, basic_profiles, default_config):
    stats = compute_all_statistics(basic_df, basic_profiles, default_config)
    for col in basic_df.columns:
        assert col in stats
        assert "type" in stats[col]
