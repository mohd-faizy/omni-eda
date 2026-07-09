from __future__ import annotations

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnTypeDetector
from omni_eda.target_analysis import analyze_target, class_imbalance_report


def test_class_imbalance_report():
    import pandas as pd

    s = pd.Series(["a"] * 90 + ["b"] * 10)
    report = class_imbalance_report(s)
    assert report["majority_class"] == "a"
    assert report["majority_pct"] == 90.0


def test_analyze_target_classification(target_df):
    cfg = EDAConfig(verbose=False, target_column="target", max_rows_for_expensive_ops=1000)
    profiles = ColumnTypeDetector(cfg).profile_dataframe(target_df)
    result = analyze_target(target_df, profiles, cfg)
    assert result["is_classification"] is True
    assert "class_imbalance" in result
    assert "feature_associations" in result
    assert result["feature_importance"] is not None


def test_analyze_target_returns_empty_without_target(target_df):
    cfg = EDAConfig(verbose=False, target_column=None)
    profiles = ColumnTypeDetector(cfg).profile_dataframe(target_df)
    result = analyze_target(target_df, profiles, cfg)
    assert result == {}


def test_analyze_target_regression():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] * 2 + df["x2"] * -1 + rng.normal(0, 0.1, n)
    cfg = EDAConfig(verbose=False, target_column="y", max_rows_for_expensive_ops=1000)
    profiles = ColumnTypeDetector(cfg).profile_dataframe(df)
    result = analyze_target(df, profiles, cfg)
    assert result["is_classification"] is False
    assert result["feature_importance"] is not None
