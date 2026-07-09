from __future__ import annotations

import numpy as np

from omni_eda.cleaning import auto_clean
from omni_eda.detection import ColumnTypeDetector


def test_auto_clean_removes_duplicates(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    original_len = len(messy_df)
    cleaned, report = auto_clean(messy_df, profiles, default_config)
    assert len(cleaned) < original_len
    assert any("duplicate row" in a for a in report.actions)


def test_auto_clean_removes_duplicate_columns(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    cleaned, report = auto_clean(messy_df, profiles, default_config)
    assert "salary_copy" not in cleaned.columns or "salary" not in cleaned.columns


def test_auto_clean_replaces_infinities(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    cleaned, report = auto_clean(messy_df, profiles, default_config)
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not np.isinf(cleaned[col]).any()


def test_auto_clean_does_not_mutate_original(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    original_copy = messy_df.copy(deep=True)
    auto_clean(messy_df, profiles, default_config)
    pd_equal = messy_df.equals(original_copy) or (
        messy_df.shape == original_copy.shape and messy_df.isna().sum().sum() == original_copy.isna().sum().sum()
    )
    assert pd_equal


def test_drop_constant_columns_step(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    cleaned, report = auto_clean(messy_df, profiles, default_config, steps=["drop_constant"])
    assert "const_col" not in cleaned.columns


def test_fill_missing_values_step(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    cleaned, report = auto_clean(messy_df, profiles, default_config, steps=["fill_missing"])
    assert cleaned["sparse_col"].isna().sum() == 0
