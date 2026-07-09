from __future__ import annotations

from omni_eda.detection import ColumnTypeDetector
from omni_eda.feature_engineering import suggest_features
from omni_eda.statistics import compute_all_statistics


def test_suggests_dropping_constant_column(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    stats = compute_all_statistics(messy_df, profiles, default_config)
    suggestions = suggest_features(messy_df, profiles, stats, default_config)
    drop_suggestions = [s for s in suggestions if s.action == "drop_column" and s.column == "const_col"]
    assert drop_suggestions


def test_suggests_dropping_id_column(basic_df, basic_profiles, default_config):
    stats = compute_all_statistics(basic_df, basic_profiles, default_config)
    suggestions = suggest_features(basic_df, basic_profiles, stats, default_config)
    id_suggestions = [s for s in suggestions if s.column == "id"]
    assert any(s.action == "drop_column" for s in id_suggestions)


def test_suggests_encoding_for_categorical(basic_df, basic_profiles, default_config):
    stats = compute_all_statistics(basic_df, basic_profiles, default_config)
    suggestions = suggest_features(basic_df, basic_profiles, stats, default_config)
    cat_suggestions = [s for s in suggestions if s.column == "category"]
    assert any("encode" in s.action for s in cat_suggestions)


def test_suggests_datetime_decomposition(basic_df, basic_profiles, default_config):
    stats = compute_all_statistics(basic_df, basic_profiles, default_config)
    suggestions = suggest_features(basic_df, basic_profiles, stats, default_config)
    dt_suggestions = [s for s in suggestions if s.column == "signup_date"]
    assert any(s.action == "datetime_decomposition" for s in dt_suggestions)
