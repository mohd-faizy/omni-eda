from __future__ import annotations

from omni_eda.detection import ColumnTypeDetector
from omni_eda.quality import build_quality_report


def test_detects_duplicate_rows(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    categories = {i.category for i in report.issues}
    assert "duplicate_rows" in categories


def test_detects_duplicate_columns(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    categories = {i.category for i in report.issues}
    assert "duplicate_columns" in categories


def test_detects_constant_column(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    const_issues = [i for i in report.issues if i.category == "constant_column"]
    assert any(i.column == "const_col" for i in const_issues)


def test_detects_infinite_values(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    inf_issues = [i for i in report.issues if i.category == "infinite_values"]
    assert len(inf_issues) >= 1
    assert inf_issues[0].severity == "critical"


def test_detects_impossible_negatives(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    categories = {i.category for i in report.issues}
    assert "impossible_values" in categories


def test_clean_data_has_few_issues(basic_df, basic_profiles, default_config):
    report = build_quality_report(basic_df, basic_profiles, default_config)
    critical = [i for i in report.issues if i.severity == "critical"]
    assert critical == []  # well-formed fixture shouldn't trip critical checks


def test_summary_counts_match_issues(messy_df, default_config):
    profiles = ColumnTypeDetector(default_config).profile_dataframe(messy_df)
    report = build_quality_report(messy_df, profiles, default_config)
    assert report.summary["n_issues"] == len(report.issues)
    assert report.summary["n_critical"] + report.summary["n_warning"] + report.summary["n_info"] == len(report.issues)
