from __future__ import annotations

import pandas as pd

from omni_eda.detection import ColumnTypeDetector


def test_detects_basic_types(basic_profiles):
    p = basic_profiles
    assert p["age"].is_numeric
    assert p["income"].is_numeric
    assert p["signup_date"].is_datetime
    assert p["is_active"].is_boolean
    assert p["category"].is_categorical
    assert p["const_col"].is_constant
    assert p["id"].is_id_like


def test_detects_semantic_types(basic_profiles):
    assert basic_profiles["email"].semantic_type == "email"
    assert basic_profiles["url"].semantic_type == "url"
    assert basic_profiles["price"].semantic_type == "currency"


def test_constant_column_flagged(basic_profiles):
    assert basic_profiles["const_col"].is_constant
    assert "constant" in basic_profiles["const_col"].flags


def test_all_missing_column(default_config):
    df = pd.DataFrame({"a": [1, 2, 3], "allnull": [None, None, None]})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    assert profiles["allnull"].base_type == "constant"
    assert profiles["allnull"].n_missing == 3


def test_high_cardinality_detection(default_config):
    df = pd.DataFrame({"cat": [f"val_{i}" for i in range(200)]})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    # near-100% unique short strings are treated as id-like, not high-cardinality categorical
    assert profiles["cat"].is_id_like or profiles["cat"].is_high_cardinality


def test_mixed_type_column(default_config):
    df = pd.DataFrame({"mixed": [1, "two", 3.0, "four"] * 25})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    assert profiles["mixed"].is_mixed_type


def test_boolean_numeric_detection(default_config):
    df = pd.DataFrame({"flag": [0, 1, 1, 0, 1] * 20})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    assert profiles["flag"].is_boolean
    assert profiles["flag"].is_binary_encoded


def test_coordinate_detection(default_config):
    df = pd.DataFrame({"lat": [12.34, 56.78, -12.1], "lon": [-45.6, 12.9, 33.3]})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    assert profiles["lat"].semantic_type == "latitude"
    assert profiles["lon"].semantic_type == "longitude"


def test_single_column_dataframe(default_config):
    df = pd.DataFrame({"only": range(50)})
    profiles = ColumnTypeDetector(default_config).profile_dataframe(df)
    assert "only" in profiles
    assert profiles["only"].is_numeric
