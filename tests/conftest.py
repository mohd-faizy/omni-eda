"""Shared fixtures for the omni_eda test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnTypeDetector


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def basic_df(rng) -> pd.DataFrame:
    """A small, well-behaved mixed-type DataFrame."""
    n = 400
    df = pd.DataFrame(
        {
            "id": [f"ID{i:05d}" for i in range(n)],
            "age": rng.integers(18, 90, n),
            "income": rng.normal(50000, 15000, n),
            "email": [f"user{i}@example.com" for i in range(n)],
            "url": [f"https://example.com/p{i}" for i in range(n)],
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "is_active": rng.choice([0, 1], n),
            "category": rng.choice(["A", "B", "C"], n),
            "const_col": ["same"] * n,
            "price": [f"${x:.2f}" for x in rng.uniform(10, 100, n)],
            "text": ["Some fairly long free text field number " + str(i) for i in range(n)],
        }
    )
    return df


@pytest.fixture
def messy_df(rng) -> pd.DataFrame:
    """A DataFrame with deliberate quality problems for quality/cleaning tests."""
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "salary": rng.normal(50000, 15000, n),
            "category": rng.choice(["A", "B", "C"], n),
            "const_col": ["x"] * n,
            "sparse_col": [np.nan] * (n - 5) + [1, 2, 3, 4, 5],
        }
    )
    df.loc[0:4, "age"] = -1  # impossible negative
    df.loc[5, "salary"] = np.inf
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
    df["salary_copy"] = df["salary"]  # duplicate column
    df.loc[10:15, "category"] = "  b  "  # whitespace issues
    return df


@pytest.fixture
def target_df(rng) -> pd.DataFrame:
    """A DataFrame with a binary classification target that has real signal."""
    n = 600
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "income": rng.normal(50000, 15000, n),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )
    score = (df["age"] - 50) * 0.02 + (df["income"] - 50000) / 15000 * 0.6 + rng.normal(0, 1, n)
    df["target"] = (score > np.median(score)).astype(int)
    return df


@pytest.fixture
def timeseries_df(rng) -> pd.DataFrame:
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    values = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame({"date": dates, "value": values, "aux": rng.normal(0, 1, n)})


@pytest.fixture
def default_config() -> EDAConfig:
    return EDAConfig(verbose=False, sample_for_plots=500, max_rows_for_expensive_ops=2000)


@pytest.fixture
def basic_profiles(basic_df, default_config):
    return ColumnTypeDetector(default_config).profile_dataframe(basic_df)
