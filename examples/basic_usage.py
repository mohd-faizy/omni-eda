"""Basic omni_eda usage: analyze a DataFrame and generate an HTML report.

Run with:  python examples/basic_usage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda import OmniEDA


def make_sample_dataframe(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "age": rng.integers(18, 85, n),
            "annual_income": rng.normal(60_000, 22_000, n).round(2),
            "signup_date": pd.date_range("2019-01-01", periods=n, freq="6h"),
            "country": rng.choice(["US", "UK", "DE", "FR", "IN"], n, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
            "email": [f"user{i}@example.com" for i in range(n)],
            "is_premium": rng.choice([True, False], n, p=[0.3, 0.7]),
            "satisfaction_score": rng.integers(1, 6, n),
        }
    )
    # sprinkle in some realistic messiness
    df.loc[rng.choice(n, size=int(n * 0.05), replace=False), "annual_income"] = np.nan
    df.loc[rng.choice(n, size=5, replace=False), "age"] = -1
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # a couple of duplicate rows
    return df


def main() -> None:
    df = make_sample_dataframe()

    # One line: analyze + write report.
    eda = OmniEDA()
    eda.run(df)
    eda.summary()  # quick console overview
    eda.generate_report("examples_output/basic_report.html")

    print("\nReport written to examples_output/basic_report.html")


if __name__ == "__main__":
    main()
