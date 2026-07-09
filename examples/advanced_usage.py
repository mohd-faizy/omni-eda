"""Advanced omni_eda usage: custom configuration, a classification target,
opt-in cleaning, and exporting to every supported format.

Run with:  python examples/advanced_usage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from omni_eda import OmniEDA, EDAConfig


def make_churn_dataframe(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(0, 72, n)
    monthly_charge = rng.normal(70, 25, n).clip(10, None)
    support_tickets = rng.poisson(1.5, n)

    # churn probability genuinely depends on the features, so target analysis
    # (feature importance / ROC AUC) has real signal to report on.
    logit = -0.03 * tenure_months + 0.02 * monthly_charge + 0.35 * support_tickets - 1.5
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge.round(2),
            "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2]),
            "support_tickets": support_tickets,
            "region": rng.choice(["north", "south", "east", "west"], n),
            "churned": churn,
        }
    )
    df.loc[rng.choice(n, size=int(n * 0.03), replace=False), "monthly_charge"] = np.nan
    return df


def main() -> None:
    df = make_churn_dataframe()

    config = EDAConfig(
        title="Customer Churn - Automated EDA",
        output_dir="examples_output/advanced",
        target_column="churned",
        theme="corporate",
        high_correlation_threshold=0.8,
        outlier_methods=["iqr", "zscore", "isolation_forest", "lof"],
        export_formats=["html", "markdown", "json", "excel", "csv", "dashboard"],
        ignore_columns=["customer_id"],
        n_jobs=4,
    )

    eda = OmniEDA(config=config)
    results = eda.run(df)

    print(f"Critical issues: {results['quality'].summary['n_critical']}")
    print(f"Warnings: {results['quality'].summary['n_warning']}")
    if results["target_analysis"] and results["target_analysis"].get("curves", {}).get("roc_auc"):
        print(f"Baseline ROC AUC on 'churned': {results['target_analysis']['curves']['roc_auc']:.3f}")

    # Optional, explicit, auditable cleaning -- never runs unless you ask for it.
    cleaned_df = eda.clean(steps=["dedup_rows", "convert_dtypes", "infinities"])
    print(f"Cleaned shape: {cleaned_df.shape} (was {df.shape})")

    written = eda.export()  # every format listed in config.export_formats
    for fmt, path in written.items():
        print(f"  [{fmt}] -> {path}")


if __name__ == "__main__":
    main()
