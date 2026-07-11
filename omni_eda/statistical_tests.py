"""Automated statistical testing engine for omni_eda.

Automatically selects and runs appropriate statistical tests based on:
- Column data types (numeric vs. categorical)
- Number of groups (2 vs. 3+)
- Sample size (parametric vs. non-parametric selection)
- Distribution normality

Every test returns a structured result dict with: test name, statistic,
p-value, effect size, interpretation text, and significance flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger, stage
from omni_eda.utils import sample_df


@dataclass
class TestResult:
    """A single statistical test result."""

    test_name: str
    column_a: str
    column_b: str | None
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_size_label: str | None = None  # "small" | "medium" | "large"
    significant: bool = False
    interpretation: str = ""
    category: str = ""  # "normality" | "comparison" | "association" | "correlation" | "variance"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "column_a": self.column_a,
            "column_b": self.column_b,
            "statistic": round(self.statistic, 6) if self.statistic is not None else None,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None else None,
            "effect_size_label": self.effect_size_label,
            "significant": self.significant,
            "interpretation": self.interpretation,
            "category": self.category,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------
def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def _eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """Eta-squared effect size from ANOVA F-statistic."""
    if df_within == 0:
        return 0.0
    return float((f_stat * df_between) / (f_stat * df_between + df_within))


def _cohens_d_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _eta_squared_label(eta2: float) -> str:
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


def _cramers_v_from_chi2(chi2: float, n: int, min_dim: int) -> float:
    """Compute Cramér's V from chi-square statistic."""
    if n == 0 or min_dim <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min_dim - 1))))


def _cramers_v_label(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def _test_normality(series: pd.Series, col_name: str, alpha: float = 0.05) -> list[TestResult]:
    """Run normality tests on a numeric series."""
    results = []
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 8:
        return results

    values = s.to_numpy()

    # Shapiro-Wilk (best for n <= 5000)
    if len(values) <= 5000:
        try:
            stat, p = scipy_stats.shapiro(values[:5000])
            is_normal = p > alpha
            results.append(TestResult(
                test_name="Shapiro-Wilk",
                column_a=col_name,
                column_b=None,
                statistic=float(stat),
                p_value=float(p),
                significant=not is_normal,
                category="normality",
                interpretation=(
                    f"'{col_name}' follows a normal distribution (p={p:.4f}, fail to reject H₀)."
                    if is_normal else
                    f"'{col_name}' does NOT follow a normal distribution (p={p:.4g}, reject H₀)."
                ),
            ))
        except Exception:
            pass

    # Anderson-Darling
    try:
        result = scipy_stats.anderson(values, dist="norm")
        # Use the 5% significance level
        idx = list(result.significance_level).index(5.0) if 5.0 in result.significance_level else 2
        critical = result.critical_values[idx]
        is_normal = result.statistic < critical
        results.append(TestResult(
            test_name="Anderson-Darling",
            column_a=col_name,
            column_b=None,
            statistic=float(result.statistic),
            p_value=-1.0,  # AD doesn't give a simple p-value
            significant=not is_normal,
            category="normality",
            interpretation=(
                f"'{col_name}' is consistent with normality (AD={result.statistic:.4f} < critical={critical:.4f})."
                if is_normal else
                f"'{col_name}' departs from normality (AD={result.statistic:.4f} ≥ critical={critical:.4f})."
            ),
            detail={"critical_value_5pct": float(critical)},
        ))
    except Exception:
        pass

    # D'Agostino-Pearson (n >= 20)
    if len(values) >= 20:
        try:
            stat, p = scipy_stats.normaltest(values)
            is_normal = p > alpha
            results.append(TestResult(
                test_name="D'Agostino-Pearson",
                column_a=col_name,
                column_b=None,
                statistic=float(stat),
                p_value=float(p),
                significant=not is_normal,
                category="normality",
                interpretation=(
                    f"'{col_name}' passes D'Agostino-Pearson normality test (p={p:.4f})."
                    if is_normal else
                    f"'{col_name}' fails D'Agostino-Pearson normality test (p={p:.4g})."
                ),
            ))
        except Exception:
            pass

    return results


def _test_ttest(
    df: pd.DataFrame, numeric_col: str, binary_col: str, alpha: float = 0.05
) -> list[TestResult]:
    """Independent samples t-test for numeric column grouped by binary column."""
    results = []
    data = df[[numeric_col, binary_col]].dropna()
    data[numeric_col] = pd.to_numeric(data[numeric_col], errors="coerce")
    data = data.dropna()

    groups = data[binary_col].unique()
    if len(groups) != 2:
        return results

    g1 = data[data[binary_col] == groups[0]][numeric_col].to_numpy()
    g2 = data[data[binary_col] == groups[1]][numeric_col].to_numpy()

    if len(g1) < 3 or len(g2) < 3:
        return results

    # Levene's test for equal variances
    try:
        lev_stat, lev_p = scipy_stats.levene(g1, g2)
        equal_var = lev_p > alpha
        results.append(TestResult(
            test_name="Levene's Test",
            column_a=numeric_col,
            column_b=binary_col,
            statistic=float(lev_stat),
            p_value=float(lev_p),
            significant=not equal_var,
            category="variance",
            interpretation=(
                f"Variances of '{numeric_col}' are equal across '{binary_col}' groups (p={lev_p:.4f})."
                if equal_var else
                f"Variances of '{numeric_col}' differ across '{binary_col}' groups (p={lev_p:.4g}). Using Welch's t-test."
            ),
        ))
    except Exception:
        equal_var = True

    # T-test
    try:
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=equal_var)
        d = _cohens_d(g1, g2)
        d_label = _cohens_d_label(d)
        sig = p < alpha
        test_name = "Independent t-test" if equal_var else "Welch's t-test"
        results.append(TestResult(
            test_name=test_name,
            column_a=numeric_col,
            column_b=binary_col,
            statistic=float(stat),
            p_value=float(p),
            effect_size=abs(d),
            effect_size_label=d_label,
            significant=sig,
            category="comparison",
            interpretation=(
                f"Significant difference in '{numeric_col}' between '{groups[0]}' and '{groups[1]}' "
                f"(p={p:.4g}, Cohen's d={abs(d):.3f} [{d_label}])."
                if sig else
                f"No significant difference in '{numeric_col}' between '{groups[0]}' and '{groups[1]}' "
                f"(p={p:.4f}, Cohen's d={abs(d):.3f} [{d_label}])."
            ),
            detail={"group_1": str(groups[0]), "group_2": str(groups[1]),
                     "mean_1": float(g1.mean()), "mean_2": float(g2.mean()),
                     "n_1": len(g1), "n_2": len(g2)},
        ))
    except Exception:
        pass

    # Mann-Whitney U (non-parametric alternative)
    try:
        stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        sig = p < alpha
        # Rank-biserial correlation as effect size
        n1, n2 = len(g1), len(g2)
        r_rb = 1 - (2 * stat) / (n1 * n2) if n1 * n2 > 0 else 0.0
        results.append(TestResult(
            test_name="Mann-Whitney U",
            column_a=numeric_col,
            column_b=binary_col,
            statistic=float(stat),
            p_value=float(p),
            effect_size=abs(r_rb),
            effect_size_label=_cohens_d_label(abs(r_rb)),  # approximate
            significant=sig,
            category="comparison",
            interpretation=(
                f"Significant rank difference in '{numeric_col}' between groups of '{binary_col}' "
                f"(U={stat:.1f}, p={p:.4g})."
                if sig else
                f"No significant rank difference in '{numeric_col}' between groups of '{binary_col}' "
                f"(U={stat:.1f}, p={p:.4f})."
            ),
        ))
    except Exception:
        pass

    return results


def _test_anova(
    df: pd.DataFrame, numeric_col: str, categorical_col: str, alpha: float = 0.05
) -> list[TestResult]:
    """One-way ANOVA and Kruskal-Wallis for numeric column grouped by categorical column."""
    results = []
    data = df[[numeric_col, categorical_col]].dropna()
    data[numeric_col] = pd.to_numeric(data[numeric_col], errors="coerce")
    data = data.dropna()

    groups = [g[numeric_col].to_numpy() for _, g in data.groupby(categorical_col, observed=True)]
    groups = [g for g in groups if len(g) >= 2]
    n_groups = len(groups)

    if n_groups < 2:
        return results

    # One-way ANOVA
    try:
        f_stat, p = scipy_stats.f_oneway(*groups)
        df_between = n_groups - 1
        df_within = sum(len(g) for g in groups) - n_groups
        eta2 = _eta_squared(f_stat, df_between, df_within)
        eta2_label = _eta_squared_label(eta2)
        sig = p < alpha
        results.append(TestResult(
            test_name="One-way ANOVA",
            column_a=numeric_col,
            column_b=categorical_col,
            statistic=float(f_stat),
            p_value=float(p),
            effect_size=eta2,
            effect_size_label=eta2_label,
            significant=sig,
            category="comparison",
            interpretation=(
                f"Significant differences in '{numeric_col}' across {n_groups} groups of '{categorical_col}' "
                f"(F={f_stat:.3f}, p={p:.4g}, η²={eta2:.4f} [{eta2_label}])."
                if sig else
                f"No significant differences in '{numeric_col}' across {n_groups} groups of '{categorical_col}' "
                f"(F={f_stat:.3f}, p={p:.4f}, η²={eta2:.4f} [{eta2_label}])."
            ),
            detail={"n_groups": n_groups, "df_between": df_between, "df_within": df_within},
        ))
    except Exception:
        pass

    # Kruskal-Wallis (non-parametric alternative)
    try:
        stat, p = scipy_stats.kruskal(*groups)
        sig = p < alpha
        # Epsilon-squared as effect size
        n_total = sum(len(g) for g in groups)
        eps2 = (stat - n_groups + 1) / (n_total - n_groups) if n_total > n_groups else 0.0
        eps2 = max(0.0, eps2)
        results.append(TestResult(
            test_name="Kruskal-Wallis",
            column_a=numeric_col,
            column_b=categorical_col,
            statistic=float(stat),
            p_value=float(p),
            effect_size=eps2,
            effect_size_label=_eta_squared_label(eps2),
            significant=sig,
            category="comparison",
            interpretation=(
                f"Significant rank differences in '{numeric_col}' across groups of '{categorical_col}' "
                f"(H={stat:.3f}, p={p:.4g})."
                if sig else
                f"No significant rank differences in '{numeric_col}' across groups of '{categorical_col}' "
                f"(H={stat:.3f}, p={p:.4f})."
            ),
        ))
    except Exception:
        pass

    return results


def _test_chi_square(
    df: pd.DataFrame, cat_col1: str, cat_col2: str, alpha: float = 0.05, max_categories: int = 20
) -> list[TestResult]:
    """Chi-square test of independence for two categorical columns."""
    results = []
    data = df[[cat_col1, cat_col2]].dropna()
    if data.empty:
        return results

    # Limit to top categories
    for col in [cat_col1, cat_col2]:
        top = data[col].value_counts().head(max_categories).index
        data = data[data[col].isin(top)]

    if data.empty:
        return results

    try:
        contingency = pd.crosstab(data[cat_col1], data[cat_col2])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return results

        chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        v = _cramers_v_from_chi2(chi2, n, min(contingency.shape))
        v_label = _cramers_v_label(v)
        sig = p < alpha

        results.append(TestResult(
            test_name="Chi-square Test",
            column_a=cat_col1,
            column_b=cat_col2,
            statistic=float(chi2),
            p_value=float(p),
            effect_size=v,
            effect_size_label=v_label,
            significant=sig,
            category="association",
            interpretation=(
                f"Significant association between '{cat_col1}' and '{cat_col2}' "
                f"(χ²={chi2:.2f}, p={p:.4g}, Cramér's V={v:.3f} [{v_label}])."
                if sig else
                f"No significant association between '{cat_col1}' and '{cat_col2}' "
                f"(χ²={chi2:.2f}, p={p:.4f}, Cramér's V={v:.3f} [{v_label}])."
            ),
            detail={"dof": int(dof), "n": int(n)},
        ))
    except Exception:
        pass

    return results


def _test_correlation_significance(
    df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05
) -> list[TestResult]:
    """Test significance of Pearson and Spearman correlations."""
    results = []
    data = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5:
        return results

    x, y = data[col_a].to_numpy(), data[col_b].to_numpy()

    # Pearson
    try:
        r, p = scipy_stats.pearsonr(x, y)
        sig = p < alpha
        results.append(TestResult(
            test_name="Pearson Correlation",
            column_a=col_a,
            column_b=col_b,
            statistic=float(r),
            p_value=float(p),
            effect_size=abs(float(r)),
            effect_size_label=_cramers_v_label(abs(float(r))),  # same thresholds work
            significant=sig,
            category="correlation",
            interpretation=(
                f"Significant linear correlation between '{col_a}' and '{col_b}' "
                f"(r={r:.4f}, p={p:.4g})."
                if sig else
                f"No significant linear correlation between '{col_a}' and '{col_b}' "
                f"(r={r:.4f}, p={p:.4f})."
            ),
        ))
    except Exception:
        pass

    # Spearman
    try:
        rho, p = scipy_stats.spearmanr(x, y)
        sig = p < alpha
        results.append(TestResult(
            test_name="Spearman Correlation",
            column_a=col_a,
            column_b=col_b,
            statistic=float(rho),
            p_value=float(p),
            effect_size=abs(float(rho)),
            effect_size_label=_cramers_v_label(abs(float(rho))),
            significant=sig,
            category="correlation",
            interpretation=(
                f"Significant monotonic correlation between '{col_a}' and '{col_b}' "
                f"(ρ={rho:.4f}, p={p:.4g})."
                if sig else
                f"No significant monotonic correlation between '{col_a}' and '{col_b}' "
                f"(ρ={rho:.4f}, p={p:.4f})."
            ),
        ))
    except Exception:
        pass

    return results


def _bootstrap_confidence_interval(
    series: pd.Series, col_name: str, n_bootstrap: int = 1000, ci: float = 0.95, random_state: int = 42
) -> TestResult | None:
    """Bootstrap confidence interval for the mean."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return None

    rng = np.random.RandomState(random_state)
    values = s.to_numpy()
    n = len(values)

    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    sample_mean = float(values.mean())

    return TestResult(
        test_name=f"Bootstrap {int(ci*100)}% CI (Mean)",
        column_a=col_name,
        column_b=None,
        statistic=sample_mean,
        p_value=-1.0,  # Not applicable
        significant=False,
        category="confidence_interval",
        interpretation=(
            f"The mean of '{col_name}' is {sample_mean:.4f} "
            f"with {int(ci*100)}% CI [{ci_lower:.4f}, {ci_upper:.4f}]."
        ),
        detail={"ci_lower": ci_lower, "ci_upper": ci_upper, "n_bootstrap": n_bootstrap,
                "boot_std": float(boot_means.std())},
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_all_tests(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
) -> dict[str, Any]:
    """Run all applicable statistical tests and return structured results.

    Returns a dict with:
        - "results": list of TestResult dicts
        - "summary": high-level summary (counts by category, etc.)
        - "by_category": dict grouping results by category
    """
    cfg = config or EDAConfig()
    logger = get_logger()
    alpha = getattr(cfg, "significance_level", 0.05)
    max_test_cats = getattr(cfg, "max_test_categories", 20)
    n_bootstrap = getattr(cfg, "bootstrap_n_iterations", 1000)

    all_results: list[TestResult] = []

    # Work on a sample for expensive tests
    working = sample_df(df, cfg.max_rows_for_expensive_ops, cfg.random_state)

    numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant and not cfg.is_ignored(c)]
    categorical_cols = [
        c for c, p in profiles.items()
        if (p.is_categorical or p.is_boolean) and not p.is_constant and not cfg.is_ignored(c) and p.n_unique <= max_test_cats
    ]
    binary_cols = [c for c in categorical_cols if profiles[c].n_unique == 2]

    # 1. Normality tests for numeric columns
    for col in numeric_cols[:15]:  # limit to avoid excessive output
        try:
            all_results.extend(_test_normality(working[col], col, alpha))
        except Exception:
            pass

    # 2. Comparison tests: numeric × binary (t-test, Mann-Whitney)
    for num_col in numeric_cols[:10]:
        for bin_col in binary_cols[:5]:
            if num_col == bin_col:
                continue
            try:
                all_results.extend(_test_ttest(working, num_col, bin_col, alpha))
            except Exception:
                pass

    # 3. Comparison tests: numeric × multi-category (ANOVA, Kruskal-Wallis)
    multi_cat_cols = [c for c in categorical_cols if profiles[c].n_unique > 2]
    for num_col in numeric_cols[:8]:
        for cat_col in multi_cat_cols[:5]:
            if num_col == cat_col:
                continue
            try:
                all_results.extend(_test_anova(working, num_col, cat_col, alpha))
            except Exception:
                pass

    # 4. Association tests: categorical × categorical (Chi-square)
    tested_pairs = set()
    for i, cat1 in enumerate(categorical_cols[:10]):
        for cat2 in categorical_cols[i + 1:10]:
            pair = tuple(sorted([cat1, cat2]))
            if pair in tested_pairs:
                continue
            tested_pairs.add(pair)
            try:
                all_results.extend(_test_chi_square(working, cat1, cat2, alpha, max_test_cats))
            except Exception:
                pass

    # 5. Correlation significance: top numeric pairs
    from itertools import combinations
    for col_a, col_b in list(combinations(numeric_cols[:10], 2))[:15]:
        try:
            all_results.extend(_test_correlation_significance(working, col_a, col_b, alpha))
        except Exception:
            pass

    # 6. Confidence intervals for numeric columns
    for col in numeric_cols[:10]:
        try:
            result = _bootstrap_confidence_interval(working[col], col, n_bootstrap=n_bootstrap)
            if result:
                all_results.append(result)
        except Exception:
            pass

    # Build summary
    by_category: dict[str, list[dict]] = {}
    n_significant = 0
    for r in all_results:
        rd = r.to_dict()
        by_category.setdefault(r.category, []).append(rd)
        if r.significant:
            n_significant += 1

    summary = {
        "total_tests": len(all_results),
        "n_significant": n_significant,
        "n_not_significant": len(all_results) - n_significant,
        "categories": {cat: len(tests) for cat, tests in by_category.items()},
    }

    return {
        "results": [r.to_dict() for r in all_results],
        "summary": summary,
        "by_category": by_category,
    }
