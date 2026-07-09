"""Target-aware analysis: run only when the user supplies a target column."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from omni_eda.config import EDAConfig
from omni_eda.correlation import cramers_v
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger
from omni_eda.utils import sample_df
from omni_eda.visualization import _new_fig, fig_to_base64


def _is_classification(target_profile: ColumnProfile, series: pd.Series) -> bool:
    return bool(target_profile.is_categorical or target_profile.is_boolean or series.nunique(dropna=True) <= 20)


def class_imbalance_report(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts()
    proportions = series.value_counts(normalize=True)
    return {
        "n_classes": int(counts.shape[0]),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "proportions": {str(k): float(v) for k, v in proportions.items()},
        "majority_class": str(proportions.index[0]) if not proportions.empty else None,
        "majority_pct": float(proportions.iloc[0] * 100) if not proportions.empty else 0.0,
        "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]) if len(counts) > 1 and counts.iloc[-1] > 0 else None,
    }


def anova_numeric_vs_categorical(df: pd.DataFrame, numeric_col: str, categorical_col: str) -> dict[str, Any] | None:
    groups = [g.dropna().to_numpy() for _, g in df.groupby(categorical_col, observed=True)[numeric_col] if g.notna().sum() > 1]
    groups = [pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy() for g in groups]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    try:
        f_stat, p_value = scipy_stats.f_oneway(*groups)
    except Exception:
        return None
    return {"f_statistic": float(f_stat), "p_value": float(p_value), "n_groups": len(groups)}


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any] | None:
    contingency = pd.crosstab(df[col_a], df[col_b])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    try:
        chi2, p_value, dof, _ = scipy_stats.chi2_contingency(contingency)
    except Exception:
        return None
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)}


def feature_importance(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    target: str,
    config: EDAConfig | None = None,
) -> pd.DataFrame | None:
    """Random-forest based importance plus mutual information, on a light preprocessing pass."""
    cfg = config or EDAConfig()
    logger = get_logger()
    target_profile = profiles[target]
    working = sample_df(df, cfg.max_rows_for_expensive_ops, cfg.random_state).copy()
    is_classif = _is_classification(target_profile, working[target])

    feature_cols = [
        c
        for c in profiles
        if c != target and not cfg.is_ignored(c) and not profiles[c].is_constant and not profiles[c].is_id_like
    ]
    if not feature_cols:
        return None

    X = pd.DataFrame(index=working.index)
    for col in feature_cols:
        profile = profiles[col]
        if profile.is_numeric:
            X[col] = pd.to_numeric(working[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
        elif profile.is_datetime:
            s = pd.to_datetime(working[col], errors="coerce", format="mixed")
            X[col] = (s - s.min()).dt.days
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = working[col].astype("category").cat.codes

    y_raw = working[target]
    if is_classif:
        y = y_raw.astype("category").cat.codes
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    valid_mask = y.notna() & X.notna().all(axis=1)
    X, y = X[valid_mask], y[valid_mask]
    if len(X) < 20 or X.empty:
        return None

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        # Mutual information's kNN-based estimator is roughly O(n^2) in effect,
        # so it gets its own (smaller) sample independent of the RF's.
        mi_sample_size = min(len(X), 5000)
        mi_idx = X.sample(n=mi_sample_size, random_state=cfg.random_state).index if mi_sample_size < len(X) else X.index

        if is_classif:
            model = RandomForestClassifier(n_estimators=100, random_state=cfg.random_state, n_jobs=cfg.n_jobs, max_depth=12)
            mi = mutual_info_classif(X.loc[mi_idx], y.loc[mi_idx], random_state=cfg.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=cfg.random_state, n_jobs=cfg.n_jobs, max_depth=12)
            mi = mutual_info_regression(X.loc[mi_idx], y.loc[mi_idx], random_state=cfg.random_state)
        model.fit(X, y)
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance_rf": model.feature_importances_, "mutual_information": mi}
        ).sort_values("importance_rf", ascending=False)
        return importance_df.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature importance computation skipped: %s", exc)
        return None


def classification_curves(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    target: str,
    config: EDAConfig | None = None,
) -> dict[str, Any]:
    """Fit a lightweight baseline classifier and produce ROC / PR / lift curves + report."""
    cfg = config or EDAConfig()
    logger = get_logger()
    result: dict[str, Any] = {}
    if not cfg.enable_target_modeling:
        return result

    target_profile = profiles[target]
    if not _is_classification(target_profile, df[target]):
        return result

    n_classes = df[target].nunique(dropna=True)
    if n_classes != 2:
        return result  # ROC/PR/Lift kept to the well-defined binary case

    working = sample_df(df, cfg.max_rows_for_expensive_ops, cfg.random_state).copy()
    feature_cols = [
        c
        for c in profiles
        if c != target and not cfg.is_ignored(c) and not profiles[c].is_constant and not profiles[c].is_id_like
    ]
    if not feature_cols:
        return result

    X = pd.DataFrame(index=working.index)
    for col in feature_cols:
        profile = profiles[col]
        if profile.is_numeric:
            X[col] = pd.to_numeric(working[col], errors="coerce")
        elif profile.is_datetime:
            s = pd.to_datetime(working[col], errors="coerce", format="mixed")
            X[col] = (s - s.min()).dt.days
        else:
            X[col] = working[col].astype("category").cat.codes
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    y = working[target].astype("category").cat.codes
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid], y[valid]
    if len(X) < 40 or y.nunique() != 2:
        return result

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            PrecisionRecallDisplay,
            RocCurveDisplay,
            classification_report,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if y.nunique() == 2 else None
        )
        model = RandomForestClassifier(n_estimators=100, random_state=cfg.random_state, n_jobs=cfg.n_jobs, max_depth=12)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        result["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        result["roc_auc"] = float(roc_auc_score(y_test, y_proba))

        fig, ax = _new_fig(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("ROC curve (held-out baseline RandomForest)")
        result["roc_curve"] = fig_to_base64(fig, dpi=cfg.figure_dpi)

        fig, ax = _new_fig(figsize=(5, 5))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("Precision-Recall curve")
        result["pr_curve"] = fig_to_base64(fig, dpi=cfg.figure_dpi)

        result["lift_chart"] = _lift_chart(y_test.to_numpy(), y_proba, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classification curve generation skipped: %s", exc)

    return result


def _lift_chart(y_true: np.ndarray, y_proba: np.ndarray, config: EDAConfig, n_bins: int = 10) -> str:
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    bin_size = max(1, len(y_sorted) // n_bins)
    overall_rate = y_true.mean() if len(y_true) else 0.0

    lifts = []
    for i in range(n_bins):
        chunk = y_sorted[i * bin_size : (i + 1) * bin_size]
        if len(chunk) == 0:
            continue
        chunk_rate = chunk.mean()
        lifts.append(chunk_rate / overall_rate if overall_rate else 0.0)

    fig, ax = _new_fig()
    ax.bar(range(1, len(lifts) + 1), lifts, color="#4C72B0")
    ax.axhline(1.0, color="red", linestyle="--", label="baseline")
    ax.set_xlabel("Decile (1 = highest predicted probability)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift chart")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig, dpi=config.figure_dpi)


def analyze_target(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    config: EDAConfig | None = None,
) -> dict[str, Any]:
    """Entry point: run every target-aware analysis available for ``config.target_column``."""
    cfg = config or EDAConfig()
    if not cfg.target_column or cfg.target_column not in profiles:
        return {}

    target = cfg.target_column
    target_profile = profiles[target]
    result: dict[str, Any] = {"target": target, "is_classification": _is_classification(target_profile, df[target])}

    if result["is_classification"]:
        result["class_imbalance"] = class_imbalance_report(df[target])

    associations = []
    for col, profile in profiles.items():
        if col == target or cfg.is_ignored(col) or profile.is_constant:
            continue
        if result["is_classification"]:
            if profile.is_numeric:
                test = anova_numeric_vs_categorical(df, col, target)
                if test:
                    associations.append({"column": col, "test": "anova", **test})
            elif profile.is_categorical or profile.is_boolean:
                test = chi_square_test(df, col, target)
                v = cramers_v(df[col], df[target])
                if test:
                    associations.append({"column": col, "test": "chi_square", "cramers_v": v, **test})
        else:
            if profile.is_numeric:
                value = df[col].astype(float).corr(df[target].astype(float))
                if pd.notna(value):
                    associations.append({"column": col, "test": "pearson_corr", "value": float(value)})
    result["feature_associations"] = sorted(associations, key=lambda d: d.get("p_value", 1.0))

    importance_df = feature_importance(df, profiles, target, cfg)
    if importance_df is not None:
        result["feature_importance"] = importance_df

    if result["is_classification"]:
        result["curves"] = classification_curves(df, profiles, target, cfg)

    return result
