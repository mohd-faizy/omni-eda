"""Rule-based feature-engineering suggestions.

Nothing here mutates the DataFrame -- it inspects column profiles and
statistics and produces human-readable suggestions (with a machine-readable
``action`` tag) that a user (or a downstream pipeline) can choose to apply,
several of which are directly implemented in :mod:`omni_eda.cleaning`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.correlation import find_highly_correlated_pairs
from omni_eda.detection import ColumnProfile


@dataclass
class Suggestion:
    column: str | None
    action: str
    rationale: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "action": self.action, "rationale": self.rationale, "detail": self.detail}


def suggest_features(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    statistics: dict[str, dict[str, Any]],
    config: EDAConfig | None = None,
    correlations: dict[str, Any] | None = None,
) -> list[Suggestion]:
    cfg = config or EDAConfig()
    suggestions: list[Suggestion] = []

    for col, profile in profiles.items():
        if cfg.is_ignored(col):
            continue

        if profile.is_constant:
            suggestions.append(Suggestion(col, "drop_column", "Column has a single unique value and carries no signal."))
            continue

        if profile.is_id_like:
            suggestions.append(Suggestion(col, "drop_column", "Column looks like a unique identifier (near-100% unique values)."))
            continue

        if profile.is_high_cardinality and profile.is_categorical:
            suggestions.append(
                Suggestion(
                    col,
                    "merge_rare_categories",
                    f"'{col}' has {profile.n_unique} categories; consider grouping rare ones into 'Other' before encoding.",
                    detail={"n_unique": profile.n_unique},
                )
            )

        if profile.is_categorical and not profile.is_high_cardinality:
            action = "one_hot_encode" if profile.n_unique <= 15 else "label_encode"
            suggestions.append(
                Suggestion(col, action, f"'{col}' is a low/medium-cardinality categorical column ({profile.n_unique} levels).")
            )

        if profile.is_numeric:
            col_stats = statistics.get(col, {})
            skew = col_stats.get("skewness")
            if skew is not None and abs(skew) >= cfg.skew_threshold:
                can_log = col_stats.get("min", 0) is not None and col_stats.get("min", 0) >= 0
                suggestions.append(
                    Suggestion(
                        col,
                        "log_transform" if can_log else "power_transform",
                        f"'{col}' is skewed (skewness={skew:.2f}); a {'log' if can_log else 'Yeo-Johnson/Box-Cox'} transform may help.",
                        detail={"skewness": skew},
                    )
                )
            if profile.n_unique > 20 and not profile.is_id_like:
                suggestions.append(
                    Suggestion(
                        col, "binning", f"'{col}' is continuous; consider binning for interpretability or tree-model splits."
                    )
                )
            suggestions.append(
                Suggestion(
                    col,
                    "scale",
                    f"'{col}' is numeric; consider standard/min-max scaling for distance-based or gradient-based models.",
                )
            )

        if profile.is_datetime:
            suggestions.append(
                Suggestion(
                    col,
                    "datetime_decomposition",
                    f"Decompose '{col}' into year/month/day/day-of-week/is_weekend features.",
                )
            )

        if profile.missing_pct > 0:
            if profile.missing_pct >= cfg.missing_drop_threshold * 100:
                suggestions.append(
                    Suggestion(col, "drop_column", f"'{col}' is {profile.missing_pct:.1f}% missing; likely not salvageable.")
                )
            else:
                strategy = "median/mode imputation" if not profile.is_text else "flag-and-fill ('missing' category)"
                suggestions.append(
                    Suggestion(
                        col,
                        "impute_missing",
                        f"'{col}' is {profile.missing_pct:.1f}% missing; consider {strategy} plus a missing-indicator column.",
                        detail={"missing_pct": profile.missing_pct},
                    )
                )

        if profile.semantic_type in ("latitude", "longitude"):
            suggestions.append(
                Suggestion(
                    col,
                    "geo_feature",
                    f"'{col}' looks like a coordinate; consider deriving distance-to-center or clustering geo features.",
                )
            )

    # Interaction / polynomial suggestions from strongly correlated (but not redundant) numeric pairs
    if correlations and correlations.get("numeric"):
        primary = cfg.correlation_methods[0] if cfg.correlation_methods else "pearson"
        matrix = correlations["numeric"].get(primary)
        if matrix is not None and not matrix.empty:
            moderate_pairs = [
                p for p in find_highly_correlated_pairs(matrix, 0.3) if abs(p["value"]) < cfg.high_correlation_threshold
            ][:5]
            for pair in moderate_pairs:
                suggestions.append(
                    Suggestion(
                        None,
                        "interaction_feature",
                        f"'{pair['col_a']}' and '{pair['col_b']}' are moderately related (r={pair['value']:.2f}); an interaction term ({pair['col_a']} * {pair['col_b']}) might capture joint effects.",
                        detail=pair,
                    )
                )
            strong_pairs = find_highly_correlated_pairs(matrix, cfg.high_correlation_threshold)
            for pair in strong_pairs[:5]:
                suggestions.append(
                    Suggestion(
                        None,
                        "drop_redundant_feature",
                        f"'{pair['col_a']}' and '{pair['col_b']}' are highly correlated (r={pair['value']:.2f}); consider dropping one to reduce multicollinearity.",
                        detail=pair,
                    )
                )

    numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant]
    if 2 <= len(numeric_cols) <= 6:
        suggestions.append(
            Suggestion(
                None,
                "polynomial_features",
                f"With {len(numeric_cols)} numeric columns, degree-2 polynomial features are cheap to generate and may help linear models capture curvature.",
            )
        )

    return suggestions


def suggestions_to_frame(suggestions: list[Suggestion]) -> pd.DataFrame:
    if not suggestions:
        return pd.DataFrame(columns=["column", "action", "rationale"])
    return pd.DataFrame([s.to_dict() for s in suggestions])
