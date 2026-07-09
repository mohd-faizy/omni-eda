"""Central configuration object used across every ``omni_eda`` module.

Keeping every tunable knob in a single dataclass makes the rest of the
package easy to test (pass a custom :class:`EDAConfig` instance) and easy
to expose through the CLI (each field maps to a flag).
"""

from __future__ import annotations

import multiprocessing as _mp
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ThemeName = Literal["light", "dark", "minimal", "corporate"]
OutlierMethod = Literal["zscore", "modified_zscore", "iqr", "isolation_forest", "lof", "elliptic_envelope"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]


@dataclass
class EDAConfig:
    """Configuration for an :class:`~omni_eda.analyzer.OmniEDA` run.

    Every attribute has a sane default so ``EDAConfig()`` works out of the
    box on small-to-medium datasets, while still being fully overridable
    for large or unusual datasets.
    """

    # ---- General ---------------------------------------------------
    title: str = "Automated EDA Report"
    output_dir: str = "omni_eda_output"
    random_state: int = 42
    verbose: bool = True
    n_jobs: int = field(default_factory=lambda: max(1, (_mp.cpu_count() or 2) - 1))

    # ---- Column handling --------------------------------------------
    target_column: str | None = None
    ignore_columns: Sequence[str] = field(default_factory=list)
    id_like_uniqueness_ratio: float = 0.98
    high_cardinality_threshold: int = 50
    high_cardinality_ratio: float = 0.5
    low_variance_threshold: float = 1e-8
    constant_check: bool = True
    max_categories_for_plot: int = 20
    rare_category_threshold: float = 0.01

    # ---- Sampling / performance guards --------------------------------
    max_rows_full_scan: int = 2_000_000
    max_rows_for_expensive_ops: int = 50_000
    max_cols_for_pairwise: int = 40
    sample_for_plots: int = 20_000
    chunksize: int = 250_000

    # ---- Missing data ----------------------------------------------
    missing_warn_threshold: float = 0.2
    missing_drop_threshold: float = 0.6

    # ---- Outliers ----------------------------------------------------
    outlier_methods: list[OutlierMethod] = field(
        default_factory=lambda: ["iqr", "zscore", "modified_zscore", "isolation_forest", "lof"]
    )
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    enable_model_based_outliers: bool = True

    # ---- Correlation --------------------------------------------------
    correlation_methods: list[CorrelationMethod] = field(default_factory=lambda: ["pearson", "spearman"])
    high_correlation_threshold: float = 0.85
    leakage_correlation_threshold: float = 0.95

    # ---- Distribution ---------------------------------------------------
    skew_threshold: float = 1.0
    class_imbalance_threshold: float = 0.9

    # ---- Visualization --------------------------------------------------
    theme: ThemeName = "light"
    figure_format: Literal["png", "svg"] = "png"
    figure_dpi: int = 110
    max_plots_per_section: int = 60
    generate_pairplot: bool = True
    generate_pca: bool = True
    generate_correlation_network: bool = True

    # ---- Target / modelling ----------------------------------------------
    enable_target_modeling: bool = True
    test_size: float = 0.25

    # ---- Cleaning (opt-in, never runs unless explicitly requested) --------
    auto_clean: bool = False

    # ---- Report / export --------------------------------------------------
    export_formats: list[str] = field(default_factory=lambda: ["html"])
    embed_images_base64: bool = True

    def resolved_output_dir(self) -> Path:
        """Return the output directory as a :class:`~pathlib.Path`, creating it."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_ignored(self, column: str) -> bool:
        return column in set(self.ignore_columns)
