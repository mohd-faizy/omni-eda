"""The main ``OmniEDA`` orchestrator -- the package's primary public entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd

from omni_eda import cleaning, correlation, feature_engineering, loaders, missing, outliers, quality, statistics, target_analysis, insights, ab_testing, clustering, drift, timeseries, statistical_tests
from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnTypeDetector
from omni_eda.export import export_all
from omni_eda.logger import get_logger, set_verbosity, stage
from omni_eda.report import ReportBuilder, build_console_summary
from omni_eda.utils import sample_df
from omni_eda.visualization import PlotEngine

__version__ = "0.2.2"

DataSource = Union[str, Path, pd.DataFrame]


class OmniEDA:
    """Fully automated exploratory data analysis on any pandas DataFrame.

    Examples
    --------
    >>> eda = OmniEDA()
    >>> results = eda.run(df)
    >>> eda.generate_report("report.html")

    or, in one line::

        OmniEDA("data.csv").generate_report("report.html")
    """

    def __init__(self, source: DataSource | None = None, config: EDAConfig | None = None, **config_overrides: Any):
        self.config = config or EDAConfig(**config_overrides)
        set_verbosity(self.config.verbose)
        self.logger = get_logger(verbose=self.config.verbose)
        self.df: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}
        self._source = source
        if source is not None:
            self.df = self._load(source)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self, source: DataSource) -> pd.DataFrame:
        loaded = loaders.load_data(source, chunksize=None)
        if not isinstance(loaded, pd.DataFrame):
            # e.g. a directory of files -> take the largest table by row count, but warn.
            if isinstance(loaded, dict):
                if not loaded:
                    raise ValueError("No supported data files were found at the given source.")
                name, frame = max(loaded.items(), key=lambda kv: len(kv[1]))
                self.logger.warning(
                    "Source is a directory with %d file(s); using the largest table ('%s'). "
                    "Call omni_eda.loaders.load_directory() directly to analyze all of them.",
                    len(loaded),
                    name,
                )
                loaded = frame
            else:
                raise TypeError(f"Unsupported data returned by loader: {type(loaded)}")
        return loaded

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame | None = None, target: str | None = None) -> dict[str, Any]:
        """Run the full EDA pipeline and return the raw results dict.

        Every stage is wrapped so a failure in one (e.g. an exotic dtype
        breaking a single plot) is logged and skipped rather than aborting
        the whole run.
        """
        if df is not None:
            self.df = df
        if self.df is None:
            raise ValueError("No data to analyze. Pass a DataFrame to run(), or a source to OmniEDA(source=...).")
        if target:
            self.config.target_column = target

        df = self._prepare(self.df)
        cfg = self.config
        results: dict[str, Any] = {
            "shape": df.shape,
            "dataframe_sample": sample_df(df, min(cfg.sample_for_plots, 5000), cfg.random_state),
        }

        self.logger.info("Starting OmniEDA run on %d rows x %d columns.", *df.shape)

        # ---- 1. Detection -------------------------------------------------
        detector = ColumnTypeDetector(cfg)
        profiles = {}
        with stage(self.logger, "Detecting column types"):
            profiles = detector.profile_dataframe(df)
        results["profiles"] = profiles

        # Save data samples for the report overview
        results["df_head"] = df.head(5).copy()
        results["df_tail"] = df.tail(5).copy()
        results["df_columns"] = list(df.columns)

        numeric_cols = [c for c, p in profiles.items() if p.is_numeric and not p.is_constant]

        # ---- 2. Statistics --------------------------------------------------
        results["statistics"] = {}
        with stage(self.logger, "Computing descriptive statistics"):
            results["statistics"] = statistics.compute_all_statistics(df, profiles, cfg)

        # ---- 2.5 Dataset summary & summary tables --------------------------
        with stage(self.logger, "Building dataset summary tables"):
            results["dataset_summary"] = statistics.compute_dataset_summary(df, profiles, results["statistics"])
            results["numeric_summary_table"] = statistics.build_numeric_summary_table(profiles, results["statistics"])
            results["categorical_summary_table"] = statistics.build_categorical_summary_table(profiles, results["statistics"])
            results["quality_scorecard"] = statistics.build_quality_scorecard(df, profiles, results["statistics"])

        # ---- 3. Missing values ------------------------------------------------
        results["missing"] = {}
        with stage(self.logger, "Analyzing missing values"):
            results["missing"] = missing.compute_missing_analysis(df, cfg)

        # ---- 4. Correlation ----------------------------------------------------
        results["correlation"] = {}
        with stage(self.logger, "Computing correlations"):
            results["correlation"] = correlation.compute_correlations(df, profiles, cfg)

        # ---- 5. Outliers -----------------------------------------------------
        results["outliers_summary"] = pd.DataFrame()
        with stage(self.logger, "Detecting outliers"):
            uni_outliers = outliers.detect_univariate_outliers(df, numeric_cols, cfg.outlier_methods, cfg)
            multi_outliers = outliers.detect_multivariate_outliers(df, numeric_cols, cfg)
            results["outliers_summary"] = outliers.summarize_outliers(uni_outliers, multi_outliers)
            results["outliers_detail"] = {"univariate": uni_outliers, "multivariate": multi_outliers}
            # v0.2: Explaining multivariate outliers
            results["outlier_explanations"] = outliers.explain_outliers(df, numeric_cols, multi_outliers, cfg)

        # ---- 5.5 Clustering Tendency ------------------------------------------
        with stage(self.logger, "Evaluating clustering tendency (Hopkins Statistic)"):
            results["hopkins_statistic"] = clustering.compute_hopkins_statistic(df, profiles, cfg)

        # ---- 6. Data quality report -------------------------------------------
        with stage(self.logger, "Building data quality report"):
            results["quality"] = quality.build_quality_report(
                df,
                profiles,
                cfg,
                correlation_findings=results["correlation"].get("high_correlation_pairs"),
                target_leakage_findings=results["correlation"].get("target_leakage"),
            )
            
        with stage(self.logger, "Computing dataset health score and memory analysis"):
            results["health_score"] = quality.compute_health_score(df, profiles, results["quality"])
            results["memory_analysis"] = quality.compute_memory_analysis(df, profiles)

        # ---- 6.5 A/B Testing & Time Series Anomalies -------------------------
        results["ab_testing"] = None
        if cfg.treatment_column:
            with stage(self.logger, f"Running A/B tests on treatment column '{cfg.treatment_column}'"):
                results["ab_testing"] = ab_testing.run_ab_test(df, profiles, cfg)
                
        with stage(self.logger, "Detecting time series change points"):
            results["timeseries_changepoints"] = timeseries.detect_changepoints(df, profiles, cfg)

        # ---- 6.7 Statistical tests -------------------------------------------
        results["statistical_tests"] = None
        if getattr(cfg, 'enable_statistical_tests', True):
            with stage(self.logger, "Running automated statistical tests"):
                results["statistical_tests"] = statistical_tests.run_all_tests(df, profiles, cfg)

        # ---- 7. Visualizations --------------------------------------------------
        engine = PlotEngine(cfg)
        with stage(self.logger, "Generating univariate plots"):
            results["univariate_plots"] = engine.univariate_plots(df, profiles)
        with stage(self.logger, "Generating missing-value plots"):
            results["missing_plots"] = engine.missing_plots(df)
        with stage(self.logger, "Generating bivariate plots"):
            results["bivariate_plots"] = engine.bivariate_plots(df, profiles, results["correlation"])
        with stage(self.logger, "Generating multivariate plots"):
            results["multivariate_plots"] = engine.multivariate_plots(
                df, profiles, results["correlation"], target=cfg.target_column
            )
        with stage(self.logger, "Generating time series plots"):
            results["timeseries_plots"] = engine.timeseries_plots(df, profiles)

        # ---- 7.5 Additional result-dependent plots ---------------------------
        with stage(self.logger, "Generating additional analysis plots"):
            results["additional_plots"] = engine.additional_plots(df, profiles, results)

        # ---- 8. Feature engineering suggestions & Insights -----------------------
        with stage(self.logger, "Generating feature engineering suggestions"):
            results["suggestions"] = feature_engineering.suggest_features(
                df, profiles, results["statistics"], cfg, results["correlation"]
            )
        with stage(self.logger, "Generating automated insights"):
            results["insights"] = insights.generate_insights(
                df, profiles, results["statistics"], results["quality"], 
                health_score=results.get("health_score"),
                missing_analysis=results.get("missing"),
                correlation=results.get("correlation"),
                config=cfg,
                statistical_tests=results.get("statistical_tests"),
                outliers_summary=results.get("outliers_summary"),
            )

        # ---- 9. Target analysis (optional) -------------------------------------
        results["target_analysis"] = None
        if cfg.target_column:
            with stage(self.logger, f"Analyzing target column '{cfg.target_column}'"):
                results["target_analysis"] = target_analysis.analyze_target(df, profiles, cfg)

        results.setdefault("suggestions", [])
        self.results = results
        n_issues = results["quality"].summary.get("n_issues", 0)
        self.logger.info("Run complete: %d columns profiled, %d quality issue(s) found.", len(profiles), n_issues)
        return results
        
    def compare(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, Any]:
        """Compare two datasets and detect data drift.
        
        Args:
            reference_df: The baseline dataset (e.g. training data)
            current_df: The new dataset (e.g. testing or production data)
            
        Returns:
            A dictionary containing drift metrics (PSI, KS Test) for all features.
        """
        self.logger.info("Starting dataset comparison (drift detection)...")
        
        cfg = self.config
        detector = ColumnTypeDetector(cfg)
        
        # Profile based on reference data
        with stage(self.logger, "Profiling reference dataset for comparison"):
            profiles = detector.profile_dataframe(reference_df)
            
        with stage(self.logger, "Detecting feature drift"):
            drift_results = drift.compare_datasets(reference_df, current_df, profiles, cfg)
            
        self.results["drift"] = drift_results
        
        n_drifted = drift_results.get("n_drifted", 0)
        self.logger.info("Comparison complete: %d feature(s) drifted.", n_drifted)
        return drift_results

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Defensive guards for edge-case inputs (empty / single-row / single-column data)."""
        if df is None:
            raise ValueError("DataFrame is None.")
        if df.shape[1] == 0:
            raise ValueError("DataFrame has no columns.")
        if df.shape[0] == 0:
            self.logger.warning("DataFrame is empty (0 rows); most analyses will be skipped or trivial.")
        if df.shape[0] > self.config.max_rows_full_scan:
            self.logger.warning(
                "DataFrame has %d rows, above max_rows_full_scan=%d; heavy operations will sample down.",
                df.shape[0],
                self.config.max_rows_full_scan,
            )
        return df

    # ------------------------------------------------------------------ #
    # Cleaning (opt-in)
    # ------------------------------------------------------------------ #
    def clean(self, steps: list | None = None) -> pd.DataFrame:
        """Run the optional auto-cleaning pipeline and return the cleaned DataFrame.

        Requires :meth:`run` to have been called first so column profiles exist.
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        if not self.results.get("profiles"):
            self.run()
        cleaned, report = cleaning.auto_clean(self.df, self.results["profiles"], self.config, steps=steps)
        self.results["cleaning_report"] = report
        for action in report.actions:
            self.logger.info("[clean] %s", action)
        return cleaned

    # ------------------------------------------------------------------ #
    # Reporting / export
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Return (and print) a short console-friendly summary of the last run."""
        if not self.results:
            raise ValueError("Call run() before summary().")
        text = build_console_summary(self.results)
        print(text)
        return text

    def generate_report(self, output_path: str | Path | None = None, fmt: str = "html") -> Path:
        """Render a single report file. Runs the pipeline first if it hasn't been run yet.

        When *output_path* is a bare filename (no directory component), the
        report is placed inside :pyattr:`config.output_dir` so that all
        generated artifacts live in the same folder.
        """
        if not self.results:
            self.run()
        if output_path is None:
            output_path = self.config.resolved_output_dir() / f"report.{fmt}"
        else:
            output_path = Path(output_path)
            # Bare filename → place inside the configured output directory
            if output_path.parent == Path("."):
                output_path = self.config.resolved_output_dir() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        builder = ReportBuilder(self.config, version=__version__)
        if fmt == "html":
            output_path.write_text(builder.render_html(self.results), encoding="utf-8")
        elif fmt in ("markdown", "md"):
            output_path.write_text(builder.render_markdown(self.results), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported report format '{fmt}'. Use 'html' or 'markdown', or see export().")

        self.logger.info("Report written to %s", output_path)
        return output_path

    def export(
        self, output_dir: str | Path | None = None, formats: list | None = None, basename: str = "report"
    ) -> dict[str, Path]:
        """Export the results in one or more formats (html, markdown, json, excel, pdf, csv, figures, dashboard)."""
        if not self.results:
            self.run()
        output_dir = Path(output_dir) if output_dir else self.config.resolved_output_dir()
        formats = formats or self.config.export_formats
        written = export_all(self.results, output_dir, formats, self.config, version=__version__, basename=basename)
        for fmt, path in written.items():
            self.logger.info("Exported %s -> %s", fmt, path)
        return written

    # ------------------------------------------------------------------ #
    # Convenience dunder methods
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        shape = self.df.shape if self.df is not None else None
        return f"OmniEDA(shape={shape}, target={self.config.target_column!r})"
