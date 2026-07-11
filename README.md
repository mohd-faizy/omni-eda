<div align="center">
  <img src="https://raw.githubusercontent.com/mohd-faizy/omni-eda/main/assets/omni-eda-banner.png" alt="omni_eda Banner" width="450">
</div>

<p align="center">
  <a href="https://github.com/mohd-faizy/omni-eda/actions/workflows/ci.yml"><img src="https://github.com/mohd-faizy/omni-eda/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://pypi.org/project/omni-eda/"><img src="https://img.shields.io/pypi/v/omni-eda.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/omni-eda/"><img src="https://img.shields.io/pypi/pyversions/omni-eda.svg" alt="Python Versions"></a>
  <a href="https://github.com/mohd-faizy/omni-eda/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mohd-faizy/omni-eda.svg" alt="License"></a>
</p>

# omni_eda

**Fully automated, production-grade Exploratory Data Analysis for any pandas DataFrame.**

`omni_eda` inspects a dataset, profiles every column, finds data-quality problems, computes
statistics for every data type, generates dozens of visualizations, detects outliers and
correlations, suggests feature engineering steps, optionally evaluates a target column, and
renders everything into a polished, shareable HTML report — in one function call.

```python
from omni_eda import OmniEDA

OmniEDA("data.csv").generate_report("report.html")
```

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [What's in the report](#whats-in-the-report)
- [Export formats](#export-formats)
- [Optional cleaning](#optional-cleaning)
- [Architecture](#architecture)
- [Performance notes](#performance-notes)
- [Development](#development)
- [License](#license)

---

## Features

| Feature / Capability | Description & Details |
| :--- | :--- |
| **Broad Input Support** | CSV/TSV, Excel, Parquet, Feather, JSON/JSONL, SQL (via SQLAlchemy connection), in-memory DataFrames, folders of files, or chunked CSV reader for datasets exceeding memory. |
| **Automatic Column Profiling** | Profiling for numeric, categorical, datetime, boolean, text, and ID columns. Semantic detection of emails, URLs, phone numbers, currencies, percentages, coordinates, ZIP codes, country/state/city columns, binary/label/ordinal-encoded columns, constant/high-cardinality columns, and mixed-dtype columns. |
| **Data Quality Report** | Detection of missing values, duplicate rows/columns, infinities, impossible negatives, invalid/future dates, empty/whitespace-only strings, hidden/non-printable characters, encoding issues, skewed distributions, class imbalance, highly correlated columns, and target-leakage candidates. |
| **Descriptive Statistics** | Mean/median/mode/std/MAD/IQR/percentiles/skewness/kurtosis for numeric columns; frequency/entropy/rare-category detection for categoricals; length/word/character stats for free text; range/seasonality for datetimes. |
| **~40 Plot Types** | Univariate, bivariate, multivariate, and time-series analysis (histograms/KDE, boxplots, violin, ECDF, Q-Q, rank, lollipop, pie, scatter, hexbin, regression, residual, joint, pairplots, mosaic, cross-tab heatmaps, FacetGrid, cluster maps, parallel coordinates, Andrews curves, radar, bubble, 3D scatter, PCA/t-SNE/UMAP, correlation networks, trend, seasonality, rolling mean/std, lag plots, ACF/PACF, etc.). Plotting is skipped automatically when it wouldn't make sense. |
| **Correlation & Association** | Pearson/Spearman/Kendall, Cramer's V, correlation ratio (categorical ↔ numeric), mutual information, and distance correlation. |
| **Outlier Detection** | Z-score, modified Z-score, IQR, Isolation Forest, Local Outlier Factor, Elliptic Envelope, and DBSCAN. |
| **Feature Engineering Suggestions** | Encoding, scaling, log/power transforms, binning, datetime decomposition, interaction & polynomial feature candidates, redundant-feature and rare-category flags. |
| **Target Analysis** | Class imbalance, ANOVA / chi-square association tests, Random-Forest feature importance + mutual information, and (for binary targets) a baseline classifier with ROC, Precision-Recall, and lift charts. |
| **Advanced Statistics & A/B Testing** | Robust statistical tests for group comparisons (A/B testing), hypothesis testing, and deep statistical insights generation. |
| **Data Drift & Shifts** | Detect distribution shifts between training and serving datasets, monitor feature drift, and ensure data integrity over time. |
| **Clustering Analysis** | Automated unsupervised learning to discover natural groupings, similarities, and anomalies in the data. |
| **Multi-Format Export** | HTML report, Markdown, JSON, Excel workbook, PDF (figure bundle), raw PNG/SVG figures, CSV tables, and a self-contained interactive HTML dashboard. |
| **Built for Scale** | Sampling guards, vectorized pandas/NumPy operations, optional multiprocessing, memory-aware dtype downcasting, and defensive handling of empty, single-row, single-column, and all-null-column datasets. |
| **Optional, Auditable Cleaning** | Every cleaning step is opt-in, logged, and returns a new DataFrame; nothing is changed silently. |


## Installation

```bash
pip install omni-eda
```

or, from a local checkout:

```bash
pip install -e .
```

Some visualizations (mosaic plots, ACF/PACF, correlation networks, UMAP) and file formats
(Parquet/Feather) need extra libraries. Install everything with:

```bash
pip install "omni-eda[extra]"
```

Every feature that needs an extra dependency degrades gracefully (it's skipped, with a log
message) if that dependency isn't installed — nothing crashes.

## Quick start

```python
import pandas as pd
from omni_eda import OmniEDA

df = pd.read_csv("customers.csv")

eda = OmniEDA()
results = eda.run(df)          # run the full pipeline
eda.summary()                  # quick console overview
eda.generate_report("report.html")
```

One-liner form:

```python
from omni_eda import OmniEDA

OmniEDA("customers.csv").generate_report("report.html")
```

With a target column (enables class-imbalance checks, feature importance, ANOVA/chi-square
tests, and ROC/PR/lift curves for binary targets):

```python
from omni_eda import OmniEDA, EDAConfig

config = EDAConfig(target_column="churned", theme="corporate")
eda = OmniEDA(config=config)
eda.run(df)
eda.export(formats=["html", "excel", "dashboard"])
```

See [`examples/basic_usage.py`](examples/basic_usage.py) and
[`examples/advanced_usage.py`](examples/advanced_usage.py) for complete, runnable scripts.

## Command-line interface

```bash
# Analyze a file and write an HTML report
omni-eda run data.csv -o report_output -f html

# With a target column, several export formats, and a lighter run for a quick look
omni-eda run data.csv --target price -f html json excel --sample-rows 50000

# Run just the (conservative) auto-cleaning pipeline
omni-eda clean data.csv -o cleaned.csv
```

Run `omni-eda run --help` for the full list of flags (theme, ignored columns, output
formats, quiet mode, etc.).

## Configuration

Every tunable knob lives in a single `EDAConfig` dataclass (see `omni_eda/config.py` for the
full list with defaults):

```python
from omni_eda import EDAConfig

config = EDAConfig(
    title="Customer Churn - EDA",
    target_column="churned",
    ignore_columns=["customer_id"],
    theme="dark",                          # light | dark | minimal | corporate
    high_correlation_threshold=0.85,
    outlier_methods=["iqr", "zscore", "isolation_forest", "lof"],
    sample_for_plots=20_000,               # cap rows used for plotting on huge datasets
    max_rows_for_expensive_ops=50_000,     # cap rows used for correlation/outlier/model fits
    export_formats=["html", "json", "excel"],
)
```

## What's in the report

The generated HTML report includes: a dataset overview (rows, columns, memory, column-type
breakdown), the full data-quality issue list (severity-tagged), missing-value analysis with
matrix/heatmap/dendrogram/bar visualizations, per-column statistics and distribution plots,
correlation heatmaps and high-correlation pairs, an outlier summary table, bivariate and
multivariate visualization galleries, time-series analysis (when datetime columns are present),
target analysis (when a target column is configured), feature-engineering suggestions, and a
plain-language final summary.

## Export formats

| Format      | What you get                                                             |
|-------------|---------------------------------------------------------------------------|
| `html`      | The full visual report (self-contained, images embedded as base64)        |
| `markdown`  | A text-only version of the report (tables + suggestions, no images)       |
| `json`      | Machine-readable statistics, quality issues, correlations, suggestions    |
| `excel`     | A multi-sheet workbook (overview, statistics, issues, correlations, ...)  |
| `pdf`       | Every generated PNG figure bundled into a single multi-page PDF           |
| `csv`       | Separate CSVs for statistics, quality issues, correlations, outliers      |
| `figures`   | Every individual plot as a standalone PNG/SVG file                        |
| `dashboard` | A single-file interactive HTML dashboard (Plotly.js via CDN, no server)   |

```python
eda.export(output_dir="out", formats=["html", "excel", "dashboard"])
```

## Optional cleaning

Cleaning never happens implicitly. Call `.clean()` explicitly and get back a new DataFrame
plus a human-readable log of exactly what changed:

```python
cleaned_df = eda.clean(steps=["dedup_rows", "dedup_columns", "convert_dtypes", "infinities"])
```

Available steps: `dedup_rows`, `dedup_columns`, `drop_constant`, `fill_missing`,
`convert_dtypes`, `strip_whitespace`, `infinities` (the default pipeline runs the
non-destructive subset of these; `fill_missing` and `drop_constant` are opt-in since they
change the data more aggressively). Lower-level building blocks (`clip_outliers_iqr`,
`encode_categoricals`, `scale_numeric_columns`, ...) are available directly from
`omni_eda.cleaning` for custom pipelines.

## Architecture

```
omni_eda/
├── __init__.py            # public API: OmniEDA, EDAConfig
├── analyzer.py             # OmniEDA orchestrator - runs the full pipeline
├── config.py                # EDAConfig dataclass (every tunable setting)
├── loaders.py                # CSV/Excel/Parquet/Feather/JSON/SQL/folder loading
├── detection.py               # column type & semantic-role detection
├── statistics.py               # descriptive statistics per dtype
├── cleaning.py                   # optional, auditable cleaning operations
├── quality.py                     # data quality issue detection
├── correlation.py                  # Pearson/Spearman/Kendall/Cramer's V/MI/distance corr
├── outliers.py                      # Z-score/IQR/Isolation Forest/LOF/DBSCAN/Elliptic Env.
├── missing.py                        # missing-value analysis & visualization
├── visualization.py                   # ~40 plot functions + the PlotEngine orchestrator
├── feature_engineering.py              # rule-based feature suggestions
├── target_analysis.py                   # class imbalance, tests, importance, ROC/PR/lift
├── themes.py                             # matplotlib/seaborn/report color themes
├── report.py                              # HTML/Markdown/console report builder (Jinja2)
├── export.py                               # HTML/MD/JSON/Excel/PDF/CSV/figure export
├── dashboard.py                             # self-contained interactive HTML dashboard
├── logger.py                                 # package-wide logging + progress bars
├── utils.py                                   # shared helpers (sampling, caching, ...)
├── cli.py                                      # `omni-eda` command-line interface
└── templates/report.html.j2                     # the HTML report template
```

Every pipeline stage in `OmniEDA.run()` is wrapped so a failure in one stage (an exotic dtype
breaking a single plot, say) is logged and skipped rather than aborting the whole run.

## Performance notes

`omni_eda` is built to stay usable on large datasets without needing a distributed runtime:

- Expensive operations (correlation, outlier detection, PCA/t-SNE, model fitting) sample down
  to `max_rows_for_expensive_ops` (default 50,000 rows) rather than processing everything.
- Plotting samples down to `sample_for_plots` (default 20,000 rows).
- Duplicate-column detection hashes columns instead of transposing the DataFrame, which is
  dramatically faster on wide-ish, long DataFrames.
- CSV loading supports `chunksize` for files too large to read at once.
- Numeric downcasting and category-dtype conversion (`omni_eda.utils.optimize_dtypes`) are
  available to cut memory usage before analysis.
- Random Forest / mutual information calls in target analysis use their own bounded samples so
  a 26-column, 100k-row dataset with a target column finishes in roughly a minute on a single
  CPU core; tune `max_rows_for_expensive_ops`, `n_jobs`, and `enable_target_modeling` /
  `enable_model_based_outliers` down further for a quicker look at very large data.

## Development

```bash
git clone https://github.com/example/omni_eda.git
cd omni_eda
pip install -e ".[dev,extra]"
pre-commit install

pytest                       # run the test suite
pytest --cov=omni_eda        # with coverage
ruff check omni_eda tests    # lint
black omni_eda tests         # format
mypy omni_eda                # type-check
```

Contributions are welcome — please open an issue or pull request.

## License

MIT — see [LICENSE](LICENSE).
