# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Initial release

### Added
- Data loading for CSV/TSV, Excel, Parquet, Feather, JSON/JSONL, SQL connections, in-memory
  DataFrames, and folders of mixed files, with optional chunked CSV reading.
- Automatic column type & semantic-role detection (numeric, categorical, datetime, boolean,
  text, ID, email, URL, phone, currency, percentage, coordinates, ZIP codes, and more).
- Data quality report covering missingness, duplicates, infinities, impossible values,
  invalid/future dates, empty/whitespace strings, hidden characters, encoding issues, skew,
  class imbalance, high correlation, and target leakage.
- Descriptive statistics for numeric, categorical, text, datetime, and boolean columns.
- ~40 visualization functions across univariate, bivariate, multivariate, and time-series
  categories, orchestrated by a `PlotEngine` that skips plots that don't apply to the data.
- Correlation & association analysis: Pearson, Spearman, Kendall, Cramer's V, correlation
  ratio, mutual information, and distance correlation.
- Outlier detection: Z-score, modified Z-score, IQR, Isolation Forest, LOF, Elliptic Envelope,
  DBSCAN.
- Rule-based feature-engineering suggestions.
- Target-aware analysis: class imbalance, ANOVA/chi-square tests, Random Forest feature
  importance + mutual information, and ROC/PR/lift curves for binary classification targets.
- HTML, Markdown, JSON, Excel, PDF, CSV, PNG/SVG, and interactive-dashboard export.
- Optional, explicit, auditable auto-cleaning pipeline.
- `omni-eda` command-line interface (`run` and `clean` subcommands).
- Full test suite (pytest), CI workflow (GitHub Actions), pre-commit configuration, and
  PyPI-ready packaging.
