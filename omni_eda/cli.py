"""Command-line interface: ``omni-eda run data.csv --target price --output report.html``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omni_eda import __version__
from omni_eda.analyzer import OmniEDA
from omni_eda.config import EDAConfig
from omni_eda.logger import get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni-eda",
        description="Fully automated exploratory data analysis for any tabular dataset.",
    )
    parser.add_argument("--version", action="version", version=f"omni_eda {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Analyze a dataset and generate a report.")
    run_parser.add_argument("source", help="Path to a CSV/Excel/Parquet/Feather/JSON file, or a directory of such files.")
    run_parser.add_argument("-o", "--output", default="omni_eda_output", help="Output directory (default: ./omni_eda_output).")
    run_parser.add_argument("--basename", default="report", help="Base filename for generated reports (default: report).")
    run_parser.add_argument(
        "-f",
        "--formats",
        nargs="+",
        default=["html"],
        choices=["html", "markdown", "json", "excel", "pdf", "csv", "figures", "dashboard"],
        help="One or more export formats (default: html).",
    )
    run_parser.add_argument("--target", default=None, help="Target/label column for supervised-style analysis.")
    run_parser.add_argument("--ignore", nargs="*", default=[], help="Column names to exclude from analysis.")
    run_parser.add_argument("--title", default="Automated EDA Report", help="Report title.")
    run_parser.add_argument("--theme", default="light", choices=["light", "dark", "minimal", "corporate"], help="Visual theme.")
    run_parser.add_argument(
        "--clean", action="store_true", help="Also run the (conservative) auto-cleaning pipeline and save the cleaned data."
    )
    run_parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Cap the number of rows loaded from the source (for a quick look at huge files).",
    )
    run_parser.add_argument("--no-pairplot", action="store_true", help="Skip the (slower) pairplot / PCA / t-SNE visualizations.")
    run_parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    run_parser.add_argument("--max-rows", type=int, default=2_000_000, help="Warn threshold for full-scan operations.")

    clean_parser = subparsers.add_parser(
        "clean", help="Run the auto-cleaning pipeline and write out a cleaned file, without a full report."
    )
    clean_parser.add_argument("source", help="Path to a data file.")
    clean_parser.add_argument(
        "-o", "--output", default="cleaned.csv", help="Where to write the cleaned CSV (default: cleaned.csv)."
    )
    clean_parser.add_argument("--quiet", action="store_true")

    return parser


def _run_command(args: argparse.Namespace) -> int:
    logger = get_logger(verbose=not args.quiet)
    output_dir = Path(args.output)

    config = EDAConfig(
        title=args.title,
        output_dir=str(output_dir),
        target_column=args.target,
        ignore_columns=args.ignore,
        theme=args.theme,
        verbose=not args.quiet,
        export_formats=args.formats,
        generate_pairplot=not args.no_pairplot,
        generate_pca=not args.no_pairplot,
        max_rows_full_scan=args.max_rows,
    )

    try:
        eda = OmniEDA(config=config)
        from omni_eda import loaders

        loaded = loaders.load_data(args.source)

        if isinstance(loaded, dict):
            logger.info("Directory source detected; analyzing each file separately.")
            any_failed = False
            for name, frame in loaded.items():
                if args.sample_rows and len(frame) > args.sample_rows:
                    frame = frame.sample(args.sample_rows, random_state=config.random_state)
                sub_eda = OmniEDA(config=EDAConfig(**{**config.__dict__, "output_dir": str(output_dir / name)}))
                try:
                    sub_eda.run(frame)
                    sub_eda.export(basename=args.basename)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed analyzing '%s': %s", name, exc)
                    any_failed = True
            return 1 if any_failed else 0

        if args.sample_rows and len(loaded) > args.sample_rows:
            loaded = loaded.sample(args.sample_rows, random_state=config.random_state)

        eda.run(loaded)
        eda.summary()
        written = eda.export(basename=args.basename)

        if args.clean:
            cleaned = eda.clean()
            clean_path = output_dir / f"{args.basename}_cleaned.csv"
            cleaned.to_csv(clean_path, index=False)
            logger.info("Cleaned data written to %s", clean_path)

        for fmt, path in written.items():
            print(f"  [{fmt}] {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("omni-eda run failed: %s", exc)
        return 1


def _clean_command(args: argparse.Namespace) -> int:
    logger = get_logger(verbose=not args.quiet)
    try:
        from omni_eda import cleaning, loaders
        from omni_eda.detection import ColumnTypeDetector

        df = loaders.load_data(args.source)
        if not hasattr(df, "columns"):
            logger.error("clean only supports a single tabular file, not a directory.")
            return 1
        cfg = EDAConfig(verbose=not args.quiet)
        profiles = ColumnTypeDetector(cfg).profile_dataframe(df)
        cleaned, report = cleaning.auto_clean(df, profiles, cfg)
        for action in report.actions:
            logger.info(action)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(args.output, index=False)
        logger.info("Cleaned data (%d -> %d rows) written to %s", report.rows_before, report.rows_after, args.output)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("omni-eda clean failed: %s", exc)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "clean":
        return _clean_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
