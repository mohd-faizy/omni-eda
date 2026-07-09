"""Load data from (almost) anything into a :class:`pandas.DataFrame`.

Supported sources
------------------
* An existing :class:`pandas.DataFrame` (passthrough)
* CSV / TSV (with optional chunked reading for very large files)
* Excel (``.xlsx`` / ``.xls``)
* Parquet
* Feather
* JSON / JSON-lines
* A SQLAlchemy connection/engine + query string
* A directory containing any mix of the above -> ``dict[str, DataFrame]``
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union

import pandas as pd

from omni_eda.logger import get_logger

PathLike = Union[str, Path]

_READERS = {
    ".csv": "read_csv",
    ".tsv": "read_csv",
    ".txt": "read_csv",
    ".parquet": "read_parquet",
    ".pq": "read_parquet",
    ".feather": "read_feather",
    ".json": "read_json",
    ".jsonl": "read_json",
    ".xlsx": "read_excel",
    ".xls": "read_excel",
    ".xlsm": "read_excel",
}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension has no registered reader."""


def load_data(
    source: PathLike | pd.DataFrame | sqlalchemy.engine.Engine,  # noqa: F821
    *,
    query: str | None = None,
    chunksize: int | None = None,
    **read_kwargs: Any,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load ``source`` into a DataFrame (or an iterator of chunks).

    Parameters
    ----------
    source:
        A DataFrame (returned as-is), a file path, or a SQLAlchemy engine/connection.
    query:
        Required when ``source`` is a database connection.
    chunksize:
        If given for a CSV/TSV source, returns an iterator of DataFrame chunks
        instead of a single DataFrame, so arbitrarily large files can be
        processed without exhausting memory.
    """
    logger = get_logger()

    if isinstance(source, pd.DataFrame):
        return source

    # SQLAlchemy engine/connection duck-typing: has `.connect` or `.execute`.
    if hasattr(source, "connect") or hasattr(source, "execute"):
        if not query:
            raise ValueError("A `query` is required when loading data from a database connection.")
        logger.info("Loading data via SQL query.")
        return pd.read_sql(query, source, **read_kwargs)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    if path.is_dir():
        return load_directory(path, **read_kwargs)

    return _load_file(path, chunksize=chunksize, **read_kwargs)


def _load_file(path: Path, *, chunksize: int | None = None, **read_kwargs: Any):
    logger = get_logger()
    suffix = path.suffix.lower()
    reader_name = _READERS.get(suffix)
    if reader_name is None:
        raise UnsupportedFileTypeError(f"Unsupported file extension '{suffix}'. Supported: {sorted(_READERS)}")

    reader = getattr(pd, reader_name)
    logger.info("Loading '%s' with pandas.%s", path.name, reader_name)

    if reader_name == "read_csv":
        sep = read_kwargs.pop("sep", "\t" if suffix == ".tsv" else read_kwargs.pop("sep", ","))
        if chunksize:
            return reader(path, sep=sep, chunksize=chunksize, **read_kwargs)
        return reader(path, sep=sep, **read_kwargs)

    if reader_name == "read_json" and suffix == ".jsonl":
        read_kwargs.setdefault("lines", True)

    return reader(path, **read_kwargs)


def load_directory(directory: PathLike, **read_kwargs: Any) -> dict[str, pd.DataFrame]:
    """Load every recognized data file in ``directory`` into a dict keyed by filename."""
    logger = get_logger()
    directory = Path(directory)
    frames: dict[str, pd.DataFrame] = {}

    for pattern_ext in _READERS:
        for filepath in sorted(glob.glob(str(directory / f"*{pattern_ext}"))):
            fp = Path(filepath)
            try:
                frames[fp.stem] = _load_file(fp, **read_kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping '%s': %s", fp.name, exc)

    if not frames:
        logger.warning("No supported data files found in '%s'.", directory)
    return frames


def concat_chunks(chunks: Iterator[pd.DataFrame], max_rows: int | None = None) -> pd.DataFrame:
    """Materialize a chunked reader into a single DataFrame, optionally capped at ``max_rows``."""
    parts = []
    total = 0
    for chunk in chunks:
        parts.append(chunk)
        total += len(chunk)
        if max_rows and total >= max_rows:
            break
    df = pd.concat(parts, ignore_index=True)
    return df.iloc[:max_rows] if max_rows else df
