"""Logging utilities for ``omni_eda``.

Named ``logger.py`` rather than ``logging.py`` on purpose: shadowing the
standard-library ``logging`` module with a same-named top-level module is a
classic footgun (any ``import logging`` elsewhere in the package would
import *this* file instead). The public surface still behaves the way the
spec asks for -- a single call to get a configured, package-wide logger.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

try:
    from tqdm.auto import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - tqdm is a core dependency, but degrade gracefully
    _HAS_TQDM = False

_PACKAGE_LOGGER_NAME = "omni_eda"
_CONFIGURED = False

T = TypeVar("T")


def get_logger(name: str = _PACKAGE_LOGGER_NAME, verbose: bool | None = None) -> logging.Logger:
    """Return a configured logger, initializing package-wide handlers once.

    ``verbose`` only *changes* the current level when explicitly passed
    (``True``/``False``). Calling ``get_logger()`` with no argument -- as
    most internal modules do, just to log a message -- must never silently
    reset a level someone else already configured via
    :func:`set_verbosity` or an explicit ``verbose=`` earlier in the run.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        root = logging.getLogger(_PACKAGE_LOGGER_NAME)
        root.handlers.clear()
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.propagate = False
        root.setLevel(logging.INFO)  # sane default until set_verbosity()/explicit verbose= says otherwise
        _CONFIGURED = True

    if verbose is not None:
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


def set_verbosity(verbose: bool) -> None:
    logging.getLogger(_PACKAGE_LOGGER_NAME).setLevel(logging.INFO if verbose else logging.WARNING)


def progress(iterable: Iterable[T], desc: str = "", total: int | None = None, enabled: bool = True) -> Iterator[T]:
    """Wrap ``iterable`` in a tqdm progress bar when tqdm is available and enabled."""
    if enabled and _HAS_TQDM:
        yield from _tqdm(iterable, desc=desc, total=total, leave=False)
    else:
        yield from iterable


@contextmanager
def stage(logger: logging.Logger, name: str):
    """Context manager that logs the start/end (and duration) of a pipeline stage."""
    import time

    logger.info("-> %s ...", name)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - deliberately broad, pipeline stages must not crash the run
        logger.warning("Stage '%s' failed and was skipped: %s", name, exc)
    else:
        elapsed = time.perf_counter() - start
        logger.info("   %s done in %.2fs", name, elapsed)
