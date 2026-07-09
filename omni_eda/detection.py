"""Automatic column type & semantic-role detection.

This is the engine every other module (statistics, visualization, quality,
correlation...) queries to decide *how* to treat a column, so it runs once
per analysis and its results (a :class:`ColumnProfile` per column) are
passed around rather than recomputed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.utils import is_effectively_numeric

# --------------------------------------------------------------------------- #
# Regex patterns for "semantic" text columns
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^(https?://|www\.)[^\s]+$", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,15}$")
_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")
_ZIP_UK_RE = re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}$")
_COORD_RE = re.compile(r"^-?\d{1,3}\.\d+$")
_CURRENCY_RE = re.compile(r"^[\$€£¥]\s?-?\d[\d,]*\.?\d*$|^-?\d[\d,]*\.?\d*\s?[\$€£¥]$")
_PERCENT_RE = re.compile(r"^-?\d+\.?\d*\s?%$")
_HIDDEN_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\u200c\u200d\ufeff]")

_COUNTRY_NAME_HINTS = {"country", "nation", "nationality"}
_STATE_NAME_HINTS = {"state", "province", "region"}
_CITY_NAME_HINTS = {"city", "town", "municipality"}
_ID_NAME_HINTS = {"id", "uuid", "guid", "key", "index", "pk"}


@dataclass
class ColumnProfile:
    """Everything omni_eda knows about a single column."""

    name: str
    pandas_dtype: str
    n: int
    n_missing: int
    missing_pct: float
    n_unique: int
    unique_ratio: float

    base_type: str = "unknown"  # numeric | categorical | datetime | boolean | text | id | constant
    semantic_type: str | None = (
        None  # email | url | phone | currency | percentage | coordinate | zip | country | state | city | binary | ordinal | cyclic
    )
    is_numeric: bool = False
    is_datetime: bool = False
    is_boolean: bool = False
    is_categorical: bool = False
    is_text: bool = False
    is_id_like: bool = False
    is_constant: bool = False
    is_single_unique: bool = False
    is_high_cardinality: bool = False
    is_low_variance: bool = False
    is_zero_variance: bool = False
    is_mixed_type: bool = False
    is_binary_encoded: bool = False
    is_label_encoded: bool = False
    is_ordinal_candidate: bool = False
    is_cyclic_candidate: bool = False
    flags: list[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


def _regex_match_ratio(series: pd.Series, pattern: re.Pattern, sample_size: int = 500) -> float:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    sample = non_null.sample(n=min(sample_size, len(non_null)), random_state=0)
    matches = sample.str.match(pattern)
    return float(matches.mean())


def _detect_semantic_text_type(series: pd.Series, name: str) -> str | None:
    lname = name.lower()
    if _regex_match_ratio(series, _EMAIL_RE) > 0.8:
        return "email"
    if _regex_match_ratio(series, _URL_RE) > 0.8:
        return "url"
    if any(h in lname for h in ("phone", "tel", "mobile", "fax")) and _regex_match_ratio(series, _PHONE_RE) > 0.6:
        return "phone"
    if any(h in lname for h in ("zip", "postal")) and (
        _regex_match_ratio(series, _ZIP_RE) > 0.6 or _regex_match_ratio(series, _ZIP_UK_RE) > 0.6
    ):
        return "zipcode"
    if any(h in lname for h in _COUNTRY_NAME_HINTS):
        return "country"
    if any(h in lname for h in _STATE_NAME_HINTS):
        return "state"
    if any(h in lname for h in _CITY_NAME_HINTS):
        return "city"
    if _regex_match_ratio(series, _CURRENCY_RE) > 0.6 or any(
        h in lname for h in ("price", "cost", "revenue", "salary", "amount")
    ):
        if _regex_match_ratio(series, _CURRENCY_RE) > 0.3:
            return "currency"
    if _regex_match_ratio(series, _PERCENT_RE) > 0.6:
        return "percentage"
    return None


def _detect_coordinate_pair(columns: list[str]) -> dict[str, str]:
    """Heuristically flag lat/lon columns based on naming."""
    result: dict[str, str] = {}
    lat_hints = {"lat", "latitude"}
    lon_hints = {"lon", "lng", "long", "longitude"}
    for col in columns:
        lname = col.lower()
        if any(h == lname or lname.endswith("_" + h) or lname.startswith(h) for h in lat_hints):
            result[col] = "latitude"
        elif any(h == lname or lname.endswith("_" + h) or lname.startswith(h) for h in lon_hints):
            result[col] = "longitude"
    return result


def _looks_like_datetime(series: pd.Series, sample_size: int = 200) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.sample(n=min(sample_size, len(non_null)), random_state=0)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.85


def _is_mixed_type(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty or not pd.api.types.is_object_dtype(series):
        return False
    types = non_null.map(type).unique()
    return len(types) > 1


class ColumnTypeDetector:
    """Runs the full detection sweep over a DataFrame."""

    def __init__(self, config: EDAConfig | None = None):
        self.config = config or EDAConfig()

    def profile_dataframe(self, df: pd.DataFrame) -> dict[str, ColumnProfile]:
        coord_hints = _detect_coordinate_pair(list(df.columns))
        profiles: dict[str, ColumnProfile] = {}
        n_rows = len(df)

        for col in df.columns:
            profiles[col] = self.profile_column(df[col], name=col, n_rows=n_rows, coord_hint=coord_hints.get(col))
        return profiles

    def profile_column(
        self,
        series: pd.Series,
        name: str,
        n_rows: int | None = None,
        coord_hint: str | None = None,
    ) -> ColumnProfile:
        cfg = self.config
        n = n_rows if n_rows is not None else len(series)
        n_missing = int(series.isna().sum())
        missing_pct = (n_missing / n * 100.0) if n else 0.0
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = (n_unique / n) if n else 0.0

        profile = ColumnProfile(
            name=name,
            pandas_dtype=str(series.dtype),
            n=n,
            n_missing=n_missing,
            missing_pct=missing_pct,
            n_unique=n_unique,
            unique_ratio=unique_ratio,
        )

        non_null = series.dropna()

        # ---- constant / single-unique --------------------------------
        if n_unique == 0:
            profile.base_type = "constant"
            profile.is_constant = True
            profile.add_flag("all_missing")
            return profile
        if n_unique == 1:
            profile.is_single_unique = True
            profile.is_constant = True
            profile.add_flag("constant")

        # ---- boolean -----------------------------------------------------
        if pd.api.types.is_bool_dtype(series) or (
            n_unique == 2 and set(non_null.unique().tolist()) <= {0, 1, "True", "False", "true", "false"}
        ):
            profile.base_type = "boolean"
            profile.is_boolean = True
            profile.is_binary_encoded = pd.api.types.is_numeric_dtype(series)
            return profile

        # ---- datetime ------------------------------------------------
        if _looks_like_datetime(series):
            profile.base_type = "datetime"
            profile.is_datetime = True
            return profile

        # ---- mixed types (object column with several python types) ------
        if _is_mixed_type(series):
            profile.is_mixed_type = True
            profile.add_flag("mixed_dtype")

        # ---- numeric -------------------------------------------------------
        if pd.api.types.is_numeric_dtype(series) or is_effectively_numeric(series):
            numeric_series = pd.to_numeric(non_null, errors="coerce") if not pd.api.types.is_numeric_dtype(series) else non_null
            profile.base_type = "numeric"
            profile.is_numeric = True

            variance = float(np.nanvar(numeric_series)) if len(numeric_series) > 1 else 0.0
            if variance == 0.0:
                profile.is_zero_variance = True
                profile.add_flag("zero_variance")
            elif variance < cfg.low_variance_threshold:
                profile.is_low_variance = True
                profile.add_flag("low_variance")

            unique_vals = set(numeric_series.unique().tolist())
            if unique_vals <= {0, 1}:
                profile.is_binary_encoded = True
            elif n_unique <= 15 and np.array_equal(
                np.sort(list(unique_vals)), np.arange(min(unique_vals), min(unique_vals) + n_unique)
            ):
                profile.is_label_encoded = True
                profile.is_ordinal_candidate = True

            lname = name.lower()
            if any(h in lname for h in ("hour", "month", "day_of_week", "dow", "angle", "degree")):
                profile.is_cyclic_candidate = True

            if coord_hint:
                profile.semantic_type = coord_hint

            if n_unique <= max(2, int(0.02 * n)) and n_unique <= 25:
                profile.add_flag("possible_categorical_numeric")
            return profile

        # ---- text-ish object column: decide categorical vs free text vs id --
        avg_len = non_null.astype(str).str.len().mean() if len(non_null) else 0.0
        semantic = _detect_semantic_text_type(non_null, name)
        if semantic:
            profile.semantic_type = semantic
            profile.base_type = "text"
            profile.is_text = True
            return profile

        lname = name.lower()
        if unique_ratio >= cfg.id_like_uniqueness_ratio and (any(h in lname for h in _ID_NAME_HINTS) or avg_len <= 40):
            profile.base_type = "id"
            profile.is_id_like = True
            return profile

        is_high_card = n_unique > cfg.high_cardinality_threshold and unique_ratio > cfg.high_cardinality_ratio
        if is_high_card and avg_len > 25:
            profile.base_type = "text"
            profile.is_text = True
            return profile

        profile.base_type = "categorical"
        profile.is_categorical = True
        if is_high_card:
            profile.is_high_cardinality = True
            profile.add_flag("high_cardinality")
        return profile


def detect_hidden_characters(series: pd.Series, sample_size: int = 500) -> float:
    """Fraction of sampled string values containing non-printable / zero-width chars."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    sample = non_null.sample(n=min(sample_size, len(non_null)), random_state=0)
    return float(sample.str.contains(_HIDDEN_CHAR_RE).mean())


def group_columns_by_base_type(profiles: dict[str, ColumnProfile]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, profile in profiles.items():
        groups.setdefault(profile.base_type, []).append(name)
    return groups
