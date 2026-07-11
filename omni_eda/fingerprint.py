import hashlib
import json
from typing import Any, Dict

import pandas as pd
from omni_eda.detection import ColumnTypeDetector

class DatasetFingerprint:
    """Generates a unique fingerprint and metadata summary for a dataset."""

    def __init__(self, df: pd.DataFrame, profiles: Dict[str, Any] = None):
        self.df = df
        if profiles:
            self.col_types = {k: "numeric" if getattr(v, "is_numeric", False) else "categorical" for k, v in profiles.items()}
        else:
            self.col_types = {k: str(v) for k, v in df.dtypes.items()}

    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """Helper to create a deterministic hash of a dictionary."""
        d_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(d_str.encode("utf-8")).hexdigest()

    def generate_schema_hash(self) -> str:
        """Hash based on column names and their detected semantic types."""
        schema = {col: str(ctype) for col, ctype in self.col_types.items()}
        return self._hash_dict(schema)

    def generate_feature_signature(self) -> str:
        """Hash including basic descriptive statistics to fingerprint the content."""
        stats = {}
        for col in self.df.columns:
            # Simple handling of missing values in calculations
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats[col] = {
                    "mean": float(self.df[col].mean()) if not self.df[col].empty else 0.0,
                    "std": float(self.df[col].std()) if not self.df[col].empty else 0.0,
                    "nulls": int(self.df[col].isnull().sum())
                }
            else:
                stats[col] = {
                    "nunique": int(self.df[col].nunique()),
                    "nulls": int(self.df[col].isnull().sum())
                }
        return self._hash_dict(stats)

    def generate_dataset_version_hash(self) -> str:
        """Comprehensive hash of schema + feature signature + shape."""
        version_data = {
            "schema_hash": self.generate_schema_hash(),
            "feature_signature": self.generate_feature_signature(),
            "shape": self.df.shape
        }
        return self._hash_dict(version_data)

    def get_dtype_summary(self) -> Dict[str, int]:
        """Summary of pandas dtypes."""
        counts = self.df.dtypes.value_counts().to_dict()
        return {str(k): v for k, v in counts.items()}

    def get_semantic_summary(self) -> Dict[str, int]:
        """Summary of detected semantic types."""
        counts: Dict[str, int] = {}
        for ctype in self.col_types.values():
            type_str = str(ctype)
            counts[type_str] = counts.get(type_str, 0) + 1
        return counts

    def get_metadata_report(self) -> Dict[str, Any]:
        """Returns the complete fingerprint and metadata report."""
        return {
            "schema_hash": self.generate_schema_hash(),
            "feature_signature": self.generate_feature_signature(),
            "dataset_version_hash": self.generate_dataset_version_hash(),
            "dtype_summary": self.get_dtype_summary(),
            "semantic_summary": self.get_semantic_summary(),
            "shape": self.df.shape,
            "memory_usage_bytes": int(self.df.memory_usage(deep=True).sum()),
        }

    @staticmethod
    def compare_schemas(schema1: Dict[str, str], schema2: Dict[str, str]) -> Dict[str, Any]:
        """Detect column evolution between two schemas."""
        cols1 = set(schema1.keys())
        cols2 = set(schema2.keys())
        
        added = list(cols2 - cols1)
        removed = list(cols1 - cols2)
        common = cols1.intersection(cols2)
        
        type_changes = {
            col: {"from": schema1[col], "to": schema2[col]}
            for col in common if schema1[col] != schema2[col]
        }
        
        return {
            "is_identical": not (added or removed or type_changes),
            "added_columns": added,
            "removed_columns": removed,
            "type_changes": type_changes
        }
