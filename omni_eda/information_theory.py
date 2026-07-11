import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

try:
    from sklearn.metrics import (
        mutual_info_score,
        normalized_mutual_info_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class InformationTheoryAnalysis:
    """Computes information theoretic metrics for dataset features."""

    def __init__(self, df: pd.DataFrame, bins: int = 10):
        self.df = df
        self.bins = bins
        self._discretized_df: Optional[pd.DataFrame] = None

    def _discretize(self) -> pd.DataFrame:
        """Discretizes continuous data for IT calculations."""
        if self._discretized_df is not None:
            return self._discretized_df

        discretized = pd.DataFrame()
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Use qcut if possible, fallback to cut
                try:
                    discretized[col] = pd.qcut(self.df[col].dropna(), q=self.bins, labels=False, duplicates='drop')
                except ValueError:
                    discretized[col] = pd.cut(self.df[col].dropna(), bins=self.bins, labels=False)
            else:
                discretized[col] = self.df[col].astype("category").cat.codes
                
        # Fill missing values with a distinct category (e.g., -1)
        self._discretized_df = discretized.fillna(-1).astype(int)
        return self._discretized_df

    def _get_probs(self, series: pd.Series) -> np.ndarray:
        """Returns the probability distribution of a series."""
        counts = series.value_counts(normalize=True).values
        return counts

    def get_entropies(self) -> Dict[str, float]:
        """Calculates Shannon entropy for each feature."""
        df_disc = self._discretize()
        results = {}
        for col in df_disc.columns:
            probs = self._get_probs(df_disc[col])
            results[col] = float(entropy(probs, base=2))
        return results

    def get_mutual_information_matrix(self, normalized: bool = True) -> pd.DataFrame:
        """Computes pairwise Mutual Information or Normalized Mutual Information."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for mutual information.")
            
        df_disc = self._discretize()
        cols = df_disc.columns
        n = len(cols)
        matrix = np.zeros((n, n))

        metric = normalized_mutual_info_score if normalized else mutual_info_score

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i == j:
                    if normalized:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = self.get_entropies()[col1]
                elif i < j:
                    score = metric(df_disc[col1], df_disc[col2])
                    matrix[i, j] = score
                    matrix[j, i] = score

        return pd.DataFrame(matrix, index=cols, columns=cols)

    def get_uncertainty_coefficients(self) -> pd.DataFrame:
        """
        Calculates Theil's U (Uncertainty Coefficient) matrix.
        U(X|Y) = I(X;Y) / H(X)
        """
        df_disc = self._discretize()
        cols = df_disc.columns
        n = len(cols)
        matrix = np.zeros((n, n))
        
        entropies = self.get_entropies()
        mi_matrix = self.get_mutual_information_matrix(normalized=False)

        for i, x in enumerate(cols):
            h_x = entropies.get(x, 0)
            for j, y in enumerate(cols):
                if i == j:
                    matrix[i, j] = 1.0
                elif h_x > 0:
                    matrix[i, j] = mi_matrix.loc[x, y] / h_x
                else:
                    matrix[i, j] = 0.0

        return pd.DataFrame(matrix, index=cols, columns=cols)

    def get_divergences(self, col1: str, col2: str) -> Dict[str, float]:
        """Calculates KL and JS divergence between two features' distributions."""
        df_disc = self._discretize()
        if col1 not in df_disc.columns or col2 not in df_disc.columns:
            raise ValueError("Columns not found in dataset.")

        # To compare distributions, they need the same support space
        # We find the union of all unique values
        val1 = df_disc[col1].value_counts(normalize=True)
        val2 = df_disc[col2].value_counts(normalize=True)
        
        all_vals = set(val1.index).union(set(val2.index))
        
        p = np.array([val1.get(v, 1e-9) for v in all_vals])
        q = np.array([val2.get(v, 1e-9) for v in all_vals])
        
        # Normalize just in case
        p /= p.sum()
        q /= q.sum()

        kl_div = float(entropy(p, q))
        js_div = float(jensenshannon(p, q) ** 2)

        return {
            "KL_divergence": kl_div,
            "JS_divergence": js_div
        }

    def get_report(self) -> Dict[str, Any]:
        """Returns the full information theory report."""
        try:
            return {
                "entropies": self.get_entropies(),
                "nmi_matrix": self.get_mutual_information_matrix(normalized=True).to_dict() if HAS_SKLEARN else {},
                "uncertainty_coefficients": self.get_uncertainty_coefficients().to_dict() if HAS_SKLEARN else {}
            }
        except Exception as e:
            return {"error": str(e)}
