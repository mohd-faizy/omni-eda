import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DatasetComplexity:
    """Estimates the complexity and intrinsic dimensionality of a dataset."""

    def __init__(self, df: pd.DataFrame):
        # We only consider numerical columns for complexity estimation
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.n_samples, self.n_features = self.numeric_df.shape
        self._pca_result: Optional[PCA] = None
        self._scaled_data: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None

    def _prepare_data(self) -> None:
        """Scales data and handles NaNs for PCA."""
        if self._scaled_data is not None:
            return

        if self.n_features == 0 or self.n_samples < 2:
            raise ValueError("Insufficient numeric data for complexity analysis.")

        # Fill NaNs with mean to allow PCA
        data = self.numeric_df.fillna(self.numeric_df.mean()).values
        
        # Scale
        scaler = StandardScaler()
        self._scaled_data = scaler.fit_transform(data)
        
        # Compute PCA
        self._pca_result = PCA()
        self._pca_result.fit(self._scaled_data)
        
        # Eigenvalues (variance explained)
        self._eigenvalues = self._pca_result.explained_variance_

    def get_pca_spectrum(self) -> Dict[str, Any]:
        """Returns the PCA variance spectrum."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn is not installed."}
        
        try:
            self._prepare_data()
        except ValueError as e:
            return {"error": str(e)}

        explained_variance_ratio = self._pca_result.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        return {
            "eigenvalues": self._eigenvalues.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance": cumulative_variance.tolist()
        }

    def estimate_effective_rank(self, threshold: float = 0.95) -> int:
        """Number of components needed to explain `threshold` of variance."""
        if not HAS_SKLEARN:
            return -1
        
        spectrum = self.get_pca_spectrum()
        if "error" in spectrum:
            return -1
            
        cumulative = np.array(spectrum["cumulative_variance"])
        return int(np.argmax(cumulative >= threshold)) + 1

    def calculate_participation_ratio(self) -> float:
        """
        Participation ratio (PR) measures the effective dimensionality.
        PR = (sum(lambda_i))^2 / sum(lambda_i^2)
        """
        if not HAS_SKLEARN:
            return -1.0
            
        try:
            self._prepare_data()
        except ValueError:
            return -1.0
            
        eigenvalues = self._eigenvalues
        if np.sum(eigenvalues) == 0:
            return 0.0
            
        pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        return float(pr)

    def calculate_entropy_score(self) -> float:
        """
        Calculates entropy of the eigenvalue spectrum. 
        Higher entropy means more uniform variance (higher complexity).
        """
        if not HAS_SKLEARN:
            return -1.0
            
        try:
            self._prepare_data()
        except ValueError:
            return -1.0
            
        # Normalize eigenvalues to act as probabilities
        probs = self._eigenvalues / np.sum(self._eigenvalues)
        return float(entropy(probs))

    def get_feature_redundancy(self) -> float:
        """
        Redundancy is 1.0 - (effective_rank / n_features).
        High redundancy means fewer intrinsic dimensions.
        """
        if not HAS_SKLEARN:
            return -1.0
            
        rank = self.estimate_effective_rank(threshold=0.95)
        if rank <= 0 or self.n_features == 0:
            return 0.0
            
        return max(0.0, 1.0 - (rank / self.n_features))

    def get_complexity_report(self) -> Dict[str, Any]:
        """Returns the full complexity analysis report."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn is required for complexity analysis."}
            
        try:
            self._prepare_data()
            return {
                "n_numeric_features": self.n_features,
                "effective_rank_95": self.estimate_effective_rank(0.95),
                "participation_ratio": self.calculate_participation_ratio(),
                "entropy_score": self.calculate_entropy_score(),
                "feature_redundancy": self.get_feature_redundancy(),
                "pca_spectrum": self.get_pca_spectrum()
            }
        except ValueError as e:
            return {"error": str(e)}
