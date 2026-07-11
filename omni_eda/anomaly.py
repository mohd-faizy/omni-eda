import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class AnomalyAnalysis:
    """Advanced anomaly detection including density, contextual, and ensemble methods."""

    def __init__(self, df: pd.DataFrame, contamination: float = 0.05):
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.n_samples, self.n_features = self.numeric_df.shape
        self.contamination = contamination
        self._scaled_data = None

    def _prepare_data(self) -> None:
        if self._scaled_data is not None:
            return
            
        data = self.numeric_df.fillna(self.numeric_df.mean()).values
        
        # Subsample if large for performance
        if len(data) > 10000:
            np.random.seed(42)
            indices = np.random.choice(len(data), 10000, replace=False)
            data = data[indices]
            
        scaler = StandardScaler()
        self._scaled_data = scaler.fit_transform(data)

    def detect_density_anomalies(self) -> np.ndarray:
        """Local Outlier Factor (density-based)."""
        if not HAS_SKLEARN or self.n_samples < 20 or self.n_features < 1:
            return np.array([])
            
        self._prepare_data()
        lof = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        return lof.fit_predict(self._scaled_data)

    def detect_isolation_anomalies(self) -> np.ndarray:
        """Isolation Forest (tree-based)."""
        if not HAS_SKLEARN or self.n_samples < 20 or self.n_features < 1:
            return np.array([])
            
        self._prepare_data()
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        return iso.fit_predict(self._scaled_data)

    def detect_svm_anomalies(self) -> np.ndarray:
        """One-Class SVM (boundary-based)."""
        if not HAS_SKLEARN or self.n_samples < 20 or self.n_features < 1:
            return np.array([])
            
        self._prepare_data()
        svm = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
        return svm.fit_predict(self._scaled_data)

    def get_ensemble_scores(self) -> Dict[str, Any]:
        """Calculates a consensus anomaly score from multiple models."""
        if not HAS_SKLEARN or self.n_samples < 20 or self.n_features < 1:
            return {"error": "Insufficient data or scikit-learn missing."}

        # Predictions: -1 for outlier, 1 for inlier
        preds_lof = self.detect_density_anomalies()
        preds_iso = self.detect_isolation_anomalies()
        preds_svm = self.detect_svm_anomalies()

        if len(preds_lof) == 0:
            return {}

        # Convert to 1 (outlier) and 0 (inlier)
        out_lof = (preds_lof == -1).astype(int)
        out_iso = (preds_iso == -1).astype(int)
        out_svm = (preds_svm == -1).astype(int)

        consensus_score = out_lof + out_iso + out_svm
        
        # Collective anomalies: flagged by all 3
        strong_anomalies = int(np.sum(consensus_score == 3))
        moderate_anomalies = int(np.sum(consensus_score == 2))
        weak_anomalies = int(np.sum(consensus_score == 1))
        
        return {
            "total_analyzed": len(preds_lof),
            "strong_anomalies": strong_anomalies,
            "moderate_anomalies": moderate_anomalies,
            "weak_anomalies": weak_anomalies,
            "anomaly_rate": float(strong_anomalies / len(preds_lof)),
            "methods_used": ["LocalOutlierFactor", "IsolationForest", "OneClassSVM"]
        }

    def get_report(self) -> Dict[str, Any]:
        return {
            "ensemble_anomaly_analysis": self.get_ensemble_scores()
        }
