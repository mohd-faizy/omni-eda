from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SyntheticDataDiagnostics:
    """Evaluates privacy risks, memorization risks, and synthesis difficulty of a dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_samples = len(df)
        self.numeric_df = df.select_dtypes(include=[np.number])

    def calculate_uniqueness(self) -> float:
        """Percentage of rows that are completely unique across all columns."""
        if self.n_samples == 0:
            return 0.0
        # Count rows that occur exactly once
        unique_counts = self.df.value_counts(dropna=False)
        strictly_unique = (unique_counts == 1).sum()
        return float(strictly_unique / self.n_samples)

    def calculate_memorization_risk(self) -> float:
        """Percentage of rows that are duplicated (indicating potential memorization)."""
        if self.n_samples == 0:
            return 0.0
        duplicates = self.df.duplicated(keep=False).sum()
        return float(duplicates / self.n_samples)

    def calculate_nearest_neighbor_distance(self) -> float:
        """Average distance to the 1st nearest neighbor (excluding self)."""
        if not HAS_SKLEARN or self.n_samples < 2 or self.numeric_df.empty:
            return -1.0
            
        try:
            # Drop rows with NaNs for distance calculation
            data = self.numeric_df.dropna().values
            if len(data) < 2:
                return -1.0
                
            # Sample if data is too large to keep it fast
            if len(data) > 10000:
                np.random.seed(42)
                indices = np.random.choice(len(data), 10000, replace=False)
                data = data[indices]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Find 1st neighbor (k=2 because 1st is the point itself)
            nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
            nn.fit(scaled_data)
            distances, _ = nn.kneighbors(scaled_data)
            
            # Mean distance to the nearest neighbor
            return float(np.mean(distances[:, 1]))
        except Exception:
            return -1.0

    def calculate_privacy_score(self, uniqueness: float, nn_distance: float) -> float:
        """
        Heuristic privacy score (0 to 1). 
        1.0 means highly private (hard to identify individuals).
        0.0 means high privacy risk (highly unique, very sparse).
        """
        # High uniqueness -> lower privacy
        # High nn_distance -> sparser space -> lower privacy (easier to isolate)
        
        # Base score on non-uniqueness
        score = 1.0 - uniqueness
        
        # Penalize if nn_distance is very large (outliers are easily re-identified)
        if nn_distance > 0:
            # Empirical scaling factor, assume distance > 3 is highly isolated
            isolation_penalty = min(0.5, nn_distance / 10.0) 
            score = max(0.0, score - isolation_penalty)
            
        return score

    def estimate_synthesis_difficulty(self, uniqueness: float, memorization: float, n_cols: int) -> str:
        """Categorical estimation of how hard it is to train a GAN/Diffusion model on this."""
        if self.n_samples < 100:
            return "Extreme (Too few samples)"
        if n_cols > 100 and uniqueness > 0.99:
            return "High (High dimensionality and uniqueness)"
        if memorization > 0.5:
            return "Medium (High duplication might lead to mode collapse)"
        if uniqueness < 0.1:
            return "Low (Highly repetitive data)"
        
        return "Moderate"

    def get_report(self) -> Dict[str, Any]:
        """Returns the full synthetic data diagnostics report."""
        uniqueness = self.calculate_uniqueness()
        memorization = self.calculate_memorization_risk()
        nn_distance = self.calculate_nearest_neighbor_distance()
        privacy_score = self.calculate_privacy_score(uniqueness, nn_distance)
        synthesis_difficulty = self.estimate_synthesis_difficulty(uniqueness, memorization, len(self.df.columns))
        
        return {
            "uniqueness_ratio": uniqueness,
            "memorization_risk_ratio": memorization,
            "avg_nearest_neighbor_distance": nn_distance,
            "privacy_score": privacy_score,
            "synthesis_difficulty": synthesis_difficulty
        }
