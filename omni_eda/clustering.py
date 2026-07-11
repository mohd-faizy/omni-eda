"""Clustering analysis for features and samples."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    import scipy.cluster.hierarchy as sch
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class FeatureClustering:
    """Clusters features to find redundancy and feature families."""

    def __init__(self, df: pd.DataFrame, correlation_matrix: pd.DataFrame = None):
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.cols = self.numeric_df.columns.tolist()
        self.corr_matrix = correlation_matrix

    def cluster_hierarchical(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Hierarchical clustering based on correlation distances."""
        if not HAS_SKLEARN or self.corr_matrix is None or len(self.cols) < 2:
            return {}

        try:
            # Convert correlation to distance matrix
            dist_matrix = 1 - np.abs(self.corr_matrix)
            dist_matrix = dist_matrix.fillna(0)
            
            # Extract condensed distance matrix for linkage
            import scipy.spatial.distance as ssd
            dist_array = ssd.squareform(dist_matrix)
            
            linkage = sch.linkage(dist_array, method='complete')
            clusters = sch.fcluster(linkage, t=threshold, criterion='distance')
            
            families = {}
            for i, cluster_id in enumerate(clusters):
                c = int(cluster_id)
                if c not in families:
                    families[c] = []
                families[c].append(self.cols[i])
                
            return {
                "n_clusters": len(families),
                "families": {f"Family_{k}": v for k, v in families.items()},
                "linkage": linkage.tolist()
            }
        except Exception as e:
            return {"error": str(e)}

    def get_report(self) -> Dict[str, Any]:
        """Returns the feature clustering report."""
        return {
            "hierarchical_correlation": self.cluster_hierarchical(threshold=0.3)
        }


class SampleClustering:
    """Clusters observations using multiple algorithms."""

    def __init__(self, df: pd.DataFrame):
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.n_samples, self.n_features = self.numeric_df.shape
        self._scaled_data: Optional[np.ndarray] = None

    def _prepare_data(self) -> None:
        if self._scaled_data is not None:
            return
        data = self.numeric_df.fillna(self.numeric_df.mean()).values
        
        # Subsample if large
        if len(data) > 10000:
            np.random.seed(42)
            indices = np.random.choice(len(data), 10000, replace=False)
            data = data[indices]
            
        scaler = StandardScaler()
        self._scaled_data = scaler.fit_transform(data)

    def run_clustering(self) -> Dict[str, Any]:
        if not HAS_SKLEARN or self.n_samples < 10 or self.n_features < 1:
            return {}
            
        self._prepare_data()
        data = self._scaled_data
        
        results = {}
        
        algorithms = {
            "kmeans_3": KMeans(n_clusters=3, random_state=42, n_init='auto'),
            "gmm_3": GaussianMixture(n_components=3, random_state=42),
            "agglomerative_3": AgglomerativeClustering(n_clusters=3)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, algo in algorithms.items():
                try:
                    if hasattr(algo, "fit_predict"):
                        labels = algo.fit_predict(data)
                    else:
                        labels = algo.fit(data).predict(data)
                        
                    n_clusters = len(np.unique(labels))
                    if 1 < n_clusters < len(data):
                        score = silhouette_score(data, labels)
                    else:
                        score = -1.0
                        
                    counts = np.bincount(labels[labels >= 0])
                    
                    results[name] = {
                        "n_clusters": n_clusters,
                        "silhouette_score": float(score),
                        "cluster_sizes": counts.tolist()
                    }
                except Exception:
                    continue
                    
        return results


def compute_hopkins_statistic(
    df: pd.DataFrame, 
    sample_size: int = 500
) -> float | None:
    """Calculate the Hopkins Statistic for clustering tendency."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
        
    data = numeric_df.dropna().values
    n, d = data.shape
    if n < sample_size * 2:
        return None
        
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    data_scaled = (data - mean) / std

    m = min(n // 2, sample_size)
    random_indices = np.random.choice(n, m, replace=False)
    real_sample = data_scaled[random_indices, :]

    mins = np.min(data_scaled, axis=0)
    maxs = np.max(data_scaled, axis=0)
    artificial_sample = np.random.uniform(mins, maxs, (m, d))

    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(data_scaled)
    u_distances, _ = nbrs.kneighbors(artificial_sample, n_neighbors=1)
    u = u_distances[:, 0] ** 2

    w_distances, _ = nbrs.kneighbors(real_sample, n_neighbors=2)
    w = w_distances[:, 1] ** 2

    sum_u = np.sum(u)
    sum_w = np.sum(w)
    
    if (sum_u + sum_w) == 0:
        return None

    return float(sum_u / (sum_u + sum_w))
