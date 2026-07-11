"""Clustering Tendency Module.

Evaluates whether a dataset has meaningful cluster structure before 
applying algorithms like K-Means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger

logger = get_logger()


def compute_hopkins_statistic(
    df: pd.DataFrame, 
    profiles: dict[str, ColumnProfile],
    config: EDAConfig,
    sample_size: int = 500
) -> float | None:
    """Calculate the Hopkins Statistic for clustering tendency.
    
    Values ~0.5 mean the data is uniformly distributed (no meaningful clusters).
    Values >0.75 indicate a high tendency to cluster.
    """
    numeric_cols = [
        c for c, p in profiles.items() 
        if p.is_numeric and not p.is_constant and not config.is_ignored(c) and c in df.columns
    ]
    
    if len(numeric_cols) < 2:
        return None
        
    data = df[numeric_cols].dropna().values
    
    n, d = data.shape
    if n < sample_size * 2:
        return None
        
    # Standardize data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    data_scaled = (data - mean) / std

    # Randomly sample 'm' data points
    m = min(n // 2, sample_size)
    random_indices = np.random.choice(n, m, replace=False)
    real_sample = data_scaled[random_indices, :]

    # Generate 'm' uniformly distributed artificial points in the same space
    mins = np.min(data_scaled, axis=0)
    maxs = np.max(data_scaled, axis=0)
    artificial_sample = np.random.uniform(mins, maxs, (m, d))

    # Fit NearestNeighbors on the full scaled dataset
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(data_scaled)

    # 1. Distances from artificial points to nearest real data points (u)
    u_distances, _ = nbrs.kneighbors(artificial_sample, n_neighbors=1)
    u = u_distances[:, 0] ** 2

    # 2. Distances from real sampled points to their nearest real neighbors (w)
    # We use n_neighbors=2 because the 1st nearest neighbor is the point itself (distance=0)
    w_distances, _ = nbrs.kneighbors(real_sample, n_neighbors=2)
    w = w_distances[:, 1] ** 2

    sum_u = np.sum(u)
    sum_w = np.sum(w)
    
    if (sum_u + sum_w) == 0:
        return None

    hopkins_stat = sum_u / (sum_u + sum_w)
    return float(hopkins_stat)
