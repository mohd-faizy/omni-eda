"""Time Series Anomaly and Change Point Detection Module.

Uses the 'ruptures' library (if installed) to find structural breaks
in time series data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from omni_eda.config import EDAConfig
from omni_eda.detection import ColumnProfile
from omni_eda.logger import get_logger

logger = get_logger()


def detect_changepoints(
    df: pd.DataFrame, 
    profiles: dict[str, ColumnProfile],
    config: EDAConfig
) -> dict[str, Any] | None:
    """Identify structural breaks in numeric series over time."""
    try:
        import ruptures as rpt
    except ImportError:
        logger.warning("Optional dependency 'ruptures' not installed. Skipping change point detection.")
        return None

    # Find a datetime column
    date_cols = [c for c, p in profiles.items() if p.is_datetime and c in df.columns]
    if not date_cols:
        return None
        
    date_col = date_cols[0]  # Just use the first one as the primary index
    
    numeric_cols = [
        c for c, p in profiles.items() 
        if p.is_numeric and not p.is_constant and not config.is_ignored(c) and c in df.columns
    ]
    
    if not numeric_cols:
        return None
        
    # Sort by date
    ts_df = df[[date_col] + numeric_cols].dropna(subset=[date_col]).sort_values(by=date_col)
    
    results = {}
    
    for metric in numeric_cols:
        series = pd.to_numeric(ts_df[metric], errors="coerce").dropna()
        if len(series) < 100:  # Need enough data for reliable breaks
            continue
            
        signal = series.values
        
        # Use Pelt search method for unknown number of change points
        try:
            algo = rpt.Pelt(model="rbf").fit(signal)
            # Penalty value - controls sensitivity. Higher penalty = fewer points.
            penalty = np.log(len(signal)) * 3 * np.std(signal)**2 if np.std(signal) > 0 else 10
            
            # Predict breakpoints (indices)
            breakpoints = algo.predict(pen=penalty)
            
            # Remove the last breakpoint which is always the end of the series
            if breakpoints and breakpoints[-1] == len(signal):
                breakpoints = breakpoints[:-1]
                
            if breakpoints:
                # Map indices back to dates
                dates = ts_df.iloc[series.index.intersection(ts_df.index)][date_col].iloc[breakpoints].tolist()
                
                results[metric] = {
                    "n_breaks": len(breakpoints),
                    "break_dates": [str(d) for d in dates]
                }
        except Exception as e:
            logger.debug(f"Failed to detect changepoints for {metric}: {e}")
            
    if not results:
        return None
        
    return {
        "time_column": date_col,
        "changepoints": results
    }
