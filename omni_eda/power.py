from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st

try:
    from statsmodels.stats.power import TTestIndPower
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

class StatisticalPowerAnalysis:
    """Computes effect size, statistical power, and required sample size."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])

    def calculate_cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cohen's d effect size between two arrays."""
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        
        if dof <= 0:
            return 0.0
            
        pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + 
                              (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
        
        if pooled_std == 0:
            return 0.0
            
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    def calculate_power(self, effect_size: float, nobs1: int, alpha: float = 0.05) -> float:
        """Calculates statistical power for a given effect size and sample size."""
        if not HAS_STATSMODELS or effect_size == 0.0:
            return -1.0
            
        analysis = TTestIndPower()
        power = analysis.solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha)
        return float(power)

    def calculate_min_sample_size(self, effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculates required sample size per group to achieve desired power."""
        if not HAS_STATSMODELS or effect_size == 0.0:
            return -1
            
        analysis = TTestIndPower()
        try:
            nobs = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
            return int(np.ceil(nobs))
        except Exception:
            return -1

    def bootstrap_confidence_interval(self, series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculates bootstrap confidence interval for the mean."""
        data = series.dropna().values
        if len(data) < 2:
            return (0.0, 0.0)
            
        res = st.bootstrap((data,), np.mean, confidence_level=confidence, method='BCa')
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))

    def get_report(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns the power analysis report.
        If target_col is provided (and is categorical binary), it calculates
        effect size and power between the two groups for each numeric feature.
        """
        report = {}
        
        for col in self.numeric_df.columns:
            series = self.numeric_df[col]
            ci_low, ci_high = self.bootstrap_confidence_interval(series)
            
            feat_report = {
                "bootstrap_ci_95": {"low": ci_low, "high": ci_high}
            }
            
            # If a binary target is provided, calculate effect size and power
            if target_col and target_col in self.df.columns and target_col != col:
                target_series = self.df[target_col].dropna()
                unique_vals = target_series.unique()
                
                if len(unique_vals) == 2:
                    val1, val2 = unique_vals
                    group1 = self.df[self.df[target_col] == val1][col].dropna().values
                    group2 = self.df[self.df[target_col] == val2][col].dropna().values
                    
                    if len(group1) > 1 and len(group2) > 1:
                        effect_size = self.calculate_cohens_d(group1, group2)
                        
                        feat_report.update({
                            "effect_size_vs_target": effect_size,
                            "current_power_alpha_0_05": self.calculate_power(effect_size, min(len(group1), len(group2))),
                            "required_sample_size_power_0_8": self.calculate_min_sample_size(effect_size, 0.8)
                        })
            
            report[col] = feat_report
            
        return report
