import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.signal import find_peaks

class DistributionDiagnostics:
    """Advanced distribution diagnostics and fitting for numerical features."""

    def __init__(self, df: pd.DataFrame):
        self.numeric_df = df.select_dtypes(include=[np.number])

    def detect_modality(self, series: pd.Series) -> Dict[str, Any]:
        """Detects if a distribution is unimodal, bimodal, or multimodal."""
        data = series.dropna().values
        if len(data) < 10:
            return {"modality": "unknown", "n_peaks": 0}

        try:
            kde = st.gaussian_kde(data)
            x_eval = np.linspace(data.min(), data.max(), 200)
            y_eval = kde(x_eval)
            
            # Find peaks
            peaks, _ = find_peaks(y_eval, prominence=np.max(y_eval)*0.05)
            n_peaks = len(peaks)
            
            if n_peaks == 1:
                mod = "unimodal"
            elif n_peaks == 2:
                mod = "bimodal"
            elif n_peaks > 2:
                mod = "multimodal"
            else:
                mod = "unknown"
                
            return {"modality": mod, "n_peaks": n_peaks}
        except Exception:
            return {"modality": "error", "n_peaks": -1}

    def check_zero_inflation(self, series: pd.Series, threshold: float = 0.15) -> bool:
        """Checks if there's an unusually high number of exact zeros."""
        data = series.dropna()
        if len(data) == 0:
            return False
        zero_prop = (data == 0).sum() / len(data)
        return bool(zero_prop > threshold)

    def check_heavy_tailed(self, series: pd.Series) -> bool:
        """Checks for heavy tails using excess kurtosis."""
        data = series.dropna()
        if len(data) < 10:
            return False
        kurtosis = st.kurtosis(data, fisher=True) # Fisher's is excess kurtosis
        return bool(kurtosis > 3.0)

    def fit_best_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """
        Fits common continuous distributions and finds the best fit based on AIC.
        Includes normal, uniform, exponential, gamma, beta, lognormal.
        """
        data = series.dropna().values
        if len(data) < 20:
            return {"error": "Not enough data to fit distributions."}

        # Sub-sample if data is too large for fast fitting
        if len(data) > 5000:
            np.random.seed(42)
            data = np.random.choice(data, size=5000, replace=False)

        distributions = {
            "norm": st.norm,
            "uniform": st.uniform,
            "expon": st.expon,
            "gamma": st.gamma,
            "beta": st.beta,
            "lognorm": st.lognorm
        }

        best_dist = None
        best_aic = np.inf
        best_bic = np.inf
        best_ks = np.inf
        best_params = {}
        results = []

        n = len(data)

        # Suppress warnings from scipy fitting on bad data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, dist in distributions.items():
                try:
                    # Some distributions need strictly positive data
                    if name in ["gamma", "lognorm"] and data.min() <= 0:
                        shifted_data = data - data.min() + 1e-5
                        params = dist.fit(shifted_data)
                        loglik = np.sum(dist.logpdf(shifted_data, *params))
                        # KS test
                        D, _ = st.kstest(shifted_data, name, args=params)
                    elif name == "beta":
                        # Beta needs data in [0, 1]
                        d_min, d_max = data.min(), data.max()
                        if d_min == d_max: continue
                        scaled_data = (data - d_min) / (d_max - d_min)
                        # Avoid exact 0 or 1
                        scaled_data = np.clip(scaled_data, 1e-5, 1 - 1e-5)
                        params = dist.fit(scaled_data)
                        loglik = np.sum(dist.logpdf(scaled_data, *params))
                        D, _ = st.kstest(scaled_data, name, args=params)
                    else:
                        params = dist.fit(data)
                        loglik = np.sum(dist.logpdf(data, *params))
                        D, _ = st.kstest(data, name, args=params)
                        
                    k = len(params)
                    aic = 2 * k - 2 * loglik
                    bic = k * np.log(n) - 2 * loglik

                    results.append({
                        "distribution": name,
                        "aic": aic,
                        "bic": bic,
                        "ks_stat": D,
                        "params": params
                    })

                    if aic < best_aic:
                        best_aic = aic
                        best_bic = bic
                        best_dist = name
                        best_ks = D
                        best_params = params

                except Exception:
                    continue

        if not best_dist:
            return {"error": "Failed to fit distributions."}

        return {
            "best_distribution": best_dist,
            "best_aic": best_aic,
            "best_bic": best_bic,
            "best_ks_stat": best_ks,
            "best_params": best_params,
            "all_results": [
                {k: v for k, v in r.items() if k != 'params'} for r in results
            ]
        }

    def get_report(self) -> Dict[str, Any]:
        """Returns the distribution diagnostics report."""
        report = {}
        for col in self.numeric_df.columns:
            series = self.numeric_df[col]
            modality = self.detect_modality(series)
            
            report[col] = {
                "modality": modality["modality"],
                "n_peaks": modality["n_peaks"],
                "zero_inflated": self.check_zero_inflation(series),
                "heavy_tailed": self.check_heavy_tailed(series),
                "fit": self.fit_best_distribution(series)
            }
        return report
