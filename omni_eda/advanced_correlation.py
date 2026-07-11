from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from minepy import MINE
    HAS_MINEPY = True
except ImportError:
    HAS_MINEPY = False

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

class AdvancedCorrelation:
    """Computes advanced nonlinear and partial correlations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.cols = self.numeric_df.columns.tolist()

    def _compute_blomqvist_beta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Blomqvist's beta (medial correlation)."""
        x_med = np.median(x)
        y_med = np.median(y)
        
        # Avoid exact medians for stability
        x_diff = x - x_med
        y_diff = y - y_med
        
        # Remove points that are exactly on the median
        valid = (x_diff != 0) & (y_diff != 0)
        if not np.any(valid):
            return 0.0
            
        x_diff = x_diff[valid]
        y_diff = y_diff[valid]
        
        concordant = np.sum((x_diff > 0) == (y_diff > 0))
        discordant = np.sum((x_diff > 0) != (y_diff > 0))
        
        return float((concordant - discordant) / (concordant + discordant))

    def _compute_rv_coefficient(self, X: np.ndarray, Y: np.ndarray) -> float:
        """RV coefficient between two sets of variables (here 1D)."""
        # Center the vectors
        X_c = X - np.mean(X)
        Y_c = Y - np.mean(Y)
        
        # Covariance matrices (1x1 for 1D)
        cov_xx = np.dot(X_c, X_c)
        cov_yy = np.dot(Y_c, Y_c)
        cov_xy = np.dot(X_c, Y_c)
        
        if cov_xx == 0 or cov_yy == 0:
            return 0.0
            
        rv = (cov_xy ** 2) / (cov_xx * cov_yy)
        return float(rv)

    def compute_mic_matrix(self) -> pd.DataFrame:
        """Maximal Information Coefficient matrix."""
        n = len(self.cols)
        matrix = pd.DataFrame(np.eye(n), index=self.cols, columns=self.cols)
        
        if not HAS_MINEPY or n < 2:
            return matrix
            
        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        
        for i, col1 in enumerate(self.cols):
            x = self.numeric_df[col1].dropna().values
            for j, col2 in enumerate(self.cols[i+1:], start=i+1):
                y = self.numeric_df[col2].dropna().values
                
                # Need to align indices if we dropped NaNs independently
                df_pair = self.numeric_df[[col1, col2]].dropna()
                if len(df_pair) < 2:
                    continue
                    
                mine.compute_score(df_pair[col1].values, df_pair[col2].values)
                score = mine.mic()
                matrix.loc[col1, col2] = score
                matrix.loc[col2, col1] = score
                
        return matrix

    def compute_partial_correlations(self) -> pd.DataFrame:
        """Partial correlation matrix using pingouin."""
        n = len(self.cols)
        matrix = pd.DataFrame(np.eye(n), index=self.cols, columns=self.cols)
        
        if not HAS_PINGOUIN or n < 3:
            return matrix
            
        try:
            # pcorr() computes partial correlations
            matrix = self.numeric_df.pcorr()
        except Exception:
            pass
            
        return matrix
        
    def compute_biweight_midcorrelation(self) -> pd.DataFrame:
        """Biweight midcorrelation matrix using pingouin."""
        n = len(self.cols)
        matrix = pd.DataFrame(np.eye(n), index=self.cols, columns=self.cols)
        
        if not HAS_PINGOUIN or n < 2:
            return matrix
            
        for i, col1 in enumerate(self.cols):
            for j, col2 in enumerate(self.cols[i+1:], start=i+1):
                try:
                    df_pair = self.numeric_df[[col1, col2]].dropna()
                    if len(df_pair) < 3:
                        continue
                    res = pg.corr(df_pair[col1], df_pair[col2], method='bicor')
                    val = float(res['r'].values[0])
                    matrix.loc[col1, col2] = val
                    matrix.loc[col2, col1] = val
                except Exception:
                    pass
        return matrix

    def compute_blomqvist_matrix(self) -> pd.DataFrame:
        """Blomqvist's beta matrix."""
        n = len(self.cols)
        matrix = pd.DataFrame(np.eye(n), index=self.cols, columns=self.cols)
        
        for i, col1 in enumerate(self.cols):
            for j, col2 in enumerate(self.cols[i+1:], start=i+1):
                df_pair = self.numeric_df[[col1, col2]].dropna()
                if len(df_pair) < 3:
                    continue
                val = self._compute_blomqvist_beta(df_pair[col1].values, df_pair[col2].values)
                matrix.loc[col1, col2] = val
                matrix.loc[col2, col1] = val
                
        return matrix

    def get_report(self) -> Dict[str, Any]:
        """Returns the advanced correlations report."""
        if self.numeric_df.empty or len(self.cols) < 2:
            return {}
            
        return {
            "mic": self.compute_mic_matrix().to_dict() if HAS_MINEPY else {},
            "partial": self.compute_partial_correlations().to_dict() if HAS_PINGOUIN else {},
            "bicor": self.compute_biweight_midcorrelation().to_dict() if HAS_PINGOUIN else {},
            "blomqvist": self.compute_blomqvist_matrix().to_dict()
        }
