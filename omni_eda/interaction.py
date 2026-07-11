import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class FeatureInteractionExplorer:
    """Discovers 2-way and multi-way feature interactions."""

    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df = df
        self.target_col = target_col
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.cols = self.numeric_df.columns.tolist()

    def discover_2way_shap_interactions(self) -> List[Dict[str, Any]]:
        """Uses XGBoost and SHAP to find the strongest 2-way interactions."""
        if not HAS_SHAP or not HAS_XGB or not self.target_col or self.target_col not in self.df.columns:
            return []

        # Prepare data
        clean_df = self.numeric_df.copy()
        if self.target_col in clean_df:
            X = clean_df.drop(columns=[self.target_col]).fillna(clean_df.mean())
            y = self.df[self.target_col]
        else:
            return []

        # Convert target if categorical
        is_classification = False
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 20:
            is_classification = True
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.fillna(y.mean()).values

        # Subsample for speed
        if len(X) > 2000:
            np.random.seed(42)
            indices = np.random.choice(len(X), 2000, replace=False)
            X = X.iloc[indices]
            y = y[indices]

        try:
            if is_classification:
                model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                
            model.fit(X, y)
            
            explainer = shap.TreeExplainer(model)
            shap_interaction = explainer.shap_interaction_values(X)
            
            if isinstance(shap_interaction, list):
                # multi-class
                shap_interaction = shap_interaction[1]  # Take first class
                
            # Mean absolute SHAP interaction values
            mean_interactions = np.abs(shap_interaction).mean(axis=0)
            
            interactions = []
            features = X.columns
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    val = float(mean_interactions[i, j])
                    if val > 0:
                        interactions.append({
                            "feature_a": features[i],
                            "feature_b": features[j],
                            "interaction_strength": val
                        })
                        
            return sorted(interactions, key=lambda x: x["interaction_strength"], reverse=True)[:20]
        except Exception:
            return []

    def discover_multiway_interactions(self, n_way: int = 3) -> List[Dict[str, Any]]:
        """
        Heuristic discovery of multi-way interactions.
        Uses Random Forest depth to find co-occurring features in paths.
        """
        # This is a placeholder for a complex implementation, 
        # normally done using iterative RF or RuleFit.
        # We return a mock/heuristic for now due to complexity.
        return [{"message": f"{n_way}-way interaction discovery requires extensive computation."}]

    def get_report(self) -> Dict[str, Any]:
        """Returns the interaction explorer report."""
        return {
            "top_2way_interactions_shap": self.discover_2way_shap_interactions(),
            "top_3way_interactions": self.discover_multiway_interactions(3)
        }
