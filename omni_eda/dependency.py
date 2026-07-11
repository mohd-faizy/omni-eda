import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

class DependencyDiscovery:
    """Discovers directional dependencies between features."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cols = self.df.columns.tolist()

    def discover_tree_dependencies(self, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Tests if Feature A can predict Feature B using a single-feature Decision Tree.
        Returns a list of dependencies with their predictive strength.
        """
        dependencies = []
        
        # We need a clean version of the data
        clean_df = self.df.copy()
        encoders = {}
        
        for col in self.cols:
            if not pd.api.types.is_numeric_dtype(clean_df[col]):
                le = LabelEncoder()
                # Treat NaN as a separate category if any
                clean_col = clean_df[col].astype(str)
                clean_df[col] = le.fit_transform(clean_col)
                encoders[col] = le
            else:
                # Fill missing for quick tree building
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())

        # For large data, sub-sample for speed
        if len(clean_df) > 5000:
            clean_df = clean_df.sample(n=5000, random_state=42)

        for col_b in self.cols:
            y = clean_df[col_b].values
            is_classification = col_b in encoders or clean_df[col_b].nunique() < 20
            
            for col_a in self.cols:
                if col_a == col_b:
                    continue
                    
                X = clean_df[[col_a]].values
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if is_classification:
                        model = DecisionTreeClassifier(max_depth=5, random_state=42)
                        # CV using 3 folds, KFold ignores class labels for splitting
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)
                        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                    else:
                        model = DecisionTreeRegressor(max_depth=5, random_state=42)
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)
                        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                        
                avg_score = np.mean(scores)
                
                if avg_score >= min_score:
                    dependencies.append({
                        "predictor (A)": col_a,
                        "target (B)": col_b,
                        "dependency_strength": float(avg_score),
                        "type": "classification (accuracy)" if is_classification else "regression (R2)"
                    })
                    
        return sorted(dependencies, key=lambda x: x['dependency_strength'], reverse=True)

    def discover_association_rules(self, min_support: float = 0.05, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Discovers association rules for categorical columns (simplified).
        """
        # Select categorical columns
        cat_cols = [c for c in self.cols if not pd.api.types.is_numeric_dtype(self.df[c]) or self.df[c].nunique() < 10]
        if len(cat_cols) < 2:
            return []

        rules = []
        n = len(self.df)
        
        # We only do 1-to-1 rules to keep it simple and fast without external heavy libraries
        for col_a in cat_cols:
            for col_b in cat_cols:
                if col_a == col_b:
                    continue
                    
                # Calculate joint frequencies
                joint = pd.crosstab(self.df[col_a], self.df[col_b])
                
                for val_a in joint.index:
                    support_a = self.df[col_a].value_counts().get(val_a, 0) / n
                    if support_a < min_support:
                        continue
                        
                    for val_b in joint.columns:
                        support_b = self.df[col_b].value_counts().get(val_b, 0) / n
                        support_ab = joint.loc[val_a, val_b] / n
                        
                        if support_ab < min_support:
                            continue
                            
                        confidence = support_ab / support_a
                        if confidence >= min_confidence:
                            lift = confidence / support_b if support_b > 0 else 0
                            
                            rules.append({
                                "rule": f"IF {col_a}='{val_a}' THEN {col_b}='{val_b}'",
                                "support": float(support_ab),
                                "confidence": float(confidence),
                                "lift": float(lift)
                            })
                            
        return sorted(rules, key=lambda x: x['lift'], reverse=True)

    def get_report(self) -> Dict[str, Any]:
        """Returns the dependency discovery report."""
        return {
            "tree_dependencies": self.discover_tree_dependencies(),
            "association_rules": self.discover_association_rules()
        }
