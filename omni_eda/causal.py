import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.FCMBased import lingam
    from causallearn.utils.GraphUtils import GraphUtils
    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False


class CausalDiscovery:
    """Discovers causal relationships using various algorithms (PC, LiNGAM)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.cols = self.numeric_df.columns.tolist()

    def discover_pc_algorithm(self) -> Dict[str, Any]:
        """Runs the PC Algorithm for causal discovery."""
        if not HAS_CAUSAL_LEARN or len(self.cols) < 2 or len(self.numeric_df) < 50:
            return {"error": "causal-learn is not installed or insufficient data."}

        # Subsample to keep it tractable
        data = self.numeric_df.dropna().values
        if len(data) > 2000:
            np.random.seed(42)
            indices = np.random.choice(len(data), 2000, replace=False)
            data = data[indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # PC algorithm with Fisher-Z test
                cg = pc(data, 0.05, "fisherz", node_names=self.cols, show_progress=False)
                
            # Parse graph edges
            edges = []
            if cg.G is not None:
                graph_nodes = cg.G.get_nodes()
                for i in range(len(graph_nodes)):
                    for j in range(len(graph_nodes)):
                        if i == j:
                            continue
                        edge = cg.G.get_edge(graph_nodes[i], graph_nodes[j])
                        if edge is not None:
                            # edge type can be directed (-->), undirected (---), etc.
                            edges.append({
                                "source": self.cols[i],
                                "target": self.cols[j],
                                "type": "directed" if edge.get_endpoint1() == 1 else "undirected"
                            })
                            
            return {
                "algorithm": "PC Algorithm",
                "edges": edges,
                "confidence_alpha": 0.05,
                "limitations": "Assumes no latent confounders (Causal Sufficiency) and linear Gaussian relations."
            }
        except Exception as e:
            return {"error": str(e)}

    def discover_lingam(self) -> Dict[str, Any]:
        """Runs DirectLiNGAM for non-Gaussian causal discovery."""
        if not HAS_CAUSAL_LEARN or len(self.cols) < 2 or len(self.numeric_df) < 50:
            return {"error": "causal-learn is not installed or insufficient data."}

        data = self.numeric_df.dropna().values
        if len(data) > 2000:
            np.random.seed(42)
            indices = np.random.choice(len(data), 2000, replace=False)
            data = data[indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = lingam.DirectLiNGAM()
                model.fit(data)
                
            adjacency_matrix = model.adjacency_matrix_
            edges = []
            
            for i in range(len(self.cols)):
                for j in range(len(self.cols)):
                    if i == j:
                        continue
                    weight = adjacency_matrix[i, j]
                    if abs(weight) > 0.01:
                        edges.append({
                            "source": self.cols[j], # LiNGAM matrix is typically (target, source)
                            "target": self.cols[i],
                            "weight": float(weight),
                            "type": "directed"
                        })
                        
            return {
                "algorithm": "DirectLiNGAM",
                "edges": edges,
                "limitations": "Assumes non-Gaussian continuous variables and acyclic structure."
            }
        except Exception as e:
            return {"error": str(e)}

    def get_report(self) -> Dict[str, Any]:
        """Returns the causal discovery report."""
        return {
            "pc_algorithm": self.discover_pc_algorithm(),
            "lingam": self.discover_lingam()
        }
