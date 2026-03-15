"""
Multi-Objective Optimization for Perovskites
=============================================

Optimize multiple properties simultaneously:
- Bandgap (target matching)
- Stability (tolerance factor)
- Synthesizability (decomposition energy proxy)
- Cost (material cost estimate)

Features:
- Pareto front calculation
- Weighted scalarization
- 2D and 3D Pareto visualization
- User-defined weights/priorities

Author: OpenClaw Agent
Date: 2026-03-15 (V5)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull

from ml_models import CompositionFeaturizer


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for perovskite materials.
    
    Objectives:
    1. Bandgap match: |Eg - target| → minimize
    2. Stability: tolerance factor deviation from ideal (0.9-1.0) → minimize
    3. Synthesizability: decomposition energy proxy → minimize
    4. Cost: raw material cost → minimize
    """
    
    # Material costs ($/kg, rough estimates)
    MATERIAL_COSTS = {
        'MA': 500,   # Methylammonium iodide
        'FA': 600,   # Formamidinium iodide
        'Cs': 2000,  # Cesium
        'Rb': 1500,  # Rubidium
        'K': 100,    # Potassium
        'Pb': 5,     # Lead
        'Sn': 20,    # Tin
        'Ge': 1500,  # Germanium
        'I': 50,     # Iodine
        'Br': 30,    # Bromine
        'Cl': 10,    # Chlorine
        'F': 20      # Fluorine
    }
    
    def __init__(self, target_bandgap: float = 1.68):
        """
        Args:
            target_bandgap: Target bandgap (eV)
        """
        self.target_bandgap = target_bandgap
        self.featurizer = CompositionFeaturizer()
    
    def evaluate_objectives(self, formulas: List[str], 
                           bandgaps: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate all objectives for list of compositions.
        
        Args:
            formulas: List of chemical formulas
            bandgaps: Optional array of bandgaps (if already predicted)
        
        Returns:
            DataFrame with all objective values
        """
        results = []
        
        for i, formula in enumerate(formulas):
            # Parse composition
            comp = self.featurizer._parse_composition(formula)
            features = self.featurizer.featurize(formula)
            
            # Get bandgap
            if bandgaps is not None:
                bandgap = bandgaps[i]
            else:
                bandgap = np.nan
            
            # Objective 1: Bandgap match
            obj_bandgap = abs(bandgap - self.target_bandgap) if not np.isnan(bandgap) else np.nan
            
            # Objective 2: Stability (tolerance factor)
            tolerance = features[16]  # tolerance_factor feature
            obj_stability = abs(tolerance - 0.95)  # Ideal tolerance factor ≈ 0.95
            
            # Objective 3: Synthesizability (decomposition energy proxy)
            # Use A-site mixing entropy as proxy: higher entropy = harder to synthesize
            a_entropy = features[4]  # A_mixing_entropy
            obj_synthesizability = a_entropy
            
            # Objective 4: Cost
            obj_cost = self._estimate_cost(comp)
            
            results.append({
                'formula': formula,
                'bandgap': bandgap,
                'tolerance_factor': tolerance,
                'obj_bandgap_match': obj_bandgap,
                'obj_stability': obj_stability,
                'obj_synthesizability': obj_synthesizability,
                'obj_cost': obj_cost
            })
        
        return pd.DataFrame(results)
    
    def _estimate_cost(self, comp: Dict[str, Dict[str, float]]) -> float:
        """
        Estimate raw material cost ($/kg) from composition.
        Weighted average of elemental costs.
        """
        total_cost = 0.0
        total_fraction = 0.0
        
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                cost = self.MATERIAL_COSTS.get(elem, 1000)  # Default high cost for unknown
                total_cost += frac * cost
                total_fraction += frac
        
        if total_fraction > 0:
            return total_cost / total_fraction
        else:
            return 1000  # Fallback
    
    def calculate_pareto_front(self, df: pd.DataFrame, 
                               objectives: List[str]) -> pd.DataFrame:
        """
        Calculate Pareto-optimal solutions.
        
        Args:
            df: DataFrame with objective columns
            objectives: List of objective column names (all minimization)
        
        Returns:
            DataFrame with only Pareto-optimal rows
        """
        # Extract objective matrix
        obj_matrix = df[objectives].values
        
        # Find Pareto-optimal points
        is_pareto = self._is_pareto_efficient(obj_matrix)
        
        # Return Pareto front
        pareto_df = df[is_pareto].copy()
        pareto_df['pareto_optimal'] = True
        
        return pareto_df
    
    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto-efficient points (minimization).
        A point is Pareto-efficient if no other point dominates it.
        
        Args:
            costs: (n_points, n_objectives) matrix
        
        Returns:
            Boolean array (True = Pareto-efficient)
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Point i is dominated if any other point is better in all objectives
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                ) | np.all(costs[is_efficient] <= c, axis=1)
                is_efficient[i] = True  # Keep current point
        
        return is_efficient
    
    def weighted_scalarization(self, df: pd.DataFrame, 
                               weights: Dict[str, float]) -> pd.DataFrame:
        """
        Combine multiple objectives into single weighted score.
        
        Args:
            df: DataFrame with objective columns
            weights: Dict mapping objective names to weights (sum to 1)
        
        Returns:
            DataFrame with 'weighted_score' column (lower = better)
        """
        df_copy = df.copy()
        
        # Normalize objectives to [0, 1]
        normalized_objs = {}
        
        for obj, weight in weights.items():
            if obj in df_copy.columns:
                values = df_copy[obj].values
                # Min-max normalization
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)
                
                normalized_objs[obj] = normalized * weight
        
        # Weighted sum
        df_copy['weighted_score'] = sum(normalized_objs.values())
        
        return df_copy.sort_values('weighted_score')
    
    def plot_pareto_front_2d(self, df: pd.DataFrame, 
                            obj_x: str, obj_y: str,
                            pareto_df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Plot 2D Pareto front.
        
        Args:
            df: Full DataFrame
            obj_x: X-axis objective
            obj_y: Y-axis objective
            pareto_df: Pareto-optimal subset (if already computed)
        
        Returns:
            Plotly figure
        """
        if pareto_df is None:
            pareto_df = self.calculate_pareto_front(df, [obj_x, obj_y])
        
        fig = go.Figure()
        
        # All points
        fig.add_trace(go.Scatter(
            x=df[obj_x],
            y=df[obj_y],
            mode='markers',
            name='All Materials',
            marker=dict(
                size=8,
                color='lightblue',
                opacity=0.5,
                line=dict(width=0.5, color='#333')
            ),
            text=df['formula'],
            hovertemplate='<b>%{text}</b><br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>'
        ))
        
        # Pareto front
        if not pareto_df.empty:
            # Sort Pareto points for line connection
            pareto_sorted = pareto_df.sort_values(obj_x)
            
            fig.add_trace(go.Scatter(
                x=pareto_sorted[obj_x],
                y=pareto_sorted[obj_y],
                mode='markers+lines',
                name='Pareto Front',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                line=dict(color='red', width=2, dash='dash'),
                text=pareto_sorted['formula'],
                hovertemplate='<b>PARETO: %{text}</b><br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"2D Pareto Front: {obj_x} vs {obj_y}",
            xaxis_title=obj_x.replace('obj_', '').replace('_', ' ').title(),
            yaxis_title=obj_y.replace('obj_', '').replace('_', ' ').title(),
            height=500,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a2e'),
            hovermode='closest'
        )
        
        return fig
    
    def plot_pareto_front_3d(self, df: pd.DataFrame, 
                            obj_x: str, obj_y: str, obj_z: str,
                            pareto_df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Plot 3D Pareto front.
        
        Args:
            df: Full DataFrame
            obj_x, obj_y, obj_z: Three objectives
            pareto_df: Pareto-optimal subset
        
        Returns:
            Plotly 3D scatter figure
        """
        if pareto_df is None:
            pareto_df = self.calculate_pareto_front(df, [obj_x, obj_y, obj_z])
        
        fig = go.Figure()
        
        # All points
        fig.add_trace(go.Scatter3d(
            x=df[obj_x],
            y=df[obj_y],
            z=df[obj_z],
            mode='markers',
            name='All Materials',
            marker=dict(
                size=5,
                color='lightblue',
                opacity=0.4
            ),
            text=df['formula'],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Pareto front
        if not pareto_df.empty:
            fig.add_trace(go.Scatter3d(
                x=pareto_df[obj_x],
                y=pareto_df[obj_y],
                z=pareto_df[obj_z],
                mode='markers',
                name='Pareto Front',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                text=pareto_df['formula'],
                hovertemplate='<b>PARETO: %{text}</b><extra></extra>'
            ))
        
        fig.update_layout(
            title="3D Pareto Front",
            scene=dict(
                xaxis_title=obj_x.replace('obj_', '').replace('_', ' ').title(),
                yaxis_title=obj_y.replace('obj_', '').replace('_', ' ').title(),
                zaxis_title=obj_z.replace('obj_', '').replace('_', ' ').title()
            ),
            height=600,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff'
        )
        
        return fig
    
    def plot_objective_tradeoffs(self, df: pd.DataFrame, 
                                objectives: List[str]) -> go.Figure:
        """
        Plot pairwise objective trade-offs (scatter matrix).
        
        Args:
            df: DataFrame with objectives
            objectives: List of objective column names
        
        Returns:
            Plotly figure with subplot matrix
        """
        n_obj = len(objectives)
        
        fig = make_subplots(
            rows=n_obj,
            cols=n_obj,
            subplot_titles=[f"{o1} vs {o2}" 
                          for o1 in objectives for o2 in objectives]
        )
        
        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(
                            x=df[obj1],
                            name=obj1,
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter
                    fig.add_trace(
                        go.Scatter(
                            x=df[obj2],
                            y=df[obj1],
                            mode='markers',
                            marker=dict(size=5, opacity=0.6),
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title="Objective Trade-offs Matrix",
            height=800,
            showlegend=False,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff'
        )
        
        return fig
    
    def get_recommendations(self, df: pd.DataFrame, 
                          weights: Dict[str, float],
                          n_top: int = 10) -> pd.DataFrame:
        """
        Get top recommendations based on weighted objectives.
        
        Args:
            df: DataFrame with objectives
            weights: Objective weights
            n_top: Number of top materials to return
        
        Returns:
            Top materials sorted by weighted score
        """
        df_scored = self.weighted_scalarization(df, weights)
        
        top_materials = df_scored.head(n_top).copy()
        
        # Add ranking
        top_materials['rank'] = range(1, len(top_materials) + 1)
        
        return top_materials


def default_weights() -> Dict[str, float]:
    """Return default objective weights (equal importance)"""
    return {
        'obj_bandgap_match': 0.4,
        'obj_stability': 0.3,
        'obj_synthesizability': 0.2,
        'obj_cost': 0.1
    }
