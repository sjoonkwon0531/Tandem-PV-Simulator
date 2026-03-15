"""
Generative Inverse Design for Perovskites
==========================================

Given TARGET properties, generate candidate compositions that satisfy ALL constraints.

Features:
- Constrained optimization (scipy.optimize)
- Multi-constraint inverse design (bandgap + stability + cost)
- Candidate ranking by feasibility + confidence
- Integration with GP surrogate from V5

Author: OpenClaw Agent
Date: 2026-03-15 (V6)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
import plotly.graph_objects as go
from itertools import product

from ml_models import CompositionFeaturizer


class InverseDesignEngine:
    """
    Inverse design: Target properties → Candidate compositions
    
    Standard workflow:
    User: "I want bandgap=1.35 eV, stability>0.9, cost<$50/kg"
    Engine: Generates candidates that satisfy ALL constraints
    
    Methods:
    1. Constrained sampling (rejection sampling)
    2. Constrained optimization (scipy.optimize + GP surrogate)
    3. Genetic algorithm (differential_evolution)
    """
    
    def __init__(self, gp_model: Optional[GaussianProcessRegressor] = None,
                 featurizer: Optional[CompositionFeaturizer] = None):
        """
        Args:
            gp_model: Trained GP surrogate (from V5 BO)
            featurizer: Composition featurizer
        """
        self.gp_model = gp_model
        self.featurizer = featurizer or CompositionFeaturizer()
        
        # Composition search space
        self.search_space = {
            'A': ['MA', 'FA', 'Cs', 'Rb'],
            'B': ['Pb', 'Sn'],
            'X': ['I', 'Br', 'Cl']
        }
        
        # Material costs ($/kg)
        self.material_costs = {
            'MA': 500, 'FA': 600, 'Cs': 2000, 'Rb': 1500,
            'Pb': 5, 'Sn': 20, 'Ge': 1500,
            'I': 50, 'Br': 30, 'Cl': 10
        }
    
    def generate_candidates(self, 
                           target_bandgap: float,
                           bandgap_tolerance: float = 0.05,
                           min_stability: float = 0.85,
                           max_cost: float = 100,
                           n_candidates: int = 1000,
                           method: str = 'rejection') -> pd.DataFrame:
        """
        Generate candidate compositions satisfying constraints.
        
        Args:
            target_bandgap: Target bandgap (eV)
            bandgap_tolerance: Allowed deviation (eV)
            min_stability: Minimum stability score (0-1)
            max_cost: Maximum cost ($/kg)
            n_candidates: Number of candidates to screen
            method: 'rejection' (fast), 'genetic' (thorough)
        
        Returns:
            DataFrame with valid candidates ranked by confidence
        """
        
        if method == 'rejection':
            return self._rejection_sampling(
                target_bandgap, bandgap_tolerance, min_stability, 
                max_cost, n_candidates
            )
        elif method == 'genetic':
            return self._genetic_algorithm(
                target_bandgap, bandgap_tolerance, min_stability, max_cost
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _rejection_sampling(self, target_bg: float, tol_bg: float,
                           min_stab: float, max_cost: float,
                           n_total: int) -> pd.DataFrame:
        """
        Rejection sampling: Generate random compositions, filter by constraints.
        
        Fast but may have low acceptance rate for tight constraints.
        """
        candidates = []
        formulas_generated = set()
        
        # Generate random compositions
        for _ in range(n_total):
            # Random mixing ratios
            n_a = np.random.choice([1, 2, 3])
            n_b = np.random.choice([1, 2])
            n_x = np.random.choice([1, 2, 3])
            
            a_species = np.random.choice(self.search_space['A'], size=n_a, replace=False)
            b_species = np.random.choice(self.search_space['B'], size=n_b, replace=False)
            x_species = np.random.choice(self.search_space['X'], size=n_x, replace=False)
            
            # Random fractions (Dirichlet distribution for natural mixing)
            a_fracs = np.random.dirichlet(np.ones(n_a))
            b_fracs = np.random.dirichlet(np.ones(n_b))
            x_fracs = np.random.dirichlet(np.ones(n_x))
            
            # Build formula
            formula = self._build_formula(
                a_species, a_fracs, b_species, b_fracs, x_species, x_fracs
            )
            
            if formula in formulas_generated:
                continue
            
            formulas_generated.add(formula)
            
            # Evaluate properties
            features = self.featurizer.featurize(formula)
            
            # Predict bandgap
            if self.gp_model is not None:
                X_test = features.reshape(1, -1)
                bg_pred, bg_std = self.gp_model.predict(X_test, return_std=True)
                bg_pred = bg_pred[0]
                bg_std = bg_std[0]
            else:
                # Fallback: estimate from features (crude)
                bg_pred = self._estimate_bandgap_from_features(features)
                bg_std = 0.3  # High uncertainty
            
            # Calculate stability (tolerance factor)
            tolerance = features[16]
            stability_score = 1.0 - abs(tolerance - 0.95)  # Ideal = 0.95
            stability_score = np.clip(stability_score, 0, 1)
            
            # Calculate cost
            cost = self._calculate_cost(formula)
            
            # Check constraints
            bg_ok = abs(bg_pred - target_bg) <= tol_bg
            stab_ok = stability_score >= min_stab
            cost_ok = cost <= max_cost
            
            if bg_ok and stab_ok and cost_ok:
                candidates.append({
                    'formula': formula,
                    'predicted_bandgap': bg_pred,
                    'bandgap_uncertainty': bg_std,
                    'stability_score': stability_score,
                    'tolerance_factor': tolerance,
                    'cost_per_kg': cost,
                    'bandgap_error': abs(bg_pred - target_bg),
                    'feasibility_score': self._calculate_feasibility(
                        bg_pred, target_bg, tol_bg, stability_score, min_stab, cost, max_cost
                    ),
                    'confidence': 1.0 / (1.0 + bg_std)  # Lower uncertainty = higher confidence
                })
        
        if not candidates:
            return pd.DataFrame()  # No valid candidates
        
        df = pd.DataFrame(candidates)
        
        # Rank by combined score
        df['combined_score'] = 0.4 * df['feasibility_score'] + 0.6 * df['confidence']
        df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _genetic_algorithm(self, target_bg: float, tol_bg: float,
                          min_stab: float, max_cost: float) -> pd.DataFrame:
        """
        Genetic algorithm for constrained composition optimization.
        
        More thorough but slower than rejection sampling.
        """
        # Define bounds for continuous mixing fractions
        # Encoding: [frac_MA, frac_FA, frac_Cs, frac_Rb, frac_Pb, frac_Sn, frac_I, frac_Br, frac_Cl]
        # Each fraction in [0, 1], must sum to 1 per site
        
        bounds = [(0, 1)] * 9  # 4 A-site + 2 B-site + 3 X-site
        
        def objective(x):
            """
            Minimize: weighted penalty for constraint violations
            """
            # Parse fractions
            a_fracs = x[0:4]
            b_fracs = x[4:6]
            x_fracs = x[6:9]
            
            # Normalize to sum=1
            a_fracs = a_fracs / (a_fracs.sum() + 1e-8)
            b_fracs = b_fracs / (b_fracs.sum() + 1e-8)
            x_fracs = x_fracs / (x_fracs.sum() + 1e-8)
            
            # Build formula
            a_species = ['MA', 'FA', 'Cs', 'Rb']
            b_species = ['Pb', 'Sn']
            x_species = ['I', 'Br', 'Cl']
            
            # Filter out near-zero fractions
            a_active = [(s, f) for s, f in zip(a_species, a_fracs) if f > 0.01]
            b_active = [(s, f) for s, f in zip(b_species, b_fracs) if f > 0.01]
            x_active = [(s, f) for s, f in zip(x_species, x_fracs) if f > 0.01]
            
            if not (a_active and b_active and x_active):
                return 1e6  # Invalid composition
            
            formula = self._build_formula(
                [s for s, f in a_active], [f for s, f in a_active],
                [s for s, f in b_active], [f for s, f in b_active],
                [s for s, f in x_active], [f for s, f in x_active]
            )
            
            # Evaluate
            features = self.featurizer.featurize(formula)
            
            if self.gp_model is not None:
                X_test = features.reshape(1, -1)
                bg_pred, _ = self.gp_model.predict(X_test, return_std=True)
                bg_pred = bg_pred[0]
            else:
                bg_pred = self._estimate_bandgap_from_features(features)
            
            tolerance = features[16]
            stability_score = 1.0 - abs(tolerance - 0.95)
            cost = self._calculate_cost(formula)
            
            # Penalty function
            penalty = 0
            
            # Bandgap constraint
            bg_error = abs(bg_pred - target_bg)
            if bg_error > tol_bg:
                penalty += 10 * (bg_error - tol_bg)**2
            
            # Stability constraint
            if stability_score < min_stab:
                penalty += 10 * (min_stab - stability_score)**2
            
            # Cost constraint
            if cost > max_cost:
                penalty += 0.1 * (cost - max_cost)**2
            
            return penalty
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=30,
            seed=42,
            atol=1e-4,
            tol=1e-4
        )
        
        # Extract best solution(s)
        # Note: DE finds ONE optimum, but we can extract population diversity
        # For simplicity, return top solution + random perturbations
        
        best_x = result.x
        candidates = []
        
        for i in range(50):
            # Perturb best solution
            if i == 0:
                x_test = best_x
            else:
                x_test = best_x + np.random.normal(0, 0.1, size=best_x.shape)
                x_test = np.clip(x_test, 0, 1)
            
            # Build candidate
            a_fracs = x_test[0:4]
            b_fracs = x_test[4:6]
            x_fracs = x_test[6:9]
            
            a_fracs = a_fracs / (a_fracs.sum() + 1e-8)
            b_fracs = b_fracs / (b_fracs.sum() + 1e-8)
            x_fracs = x_fracs / (x_fracs.sum() + 1e-8)
            
            a_species = ['MA', 'FA', 'Cs', 'Rb']
            b_species = ['Pb', 'Sn']
            x_species = ['I', 'Br', 'Cl']
            
            a_active = [(s, f) for s, f in zip(a_species, a_fracs) if f > 0.01]
            b_active = [(s, f) for s, f in zip(b_species, b_fracs) if f > 0.01]
            x_active = [(s, f) for s, f in zip(x_species, x_fracs) if f > 0.01]
            
            if not (a_active and b_active and x_active):
                continue
            
            formula = self._build_formula(
                [s for s, f in a_active], [f for s, f in a_active],
                [s for s, f in b_active], [f for s, f in b_active],
                [s for s, f in x_active], [f for s, f in x_active]
            )
            
            features = self.featurizer.featurize(formula)
            
            if self.gp_model is not None:
                X_test_feat = features.reshape(1, -1)
                bg_pred, bg_std = self.gp_model.predict(X_test_feat, return_std=True)
                bg_pred = bg_pred[0]
                bg_std = bg_std[0]
            else:
                bg_pred = self._estimate_bandgap_from_features(features)
                bg_std = 0.3
            
            tolerance = features[16]
            stability_score = 1.0 - abs(tolerance - 0.95)
            stability_score = np.clip(stability_score, 0, 1)
            cost = self._calculate_cost(formula)
            
            # Check constraints
            bg_ok = abs(bg_pred - target_bg) <= tol_bg
            stab_ok = stability_score >= min_stab
            cost_ok = cost <= max_cost
            
            if bg_ok and stab_ok and cost_ok:
                candidates.append({
                    'formula': formula,
                    'predicted_bandgap': bg_pred,
                    'bandgap_uncertainty': bg_std,
                    'stability_score': stability_score,
                    'tolerance_factor': tolerance,
                    'cost_per_kg': cost,
                    'bandgap_error': abs(bg_pred - target_bg),
                    'feasibility_score': self._calculate_feasibility(
                        bg_pred, target_bg, tol_bg, stability_score, min_stab, cost, max_cost
                    ),
                    'confidence': 1.0 / (1.0 + bg_std)
                })
        
        if not candidates:
            return pd.DataFrame()
        
        df = pd.DataFrame(candidates)
        df['combined_score'] = 0.4 * df['feasibility_score'] + 0.6 * df['confidence']
        df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        df.drop_duplicates(subset=['formula'], inplace=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _build_formula(self, a_species, a_fracs, b_species, b_fracs, x_species, x_fracs) -> str:
        """Build perovskite formula string."""
        parts = []
        
        # A-site
        for species, frac in zip(a_species, a_fracs):
            if frac > 0.01:
                if abs(frac - 1.0) < 0.01:
                    parts.append(species)
                else:
                    parts.append(f"{species}{frac:.2f}")
        
        # B-site
        for species, frac in zip(b_species, b_fracs):
            if frac > 0.01:
                if abs(frac - 1.0) < 0.01:
                    parts.append(species)
                else:
                    parts.append(f"{species}{frac:.2f}")
        
        # X-site (always 3 total)
        x_parts = []
        for species, frac in zip(x_species, x_fracs):
            if frac > 0.01:
                if abs(frac - 1.0) < 0.01:
                    x_parts.append(f"{species}3")
                else:
                    x_parts.append(f"{species}{frac*3:.2f}")
        
        parts.extend(x_parts)
        
        return "".join(parts)
    
    def _calculate_cost(self, formula: str) -> float:
        """Estimate material cost ($/kg) from composition."""
        comp = self.featurizer._parse_composition(formula)
        
        total_cost = 0
        total_mass = 0
        
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                # Atomic masses (simplified)
                masses = {
                    'MA': 31, 'FA': 45, 'Cs': 133, 'Rb': 85, 'K': 39,
                    'Pb': 207, 'Sn': 119, 'Ge': 73,
                    'I': 127, 'Br': 80, 'Cl': 35, 'F': 19
                }
                
                mass = masses.get(elem, 100) * frac
                cost_per_kg = self.material_costs.get(elem, 100)
                
                total_mass += mass
                total_cost += mass * cost_per_kg
        
        if total_mass == 0:
            return 999
        
        return total_cost / total_mass
    
    def _calculate_feasibility(self, bg_pred, target_bg, tol_bg, 
                               stab, min_stab, cost, max_cost) -> float:
        """
        Calculate feasibility score (0-1).
        
        1.0 = perfectly meets all constraints
        0.0 = violates all constraints
        """
        # Bandgap score
        bg_score = 1.0 - min(abs(bg_pred - target_bg) / tol_bg, 1.0)
        
        # Stability score
        stab_score = min((stab - min_stab) / (1.0 - min_stab), 1.0) if stab >= min_stab else 0.0
        
        # Cost score
        cost_score = 1.0 - min(cost / max_cost, 1.0)
        
        # Weighted average
        return 0.5 * bg_score + 0.3 * stab_score + 0.2 * cost_score
    
    def _estimate_bandgap_from_features(self, features: np.ndarray) -> float:
        """
        Crude bandgap estimate from features (when GP not available).
        
        Based on empirical correlations:
        - Larger X-site radius → smaller bandgap
        - Higher X-site electronegativity → larger bandgap
        """
        x_radius = features[6]  # X_avg_radius
        x_en = features[7]  # X_avg_electronegativity
        
        # Rough linear model (I=2.2Å/2.66 → 1.6eV, Cl=1.81Å/3.16 → 2.3eV)
        bg = 3.5 - 0.5 * x_radius + 0.2 * x_en
        
        return np.clip(bg, 0.5, 3.5)
    
    def visualize_target_region(self, df_candidates: pd.DataFrame,
                                target_bandgap: float,
                                bandgap_tolerance: float,
                                min_stability: float,
                                max_cost: float) -> go.Figure:
        """
        Visualize target region in property space + generated candidates.
        
        3D plot: bandgap vs stability vs cost
        Target region highlighted
        Valid candidates shown as points
        """
        fig = go.Figure()
        
        # Target region (wireframe box)
        bg_min = target_bandgap - bandgap_tolerance
        bg_max = target_bandgap + bandgap_tolerance
        
        # Draw target box
        corners = [
            [bg_min, min_stability, 0],
            [bg_max, min_stability, 0],
            [bg_max, 1.0, 0],
            [bg_min, 1.0, 0],
            [bg_min, min_stability, max_cost],
            [bg_max, min_stability, max_cost],
            [bg_max, 1.0, max_cost],
            [bg_min, 1.0, max_cost]
        ]
        
        # Draw edges of box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]
        
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode='lines',
                line=dict(color='rgba(100, 200, 100, 0.3)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Candidates
        if not df_candidates.empty:
            fig.add_trace(go.Scatter3d(
                x=df_candidates['predicted_bandgap'],
                y=df_candidates['stability_score'],
                z=df_candidates['cost_per_kg'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df_candidates['combined_score'],
                    colorscale='Viridis',
                    colorbar=dict(title='Score'),
                    showscale=True
                ),
                text=df_candidates['formula'],
                hovertemplate='<b>%{text}</b><br>Bandgap: %{x:.3f} eV<br>Stability: %{y:.3f}<br>Cost: $%{z:.1f}/kg<extra></extra>',
                name='Candidates'
            ))
        
        fig.update_layout(
            title='Inverse Design: Target Region + Generated Candidates',
            scene=dict(
                xaxis_title='Bandgap (eV)',
                yaxis_title='Stability Score',
                zaxis_title='Cost ($/kg)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=900,
            height=700
        )
        
        return fig


def compare_inverse_methods(target_bandgap: float, bandgap_tolerance: float,
                            min_stability: float, max_cost: float,
                            gp_model=None, featurizer=None) -> Dict:
    """
    Compare rejection sampling vs genetic algorithm.
    
    Returns:
        Dict with comparison metrics
    """
    engine = InverseDesignEngine(gp_model, featurizer)
    
    # Rejection sampling
    import time
    t0 = time.time()
    df_rejection = engine.generate_candidates(
        target_bandgap, bandgap_tolerance, min_stability, max_cost,
        n_candidates=2000, method='rejection'
    )
    t_rejection = time.time() - t0
    
    # Genetic algorithm
    t0 = time.time()
    df_genetic = engine.generate_candidates(
        target_bandgap, bandgap_tolerance, min_stability, max_cost,
        method='genetic'
    )
    t_genetic = time.time() - t0
    
    return {
        'rejection': {
            'n_valid': len(df_rejection),
            'time': t_rejection,
            'acceptance_rate': len(df_rejection) / 2000,
            'top_score': df_rejection['combined_score'].max() if not df_rejection.empty else 0
        },
        'genetic': {
            'n_valid': len(df_genetic),
            'time': t_genetic,
            'acceptance_rate': len(df_genetic) / 50,
            'top_score': df_genetic['combined_score'].max() if not df_genetic.empty else 0
        }
    }
