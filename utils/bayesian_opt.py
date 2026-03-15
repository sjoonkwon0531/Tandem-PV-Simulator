"""
Bayesian Optimization for Perovskite Discovery
===============================================

Lightweight BO implementation using sklearn GaussianProcessRegressor.
Suggests next experiments based on uploaded data.

Features:
- Acquisition functions: Expected Improvement, UCB, Thompson Sampling
- Composition space search
- Integration with user data

Author: OpenClaw Agent
Date: 2026-03-15 (V5)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

from ml_models import CompositionFeaturizer


class BayesianOptimizer:
    """
    Bayesian Optimization for bandgap-targeted composition search.
    
    Workflow:
    1. User uploads experimental data
    2. Fit GP surrogate on user data
    3. Suggest next experiments via acquisition function
    4. User runs experiments, uploads new data
    5. Repeat (closed-loop learning)
    """
    
    def __init__(self, target_bandgap: float = 1.68, acq_function: str = 'ei'):
        """
        Args:
            target_bandgap: Target bandgap (eV) for optimization
            acq_function: 'ei' (Expected Improvement), 'ucb' (Upper Confidence Bound), 
                         'ts' (Thompson Sampling)
        """
        self.target_bandgap = target_bandgap
        self.acq_function = acq_function
        
        self.featurizer = CompositionFeaturizer()
        self.scaler = StandardScaler()
        
        # GP model with Matérn kernel (smoother than RBF for real experiments)
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5  # Twice differentiable
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=0.01,  # Noise variance (experimental uncertainty)
            normalize_y=True
        )
        
        self.fitted = False
        self.X_train = None
        self.y_train = None
        self.formulas_train = None
    
    def fit(self, df: pd.DataFrame, formula_col: str = 'formula', 
            target_col: str = 'bandgap'):
        """
        Fit GP model on user's experimental data.
        
        Args:
            df: DataFrame with experimental results
            formula_col: Column name for chemical formula
            target_col: Column name for bandgap
        """
        # Clean data
        df_clean = df[[formula_col, target_col]].dropna()
        
        # Featurize compositions
        X = np.array([self.featurizer.featurize(f) for f in df_clean[formula_col]])
        y = df_clean[target_col].values
        
        # Scale features
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y
        self.formulas_train = df_clean[formula_col].tolist()
        
        # Fit GP
        self.gp.fit(self.X_train, self.y_train)
        
        self.fitted = True
        
        return {
            'n_samples': len(y),
            'bandgap_mean': y.mean(),
            'bandgap_std': y.std(),
            'kernel_params': self.gp.kernel_.get_params()
        }
    
    def predict(self, formulas: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict bandgap and uncertainty for new compositions.
        
        Returns:
            (predictions, uncertainties)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet!")
        
        # Featurize
        X = np.array([self.featurizer.featurize(f) for f in formulas])
        X_scaled = self.scaler.transform(X)
        
        # GP prediction with uncertainty
        y_pred, y_std = self.gp.predict(X_scaled, return_std=True)
        
        return y_pred, y_std
    
    def acquisition_function(self, formulas: List[str], 
                            acquisition: Optional[str] = None) -> np.ndarray:
        """
        Calculate acquisition function values for candidate compositions.
        
        Args:
            formulas: List of candidate formulas
            acquisition: Override default acquisition function
        
        Returns:
            Acquisition values (higher = better)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet!")
        
        acq = acquisition or self.acq_function
        
        # Get predictions and uncertainties
        y_pred, y_std = self.predict(formulas)
        
        if acq == 'ei':
            # Expected Improvement
            # How much better than current best?
            current_best = np.max(self._objective(self.y_train))
            
            z = (self._objective(y_pred) - current_best) / (y_std + 1e-9)
            ei = (self._objective(y_pred) - current_best) * norm.cdf(z) + y_std * norm.pdf(z)
            
            return ei
        
        elif acq == 'ucb':
            # Upper Confidence Bound
            # Balance exploration (uncertainty) and exploitation (predicted value)
            kappa = 2.0  # Exploration parameter (higher = more exploration)
            
            ucb = self._objective(y_pred) + kappa * y_std
            
            return ucb
        
        elif acq == 'ts':
            # Thompson Sampling
            # Sample from posterior distribution
            y_sample = y_pred + np.random.randn(len(y_pred)) * y_std
            
            return self._objective(y_sample)
        
        else:
            raise ValueError(f"Unknown acquisition function: {acq}")
    
    def _objective(self, bandgaps: np.ndarray) -> np.ndarray:
        """
        Objective function: maximize closeness to target bandgap.
        Objective = -|bandgap - target|
        """
        return -np.abs(bandgaps - self.target_bandgap)
    
    def suggest_next(self, candidates: List[str], n_suggestions: int = 5) -> pd.DataFrame:
        """
        Suggest next experiments from candidate pool.
        
        Args:
            candidates: List of candidate formulas to evaluate
            n_suggestions: Number of experiments to suggest
        
        Returns:
            DataFrame with top suggestions sorted by acquisition value
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet!")
        
        # Calculate acquisition values
        acq_values = self.acquisition_function(candidates)
        
        # Get predictions
        y_pred, y_std = self.predict(candidates)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'formula': candidates,
            'predicted_bandgap': y_pred,
            'uncertainty': y_std,
            'acquisition_value': acq_values,
            'distance_to_target': np.abs(y_pred - self.target_bandgap)
        })
        
        # Sort by acquisition value (descending)
        results = results.sort_values('acquisition_value', ascending=False)
        
        # Add rank
        results['rank'] = range(1, len(results) + 1)
        
        return results.head(n_suggestions)
    
    def optimize_composition(self, search_space: Dict[str, List[str]], 
                            n_samples: int = 1000) -> pd.DataFrame:
        """
        Optimize composition by sampling search space.
        
        Args:
            search_space: Dict with A-site, B-site, X-site options
                Example: {'A': ['MA', 'FA', 'Cs'], 'B': ['Pb', 'Sn'], 'X': ['I', 'Br']}
            n_samples: Number of random compositions to evaluate
        
        Returns:
            Top suggestions DataFrame
        """
        # Generate random compositions
        candidates = self._generate_candidates(search_space, n_samples)
        
        # Suggest best
        return self.suggest_next(candidates)
    
    def _generate_candidates(self, search_space: Dict[str, List[str]], 
                            n_samples: int) -> List[str]:
        """
        Generate random ABX3 compositions from search space.
        Supports mixing (e.g., FA0.8Cs0.2PbI3)
        """
        candidates = []
        
        a_options = search_space.get('A', ['MA', 'FA', 'Cs'])
        b_options = search_space.get('B', ['Pb', 'Sn'])
        x_options = search_space.get('X', ['I', 'Br', 'Cl'])
        
        for _ in range(n_samples):
            # Random A-site composition (1-2 species)
            n_a_species = np.random.choice([1, 2], p=[0.6, 0.4])
            a_species = np.random.choice(a_options, size=n_a_species, replace=False)
            
            if n_a_species == 1:
                a_str = a_species[0]
            else:
                # Random mixing ratio
                ratio = np.random.uniform(0.1, 0.9)
                a_str = f"{a_species[0]}{ratio:.2f}{a_species[1]}{1-ratio:.2f}"
            
            # B-site (usually pure)
            b_str = np.random.choice(b_options)
            
            # Random X-site composition (1-2 species)
            n_x_species = np.random.choice([1, 2], p=[0.5, 0.5])
            x_species = np.random.choice(x_options, size=n_x_species, replace=False)
            
            if n_x_species == 1:
                x_str = f"{x_species[0]}3"
            else:
                # Random mixing ratio
                ratio = np.random.uniform(0.1, 0.9)
                x_str = f"({x_species[0]}{ratio:.2f}{x_species[1]}{1-ratio:.2f})3"
            
            formula = f"{a_str}{b_str}{x_str}"
            candidates.append(formula)
        
        return candidates
    
    def plot_acquisition_landscape(self, candidates: List[str], 
                                   feature_indices: Tuple[int, int] = (0, 10)):
        """
        Plot 2D slice of acquisition function landscape.
        
        Args:
            candidates: Candidate formulas
            feature_indices: Which 2 features to plot (default: A_radius, X_radius)
        
        Returns:
            Plotly figure
        """
        import plotly.graph_objects as go
        
        # Get features and acquisition values
        X = np.array([self.featurizer.featurize(f) for f in candidates])
        acq_values = self.acquisition_function(candidates)
        
        fig = go.Figure()
        
        feature_names = self.featurizer.get_feature_names()
        
        fig.add_trace(go.Scatter(
            x=X[:, feature_indices[0]],
            y=X[:, feature_indices[1]],
            mode='markers',
            marker=dict(
                size=10,
                color=acq_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Acquisition Value"),
                line=dict(width=1, color='#333')
            ),
            text=[f"{f}<br>Acq: {a:.3f}" for f, a in zip(candidates, acq_values)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Acquisition Function Landscape ({self.acq_function.upper()})",
            xaxis_title=feature_names[feature_indices[0]],
            yaxis_title=feature_names[feature_indices[1]],
            height=500,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff'
        )
        
        return fig
    
    def get_convergence_plot(self) -> 'plotly.graph_objects.Figure':
        """
        Plot optimization convergence (best objective value vs iteration).
        Useful for visualizing learning progress.
        """
        import plotly.graph_objects as go
        
        if not self.fitted:
            return go.Figure()
        
        # Calculate cumulative best objective
        objectives = self._objective(self.y_train)
        cumulative_best = np.maximum.accumulate(objectives)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_best) + 1)),
            y=cumulative_best,
            mode='lines+markers',
            name='Best Objective',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        # Add target line
        target_obj = 0  # Perfect match
        fig.add_hline(
            y=target_obj,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target (Eg={self.target_bandgap} eV)"
        )
        
        fig.update_layout(
            title="Optimization Convergence",
            xaxis_title="Experiment Number",
            yaxis_title="Best Objective (closer to 0 = better)",
            height=400,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff'
        )
        
        return fig


def compare_acquisition_functions(bo: BayesianOptimizer, 
                                  candidates: List[str]) -> pd.DataFrame:
    """
    Compare different acquisition functions on same candidates.
    
    Args:
        bo: Fitted BayesianOptimizer
        candidates: List of candidate formulas
    
    Returns:
        DataFrame with rankings from each acquisition function
    """
    results = pd.DataFrame({'formula': candidates})
    
    for acq in ['ei', 'ucb', 'ts']:
        acq_values = bo.acquisition_function(candidates, acquisition=acq)
        results[f'{acq}_value'] = acq_values
        results[f'{acq}_rank'] = acq_values.argsort()[::-1].argsort() + 1
    
    # Add predictions
    y_pred, y_std = bo.predict(candidates)
    results['predicted_bandgap'] = y_pred
    results['uncertainty'] = y_std
    
    return results
