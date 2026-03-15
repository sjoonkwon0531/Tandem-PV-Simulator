"""
Autonomous Experiment Scheduler
================================

Full closed-loop autonomous experimentation:
Predict → Suggest → Simulate → Evaluate → Learn → Repeat

Features:
- Autonomous BO loop (no human intervention)
- Batch mode: run N experiments automatically
- Convergence tracking (improvement over iterations)
- Budget-aware optimization (max experiments constraint)
- Stopping criteria: convergence threshold, diminishing returns

Author: OpenClaw Agent
Date: 2026-03-15 (V7)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from bayesian_opt import BayesianOptimizer
from ml_models import BandgapPredictor


class AutonomousScheduler:
    """
    Autonomous experiment scheduler with closed-loop optimization.
    
    Workflow:
    1. Initialize with training data + ML model
    2. Run autonomous loop:
       - BO suggests next experiment
       - Simulate experiment (or wait for real result)
       - Update model with new data
       - Check convergence
       - Repeat until budget exhausted or converged
    3. Report final best candidates
    
    Stopping criteria:
    - Max iterations reached
    - Convergence (no improvement for N iterations)
    - Diminishing returns (improvement < threshold)
    """
    
    def __init__(self, 
                 ml_model: BandgapPredictor,
                 initial_data: pd.DataFrame,
                 target_bandgap: float = 1.35,
                 acq_function: str = 'ei'):
        """
        Args:
            ml_model: Trained ML surrogate model
            initial_data: Initial training data (formula + bandgap)
            target_bandgap: Target bandgap (eV)
            acq_function: Acquisition function ('ei', 'ucb', 'ts')
        """
        self.ml_model = ml_model
        self.initial_data = initial_data.copy()
        self.target_bandgap = target_bandgap
        self.acq_function = acq_function
        
        # Experiment history
        self.experiment_history = initial_data.copy()
        self.iteration_history = []
        
        # BO optimizer
        self.bo = BayesianOptimizer(
            target_bandgap=target_bandgap,
            acq_function=acq_function
        )
        self.bo.fit(initial_data, formula_col='formula', target_col='bandgap')
        
        # Search space
        self.search_space = {
            'A': ['MA', 'FA', 'Cs'],
            'B': ['Pb', 'Sn'],
            'X': ['I', 'Br', 'Cl']
        }
    
    def run_autonomous_loop(self,
                           max_iterations: int = 20,
                           convergence_window: int = 5,
                           improvement_threshold: float = 0.01,
                           batch_size: int = 1,
                           simulator: Optional[Callable] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Run autonomous experiment loop.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_window: Check convergence over this many iterations
            improvement_threshold: Min improvement to continue (eV)
            batch_size: Experiments per iteration
            simulator: Function(formula) -> bandgap (if None, use ML model)
            verbose: Print progress
        
        Returns:
            DataFrame with iteration history
        """
        
        if simulator is None:
            # Default: use ML model as "ground truth" with noise
            def simulator(formula):
                pred, _ = self.ml_model.predict([formula])
                # Add realistic noise (±0.05 eV)
                noise = np.random.normal(0, 0.05)
                return float(pred[0] + noise)
        
        # Reset iteration history
        self.iteration_history = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
            
            # 1. PREDICT: BO suggests next experiments
            suggestions = self.bo.optimize_composition(
                search_space=self.search_space,
                n_samples=1000
            )
            
            top_candidates = suggestions.head(batch_size)
            
            if verbose:
                print(f"\n🔮 BO Suggestions (batch size {batch_size}):")
                print(top_candidates[['formula', 'predicted_bandgap', 'acquisition_value']].to_string(index=False))
            
            # 2. SIMULATE: Run experiments (or simulate)
            new_experiments = []
            
            for idx, row in top_candidates.iterrows():
                formula = row['formula']
                
                # Simulate experiment
                measured_bandgap = simulator(formula)
                
                new_experiments.append({
                    'formula': formula,
                    'bandgap': measured_bandgap,
                    'iteration': iteration + 1,
                    'predicted_bandgap': row['predicted_bandgap'],
                    'acquisition_value': row['acquisition_value'],
                    'timestamp': datetime.now().isoformat()
                })
                
                if verbose:
                    error = abs(measured_bandgap - row['predicted_bandgap'])
                    print(f"  • {formula}: Predicted {row['predicted_bandgap']:.3f} eV, Measured {measured_bandgap:.3f} eV (error: {error:.3f} eV)")
            
            df_new = pd.DataFrame(new_experiments)
            
            # 3. EVALUATE: Calculate best-so-far
            self.experiment_history = pd.concat([self.experiment_history, df_new], ignore_index=True)
            
            # Best candidate so far
            self.experiment_history['abs_error'] = abs(self.experiment_history['bandgap'] - self.target_bandgap)
            best_idx = self.experiment_history['abs_error'].idxmin()
            best = self.experiment_history.loc[best_idx]
            
            if verbose:
                print(f"\n🏆 Best So Far: {best['formula']} (Eg = {best['bandgap']:.3f} eV, error = {best['abs_error']:.3f} eV)")
            
            # 4. LEARN: Update BO model with new data
            self.bo.fit(self.experiment_history, formula_col='formula', target_col='bandgap')
            
            # Track iteration metrics
            iteration_metrics = {
                'iteration': iteration + 1,
                'best_formula': best['formula'],
                'best_bandgap': best['bandgap'],
                'best_error': best['abs_error'],
                'n_experiments_total': len(self.experiment_history),
                'batch_size': batch_size,
                'improvement_vs_initial': self._calculate_improvement(iteration),
            }
            
            self.iteration_history.append(iteration_metrics)
            
            # 5. CHECK CONVERGENCE
            if self._check_convergence(convergence_window, improvement_threshold):
                if verbose:
                    print(f"\n✅ CONVERGED after {iteration + 1} iterations!")
                    print(f"   No improvement > {improvement_threshold} eV in last {convergence_window} iterations")
                break
            
            # Check diminishing returns
            if iteration >= convergence_window:
                recent_improvement = self._recent_improvement_rate(convergence_window)
                if recent_improvement < improvement_threshold / convergence_window:
                    if verbose:
                        print(f"\n⚠️ DIMINISHING RETURNS detected!")
                        print(f"   Recent improvement rate: {recent_improvement:.4f} eV/iter (threshold: {improvement_threshold / convergence_window:.4f})")
                    break
        
        # Final summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"AUTONOMOUS LOOP COMPLETE")
            print(f"{'='*60}")
            print(f"Total iterations: {len(self.iteration_history)}")
            print(f"Total experiments: {len(self.experiment_history)}")
            print(f"Best candidate: {best['formula']} (Eg = {best['bandgap']:.3f} eV)")
            print(f"Target bandgap: {self.target_bandgap} eV")
            print(f"Final error: {best['abs_error']:.3f} eV")
        
        return pd.DataFrame(self.iteration_history)
    
    def _calculate_improvement(self, current_iteration: int) -> float:
        """Calculate improvement vs initial best."""
        initial_best_error = abs(self.initial_data['bandgap'] - self.target_bandgap).min()
        current_best_error = self.iteration_history[current_iteration]['best_error']
        return float(initial_best_error - current_best_error)
    
    def _check_convergence(self, window: int, threshold: float) -> bool:
        """Check if converged (no improvement in last N iterations)."""
        if len(self.iteration_history) < window:
            return False
        
        recent_errors = [it['best_error'] for it in self.iteration_history[-window:]]
        improvement = max(recent_errors) - min(recent_errors)
        
        return improvement < threshold
    
    def _recent_improvement_rate(self, window: int) -> float:
        """Calculate recent improvement rate (eV/iteration)."""
        if len(self.iteration_history) < window:
            return float('inf')
        
        recent_errors = [it['best_error'] for it in self.iteration_history[-window:]]
        improvement = recent_errors[0] - recent_errors[-1]
        
        return improvement / window
    
    def plot_convergence(self, iteration_df: pd.DataFrame) -> go.Figure:
        """
        Plot convergence: best error vs iteration.
        
        Args:
            iteration_df: DataFrame from run_autonomous_loop
        
        Returns:
            Plotly figure
        """
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Convergence: Best Error vs Iteration', 'Cumulative Experiments'),
            vertical_spacing=0.15
        )
        
        # Convergence plot
        fig.add_trace(
            go.Scatter(
                x=iteration_df['iteration'],
                y=iteration_df['best_error'],
                mode='lines+markers',
                line=dict(color='cyan', width=3),
                marker=dict(size=8, color='cyan', symbol='circle'),
                name='Best Error'
            ),
            row=1, col=1
        )
        
        # Target line (zero error)
        fig.add_hline(
            y=0, line_dash="dash", line_color="lime",
            annotation_text="Target (Perfect Match)",
            row=1, col=1
        )
        
        # Cumulative experiments
        fig.add_trace(
            go.Scatter(
                x=iteration_df['iteration'],
                y=iteration_df['n_experiments_total'],
                mode='lines+markers',
                line=dict(color='orange', width=3),
                marker=dict(size=8, color='orange', symbol='square'),
                name='Total Experiments',
                fill='tozeroy',
                fillcolor='rgba(255,165,0,0.2)'
            ),
            row=2, col=1
        )
        
        # Layout
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Best Error (eV)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Experiments", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            title='Autonomous Loop Convergence'
        )
        
        return fig
    
    def plot_exploration_vs_exploitation(self) -> go.Figure:
        """
        Visualize exploration (high uncertainty) vs exploitation (low uncertainty) over iterations.
        
        Returns:
            Plotly figure
        """
        
        # Add iteration info to experiment history
        df = self.experiment_history.copy()
        
        # Scatter: iteration vs bandgap, color by uncertainty (if available)
        fig = go.Figure()
        
        # Group by iteration
        if 'iteration' in df.columns:
            for iteration in sorted(df['iteration'].unique()):
                df_iter = df[df['iteration'] == iteration]
                
                fig.add_trace(go.Scatter(
                    x=[iteration] * len(df_iter),
                    y=df_iter['bandgap'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=iteration,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Iteration')
                    ),
                    text=df_iter['formula'],
                    hovertemplate='<b>%{text}</b><br>Iteration: %{x}<br>Bandgap: %{y:.3f} eV<extra></extra>',
                    name=f'Iter {iteration}'
                ))
        
        # Target line
        fig.add_hline(
            y=self.target_bandgap,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {self.target_bandgap} eV"
        )
        
        fig.update_layout(
            title='Exploration vs Exploitation Over Iterations',
            xaxis_title='Iteration',
            yaxis_title='Measured Bandgap (eV)',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def get_top_discoveries(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N discoveries from autonomous loop.
        
        Args:
            n: Number of top candidates
        
        Returns:
            DataFrame with top candidates
        """
        
        df = self.experiment_history.copy()
        df = df.sort_values('abs_error')
        
        return df.head(n)[['formula', 'bandgap', 'abs_error', 'iteration']]


class BatchScheduler:
    """
    Parallel batch experiment scheduler.
    
    For labs with parallel synthesis capability:
    - Suggest N experiments per iteration (batch)
    - Each batch explores different regions of space
    - Diversity-aware suggestions (avoid redundant experiments)
    """
    
    def __init__(self, 
                 scheduler: AutonomousScheduler,
                 batch_size: int = 5):
        """
        Args:
            scheduler: AutonomousScheduler instance
            batch_size: Experiments per batch
        """
        self.scheduler = scheduler
        self.batch_size = batch_size
    
    def suggest_diverse_batch(self, n: int = None) -> pd.DataFrame:
        """
        Suggest diverse batch of experiments.
        
        Diversity strategy:
        - Top candidate (max acquisition)
        - Random sampling from high-acquisition region
        - Uncertainty sampling (high GP variance)
        
        Args:
            n: Batch size (default: self.batch_size)
        
        Returns:
            DataFrame with diverse batch
        """
        
        n = n or self.batch_size
        
        # Get large pool of candidates
        suggestions = self.scheduler.bo.optimize_composition(
            search_space=self.scheduler.search_space,
            n_samples=2000
        )
        
        # Diverse selection
        batch = []
        
        # 1. Best candidate (max acquisition)
        batch.append(suggestions.iloc[0])
        
        # 2. High-uncertainty candidates (exploration)
        suggestions_sorted_unc = suggestions.sort_values('uncertainty', ascending=False)
        batch.append(suggestions_sorted_unc.iloc[0])
        
        # 3. Random from top 20% (diversity)
        top_20pct = suggestions.head(int(len(suggestions) * 0.2))
        remaining = n - len(batch)
        
        if remaining > 0:
            sampled = top_20pct.sample(min(remaining, len(top_20pct)))
            for idx, row in sampled.iterrows():
                batch.append(row)
        
        return pd.DataFrame(batch).reset_index(drop=True)
