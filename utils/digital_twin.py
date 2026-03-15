"""
Digital Twin: Real-Time Process Simulation
===========================================

Simulate perovskite film formation from precursor to final crystal structure.

Process stages:
1. Spin-coating (precursor solution → wet film)
2. Nucleation (crystal seeds formation)
3. Grain growth (crystal expansion)
4. Annealing (solvent evaporation, crystallization)

Physics models:
- Coupled ODEs for concentration, nucleation rate, grain size
- Film thickness evolution (spin speed dependent)
- Temperature-dependent crystallization kinetics
- Final property prediction (grain size, roughness, bandgap, defects)

Author: OpenClaw Agent
Date: 2026-03-15 (V7)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DigitalTwin:
    """
    Real-time simulation of perovskite film formation process.
    
    Physical model:
    - dC/dt = -k_evap(T) * C                    (Precursor evaporation)
    - dN/dt = k_nuc(C, T) * (1 - N/N_max)       (Nucleation)
    - dG/dt = k_growth(C, T) * sqrt(N)          (Grain growth)
    - dh/dt = -k_thin(omega) * h                (Film thinning during spin)
    
    Where:
    - C = precursor concentration (mol/L)
    - N = nucleation density (#/cm²)
    - G = average grain size (nm)
    - h = film thickness (nm)
    - T = temperature (°C)
    - omega = spin speed (rpm)
    """
    
    def __init__(self):
        """Initialize digital twin with physics constants."""
        
        # Physics constants
        self.k_evap_0 = 0.01        # Base evaporation rate (1/s)
        self.E_evap = 0.5           # Activation energy (eV)
        self.k_nuc_0 = 1e6          # Nucleation rate constant (#/cm²/s)
        self.C_crit = 0.5           # Critical concentration for nucleation
        self.N_max = 1e10           # Maximum nucleation density
        self.k_growth_0 = 0.5       # Growth rate constant (nm/s)
        self.k_thin = 1e-4          # Spin thinning constant
        
        # Process parameters (defaults)
        self.default_params = {
            'spin_speed': 3000,         # rpm
            'spin_time': 30,            # seconds
            'anneal_temp': 100,         # °C
            'anneal_time': 600,         # seconds (10 min)
            'precursor_conc': 1.0,      # mol/L
            'film_thickness_0': 500,    # nm (initial wet film)
        }
    
    def simulate_formation(self, 
                          spin_speed: float = 3000,
                          anneal_temp: float = 100,
                          anneal_time: float = 600,
                          precursor_conc: float = 1.0,
                          n_points: int = 1000) -> pd.DataFrame:
        """
        Simulate complete film formation process.
        
        Args:
            spin_speed: Spin-coating speed (rpm)
            anneal_temp: Annealing temperature (°C)
            anneal_time: Annealing duration (s)
            precursor_conc: Initial precursor concentration (mol/L)
            n_points: Time points for simulation
        
        Returns:
            DataFrame with time-resolved film properties
        """
        
        # Total simulation time
        spin_time = 30  # Fixed spin-coating time
        total_time = spin_time + anneal_time
        
        # Time grid
        t_grid = np.linspace(0, total_time, n_points)
        
        # Initial conditions [C, N, G, h]
        y0 = [
            precursor_conc,      # C: precursor concentration
            0,                   # N: nucleation density
            0,                   # G: grain size
            500                  # h: film thickness (nm)
        ]
        
        # Solve ODEs
        def ode_system(t, y):
            C, N, G, h = y
            
            # Determine current temperature (room temp during spin, anneal temp during annealing)
            T = 25 if t < spin_time else anneal_temp
            
            # Evaporation rate (Arrhenius)
            k_evap = self.k_evap_0 * np.exp(-self.E_evap / (8.617e-5 * (T + 273)))
            
            # Nucleation rate (supersaturation-dependent)
            if C > self.C_crit:
                k_nuc = self.k_nuc_0 * (C - self.C_crit) * np.exp(-0.3 / (8.617e-5 * (T + 273)))
            else:
                k_nuc = 0
            
            # Growth rate (concentration and temperature dependent)
            k_growth = self.k_growth_0 * C * np.exp(-0.2 / (8.617e-5 * (T + 273)))
            
            # Thinning rate (only during spin)
            omega = spin_speed if t < spin_time else 0
            k_thin = self.k_thin * omega / 1000  # Normalize by 1000 rpm
            
            # ODEs
            dC_dt = -k_evap * C
            dN_dt = k_nuc * (1 - N / self.N_max) if N < self.N_max else 0
            dG_dt = k_growth * np.sqrt(max(N, 1)) if C > 0.01 else 0
            dh_dt = -k_thin * h
            
            return [dC_dt, dN_dt, dG_dt, dh_dt]
        
        # Integrate
        solution = solve_ivp(
            ode_system, 
            (0, total_time), 
            y0, 
            t_eval=t_grid,
            method='RK45'
        )
        
        # Extract results
        C_t = solution.y[0]
        N_t = solution.y[1]
        G_t = solution.y[2]
        h_t = solution.y[3]
        
        # Calculate derived properties
        T_t = np.where(solution.t < spin_time, 25, anneal_temp)
        
        # Crystallinity (fraction of crystallized material)
        crystallinity = 1 - np.exp(-N_t / (self.N_max * 0.5))
        
        # Roughness (larger grains = rougher surface, but normalized)
        roughness = G_t / 1000 * (1 - crystallinity)  # nm
        
        # Build DataFrame
        df = pd.DataFrame({
            'time': solution.t,
            'temperature': T_t,
            'concentration': C_t,
            'nucleation_density': N_t,
            'grain_size': G_t,
            'film_thickness': h_t,
            'crystallinity': crystallinity,
            'roughness': roughness
        })
        
        return df
    
    def predict_final_properties(self, df: pd.DataFrame, 
                                 target_bandgap: float = 1.55) -> Dict:
        """
        Predict final film properties from simulation results.
        
        Args:
            df: Simulation DataFrame (from simulate_formation)
            target_bandgap: Target bandgap (eV) for composition
        
        Returns:
            Dict with final property predictions
        """
        
        # Extract final state
        final = df.iloc[-1]
        
        # Grain size (nm)
        grain_size = final['grain_size']
        
        # Roughness (nm RMS)
        roughness = final['roughness']
        
        # Crystallinity (0-1)
        crystallinity = final['crystallinity']
        
        # Film thickness (nm)
        thickness = final['film_thickness']
        
        # Bandgap shift from grain boundaries (smaller grains = more grain boundaries = higher Eg)
        # Empirical: ΔEg ∝ 1/grain_size
        delta_Eg_grain = 0.05 * (500 / max(grain_size, 50))  # Max shift 0.05 eV
        
        # Bandgap shift from defects (lower crystallinity = more defects = bandgap widening)
        delta_Eg_defect = 0.03 * (1 - crystallinity)
        
        # Final bandgap
        final_bandgap = target_bandgap + delta_Eg_grain + delta_Eg_defect
        
        # Defect density (inversely proportional to crystallinity)
        defect_density = 1e16 * (1 - crystallinity)  # cm⁻³
        
        # Quality score (0-1, higher is better)
        quality_score = (
            0.4 * crystallinity +
            0.3 * min(grain_size / 500, 1.0) +  # Larger grains better
            0.2 * (1 - min(roughness / 50, 1.0)) +  # Smoother better
            0.1 * min(thickness / 300, 1.0)  # Thicker better (to a point)
        )
        
        return {
            'grain_size_nm': float(grain_size),
            'roughness_nm': float(roughness),
            'film_thickness_nm': float(thickness),
            'crystallinity': float(crystallinity),
            'final_bandgap_eV': float(final_bandgap),
            'defect_density_cm3': float(defect_density),
            'quality_score': float(quality_score)
        }
    
    def create_animation(self, df: pd.DataFrame) -> go.Figure:
        """
        Create animated visualization of film formation.
        
        Args:
            df: Simulation DataFrame
        
        Returns:
            Plotly animated scatter plot
        """
        
        # Sample frames (every 20th point for smooth animation)
        frames_idx = np.arange(0, len(df), max(1, len(df) // 50))
        
        frames = []
        for idx in frames_idx:
            row = df.iloc[idx]
            
            # Simulate grain positions (random for visualization)
            n_grains = min(int(row['nucleation_density'] / 1e7), 200)  # Cap at 200 for performance
            if n_grains > 0:
                np.random.seed(42 + idx)  # Reproducible positions
                x_grains = np.random.uniform(0, 100, n_grains)
                y_grains = np.random.uniform(0, 100, n_grains)
                sizes = np.full(n_grains, max(row['grain_size'] / 10, 1))
                
                frame = go.Frame(
                    data=[go.Scatter(
                        x=x_grains,
                        y=y_grains,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            color=row['crystallinity'],
                            colorscale='Viridis',
                            cmin=0,
                            cmax=1,
                            colorbar=dict(title='Crystallinity'),
                            line=dict(width=0.5, color='white')
                        ),
                        name='Grains'
                    )],
                    name=f"t={row['time']:.1f}s"
                )
            else:
                # No grains yet (early in process)
                frame = go.Frame(
                    data=[go.Scatter(x=[], y=[], mode='markers')],
                    name=f"t={row['time']:.1f}s"
                )
            
            frames.append(frame)
        
        # Initial frame
        initial_frame = frames[0] if frames else go.Scatter(x=[], y=[])
        
        fig = go.Figure(
            data=initial_frame.data if hasattr(initial_frame, 'data') else [initial_frame],
            frames=frames
        )
        
        # Layout
        fig.update_layout(
            title='Film Formation Animation (Top View)',
            xaxis=dict(range=[0, 100], title='x (μm)', showgrid=False),
            yaxis=dict(range=[0, 100], title='y (μm)', showgrid=False),
            width=600,
            height=600,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
        
        return fig
    
    def plot_time_series(self, df: pd.DataFrame) -> go.Figure:
        """
        Plot time-resolved properties during formation.
        
        Args:
            df: Simulation DataFrame
        
        Returns:
            Plotly subplot figure
        """
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Precursor Concentration', 'Temperature Profile',
                'Nucleation Density', 'Grain Size',
                'Film Thickness', 'Crystallinity'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Concentration
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['concentration'], 
                      line=dict(color='cyan', width=2), name='Concentration'),
            row=1, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['temperature'], 
                      line=dict(color='orange', width=2), name='Temperature'),
            row=1, col=2
        )
        
        # Nucleation
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['nucleation_density'], 
                      line=dict(color='magenta', width=2), name='Nucleation'),
            row=2, col=1
        )
        
        # Grain size
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['grain_size'], 
                      line=dict(color='lime', width=2), name='Grain Size'),
            row=2, col=2
        )
        
        # Thickness
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['film_thickness'], 
                      line=dict(color='yellow', width=2), name='Thickness'),
            row=3, col=1
        )
        
        # Crystallinity
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['crystallinity'], 
                      line=dict(color='violet', width=2), name='Crystallinity'),
            row=3, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        
        fig.update_yaxes(title_text="C (mol/L)", row=1, col=1)
        fig.update_yaxes(title_text="T (°C)", row=1, col=2)
        fig.update_yaxes(title_text="N (#/cm²)", row=2, col=1)
        fig.update_yaxes(title_text="G (nm)", row=2, col=2)
        fig.update_yaxes(title_text="h (nm)", row=3, col=1)
        fig.update_yaxes(title_text="Crystallinity", row=3, col=2)
        
        fig.update_layout(
            title='Process Simulation Time Series',
            height=800,
            showlegend=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white'
        )
        
        return fig


def compare_process_conditions(twin: DigitalTwin, 
                               conditions: List[Dict],
                               labels: List[str]) -> go.Figure:
    """
    Compare multiple process conditions side-by-side.
    
    Args:
        twin: DigitalTwin instance
        conditions: List of parameter dicts
        labels: Condition labels
    
    Returns:
        Comparison plot
    """
    
    results = []
    
    for cond, label in zip(conditions, labels):
        df = twin.simulate_formation(**cond)
        props = twin.predict_final_properties(df)
        results.append({**props, 'label': label})
    
    df_compare = pd.DataFrame(results)
    
    # Bar chart comparison
    fig = go.Figure()
    
    metrics = ['grain_size_nm', 'crystallinity', 'quality_score']
    metric_labels = ['Grain Size (nm)', 'Crystallinity', 'Quality Score']
    
    for metric, label_metric in zip(metrics, metric_labels):
        fig.add_trace(go.Bar(
            x=df_compare['label'],
            y=df_compare[metric],
            name=label_metric
        ))
    
    fig.update_layout(
        title='Process Condition Comparison',
        barmode='group',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='white'
    )
    
    return fig
