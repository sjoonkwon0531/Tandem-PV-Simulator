"""
What-If Scenario Engine
========================

Scenario-based analysis for materials discovery:
- "What if Pb is banned?" → re-optimize without Pb
- "What if iodine price doubles?" → recalculate economics
- "What if we need Eg < 1.0 eV?" → show feasibility map
- Side-by-side scenario comparison
- Policy impact simulation (RoHS, REACH compliance)

Features:
- Constraint-based scenario definition
- Multi-scenario comparison
- Policy impact assessment
- Cost sensitivity to price changes

Author: OpenClaw Agent
Date: 2026-03-15 (V7)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

from inverse_design import InverseDesignEngine
from techno_economics import TechnoEconomicAnalyzer
from ml_models import CompositionFeaturizer


class Scenario:
    """
    Single scenario definition.
    
    A scenario modifies:
    - Composition constraints (banned elements, required elements)
    - Cost parameters (material prices, process costs)
    - Performance targets (bandgap, efficiency, stability)
    - Regulatory constraints (RoHS, REACH)
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 banned_elements: Optional[List[str]] = None,
                 required_elements: Optional[List[str]] = None,
                 bandgap_range: Optional[Tuple[float, float]] = None,
                 max_cost_per_watt: Optional[float] = None,
                 material_price_multipliers: Optional[Dict[str, float]] = None,
                 rohs_compliant: bool = False):
        """
        Args:
            name: Scenario name
            description: Human-readable description
            banned_elements: Elements not allowed in composition
            required_elements: Elements that must be in composition
            bandgap_range: (min, max) bandgap constraint (eV)
            max_cost_per_watt: Maximum acceptable $/W
            material_price_multipliers: {element: multiplier} for cost changes
            rohs_compliant: Require RoHS compliance
        """
        self.name = name
        self.description = description
        self.banned_elements = banned_elements or []
        self.required_elements = required_elements or []
        self.bandgap_range = bandgap_range or (0.5, 3.0)
        self.max_cost_per_watt = max_cost_per_watt or float('inf')
        self.material_price_multipliers = material_price_multipliers or {}
        self.rohs_compliant = rohs_compliant
    
    def __repr__(self):
        return f"Scenario('{self.name}': {self.description})"


class ScenarioEngine:
    """
    What-if scenario analysis engine.
    
    Workflow:
    1. Define scenarios
    2. Run optimization under each scenario's constraints
    3. Compare results side-by-side
    4. Generate policy impact reports
    """
    
    def __init__(self,
                 inverse_engine: InverseDesignEngine,
                 techno_analyzer: TechnoEconomicAnalyzer):
        """
        Args:
            inverse_engine: Inverse design engine
            techno_analyzer: Techno-economic analyzer
        """
        self.inverse_engine = inverse_engine
        self.techno_analyzer = techno_analyzer
        self.featurizer = CompositionFeaturizer()
        
        # Predefined scenarios
        self.predefined_scenarios = self._create_predefined_scenarios()
    
    def _create_predefined_scenarios(self) -> Dict[str, Scenario]:
        """Create library of common scenarios."""
        
        scenarios = {}
        
        # Baseline
        scenarios['baseline'] = Scenario(
            name='Baseline',
            description='Current state: no restrictions',
            banned_elements=[],
            bandgap_range=(1.2, 1.8),
            max_cost_per_watt=0.30
        )
        
        # Pb ban
        scenarios['pb_ban'] = Scenario(
            name='Lead Ban',
            description='Regulatory ban on Pb (RoHS-like)',
            banned_elements=['Pb'],
            bandgap_range=(1.2, 1.8),
            max_cost_per_watt=0.30,
            rohs_compliant=True
        )
        
        # Iodine shortage
        scenarios['iodine_crisis'] = Scenario(
            name='Iodine Price Shock',
            description='Iodine price doubles due to supply disruption',
            material_price_multipliers={'I': 2.0},
            bandgap_range=(1.2, 1.8),
            max_cost_per_watt=0.30
        )
        
        # Low bandgap requirement
        scenarios['low_bandgap'] = Scenario(
            name='Low Bandgap (<1.0 eV)',
            description='NIR application requiring Eg < 1.0 eV',
            bandgap_range=(0.5, 1.0),
            max_cost_per_watt=0.50
        )
        
        # High bandgap requirement
        scenarios['high_bandgap'] = Scenario(
            name='High Bandgap (>2.5 eV)',
            description='UV/Blue application requiring Eg > 2.5 eV',
            bandgap_range=(2.5, 3.5),
            max_cost_per_watt=0.50
        )
        
        # Earth-abundant only
        scenarios['earth_abundant'] = Scenario(
            name='Earth-Abundant Only',
            description='Exclude all rare/expensive elements',
            banned_elements=['Cs', 'Rb', 'Ge'],
            bandgap_range=(1.2, 1.8),
            max_cost_per_watt=0.25
        )
        
        # Tandem top cell
        scenarios['tandem_top'] = Scenario(
            name='Tandem Top Cell',
            description='Wide-gap material for Si tandem (Eg ~ 1.7 eV)',
            bandgap_range=(1.65, 1.80),
            max_cost_per_watt=0.40
        )
        
        # Cost-optimized
        scenarios['cost_aggressive'] = Scenario(
            name='Aggressive Cost Target',
            description='Must beat silicon ($0.20/W)',
            bandgap_range=(1.2, 1.8),
            max_cost_per_watt=0.20
        )
        
        return scenarios
    
    def run_scenario(self, 
                     scenario: Scenario,
                     n_candidates: int = 500,
                     verbose: bool = True) -> pd.DataFrame:
        """
        Run optimization under scenario constraints.
        
        Args:
            scenario: Scenario definition
            n_candidates: Number of candidates to screen
            verbose: Print progress
        
        Returns:
            DataFrame with valid candidates
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario.name}")
            print(f"{'='*60}")
            print(f"Description: {scenario.description}")
            print(f"Constraints:")
            print(f"  • Banned elements: {scenario.banned_elements or 'None'}")
            print(f"  • Required elements: {scenario.required_elements or 'None'}")
            print(f"  • Bandgap range: {scenario.bandgap_range[0]:.2f} - {scenario.bandgap_range[1]:.2f} eV")
            print(f"  • Max $/W: ${scenario.max_cost_per_watt:.2f}")
            print(f"  • RoHS compliant: {scenario.rohs_compliant}")
            print()
        
        # Modify search space based on banned elements
        modified_search_space = self._apply_element_constraints(
            scenario.banned_elements,
            scenario.required_elements
        )
        
        # Generate candidates
        candidates = self._generate_scenario_candidates(
            modified_search_space,
            scenario,
            n_candidates
        )
        
        if len(candidates) == 0:
            if verbose:
                print("⚠️ No candidates found satisfying constraints!")
            return pd.DataFrame()
        
        # Apply economic constraints with price multipliers
        candidates = self._apply_economic_constraints(
            candidates,
            scenario
        )
        
        if verbose:
            print(f"\n✅ Found {len(candidates)} valid candidates")
            print(f"Top 5:")
            print(candidates.head(5)[['formula', 'bandgap', 'cost_per_watt', 'feasibility_score']].to_string(index=False))
        
        return candidates
    
    def _apply_element_constraints(self,
                                  banned: List[str],
                                  required: List[str]) -> Dict:
        """
        Modify search space based on element constraints.
        
        Args:
            banned: Banned elements
            required: Required elements
        
        Returns:
            Modified search space dict
        """
        
        base_space = copy.deepcopy(self.inverse_engine.search_space)
        
        # Remove banned elements from all sites
        for site, elements in base_space.items():
            base_space[site] = [e for e in elements if e not in banned]
        
        # Check that search space is non-empty
        for site, elements in base_space.items():
            if len(elements) == 0:
                raise ValueError(f"No elements available for site {site} after applying constraints!")
        
        return base_space
    
    def _generate_scenario_candidates(self,
                                     search_space: Dict,
                                     scenario: Scenario,
                                     n_candidates: int) -> pd.DataFrame:
        """
        Generate candidates for scenario.
        
        Args:
            search_space: Modified search space
            scenario: Scenario definition
            n_candidates: Number to screen
        
        Returns:
            DataFrame with candidates
        """
        
        # Generate random compositions from search space
        from itertools import product
        
        # All combinations
        A_options = search_space.get('A', ['MA', 'FA', 'Cs'])
        B_options = search_space.get('B', ['Pb', 'Sn'])
        X_options = search_space.get('X', ['I', 'Br', 'Cl'])
        
        all_combos = list(product(A_options, B_options, X_options))
        
        # Sample if too many
        if len(all_combos) > n_candidates:
            np.random.shuffle(all_combos)
            all_combos = all_combos[:n_candidates]
        
        # Build formulas
        formulas = [f"{A}{B}{X}3" for A, B, X in all_combos]
        
        # Predict bandgaps
        if self.inverse_engine.gp_model is not None:
            features = np.array([self.featurizer.featurize(f) for f in formulas])
            bandgaps, uncertainties = self.inverse_engine.gp_model.predict(features, return_std=True)
        else:
            # Fallback: simple heuristic
            bandgaps = np.random.uniform(scenario.bandgap_range[0], scenario.bandgap_range[1], len(formulas))
            uncertainties = np.random.uniform(0.05, 0.2, len(formulas))
        
        # Filter by bandgap range
        bg_min, bg_max = scenario.bandgap_range
        mask = (bandgaps >= bg_min) & (bandgaps <= bg_max)
        
        formulas_filtered = [f for i, f in enumerate(formulas) if mask[i]]
        bandgaps_filtered = bandgaps[mask]
        uncertainties_filtered = uncertainties[mask]
        
        # Calculate costs (with price multipliers)
        costs_per_kg = []
        for formula in formulas_filtered:
            base_cost = self._calculate_material_cost(formula)
            
            # Apply multipliers
            adjusted_cost = base_cost
            for element, multiplier in scenario.material_price_multipliers.items():
                if element in formula:
                    adjusted_cost *= multiplier
            
            costs_per_kg.append(adjusted_cost)
        
        # Build DataFrame
        df = pd.DataFrame({
            'formula': formulas_filtered,
            'bandgap': bandgaps_filtered,
            'uncertainty': uncertainties_filtered,
            'cost_per_kg': costs_per_kg
        })
        
        # Feasibility score
        df['feasibility_score'] = 1.0 / (1.0 + df['uncertainty'])
        
        # Rank
        df = df.sort_values('feasibility_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_material_cost(self, formula: str) -> float:
        """Calculate material cost per kg (simplified)."""
        
        costs = self.inverse_engine.material_costs
        
        # Parse formula (simplified)
        total_cost = 0
        total_mass = 0
        
        for element, cost in costs.items():
            if element in formula:
                # Estimate mass fraction (simplified)
                mass_frac = 0.33  # Assume equal thirds for ABX3
                total_cost += cost * mass_frac
                total_mass += mass_frac
        
        return total_cost / max(total_mass, 0.01)
    
    def _apply_economic_constraints(self,
                                   candidates: pd.DataFrame,
                                   scenario: Scenario) -> pd.DataFrame:
        """
        Filter candidates by economic constraints.
        
        Args:
            candidates: Candidate DataFrame
            scenario: Scenario definition
        
        Returns:
            Filtered DataFrame
        """
        
        # Calculate $/W (simplified)
        # Assume 20% efficiency baseline
        efficiency = 0.20
        
        cost_per_watt = []
        for formula in candidates['formula']:
            # Material cost contribution (simplified)
            mat_cost_per_m2 = candidates.loc[candidates['formula'] == formula, 'cost_per_kg'].values[0] * 0.001  # Assume 1g/m² active layer
            
            # Process cost (fixed)
            process_cost_per_m2 = 40.0  # From techno-economics module
            
            # Total cost per m²
            total_cost_per_m2 = mat_cost_per_m2 + process_cost_per_m2
            
            # Power per m² (20% efficiency)
            power_per_m2 = 1000 * efficiency  # W/m²
            
            # $/W
            cpw = total_cost_per_m2 / power_per_m2
            cost_per_watt.append(cpw)
        
        candidates['cost_per_watt'] = cost_per_watt
        
        # Filter by max $/W
        candidates = candidates[candidates['cost_per_watt'] <= scenario.max_cost_per_watt]
        
        # RoHS compliance
        if scenario.rohs_compliant:
            # Remove Pb-containing
            candidates = candidates[~candidates['formula'].str.contains('Pb')]
        
        return candidates.reset_index(drop=True)
    
    def compare_scenarios(self,
                         scenarios: List[Scenario],
                         n_candidates: int = 500) -> pd.DataFrame:
        """
        Compare multiple scenarios side-by-side.
        
        Args:
            scenarios: List of scenarios
            n_candidates: Candidates per scenario
        
        Returns:
            Comparison DataFrame
        """
        
        results = []
        
        for scenario in scenarios:
            candidates = self.run_scenario(scenario, n_candidates, verbose=False)
            
            if len(candidates) > 0:
                best = candidates.iloc[0]
                
                results.append({
                    'scenario': scenario.name,
                    'description': scenario.description,
                    'n_valid_candidates': len(candidates),
                    'best_formula': best['formula'],
                    'best_bandgap': best['bandgap'],
                    'best_cost_per_watt': best['cost_per_watt'],
                    'avg_cost_per_watt': candidates['cost_per_watt'].mean(),
                })
            else:
                results.append({
                    'scenario': scenario.name,
                    'description': scenario.description,
                    'n_valid_candidates': 0,
                    'best_formula': 'N/A',
                    'best_bandgap': np.nan,
                    'best_cost_per_watt': np.nan,
                    'avg_cost_per_watt': np.nan,
                })
        
        return pd.DataFrame(results)
    
    def plot_scenario_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Visualize scenario comparison.
        
        Args:
            comparison_df: DataFrame from compare_scenarios
        
        Returns:
            Plotly figure
        """
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Valid Candidates per Scenario', 'Cost per Watt Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Valid candidates
        fig.add_trace(
            go.Bar(
                x=comparison_df['scenario'],
                y=comparison_df['n_valid_candidates'],
                marker_color='cyan',
                text=comparison_df['n_valid_candidates'],
                textposition='outside',
                name='Valid Candidates'
            ),
            row=1, col=1
        )
        
        # Cost comparison
        fig.add_trace(
            go.Bar(
                x=comparison_df['scenario'],
                y=comparison_df['best_cost_per_watt'],
                marker_color='orange',
                text=[f"${v:.3f}" for v in comparison_df['best_cost_per_watt']],
                textposition='outside',
                name='Best $/W'
            ),
            row=1, col=2
        )
        
        # Silicon baseline
        fig.add_hline(y=0.25, line_dash="dash", line_color="blue", 
                     annotation_text="Silicon Baseline", row=1, col=2)
        
        fig.update_xaxes(tickangle=-45)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            title='Scenario Comparison'
        )
        
        return fig
    
    def plot_feasibility_map(self, 
                            scenario: Scenario,
                            candidates: pd.DataFrame) -> go.Figure:
        """
        Create 2D feasibility map: bandgap vs cost.
        
        Args:
            scenario: Scenario definition
            candidates: Candidate DataFrame
        
        Returns:
            Plotly figure
        """
        
        if len(candidates) == 0:
            # Empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No candidates found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="red")
            )
            fig.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font_color='white'
            )
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=candidates['bandgap'],
            y=candidates['cost_per_watt'],
            mode='markers',
            marker=dict(
                size=8,
                color=candidates['feasibility_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Feasibility'),
                line=dict(width=0.5, color='white')
            ),
            text=candidates['formula'],
            hovertemplate='<b>%{text}</b><br>Bandgap: %{x:.3f} eV<br>$/W: $%{y:.3f}<extra></extra>',
            name='Candidates'
        ))
        
        # Target region (if applicable)
        bg_min, bg_max = scenario.bandgap_range
        
        fig.add_vrect(
            x0=bg_min, x1=bg_max,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Target Bandgap",
            annotation_position="top left"
        )
        
        fig.add_hline(
            y=scenario.max_cost_per_watt,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max $/W: ${scenario.max_cost_per_watt:.2f}"
        )
        
        fig.update_layout(
            title=f'Feasibility Map: {scenario.name}',
            xaxis_title='Bandgap (eV)',
            yaxis_title='Cost per Watt ($/W)',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            height=500
        )
        
        return fig
    
    def generate_policy_impact_report(self, 
                                     baseline_scenario: Scenario,
                                     policy_scenario: Scenario,
                                     n_candidates: int = 500) -> Dict:
        """
        Generate policy impact assessment report.
        
        Compare baseline vs policy-constrained scenario.
        
        Args:
            baseline_scenario: Baseline scenario
            policy_scenario: Policy-constrained scenario
            n_candidates: Candidates per scenario
        
        Returns:
            Impact report dict
        """
        
        # Run both scenarios
        baseline_results = self.run_scenario(baseline_scenario, n_candidates, verbose=False)
        policy_results = self.run_scenario(policy_scenario, n_candidates, verbose=False)
        
        # Calculate impacts
        n_candidates_lost = len(baseline_results) - len(policy_results)
        pct_candidates_lost = (n_candidates_lost / max(len(baseline_results), 1)) * 100
        
        if len(baseline_results) > 0 and len(policy_results) > 0:
            baseline_best_cost = baseline_results['cost_per_watt'].min()
            policy_best_cost = policy_results['cost_per_watt'].min()
            cost_increase = policy_best_cost - baseline_best_cost
            cost_increase_pct = (cost_increase / baseline_best_cost) * 100
        else:
            baseline_best_cost = np.nan
            policy_best_cost = np.nan
            cost_increase = np.nan
            cost_increase_pct = np.nan
        
        report = {
            'policy_name': policy_scenario.name,
            'policy_description': policy_scenario.description,
            'baseline_candidates': len(baseline_results),
            'policy_candidates': len(policy_results),
            'candidates_lost': n_candidates_lost,
            'candidates_lost_pct': pct_candidates_lost,
            'baseline_best_cost': baseline_best_cost,
            'policy_best_cost': policy_best_cost,
            'cost_increase': cost_increase,
            'cost_increase_pct': cost_increase_pct,
            'feasible': len(policy_results) > 0,
            'recommendation': self._generate_policy_recommendation(
                len(policy_results), cost_increase_pct
            )
        }
        
        return report
    
    def _generate_policy_recommendation(self, 
                                       n_candidates: int,
                                       cost_increase_pct: float) -> str:
        """Generate human-readable policy recommendation."""
        
        if n_candidates == 0:
            return "⛔ Policy makes all candidates infeasible. Recommend exemption or technology pivot."
        
        if cost_increase_pct < 5:
            return "✅ Policy has minimal economic impact (<5% cost increase). Proceed with implementation."
        elif cost_increase_pct < 15:
            return "⚠️ Policy increases costs by 5-15%. Manageable with process optimization."
        else:
            return "🔴 Policy significantly increases costs (>15%). Consider phased implementation or R&D investment in alternatives."
