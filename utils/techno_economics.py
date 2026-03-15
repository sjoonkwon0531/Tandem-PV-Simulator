"""
Techno-Economic Analysis for Perovskite Solar Cells
====================================================

Supply chain + manufacturing cost modeling + $/Watt calculation

Features:
- Raw material cost database
- Manufacturing cost model (deposition, annealing, encapsulation)
- $/Watt calculation for each composition
- Sensitivity analysis (cost drivers)
- Comparison vs silicon baseline
- Supply chain risk assessment
- Toxicity scoring
- Manufacturing readiness level (TRL)

Author: OpenClaw Agent
Date: 2026-03-15 (V6)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_models import CompositionFeaturizer


class TechnoEconomicAnalyzer:
    """
    Comprehensive techno-economic analysis for perovskite solar cells.
    
    Cost breakdown:
    1. Raw materials ($/kg)
    2. Processing (deposition, annealing, encapsulation)
    3. Characterization & QC
    4. Overhead (labor, facilities, utilities)
    
    Metrics:
    - $/m² (module area cost)
    - $/Watt (levelized cost)
    - $/kWh (LCOE analog)
    """
    
    # Raw material costs ($/kg, 2026 estimates)
    RAW_MATERIAL_COSTS = {
        # A-site cations
        'MA': 500,    # Methylammonium iodide (synthesized)
        'FA': 600,    # Formamidinium iodide (synthesized)
        'Cs': 2000,   # Cesium carbonate/iodide
        'Rb': 1500,   # Rubidium iodide
        'K': 100,     # Potassium iodide
        
        # B-site metals
        'Pb': 5,      # Lead (commodity price)
        'Sn': 20,     # Tin (higher purity)
        'Ge': 1500,   # Germanium (semiconductor grade)
        'Sr': 50,     # Strontium
        'Ca': 30,     # Calcium
        
        # X-site halides
        'I': 50,      # Iodine (purified)
        'Br': 30,     # Bromine (purified)
        'Cl': 10,     # Chlorine salts
        'F': 20       # Fluorine salts
    }
    
    # Atomic masses (g/mol)
    ATOMIC_MASSES = {
        'MA': 31, 'FA': 45, 'Cs': 133, 'Rb': 85, 'K': 39,
        'Pb': 207, 'Sn': 119, 'Ge': 73, 'Sr': 88, 'Ca': 40,
        'I': 127, 'Br': 80, 'Cl': 35, 'F': 19
    }
    
    # Supply chain risk scores (0-1, higher = riskier)
    SUPPLY_RISK = {
        'MA': 0.3,  # Synthetic, multiple suppliers
        'FA': 0.3,
        'Cs': 0.7,  # Limited sources, geopolitical
        'Rb': 0.6,
        'K': 0.1,   # Abundant
        'Pb': 0.4,  # Commodity, but environmental concerns
        'Sn': 0.3,
        'Ge': 0.8,  # Rare, concentrated supply
        'I': 0.5,   # Few producers (Chile, Japan)
        'Br': 0.4,
        'Cl': 0.1,
        'F': 0.2
    }
    
    # Manufacturing process costs ($/m²)
    PROCESS_COSTS = {
        'substrate': 5.0,          # ITO glass
        'hole_transport': 3.0,     # HTL (Spiro-OMeTAD, PTAA, etc.)
        'perovskite_deposition': 8.0,  # Spin-coating, slot-die, vapor deposition
        'electron_transport': 2.0,  # ETL (TiO2, SnO2)
        'electrode': 4.0,          # Metal contact (Au, Ag)
        'encapsulation': 10.0,     # Glass sealing + edge seal
        'characterization': 5.0,   # I-V testing, EQE
        'overhead': 15.0           # Labor, facilities, utilities
    }
    
    # Silicon baseline (for comparison)
    SILICON_BASELINE = {
        'cost_per_watt': 0.25,  # $/W (2026 typical)
        'efficiency': 0.22,     # 22% (commercial mono-Si)
        'lifetime': 25          # years
    }
    
    def __init__(self, featurizer: Optional[CompositionFeaturizer] = None):
        """
        Args:
            featurizer: Composition featurizer for property extraction
        """
        self.featurizer = featurizer or CompositionFeaturizer()
    
    def calculate_material_cost(self, formula: str) -> Dict[str, float]:
        """
        Calculate raw material cost for composition.
        
        Returns:
            Dict with cost breakdown
        """
        comp = self.featurizer._parse_composition(formula)
        
        # Calculate molar mass
        molar_mass = 0
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                molar_mass += self.ATOMIC_MASSES.get(elem, 100) * frac
        
        # Calculate cost per mole
        cost_per_mole = 0
        cost_breakdown = {}
        
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                mass_contrib = self.ATOMIC_MASSES.get(elem, 100) * frac
                cost_per_kg = self.RAW_MATERIAL_COSTS.get(elem, 100)
                
                # Cost for this element ($/mol)
                elem_cost = (mass_contrib / 1000) * cost_per_kg
                cost_per_mole += elem_cost
                cost_breakdown[elem] = elem_cost
        
        # Convert to $/kg
        cost_per_kg = (cost_per_mole / molar_mass) * 1000 if molar_mass > 0 else 999
        
        return {
            'cost_per_kg': cost_per_kg,
            'cost_per_mole': cost_per_mole,
            'molar_mass': molar_mass,
            'breakdown': cost_breakdown
        }
    
    def calculate_module_cost(self, formula: str, 
                             perovskite_thickness_nm: float = 500,
                             perovskite_density_g_cm3: float = 4.0,
                             active_area_fraction: float = 0.9) -> Dict[str, float]:
        """
        Calculate full module cost ($/m²).
        
        Args:
            formula: Chemical formula
            perovskite_thickness_nm: Active layer thickness (nm)
            perovskite_density_g_cm3: Material density (g/cm³)
            active_area_fraction: Fraction of module that's active (vs dead area)
        
        Returns:
            Dict with module cost breakdown
        """
        # Material cost
        mat_cost = self.calculate_material_cost(formula)
        
        # Volume of perovskite per m² (cm³)
        thickness_cm = perovskite_thickness_nm * 1e-7
        area_cm2 = 1e4  # 1 m² = 10,000 cm²
        volume_cm3 = thickness_cm * area_cm2
        
        # Mass of perovskite (kg)
        mass_kg = (volume_cm3 * perovskite_density_g_cm3) / 1000
        
        # Material cost per m²
        perovskite_cost = mass_kg * mat_cost['cost_per_kg']
        
        # Total process cost
        total_process_cost = sum(self.PROCESS_COSTS.values())
        
        # Total module cost
        total_cost_per_m2 = perovskite_cost + total_process_cost
        
        return {
            'perovskite_material_cost': perovskite_cost,
            'process_cost': total_process_cost,
            'total_cost_per_m2': total_cost_per_m2,
            'perovskite_mass_kg': mass_kg,
            'process_breakdown': self.PROCESS_COSTS.copy(),
            'material_breakdown': mat_cost['breakdown']
        }
    
    def calculate_cost_per_watt(self, formula: str, 
                                efficiency: float,
                                perovskite_thickness_nm: float = 500,
                                irradiance_W_m2: float = 1000) -> Dict[str, float]:
        """
        Calculate $/Watt (levelized cost per watt-peak).
        
        Args:
            formula: Chemical formula
            efficiency: Power conversion efficiency (0-1)
            perovskite_thickness_nm: Active layer thickness (nm)
            irradiance_W_m2: Standard test irradiance (1000 W/m² = 1 sun)
        
        Returns:
            Dict with $/W calculation
        """
        # Module cost
        module_cost = self.calculate_module_cost(formula, perovskite_thickness_nm)
        
        # Power output per m² (W)
        power_per_m2 = irradiance_W_m2 * efficiency
        
        # Cost per watt
        if power_per_m2 > 0:
            cost_per_watt = module_cost['total_cost_per_m2'] / power_per_m2
        else:
            cost_per_watt = 999
        
        # Compare to silicon
        vs_silicon_ratio = cost_per_watt / self.SILICON_BASELINE['cost_per_watt']
        
        return {
            'cost_per_watt': cost_per_watt,
            'efficiency': efficiency,
            'power_per_m2': power_per_m2,
            'module_cost_per_m2': module_cost['total_cost_per_m2'],
            'silicon_baseline': self.SILICON_BASELINE['cost_per_watt'],
            'vs_silicon_ratio': vs_silicon_ratio,
            'competitive': cost_per_watt <= self.SILICON_BASELINE['cost_per_watt']
        }
    
    def sensitivity_analysis(self, formula: str, efficiency: float) -> pd.DataFrame:
        """
        Sensitivity analysis: Which cost drivers matter most?
        
        Vary each parameter ±20% and measure impact on $/W.
        
        Returns:
            DataFrame with sensitivity coefficients
        """
        base_cost = self.calculate_cost_per_watt(formula, efficiency)
        base_value = base_cost['cost_per_watt']
        
        sensitivities = []
        
        # 1. Material cost sensitivity
        original_costs = self.RAW_MATERIAL_COSTS.copy()
        
        # Find dominant material
        mat_cost = self.calculate_material_cost(formula)
        dominant_elem = max(mat_cost['breakdown'], key=mat_cost['breakdown'].get)
        
        # Perturb dominant material cost
        for delta_pct in [-20, 20]:
            self.RAW_MATERIAL_COSTS[dominant_elem] = original_costs[dominant_elem] * (1 + delta_pct/100)
            new_cost = self.calculate_cost_per_watt(formula, efficiency)
            sensitivity = (new_cost['cost_per_watt'] - base_value) / base_value * 100
            
            sensitivities.append({
                'parameter': f'{dominant_elem} cost',
                'delta_pct': delta_pct,
                'new_cost_per_watt': new_cost['cost_per_watt'],
                'sensitivity_pct': sensitivity
            })
        
        # Restore original
        self.RAW_MATERIAL_COSTS = original_costs.copy()
        
        # 2. Efficiency sensitivity
        for delta_pct in [-20, 20]:
            new_eff = efficiency * (1 + delta_pct/100)
            new_cost = self.calculate_cost_per_watt(formula, new_eff)
            sensitivity = (new_cost['cost_per_watt'] - base_value) / base_value * 100
            
            sensitivities.append({
                'parameter': 'Efficiency',
                'delta_pct': delta_pct,
                'new_cost_per_watt': new_cost['cost_per_watt'],
                'sensitivity_pct': sensitivity
            })
        
        # 3. Process cost sensitivity (encapsulation)
        original_enc_cost = self.PROCESS_COSTS['encapsulation']
        
        for delta_pct in [-20, 20]:
            self.PROCESS_COSTS['encapsulation'] = original_enc_cost * (1 + delta_pct/100)
            new_cost = self.calculate_cost_per_watt(formula, efficiency)
            sensitivity = (new_cost['cost_per_watt'] - base_value) / base_value * 100
            
            sensitivities.append({
                'parameter': 'Encapsulation cost',
                'delta_pct': delta_pct,
                'new_cost_per_watt': new_cost['cost_per_watt'],
                'sensitivity_pct': sensitivity
            })
        
        # Restore
        self.PROCESS_COSTS['encapsulation'] = original_enc_cost
        
        # 4. Thickness sensitivity
        for delta_pct in [-20, 20]:
            new_thickness = 500 * (1 + delta_pct/100)
            new_cost = self.calculate_cost_per_watt(formula, efficiency, new_thickness)
            sensitivity = (new_cost['cost_per_watt'] - base_value) / base_value * 100
            
            sensitivities.append({
                'parameter': 'Perovskite thickness',
                'delta_pct': delta_pct,
                'new_cost_per_watt': new_cost['cost_per_watt'],
                'sensitivity_pct': sensitivity
            })
        
        df = pd.DataFrame(sensitivities)
        
        # Calculate tornado metrics (absolute sensitivity)
        tornado_data = []
        for param in df['parameter'].unique():
            subset = df[df['parameter'] == param]
            if len(subset) == 2:
                sens_low = subset[subset['delta_pct'] == -20]['sensitivity_pct'].values[0]
                sens_high = subset[subset['delta_pct'] == 20]['sensitivity_pct'].values[0]
                
                tornado_data.append({
                    'parameter': param,
                    'sensitivity_magnitude': abs(sens_high - sens_low) / 2,
                    'direction': 'favorable' if sens_high < 0 else 'unfavorable'
                })
        
        df_tornado = pd.DataFrame(tornado_data)
        df_tornado = df_tornado.sort_values('sensitivity_magnitude', ascending=False)
        
        return df, df_tornado
    
    def calculate_supply_risk(self, formula: str) -> Dict[str, float]:
        """
        Calculate supply chain risk score.
        
        Factors:
        - Element availability (geopolitical)
        - Supplier concentration
        - Price volatility
        
        Returns:
            Dict with risk scores
        """
        comp = self.featurizer._parse_composition(formula)
        
        risk_scores = {}
        total_risk = 0
        total_weight = 0
        
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                risk = self.SUPPLY_RISK.get(elem, 0.5)
                mass = self.ATOMIC_MASSES.get(elem, 100) * frac
                
                risk_scores[elem] = risk
                total_risk += risk * mass
                total_weight += mass
        
        overall_risk = total_risk / total_weight if total_weight > 0 else 0.5
        
        # Risk categories
        if overall_risk < 0.3:
            risk_level = 'Low'
        elif overall_risk < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'element_risks': risk_scores,
            'high_risk_elements': [e for e, r in risk_scores.items() if r > 0.6]
        }
    
    def calculate_toxicity_score(self, formula: str) -> Dict[str, float]:
        """
        Calculate toxicity score (Pb content penalty).
        
        Returns:
            Dict with toxicity metrics
        """
        comp = self.featurizer._parse_composition(formula)
        
        # Toxicity weights (0-1, higher = more toxic)
        toxicity_weights = {
            'Pb': 1.0,   # High toxicity
            'Sn': 0.3,   # Low toxicity
            'Ge': 0.2,
            'MA': 0.1,
            'FA': 0.1,
            'Cs': 0.2,
            'I': 0.3,
            'Br': 0.4,
            'Cl': 0.3
        }
        
        total_toxicity = 0
        total_mass = 0
        
        for site in ['A', 'B', 'X']:
            for elem, frac in comp.get(site, {}).items():
                tox = toxicity_weights.get(elem, 0.1)
                mass = self.ATOMIC_MASSES.get(elem, 100) * frac
                
                total_toxicity += tox * mass
                total_mass += mass
        
        toxicity_score = total_toxicity / total_mass if total_mass > 0 else 0
        
        # Pb content (mass fraction)
        pb_frac = comp.get('B', {}).get('Pb', 0)
        pb_mass_fraction = (self.ATOMIC_MASSES['Pb'] * pb_frac) / total_mass if total_mass > 0 else 0
        
        # Classification
        if toxicity_score < 0.3:
            toxicity_level = 'Low (Pb-free or Pb-minimal)'
        elif toxicity_score < 0.7:
            toxicity_level = 'Medium (Mixed Pb/Sn)'
        else:
            toxicity_level = 'High (Pb-rich)'
        
        return {
            'toxicity_score': toxicity_score,
            'toxicity_level': toxicity_level,
            'pb_mass_fraction': pb_mass_fraction,
            'pb_free': pb_frac == 0
        }
    
    def calculate_trl(self, formula: str, has_experimental_data: bool = False) -> Dict[str, any]:
        """
        Estimate Technology Readiness Level (TRL) analog.
        
        TRL scale (1-9):
        1-3: Basic research
        4-6: Technology development
        7-8: System demonstration
        9: Commercial deployment
        
        Heuristics:
        - Pure MAPbI3, FAPbI3 → TRL 7-8 (widely studied)
        - Mixed cation/halide → TRL 5-6 (optimization phase)
        - Novel compositions → TRL 3-4 (early research)
        - Experimental data → +1 TRL
        
        Returns:
            Dict with TRL estimate
        """
        # Simple heuristics based on composition
        formula_lower = formula.lower()
        
        # High TRL compositions (literature champions)
        if formula in ['MAPbI3', 'FAPbI3', 'CsPbI3']:
            trl = 7
            description = 'Well-established champion composition'
        elif 'ma' in formula_lower and 'fa' in formula_lower and 'pb' in formula_lower and 'i' in formula_lower:
            trl = 6
            description = 'Mixed cation compositions (under optimization)'
        elif 'sn' in formula_lower and 'pb' in formula_lower:
            trl = 5
            description = 'Mixed Pb/Sn (stability challenges)'
        elif 'sn' in formula_lower and 'pb' not in formula_lower:
            trl = 4
            description = 'Pb-free Sn-based (early development)'
        elif 'ge' in formula_lower:
            trl = 3
            description = 'Ge-based (exploratory research)'
        else:
            trl = 4
            description = 'Novel composition (early development)'
        
        # Boost TRL if experimental data exists
        if has_experimental_data:
            trl = min(trl + 1, 9)
            description += ' (experimental validation available)'
        
        return {
            'trl': trl,
            'description': description,
            'ready_for_pilot': trl >= 6,
            'ready_for_commercialization': trl >= 8
        }
    
    def calculate_regulatory_compliance(self, formula: str) -> Dict[str, any]:
        """
        Estimate regulatory compliance challenges.
        
        Key regulations:
        - RoHS (Restriction of Hazardous Substances): Pb content limits
        - REACH (EU): Chemical safety
        - EPA (US): Toxicity
        
        Returns:
            Dict with compliance indicators
        """
        tox = self.calculate_toxicity_score(formula)
        
        # RoHS compliance (Pb content)
        # RoHS allows <0.1% Pb by weight in homogeneous materials
        # Perovskites typically have 50-60% Pb → NOT RoHS compliant
        
        rohs_compliant = tox['pb_free']
        
        # REACH compliance (simplified)
        # High toxicity = requires more documentation/testing
        reach_complexity = 'High' if tox['toxicity_score'] > 0.7 else 'Medium' if tox['toxicity_score'] > 0.3 else 'Low'
        
        # Overall regulatory risk
        if rohs_compliant:
            regulatory_risk = 'Low'
        elif tox['pb_mass_fraction'] < 0.3:  # Mixed Pb/Sn
            regulatory_risk = 'Medium'
        else:
            regulatory_risk = 'High'
        
        return {
            'rohs_compliant': rohs_compliant,
            'reach_complexity': reach_complexity,
            'regulatory_risk': regulatory_risk,
            'pb_mass_fraction': tox['pb_mass_fraction'],
            'notes': 'Pb-based perovskites require environmental impact mitigation (encapsulation, recycling)'
        }
    
    def plot_cost_waterfall(self, formula: str, efficiency: float) -> go.Figure:
        """
        Waterfall chart: Cost breakdown from raw materials to $/W.
        """
        module_cost = self.calculate_module_cost(formula)
        cost_per_w = self.calculate_cost_per_watt(formula, efficiency)
        
        # Build waterfall data
        categories = []
        values = []
        
        # Raw materials
        categories.append('Perovskite Material')
        values.append(module_cost['perovskite_material_cost'])
        
        # Process steps
        for process, cost in module_cost['process_breakdown'].items():
            categories.append(process.replace('_', ' ').title())
            values.append(cost)
        
        # Total
        categories.append('Total ($/m²)')
        values.append(module_cost['total_cost_per_m2'])
        
        # Create waterfall
        fig = go.Figure(go.Waterfall(
            x=categories,
            y=values,
            text=[f"${v:.2f}" for v in values],
            textposition='outside',
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
        ))
        
        fig.update_layout(
            title=f'Cost Breakdown: {formula}<br>→ ${cost_per_w["cost_per_watt"]:.3f}/W @ {efficiency*100:.1f}% efficiency',
            yaxis_title='Cost ($/m²)',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        
        return fig
    
    def plot_tornado_sensitivity(self, df_tornado: pd.DataFrame, formula: str) -> go.Figure:
        """
        Tornado diagram: Which parameters drive cost most?
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_tornado['parameter'],
            x=df_tornado['sensitivity_magnitude'],
            orientation='h',
            marker=dict(
                color=df_tornado['sensitivity_magnitude'],
                colorscale='RdYlGn_r',
                showscale=False
            ),
            text=[f"{v:.1f}%" for v in df_tornado['sensitivity_magnitude']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Cost Sensitivity Analysis: {formula}<br>(Impact of ±20% parameter change on $/W)',
            xaxis_title='Sensitivity (%)',
            yaxis_title='Parameter',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig
    
    def plot_scale_up_risk_radar(self, formula: str, has_experimental_data: bool = False) -> go.Figure:
        """
        Spider/radar chart: Scale-up risk assessment across dimensions.
        """
        # Calculate all risk dimensions
        supply_risk = self.calculate_supply_risk(formula)
        tox = self.calculate_toxicity_score(formula)
        trl = self.calculate_trl(formula, has_experimental_data)
        reg = self.calculate_regulatory_compliance(formula)
        
        # Normalize scores to 0-1 (higher = better)
        scores = {
            'Supply Chain': 1.0 - supply_risk['overall_risk_score'],
            'Environmental': 1.0 - tox['toxicity_score'],
            'Readiness (TRL)': trl['trl'] / 9.0,
            'Regulatory': 1.0 if reg['rohs_compliant'] else 0.5 if reg['regulatory_risk'] == 'Medium' else 0.2,
            'Stability': 0.7  # Placeholder (would need real data)
        }
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            name=formula,
            marker=dict(color='rgba(100, 150, 200, 0.6)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f'Scale-Up Risk Assessment: {formula}<br>(Higher = Better)',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        
        return fig


def compare_to_silicon(formulas: List[str], efficiencies: List[float]) -> pd.DataFrame:
    """
    Compare multiple perovskite compositions to silicon baseline.
    
    Args:
        formulas: List of chemical formulas
        efficiencies: Corresponding efficiencies (0-1)
    
    Returns:
        DataFrame with comparison
    """
    analyzer = TechnoEconomicAnalyzer()
    
    results = []
    
    for formula, eff in zip(formulas, efficiencies):
        cost_data = analyzer.calculate_cost_per_watt(formula, eff)
        
        results.append({
            'formula': formula,
            'efficiency': eff,
            'cost_per_watt': cost_data['cost_per_watt'],
            'vs_silicon_ratio': cost_data['vs_silicon_ratio'],
            'competitive': cost_data['competitive']
        })
    
    df = pd.DataFrame(results)
    
    # Add silicon baseline
    df_silicon = pd.DataFrame([{
        'formula': 'Silicon (baseline)',
        'efficiency': analyzer.SILICON_BASELINE['efficiency'],
        'cost_per_watt': analyzer.SILICON_BASELINE['cost_per_watt'],
        'vs_silicon_ratio': 1.0,
        'competitive': True
    }])
    
    df = pd.concat([df, df_silicon], ignore_index=True)
    
    return df
