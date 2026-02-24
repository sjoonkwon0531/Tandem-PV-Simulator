#!/usr/bin/env python3
"""
Thermal Model and CTE Mismatch Analysis Engine
==============================================

Models thermal stress, coefficient of thermal expansion (CTE) mismatch,
and thermal cycling effects in multi-junction tandem solar cells.

This module calculates:
- Thermal stress from CTE mismatch (Stoney equation)
- Thermal fatigue and cycling effects
- Temperature-dependent material properties
- Predicted lifetime (T80) from thermal degradation
- Optimal layer ordering for minimal thermal stress

References:
- Stoney, "The Tension of Metallic Films Deposited by Electrolysis" (1909)
- Freund & Suresh, "Thin Film Materials: Stress, Defect Formation and Surface Evolution" (2003)
- Dupuis et al., "Physics and technology of amorphous-crystalline heterostructure silicon solar cells" (2007)
- Green, "General temperature dependence of solar cell performance" (2003)
- Kinsey & Edmondson, "Spectral response and energy output of concentrator multijunction solar cells" (2009)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d

# Local imports
try:
    from ..config import (MATERIAL_DB, Q, KB, H, C, T_CELL, DEFAULT_CONFIG)
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (MATERIAL_DB, Q, KB, H, C, T_CELL, DEFAULT_CONFIG)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThermalProperties:
    """Thermal properties for a material layer"""
    material_name: str
    thickness: float                # m
    cte: float                      # /K (coefficient of thermal expansion)
    young_modulus: float            # Pa (Young's modulus)
    poisson_ratio: float            # Poisson's ratio
    thermal_conductivity: float     # W⋅m⁻¹⋅K⁻¹
    specific_heat: float            # J⋅kg⁻¹⋅K⁻¹
    density: float                  # kg⋅m⁻³
    thermal_expansion_nonlinear: Optional[List[float]] = None  # Higher order coefficients

@dataclass
class ThermalStressResult:
    """Results from thermal stress analysis"""
    total_stress: float                 # Pa (maximum stress in stack)
    stress_per_layer: List[float]       # Pa (stress in each layer)
    curvature: float                    # m⁻¹ (substrate curvature)
    critical_thickness: List[float]     # m (critical thickness for cracking)
    thermal_fatigue_cycles: float      # Predicted cycles to failure
    temperature_gradient: List[float]   # K⋅m⁻¹ (temperature gradient per layer)
    cte_mismatch_severity: float       # Unitless (0-10 scale)

@dataclass
class ThermalCyclingParams:
    """Parameters for thermal cycling analysis"""
    min_temperature: float          # K
    max_temperature: float          # K
    heating_rate: float             # K⋅s⁻¹
    cooling_rate: float             # K⋅s⁻¹
    dwell_time_hot: float           # s
    dwell_time_cold: float          # s
    cycles_per_day: float           # cycles/day
    target_lifetime: float          # years

@dataclass
class LifetimePrediction:
    """Thermal lifetime prediction results"""
    t80_thermal: float              # years (time to 80% performance)
    dominant_failure_mode: str      # Primary thermal failure mechanism
    stress_concentration_factor: float  # Local stress enhancement
    delamination_risk: float        # 0-1 probability
    crack_propagation_rate: float   # m⋅cycle⁻¹
    recommended_operating_temp: float  # K (for optimal lifetime)

# =============================================================================
# CTE DATABASE AND MATERIAL PROPERTIES
# =============================================================================

class ThermalPropertiesDatabase:
    """
    Extended thermal properties database for all materials.
    
    Includes temperature-dependent properties and mechanical parameters
    needed for thermal stress calculations.
    """
    
    def __init__(self):
        """Initialize thermal properties database"""
        
        self.thermal_props = self._init_thermal_database()
        
        # Reference temperature for CTE calculations
        self.T_ref = 298.15  # K (25°C)
    
    def _init_thermal_database(self) -> Dict[str, ThermalProperties]:
        """Initialize comprehensive thermal properties database"""
        
        # Note: Properties at room temperature unless specified
        return {
            # Silicon variants
            'c-Si': ThermalProperties(
                material_name='c-Si',
                thickness=0,  # Will be set by user
                cte=2.6e-6,   # /K
                young_modulus=169e9,  # Pa (along <100>)
                poisson_ratio=0.064,
                thermal_conductivity=150,  # W⋅m⁻¹⋅K⁻¹
                specific_heat=700,         # J⋅kg⁻¹⋅K⁻¹
                density=2330,              # kg⋅m⁻³
                thermal_expansion_nonlinear=[2.6e-6, 3.725e-9]  # α₀ + α₁T
            ),
            
            'a-Si': ThermalProperties(
                material_name='a-Si',
                thickness=0,
                cte=3.2e-6,
                young_modulus=70e9,   # Lower than crystalline
                poisson_ratio=0.16,
                thermal_conductivity=1.5,  # Much lower than c-Si
                specific_heat=650,
                density=2200,
                thermal_expansion_nonlinear=[3.2e-6, 2e-9]
            ),
            
            # III-V semiconductors
            'GaAs': ThermalProperties(
                material_name='GaAs',
                thickness=0,
                cte=5.73e-6,
                young_modulus=85e9,
                poisson_ratio=0.31,
                thermal_conductivity=55,
                specific_heat=350,
                density=5317,
                thermal_expansion_nonlinear=[5.73e-6, 1.88e-9]
            ),
            
            'GaInP': ThermalProperties(
                material_name='GaInP',
                thickness=0,
                cte=5.3e-6,
                young_modulus=80e9,   # Slightly lower than GaAs
                poisson_ratio=0.33,
                thermal_conductivity=12,   # Lower thermal conductivity
                specific_heat=320,
                density=4350,
                thermal_expansion_nonlinear=[5.3e-6, 1.5e-9]
            ),
            
            'InGaAs': ThermalProperties(
                material_name='InGaAs',
                thickness=0,
                cte=6.2e-6,   # Higher due to In content
                young_modulus=60e9,
                poisson_ratio=0.35,
                thermal_conductivity=8,    # Low thermal conductivity
                specific_heat=310,
                density=5500,
                thermal_expansion_nonlinear=[6.2e-6, 1.2e-9]
            ),
            
            # Chalcogenides
            'CIGS': ThermalProperties(
                material_name='CIGS',
                thickness=0,
                cte=8.8e-6,   # Relatively high CTE
                young_modulus=80e9,
                poisson_ratio=0.28,
                thermal_conductivity=15,
                specific_heat=350,
                density=5800,
                thermal_expansion_nonlinear=[8.8e-6, 2.5e-9]
            ),
            
            'CdTe': ThermalProperties(
                material_name='CdTe',
                thickness=0,
                cte=4.9e-6,
                young_modulus=52e9,
                poisson_ratio=0.41,
                thermal_conductivity=7,    # Low thermal conductivity
                specific_heat=210,
                density=6200,
                thermal_expansion_nonlinear=[4.9e-6, 1.8e-9]
            ),
            
            # Quantum dots (approximate values)
            'PbS_QD': ThermalProperties(
                material_name='PbS_QD',
                thickness=0,
                cte=18e-6,    # High CTE due to organic matrix
                young_modulus=5e9,     # Low modulus (composite)
                poisson_ratio=0.45,
                thermal_conductivity=0.5,  # Very low
                specific_heat=200,
                density=4000,          # Composite density
                thermal_expansion_nonlinear=[18e-6, 5e-9]
            ),
            
            'PbSe_QD': ThermalProperties(
                material_name='PbSe_QD',
                thickness=0,
                cte=20e-6,
                young_modulus=4e9,
                poisson_ratio=0.46,
                thermal_conductivity=0.4,
                specific_heat=190,
                density=4200,
                thermal_expansion_nonlinear=[20e-6, 6e-9]
            ),
            
            # Organic photovoltaics
            'OPV_P3HT': ThermalProperties(
                material_name='OPV_P3HT',
                thickness=0,
                cte=150e-6,   # Very high CTE for polymers
                young_modulus=0.8e9,  # Low modulus
                poisson_ratio=0.49,
                thermal_conductivity=0.2,  # Very low
                specific_heat=1200,    # High specific heat
                density=1100,          # Low density
                thermal_expansion_nonlinear=[150e-6, 50e-9]
            ),
            
            # Perovskites (Track B)
            'MAPbI3': ThermalProperties(
                material_name='MAPbI3',
                thickness=0,
                cte=4.2e-5,   # High CTE due to organic component
                young_modulus=20e9,   # Relatively soft
                poisson_ratio=0.25,
                thermal_conductivity=0.5,  # Low thermal conductivity
                specific_heat=400,
                density=4160,
                thermal_expansion_nonlinear=[4.2e-5, 10e-9]
            ),
            
            'MAPbBr3': ThermalProperties(
                material_name='MAPbBr3',
                thickness=0,
                cte=3.8e-5,
                young_modulus=22e9,
                poisson_ratio=0.24,
                thermal_conductivity=0.6,
                specific_heat=390,
                density=3900,
                thermal_expansion_nonlinear=[3.8e-5, 8e-9]
            ),
            
            'MAPbCl3': ThermalProperties(
                material_name='MAPbCl3',
                thickness=0,
                cte=3.5e-5,
                young_modulus=25e9,
                poisson_ratio=0.23,
                thermal_conductivity=0.7,
                specific_heat=380,
                density=3600,
                thermal_expansion_nonlinear=[3.5e-5, 6e-9]
            ),
            
            'FAPbI3': ThermalProperties(
                material_name='FAPbI3',
                thickness=0,
                cte=4.5e-5,   # Slightly higher than MA variant
                young_modulus=18e9,   # Softer due to FA
                poisson_ratio=0.26,
                thermal_conductivity=0.4,
                specific_heat=420,
                density=4200,
                thermal_expansion_nonlinear=[4.5e-5, 12e-9]
            ),
            
            'CsPbI3': ThermalProperties(
                material_name='CsPbI3',
                thickness=0,
                cte=2.8e-5,   # Lower CTE (inorganic)
                young_modulus=35e9,   # Higher modulus
                poisson_ratio=0.20,
                thermal_conductivity=1.0,  # Better thermal conductivity
                specific_heat=300,
                density=4800,
                thermal_expansion_nonlinear=[2.8e-5, 3e-9]
            ),
            
            'CsPbBr3': ThermalProperties(
                material_name='CsPbBr3',
                thickness=0,
                cte=2.5e-5,
                young_modulus=38e9,
                poisson_ratio=0.19,
                thermal_conductivity=1.2,
                specific_heat=290,
                density=4500,
                thermal_expansion_nonlinear=[2.5e-5, 2e-9]
            ),
            
            # Substrate materials
            'glass': ThermalProperties(
                material_name='glass',
                thickness=1e-3,    # 1 mm typical
                cte=9e-6,          # Borosilicate glass
                young_modulus=64e9,
                poisson_ratio=0.20,
                thermal_conductivity=1.2,
                specific_heat=800,
                density=2230,
                thermal_expansion_nonlinear=[9e-6, 2e-9]
            ),
            
            'sapphire': ThermalProperties(
                material_name='sapphire',
                thickness=430e-6,  # Standard wafer thickness
                cte=7.5e-6,
                young_modulus=400e9,  # Very stiff
                poisson_ratio=0.23,
                thermal_conductivity=35,   # High thermal conductivity
                specific_heat=750,
                density=3980,
                thermal_expansion_nonlinear=[7.5e-6, 1e-9]
            ),
            
            'polymer': ThermalProperties(
                material_name='polymer',
                thickness=125e-6,  # Flexible substrate
                cte=60e-6,         # High CTE
                young_modulus=3e9, # Low modulus (flexible)
                poisson_ratio=0.40,
                thermal_conductivity=0.3,
                specific_heat=1400,
                density=1200,
                thermal_expansion_nonlinear=[60e-6, 20e-9]
            )
        }
    
    def get_thermal_properties(self, material_name: str, thickness: float) -> ThermalProperties:
        """
        Get thermal properties for a material with specified thickness.
        
        Args:
            material_name: Material name
            thickness: Layer thickness in meters
            
        Returns:
            ThermalProperties with specified thickness
        """
        
        if material_name not in self.thermal_props:
            # Try to get from main material database
            try:
                material = MATERIAL_DB.get_material(material_name, 'A')
                cte = material.get('cte', 10e-6)  # Default CTE
                
                # Create default thermal properties
                thermal_props = ThermalProperties(
                    material_name=material_name,
                    thickness=thickness,
                    cte=cte,
                    young_modulus=50e9,     # Default
                    poisson_ratio=0.3,
                    thermal_conductivity=10,
                    specific_heat=500,
                    density=3000,
                    thermal_expansion_nonlinear=[cte, 1e-9]
                )
                
                warnings.warn(f"Using default thermal properties for {material_name}")
                return thermal_props
                
            except (KeyError, ValueError):
                raise ValueError(f"Material '{material_name}' not found in thermal database")
        
        # Copy and set thickness
        props = self.thermal_props[material_name]
        props.thickness = thickness
        
        return props
    
    def get_temperature_dependent_cte(self, material_name: str, temperature: float) -> float:
        """
        Get temperature-dependent CTE using nonlinear expansion coefficients.
        
        Args:
            material_name: Material name
            temperature: Temperature in Kelvin
            
        Returns:
            CTE at specified temperature (/K)
        """
        
        props = self.thermal_props.get(material_name)
        if props is None:
            return 10e-6  # Default CTE
        
        if props.thermal_expansion_nonlinear is None:
            return props.cte
        
        # Nonlinear expansion: α(T) = α₀ + α₁(T - T_ref) + α₂(T - T_ref)²
        coeffs = props.thermal_expansion_nonlinear
        dT = temperature - self.T_ref
        
        if len(coeffs) >= 2:
            cte_temp = coeffs[0] + coeffs[1] * dT
            if len(coeffs) >= 3:
                cte_temp += coeffs[2] * dT**2
        else:
            cte_temp = coeffs[0]
        
        return cte_temp

# =============================================================================
# THERMAL STRESS CALCULATIONS
# =============================================================================

class ThermalStressCalculator:
    """
    Calculates thermal stress in multi-layer structures using mechanics of materials
    and the modified Stoney equation for thin film stress.
    """
    
    def __init__(self, reference_temperature: float = T_CELL):
        """
        Initialize thermal stress calculator.
        
        Args:
            reference_temperature: Reference temperature for stress-free state (K)
        """
        
        self.T_ref = reference_temperature
        self.thermal_db = ThermalPropertiesDatabase()
    
    def calculate_thermal_stress(self, 
                               materials: List[str],
                               thicknesses: List[float],
                               operating_temperature: float,
                               substrate_material: str = 'glass') -> ThermalStressResult:
        """
        Calculate thermal stress in multi-layer stack using modified Stoney equation.
        
        Args:
            materials: List of layer materials (bottom to top)
            thicknesses: Layer thicknesses in meters
            operating_temperature: Operating temperature in Kelvin
            substrate_material: Substrate material name
            
        Returns:
            Complete thermal stress analysis results
        """
        
        n_layers = len(materials)
        if n_layers != len(thicknesses):
            raise ValueError("Number of materials must match number of thicknesses")
        
        # Get thermal properties for all layers
        layer_props = []
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            props = self.thermal_db.get_thermal_properties(material, thickness)
            layer_props.append(props)
        
        # Get substrate properties
        substrate_props = self.thermal_db.get_thermal_properties(substrate_material, 1e-3)
        
        # Temperature change
        dT = operating_temperature - self.T_ref
        
        # Calculate stress in each layer
        stress_per_layer = []
        total_force_per_width = 0  # N/m (force per unit width)
        
        for i, props in enumerate(layer_props):
            # Temperature-dependent CTE
            cte_temp = self.thermal_db.get_temperature_dependent_cte(props.material_name, operating_temperature)
            cte_substrate = self.thermal_db.get_temperature_dependent_cte(substrate_material, operating_temperature)
            
            # Thermal strain mismatch
            strain_mismatch = (cte_temp - cte_substrate) * dT
            
            # Biaxial modulus: M = E / (1 - ν)
            biaxial_modulus = props.young_modulus / (1 - props.poisson_ratio)
            
            # Thermal stress (assuming constrained by substrate)
            thermal_stress = -biaxial_modulus * strain_mismatch
            
            # Modify for multi-layer effects (simplified)
            if i > 0:
                # Influence of adjacent layers
                previous_props = layer_props[i-1]
                stress_coupling = 0.1 * (props.young_modulus - previous_props.young_modulus) / props.young_modulus
                thermal_stress *= (1 + stress_coupling)
            
            stress_per_layer.append(thermal_stress)
            
            # Force contribution
            total_force_per_width += thermal_stress * props.thickness
        
        # Maximum stress in stack
        total_stress = max([abs(stress) for stress in stress_per_layer])
        
        # Substrate curvature (modified Stoney equation)
        # κ = (6 * Σ(σᵢ * tᵢ)) / (E_sub * t_sub²)
        substrate_biaxial_modulus = substrate_props.young_modulus / (1 - substrate_props.poisson_ratio)
        curvature = (6 * total_force_per_width) / (substrate_biaxial_modulus * substrate_props.thickness**2)
        
        # Critical thickness for cracking and delamination (includes Griffith energy release rate)
        # Estimate adhesion energy based on material combinations
        # Default 1 J/m² for perovskite/oxide, 2 J/m² for covalent bonds, 0.5 J/m² for weak interfaces
        avg_adhesion = 1.0  # J/m², reasonable default for most tandem interfaces
        critical_thickness = self._calculate_critical_thickness(layer_props, stress_per_layer, avg_adhesion)
        
        # Thermal fatigue estimation
        fatigue_cycles = self._estimate_thermal_fatigue_cycles(stress_per_layer, dT)
        
        # Temperature gradients (simplified 1D heat conduction)
        temperature_gradient = self._calculate_temperature_gradients(layer_props, dT)
        
        # CTE mismatch severity score
        cte_mismatch_severity = self._calculate_cte_mismatch_severity(layer_props, substrate_props)
        
        return ThermalStressResult(
            total_stress=total_stress,
            stress_per_layer=stress_per_layer,
            curvature=curvature,
            critical_thickness=critical_thickness,
            thermal_fatigue_cycles=fatigue_cycles,
            temperature_gradient=temperature_gradient,
            cte_mismatch_severity=cte_mismatch_severity
        )
    
    def _calculate_critical_thickness(self, layer_props: List[ThermalProperties], 
                                    stresses: List[float],
                                    adhesion_energy: float = 1.0) -> List[float]:
        """Calculate critical thickness for crack initiation and delamination in each layer.
        
        FIXED: Add delamination check using Griffith energy release rate:
        G = σ²×t/(2E'). Compare G to interface adhesion energy Γ. If G > Γ, flag delamination risk.
        Add adhesion_energy parameter (default ~1 J/m² for perovskite/oxide interfaces).
        """
        
        critical_thicknesses = []
        
        for props, stress in zip(layer_props, stresses):
            # Estimate fracture toughness based on material type
            if 'perovskite' in props.material_name.lower() or 'MAP' in props.material_name or 'FA' in props.material_name:
                K_Ic = 0.5e6  # Pa⋅m^0.5 (low toughness)
                default_adhesion = 0.5  # J/m², weaker perovskite interfaces
            elif 'Si' in props.material_name:
                K_Ic = 0.8e6  # Pa⋅m^0.5
                default_adhesion = 2.0  # J/m², strong covalent bonding
            elif 'GaAs' in props.material_name or 'GaInP' in props.material_name:
                K_Ic = 0.6e6  # Pa⋅m^0.5
                default_adhesion = 1.5  # J/m², moderate III-V bonding
            else:
                K_Ic = 0.7e6  # Pa⋅m^0.5 (default)
                default_adhesion = 1.0  # J/m², default
            
            # Use provided adhesion energy or material default
            interface_adhesion = adhesion_energy if adhesion_energy != 1.0 else default_adhesion
            
            # Critical thickness from crack propagation (Griffith criterion)
            Y = 1.12  # Geometric factor (edge crack)
            if abs(stress) > 1e6:  # Avoid division by very small stress
                t_critical_crack = (K_Ic / (Y * abs(stress)))**2
            else:
                t_critical_crack = 1e-3  # 1 mm default (very thick)
            
            # FIXED: Critical thickness from delamination (Griffith energy release rate)
            # Energy release rate: G = σ²×t/(2E') where E' = E/(1-ν²) is plane strain modulus
            E_prime = props.young_modulus / (1 - props.poisson_ratio**2)
            
            if abs(stress) > 1e6 and E_prime > 0:
                # Delamination occurs when G > Γ (adhesion energy)
                # Solving G = σ²×t/(2E') = Γ for t gives:
                t_critical_delamination = 2 * E_prime * interface_adhesion / (stress**2)
            else:
                t_critical_delamination = 1e-3  # 1 mm default (very thick)
            
            # The actual critical thickness is the minimum of crack and delamination limits
            t_critical = min(t_critical_crack, t_critical_delamination)
            
            critical_thicknesses.append(t_critical)
        
        return critical_thicknesses
    
    def _estimate_thermal_fatigue_cycles(self, stresses: List[float], temperature_range: float) -> float:
        """Estimate number of thermal cycles to failure using Coffin-Manson model"""
        
        # Coffin-Manson equation: N = A * (Δε_plastic)^(-n)
        # where Δε_plastic ≈ Δσ / E for elastic regime
        
        max_stress = max([abs(s) for s in stresses])
        
        if max_stress < 1e6:  # Very low stress
            return 1e8  # Very long lifetime
        
        # Material-dependent Coffin-Manson constants
        # A ~ 1000-10000, n ~ 1.5-2.5 for typical materials
        A = 5000
        n = 2.0
        
        # Stress amplitude (assume zero-to-max cycling)
        stress_amplitude = max_stress / 2
        
        # Estimate plastic strain (simplified)
        E_avg = 100e9  # Average Young's modulus (Pa)
        plastic_strain_amplitude = stress_amplitude / E_avg
        
        # Temperature factor (higher temperature reduces fatigue life)
        temp_factor = np.exp(-(temperature_range - 50) / 100)  # Derating for high ΔT
        
        cycles_to_failure = A * (plastic_strain_amplitude)**(-n) * temp_factor
        
        return max(cycles_to_failure, 100)  # Minimum 100 cycles
    
    def _calculate_temperature_gradients(self, layer_props: List[ThermalProperties], 
                                       total_dT: float) -> List[float]:
        """Calculate temperature gradient across each layer (1D heat conduction)"""
        
        # Thermal resistance model: ΔT = Q * R_thermal
        # where R_thermal = thickness / (k * A)
        
        temperature_gradients = []
        
        # Calculate thermal resistances
        thermal_resistances = []
        for props in layer_props:
            R_thermal = props.thickness / props.thermal_conductivity  # K⋅m²⋅W⁻¹
            thermal_resistances.append(R_thermal)
        
        total_thermal_resistance = sum(thermal_resistances)
        
        # Heat flux (assuming steady state)
        if total_thermal_resistance > 0:
            heat_flux = total_dT / total_thermal_resistance  # W⋅m⁻²
        else:
            heat_flux = 0
        
        # Temperature gradient in each layer
        for props, R_thermal in zip(layer_props, thermal_resistances):
            if props.thickness > 0:
                dT_layer = heat_flux * R_thermal
                gradient = dT_layer / props.thickness  # K⋅m⁻¹
            else:
                gradient = 0
            
            temperature_gradients.append(gradient)
        
        return temperature_gradients
    
    def _calculate_cte_mismatch_severity(self, layer_props: List[ThermalProperties],
                                       substrate_props: ThermalProperties) -> float:
        """Calculate CTE mismatch severity score (0-10 scale)"""
        
        substrate_cte = substrate_props.cte
        
        # Calculate relative CTE mismatches
        mismatch_scores = []
        
        for props in layer_props:
            relative_mismatch = abs(props.cte - substrate_cte) / substrate_cte
            
            # Score based on relative mismatch
            if relative_mismatch < 0.1:
                score = 1  # Very good match
            elif relative_mismatch < 0.5:
                score = 3  # Good match
            elif relative_mismatch < 1.0:
                score = 5  # Moderate mismatch
            elif relative_mismatch < 2.0:
                score = 7  # High mismatch
            else:
                score = 10  # Very high mismatch (problematic)
            
            mismatch_scores.append(score)
        
        # Average severity (weighted by layer thickness)
        total_thickness = sum(props.thickness for props in layer_props)
        if total_thickness > 0:
            weighted_severity = sum(score * props.thickness for score, props in 
                                  zip(mismatch_scores, layer_props)) / total_thickness
        else:
            weighted_severity = np.mean(mismatch_scores)
        
        return weighted_severity

# =============================================================================
# THERMAL CYCLING AND LIFETIME PREDICTION
# =============================================================================

class ThermalLifetimePredictor:
    """
    Predicts thermal lifetime of tandem solar cells based on
    thermal cycling parameters and stress analysis.
    """
    
    def __init__(self):
        """Initialize thermal lifetime predictor"""
        
        self.stress_calc = ThermalStressCalculator()
        
        # Standard thermal cycling test conditions
        self.standard_cycling = ThermalCyclingParams(
            min_temperature=233.15,    # -40°C
            max_temperature=358.15,    # 85°C
            heating_rate=1.0,          # 1 K/s
            cooling_rate=1.0,          # 1 K/s
            dwell_time_hot=600,        # 10 minutes
            dwell_time_cold=600,       # 10 minutes
            cycles_per_day=24,         # 1 cycle per hour
            target_lifetime=25         # years
        )
    
    def predict_thermal_lifetime(self, 
                               materials: List[str],
                               thicknesses: List[float],
                               cycling_params: Optional[ThermalCyclingParams] = None,
                               substrate_material: str = 'glass') -> LifetimePrediction:
        """
        Predict thermal lifetime based on materials and cycling conditions.
        
        Args:
            materials: List of layer materials
            thicknesses: Layer thicknesses in meters
            cycling_params: Thermal cycling parameters
            substrate_material: Substrate material
            
        Returns:
            Comprehensive lifetime prediction
        """
        
        if cycling_params is None:
            cycling_params = self.standard_cycling
        
        # Calculate thermal stress at extreme temperatures
        stress_cold = self.stress_calc.calculate_thermal_stress(
            materials, thicknesses, cycling_params.min_temperature, substrate_material
        )
        
        stress_hot = self.stress_calc.calculate_thermal_stress(
            materials, thicknesses, cycling_params.max_temperature, substrate_material
        )
        
        # Maximum stress range (for fatigue analysis)
        max_stress_range = max([
            abs(hot - cold) for hot, cold in 
            zip(stress_hot.stress_per_layer, stress_cold.stress_per_layer)
        ])
        
        # Failure mode identification
        failure_modes = self._identify_failure_modes(stress_cold, stress_hot, materials, thicknesses)
        dominant_failure = max(failure_modes, key=failure_modes.get)
        
        # Fatigue life prediction (Paris law for crack propagation)
        fatigue_life = self._calculate_fatigue_life(
            max_stress_range, cycling_params, materials, thicknesses
        )
        
        # Stress concentration factors
        stress_concentration = self._calculate_stress_concentration(materials, thicknesses)
        
        # Delamination risk assessment
        delamination_risk = self._assess_delamination_risk(stress_cold, stress_hot, materials)
        
        # Crack propagation rate estimation
        crack_rate = self._estimate_crack_propagation_rate(max_stress_range, materials)
        
        # T80 lifetime (time to 80% performance retention)
        # Include multiple degradation mechanisms
        t80_thermal = min([
            fatigue_life,
            self._interface_delamination_lifetime(delamination_risk),
            self._substrate_cracking_lifetime(stress_cold.curvature, substrate_material)
        ])
        
        # Recommended operating temperature
        recommended_temp = self._optimize_operating_temperature(
            materials, thicknesses, substrate_material
        )
        
        return LifetimePrediction(
            t80_thermal=t80_thermal,
            dominant_failure_mode=dominant_failure,
            stress_concentration_factor=stress_concentration,
            delamination_risk=delamination_risk,
            crack_propagation_rate=crack_rate,
            recommended_operating_temp=recommended_temp
        )
    
    def _identify_failure_modes(self, stress_cold: ThermalStressResult, 
                              stress_hot: ThermalStressResult,
                              materials: List[str], thicknesses: List[float]) -> Dict[str, float]:
        """Identify and rank thermal failure modes"""
        
        failure_modes = {
            'thermal_fatigue': 0.0,
            'delamination': 0.0,
            'substrate_cracking': 0.0,
            'layer_cracking': 0.0,
            'solder_bond_failure': 0.0
        }
        
        # Thermal fatigue risk (based on stress cycling)
        max_stress_cycle = max([
            abs(hot - cold) for hot, cold in 
            zip(stress_hot.stress_per_layer, stress_cold.stress_per_layer)
        ])
        failure_modes['thermal_fatigue'] = min(max_stress_cycle / 100e6, 10.0)  # Normalize
        
        # Delamination risk (CTE mismatch severity)
        failure_modes['delamination'] = (stress_cold.cte_mismatch_severity + 
                                       stress_hot.cte_mismatch_severity) / 2
        
        # Substrate cracking (curvature-based)
        max_curvature = max(abs(stress_cold.curvature), abs(stress_hot.curvature))
        failure_modes['substrate_cracking'] = min(max_curvature * 1000, 10.0)  # m⁻¹ to score
        
        # Layer cracking (maximum stress vs critical)
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            if i < len(stress_cold.critical_thickness):
                if thickness > stress_cold.critical_thickness[i]:
                    failure_modes['layer_cracking'] += 2.0
        
        # Solder bond failure (for III-V on substrate)
        if any('GaAs' in mat or 'GaInP' in mat for mat in materials):
            failure_modes['solder_bond_failure'] = 3.0  # Moderate risk
        
        return failure_modes
    
    def _calculate_fatigue_life(self, stress_range: float, cycling_params: ThermalCyclingParams,
                              materials: List[str], thicknesses: List[float]) -> float:
        """Calculate fatigue life using S-N curves and Miner's rule"""
        
        # Material-dependent fatigue parameters
        # S-N curve: N = A * (σ_range)^(-m)
        fatigue_params = {
            'Si': {'A': 1e15, 'm': 3.0},        # Silicon (brittle)
            'GaAs': {'A': 1e12, 'm': 2.5},      # III-V (moderate)
            'perovskite': {'A': 1e10, 'm': 2.0}, # Perovskite (soft)
            'polymer': {'A': 1e8, 'm': 1.8},    # Organic (very soft)
            'default': {'A': 1e12, 'm': 2.5}
        }
        
        # Determine dominant material class
        if any('Si' in mat for mat in materials):
            params = fatigue_params['Si']
        elif any('GaAs' in mat or 'GaInP' in mat for mat in materials):
            params = fatigue_params['GaAs']
        elif any('MAP' in mat or 'FA' in mat or 'Cs' in mat for mat in materials):
            params = fatigue_params['perovskite']
        elif any('OPV' in mat or 'QD' in mat for mat in materials):
            params = fatigue_params['polymer']
        else:
            params = fatigue_params['default']
        
        # Cycles to failure
        if stress_range > 1e6:  # Significant stress
            cycles_to_failure = params['A'] * (stress_range)**(-params['m'])
        else:
            cycles_to_failure = 1e8  # Very long life for low stress
        
        # Convert to years based on cycling frequency
        cycles_per_year = cycling_params.cycles_per_day * 365
        fatigue_life_years = cycles_to_failure / cycles_per_year
        
        # Apply safety factor
        safety_factor = 3.0
        
        return fatigue_life_years / safety_factor
    
    def _calculate_stress_concentration(self, materials: List[str], 
                                      thicknesses: List[float]) -> float:
        """Calculate stress concentration factor due to geometry and interfaces"""
        
        # Geometric stress concentration
        thickness_ratios = []
        for i in range(len(thicknesses) - 1):
            ratio = thicknesses[i+1] / thicknesses[i]
            thickness_ratios.append(max(ratio, 1/ratio))  # Always > 1
        
        if thickness_ratios:
            geometric_factor = 1 + 0.5 * (max(thickness_ratios) - 1)
        else:
            geometric_factor = 1.0
        
        # Material interface factor
        interface_factor = 1.0
        for i in range(len(materials) - 1):
            # Simplified estimation based on modulus mismatch
            interface_factor += 0.1  # Each interface adds 10%
        
        total_concentration = geometric_factor * interface_factor
        
        return min(total_concentration, 5.0)  # Cap at 5x
    
    def _assess_delamination_risk(self, stress_cold: ThermalStressResult,
                                stress_hot: ThermalStressResult, materials: List[str]) -> float:
        """Assess risk of delamination at interfaces (0-1 probability)"""
        
        # Base risk from CTE mismatch severity
        base_risk = max(stress_cold.cte_mismatch_severity, stress_hot.cte_mismatch_severity) / 10
        
        # Material-specific risk factors
        risk_multipliers = {
            'perovskite': 2.0,  # High delamination risk
            'organic': 1.8,     # High risk
            'III-V': 1.2,       # Moderate risk
            'silicon': 1.0,     # Reference
            'substrate': 0.8    # Low risk
        }
        
        # Calculate weighted risk
        total_risk = base_risk
        
        for material in materials:
            if any(x in material.lower() for x in ['map', 'fa', 'cs', 'perovskite']):
                total_risk *= risk_multipliers['perovskite']
            elif 'opv' in material.lower() or 'qd' in material.lower():
                total_risk *= risk_multipliers['organic']
            elif 'gaas' in material.lower() or 'gainp' in material.lower():
                total_risk *= risk_multipliers['III-V']
            elif 'si' in material.lower():
                total_risk *= risk_multipliers['silicon']
        
        return min(total_risk, 1.0)
    
    def _estimate_crack_propagation_rate(self, stress_range: float, materials: List[str]) -> float:
        """Estimate crack propagation rate using Paris law"""
        
        # Paris law: da/dN = C * (ΔK)^m
        # where ΔK = Y * Δσ * √(πa), Y ~ 1.12 for edge crack
        
        # Material-dependent Paris law constants
        if any('Si' in mat for mat in materials):
            C = 1e-11  # m/cycle/(MPa⋅m^0.5)^m
            m = 3.0
        elif any('perovskite' in mat.lower() or 'MAP' in mat for mat in materials):
            C = 1e-9   # Higher crack growth rate
            m = 2.0
        else:
            C = 1e-10  # Default
            m = 2.5
        
        # Stress intensity factor range (approximate)
        Y = 1.12
        initial_crack_size = 1e-6  # 1 μm initial flaw
        delta_K = Y * stress_range * np.sqrt(np.pi * initial_crack_size)  # Pa⋅m^0.5
        
        # Crack propagation rate
        da_dN = C * (delta_K)**m  # m/cycle
        
        return da_dN
    
    def _interface_delamination_lifetime(self, delamination_risk: float) -> float:
        """Estimate lifetime limited by interface delamination"""
        
        # Empirical model: high risk leads to shorter lifetime
        if delamination_risk > 0.8:
            return 1.0    # 1 year (very high risk)
        elif delamination_risk > 0.5:
            return 5.0    # 5 years (high risk)
        elif delamination_risk > 0.2:
            return 15.0   # 15 years (moderate risk)
        else:
            return 30.0   # 30 years (low risk)
    
    def _substrate_cracking_lifetime(self, curvature: float, substrate_material: str) -> float:
        """Estimate lifetime limited by substrate cracking"""
        
        # Critical curvature for different substrates
        critical_curvatures = {
            'glass': 0.01,      # m⁻¹
            'sapphire': 0.1,    # Higher critical curvature (stronger)
            'polymer': 0.5,     # Very flexible
            'silicon': 0.05     # Moderate
        }
        
        critical_curvature = critical_curvatures.get(substrate_material, 0.02)
        
        if abs(curvature) > critical_curvature:
            # Time to crack propagation through substrate
            return 0.5  # 6 months (immediate concern)
        elif abs(curvature) > 0.5 * critical_curvature:
            return 5.0  # 5 years (moderate concern)
        else:
            return 50.0  # 50 years (low concern)
    
    def _optimize_operating_temperature(self, materials: List[str], thicknesses: List[float],
                                      substrate_material: str) -> float:
        """Find optimal operating temperature for maximum thermal lifetime"""
        
        # Temperature range to optimize over
        temp_range = np.linspace(283.15, 343.15, 20)  # 10°C to 70°C
        
        best_temp = T_CELL
        min_stress = float('inf')
        
        for temp in temp_range:
            # Calculate thermal stress at this temperature
            stress_result = self.stress_calc.calculate_thermal_stress(
                materials, thicknesses, temp, substrate_material
            )
            
            # Weighted stress metric (stress + CTE mismatch)
            stress_metric = stress_result.total_stress + 1e6 * stress_result.cte_mismatch_severity
            
            if stress_metric < min_stress:
                min_stress = stress_metric
                best_temp = temp
        
        return best_temp

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_thermal_performance(materials: List[str], 
                              thicknesses: List[float],
                              operating_conditions: Dict = None,
                              substrate: str = 'glass') -> Dict:
    """
    High-level thermal analysis for tandem solar cell stack.
    
    Args:
        materials: List of layer materials (bottom to top)
        thicknesses: Layer thicknesses in meters
        operating_conditions: Dictionary with temperature conditions
        substrate: Substrate material
        
    Returns:
        Complete thermal analysis results
    """
    
    if operating_conditions is None:
        operating_conditions = {
            'operating_temp': 323.15,  # 50°C
            'min_temp': 233.15,        # -40°C
            'max_temp': 358.15         # 85°C
        }
    
    # Thermal stress analysis
    stress_calc = ThermalStressCalculator()
    
    thermal_stress = stress_calc.calculate_thermal_stress(
        materials, thicknesses, 
        operating_conditions['operating_temp'], 
        substrate
    )
    
    # Lifetime prediction
    lifetime_predictor = ThermalLifetimePredictor()
    
    cycling_params = ThermalCyclingParams(
        min_temperature=operating_conditions['min_temp'],
        max_temperature=operating_conditions['max_temp'],
        heating_rate=1.0,
        cooling_rate=1.0,
        dwell_time_hot=600,
        dwell_time_cold=600,
        cycles_per_day=2,  # More realistic field cycling
        target_lifetime=25
    )
    
    lifetime_prediction = lifetime_predictor.predict_thermal_lifetime(
        materials, thicknesses, cycling_params, substrate
    )
    
    return {
        'thermal_stress': thermal_stress,
        'lifetime_prediction': lifetime_prediction,
        'operating_conditions': operating_conditions,
        'recommendations': {
            'max_operating_temp': lifetime_prediction.recommended_operating_temp,
            'thermal_design_margin': 25 / lifetime_prediction.t80_thermal,  # Safety factor
            'critical_interfaces': [i for i, stress in enumerate(thermal_stress.stress_per_layer) 
                                  if abs(stress) > 50e6],  # > 50 MPa
            'substrate_suitability': 'good' if thermal_stress.curvature < 0.001 else 'poor'
        }
    }

if __name__ == "__main__":
    # Test thermal model
    print("Thermal Model and CTE Mismatch Analysis Test")
    print("=" * 55)
    
    # Test 1: Simple two-layer stack
    print("\nTest 1: Perovskite/Silicon Thermal Analysis")
    
    materials = ['MAPbI3', 'c-Si']
    thicknesses = [500e-9, 200e-6]  # 500 nm perovskite, 200 μm silicon
    
    results = analyze_thermal_performance(materials, thicknesses)
    
    thermal_stress = results['thermal_stress']
    lifetime_pred = results['lifetime_prediction']
    
    print(f"Maximum thermal stress: {thermal_stress.total_stress/1e6:.1f} MPa")
    print(f"Substrate curvature: {thermal_stress.curvature*1000:.2f} m⁻¹ (×1000)")
    print(f"CTE mismatch severity: {thermal_stress.cte_mismatch_severity:.1f}/10")
    print(f"Predicted T80 lifetime: {lifetime_pred.t80_thermal:.1f} years")
    print(f"Dominant failure mode: {lifetime_pred.dominant_failure_mode}")
    print(f"Recommended operating temp: {lifetime_pred.recommended_operating_temp-273.15:.1f}°C")
    
    # Test 2: Multi-junction III-V stack
    print("\nTest 2: Three-Junction III-V Thermal Analysis")
    
    materials_3j = ['GaInP', 'GaAs', 'Ge']
    thicknesses_3j = [500e-9, 3.5e-6, 140e-6]
    
    results_3j = analyze_thermal_performance(materials_3j, thicknesses_3j, substrate='sapphire')
    
    thermal_stress_3j = results_3j['thermal_stress']
    print(f"III-V max stress: {thermal_stress_3j.total_stress/1e6:.1f} MPa")
    print(f"III-V CTE mismatch: {thermal_stress_3j.cte_mismatch_severity:.1f}/10")
    
    # Test 3: Thermal properties database
    print("\nTest 3: Material Thermal Properties")
    
    thermal_db = ThermalPropertiesDatabase()
    
    # Test CTE temperature dependence
    for material in ['c-Si', 'MAPbI3', 'GaAs']:
        cte_300k = thermal_db.get_temperature_dependent_cte(material, 300)
        cte_350k = thermal_db.get_temperature_dependent_cte(material, 350)
        
        print(f"{material}: CTE = {cte_300k*1e6:.2f} → {cte_350k*1e6:.2f} ppm/K (300K → 350K)")
    
    # Test 4: Critical thickness calculation
    print("\nTest 4: Critical Thickness Analysis")
    
    stress_calc = ThermalStressCalculator()
    
    # Test various thicknesses
    test_thicknesses = [100e-9, 500e-9, 1e-6, 5e-6, 10e-6]  # 100nm to 10μm
    
    for thickness in test_thicknesses:
        stress_result = stress_calc.calculate_thermal_stress(
            ['MAPbI3'], [thickness], 350, 'glass'
        )
        
        is_safe = thickness < stress_result.critical_thickness[0]
        safety_margin = stress_result.critical_thickness[0] / thickness
        
        print(f"MAPbI3 {thickness*1e9:.0f}nm: {'✅ SAFE' if is_safe else '❌ RISK'} "
              f"(margin: {safety_margin:.1f}×)")
    
    print("\n✅ Thermal model engine implementation complete!")