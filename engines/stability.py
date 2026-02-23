#!/usr/bin/env python3
"""
Stability and Degradation Mechanisms Engine
===========================================

Models long-term stability and degradation mechanisms in tandem photovoltaic devices:
- Humidity degradation and moisture ingress
- Ion migration in perovskites (Hoke effect, phase segregation)
- Light-induced degradation (photochemical effects)
- Temperature-accelerated degradation (Arrhenius kinetics)
- Encapsulation effectiveness modeling
- PCE/EQE decay ODEs with multiple degradation pathways

References:
- Hoke et al., "Reversible photo-induced trap formation in mixed-halide hybrid perovskites" (2015)
- Aristidou et al., "The role of oxygen in the degradation of methylammonium lead trihalide perovskite photoactive layers" (2017)
- Khenkin et al., "Consensus statement for stability assessment and reporting" (2020)
- Dunlap-Shohl et al., "Synthetic approaches for halide perovskite thin films" (2019)
- Jordan & Kurtz, "Photovoltaic degradation rates" (2013)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings
from dataclasses import dataclass
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize_scalar, curve_fit
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
class DegradationParameters:
    """Parameters for material degradation modeling"""
    material_name: str
    humidity_degradation_rate: float    # /(%RH⋅year) base degradation rate
    humidity_activation_energy: float   # eV (Arrhenius activation energy)
    light_degradation_rate: float       # /(W⋅m⁻²⋅year) at 1000 W/m²
    thermal_degradation_rate: float     # /year at reference temperature
    thermal_activation_energy: float    # eV (thermal degradation)
    oxygen_sensitivity: float           # Relative sensitivity to oxygen
    moisture_permeability: float        # g⋅m⁻¹⋅s⁻¹⋅Pa⁻¹ (water vapor)
    ion_mobility: Optional[float] = None           # m²⋅V⁻¹⋅s⁻¹ (for perovskites)
    phase_segregation_threshold: Optional[float] = None  # Light intensity threshold

@dataclass
class EnvironmentalConditions:
    """Environmental operating conditions for degradation modeling"""
    temperature: float              # K
    relative_humidity: float        # % (0-100)
    light_intensity: float          # W⋅m⁻²
    oxygen_partial_pressure: float  # Pa
    uv_fraction: float              # Fraction of total light that is UV
    encapsulation_quality: float    # 0-1 (0=none, 1=perfect)
    
@dataclass
class StabilityResult:
    """Results from stability analysis"""
    initial_efficiency: float          # Initial PCE
    final_efficiency: float            # Final PCE after degradation time
    t80_years: float                   # Time to 80% performance retention
    t90_years: float                   # Time to 90% performance retention
    degradation_rate_per_year: float   # %/year compound degradation
    dominant_mechanism: str            # Primary degradation mechanism
    degradation_contributions: Dict[str, float]  # Fractional contribution of each mechanism
    projected_lifetime: float          # Years to complete failure

@dataclass
class IonMigrationModel:
    """Ion migration and phase segregation model parameters"""
    ion_species: List[str]             # ['I-', 'Br-', 'MA+', etc.]
    diffusion_coefficients: List[float] # m²⋅s⁻¹ for each species
    activation_energies: List[float]   # eV for thermally activated diffusion
    segregation_driving_force: float   # eV (chemical potential difference)
    critical_field_strength: float     # V⋅m⁻¹ (field for migration)
    reversibility_factor: float        # 0-1 (0=irreversible, 1=fully reversible)

# =============================================================================
# DEGRADATION PARAMETERS DATABASE
# =============================================================================

class DegradationDatabase:
    """
    Comprehensive database of material degradation parameters.
    
    Based on literature data and accelerated aging tests.
    All rates are given at reference conditions (25°C, 50% RH, 1000 W/m²).
    """
    
    def __init__(self):
        """Initialize degradation parameters database"""
        
        self.degradation_params = self._init_degradation_database()
        
        # Reference conditions for rate normalization
        self.T_ref = 298.15      # K (25°C)
        self.RH_ref = 50.0       # % RH
        self.light_ref = 1000.0  # W/m²
    
    def _init_degradation_database(self) -> Dict[str, DegradationParameters]:
        """Initialize comprehensive degradation parameters"""
        
        return {
            # Silicon variants (very stable)
            'c-Si': DegradationParameters(
                material_name='c-Si',
                humidity_degradation_rate=0.001,    # Very low humidity sensitivity
                humidity_activation_energy=0.65,    # eV (surface oxidation)
                light_degradation_rate=0.005,       # Light-induced degradation
                thermal_degradation_rate=0.002,     # Very stable thermally
                thermal_activation_energy=1.2,      # eV (high activation energy)
                oxygen_sensitivity=0.1,             # Low oxygen sensitivity
                moisture_permeability=1e-15         # Very low permeability
            ),
            
            'a-Si': DegradationParameters(
                material_name='a-Si',
                humidity_degradation_rate=0.02,     # Moderate humidity degradation
                humidity_activation_energy=0.4,     # Lower than c-Si
                light_degradation_rate=0.05,        # Staebler-Wronski effect
                thermal_degradation_rate=0.01,      # Annealing effects
                thermal_activation_energy=0.8,
                oxygen_sensitivity=0.3,
                moisture_permeability=5e-14
            ),
            
            # III-V semiconductors (stable but expensive)
            'GaAs': DegradationParameters(
                material_name='GaAs',
                humidity_degradation_rate=0.005,    # Good humidity stability
                humidity_activation_energy=0.8,
                light_degradation_rate=0.01,        # Some UV degradation
                thermal_degradation_rate=0.003,
                thermal_activation_energy=1.1,
                oxygen_sensitivity=0.2,
                moisture_permeability=2e-15
            ),
            
            'GaInP': DegradationParameters(
                material_name='GaInP',
                humidity_degradation_rate=0.008,
                humidity_activation_energy=0.7,
                light_degradation_rate=0.015,       # Slightly more UV sensitive
                thermal_degradation_rate=0.004,
                thermal_activation_energy=1.0,
                oxygen_sensitivity=0.25,
                moisture_permeability=3e-15
            ),
            
            'InGaAs': DegradationParameters(
                material_name='InGaAs',
                humidity_degradation_rate=0.012,    # Less stable due to In
                humidity_activation_energy=0.6,
                light_degradation_rate=0.02,
                thermal_degradation_rate=0.006,
                thermal_activation_energy=0.9,
                oxygen_sensitivity=0.3,
                moisture_permeability=4e-15
            ),
            
            # Chalcogenides (moderate stability)
            'CIGS': DegradationParameters(
                material_name='CIGS',
                humidity_degradation_rate=0.05,     # Moderate humidity sensitivity
                humidity_activation_energy=0.5,
                light_degradation_rate=0.02,        # Light soaking effects
                thermal_degradation_rate=0.01,
                thermal_activation_energy=0.7,
                oxygen_sensitivity=0.4,
                moisture_permeability=1e-13
            ),
            
            'CdTe': DegradationParameters(
                material_name='CdTe',
                humidity_degradation_rate=0.03,
                humidity_activation_energy=0.6,
                light_degradation_rate=0.015,
                thermal_degradation_rate=0.008,
                thermal_activation_energy=0.8,
                oxygen_sensitivity=0.3,
                moisture_permeability=8e-14
            ),
            
            # Quantum dots (stability depends on surface ligands)
            'PbS_QD': DegradationParameters(
                material_name='PbS_QD',
                humidity_degradation_rate=0.2,      # High sensitivity to moisture
                humidity_activation_energy=0.3,     # Low activation energy
                light_degradation_rate=0.1,         # Photo-oxidation
                thermal_degradation_rate=0.05,      # Ligand desorption
                thermal_activation_energy=0.5,
                oxygen_sensitivity=0.8,             # Very sensitive to oxygen
                moisture_permeability=1e-11
            ),
            
            'PbSe_QD': DegradationParameters(
                material_name='PbSe_QD',
                humidity_degradation_rate=0.25,
                humidity_activation_energy=0.28,
                light_degradation_rate=0.12,
                thermal_degradation_rate=0.06,
                thermal_activation_energy=0.45,
                oxygen_sensitivity=0.9,
                moisture_permeability=1.5e-11
            ),
            
            # Organic photovoltaics (low stability)
            'OPV_P3HT': DegradationParameters(
                material_name='OPV_P3HT',
                humidity_degradation_rate=0.5,      # Very high moisture sensitivity
                humidity_activation_energy=0.25,    # Low barrier
                light_degradation_rate=0.3,         # Photo-degradation
                thermal_degradation_rate=0.1,       # Chain scission
                thermal_activation_energy=0.4,
                oxygen_sensitivity=1.0,             # Extremely sensitive
                moisture_permeability=1e-10
            ),
            
            # Perovskites - Track B (stability challenges)
            'MAPbI3': DegradationParameters(
                material_name='MAPbI3',
                humidity_degradation_rate=1.0,      # Very high humidity sensitivity
                humidity_activation_energy=0.2,     # Low activation energy
                light_degradation_rate=0.15,        # Photo-degradation
                thermal_degradation_rate=0.2,       # MA volatilization
                thermal_activation_energy=0.3,
                oxygen_sensitivity=0.7,
                moisture_permeability=5e-11,
                ion_mobility=1e-12,                 # m²⋅V⁻¹⋅s⁻¹
                phase_segregation_threshold=100     # W⋅m⁻² threshold
            ),
            
            'MAPbBr3': DegradationParameters(
                material_name='MAPbBr3',
                humidity_degradation_rate=0.3,      # Better than iodide
                humidity_activation_energy=0.35,
                light_degradation_rate=0.08,        # More stable to light
                thermal_degradation_rate=0.1,
                thermal_activation_energy=0.45,
                oxygen_sensitivity=0.4,
                moisture_permeability=2e-11,
                ion_mobility=5e-13,
                phase_segregation_threshold=500     # Higher threshold
            ),
            
            'MAPbCl3': DegradationParameters(
                material_name='MAPbCl3',
                humidity_degradation_rate=0.1,      # Most stable halide
                humidity_activation_energy=0.5,
                light_degradation_rate=0.05,
                thermal_degradation_rate=0.08,
                thermal_activation_energy=0.6,
                oxygen_sensitivity=0.3,
                moisture_permeability=1e-11,
                ion_mobility=1e-13,
                phase_segregation_threshold=1000    # Very stable
            ),
            
            'FAPbI3': DegradationParameters(
                material_name='FAPbI3',
                humidity_degradation_rate=0.8,      # Better than MA variant
                humidity_activation_energy=0.25,
                light_degradation_rate=0.12,
                thermal_degradation_rate=0.15,      # FA more stable than MA
                thermal_activation_energy=0.4,
                oxygen_sensitivity=0.6,
                moisture_permeability=4e-11,
                ion_mobility=8e-13,
                phase_segregation_threshold=150
            ),
            
            'CsPbI3': DegradationParameters(
                material_name='CsPbI3',
                humidity_degradation_rate=0.2,      # Inorganic - better moisture stability
                humidity_activation_energy=0.45,
                light_degradation_rate=0.06,
                thermal_degradation_rate=0.05,      # No volatile organics
                thermal_activation_energy=0.8,
                oxygen_sensitivity=0.3,
                moisture_permeability=1.5e-11,
                ion_mobility=2e-13,                 # Lower ionic mobility
                phase_segregation_threshold=800
            ),
            
            'CsPbBr3': DegradationParameters(
                material_name='CsPbBr3',
                humidity_degradation_rate=0.05,     # Best overall stability
                humidity_activation_energy=0.6,
                light_degradation_rate=0.03,
                thermal_degradation_rate=0.02,
                thermal_activation_energy=1.0,
                oxygen_sensitivity=0.2,
                moisture_permeability=8e-12,
                ion_mobility=5e-14,
                phase_segregation_threshold=2000   # Very high threshold
            )
        }
    
    def get_degradation_params(self, material_name: str) -> DegradationParameters:
        """Get degradation parameters for a material"""
        
        if material_name not in self.degradation_params:
            # Try to create default parameters from material database
            try:
                material = MATERIAL_DB.get_material(material_name, 'A')
                humidity_score = material.get('humidity_score', 5.0)
                
                # Convert humidity score to degradation rate (inverse relationship)
                humidity_rate = 0.5 / max(humidity_score, 1.0)  # Higher score = lower rate
                
                default_params = DegradationParameters(
                    material_name=material_name,
                    humidity_degradation_rate=humidity_rate,
                    humidity_activation_energy=0.5,
                    light_degradation_rate=0.05,
                    thermal_degradation_rate=0.01,
                    thermal_activation_energy=0.7,
                    oxygen_sensitivity=0.3,
                    moisture_permeability=1e-12
                )
                
                warnings.warn(f"Using estimated degradation parameters for {material_name}")
                return default_params
                
            except (KeyError, ValueError):
                raise ValueError(f"Degradation parameters not found for {material_name}")
        
        return self.degradation_params[material_name]

# =============================================================================
# DEGRADATION KINETICS MODELS
# =============================================================================

class DegradationKineticsCalculator:
    """
    Calculates degradation kinetics using multiple degradation mechanisms:
    - Arrhenius temperature dependence
    - Humidity acceleration (Peck model)
    - Light intensity dependence
    - Oxygen partial pressure effects
    """
    
    def __init__(self, degradation_db: DegradationDatabase = None):
        """Initialize degradation kinetics calculator"""
        
        if degradation_db is None:
            self.degradation_db = DegradationDatabase()
        else:
            self.degradation_db = degradation_db
    
    def calculate_degradation_rate(self, 
                                 material_name: str,
                                 conditions: EnvironmentalConditions) -> Dict[str, float]:
        """
        Calculate material-specific degradation rate under given conditions.
        
        Args:
            material_name: Material identifier
            conditions: Environmental conditions
            
        Returns:
            Dictionary with breakdown of degradation rates by mechanism
        """
        
        params = self.degradation_db.get_degradation_params(material_name)
        
        # Temperature factor (Arrhenius)
        temp_factor_humidity = np.exp(
            -params.humidity_activation_energy * Q / (KB * conditions.temperature) +
            params.humidity_activation_energy * Q / (KB * self.degradation_db.T_ref)
        )
        
        temp_factor_thermal = np.exp(
            -params.thermal_activation_energy * Q / (KB * conditions.temperature) +
            params.thermal_activation_energy * Q / (KB * self.degradation_db.T_ref)
        )
        
        # Humidity acceleration (Peck model: RH^n where n~2-3)
        humidity_power = 2.5  # Typical value for polymeric materials
        humidity_factor = (conditions.relative_humidity / self.degradation_db.RH_ref) ** humidity_power
        
        # Light intensity factor (linear assumption)
        light_factor = conditions.light_intensity / self.degradation_db.light_ref
        
        # UV enhancement factor
        uv_enhancement = 1 + 2 * conditions.uv_fraction  # UV is 2x more damaging
        
        # Oxygen partial pressure effect (square root dependence for diffusion-limited)
        oxygen_factor = np.sqrt(conditions.oxygen_partial_pressure / 21000)  # Normalized to air
        
        # Encapsulation protection factor
        protection_factor = 1 - conditions.encapsulation_quality * 0.9  # Max 90% protection
        
        # Calculate individual degradation rates (/year)
        rate_humidity = (params.humidity_degradation_rate * 
                        temp_factor_humidity * 
                        humidity_factor * 
                        protection_factor)
        
        rate_thermal = (params.thermal_degradation_rate * 
                       temp_factor_thermal)
        
        rate_light = (params.light_degradation_rate * 
                     light_factor * 
                     uv_enhancement * 
                     protection_factor)
        
        rate_oxygen = (params.oxygen_sensitivity * 
                      oxygen_factor * 
                      0.01 *  # Base rate 1%/year
                      protection_factor)
        
        return {
            'humidity': rate_humidity,
            'thermal': rate_thermal,
            'light': rate_light,
            'oxygen': rate_oxygen,
            'total': rate_humidity + rate_thermal + rate_light + rate_oxygen
        }
    
    def calculate_multi_layer_degradation(self,
                                        materials: List[str],
                                        thicknesses: List[float],
                                        conditions: EnvironmentalConditions,
                                        simulation_time: float) -> Dict:
        """
        Calculate degradation for multi-layer stack with interaction effects.
        
        Args:
            materials: List of layer materials
            thicknesses: Layer thicknesses (m)
            conditions: Environmental conditions
            simulation_time: Simulation time (years)
            
        Returns:
            Multi-layer degradation results
        """
        
        layer_degradation = []
        overall_degradation = 0
        
        # Calculate degradation for each layer
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            # Get base degradation rates
            rates = self.calculate_degradation_rate(material, conditions)
            
            # Layer-specific modifications
            modified_conditions = self._modify_conditions_for_layer(
                conditions, materials, thicknesses, i
            )
            
            if modified_conditions != conditions:
                rates = self.calculate_degradation_rate(material, modified_conditions)
            
            # Time-dependent degradation (exponential decay)
            efficiency_retention = np.exp(-rates['total'] * simulation_time)
            
            layer_result = {
                'material': material,
                'thickness': thickness,
                'rates': rates,
                'efficiency_retention': efficiency_retention,
                'conditions': modified_conditions
            }
            
            layer_degradation.append(layer_result)
            
            # Series model: overall efficiency is product of layer efficiencies
            # (for current-matched tandems)
            if i == 0:
                overall_degradation = 1 - efficiency_retention
            else:
                overall_degradation = 1 - (1 - overall_degradation) * efficiency_retention
        
        # Weakest link analysis
        weakest_layer_idx = min(range(len(layer_degradation)), 
                              key=lambda i: layer_degradation[i]['efficiency_retention'])
        
        return {
            'layer_degradation': layer_degradation,
            'overall_efficiency_retention': 1 - overall_degradation,
            'overall_degradation_rate': overall_degradation / simulation_time if simulation_time > 0 else 0,
            'weakest_layer': weakest_layer_idx,
            'dominant_mechanism': self._find_dominant_mechanism(layer_degradation)
        }
    
    def _modify_conditions_for_layer(self, 
                                   base_conditions: EnvironmentalConditions,
                                   materials: List[str], 
                                   thicknesses: List[float],
                                   layer_index: int) -> EnvironmentalConditions:
        """Modify environmental conditions for specific layer in stack"""
        
        # Copy base conditions
        modified = EnvironmentalConditions(
            temperature=base_conditions.temperature,
            relative_humidity=base_conditions.relative_humidity,
            light_intensity=base_conditions.light_intensity,
            oxygen_partial_pressure=base_conditions.oxygen_partial_pressure,
            uv_fraction=base_conditions.uv_fraction,
            encapsulation_quality=base_conditions.encapsulation_quality
        )
        
        # Light attenuation through upper layers
        for i in range(layer_index):
            # Simple exponential attenuation (Beer's law approximation)
            material = materials[i]
            thickness = thicknesses[i]
            
            # Estimate absorption coefficient from material database
            try:
                mat_props = MATERIAL_DB.get_material(material, 'A')
                if 'absorption_coefficient' in mat_props:
                    alpha = mat_props['absorption_coefficient']  # cm⁻¹
                else:
                    alpha = 1e4  # Default
            except:
                alpha = 1e4  # Default
            
            # Light attenuation
            attenuation = np.exp(-alpha * thickness * 100)  # Convert thickness to cm
            modified.light_intensity *= attenuation
        
        # Moisture and oxygen diffusion (simplified)
        if layer_index > 0:
            # Inner layers have reduced exposure
            protection_factor = 0.8 ** layer_index  # Each layer provides 20% protection
            modified.relative_humidity *= protection_factor
            modified.oxygen_partial_pressure *= protection_factor
        
        return modified
    
    def _find_dominant_mechanism(self, layer_degradation: List[Dict]) -> str:
        """Find the dominant degradation mechanism across all layers"""
        
        mechanism_totals = {'humidity': 0, 'thermal': 0, 'light': 0, 'oxygen': 0}
        
        for layer in layer_degradation:
            rates = layer['rates']
            for mechanism in mechanism_totals:
                mechanism_totals[mechanism] += rates[mechanism]
        
        return max(mechanism_totals, key=mechanism_totals.get)

# =============================================================================
# PEROVSKITE-SPECIFIC DEGRADATION MODELS
# =============================================================================

class PerovskiteStabilityCalculator:
    """
    Specialized calculator for perovskite-specific degradation mechanisms:
    - Ion migration and phase segregation (Hoke effect)
    - Moisture-induced decomposition
    - Thermal instability and phase transitions
    - Light-induced trap formation
    """
    
    def __init__(self):
        """Initialize perovskite stability calculator"""
        
        self.degradation_db = DegradationDatabase()
        
        # Ion migration models for common perovskites
        self.ion_migration_models = {
            'MAPbI3': IonMigrationModel(
                ion_species=['I-', 'MA+', 'Pb2+'],
                diffusion_coefficients=[1e-12, 5e-14, 1e-16],  # m²⋅s⁻¹
                activation_energies=[0.58, 0.84, 1.2],         # eV
                segregation_driving_force=0.1,                  # eV
                critical_field_strength=1e5,                    # V⋅m⁻¹
                reversibility_factor=0.7                        # Partially reversible
            ),
            
            'MAPbBr3': IonMigrationModel(
                ion_species=['Br-', 'MA+', 'Pb2+'],
                diffusion_coefficients=[5e-13, 5e-14, 1e-16],
                activation_energies=[0.65, 0.84, 1.2],
                segregation_driving_force=0.05,                 # Lower driving force
                critical_field_strength=2e5,                    # Higher threshold
                reversibility_factor=0.8
            ),
            
            'FAPbI3': IonMigrationModel(
                ion_species=['I-', 'FA+', 'Pb2+'],
                diffusion_coefficients=[8e-13, 2e-14, 1e-16],  # FA less mobile
                activation_energies=[0.6, 0.9, 1.2],
                segregation_driving_force=0.08,
                critical_field_strength=1.5e5,
                reversibility_factor=0.75
            ),
            
            'CsPbI3': IonMigrationModel(
                ion_species=['I-', 'Cs+', 'Pb2+'],
                diffusion_coefficients=[3e-13, 1e-15, 1e-16],  # Cs much less mobile
                activation_energies=[0.7, 1.1, 1.2],
                segregation_driving_force=0.03,                 # Most stable
                critical_field_strength=5e5,
                reversibility_factor=0.9
            )
        }
    
    def calculate_ion_migration_rate(self, 
                                   material_name: str,
                                   temperature: float,
                                   electric_field: float) -> Dict[str, float]:
        """
        Calculate ion migration rates for each species.
        
        Args:
            material_name: Perovskite material name
            temperature: Temperature in Kelvin
            electric_field: Electric field strength in V/m
            
        Returns:
            Migration rates for each ion species
        """
        
        if material_name not in self.ion_migration_models:
            return {'total_migration_rate': 0.0}
        
        model = self.ion_migration_models[material_name]
        migration_rates = {}
        
        for i, (species, D0, Ea) in enumerate(zip(
            model.ion_species, 
            model.diffusion_coefficients, 
            model.activation_energies
        )):
            # Einstein relation: mobility = qD/(kT)
            # Diffusion coefficient with Arrhenius temperature dependence
            D_temp = D0 * np.exp(-Ea * Q / (KB * temperature))
            
            # Drift velocity under electric field: v = μE where μ = qD/(kT)
            mobility = Q * D_temp / (KB * temperature)
            drift_velocity = mobility * electric_field  # m/s
            
            # Migration rate (simplified as flux per unit area)
            migration_rate = drift_velocity * 1e19  # Convert to practical units
            
            migration_rates[species] = migration_rate
        
        migration_rates['total_migration_rate'] = sum(migration_rates.values())
        
        return migration_rates
    
    def calculate_phase_segregation_kinetics(self,
                                           composition: Dict[str, float],
                                           light_intensity: float,
                                           temperature: float) -> Dict:
        """
        Calculate phase segregation kinetics for mixed halide perovskites.
        
        Based on Hoke et al. model for light-induced phase segregation.
        
        Args:
            composition: Halide composition {'I': x, 'Br': y, ...}
            light_intensity: Light intensity in W/m²
            temperature: Temperature in Kelvin
            
        Returns:
            Phase segregation kinetics results
        """
        
        # Check if material is prone to phase segregation
        if 'I' not in composition or 'Br' not in composition:
            return {
                'segregation_rate': 0.0,
                'time_to_segregation': float('inf'),
                'reversible': True,
                'segregated_phases': []
            }
        
        x_I = composition['I']
        x_Br = composition['Br']
        
        # Hoke effect model parameters
        # Segregation is most pronounced around 50:50 I:Br
        segregation_susceptibility = 4 * x_I * x_Br  # Maximum at x=0.5
        
        # Light intensity dependence (threshold behavior)
        light_threshold = 100  # W/m² (typical threshold)
        if light_intensity > light_threshold:
            light_factor = np.log(light_intensity / light_threshold) + 1
        else:
            light_factor = 0  # No segregation below threshold
        
        # Temperature dependence (higher T promotes mixing)
        # Segregation suppressed at higher temperatures
        temp_factor = np.exp(-(temperature - 298.15) / 50)  # Decreases with T
        
        # Overall segregation rate (/s)
        segregation_rate = (1e-6 *                    # Base rate
                           segregation_susceptibility * 
                           light_factor * 
                           temp_factor)
        
        # Time to significant segregation (10% composition change)
        if segregation_rate > 0:
            time_to_segregation = 0.1 / (segregation_rate * 3600)  # hours
        else:
            time_to_segregation = float('inf')
        
        # Reversibility (segregation can be reversed by annealing)
        reversible = temperature > 350  # K (above ~77°C)
        
        # Predicted segregated phases
        if segregation_rate > 1e-8:  # Significant segregation
            if x_I > 0.6:
                segregated_phases = ['I-rich', 'Br-rich']
            else:
                segregated_phases = ['Br-rich', 'I-rich']
        else:
            segregated_phases = []
        
        return {
            'segregation_rate': segregation_rate,
            'time_to_segregation': time_to_segregation,
            'reversible': reversible,
            'segregated_phases': segregated_phases,
            'susceptibility': segregation_susceptibility
        }
    
    def calculate_moisture_degradation_kinetics(self,
                                              material_name: str,
                                              relative_humidity: float,
                                              temperature: float) -> Dict:
        """
        Calculate moisture-induced degradation kinetics for perovskites.
        
        Models the reaction: MAPbI3 + H2O → PbI2 + MAI + HI
        """
        
        params = self.degradation_db.get_degradation_params(material_name)
        
        # Reaction rate constants (Arrhenius form)
        # k = A * exp(-Ea/(kT)) * [H2O]^n
        
        # Water concentration from relative humidity (simplified)
        # Assuming Henry's law and surface adsorption
        water_partial_pressure = relative_humidity * 3167 / 100  # Pa at 25°C
        water_concentration = water_partial_pressure / (KB * temperature)  # m⁻³
        
        # Reaction rate constant
        A_factor = 1e-15  # Pre-exponential factor (m³⋅s⁻¹ per water molecule)
        activation_energy = params.humidity_activation_energy * Q  # J
        
        rate_constant = A_factor * np.exp(-activation_energy / (KB * temperature))
        
        # Reaction rate (assuming first order in water)
        reaction_rate = rate_constant * water_concentration  # s⁻¹
        
        # Time constants
        half_life = np.log(2) / reaction_rate if reaction_rate > 0 else float('inf')
        t90 = np.log(10) / reaction_rate if reaction_rate > 0 else float('inf')
        
        # Product formation
        products = ['PbI2', 'MAI'] if 'MA' in material_name else ['PbI2', 'FAI']
        if 'Br' in material_name:
            products.append('PbBr2')
        
        return {
            'reaction_rate': reaction_rate,
            'half_life_hours': half_life / 3600,
            't90_hours': t90 / 3600,
            'degradation_products': products,
            'rate_constant': rate_constant,
            'water_concentration': water_concentration
        }

# =============================================================================
# LONG-TERM STABILITY PREDICTION
# =============================================================================

class StabilityPredictor:
    """
    Comprehensive stability predictor combining all degradation mechanisms
    into unified lifetime predictions with uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize stability predictor"""
        
        self.degradation_calc = DegradationKineticsCalculator()
        self.perovskite_calc = PerovskiteStabilityCalculator()
    
    def predict_long_term_stability(self,
                                  materials: List[str],
                                  thicknesses: List[float],
                                  operating_conditions: EnvironmentalConditions,
                                  simulation_years: float = 25) -> StabilityResult:
        """
        Predict long-term stability incorporating all degradation mechanisms.
        
        Args:
            materials: List of layer materials
            thicknesses: Layer thicknesses (m)
            operating_conditions: Environmental conditions
            simulation_years: Simulation duration (years)
            
        Returns:
            Comprehensive stability analysis results
        """
        
        # Initial efficiency (assume 100% for relative calculations)
        initial_efficiency = 1.0
        
        # Calculate multi-layer degradation
        degradation_result = self.degradation_calc.calculate_multi_layer_degradation(
            materials, thicknesses, operating_conditions, simulation_years
        )
        
        final_efficiency = degradation_result['overall_efficiency_retention']
        
        # Calculate T80 and T90 lifetimes using exponential extrapolation
        overall_rate = degradation_result['overall_degradation_rate']
        
        if overall_rate > 0:
            # Time to reach 80% and 90% retention
            t80_years = -np.log(0.8) / overall_rate
            t90_years = -np.log(0.9) / overall_rate
            
            # Time to complete failure (20% retention)
            projected_lifetime = -np.log(0.2) / overall_rate
        else:
            t80_years = float('inf')
            t90_years = float('inf')
            projected_lifetime = float('inf')
        
        # Analyze degradation contributions
        degradation_contributions = self._analyze_degradation_contributions(
            degradation_result['layer_degradation']
        )
        
        # Dominant mechanism
        dominant_mechanism = degradation_result['dominant_mechanism']
        
        # Perovskite-specific analysis
        if any('MAP' in mat or 'FA' in mat or 'Cs' in mat for mat in materials):
            perovskite_effects = self._analyze_perovskite_effects(
                materials, operating_conditions
            )
            
            # Modify predictions based on perovskite effects
            if perovskite_effects['has_severe_degradation']:
                t80_years = min(t80_years, perovskite_effects['minimum_lifetime'])
                dominant_mechanism = perovskite_effects['dominant_perovskite_mechanism']
        
        return StabilityResult(
            initial_efficiency=initial_efficiency,
            final_efficiency=final_efficiency,
            t80_years=t80_years,
            t90_years=t90_years,
            degradation_rate_per_year=overall_rate * 100,  # Convert to %/year
            dominant_mechanism=dominant_mechanism,
            degradation_contributions=degradation_contributions,
            projected_lifetime=projected_lifetime
        )
    
    def _analyze_degradation_contributions(self, layer_degradation: List[Dict]) -> Dict[str, float]:
        """Analyze relative contributions of different degradation mechanisms"""
        
        total_rates = {'humidity': 0, 'thermal': 0, 'light': 0, 'oxygen': 0}
        
        # Sum rates across all layers
        for layer in layer_degradation:
            rates = layer['rates']
            for mechanism in total_rates:
                total_rates[mechanism] += rates[mechanism]
        
        # Normalize to get fractional contributions
        total_rate = sum(total_rates.values())
        if total_rate > 0:
            contributions = {mech: rate/total_rate for mech, rate in total_rates.items()}
        else:
            contributions = {mech: 0.25 for mech in total_rates}  # Equal if no degradation
        
        return contributions
    
    def _analyze_perovskite_effects(self, 
                                  materials: List[str],
                                  conditions: EnvironmentalConditions) -> Dict:
        """Analyze perovskite-specific degradation effects"""
        
        perovskite_materials = [mat for mat in materials 
                              if any(x in mat for x in ['MAP', 'FA', 'Cs', 'Pb'])]
        
        if not perovskite_materials:
            return {'has_severe_degradation': False}
        
        severe_degradation = False
        minimum_lifetime = float('inf')
        dominant_mechanism = 'none'
        
        for material in perovskite_materials:
            # Check for moisture sensitivity
            moisture_kinetics = self.perovskite_calc.calculate_moisture_degradation_kinetics(
                material, conditions.relative_humidity, conditions.temperature
            )
            
            if moisture_kinetics['t90_hours'] < 8760:  # Less than 1 year
                severe_degradation = True
                minimum_lifetime = min(minimum_lifetime, moisture_kinetics['t90_hours'] / 8760)
                dominant_mechanism = 'moisture_decomposition'
            
            # Check for phase segregation (if mixed halide)
            if 'I' in material and 'Br' in material:
                # Approximate composition
                composition = {'I': 0.5, 'Br': 0.5}  # Simplified
                
                segregation_kinetics = self.perovskite_calc.calculate_phase_segregation_kinetics(
                    composition, conditions.light_intensity, conditions.temperature
                )
                
                if segregation_kinetics['time_to_segregation'] < 8760:  # Less than 1 year
                    severe_degradation = True
                    segregation_lifetime = segregation_kinetics['time_to_segregation'] / 8760
                    minimum_lifetime = min(minimum_lifetime, segregation_lifetime)
                    dominant_mechanism = 'phase_segregation'
        
        return {
            'has_severe_degradation': severe_degradation,
            'minimum_lifetime': minimum_lifetime,
            'dominant_perovskite_mechanism': dominant_mechanism,
            'perovskite_materials': perovskite_materials
        }
    
    def optimize_encapsulation(self, 
                             materials: List[str],
                             thicknesses: List[float],
                             target_lifetime: float = 25) -> Dict:
        """
        Optimize encapsulation strategy for target lifetime.
        
        Args:
            materials: Layer materials
            thicknesses: Layer thicknesses (m)
            target_lifetime: Target lifetime in years
            
        Returns:
            Optimal encapsulation recommendations
        """
        
        # Test different encapsulation qualities
        encapsulation_levels = np.linspace(0.5, 0.99, 20)
        
        # Standard harsh outdoor conditions
        harsh_conditions = EnvironmentalConditions(
            temperature=333.15,         # 60°C
            relative_humidity=85,       # 85% RH
            light_intensity=1200,       # 1.2 suns peak
            oxygen_partial_pressure=21000,  # Air
            uv_fraction=0.05,          # 5% UV
            encapsulation_quality=0.8   # Will be varied
        )
        
        optimal_encapsulation = 0.8
        achieved_lifetime = 0
        
        for encap_quality in encapsulation_levels:
            test_conditions = EnvironmentalConditions(
                temperature=harsh_conditions.temperature,
                relative_humidity=harsh_conditions.relative_humidity,
                light_intensity=harsh_conditions.light_intensity,
                oxygen_partial_pressure=harsh_conditions.oxygen_partial_pressure,
                uv_fraction=harsh_conditions.uv_fraction,
                encapsulation_quality=encap_quality
            )
            
            stability = self.predict_long_term_stability(
                materials, thicknesses, test_conditions
            )
            
            if stability.t80_years >= target_lifetime:
                optimal_encapsulation = encap_quality
                achieved_lifetime = stability.t80_years
                break
        
        # Encapsulation recommendations
        if optimal_encapsulation < 0.7:
            encap_type = 'basic_polymer'
            cost_factor = 1.0
        elif optimal_encapsulation < 0.85:
            encap_type = 'eva_glass'
            cost_factor = 1.5
        elif optimal_encapsulation < 0.95:
            encap_type = 'advanced_multilayer'
            cost_factor = 2.5
        else:
            encap_type = 'hermetic_sealing'
            cost_factor = 5.0
        
        return {
            'optimal_encapsulation_quality': optimal_encapsulation,
            'encapsulation_type': encap_type,
            'cost_factor': cost_factor,
            'achieved_lifetime': achieved_lifetime,
            'target_lifetime': target_lifetime,
            'meets_target': achieved_lifetime >= target_lifetime
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_material_stability(material_name: str,
                               operating_years: float = 25,
                               conditions: Dict = None) -> Dict:
    """
    Quick stability evaluation for a single material.
    
    Args:
        material_name: Material to evaluate
        operating_years: Operating period (years)
        conditions: Environmental conditions dictionary
        
    Returns:
        Stability assessment summary
    """
    
    if conditions is None:
        conditions = {
            'temperature': 323.15,     # 50°C
            'humidity': 60,            # 60% RH
            'light_intensity': 1000,   # 1 sun
            'uv_fraction': 0.03,      # 3% UV
            'encapsulation': 0.8       # Good encapsulation
        }
    
    env_conditions = EnvironmentalConditions(
        temperature=conditions['temperature'],
        relative_humidity=conditions['humidity'],
        light_intensity=conditions['light_intensity'],
        oxygen_partial_pressure=21000,
        uv_fraction=conditions['uv_fraction'],
        encapsulation_quality=conditions['encapsulation']
    )
    
    # Single layer analysis
    predictor = StabilityPredictor()
    
    stability = predictor.predict_long_term_stability(
        materials=[material_name],
        thicknesses=[1e-6],  # 1 μm reference thickness
        operating_conditions=env_conditions,
        simulation_years=operating_years
    )
    
    # Stability rating
    if stability.t80_years >= 25:
        stability_rating = 'excellent'
    elif stability.t80_years >= 15:
        stability_rating = 'good'
    elif stability.t80_years >= 10:
        stability_rating = 'fair'
    else:
        stability_rating = 'poor'
    
    return {
        'material': material_name,
        'stability_rating': stability_rating,
        't80_years': stability.t80_years,
        'degradation_rate_percent_per_year': stability.degradation_rate_per_year,
        'dominant_mechanism': stability.dominant_mechanism,
        'encapsulation_required': stability.t80_years < 20
    }

if __name__ == "__main__":
    # Test stability analysis
    print("Stability and Degradation Mechanisms Engine Test")
    print("=" * 55)
    
    # Test 1: Single material stability
    print("\nTest 1: Individual Material Stability")
    
    test_materials = ['c-Si', 'MAPbI3', 'CsPbBr3', 'GaAs', 'OPV_P3HT']
    
    for material in test_materials:
        try:
            stability = evaluate_material_stability(material)
            print(f"{material:12} | {stability['stability_rating']:10} | "
                  f"T80: {stability['t80_years']:6.1f}y | "
                  f"Rate: {stability['degradation_rate_percent_per_year']:5.2f}%/y | "
                  f"{stability['dominant_mechanism']}")
        except Exception as e:
            print(f"{material:12} | ERROR: {e}")
    
    # Test 2: Tandem stack stability
    print("\nTest 2: Perovskite/Silicon Tandem Stability")
    
    predictor = StabilityPredictor()
    
    # Harsh outdoor conditions
    harsh_conditions = EnvironmentalConditions(
        temperature=333.15,         # 60°C
        relative_humidity=85,       # 85% RH
        light_intensity=1200,       # 1.2 suns
        oxygen_partial_pressure=21000,
        uv_fraction=0.05,          # 5% UV
        encapsulation_quality=0.8
    )
    
    tandem_stability = predictor.predict_long_term_stability(
        materials=['MAPbI3', 'c-Si'],
        thicknesses=[500e-9, 200e-6],
        operating_conditions=harsh_conditions
    )
    
    print(f"Tandem T80 lifetime: {tandem_stability.t80_years:.1f} years")
    print(f"Final efficiency retention: {tandem_stability.final_efficiency:.3f}")
    print(f"Dominant mechanism: {tandem_stability.dominant_mechanism}")
    print(f"Degradation contributions:")
    for mech, contrib in tandem_stability.degradation_contributions.items():
        print(f"  {mech}: {contrib*100:.1f}%")
    
    # Test 3: Perovskite phase segregation
    print("\nTest 3: Mixed Halide Phase Segregation")
    
    perov_calc = PerovskiteStabilityCalculator()
    
    compositions = [
        {'I': 1.0, 'Br': 0.0},      # Pure iodide
        {'I': 0.7, 'Br': 0.3},      # I-rich
        {'I': 0.5, 'Br': 0.5},      # 50:50 (worst case)
        {'I': 0.3, 'Br': 0.7},      # Br-rich
        {'I': 0.0, 'Br': 1.0}       # Pure bromide
    ]
    
    for comp in compositions:
        segregation = perov_calc.calculate_phase_segregation_kinetics(
            comp, light_intensity=500, temperature=298.15
        )
        
        print(f"I{comp['I']:.1f}Br{comp['Br']:.1f}: "
              f"Time to segregation = {segregation['time_to_segregation']:.1f} hours, "
              f"Susceptibility = {segregation['susceptibility']:.2f}")
    
    # Test 4: Encapsulation optimization
    print("\nTest 4: Encapsulation Optimization")
    
    encap_optimization = predictor.optimize_encapsulation(
        materials=['MAPbI3', 'c-Si'],
        thicknesses=[500e-9, 200e-6],
        target_lifetime=25
    )
    
    print(f"Target: 25-year lifetime")
    print(f"Required encapsulation quality: {encap_optimization['optimal_encapsulation_quality']:.3f}")
    print(f"Recommended type: {encap_optimization['encapsulation_type']}")
    print(f"Cost factor: {encap_optimization['cost_factor']:.1f}×")
    print(f"Achieved lifetime: {encap_optimization['achieved_lifetime']:.1f} years")
    print(f"Meets target: {'✅ YES' if encap_optimization['meets_target'] else '❌ NO'}")
    
    print("\n✅ Stability engine implementation complete!")