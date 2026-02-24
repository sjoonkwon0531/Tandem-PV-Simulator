#!/usr/bin/env python3
"""
Interface Loss and Tunneling Junction Engine
============================================

Models loss mechanisms at interfaces between subcells in tandem photovoltaic devices:
- Recombination losses at tunnel junctions
- Series resistance from tunnel junctions
- Contact resistance effects
- Shunt resistance across interfaces
- Band alignment effects on carrier transport

These losses reduce the theoretical Shockley-Queisser limit to practical efficiencies.

References:
- Essig et al., "Raising the one-sun conversion efficiency of III–V/Si solar cells" (2017)
- Geisz et al., "Six-junction III–V solar cells with 47.1% conversion efficiency" (2020)
- Yamaguchi et al., "Multi-junction III–V solar cells: current status and future potential" (2005)
- Francia et al., "Tunnel junction properties optimized for III-V multijunction solar cells" (2012)
- Green, "Third generation photovoltaics: solar cells for 2020 and beyond" (2001)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, fsolve
from scipy.constants import epsilon_0

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
class TunnelJunctionParams:
    """Parameters for tunnel junction between subcells"""
    top_material: str           # Top subcell material
    bottom_material: str        # Bottom subcell material
    junction_type: str          # 'p++/n++', 'ohmic', 'heterojunction'
    doping_top: float           # Doping concentration top layer (cm⁻³)
    doping_bottom: float        # Doping concentration bottom layer (cm⁻³)
    thickness: float            # Junction thickness (m)
    barrier_height: float       # Tunneling barrier height (eV)
    interface_recombination_velocity: float  # cm/s
    effective_mass_ratio: float # Tunneling effective mass ratio

@dataclass
class InterfaceLossResult:
    """Complete interface loss analysis results"""
    total_voltage_loss: float           # Total voltage loss (V)
    total_series_resistance: float      # Total series resistance (Ω⋅cm²)
    total_shunt_conductance: float      # Total shunt conductance (S⋅cm²)
    interface_losses: List[Dict]        # Per-interface loss breakdown
    tunnel_junction_efficiency: float  # Overall TJ efficiency
    fill_factor_loss: float            # Fill factor degradation
    current_matching_penalty: float    # Current mismatch penalty
    thermal_voltage_loss: float        # Temperature-dependent losses
    
@dataclass
class CarrierTransportModel:
    """Carrier transport parameters at interfaces"""
    diffusion_length_electrons: float  # m
    diffusion_length_holes: float      # m
    surface_recombination_velocity: float  # m/s
    interface_trap_density: float      # cm⁻²
    band_offset_conduction: float      # eV
    band_offset_valence: float         # eV
    
# =============================================================================
# TUNNEL JUNCTION MODELING
# =============================================================================

class TunnelJunctionCalculator:
    """
    Calculates electrical properties of tunnel junctions between subcells.
    
    Implements WKB approximation for tunneling probability and 
    recombination-generation models for interface losses.
    """
    
    def __init__(self, temperature: float = T_CELL):
        """
        Initialize tunnel junction calculator.
        
        Args:
            temperature: Operating temperature in Kelvin
        """
        
        self.T = temperature
        self.Vt = KB * temperature / Q  # Thermal voltage (V)
        
        # Default tunnel junction parameters
        self.default_params = {
            'doping_concentration': 1e19,    # cm⁻³ (heavily doped)
            'effective_mass_ratio': 0.1,    # m*/m₀
            'barrier_height': 0.3,           # eV (typical for III-V)
            'interface_velocity': 1e3,       # cm/s
            'junction_thickness': 10e-9,     # 10 nm
        }
    
    def calculate_tunneling_resistance(self, junction_params: TunnelJunctionParams) -> float:
        """
        Calculate series resistance of tunnel junction using WKB approximation.
        
        FIXED: Include electric field effect on barrier shape for field-assisted tunneling
        (Fowler-Nordheim regime). Effective barrier width under field F: w_eff = φ_B / (q × F) 
        for triangular barrier. Add field parameter and use trapezoidal barrier approximation 
        when field is moderate.
        
        Args:
            junction_params: Tunnel junction parameters
            
        Returns:
            Series resistance in Ω⋅cm²
        """
        
        # Extract parameters
        N_top = junction_params.doping_top        # cm⁻³
        N_bottom = junction_params.doping_bottom  # cm⁻³
        thickness = junction_params.thickness     # m
        barrier_height = junction_params.barrier_height  # eV
        m_eff_ratio = junction_params.effective_mass_ratio
        
        # Effective tunneling mass
        m_eff = m_eff_ratio * 9.109e-31  # kg
        
        # Debye screening lengths
        eps_r = 12  # Relative permittivity (typical for III-V)
        eps = eps_r * epsilon_0
        
        L_D_top = np.sqrt(eps * self.Vt / (Q * N_top * 1e6))      # m
        L_D_bottom = np.sqrt(eps * self.Vt / (Q * N_bottom * 1e6))  # m
        
        # Built-in potential (simplified)
        V_bi = self.Vt * np.log((N_top * N_bottom) / (1e20)**2)  # V
        
        # Electric field across junction
        electric_field = V_bi / thickness  # V/m
        
        # FIXED: Field-assisted tunneling with proper barrier shape
        if thickness < L_D_top + L_D_bottom:
            # Ultra-thin junction - direct tunneling dominates
            # Rectangular barrier approximation
            barrier_width = thickness
            effective_barrier = barrier_height
            tunneling_regime = 'direct'
            
        elif electric_field * thickness > barrier_height:
            # High-field regime: Fowler-Nordheim tunneling
            # Triangular barrier: barrier drops linearly with distance
            # w_eff = φ_B / (q × F) for triangular barrier
            barrier_width = barrier_height / (Q * electric_field * 1e-9)  # Convert eV to J
            if barrier_width > thickness:
                barrier_width = thickness
            effective_barrier = barrier_height  # Peak barrier height
            tunneling_regime = 'fowler_nordheim'
            
        else:
            # Moderate field: Trapezoidal barrier approximation  
            # Barrier has flat top with field-assisted thinning at edges
            field_reduction = min(0.5, Q * electric_field * thickness / (2 * barrier_height))
            barrier_width = thickness * (1 - field_reduction)
            effective_barrier = barrier_height * (1 - 0.5 * field_reduction)
            tunneling_regime = 'trapezoidal'
        
        # WKB tunneling probability calculation depends on barrier shape
        if tunneling_regime == 'fowler_nordheim':
            # For triangular barrier: ∫√(2m(V-qFx)) dx from 0 to w_eff
            # Analytical result for triangular barrier
            gamma_integral = (4/3) * np.sqrt(2 * m_eff) / H * (barrier_height * Q)**(3/2) / (Q * electric_field)
            tunneling_probability = np.exp(-2 * gamma_integral)
            
        elif tunneling_regime == 'trapezoidal':
            # Trapezoidal barrier: flat top + field-assisted edges
            # Approximate as effective rectangular barrier
            gamma = 2 * np.sqrt(2 * m_eff * effective_barrier * Q) / H  # m⁻¹
            tunneling_probability = np.exp(-gamma * barrier_width)
            
        else:  # direct tunneling
            # Rectangular barrier
            gamma = 2 * np.sqrt(2 * m_eff * effective_barrier * Q) / H  # m⁻¹
            tunneling_probability = np.exp(-gamma * barrier_width)
        
        # Current density from Tsu-Esaki formula
        # J = (q*m_eff*k*T/(2*π*ħ²)) * ln(1 + exp(qV/kT)) * T_tunnel
        prefactor = Q * m_eff * KB * self.T / (2 * np.pi * H**2)  # A⋅m⁻²⋅V⁻¹
        
        # Field enhancement factor for high-field regime
        if tunneling_regime == 'fowler_nordheim':
            # Field enhancement increases prefactor
            field_enhancement = 1 + 0.1 * electric_field / 1e7  # Modest enhancement per 10 MV/m
            prefactor *= field_enhancement
        
        # Differential conductance (dJ/dV at V=0)
        conductance_per_area = prefactor * tunneling_probability  # S⋅m⁻²
        
        # Ensure minimum conductance for numerical stability
        min_conductance = 1e-10  # S⋅m⁻²
        conductance_per_area = max(conductance_per_area, min_conductance)
        
        # Convert to resistance
        resistance_area = 1 / conductance_per_area * 1e-4  # Ω⋅cm²
        
        return resistance_area
    
    def calculate_recombination_current(self, junction_params: TunnelJunctionParams,
                                      current_density: float,
                                      defect_density: float = 1e12,
                                      trap_energy: float = 0.5) -> float:
        """
        Calculate recombination current at tunnel junction interface using proper SRH theory.
        
        FIXED: Add Shockley-Read-Hall recombination at interface:
        R_SRH = (n×p - ni²) / (τ_p×(n+n1) + τ_n×(p+p1))
        where n1, p1 are trap-level carrier densities. Add defect_density and trap_energy 
        parameters with reasonable defaults.
        
        Args:
            junction_params: Tunnel junction parameters
            current_density: Operating current density (A/cm²)
            defect_density: Interface defect density (cm⁻²)
            trap_energy: Trap energy level relative to intrinsic level (eV)
            
        Returns:
            Recombination current density (A/cm²)
        """
        
        # Interface recombination velocity (for simple surface recombination)
        S_eff = junction_params.interface_recombination_velocity  # cm/s
        
        # Material properties for SRH recombination
        Eg_eff = 1.2  # eV, effective bandgap at interface (reasonable for most junctions)
        ni_eff = 1e10 * np.exp(-Eg_eff / (2 * self.Vt))  # cm⁻³, intrinsic carrier concentration
        
        # Excess carrier concentration from current density
        # Δn ≈ Δp ≈ J / (q * v_th) where v_th is thermal velocity
        v_th = np.sqrt(3 * KB * self.T / 9.109e-31) * 1e2  # cm/s, electron thermal velocity
        excess_carriers = current_density / (Q * v_th)  # cm⁻³
        
        # Total carrier concentrations (equilibrium + excess)
        n_total = ni_eff + excess_carriers  # electrons
        p_total = ni_eff + excess_carriers  # holes (assume charge neutrality)
        
        # FIXED: Proper SRH recombination parameters
        # Trap energy level carrier concentrations
        n1 = ni_eff * np.exp(trap_energy / self.Vt)   # electrons when trap is occupied
        p1 = ni_eff * np.exp(-trap_energy / self.Vt)  # holes when trap is empty
        
        # SRH lifetimes from defect density and capture cross-sections
        # τ = 1 / (σ × v_th × N_t) where σ is capture cross-section, N_t is trap density
        sigma_n = 1e-15  # cm², electron capture cross-section (typical)
        sigma_p = 1e-15  # cm², hole capture cross-section
        
        # Convert defect density from cm⁻² to cm⁻³ assuming interface thickness
        interface_thickness = 1e-7  # 1 nm effective interface thickness
        N_t = defect_density / interface_thickness  # cm⁻³
        
        # SRH lifetimes
        tau_n = 1 / (sigma_n * v_th * N_t)  # electron lifetime
        tau_p = 1 / (sigma_p * v_th * N_t)  # hole lifetime
        
        # SRH recombination rate (carriers per cm³ per second)
        if (tau_p * (n_total + n1) + tau_n * (p_total + p1)) > 0:
            R_SRH = (n_total * p_total - ni_eff**2) / (tau_p * (n_total + n1) + tau_n * (p_total + p1))
        else:
            R_SRH = 0.0
        
        # Convert recombination rate to current density
        # J_rec = q × R_SRH × interface_thickness
        recombination_current_SRH = Q * R_SRH * interface_thickness * 1e1  # A/cm² (unit conversion)
        
        # Also include simple surface recombination (empirical)
        # J_rec_surface = q * S_eff * excess_carriers
        recombination_current_surface = Q * S_eff * excess_carriers * 1e-3  # A/cm²
        
        # Total recombination current is sum of SRH and surface recombination
        total_recombination_current = recombination_current_SRH + recombination_current_surface
        
        return total_recombination_current
    
    def calculate_band_alignment_loss(self, top_material: str, bottom_material: str,
                                    track: str = 'A') -> Dict[str, float]:
        """
        Calculate voltage loss from band alignment at heterojunction interface.
        
        Args:
            top_material: Top subcell material name
            bottom_material: Bottom subcell material name
            track: Material track
            
        Returns:
            Dictionary with band alignment loss components
        """
        
        # Get material properties
        try:
            top_props = MATERIAL_DB.get_material(top_material, track)
            bottom_props = MATERIAL_DB.get_material(bottom_material, track)
        except (KeyError, ValueError):
            warnings.warn(f"Material properties not found, using defaults")
            return {'conduction_offset': 0.1, 'valence_offset': 0.1, 'voltage_loss': 0.05}
        
        # Bandgap difference
        Eg_top = top_props['bandgap']
        Eg_bottom = bottom_props['bandgap']
        
        # Electron affinity (approximation based on bandgap and material type)
        # For III-V semiconductors: χ ≈ 4.0 - 0.5*Eg (rough approximation)
        # For perovskites: χ ≈ 3.9 - 0.2*Eg
        
        if track == 'A':
            # Multi-material track
            if 'GaAs' in top_material or 'GaAs' in bottom_material:
                chi_top = 4.0 - 0.5 * Eg_top
                chi_bottom = 4.0 - 0.5 * Eg_bottom
            else:
                # General approximation
                chi_top = 4.2 - 0.3 * Eg_top
                chi_bottom = 4.2 - 0.3 * Eg_bottom
        else:
            # Perovskite track
            chi_top = 3.9 - 0.2 * Eg_top
            chi_bottom = 3.9 - 0.2 * Eg_bottom
        
        # Band offsets
        conduction_offset = abs(chi_top - chi_bottom)  # eV
        valence_offset = abs((chi_top + Eg_top) - (chi_bottom + Eg_bottom))  # eV
        
        # Voltage loss from carrier thermalization and interface states
        # Empirical relation: V_loss ≈ 0.1 * max(ΔE_c, ΔE_v)
        voltage_loss = 0.1 * max(conduction_offset, valence_offset)  # V
        
        return {
            'conduction_offset': conduction_offset,
            'valence_offset': valence_offset,
            'voltage_loss': voltage_loss,
            'interface_dipole': 0.02 * abs(Eg_top - Eg_bottom)  # eV, approximation
        }

# =============================================================================
# INTERFACE LOSS AGGREGATION
# =============================================================================

class InterfaceLossCalculator:
    """
    Aggregates all interface loss mechanisms for multi-junction solar cells.
    
    Combines tunnel junction losses, recombination losses, and other
    interface effects to predict overall device performance.
    """
    
    def __init__(self, temperature: float = T_CELL):
        """Initialize interface loss calculator"""
        
        self.T = temperature
        self.Vt = KB * temperature / Q
        self.tj_calc = TunnelJunctionCalculator(temperature)
    
    def calculate_total_interface_losses(self, 
                                       materials: List[str],
                                       bandgaps: List[float],
                                       current_densities: List[float],
                                       track: str = 'A',
                                       custom_tj_params: Optional[List[TunnelJunctionParams]] = None) -> InterfaceLossResult:
        """
        Calculate total interface losses for a multi-junction stack.
        
        Args:
            materials: List of subcell materials (top to bottom)
            bandgaps: Bandgaps of each subcell (eV)
            current_densities: Current density of each subcell (mA/cm²)
            track: Material track
            custom_tj_params: Custom tunnel junction parameters
            
        Returns:
            Complete interface loss analysis
        """
        
        n_junctions = len(materials)
        if n_junctions < 2:
            # Single junction - no interface losses
            return InterfaceLossResult(
                total_voltage_loss=0.0, total_series_resistance=0.0,
                total_shunt_conductance=0.0, interface_losses=[],
                tunnel_junction_efficiency=1.0, fill_factor_loss=0.0,
                current_matching_penalty=0.0, thermal_voltage_loss=0.0
            )
        
        # Number of interfaces (N-1 for N junctions)
        n_interfaces = n_junctions - 1
        
        interface_losses = []
        total_voltage_loss = 0.0
        total_series_resistance = 0.0
        total_shunt_conductance = 0.0
        
        # Calculate losses for each interface
        for i in range(n_interfaces):
            top_material = materials[i]
            bottom_material = materials[i+1]
            top_bandgap = bandgaps[i]
            bottom_bandgap = bandgaps[i+1]
            
            # Use average current density for interface
            avg_current = (current_densities[i] + current_densities[i+1]) / 2 * 1e-1  # A/cm²
            
            # Get or create tunnel junction parameters
            if custom_tj_params and i < len(custom_tj_params):
                tj_params = custom_tj_params[i]
            else:
                tj_params = self._create_default_tj_params(
                    top_material, bottom_material, top_bandgap, bottom_bandgap, track
                )
            
            # Calculate individual loss components
            interface_loss = self._calculate_single_interface_loss(
                tj_params, avg_current, i
            )
            
            interface_losses.append(interface_loss)
            
            # Accumulate total losses
            total_voltage_loss += interface_loss['voltage_loss']
            total_series_resistance += interface_loss['series_resistance']
            total_shunt_conductance += interface_loss['shunt_conductance']
        
        # Calculate aggregate effects
        tj_efficiency = self._calculate_tunnel_junction_efficiency(interface_losses)
        fill_factor_loss = self._calculate_fill_factor_loss(total_series_resistance, avg_current)
        current_matching_penalty = self._calculate_current_matching_penalty(current_densities)
        thermal_voltage_loss = self._calculate_thermal_voltage_loss(n_interfaces)
        
        return InterfaceLossResult(
            total_voltage_loss=total_voltage_loss,
            total_series_resistance=total_series_resistance,
            total_shunt_conductance=total_shunt_conductance,
            interface_losses=interface_losses,
            tunnel_junction_efficiency=tj_efficiency,
            fill_factor_loss=fill_factor_loss,
            current_matching_penalty=current_matching_penalty,
            thermal_voltage_loss=thermal_voltage_loss
        )
    
    def _create_default_tj_params(self, top_material: str, bottom_material: str,
                                 top_bandgap: float, bottom_bandgap: float,
                                 track: str) -> TunnelJunctionParams:
        """Create default tunnel junction parameters based on materials"""
        
        # Default heavily doped tunnel junction
        default_doping = 5e19  # cm⁻³
        
        # Barrier height depends on material combination
        if track == 'A':
            # Multi-material: higher barriers typically
            barrier_height = 0.4 + 0.1 * abs(top_bandgap - bottom_bandgap)
            interface_velocity = 5e2  # cm/s
        else:
            # Perovskite: potentially lower barriers due to similar crystal structure  
            barrier_height = 0.2 + 0.05 * abs(top_bandgap - bottom_bandgap)
            interface_velocity = 1e3  # cm/s (potentially higher due to defects)
        
        # Junction type classification
        if abs(top_bandgap - bottom_bandgap) < 0.2:
            junction_type = 'p++/n++'  # Homojunction-like
            thickness = 10e-9  # 10 nm
        elif abs(top_bandgap - bottom_bandgap) < 0.5:
            junction_type = 'heterojunction'
            thickness = 15e-9  # 15 nm
        else:
            junction_type = 'ohmic'  # May need ohmic contact
            thickness = 20e-9  # 20 nm
        
        return TunnelJunctionParams(
            top_material=top_material,
            bottom_material=bottom_material,
            junction_type=junction_type,
            doping_top=default_doping,
            doping_bottom=default_doping,
            thickness=thickness,
            barrier_height=barrier_height,
            interface_recombination_velocity=interface_velocity,
            effective_mass_ratio=0.1
        )
    
    def _calculate_single_interface_loss(self, tj_params: TunnelJunctionParams,
                                       current_density: float, interface_index: int) -> Dict:
        """Calculate losses for a single interface"""
        
        # Tunneling resistance
        R_tunnel = self.tj_calc.calculate_tunneling_resistance(tj_params)
        
        # Recombination current with improved SRH model
        # Use material-dependent defect density estimates
        if 'perovskite' in tj_params.top_material.lower() or 'perovskite' in tj_params.bottom_material.lower():
            defect_density = 5e12  # Higher defect density for perovskites
            trap_energy = 0.4      # eV, typical for perovskite defects
        elif 'III-V' in tj_params.top_material or 'GaAs' in tj_params.top_material or 'GaInP' in tj_params.top_material:
            defect_density = 1e11  # Lower defect density for high-quality III-V
            trap_energy = 0.6      # eV, mid-gap states
        else:
            defect_density = 1e12  # Default moderate defect density
            trap_energy = 0.5      # eV, mid-gap
        
        J_rec = self.tj_calc.calculate_recombination_current(tj_params, current_density, defect_density, trap_energy)
        
        # Band alignment loss
        band_loss = self.tj_calc.calculate_band_alignment_loss(
            tj_params.top_material, tj_params.bottom_material
        )
        
        # Voltage losses
        voltage_loss_tunnel = current_density * R_tunnel  # V (from series resistance)
        voltage_loss_recombination = self.Vt * np.log(1 + J_rec / current_density) if current_density > 0 else 0
        voltage_loss_band = band_loss['voltage_loss']
        
        total_voltage_loss = voltage_loss_tunnel + voltage_loss_recombination + voltage_loss_band
        
        # Shunt conductance (leakage current effects)
        # Empirical relation based on interface quality
        interface_quality = 1 / tj_params.interface_recombination_velocity  # s/cm
        shunt_conductance = 1e-6 / max(interface_quality, 1e-4)  # S/cm²
        
        return {
            'interface_index': interface_index,
            'top_material': tj_params.top_material,
            'bottom_material': tj_params.bottom_material,
            'junction_type': tj_params.junction_type,
            'series_resistance': R_tunnel,
            'recombination_current': J_rec,
            'voltage_loss': total_voltage_loss,
            'voltage_loss_tunnel': voltage_loss_tunnel,
            'voltage_loss_recombination': voltage_loss_recombination,
            'voltage_loss_band_alignment': voltage_loss_band,
            'shunt_conductance': shunt_conductance,
            'band_offsets': band_loss
        }
    
    def _calculate_tunnel_junction_efficiency(self, interface_losses: List[Dict]) -> float:
        """Calculate overall tunnel junction efficiency"""
        
        if not interface_losses:
            return 1.0
        
        # Product of individual interface efficiencies
        # η_interface ≈ 1 - (V_loss / V_typical) where V_typical ~ 0.5V per junction
        total_efficiency = 1.0
        
        for loss in interface_losses:
            V_loss = loss['voltage_loss']
            interface_efficiency = max(1 - V_loss / 0.5, 0.5)  # Cap minimum at 50%
            total_efficiency *= interface_efficiency
        
        return total_efficiency
    
    def _calculate_fill_factor_loss(self, total_rs: float, current_density: float) -> float:
        """Calculate fill factor degradation from series resistance"""
        
        if total_rs <= 0 or current_density <= 0:
            return 0.0
        
        # Empirical relation: ΔFF ≈ -Rs * Jsc / Voc
        # Assume typical Voc ~ 2.5V for multi-junction
        typical_voc = 2.5  # V
        ff_loss = total_rs * current_density * 1e-1 / typical_voc  # Fractional loss
        
        return min(ff_loss, 0.5)  # Cap at 50% FF loss
    
    def _calculate_current_matching_penalty(self, current_densities: List[float]) -> float:
        """Calculate penalty from current mismatch between subcells"""
        
        if len(current_densities) < 2:
            return 0.0
        
        # Current matching penalty based on standard deviation
        mean_current = np.mean(current_densities)
        std_current = np.std(current_densities)
        
        if mean_current > 0:
            relative_mismatch = std_current / mean_current
            # Penalty increases quadratically with mismatch
            penalty = (relative_mismatch)**2 * 0.1  # Up to 10% penalty
        else:
            penalty = 0.0
        
        return min(penalty, 0.2)  # Cap at 20%
    
    def _calculate_thermal_voltage_loss(self, n_interfaces: int) -> float:
        """Calculate additional voltage loss from thermal effects"""
        
        # Each interface contributes thermal voltage loss
        # Approximately kT/q per interface at room temperature
        thermal_loss_per_interface = self.Vt * 0.1  # 10% of thermal voltage
        
        return n_interfaces * thermal_loss_per_interface

# =============================================================================
# OPTIMIZATION AND TRADE-OFFS
# =============================================================================

def optimize_tunnel_junction_design(top_material: str, bottom_material: str,
                                   target_current: float, track: str = 'A') -> Dict:
    """
    Optimize tunnel junction design for minimum losses.
    
    Args:
        top_material: Top subcell material
        bottom_material: Bottom subcell material  
        target_current: Target current density (mA/cm²)
        track: Material track
        
    Returns:
        Optimal tunnel junction parameters and expected performance
    """
    
    tj_calc = TunnelJunctionCalculator()
    
    # Define optimization variables and bounds
    # Variables: [log10(doping), thickness_nm, barrier_height]
    bounds = [
        (18, 20),      # log10(doping): 10^18 to 10^20 cm⁻³
        (5, 50),       # thickness: 5 to 50 nm
        (0.1, 0.8)     # barrier height: 0.1 to 0.8 eV
    ]
    
    def objective(params):
        """Minimize total voltage loss"""
        
        log_doping, thickness_nm, barrier_height = params
        doping = 10**log_doping
        thickness = thickness_nm * 1e-9
        
        # Create tunnel junction parameters
        tj_params = TunnelJunctionParams(
            top_material=top_material,
            bottom_material=bottom_material,
            junction_type='p++/n++',
            doping_top=doping,
            doping_bottom=doping,
            thickness=thickness,
            barrier_height=barrier_height,
            interface_recombination_velocity=1e3,
            effective_mass_ratio=0.1
        )
        
        try:
            # Calculate losses
            R_tunnel = tj_calc.calculate_tunneling_resistance(tj_params)
            J_rec = tj_calc.calculate_recombination_current(tj_params, target_current * 1e-1)
            
            # Total voltage loss
            V_loss_tunnel = target_current * 1e-1 * R_tunnel
            V_loss_rec = tj_calc.Vt * np.log(1 + J_rec / (target_current * 1e-1)) if target_current > 0 else 0
            
            total_loss = V_loss_tunnel + V_loss_rec
            
            # Add penalty for extreme parameters
            if doping > 1e20:
                total_loss += 0.1  # Penalty for very high doping
            if thickness < 8e-9:
                total_loss += 0.05  # Penalty for very thin junction
            
            return total_loss
            
        except:
            return 1.0  # High penalty for failed calculation
    
    # Simple grid search for optimization
    best_loss = float('inf')
    best_params = None
    
    for log_doping in np.linspace(18.5, 19.5, 10):
        for thickness_nm in np.linspace(8, 25, 10):
            for barrier_height in np.linspace(0.2, 0.6, 8):
                params = [log_doping, thickness_nm, barrier_height]
                loss = objective(params)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
    
    if best_params is None:
        raise RuntimeError("Optimization failed")
    
    # Extract optimal parameters
    opt_doping = 10**best_params[0]
    opt_thickness = best_params[1] * 1e-9
    opt_barrier = best_params[2]
    
    # Create optimal tunnel junction
    optimal_tj = TunnelJunctionParams(
        top_material=top_material,
        bottom_material=bottom_material,
        junction_type='p++/n++',
        doping_top=opt_doping,
        doping_bottom=opt_doping,
        thickness=opt_thickness,
        barrier_height=opt_barrier,
        interface_recombination_velocity=1e3,
        effective_mass_ratio=0.1
    )
    
    # Calculate performance
    R_opt = tj_calc.calculate_tunneling_resistance(optimal_tj)
    J_rec_opt = tj_calc.calculate_recombination_current(optimal_tj, target_current * 1e-1)
    
    return {
        'optimal_parameters': optimal_tj,
        'tunnel_resistance': R_opt,
        'recombination_current': J_rec_opt,
        'voltage_loss': best_loss,
        'doping_concentration': opt_doping,
        'thickness_nm': best_params[1],
        'barrier_height': opt_barrier
    }

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_interface_models() -> Dict:
    """Validate interface loss models against literature data"""
    
    # Test case: GaInP/GaAs tunnel junction (common in III-V tandems)
    tj_calc = TunnelJunctionCalculator()
    
    # Literature reference: typical GaInP/GaAs TJ
    reference_tj = TunnelJunctionParams(
        top_material='GaInP',
        bottom_material='GaAs',
        junction_type='p++/n++',
        doping_top=1e19,
        doping_bottom=1e19,
        thickness=15e-9,
        barrier_height=0.3,
        interface_recombination_velocity=500,
        effective_mass_ratio=0.1
    )
    
    # Calculate properties
    R_tunnel = tj_calc.calculate_tunneling_resistance(reference_tj)
    J_rec = tj_calc.calculate_recombination_current(reference_tj, 0.14)  # 14 mA/cm²
    
    # Literature comparison
    literature_values = {
        'tunnel_resistance_literature': 0.05,  # Ω⋅cm² (typical for good TJ)
        'voltage_loss_literature': 0.02,       # V (typical)
    }
    
    calculated_values = {
        'tunnel_resistance_calculated': R_tunnel,
        'recombination_current_calculated': J_rec,
        'voltage_loss_calculated': 0.14 * R_tunnel + tj_calc.Vt * np.log(1 + J_rec/0.14)
    }
    
    # Test multi-junction stack
    interface_calc = InterfaceLossCalculator()
    
    # Three-junction test case
    materials = ['GaInP', 'GaAs', 'Ge']
    bandgaps = [1.81, 1.42, 0.67]
    currents = [14, 14, 28]  # mA/cm² (current matched for top two)
    
    total_losses = interface_calc.calculate_total_interface_losses(
        materials, bandgaps, currents, track='A'
    )
    
    validation_results = {
        **calculated_values,
        **literature_values,
        'multi_junction_test': {
            'total_voltage_loss': total_losses.total_voltage_loss,
            'total_series_resistance': total_losses.total_series_resistance,
            'tunnel_junction_efficiency': total_losses.tunnel_junction_efficiency,
            'fill_factor_loss': total_losses.fill_factor_loss
        }
    }
    
    return validation_results

if __name__ == "__main__":
    # Test interface loss calculations
    print("Interface Loss and Tunneling Junction Engine Test")
    print("=" * 55)
    
    # Test 1: Single tunnel junction
    print("\nTest 1: GaInP/GaAs Tunnel Junction")
    
    tj_calc = TunnelJunctionCalculator()
    
    test_tj = TunnelJunctionParams(
        top_material='GaInP',
        bottom_material='GaAs', 
        junction_type='p++/n++',
        doping_top=5e19,
        doping_bottom=5e19,
        thickness=12e-9,
        barrier_height=0.25,
        interface_recombination_velocity=300,
        effective_mass_ratio=0.08
    )
    
    R_tunnel = tj_calc.calculate_tunneling_resistance(test_tj)
    J_rec = tj_calc.calculate_recombination_current(test_tj, 0.15)
    
    print(f"Tunnel resistance: {R_tunnel:.4f} Ω⋅cm²")
    print(f"Recombination current: {J_rec*1000:.2f} mA/cm²")
    print(f"Voltage loss (15 mA/cm²): {0.15 * R_tunnel:.3f} V")
    
    # Test 2: Multi-junction interface losses
    print("\nTest 2: Three-Junction Interface Analysis")
    
    interface_calc = InterfaceLossCalculator()
    
    materials_3j = ['GaInP', 'GaAs', 'Ge']
    bandgaps_3j = [1.81, 1.42, 0.67]
    currents_3j = [14, 14, 28]
    
    losses_3j = interface_calc.calculate_total_interface_losses(
        materials_3j, bandgaps_3j, currents_3j
    )
    
    print(f"Total voltage loss: {losses_3j.total_voltage_loss:.3f} V")
    print(f"Total series resistance: {losses_3j.total_series_resistance:.4f} Ω⋅cm²")
    print(f"Tunnel junction efficiency: {losses_3j.tunnel_junction_efficiency:.3f}")
    print(f"Fill factor loss: {losses_3j.fill_factor_loss:.3f}")
    
    # Test 3: Perovskite tandem
    print("\nTest 3: Perovskite/Silicon Tandem")
    
    materials_perov = ['MAPbI3', 'c-Si']
    bandgaps_perov = [1.55, 1.12]
    currents_perov = [20, 20]  # Current matched
    
    losses_perov = interface_calc.calculate_total_interface_losses(
        materials_perov, bandgaps_perov, currents_perov, track='B'
    )
    
    print(f"Perovskite/Si voltage loss: {losses_perov.total_voltage_loss:.3f} V")
    print(f"Interface efficiency: {losses_perov.tunnel_junction_efficiency:.3f}")
    
    # Test 4: Tunnel junction optimization
    print("\nTest 4: Tunnel Junction Optimization")
    
    try:
        optimal_design = optimize_tunnel_junction_design(
            'GaInP', 'GaAs', target_current=15, track='A'
        )
        
        print(f"Optimal doping: {optimal_design['doping_concentration']:.1e} cm⁻³")
        print(f"Optimal thickness: {optimal_design['thickness_nm']:.1f} nm")
        print(f"Optimal barrier: {optimal_design['barrier_height']:.3f} eV")
        print(f"Optimized voltage loss: {optimal_design['voltage_loss']:.3f} V")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Validation
    print("\nValidation Against Literature:")
    validation = validate_interface_models()
    
    print(f"Calculated tunnel resistance: {validation['tunnel_resistance_calculated']:.4f} Ω⋅cm²")
    print(f"Literature reference: {validation['tunnel_resistance_literature']:.4f} Ω⋅cm²")
    
    multi_test = validation['multi_junction_test']
    print(f"Multi-junction total loss: {multi_test['total_voltage_loss']:.3f} V")
    print(f"Overall TJ efficiency: {multi_test['tunnel_junction_efficiency']:.3f}")
    
    print("\n✅ Interface loss engine implementation complete!")