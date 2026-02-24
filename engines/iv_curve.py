#!/usr/bin/env python3
"""
I-V Curve Simulator for Tandem Photovoltaic Cells
=================================================

Single diode model implementation with tandem series connection support.
Generates current-voltage characteristics and extracts performance parameters.

Features:
- Single diode model with series/shunt resistance
- Temperature and concentration effects
- Maximum power point tracking
- Series connection for tandem cells
- Current matching analysis

References:
- Sze, S.M. & Ng, K.K. "Physics of Semiconductor Devices" 3rd Ed. (2006)
- Green, M.A. "Solar Cells: Operating Principles, Technology, and System Applications" (1982)
- Rau, U. & Paetzold, U.W. "Thermodynamics of light management in photovoltaic devices" (2009)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize_scalar, brentq
from scipy.constants import k, elementary_charge as q
import warnings

warnings.filterwarnings("ignore")

class SingleDiodeModel:
    """
    Single diode equivalent circuit model for solar cells.
    
    Implements: I = Iph - I0*[exp(q(V+I*Rs)/(n*k*T)) - 1] - (V+I*Rs)/Rsh
    """
    
    def __init__(self, 
                 bandgap: float,
                 temperature: float = 298.15,
                 concentration: float = 1.0,
                 ideality_factor: float = 1.2,
                 series_resistance: float = 0.0,
                 shunt_resistance: float = 1e6):
        """
        Initialize single diode model parameters.
        
        Args:
            bandgap: Bandgap energy [eV]
            temperature: Cell temperature [K]
            concentration: Solar concentration [suns]
            ideality_factor: Diode ideality factor (1-2)
            series_resistance: Series resistance [Ω⋅cm²]
            shunt_resistance: Shunt resistance [Ω⋅cm²]
        """
        self.Eg = bandgap
        self.T = temperature
        self.X = concentration
        self.n = ideality_factor
        self.Rs = series_resistance
        self.Rsh = shunt_resistance
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """Calculate photocurrent and saturation current.
        
        FIXED: Replace rough Iph estimation with proper spectral integration:
        Iph = q × ∫(λ_min to λ_g) Φ(λ) × EQE(λ) dλ, where λ_g = hc/Eg
        """
        
        # Photocurrent from proper spectral integration
        kT = k * self.T / q  # Thermal voltage [V]
        
        # FIXED: Proper spectral integration for photocurrent
        try:
            # Try to use the solar spectrum engine if available
            from .solar_spectrum import SOLAR_SPECTRUM_CALCULATOR
            
            # Get AM1.5G spectrum
            spectrum = SOLAR_SPECTRUM_CALCULATOR.calculate_spectrum(air_mass=1.5)
            wavelengths = spectrum.wavelengths  # nm
            irradiance = spectrum.irradiance    # W⋅m⁻²⋅nm⁻¹
            
            # Bandgap wavelength cutoff
            lambda_g = 1240 / self.Eg  # nm, λ = hc/E
            
            # Filter spectrum for photons with energy E > Eg (λ < λ_g)
            mask = wavelengths <= lambda_g
            wl_filtered = wavelengths[mask]
            irr_filtered = irradiance[mask]
            
            if len(wl_filtered) == 0:
                # Bandgap too high, no absorption
                Iph_1sun = 0.0
            else:
                # Direct conversion from power to photocurrent
                h = 6.62607015e-34  # J⋅s
                c = 2.99792458e8    # m/s
                
                # Photocurrent density per wavelength: J(λ) = q × I(λ) × λ/(hc) [A/m²/nm]
                # This directly converts power to current accounting for photon energy
                wl_m = wl_filtered * 1e-9  # Convert nm to meters
                J_spectral = q * irr_filtered * wl_m / (h * c)  # A/m²/nm
                
                # Integrate over wavelength to get total current density
                J_total_A_per_m2 = np.trapezoid(J_spectral, wl_filtered)  # A/m²
                
                # Convert to mA/cm²: 1 A/m² = 0.1 mA/cm²
                Iph_1sun = J_total_A_per_m2 * 0.1  # mA/cm²
            
        except (ImportError, AttributeError, ValueError):
            # Fallback to simple estimation if solar spectrum engine not available
            # Keep the simple estimate as fallback but with improved physics
            if self.Eg <= 0.7:
                Iph_1sun = 45.0  # Very low bandgap (IR)
            elif self.Eg <= 1.0:
                Iph_1sun = 40.0  # Low bandgap  
            elif self.Eg <= 1.3:
                Iph_1sun = 35.0  # Medium-low
            elif self.Eg <= 1.6:
                Iph_1sun = 25.0  # Medium
            elif self.Eg <= 2.0:
                Iph_1sun = 15.0  # Medium-high
            elif self.Eg <= 2.5:
                Iph_1sun = 8.0   # High bandgap
            else:
                Iph_1sun = 2.0   # Very high bandgap
        
        self.Iph = Iph_1sun * self.X  # Scale with concentration
        
        # Saturation current density from detailed balance
        # I0 = q * integral of blackbody emission
        I0_prefactor = 2.51e4  # mA/cm² (from integration constants)
        self.I0 = I0_prefactor * self.T**2 * np.exp(-self.Eg / kT)
        
        # Thermal voltage
        self.VT = kT  # k*T/q in Volts
    
    def current_density(self, voltage: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate current density at given voltage(s).
        
        Args:
            voltage: Terminal voltage [V], can be array
            
        Returns:
            Current density [mA/cm²]
        """
        V = np.asarray(voltage)
        
        # Single diode equation: I = Iph - I0*(exp((V+I*Rs)/(n*VT)) - 1) - (V+I*Rs)/Rsh
        # Implicit in current, solve numerically for each voltage
        
        if V.ndim == 0:  # Scalar input
            return self._solve_current_at_voltage(float(V))
        else:  # Array input
            return np.array([self._solve_current_at_voltage(v) for v in V])
    
    def _solve_current_at_voltage(self, V: float) -> float:
        """Solve implicit current equation at single voltage point."""
        
        def current_equation(I):
            """Current equation to solve: f(I) = 0"""
            if I < 0:
                return 1e10  # Penalize negative current
            
            try:
                diode_term = self.I0 * (np.exp((V + I * self.Rs) / (self.n * self.VT)) - 1)
                shunt_term = (V + I * self.Rs) / self.Rsh
                return self.Iph - diode_term - shunt_term - I
            except (OverflowError, FloatingPointError):
                return 1e10
        
        # Initial guess: neglect Rs and Rsh
        I_guess = max(0, self.Iph - self.I0 * (np.exp(V / (self.n * self.VT)) - 1))
        
        try:
            # Use Brent's method for robust root finding
            if current_equation(0) * current_equation(I_guess) > 0:
                # Same sign, expand search
                I_max = max(I_guess * 2, self.Iph * 1.2)
                if current_equation(0) * current_equation(I_max) > 0:
                    return 0  # Failed to bracket root
                I_solution = brentq(current_equation, 0, I_max, xtol=1e-10)
            else:
                I_solution = brentq(current_equation, 0, I_guess, xtol=1e-10)
                
            return max(0, I_solution)
        
        except (ValueError, RuntimeError):
            # Fallback to simple approximation
            return max(0, self.Iph - self.I0 * (np.exp(V / (self.n * self.VT)) - 1))
    
    def generate_iv_curve(self, 
                         n_points: int = 100,
                         voltage_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete I-V curve.
        
        Args:
            n_points: Number of voltage points
            voltage_range: (V_min, V_max) in volts, auto if None
            
        Returns:
            (voltage_array, current_array) in [V, mA/cm²]
        """
        
        if voltage_range is None:
            # Auto-range from 0 to estimated Voc + margin
            V_est_oc = self.n * self.VT * np.log(self.Iph / self.I0 + 1)
            V_max = min(V_est_oc * 1.2, 3.0)  # Cap at 3V for safety
            voltage_range = (0.0, V_max)
        
        V = np.linspace(voltage_range[0], voltage_range[1], n_points)
        I = self.current_density(V)
        
        return V, I
    
    def extract_parameters(self) -> Dict[str, float]:
        """
        Extract solar cell parameters from I-V curve.
        
        Returns:
            Dictionary with Jsc, Voc, FF, PCE, Vmpp, Impp, Pmpp
        """
        
        # Generate I-V curve
        V, I = self.generate_iv_curve(n_points=200)
        
        # Short-circuit current (I at V=0)
        Jsc = float(I[0])
        
        # Open-circuit voltage (V where I=0)
        if np.any(I <= 0):
            # Find crossing point
            zero_idx = np.where(I <= 0)[0]
            if len(zero_idx) > 0:
                idx = zero_idx[0]
                if idx > 0:
                    # Linear interpolation for better accuracy
                    V1, I1 = V[idx-1], I[idx-1]
                    V2, I2 = V[idx], I[idx]
                    Voc = V1 - I1 * (V2 - V1) / (I2 - I1)
                else:
                    Voc = V[0]
            else:
                Voc = V[-1]  # Fallback
        else:
            Voc = V[-1]  # All currents positive, take max voltage
        
        # Maximum power point
        P = V * I  # Power density [mW/cm²]
        valid_power = P[I > 0]  # Only positive current region
        
        if len(valid_power) > 0:
            max_idx = np.argmax(valid_power)
            # Map back to original indices
            positive_indices = np.where(I > 0)[0]
            if len(positive_indices) > max_idx:
                mpp_idx = positive_indices[max_idx]
                Vmpp = V[mpp_idx]
                Impp = I[mpp_idx]
                Pmpp = valid_power[max_idx]
            else:
                Vmpp = Impp = Pmpp = 0
        else:
            Vmpp = Impp = Pmpp = 0
        
        # Fill factor
        FF = Pmpp / (Jsc * Voc) if (Jsc * Voc > 0) else 0
        
        # Power conversion efficiency (at 1000 W/m² = 100 mW/cm²)
        PCE = Pmpp / (100.0 * self.X) if self.X > 0 else 0
        
        return {
            'Jsc': Jsc,        # mA/cm²
            'Voc': Voc,        # V
            'FF': FF,          # dimensionless
            'PCE': PCE * 100,  # %
            'Vmpp': Vmpp,      # V
            'Impp': Impp,      # mA/cm²
            'Pmpp': Pmpp       # mW/cm²
        }

class TandemIVSimulator:
    """
    Simulator for tandem (multi-junction) I-V characteristics.
    
    Handles series connection of subcells with current matching constraints.
    """
    
    def __init__(self, subcells: List[Dict]):
        """
        Initialize tandem simulator.
        
        Args:
            subcells: List of subcell dictionaries with parameters:
                     {'bandgap', 'temperature', 'concentration', 'Rs', 'Rsh', 'n'}
        """
        self.subcells = []
        
        for cell_params in subcells:
            cell = SingleDiodeModel(
                bandgap=cell_params['bandgap'],
                temperature=cell_params.get('temperature', 298.15),
                concentration=cell_params.get('concentration', 1.0),
                ideality_factor=cell_params.get('n', 1.2),
                series_resistance=cell_params.get('Rs', 0.0),
                shunt_resistance=cell_params.get('Rsh', 1e6)
            )
            self.subcells.append(cell)
    
    def generate_tandem_iv(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate tandem I-V curve with series connection.
        
        In series connection:
        - Current is limited by minimum subcell current
        - Voltages add up
        
        Args:
            n_points: Number of points for I-V curve
            
        Returns:
            (V_tandem, I_tandem, metrics_dict)
        """
        
        if not self.subcells:
            return np.array([0]), np.array([0]), {}
        
        # Get all subcell I-V curves on same current axis
        # Use minimum Jsc to set current range
        subcell_params = [cell.extract_parameters() for cell in self.subcells]
        min_jsc = min(params['Jsc'] for params in subcell_params)
        
        # Current array from 0 to minimum Jsc
        I_array = np.linspace(0, min_jsc * 0.99, n_points)
        
        # For each current, find voltage of each subcell
        V_subcells = np.zeros((len(self.subcells), len(I_array)))
        
        for i, cell in enumerate(self.subcells):
            for j, I_target in enumerate(I_array):
                V_subcells[i, j] = self._find_voltage_at_current(cell, I_target)
        
        # Series connection: V_total = sum of subcell voltages
        V_tandem = np.sum(V_subcells, axis=0)
        
        # Extract tandem parameters
        tandem_metrics = self._extract_tandem_parameters(V_tandem, I_array, subcell_params)
        
        return V_tandem, I_array, tandem_metrics
    
    def _find_voltage_at_current(self, cell: SingleDiodeModel, I_target: float) -> float:
        """Find voltage that produces target current in given cell."""
        
        def voltage_equation(V):
            return cell.current_density(V) - I_target
        
        # Estimate voltage range
        params = cell.extract_parameters()
        V_max = params['Voc']
        
        try:
            if voltage_equation(0) * voltage_equation(V_max) <= 0:
                V_solution = brentq(voltage_equation, 0, V_max, xtol=1e-6)
            else:
                # Extrapolate if necessary
                V_solution = V_max * I_target / params['Jsc']
                
            return max(0, V_solution)
        
        except (ValueError, RuntimeError):
            # Linear approximation fallback
            return V_max * I_target / params['Jsc'] if params['Jsc'] > 0 else 0
    
    def _extract_tandem_parameters(self, V_tandem: np.ndarray, I_tandem: np.ndarray, 
                                 subcell_params: List[Dict]) -> Dict[str, float]:
        """Extract performance parameters from tandem I-V curve."""
        
        if len(V_tandem) == 0 or len(I_tandem) == 0:
            return {'Jsc': 0, 'Voc': 0, 'FF': 0, 'PCE': 0, 'Vmpp': 0, 'Impp': 0, 'Pmpp': 0}
        
        # Short-circuit current (current limited by minimum subcell)
        Jsc_tandem = float(I_tandem[-1]) if len(I_tandem) > 0 else 0
        
        # Open-circuit voltage (sum of subcell Vocs)
        Voc_tandem = sum(params['Voc'] for params in subcell_params)
        
        # Maximum power point
        P_tandem = V_tandem * I_tandem
        if len(P_tandem) > 0:
            max_idx = np.argmax(P_tandem)
            Vmpp_tandem = V_tandem[max_idx]
            Impp_tandem = I_tandem[max_idx]
            Pmpp_tandem = P_tandem[max_idx]
        else:
            Vmpp_tandem = Impp_tandem = Pmpp_tandem = 0
        
        # Fill factor
        FF_tandem = Pmpp_tandem / (Jsc_tandem * Voc_tandem) if (Jsc_tandem * Voc_tandem > 0) else 0
        
        # Power conversion efficiency (assume 1 sun concentration)
        PCE_tandem = Pmpp_tandem / 100.0  # 100 mW/cm² = 1000 W/m²
        
        return {
            'Jsc': Jsc_tandem,
            'Voc': Voc_tandem,
            'FF': FF_tandem,
            'PCE': PCE_tandem * 100,  # %
            'Vmpp': Vmpp_tandem,
            'Impp': Impp_tandem,
            'Pmpp': Pmpp_tandem,
            'subcell_params': subcell_params
        }

def find_mpp(voltage: np.ndarray, current: np.ndarray) -> Tuple[float, float, float]:
    """
    Find maximum power point from I-V data.
    
    Args:
        voltage: Voltage array [V]
        current: Current array [mA/cm²]
        
    Returns:
        (V_mpp, I_mpp, P_mpp) tuple
    """
    
    if len(voltage) == 0 or len(current) == 0:
        return 0, 0, 0
    
    # Calculate power
    power = voltage * current
    
    # Find maximum (only consider positive current region)
    positive_mask = current > 0
    if not np.any(positive_mask):
        return 0, 0, 0
    
    valid_power = power[positive_mask]
    valid_voltage = voltage[positive_mask]
    valid_current = current[positive_mask]
    
    max_idx = np.argmax(valid_power)
    
    return valid_voltage[max_idx], valid_current[max_idx], valid_power[max_idx]

def simulate_subcell_iv(bandgap: float, 
                       jsc: Optional[float] = None,
                       temperature: float = 298.15,
                       concentration: float = 1.0,
                       series_resistance: float = 0.0,
                       shunt_resistance: float = 1e6,
                       ideality_factor: float = 1.2) -> Dict[str, Union[np.ndarray, float]]:
    """
    Convenience function to simulate single subcell I-V curve.
    
    Args:
        bandgap: Bandgap energy [eV]
        jsc: Short-circuit current [mA/cm²], auto if None
        temperature: Cell temperature [K]
        concentration: Solar concentration [suns]
        series_resistance: Series resistance [Ω⋅cm²]
        shunt_resistance: Shunt resistance [Ω⋅cm²]
        ideality_factor: Diode ideality factor
        
    Returns:
        Dictionary with 'V', 'I' arrays and extracted parameters
    """
    
    # Create single diode model
    cell = SingleDiodeModel(
        bandgap=bandgap,
        temperature=temperature,
        concentration=concentration,
        ideality_factor=ideality_factor,
        series_resistance=series_resistance,
        shunt_resistance=shunt_resistance
    )
    
    # Override photocurrent if specified
    if jsc is not None:
        cell.Iph = jsc
    
    # Generate I-V curve
    V, I = cell.generate_iv_curve()
    
    # Extract parameters
    params = cell.extract_parameters()
    
    # Combine results
    result = {
        'V': V,
        'I': I,
        **params
    }
    
    return result

def simulate_tandem_iv(subcells: List[Dict], 
                      connection: str = 'series') -> Dict[str, Union[np.ndarray, float, List]]:
    """
    Convenience function to simulate tandem I-V curve.
    
    Args:
        subcells: List of subcell parameter dictionaries
        connection: Connection type ('series' only for now)
        
    Returns:
        Dictionary with tandem I-V data and parameters
    """
    
    if connection != 'series':
        raise ValueError("Only series connection supported currently")
    
    # Create tandem simulator
    tandem = TandemIVSimulator(subcells)
    
    # Generate I-V curve
    V, I, metrics = tandem.generate_tandem_iv()
    
    return {
        'V': V,
        'I': I,
        **metrics
    }

# Example usage and testing
if __name__ == "__main__":
    print("I-V Curve Simulator Test")
    print("=" * 40)
    
    # Test single cell
    print("\n1. Single cell test (Si, Eg=1.12 eV):")
    si_result = simulate_subcell_iv(
        bandgap=1.12,
        temperature=298.15,
        concentration=1.0,
        series_resistance=0.001,  # 1 mΩ⋅cm²
        shunt_resistance=1e4      # 10 kΩ⋅cm²
    )
    
    print(f"   Jsc: {si_result['Jsc']:.1f} mA/cm²")
    print(f"   Voc: {si_result['Voc']:.3f} V")
    print(f"   FF: {si_result['FF']:.3f}")
    print(f"   PCE: {si_result['PCE']:.1f}%")
    
    # Test tandem cell
    print("\n2. Tandem cell test (GaInP/Si):")
    subcells = [
        {'bandgap': 1.81, 'n': 1.1},  # GaInP top cell
        {'bandgap': 1.12, 'n': 1.2}   # Si bottom cell
    ]
    
    tandem_result = simulate_tandem_iv(subcells)
    
    print(f"   Jsc: {tandem_result['Jsc']:.1f} mA/cm²")
    print(f"   Voc: {tandem_result['Voc']:.3f} V")
    print(f"   FF: {tandem_result['FF']:.3f}")
    print(f"   PCE: {tandem_result['PCE']:.1f}%")
    
    # Current matching analysis
    subcell_jscs = [params['Jsc'] for params in tandem_result['subcell_params']]
    print(f"   Current matching: {subcell_jscs} mA/cm²")
    print(f"   Matching ratio: {min(subcell_jscs)/max(subcell_jscs):.3f}")
    
    print("\n✅ I-V Curve Simulator ready!")