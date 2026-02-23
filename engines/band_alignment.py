#!/usr/bin/env python3
"""
Band Alignment and Optimal Bandgap Distribution Engine
=====================================================

Calculates optimal bandgap distributions for N-junction tandem solar cells
using detailed balance theory and the Shockley-Queisser limit.

This module determines:
- Theoretical maximum efficiency vs number of junctions
- Optimal bandgap spacing for current matching
- Voltage losses from non-optimal bandgap distribution
- Trade-offs between efficiency and material availability

References:
- Shockley & Queisser, "Detailed Balance Limit of Efficiency of p‐n Junction Solar Cells" (1961)
- Henry, "Limiting efficiencies of ideal single and multiple energy gap terrestrial solar cells" (1980)
- De Vos, "Detailed balance limit of the efficiency of tandem solar cells" (1980)
- Marti & Araujo, "Limiting efficiencies for photovoltaic energy conversion in multigap systems" (1996)
- Green et al., "Solar cell efficiency tables (version 58)" (2021)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.integrate import quad

# Local imports
try:
    from ..config import (MATERIAL_DB, Q, KB, H, C, T_CELL, T_SUN, 
                         CONCENTRATION, get_am15g_spectrum, AM15G_FLUX_TOTAL)
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (MATERIAL_DB, Q, KB, H, C, T_CELL, T_SUN, 
                       CONCENTRATION, get_am15g_spectrum, AM15G_FLUX_TOTAL)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BandgapSolution:
    """Optimal bandgap distribution for N-junction cell"""
    n_junctions: int
    bandgaps: List[float]           # eV, ordered from top to bottom
    max_efficiency: float           # Theoretical maximum efficiency
    jsc_matched: float              # Current density at current matching (mA/cm²)
    voc_total: float                # Total open-circuit voltage (V)
    voltage_losses: List[float]     # Voltage loss per junction (V)
    current_densities: List[float]  # Short-circuit current per junction (mA/cm²)
    fill_factor: float              # Assumed ideal fill factor
    radiative_limit: bool           # Whether radiative recombination limit applied
    optimization_method: str        # Algorithm used for optimization

@dataclass
class DetailedBalanceResult:
    """Complete detailed balance analysis results"""
    photon_flux_incident: float     # Incident photon flux (m⁻²⋅s⁻¹)
    photon_flux_absorbed: float     # Absorbed photon flux
    radiative_recombination: float  # Radiative recombination rate
    jsc_theoretical: float          # Short-circuit current (mA/cm²)
    voc_theoretical: float          # Open-circuit voltage (V)
    max_power_density: float        # Maximum power density (mW/cm²)
    efficiency_theoretical: float   # Theoretical efficiency
    entropy_generation: float       # Entropy generation rate
    
# =============================================================================
# SHOCKLEY-QUEISSER DETAILED BALANCE CALCULATIONS
# =============================================================================

class DetailedBalanceCalculator:
    """
    Detailed balance calculations for solar cells based on Shockley-Queisser theory.
    
    Assumes:
    - Only radiative recombination (perfect cell)
    - Step-function absorption (EQE = 1 for E > Eg, 0 otherwise)  
    - Lambertian emission
    - One sun illumination (can be scaled)
    """
    
    def __init__(self, temperature: float = T_CELL, concentration: float = CONCENTRATION):
        """
        Initialize detailed balance calculator.
        
        Args:
            temperature: Cell temperature in Kelvin
            concentration: Sun concentration (1 = 1-sun)
        """
        
        self.T = temperature
        self.concentration = concentration
        
        # Wavelength range (ASCENDING order for correct integration)
        self.n_points = 1000
        self.wavelengths = np.linspace(300, 4133, self.n_points)  # nm, ascending
        self.energies = 1240 / self.wavelengths  # eV, descending
        
        # Get AM1.5G spectrum (positive for ascending wavelengths)
        self.am15g_flux = get_am15g_spectrum(self.wavelengths)  # W⋅m⁻²⋅nm⁻¹
        
        # Convert to photon flux density: Φ(λ) = I(λ) * λ / (hc)
        self.photon_flux_incident = self.am15g_flux * self.wavelengths * 1e-9 / (H * C)  # photons⋅m⁻²⋅s⁻¹⋅nm⁻¹
        
        # Pre-calculate blackbody spectrum at cell temperature
        self._calculate_blackbody_emission()
    
    def _calculate_blackbody_emission(self):
        """Pre-calculate blackbody photon flux at cell temperature.
        
        Planck's law in wavelength form:
        Φ_bb(λ) = 2πc/λ⁴ * 1/[exp(hc/λkT) - 1]  [photons⋅m⁻²⋅s⁻¹⋅nm⁻¹]
        
        Ref: Rühle (2016) Solar Energy 130, 139-147
        """
        wl_m = self.wavelengths * 1e-9  # Convert nm to meters
        exponent = H * C / (wl_m * KB * self.T)
        # Clamp exponent to avoid overflow
        exponent = np.clip(exponent, 0, 500)
        
        # Photon flux per wavelength per steradian
        phi_bb = 2 * C / wl_m**4 / (np.exp(exponent) - 1)  # photons⋅m⁻²⋅s⁻¹⋅m⁻¹⋅sr⁻¹
        
        # Convert to per nm, integrate over hemisphere (π sr)
        self.blackbody_photon_flux = phi_bb * 1e-9 * np.pi  # photons⋅m⁻²⋅s⁻¹⋅nm⁻¹
    
    def calculate_single_junction(self, bandgap: float, detailed_output: bool = False) -> Union[float, DetailedBalanceResult]:
        """
        Calculate theoretical efficiency for a single junction cell.
        
        Args:
            bandgap: Bandgap energy in eV
            detailed_output: If True, return full DetailedBalanceResult
            
        Returns:
            Efficiency (fraction) or detailed results
        """
        
        # Find wavelength cutoff: λ_cutoff = 1240/Eg
        lambda_cutoff = 1240 / bandgap  # nm
        cutoff_idx = np.searchsorted(self.wavelengths, lambda_cutoff)
        
        if cutoff_idx <= 0:
            # Bandgap too high - no absorption
            if detailed_output:
                return DetailedBalanceResult(
                    photon_flux_incident=0, photon_flux_absorbed=0,
                    radiative_recombination=0, jsc_theoretical=0,
                    voc_theoretical=0, max_power_density=0,
                    efficiency_theoretical=0, entropy_generation=0
                )
            else:
                return 0.0
        
        # Absorbed photon flux (λ < λ_cutoff, i.e., E > Eg)
        absorbed_flux = np.trapezoid(
            self.photon_flux_incident[:cutoff_idx], 
            self.wavelengths[:cutoff_idx]
        ) * self.concentration  # photons⋅m⁻²⋅s⁻¹
        
        # Short-circuit current density
        jsc = Q * absorbed_flux * 1e-1  # Convert to mA/cm²
        
        # Radiative recombination (blackbody at cell temperature, E > Eg)
        blackbody_flux_above_eg = np.trapezoid(
            self.blackbody_photon_flux[:cutoff_idx],
            self.wavelengths[:cutoff_idx]
        )  # photons⋅m⁻²⋅s⁻¹
        
        # Open-circuit voltage from detailed balance
        if blackbody_flux_above_eg > 0:
            voc = (KB * self.T / Q) * np.log(absorbed_flux / blackbody_flux_above_eg)
        else:
            voc = 0.0
        
        # Ideal fill factor (from detailed balance - typically ~0.89 for Si)
        # FF_ideal = (voc - ln(voc + 0.72))/(voc + 1) where voc is normalized by kT/q
        voc_norm = Q * voc / (KB * self.T)
        if voc_norm > 0:
            ff_ideal = (voc_norm - np.log(voc_norm + 0.72)) / (voc_norm + 1)
            ff_ideal = max(ff_ideal, 0.25)  # Minimum reasonable FF
        else:
            ff_ideal = 0.0
        
        # Maximum power density
        pmax = jsc * voc * ff_ideal  # mW/cm²
        
        # Efficiency
        efficiency = pmax / (AM15G_FLUX_TOTAL * 0.1 * self.concentration)  # Fraction
        
        if detailed_output:
            return DetailedBalanceResult(
                photon_flux_incident=absorbed_flux / self.concentration,
                photon_flux_absorbed=absorbed_flux,
                radiative_recombination=blackbody_flux_above_eg,
                jsc_theoretical=jsc,
                voc_theoretical=voc,
                max_power_density=pmax,
                efficiency_theoretical=efficiency,
                entropy_generation=self._calculate_entropy_generation(absorbed_flux, blackbody_flux_above_eg)
            )
        else:
            return efficiency
    
    def _calculate_entropy_generation(self, absorbed_flux: float, emission_flux: float) -> float:
        """Calculate entropy generation rate in the solar cell"""
        
        # Simplified entropy calculation
        # ΔS = (energy absorbed)/T_sun - (energy emitted)/T_cell
        
        # Average energy of absorbed photons (weighted by spectrum)
        cutoff_idx = 100  # Approximate for calculation
        avg_energy_absorbed = np.average(
            self.energies[cutoff_idx:], 
            weights=self.photon_flux_incident[cutoff_idx:]
        )
        
        # Average energy of emitted photons (weighted by blackbody)
        avg_energy_emitted = np.average(
            self.energies[cutoff_idx:], 
            weights=self.blackbody_photon_flux[cutoff_idx:]
        )
        
        entropy_gen = (
            absorbed_flux * avg_energy_absorbed * Q / T_SUN -
            emission_flux * avg_energy_emitted * Q / self.T
        )  # W⋅m⁻²⋅K⁻¹
        
        return entropy_gen

# =============================================================================
# MULTI-JUNCTION OPTIMIZATION
# =============================================================================

class BandgapOptimizer:
    """
    Optimizer for multi-junction bandgap distributions.
    
    Finds optimal bandgap combinations that maximize efficiency under
    current matching constraints.
    """
    
    def __init__(self, detailed_balance_calc: DetailedBalanceCalculator = None):
        """Initialize optimizer with detailed balance calculator"""
        
        if detailed_balance_calc is None:
            self.db_calc = DetailedBalanceCalculator()
        else:
            self.db_calc = detailed_balance_calc
    
    def optimize_n_junction(self, n_junctions: int, 
                          bandgap_constraints: Optional[Tuple[float, float]] = None,
                          current_matching: bool = True,
                          method: str = 'differential_evolution') -> BandgapSolution:
        """
        Find optimal bandgap distribution for N-junction tandem cell.
        
        Args:
            n_junctions: Number of junctions (subcells)
            bandgap_constraints: (min_gap, max_gap) in eV
            current_matching: If True, enforce current matching constraint
            method: Optimization method ('differential_evolution', 'basin_hopping', 'grid_search')
            
        Returns:
            BandgapSolution with optimal distribution
        """
        
        if n_junctions < 1:
            raise ValueError("Number of junctions must be at least 1")
        if n_junctions > 20:
            warnings.warn("Large number of junctions may be computationally expensive")
        
        # Default bandgap constraints
        if bandgap_constraints is None:
            bandgap_constraints = (0.5, 3.5)  # Reasonable range for solar cells
        
        min_gap, max_gap = bandgap_constraints
        
        if method == 'differential_evolution':
            result = self._optimize_differential_evolution(
                n_junctions, min_gap, max_gap, current_matching
            )
        elif method == 'grid_search':
            result = self._optimize_grid_search(
                n_junctions, min_gap, max_gap, current_matching
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return result
    
    def _optimize_differential_evolution(self, n_junctions: int, min_gap: float, 
                                       max_gap: float, current_matching: bool) -> BandgapSolution:
        """Optimize using differential evolution algorithm"""
        
        # Define optimization bounds — spread across useful range
        # Practical range: 0.5 eV (NIR) to 2.5 eV (UV-vis boundary)
        effective_min = max(min_gap, 0.5)
        effective_max = min(max_gap, 2.5)
        bounds = [(effective_min, effective_max) for _ in range(n_junctions)]
        
        def objective(bandgaps_array):
            """Objective function: negative efficiency (to minimize)"""
            
            # Sort bandgaps in descending order (top to bottom cell)
            bandgaps = np.sort(bandgaps_array)[::-1]
            
            # Ensure minimum separation to avoid numerical issues
            min_separation = 0.05  # eV
            for i in range(len(bandgaps) - 1):
                if bandgaps[i] - bandgaps[i+1] < min_separation:
                    return 1e6  # Large penalty
            
            try:
                efficiency = self._calculate_tandem_efficiency(
                    bandgaps, current_matching
                )
                return -efficiency  # Minimize negative efficiency
            except:
                return 1e6  # Penalty for failed calculation
        
        # Run optimization
        result = differential_evolution(
            objective, bounds, 
            maxiter=2000, 
            popsize=30,
            tol=1e-8,
            atol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            seed=42
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge properly")
        
        # Extract optimal solution
        optimal_bandgaps = np.sort(result.x)[::-1]  # High to low
        optimal_efficiency = -result.fun
        
        # Calculate detailed results
        detailed_results = self._calculate_detailed_tandem_results(
            optimal_bandgaps, current_matching
        )
        
        return BandgapSolution(
            n_junctions=n_junctions,
            bandgaps=optimal_bandgaps.tolist(),
            max_efficiency=optimal_efficiency,
            jsc_matched=detailed_results['jsc_matched'],
            voc_total=detailed_results['voc_total'],
            voltage_losses=detailed_results['voltage_losses'],
            current_densities=detailed_results['current_densities'],
            fill_factor=detailed_results['fill_factor'],
            radiative_limit=True,
            optimization_method='differential_evolution'
        )
    
    def _optimize_grid_search(self, n_junctions: int, min_gap: float, 
                            max_gap: float, current_matching: bool) -> BandgapSolution:
        """Optimize using grid search (for small N)"""
        
        if n_junctions > 3:
            raise ValueError("Grid search only practical for N ≤ 3 junctions")
        
        # Create grid
        n_points = 50 if n_junctions == 1 else (20 if n_junctions == 2 else 10)
        grid_points = np.linspace(min_gap, max_gap, n_points)
        
        best_efficiency = 0
        best_bandgaps = None
        
        if n_junctions == 1:
            for eg in grid_points:
                efficiency = self._calculate_tandem_efficiency([eg], current_matching)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_bandgaps = [eg]
        
        elif n_junctions == 2:
            for eg1 in grid_points:
                for eg2 in grid_points:
                    if eg1 <= eg2:  # Skip invalid combinations
                        continue
                    bandgaps = [eg1, eg2]
                    efficiency = self._calculate_tandem_efficiency(bandgaps, current_matching)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_bandgaps = bandgaps
        
        elif n_junctions == 3:
            for eg1 in grid_points:
                for eg2 in grid_points:
                    for eg3 in grid_points:
                        if eg1 <= eg2 or eg2 <= eg3:
                            continue
                        bandgaps = [eg1, eg2, eg3]
                        efficiency = self._calculate_tandem_efficiency(bandgaps, current_matching)
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_bandgaps = bandgaps
        
        if best_bandgaps is None:
            raise RuntimeError("Grid search failed to find valid solution")
        
        # Calculate detailed results
        detailed_results = self._calculate_detailed_tandem_results(
            best_bandgaps, current_matching
        )
        
        return BandgapSolution(
            n_junctions=n_junctions,
            bandgaps=best_bandgaps,
            max_efficiency=best_efficiency,
            jsc_matched=detailed_results['jsc_matched'],
            voc_total=detailed_results['voc_total'],
            voltage_losses=detailed_results['voltage_losses'],
            current_densities=detailed_results['current_densities'],
            fill_factor=detailed_results['fill_factor'],
            radiative_limit=True,
            optimization_method='grid_search'
        )
    
    def _calculate_tandem_efficiency(self, bandgaps: List[float], current_matching: bool) -> float:
        """Calculate tandem efficiency for given bandgap distribution"""
        
        if len(bandgaps) == 1:
            return self.db_calc.calculate_single_junction(bandgaps[0])
        
        # Calculate current densities for each subcell
        jsc_values = []
        voc_values = []
        
        # Top cell: only absorbs E > Eg_top
        top_result = self.db_calc.calculate_single_junction(bandgaps[0], detailed_output=True)
        jsc_values.append(top_result.jsc_theoretical)
        voc_values.append(top_result.voc_theoretical)
        
        # Middle cells: absorb photons between adjacent bandgaps
        for i in range(1, len(bandgaps) - 1):
            eg_high = bandgaps[i-1]  # Higher energy cutoff
            eg_low = bandgaps[i]     # Lower energy cutoff
            
            jsc_filtered = self._calculate_filtered_jsc(eg_high, eg_low)
            voc_filtered = self._calculate_filtered_voc(eg_low, jsc_filtered)
            
            jsc_values.append(jsc_filtered)
            voc_values.append(voc_filtered)
        
        # Bottom cell: absorbs all remaining photons (E > Eg_bottom)
        if len(bandgaps) > 1:
            eg_high = bandgaps[-2]
            eg_low = bandgaps[-1] 
            
            jsc_filtered = self._calculate_filtered_jsc(eg_high, eg_low, include_below=True)
            voc_filtered = self._calculate_filtered_voc(eg_low, jsc_filtered)
            
            jsc_values.append(jsc_filtered)
            voc_values.append(voc_filtered)
        
        # Current matching constraint
        if current_matching:
            jsc_tandem = min(jsc_values)  # Limited by lowest current
        else:
            jsc_tandem = np.mean(jsc_values)  # Average (no current matching)
        
        # Series voltage addition
        voc_tandem = sum(voc_values)
        
        # Fill factor (simplified - assumes similar for all subcells)
        voc_norm_avg = Q * voc_tandem / (len(bandgaps) * KB * self.db_calc.T)
        if voc_norm_avg > 0:
            ff_tandem = (voc_norm_avg - np.log(voc_norm_avg + 0.72)) / (voc_norm_avg + 1)
            ff_tandem = max(ff_tandem, 0.25)
        else:
            ff_tandem = 0.0
        
        # Power and efficiency
        power_density = jsc_tandem * voc_tandem * ff_tandem  # mW/cm²
        efficiency = power_density / (AM15G_FLUX_TOTAL * 0.1 * self.db_calc.concentration)
        
        return efficiency
    
    def _calculate_filtered_jsc(self, eg_high: float, eg_low: float, include_below: bool = False) -> float:
        """Calculate Jsc for photons in a specific energy range.
        
        Wavelengths are ascending, energies descending.
        E > Eg_high → λ < λ_high; E > Eg_low → λ < λ_low
        """
        lam_high = 1240 / eg_high  # shorter wavelength (higher energy cutoff)
        lam_low = 1240 / eg_low    # longer wavelength (lower energy cutoff)
        
        idx_high = np.searchsorted(self.db_calc.wavelengths, lam_high)
        idx_low = np.searchsorted(self.db_calc.wavelengths, lam_low)
        
        if include_below:
            # Bottom cell: absorb all photons with λ < lam_low (E > eg_low)
            # But exclude photons already absorbed by upper cells (λ < lam_high)
            absorbed_flux = np.trapezoid(
                self.db_calc.photon_flux_incident[idx_high:idx_low],
                self.db_calc.wavelengths[idx_high:idx_low]
            )
        else:
            # Middle cell: absorb photons with lam_high < λ < lam_low (eg_low < E < eg_high)
            absorbed_flux = np.trapezoid(
                self.db_calc.photon_flux_incident[idx_high:idx_low],
                self.db_calc.wavelengths[idx_high:idx_low]
            )
        
        absorbed_flux *= self.db_calc.concentration
        jsc = Q * absorbed_flux * 1e-1  # mA/cm²
        
        return jsc
    
    def _calculate_filtered_voc(self, bandgap: float, jsc: float) -> float:
        """Calculate Voc for filtered current density"""
        
        # Use single junction calculation but with corrected current
        single_result = self.db_calc.calculate_single_junction(bandgap, detailed_output=True)
        
        if single_result.jsc_theoretical > 0:
            # Scale by current ratio (approximate)
            voc_ratio = jsc / single_result.jsc_theoretical
            
            # Voc scales logarithmically with current
            voc_filtered = single_result.voc_theoretical + (KB * self.db_calc.T / Q) * np.log(voc_ratio)
            return max(voc_filtered, 0)
        else:
            return 0.0
    
    def _calculate_detailed_tandem_results(self, bandgaps: List[float], current_matching: bool) -> Dict:
        """Calculate detailed results for tandem cell analysis"""
        
        # Calculate individual subcell properties
        jsc_values = []
        voc_values = []
        
        for i, bandgap in enumerate(bandgaps):
            if i == 0:
                # Top cell
                result = self.db_calc.calculate_single_junction(bandgap, detailed_output=True)
                jsc_values.append(result.jsc_theoretical)
                voc_values.append(result.voc_theoretical)
            elif i == len(bandgaps) - 1:
                # Bottom cell
                jsc = self._calculate_filtered_jsc(bandgaps[i-1], bandgap, include_below=True)
                voc = self._calculate_filtered_voc(bandgap, jsc)
                jsc_values.append(jsc)
                voc_values.append(voc)
            else:
                # Middle cell
                jsc = self._calculate_filtered_jsc(bandgaps[i-1], bandgap)
                voc = self._calculate_filtered_voc(bandgap, jsc)
                jsc_values.append(jsc)
                voc_values.append(voc)
        
        # Current matching and voltage losses
        if current_matching:
            jsc_matched = min(jsc_values)
            voltage_losses = [(jsc_values[i] - jsc_matched) / jsc_values[i] * voc_values[i] 
                            for i in range(len(voc_values))]
        else:
            jsc_matched = np.mean(jsc_values)
            voltage_losses = [0.0] * len(voc_values)
        
        voc_total = sum(voc_values)
        
        # Fill factor calculation
        voc_norm_total = Q * voc_total / (KB * self.db_calc.T)
        if voc_norm_total > 0:
            ff = (voc_norm_total - np.log(voc_norm_total + 0.72)) / (voc_norm_total + 1)
            ff = max(ff, 0.25)
        else:
            ff = 0.0
        
        return {
            'jsc_matched': jsc_matched,
            'voc_total': voc_total,
            'voltage_losses': voltage_losses,
            'current_densities': jsc_values,
            'fill_factor': ff
        }

# =============================================================================
# MATERIAL-CONSTRAINED OPTIMIZATION
# =============================================================================

def find_available_bandgaps(track: str = 'A', 
                          include_mixed: bool = True,
                          mixed_resolution: int = 20) -> List[Tuple[float, str]]:
    """
    Find all available bandgaps from material database.
    
    Args:
        track: Material track ('A' or 'B')
        include_mixed: Include mixed halide compositions (Track B only)
        mixed_resolution: Number of composition points for mixed materials
        
    Returns:
        List of (bandgap, material_name) tuples
    """
    
    available_bandgaps = []
    
    # Pure materials
    materials = MATERIAL_DB.list_materials(track)
    for material_name in materials:
        material = MATERIAL_DB.get_material(material_name, track)
        bandgap = material['bandgap']
        available_bandgaps.append((bandgap, material_name))
    
    # Mixed compositions (Track B only)
    if track == 'B' and include_mixed:
        # Binary mixtures
        halides = ['I', 'Br', 'Cl']
        
        for i, halide1 in enumerate(halides):
            for halide2 in halides[i+1:]:
                for x in np.linspace(0.1, 0.9, mixed_resolution):
                    composition = {halide1: x, halide2: 1-x}
                    try:
                        mixed_props = MATERIAL_DB.get_mixed_halide_properties(composition)
                        bandgap = mixed_props['bandgap']
                        name = f"Mixed_{halide1}{x:.1f}{halide2}{1-x:.1f}"
                        available_bandgaps.append((bandgap, name))
                    except:
                        continue
        
        # Ternary mixtures (selected points)
        for x_I in [0.2, 0.4, 0.6]:
            for x_Br in np.linspace(0.1, 0.8-x_I, 5):
                x_Cl = 1 - x_I - x_Br
                if x_Cl > 0.05:  # Minimum 5% of each component
                    composition = {'I': x_I, 'Br': x_Br, 'Cl': x_Cl}
                    try:
                        mixed_props = MATERIAL_DB.get_mixed_halide_properties(composition)
                        bandgap = mixed_props['bandgap']
                        name = f"Mixed_I{x_I:.1f}Br{x_Br:.1f}Cl{x_Cl:.1f}"
                        available_bandgaps.append((bandgap, name))
                    except:
                        continue
    
    # Sort by bandgap
    available_bandgaps.sort(key=lambda x: x[0])
    
    return available_bandgaps

def optimize_with_material_constraints(n_junctions: int,
                                     track: str = 'A',
                                     include_mixed: bool = True,
                                     bandgap_tolerance: float = 0.05) -> BandgapSolution:
    """
    Optimize bandgap distribution constrained by available materials.
    
    Args:
        n_junctions: Number of junctions
        track: Material track
        include_mixed: Include mixed compositions
        bandgap_tolerance: Tolerance for matching theoretical optimal gaps (eV)
        
    Returns:
        Material-constrained optimal solution
    """
    
    # First find theoretical optimum
    optimizer = BandgapOptimizer()
    theoretical_optimum = optimizer.optimize_n_junction(n_junctions)
    
    # Get available materials
    available_materials = find_available_bandgaps(track, include_mixed)
    available_gaps = [gap for gap, name in available_materials]
    
    # Find closest available materials to theoretical optimum
    selected_materials = []
    selected_gaps = []
    
    for target_gap in theoretical_optimum.bandgaps:
        # Find closest available bandgap
        closest_idx = np.argmin([abs(gap - target_gap) for gap in available_gaps])
        closest_gap, closest_name = available_materials[closest_idx]
        
        # Check if within tolerance
        if abs(closest_gap - target_gap) <= bandgap_tolerance:
            selected_gaps.append(closest_gap)
            selected_materials.append(closest_name)
        else:
            # Use theoretical gap with warning
            selected_gaps.append(target_gap)
            selected_materials.append(f"Theoretical_{target_gap:.2f}eV")
            warnings.warn(f"No material found within tolerance for Eg = {target_gap:.2f} eV")
    
    # Calculate efficiency with material-constrained bandgaps
    constrained_efficiency = optimizer._calculate_tandem_efficiency(selected_gaps, True)
    detailed_results = optimizer._calculate_detailed_tandem_results(selected_gaps, True)
    
    return BandgapSolution(
        n_junctions=n_junctions,
        bandgaps=selected_gaps,
        max_efficiency=constrained_efficiency,
        jsc_matched=detailed_results['jsc_matched'],
        voc_total=detailed_results['voc_total'],
        voltage_losses=detailed_results['voltage_losses'],
        current_densities=detailed_results['current_densities'],
        fill_factor=detailed_results['fill_factor'],
        radiative_limit=True,
        optimization_method=f'material_constrained_track_{track}'
    )

# =============================================================================
# VALIDATION AND BENCHMARKING
# =============================================================================

def validate_shockley_queisser_limits() -> Dict[str, float]:
    """
    Validate implementation against known Shockley-Queisser limits.
    
    Returns:
        Dictionary with theoretical limits for comparison
    """
    
    db_calc = DetailedBalanceCalculator()
    optimizer = BandgapOptimizer(db_calc)
    
    results = {}
    
    # Single junction optimum (should be ~33.7% at ~1.34 eV)
    single_result = optimizer.optimize_n_junction(1, method='grid_search')
    results['1J_efficiency'] = single_result.max_efficiency
    results['1J_optimal_bandgap'] = single_result.bandgaps[0]
    
    # Two junction optimum (should be ~42.9%)
    double_result = optimizer.optimize_n_junction(2, method='differential_evolution')
    results['2J_efficiency'] = double_result.max_efficiency
    results['2J_bandgaps'] = double_result.bandgaps
    
    # Three junction optimum (should be ~49.3%)
    triple_result = optimizer.optimize_n_junction(3, method='differential_evolution')
    results['3J_efficiency'] = triple_result.max_efficiency
    results['3J_bandgaps'] = triple_result.bandgaps
    
    # Literature values for comparison
    literature_values = {
        '1J_literature': 0.337,
        '2J_literature': 0.429,
        '3J_literature': 0.493,
        'infinite_J_literature': 0.687
    }
    
    results.update(literature_values)
    
    # Calculate deviations
    for n_j in ['1J', '2J', '3J']:
        calc_eff = results[f'{n_j}_efficiency']
        lit_eff = results[f'{n_j}_literature']
        results[f'{n_j}_deviation'] = abs(calc_eff - lit_eff) / lit_eff
    
    return results

if __name__ == "__main__":
    # Test band alignment calculations
    print("Band Alignment & Shockley-Queisser Validation")
    print("=" * 55)
    
    # Test single junction
    print("\nSingle Junction Test:")
    db_calc = DetailedBalanceCalculator()
    
    # Test Si bandgap
    si_efficiency = db_calc.calculate_single_junction(1.12)
    print(f"Silicon (Eg=1.12eV): {si_efficiency:.3f} = {si_efficiency*100:.1f}%")
    
    # Find optimal single junction
    optimizer = BandgapOptimizer(db_calc)
    optimal_1j = optimizer.optimize_n_junction(1, method='grid_search')
    print(f"Optimal 1J: Eg={optimal_1j.bandgaps[0]:.2f}eV, η={optimal_1j.max_efficiency:.3f} = {optimal_1j.max_efficiency*100:.1f}%")
    
    # Test two junction
    print("\nTwo Junction Test:")
    optimal_2j = optimizer.optimize_n_junction(2, method='differential_evolution')
    print(f"Optimal 2J: Eg1={optimal_2j.bandgaps[0]:.2f}eV, Eg2={optimal_2j.bandgaps[1]:.2f}eV")
    print(f"Efficiency: {optimal_2j.max_efficiency:.3f} = {optimal_2j.max_efficiency*100:.1f}%")
    print(f"Jsc matched: {optimal_2j.jsc_matched:.1f} mA/cm²")
    print(f"Voc total: {optimal_2j.voc_total:.3f} V")
    
    # Validation against literature
    print("\nValidation Against Literature:")
    validation = validate_shockley_queisser_limits()
    
    print(f"1J: Calculated={validation['1J_efficiency']*100:.1f}%, Literature={validation['1J_literature']*100:.1f}%, Error={validation['1J_deviation']*100:.1f}%")
    print(f"2J: Calculated={validation['2J_efficiency']*100:.1f}%, Literature={validation['2J_literature']*100:.1f}%, Error={validation['2J_deviation']*100:.1f}%")
    print(f"3J: Calculated={validation['3J_efficiency']*100:.1f}%, Literature={validation['3J_literature']*100:.1f}%, Error={validation['3J_deviation']*100:.1f}%")
    
    # Test material constraints
    print("\nMaterial-Constrained Optimization (Track B):")
    available_b = find_available_bandgaps('B', include_mixed=True, mixed_resolution=10)
    print(f"Found {len(available_b)} available bandgap options")
    
    bandgap_range = (min(gap for gap, _ in available_b), max(gap for gap, _ in available_b))
    print(f"Bandgap range: {bandgap_range[0]:.2f} - {bandgap_range[1]:.2f} eV")
    
    constrained_2j = optimize_with_material_constraints(2, track='B', include_mixed=True)
    print(f"Material-constrained 2J: {constrained_2j.max_efficiency*100:.1f}% vs {optimal_2j.max_efficiency*100:.1f}% theoretical")
    
    print("\n✅ Band alignment engine implementation complete!")