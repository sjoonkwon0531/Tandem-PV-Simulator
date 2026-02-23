#!/usr/bin/env python3
"""
N-Junction Infinite Tandem PV Simulator - Configuration & Material Database
=====================================================================

Physical constants, material properties, and configuration settings.
Supports both Track A (Multi-Material) and Track B (All-Perovskite) approaches.

References:
- Shockley & Queisser (1961) - Detailed balance theory
- Green et al. (2021) - Solar cell efficiency tables  
- Vegard's law for mixed perovskites
- Phonopy materials database for CTE values
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Fundamental constants
Q = 1.602176634e-19     # Elementary charge [C]
KB = 1.380649e-23       # Boltzmann constant [J/K] 
H = 6.62607015e-34      # Planck constant [J⋅s]
C = 299792458           # Speed of light [m/s]
EV_TO_J = Q             # eV to Joules conversion
J_TO_EV = 1/Q           # Joules to eV conversion

# Solar cell parameters
T_CELL = 298.15         # Standard cell temperature [K] (25°C)
T_SUN = 5778            # Sun temperature [K]
CONCENTRATION = 1       # Sun concentration (1 = 1-sun)

# AM1.5G spectrum integration constants
AM15G_FLUX_TOTAL = 1000.37  # Total AM1.5G flux [W/m²]
AM15G_PHOTON_FLUX = 4.258e21  # Total photon flux [photons/(m²⋅s)]

# =============================================================================
# MATERIAL DATABASE - TRACK A: MULTI-MATERIAL
# =============================================================================

class MaterialDatabase:
    """
    Comprehensive material database for tandem PV simulation.
    Contains optical, thermal, electrical, and economic properties.
    """
    
    def __init__(self):
        self.wavelength_range = np.linspace(300, 1550, 126)  # 10nm steps
        self.materials_track_a = self._init_track_a_materials()
        self.materials_track_b = self._init_track_b_materials()
        
    def _init_track_a_materials(self) -> Dict:
        """Initialize Track A materials (multi-material systems)"""
        
        materials = {
            # Silicon variants
            'c-Si': {
                'name': 'Crystalline Silicon',
                'bandgap': 1.12,  # eV at 300K
                'type': 'indirect',
                'cte': 2.6e-6,    # /K (coefficient of thermal expansion)
                'humidity_score': 9.5,  # /10 (higher = more stable)
                'cost_per_cm2': 0.05,   # USD/cm² active area
                'deposition': 'wafer',
                'n_k_data': self._generate_si_nk(),
                'absorption_coefficient': 1e4,  # cm⁻¹ (typical)
                'diffusion_length': 300e-4,     # cm
                'mobility_electron': 1350,      # cm²/V⋅s
                'mobility_hole': 480,           # cm²/V⋅s
            },
            
            'a-Si': {
                'name': 'Amorphous Silicon',
                'bandgap': 1.75,
                'type': 'direct',
                'cte': 3.2e-6,
                'humidity_score': 7.0,
                'cost_per_cm2': 0.02,
                'deposition': 'pecvd',
                'n_k_data': self._generate_asi_nk(),
                'absorption_coefficient': 1e5,
                'diffusion_length': 10e-7,     # Very short in a-Si
                'dangling_bonds': 1e16,        # cm⁻³
            },
            
            # III-V semiconductors
            'GaAs': {
                'name': 'Gallium Arsenide',
                'bandgap': 1.42,
                'type': 'direct',
                'cte': 5.73e-6,
                'humidity_score': 8.5,
                'cost_per_cm2': 2.50,
                'deposition': 'mocvd',
                'n_k_data': self._generate_gaas_nk(),
                'absorption_coefficient': 1e5,
                'diffusion_length': 5e-4,
                'mobility_electron': 8500,
                'mobility_hole': 400,
            },
            
            'GaInP': {
                'name': 'Gallium Indium Phosphide',
                'bandgap': 1.81,
                'type': 'direct',
                'cte': 5.3e-6,
                'humidity_score': 8.0,
                'cost_per_cm2': 3.00,
                'deposition': 'mocvd',
                'n_k_data': self._generate_gainp_nk(),
                'absorption_coefficient': 8e4,
                'indium_fraction': 0.49,  # Ga₀.₅₁In₀.₄₉P lattice matched to GaAs
            },
            
            # Chalcogenide thin films
            'CIGS': {
                'name': 'Copper Indium Gallium Selenide',
                'bandgap': 1.15,  # Variable: 1.0-1.7 eV
                'bandgap_range': (1.0, 1.7),
                'type': 'direct',
                'cte': 8.8e-6,
                'humidity_score': 6.5,
                'cost_per_cm2': 0.35,
                'deposition': 'coevaporation',
                'n_k_data': self._generate_cigs_nk(),
                'absorption_coefficient': 1e5,
                'ga_fraction': 0.3,  # Ga/(In+Ga) ratio
            },
            
            'CdTe': {
                'name': 'Cadmium Telluride',
                'bandgap': 1.45,
                'type': 'direct',
                'cte': 4.9e-6,
                'humidity_score': 7.5,
                'cost_per_cm2': 0.25,
                'deposition': 'close_space_sublimation',
                'n_k_data': self._generate_cdte_nk(),
                'absorption_coefficient': 1e5,
                'toxicity_concern': True,
            },
            
            # Organic photovoltaics
            'OPV_P3HT': {
                'name': 'P3HT:PCBM Blend',
                'bandgap': 1.9,
                'bandgap_range': (1.2, 2.0),
                'type': 'direct',
                'cte': 150e-6,  # Much higher than inorganics
                'humidity_score': 2.0,  # Poor stability
                'cost_per_cm2': 0.10,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_opv_nk(),
                'absorption_coefficient': 1e5,
                'pcbm_ratio': 0.8,  # PCBM weight ratio
            },
            
            # Quantum dots
            'PbS_QD': {
                'name': 'Lead Sulfide Quantum Dots',
                'bandgap': 1.3,  # Size-tunable: 0.4-1.5 eV
                'bandgap_range': (0.4, 1.5),
                'type': 'direct',
                'cte': 18e-6,
                'humidity_score': 4.0,
                'cost_per_cm2': 0.80,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_pbsqd_nk(),
                'quantum_confinement': True,
                'dot_size': 3.5,  # nm (affects bandgap)
            },
            
            'PbSe_QD': {
                'name': 'Lead Selenide Quantum Dots',
                'bandgap': 0.9,  # NIR optimized
                'bandgap_range': (0.4, 1.3),
                'type': 'direct',
                'cte': 20e-6,
                'humidity_score': 4.5,
                'cost_per_cm2': 1.20,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_pbseqd_nk(),
                'quantum_confinement': True,
                'dot_size': 4.2,  # nm
            }
        }
        
        return materials
    
    def _init_track_b_materials(self) -> Dict:
        """Initialize Track B materials (all-perovskite systems)"""
        
        materials = {
            # Pure halide perovskites
            'MAPbI3': {
                'name': 'Methylammonium Lead Iodide',
                'bandgap': 1.55,
                'type': 'direct',
                'cte': 4.2e-5,  # High CTE for organics
                'humidity_score': 2.5,  # Poor humidity stability
                'cost_per_cm2': 0.15,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_mapbi3_nk(),
                'absorption_coefficient': 1e5,
                'urbach_energy': 15,  # meV (disorder parameter)
                'ion_migration': True,
                'phase_transition_temp': 327,  # K (tetragonal to cubic)
            },
            
            'MAPbBr3': {
                'name': 'Methylammonium Lead Bromide',
                'bandgap': 2.3,
                'type': 'direct',
                'cte': 3.8e-5,
                'humidity_score': 4.0,
                'cost_per_cm2': 0.20,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_mapbbr3_nk(),
                'absorption_coefficient': 8e4,
                'urbach_energy': 12,
            },
            
            'MAPbCl3': {
                'name': 'Methylammonium Lead Chloride',
                'bandgap': 2.97,
                'type': 'direct',
                'cte': 3.5e-5,
                'humidity_score': 5.0,
                'cost_per_cm2': 0.25,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_mapbcl3_nk(),
                'absorption_coefficient': 6e4,
                'urbach_energy': 10,
            },
            
            # Formamidinium perovskites
            'FAPbI3': {
                'name': 'Formamidinium Lead Iodide',
                'bandgap': 1.48,
                'type': 'direct',
                'cte': 4.5e-5,
                'humidity_score': 3.0,
                'cost_per_cm2': 0.18,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_fapbi3_nk(),
                'absorption_coefficient': 1.2e5,
                'urbach_energy': 18,
                'yellow_phase_instability': True,
            },
            
            # Cesium perovskites
            'CsPbI3': {
                'name': 'Cesium Lead Iodide',
                'bandgap': 1.73,
                'type': 'direct',
                'cte': 2.8e-5,  # Lower CTE (inorganic)
                'humidity_score': 6.0,
                'cost_per_cm2': 0.22,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_cspbi3_nk(),
                'absorption_coefficient': 9e4,
                'phase_stability_temp': 593,  # K (black phase stable above)
            },
            
            'CsPbBr3': {
                'name': 'Cesium Lead Bromide',
                'bandgap': 2.36,
                'type': 'direct',
                'cte': 2.5e-5,
                'humidity_score': 8.0,  # Best stability in Track B
                'cost_per_cm2': 0.28,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_cspbbr3_nk(),
                'absorption_coefficient': 7e4,
                'green_emission': 520,  # nm (for LED applications)
            }
        }
        
        return materials
    
    # =============================================================================
    # OPTICAL PROPERTY GENERATORS (n/k spectra)
    # =============================================================================
    
    def _generate_si_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for crystalline silicon (Palik handbook)"""
        wavelengths = self.wavelength_range
        
        # Simplified dispersion model for c-Si
        n = 3.5 + 0.1 / (wavelengths / 1000) ** 2  # Cauchy formula approximation
        
        # Absorption coefficient to k conversion: k = α⋅λ/(4π)
        alpha = np.where(
            wavelengths < 1100,  # Band edge ~1.1 μm
            1e4 * np.exp(-(1240 / wavelengths - 1.12) / 0.1),  # Exponential tail
            10  # Weak sub-bandgap absorption
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_asi_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for amorphous silicon"""
        wavelengths = self.wavelength_range
        
        n = 4.0 + 0.05 / (wavelengths / 1000) ** 2
        
        # Stronger absorption due to direct bandgap nature
        alpha = np.where(
            wavelengths < 700,  # Band edge ~1.75 eV
            5e4 * np.exp(-(1240 / wavelengths - 1.75) / 0.08),
            50
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_gaas_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for GaAs (Adachi model)"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths  # eV
        
        # Adachi critical point model (simplified)
        n = 3.3 + 0.2 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        n = np.where(photon_energy < 1.42, n, 3.8)  # Above bandgap
        
        alpha = np.where(
            photon_energy > 1.42,
            1e5 * np.maximum(photon_energy - 1.42, 0) ** 0.5,  # Direct bandgap
            10 * np.exp((photon_energy - 1.42) / 0.025)  # Urbach tail
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_gainp_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for Ga₀.₅₁In₀.₄₉P"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.1 + 0.15 / np.maximum((photon_energy - 1.2) ** 2, 1e-6)
        n = np.where(photon_energy < 1.81, n, 3.6)
        
        alpha = np.where(
            photon_energy > 1.81,
            8e4 * np.maximum(photon_energy - 1.81, 0) ** 0.5,
            5 * np.exp((photon_energy - 1.81) / 0.020)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_cigs_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CIGS (Cu(In,Ga)Se₂)"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        bandgap = 1.15  # Default, tunable with Ga content
        
        n = 2.8 + 0.3 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        n = np.where(photon_energy < bandgap, n, 3.2)
        
        alpha = np.where(
            photon_energy > bandgap,
            1e5 * np.maximum(photon_energy - bandgap, 0) ** 0.5,
            20 * np.exp((photon_energy - bandgap) / 0.030)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_cdte_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CdTe"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.7 + 0.25 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        n = np.where(photon_energy < 1.45, n, 3.1)
        
        alpha = np.where(
            photon_energy > 1.45,
            1e5 * np.maximum(photon_energy - 1.45, 0) ** 0.5,
            15 * np.exp((photon_energy - 1.45) / 0.025)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_opv_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for P3HT:PCBM organic blend"""
        wavelengths = self.wavelength_range
        
        # Organic materials have complex absorption features
        n = 1.7 + 0.02 * np.sin(2 * np.pi * wavelengths / 200)  # Weak dispersion
        
        # P3HT absorption peak ~520nm, PCBM ~330nm
        alpha_p3ht = 5e4 * np.exp(-((wavelengths - 520) / 80) ** 2)
        alpha_pcbm = 3e4 * np.exp(-((wavelengths - 330) / 60) ** 2)
        alpha = alpha_p3ht + alpha_pcbm + 1000  # Background
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_pbsqd_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PbS quantum dots"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        # Size-dependent bandgap (effective mass approximation)
        dot_size = 3.5  # nm
        bandgap = 0.41 + 1.8 / dot_size ** 2  # Quantum confinement
        
        n = 2.5 + 0.1 / np.maximum((photon_energy - 0.3) ** 2, 1e-6)
        
        alpha = np.where(
            photon_energy > bandgap,
            8e4 * np.maximum(photon_energy - bandgap, 0) ** 0.5,
            100 * np.exp((photon_energy - bandgap) / 0.040)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_pbseqd_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PbSe quantum dots"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        dot_size = 4.2  # nm
        bandgap = 0.28 + 1.5 / dot_size ** 2
        
        n = 2.8 + 0.12 / np.maximum((photon_energy - 0.2) ** 2, 1e-6)
        
        alpha = np.where(
            photon_energy > bandgap,
            9e4 * np.maximum(photon_energy - bandgap, 0) ** 0.5,
            150 * np.exp((photon_energy - bandgap) / 0.045)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    # Perovskite n/k generators
    def _generate_mapbi3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for MAPbI₃ (Ball et al. 2015)"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        # High refractive index ~2.5-2.8
        n = 2.6 + 0.05 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        n = np.where(photon_energy < 1.55, n, 2.8)
        
        # Sharp absorption onset with Urbach tail
        alpha = np.where(
            photon_energy > 1.55,
            1e5 * np.maximum(photon_energy - 1.55, 0) ** 0.5,
            10 * np.exp((photon_energy - 1.55) / 0.015)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_mapbbr3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for MAPbBr₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.4 + 0.04 / np.maximum((photon_energy - 1.5) ** 2, 1e-6)
        n = np.where(photon_energy < 2.3, n, 2.6)
        
        alpha = np.where(
            photon_energy > 2.3,
            8e4 * np.maximum(photon_energy - 2.3, 0) ** 0.5,
            5 * np.exp((photon_energy - 2.3) / 0.012)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_mapbcl3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for MAPbCl₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.2 + 0.03 / np.maximum((photon_energy - 2.0) ** 2, 1e-6)
        n = np.where(photon_energy < 2.97, n, 2.4)
        
        alpha = np.where(
            photon_energy > 2.97,
            6e4 * np.maximum(photon_energy - 2.97, 0) ** 0.5,
            2 * np.exp((photon_energy - 2.97) / 0.010)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_fapbi3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for FAPbI₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.7 + 0.06 / np.maximum((photon_energy - 0.9) ** 2, 1e-6)
        n = np.where(photon_energy < 1.48, n, 2.9)
        
        alpha = np.where(
            photon_energy > 1.48,
            1.2e5 * np.maximum(photon_energy - 1.48, 0) ** 0.5,
            12 * np.exp((photon_energy - 1.48) / 0.018)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_cspbi3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for CsPbI₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.5 + 0.05 / np.maximum((photon_energy - 1.1) ** 2, 1e-6)
        n = np.where(photon_energy < 1.73, n, 2.7)
        
        alpha = np.where(
            photon_energy > 1.73,
            9e4 * np.maximum(photon_energy - 1.73, 0) ** 0.5,
            8 * np.exp((photon_energy - 1.73) / 0.013)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    def _generate_cspbbr3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k for CsPbBr₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.3 + 0.04 / np.maximum((photon_energy - 1.6) ** 2, 1e-6)
        n = np.where(photon_energy < 2.36, n, 2.5)
        
        alpha = np.where(
            photon_energy > 2.36,
            7e4 * np.maximum(photon_energy - 2.36, 0) ** 0.5,
            4 * np.exp((photon_energy - 2.36) / 0.011)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        
        return n, k
    
    # =============================================================================
    # MIXED MATERIAL PROPERTIES (Vegard's Law)
    # =============================================================================
    
    def get_mixed_halide_properties(self, composition: Dict[str, float]) -> Dict:
        """
        Calculate properties of mixed halide perovskites using Vegard's law.
        
        Args:
            composition: Dictionary with halide fractions, e.g., {'I': 0.6, 'Br': 0.4}
            
        Returns:
            Mixed material properties with bowing parameter correction
        """
        
        if not np.isclose(sum(composition.values()), 1.0, rtol=1e-6):
            raise ValueError("Composition fractions must sum to 1.0")
        
        # Reference bandgaps (for MAPb system)
        ref_bandgaps = {'I': 1.55, 'Br': 2.30, 'Cl': 2.97}
        ref_cte = {'I': 4.2e-5, 'Br': 3.8e-5, 'Cl': 3.5e-5}
        ref_stability = {'I': 2.5, 'Br': 4.0, 'Cl': 5.0}
        
        # Linear interpolation (Vegard's law)
        bandgap_linear = sum(composition[halide] * ref_bandgaps[halide] 
                           for halide in composition)
        cte_mixed = sum(composition[halide] * ref_cte[halide] 
                       for halide in composition)
        stability_mixed = sum(composition[halide] * ref_stability[halide] 
                             for halide in composition)
        
        # Bowing parameter correction for bandgap
        # E_g(mixed) = E_linear - b * x * (1-x) where b is bowing parameter
        bowing_parameters = {
            ('I', 'Br'): 0.33,  # eV, from literature
            ('I', 'Cl'): 0.65,
            ('Br', 'Cl'): 0.23
        }
        
        bandgap_corrected = bandgap_linear
        for (halide1, halide2), bowing in bowing_parameters.items():
            if halide1 in composition and halide2 in composition:
                x1, x2 = composition[halide1], composition[halide2]
                bandgap_corrected -= bowing * x1 * x2
        
        # Generate mixed n/k data (simple interpolation)
        n_mixed = np.zeros_like(self.wavelength_range, dtype=float)
        k_mixed = np.zeros_like(self.wavelength_range, dtype=float)
        
        for halide, fraction in composition.items():
            if halide == 'I':
                n_ref, k_ref = self._generate_mapbi3_nk()
            elif halide == 'Br':
                n_ref, k_ref = self._generate_mapbbr3_nk()
            elif halide == 'Cl':
                n_ref, k_ref = self._generate_mapbcl3_nk()
            else:
                raise ValueError(f"Unknown halide: {halide}")
            
            n_mixed += fraction * n_ref
            k_mixed += fraction * k_ref
        
        return {
            'name': f"Mixed Halide {composition}",
            'bandgap': bandgap_corrected,
            'bandgap_linear': bandgap_linear,
            'bowing_correction': bandgap_linear - bandgap_corrected,
            'type': 'direct',
            'cte': cte_mixed,
            'humidity_score': stability_mixed,
            'cost_per_cm2': 0.15 + 0.05 * len(composition),  # Complexity penalty
            'deposition': 'solution_processing',
            'n_k_data': (n_mixed, k_mixed),
            'composition': composition,
            'phase_segregation_risk': self._calculate_phase_segregation_risk(composition)
        }
    
    def _calculate_phase_segregation_risk(self, composition: Dict[str, float]) -> float:
        """
        Calculate Hoke effect risk for mixed halide perovskites.
        Returns risk score (0-10, higher = more risk)
        """
        
        if 'I' in composition and 'Br' in composition:
            # I/Br mixing most prone to segregation
            i_fraction = composition['I']
            risk = 8.0 * 4 * i_fraction * (1 - i_fraction)  # Maximum at 50:50
        else:
            risk = 1.0  # Other combinations more stable
        
        return min(risk, 10.0)
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def get_material(self, name: str, track: str = 'A') -> Dict:
        """Get material properties by name and track"""
        
        materials = self.materials_track_a if track == 'A' else self.materials_track_b
        
        if name not in materials:
            available = list(materials.keys())
            raise ValueError(f"Material '{name}' not found in Track {track}. "
                           f"Available: {available}")
        
        return materials[name].copy()
    
    def list_materials(self, track: Optional[str] = None) -> List[str]:
        """List available materials"""
        
        if track == 'A':
            return list(self.materials_track_a.keys())
        elif track == 'B':
            return list(self.materials_track_b.keys())
        else:
            return list(self.materials_track_a.keys()) + list(self.materials_track_b.keys())
    
    def get_bandgap_range(self, track: str = 'A') -> Tuple[float, float]:
        """Get the range of available bandgaps for a track"""
        
        materials = self.materials_track_a if track == 'A' else self.materials_track_b
        bandgaps = [mat['bandgap'] for mat in materials.values()]
        
        return min(bandgaps), max(bandgaps)

# =============================================================================
# SOLAR SPECTRUM DATA
# =============================================================================

def get_am15g_spectrum(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Get AM1.5G solar spectrum interpolated to given wavelengths.
    
    Args:
        wavelength_nm: Wavelength array in nanometers
        
    Returns:
        Spectral irradiance in W⋅m⁻²⋅nm⁻¹
        
    Reference: ASTM G173-03 Standard Tables
    """
    
    # Simplified AM1.5G model (Planck + atmospheric absorption)
    # For production code, use NREL data tables
    
    photon_energy = 1240 / wavelength_nm  # eV
    wavelength_m = wavelength_nm * 1e-9
    
    # Blackbody at 5778K modified by atmospheric transmission
    planck = (2 * H * C**2 / wavelength_m**5) / (np.exp(H * C / (wavelength_m * KB * T_SUN)) - 1)
    
    # Simplified atmospheric transmission
    transmission = np.where(
        (wavelength_nm >= 300) & (wavelength_nm <= 1200),
        0.8 * np.exp(-0.5 * ((wavelength_nm - 500) / 300)**2),  # Peak transmission ~500nm
        0.4 * np.exp(-abs(wavelength_nm - 800) / 500)  # IR tail
    )
    
    # Convert to W⋅m⁻²⋅nm⁻¹ and normalize to AM1.5G total
    irradiance = planck * transmission * 1e-9  # J⋅s⁻¹⋅m⁻²⋅nm⁻¹ = W⋅m⁻²⋅nm⁻¹
    
    # Normalize to standard AM1.5G flux (1000.37 W/m²)
    # Use abs() because wavelength array may be descending
    total_flux = abs(np.trapezoid(irradiance, wavelength_nm))
    if total_flux > 0:
        irradiance *= AM15G_FLUX_TOTAL / total_flux
    irradiance = np.abs(irradiance)  # Ensure positive irradiance
    
    return irradiance

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Default simulation parameters
DEFAULT_CONFIG = {
    'temperature': T_CELL,                    # K
    'concentration': CONCENTRATION,           # Suns
    'wavelength_min': 300,                   # nm
    'wavelength_max': 1550,                  # nm
    'wavelength_points': 126,                # Resolution
    'max_junctions': 10,                     # Practical limit for first implementation
    'convergence_tolerance': 1e-6,           # Numerical tolerance
    'max_iterations': 1000,                  # Optimization limit
    
    # Economic parameters
    'discount_rate': 0.08,                   # WACC
    'system_lifetime': 25,                   # years
    'degradation_rate': 0.5,                 # %/year (compound)
    'installation_cost': 1.5,               # USD/Wp
    'bos_cost': 0.8,                        # USD/Wp (balance of systems)
    
    # Physical constraints
    'min_layer_thickness': 50e-9,           # m (50 nm minimum)
    'max_layer_thickness': 50e-6,           # m (50 μm maximum)
    'max_cte_mismatch': 30e-6,              # /K (thermal stress limit)
    'min_humidity_score': 5.0,              # Stability requirement
    'max_interface_resistance': 1e-3,       # Ω⋅cm² (tunneling junction)
}

# Initialize global material database
MATERIAL_DB = MaterialDatabase()

if __name__ == "__main__":
    # Configuration validation and testing
    print("N-Junction Tandem PV Simulator - Configuration Test")
    print("=" * 60)
    
    print(f"Total Track A materials: {len(MATERIAL_DB.list_materials('A'))}")
    print(f"Total Track B materials: {len(MATERIAL_DB.list_materials('B'))}")
    
    print("\nTrack A bandgap range:", MATERIAL_DB.get_bandgap_range('A'))
    print("Track B bandgap range:", MATERIAL_DB.get_bandgap_range('B'))
    
    # Test mixed halide calculation
    print("\nTesting mixed halide (50% I, 50% Br):")
    mixed = MATERIAL_DB.get_mixed_halide_properties({'I': 0.5, 'Br': 0.5})
    print(f"  Linear Eg: {mixed['bandgap_linear']:.3f} eV")
    print(f"  Corrected Eg: {mixed['bandgap']:.3f} eV")
    print(f"  Bowing correction: {mixed['bowing_correction']:.3f} eV")
    print(f"  Phase segregation risk: {mixed['phase_segregation_risk']:.1f}/10")
    
    # Test AM1.5G spectrum
    wavelengths = np.linspace(300, 1200, 100)
    spectrum = get_am15g_spectrum(wavelengths)
    total_power = np.trapezoid(spectrum, wavelengths)
    print(f"\nAM1.5G spectrum integration (300-1200nm): {total_power:.1f} W/m²")
    
    print("\n✅ Configuration loaded successfully!")