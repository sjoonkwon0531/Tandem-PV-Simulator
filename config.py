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
            # Silicon family
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
            
            'µc-Si': {
                'name': 'Microcrystalline Silicon',
                'bandgap': 1.1,
                'type': 'indirect',
                'cte': 2.8e-6,
                'humidity_score': 8.0,
                'cost_per_cm2': 0.03,
                'deposition': 'pecvd',
                'n_k_data': self._generate_si_nk(),
                'absorption_coefficient': 2e4,
                'crystalline_fraction': 0.6,   # Mixed phase
            },
            
            'SiGe': {
                'name': 'Silicon Germanium',
                'bandgap': 0.9,  # Variable 0.7-1.1 eV with Ge content
                'bandgap_range': (0.7, 1.1),
                'type': 'indirect',
                'cte': 4.2e-6,
                'humidity_score': 8.5,
                'cost_per_cm2': 0.12,
                'deposition': 'epitaxy',
                'n_k_data': self._generate_sige_nk(),
                'ge_fraction': 0.3,  # Default Ge content
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
            
            'InGaAs': {
                'name': 'Indium Gallium Arsenide',
                'bandgap': 1.0,  # Variable 0.7-1.4 eV with In content
                'bandgap_range': (0.7, 1.4),
                'type': 'direct',
                'cte': 5.9e-6,
                'humidity_score': 7.8,
                'cost_per_cm2': 4.00,
                'deposition': 'mocvd',
                'n_k_data': self._generate_ingaas_nk(),
                'indium_fraction': 0.2,  # Default composition
            },
            
            'InP': {
                'name': 'Indium Phosphide',
                'bandgap': 1.34,
                'type': 'direct',
                'cte': 4.6e-6,
                'humidity_score': 8.2,
                'cost_per_cm2': 5.00,
                'deposition': 'mocvd',
                'n_k_data': self._generate_inp_nk(),
                'lattice_constant': 5.87,  # Å
            },
            
            'AlGaAs': {
                'name': 'Aluminum Gallium Arsenide',
                'bandgap': 2.0,  # Variable 1.42-2.16 eV
                'bandgap_range': (1.42, 2.16),
                'type': 'direct',  # Becomes indirect at high Al
                'cte': 5.2e-6,
                'humidity_score': 7.5,
                'cost_per_cm2': 3.50,
                'deposition': 'mocvd',
                'n_k_data': self._generate_algaas_nk(),
                'aluminum_fraction': 0.3,
            },
            
            'GaSb': {
                'name': 'Gallium Antimonide',
                'bandgap': 0.73,
                'type': 'direct',
                'cte': 6.1e-6,
                'humidity_score': 7.0,
                'cost_per_cm2': 6.00,
                'deposition': 'mocvd',
                'n_k_data': self._generate_gasb_nk(),
                'near_infrared': True,
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
            
            'CZTS': {
                'name': 'Copper Zinc Tin Sulfide',
                'bandgap': 1.5,
                'type': 'direct',
                'cte': 9.2e-6,
                'humidity_score': 6.0,
                'cost_per_cm2': 0.20,
                'deposition': 'sputtering',
                'n_k_data': self._generate_czts_nk(),
                'earth_abundant': True,
            },
            
            'Sb2Se3': {
                'name': 'Antimony Selenide',
                'bandgap': 1.1,
                'type': 'direct',
                'cte': 12e-6,
                'humidity_score': 5.5,
                'cost_per_cm2': 0.15,
                'deposition': 'thermal_evaporation',
                'n_k_data': self._generate_sb2se3_nk(),
                'emerging_technology': True,
            },
            
            # Organic photovoltaics
            'P3HT:PCBM': {
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
            
            'PTB7': {
                'name': 'PTB7:PC71BM Blend',
                'bandgap': 1.6,
                'type': 'direct',
                'cte': 140e-6,
                'humidity_score': 2.5,
                'cost_per_cm2': 0.15,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_ptb7_nk(),
                'low_bandgap_polymer': True,
            },
            
            'PM6:Y6': {
                'name': 'PM6:Y6 Non-fullerene Blend',
                'bandgap': 1.33,
                'type': 'direct',
                'cte': 160e-6,
                'humidity_score': 3.0,
                'cost_per_cm2': 0.12,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_pm6y6_nk(),
                'non_fullerene_acceptor': True,
            },
            
            # Dye-sensitized materials
            'N719': {
                'name': 'N719 Ruthenium Dye',
                'bandgap': 1.7,  # Effective for DSSC
                'type': 'molecular',
                'cte': 100e-6,
                'humidity_score': 4.0,
                'cost_per_cm2': 0.08,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_n719_nk(),
                'dssc_technology': True,
            },
            
            'organic_dyes': {
                'name': 'Organic DSSC Dyes',
                'bandgap': 2.0,  # Variable 1.5-2.5 eV
                'bandgap_range': (1.5, 2.5),
                'type': 'molecular',
                'cte': 120e-6,
                'humidity_score': 3.5,
                'cost_per_cm2': 0.06,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_orgdye_nk(),
                'tunable_absorption': True,
            },
            
            # Quantum dots
            'PbS_QD': {
                'name': 'Lead Sulfide Quantum Dots',
                'bandgap': 1.3,  # Size-tunable: 0.5-1.5 eV
                'bandgap_range': (0.5, 1.5),
                'type': 'direct',
                'cte': 18e-6,
                'humidity_score': 4.0,
                'cost_per_cm2': 0.80,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_pbsqd_nk(),
                'quantum_confinement': True,
                'dot_size': 3.5,  # nm (affects bandgap)
            },
            
            'CsPbI3_QD': {
                'name': 'Cesium Lead Iodide Quantum Dots',
                'bandgap': 1.73,
                'type': 'direct',
                'cte': 15e-6,
                'humidity_score': 6.0,  # Better than organic perovskites
                'cost_per_cm2': 0.90,
                'deposition': 'solution_processing',
                'n_k_data': self._generate_cspbiqd_nk(),
                'quantum_confinement': True,
                'phase_stable_qd': True,
            },
            
            # Other emerging materials
            'Cu2O': {
                'name': 'Copper Oxide',
                'bandgap': 2.1,
                'type': 'direct',
                'cte': 16.9e-6,
                'humidity_score': 7.0,
                'cost_per_cm2': 0.08,
                'deposition': 'electrodeposition',
                'n_k_data': self._generate_cu2o_nk(),
                'earth_abundant': True,
                'p_type_oxide': True,
            },
            
            'SnS': {
                'name': 'Tin Sulfide',
                'bandgap': 1.3,
                'type': 'indirect',
                'cte': 14e-6,
                'humidity_score': 6.5,
                'cost_per_cm2': 0.12,
                'deposition': 'thermal_evaporation',
                'n_k_data': self._generate_sns_nk(),
                'earth_abundant': True,
            },
            
            'BaSi2': {
                'name': 'Barium Disilicide',
                'bandgap': 1.3,
                'type': 'indirect',
                'cte': 18e-6,
                'humidity_score': 8.0,
                'cost_per_cm2': 0.25,
                'deposition': 'sputter_epitaxy',
                'n_k_data': self._generate_basi2_nk(),
                'silicide_semiconductor': True,
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
    
    # Additional n/k generators for new Track A materials
    def _generate_sige_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for SiGe alloy"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        bandgap = 0.9  # Variable with Ge content
        
        # Between Si and Ge properties
        n = 4.2 + 0.15 / (wavelengths / 1000) ** 2
        alpha = np.where(
            wavelengths < 1380,  # Band edge
            2e4 * np.exp(-(1240 / wavelengths - bandgap) / 0.12),
            20
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_ingaas_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for InGaAs"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        bandgap = 1.0
        
        n = 3.6 + 0.2 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > bandgap,
            1.2e5 * np.maximum(photon_energy - bandgap, 0) ** 0.5,
            50 * np.exp((photon_energy - bandgap) / 0.035)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_inp_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for InP"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.2 + 0.18 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.34,
            8e4 * np.maximum(photon_energy - 1.34, 0) ** 0.5,
            15 * np.exp((photon_energy - 1.34) / 0.030)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_algaas_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for AlGaAs"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        bandgap = 2.0  # Variable with Al content
        
        n = 3.0 + 0.15 / np.maximum((photon_energy - 1.5) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > bandgap,
            6e4 * np.maximum(photon_energy - bandgap, 0) ** 0.5,
            8 * np.exp((photon_energy - bandgap) / 0.025)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_gasb_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for GaSb"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 4.0 + 0.3 / np.maximum((photon_energy - 0.5) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 0.73,
            1.5e5 * np.maximum(photon_energy - 0.73, 0) ** 0.5,
            200 * np.exp((photon_energy - 0.73) / 0.050)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_czts_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CZTS"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.9 + 0.25 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.5,
            1.2e5 * np.maximum(photon_energy - 1.5, 0) ** 0.5,
            30 * np.exp((photon_energy - 1.5) / 0.035)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_sb2se3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for Sb2Se3"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.5 + 0.2 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.1,
            1e5 * np.maximum(photon_energy - 1.1, 0) ** 0.5,
            100 * np.exp((photon_energy - 1.1) / 0.040)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_ptb7_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PTB7"""
        wavelengths = self.wavelength_range
        
        n = 1.8 + 0.03 * np.sin(2 * np.pi * wavelengths / 250)
        # PTB7 absorption peak ~700nm
        alpha_ptb7 = 6e4 * np.exp(-((wavelengths - 700) / 100) ** 2)
        alpha_pc71bm = 4e4 * np.exp(-((wavelengths - 380) / 80) ** 2)
        alpha = alpha_ptb7 + alpha_pc71bm + 2000
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_pm6y6_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PM6:Y6"""
        wavelengths = self.wavelength_range
        
        n = 1.9 + 0.02 * np.sin(2 * np.pi * wavelengths / 300)
        # Broad absorption 400-900nm
        alpha_pm6 = 5e4 * np.exp(-((wavelengths - 600) / 120) ** 2)
        alpha_y6 = 7e4 * np.exp(-((wavelengths - 800) / 150) ** 2)  # NIR acceptor
        alpha = alpha_pm6 + alpha_y6 + 3000
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_n719_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for N719 dye"""
        wavelengths = self.wavelength_range
        
        n = 1.6 + 0.01 * np.sin(2 * np.pi * wavelengths / 200)
        # Metal-to-ligand charge transfer bands
        alpha_mlct1 = 8e4 * np.exp(-((wavelengths - 530) / 60) ** 2)
        alpha_mlct2 = 6e4 * np.exp(-((wavelengths - 400) / 50) ** 2)
        alpha = alpha_mlct1 + alpha_mlct2 + 1000
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_orgdye_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for organic dyes"""
        wavelengths = self.wavelength_range
        
        n = 1.7 + 0.02 * np.sin(2 * np.pi * wavelengths / 180)
        # Tunable absorption peak
        alpha = 5e4 * np.exp(-((wavelengths - 550) / 80) ** 2) + 1500
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_cspbiqd_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CsPbI3 quantum dots"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.5 + 0.06 / np.maximum((photon_energy - 1.2) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.73,
            1.1e5 * np.maximum(photon_energy - 1.73, 0) ** 0.5,
            20 * np.exp((photon_energy - 1.73) / 0.020)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_cu2o_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for Cu2O"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.7 + 0.1 / np.maximum((photon_energy - 1.5) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 2.1,
            4e4 * np.maximum(photon_energy - 2.1, 0) ** 0.5,
            50 * np.exp((photon_energy - 2.1) / 0.040)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_sns_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for SnS"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.2 + 0.15 / np.maximum((photon_energy - 0.9) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.3,
            8e4 * np.maximum(photon_energy - 1.3, 0) ** 0.5,
            80 * np.exp((photon_energy - 1.3) / 0.035)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_basi2_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for BaSi2"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.8 + 0.12 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        alpha = np.where(
            photon_energy > 1.3,
            6e4 * np.maximum(photon_energy - 1.3, 0) ** 0.5,
            40 * np.exp((photon_energy - 1.3) / 0.030)
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
    
    # Missing n/k generators for Track A materials
    def _generate_sige_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for SiGe alloy"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        # SiGe with ~30% Ge (variable bandgap)
        n = 3.8 + 0.08 / (wavelengths / 1000) ** 2
        alpha = np.where(
            wavelengths < 1300,  # Band edge ~0.9 eV
            2e4 * np.exp(-(1240 / wavelengths - 0.9) / 0.12),
            50
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_ingaas_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for InGaAs"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.5 + 0.2 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        n = np.where(photon_energy < 1.0, n, 3.8)
        
        alpha = np.where(
            photon_energy > 1.0,
            9e4 * np.maximum(photon_energy - 1.0, 0) ** 0.5,
            20 * np.exp((photon_energy - 1.0) / 0.025)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_inp_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for InP"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.2 + 0.18 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        n = np.where(photon_energy < 1.34, n, 3.6)
        
        alpha = np.where(
            photon_energy > 1.34,
            9e4 * np.maximum(photon_energy - 1.34, 0) ** 0.5,
            15 * np.exp((photon_energy - 1.34) / 0.022)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_algaas_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for AlGaAs"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        # Al₀.₃Ga₀.₇As composition
        n = 3.4 + 0.15 / np.maximum((photon_energy - 1.2) ** 2, 1e-6)
        n = np.where(photon_energy < 2.0, n, 3.7)
        
        alpha = np.where(
            photon_energy > 2.0,
            7e4 * np.maximum(photon_energy - 2.0, 0) ** 0.5,
            10 * np.exp((photon_energy - 2.0) / 0.018)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_gasb_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for GaSb"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.8 + 0.25 / np.maximum((photon_energy - 0.5) ** 2, 1e-6)
        n = np.where(photon_energy < 0.73, n, 4.2)
        
        alpha = np.where(
            photon_energy > 0.73,
            1.2e5 * np.maximum(photon_energy - 0.73, 0) ** 0.5,
            30 * np.exp((photon_energy - 0.73) / 0.030)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_czts_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CZTS"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 2.9 + 0.3 / np.maximum((photon_energy - 1.0) ** 2, 1e-6)
        n = np.where(photon_energy < 1.5, n, 3.3)
        
        alpha = np.where(
            photon_energy > 1.5,
            9e4 * np.maximum(photon_energy - 1.5, 0) ** 0.5,
            25 * np.exp((photon_energy - 1.5) / 0.028)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_sb2se3_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for Sb₂Se₃"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        n = 3.5 + 0.2 / np.maximum((photon_energy - 0.8) ** 2, 1e-6)
        n = np.where(photon_energy < 1.1, n, 3.8)
        
        alpha = np.where(
            photon_energy > 1.1,
            8e4 * np.maximum(photon_energy - 1.1, 0) ** 0.5,
            40 * np.exp((photon_energy - 1.1) / 0.032)
        )
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_ptb7_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PTB7:PC71BM"""
        wavelengths = self.wavelength_range
        
        # Low bandgap polymer with broader absorption
        n = 1.8 + 0.03 * np.sin(2 * np.pi * wavelengths / 150)
        
        # PTB7 absorption peak ~700nm, PC71BM ~380nm
        alpha_ptb7 = 6e4 * np.exp(-((wavelengths - 700) / 120) ** 2)
        alpha_pcbm = 4e4 * np.exp(-((wavelengths - 380) / 80) ** 2)
        alpha = alpha_ptb7 + alpha_pcbm + 2000  # Background
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_pm6y6_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for PM6:Y6"""
        wavelengths = self.wavelength_range
        
        n = 1.75 + 0.04 * np.sin(2 * np.pi * wavelengths / 180)
        
        # PM6 ~620nm, Y6 near-IR ~850nm
        alpha_pm6 = 7e4 * np.exp(-((wavelengths - 620) / 100) ** 2)
        alpha_y6 = 5e4 * np.exp(-((wavelengths - 850) / 150) ** 2)
        alpha = alpha_pm6 + alpha_y6 + 1500  # Background
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_n719_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for N719 dye"""
        wavelengths = self.wavelength_range
        
        n = 1.6 + 0.02 * np.cos(2 * np.pi * wavelengths / 200)
        
        # N719 broad absorption 400-800nm with peak ~530nm
        alpha = 8e4 * np.exp(-((wavelengths - 530) / 180) ** 2) + 1000
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_orgdye_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for organic DSSC dyes"""
        wavelengths = self.wavelength_range
        
        n = 1.65 + 0.03 * np.sin(2 * np.pi * wavelengths / 160)
        
        # Tunable absorption, example peak ~480nm
        alpha = 6e4 * np.exp(-((wavelengths - 480) / 140) ** 2) + 800
        
        k = alpha * wavelengths * 1e-9 / (4 * np.pi)
        return n, k
    
    def _generate_cspbiqd_nk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n/k data for CsPbI₃ quantum dots"""
        wavelengths = self.wavelength_range
        photon_energy = 1240 / wavelengths
        
        # QD bandgap ~1.73 eV (size-dependent)
        n = 2.4 + 0.08 / np.maximum((photon_energy - 1.2) ** 2, 1e-6)
        
        alpha = np.where(
            photon_energy > 1.73,
            7e4 * np.maximum(photon_energy - 1.73, 0) ** 0.5,
            60 * np.exp((photon_energy - 1.73) / 0.035)
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
# ABX₃ PEROVSKITE COMPOSITIONAL SPACE
# =============================================================================

class ABX3_CompositionSpace:
    """
    Defines the ABX₃ perovskite compositional space for solid solution modeling.
    
    Includes A-site, B-site, and X-site cations/anions with their properties.
    """
    
    def __init__(self):
        # A-site cations (organic and inorganic)
        self.a_site_species = {
            'MA': {  # Methylammonium CH₃NH₃⁺
                'name': 'Methylammonium',
                'formula': 'CH₃NH₃⁺',
                'ionic_radius': 1.8,    # Å (estimated)
                'electronegativity': 2.2,
                'molecular_weight': 32.04,  # g/mol
                'stability_factor': 0.6,    # Thermal/humidity stability
                'common_combinations': ['Pb', 'Sn'],
            },
            
            'FA': {  # Formamidinium CH(NH₂)₂⁺
                'name': 'Formamidinium',
                'formula': 'CH(NH₂)₂⁺',
                'ionic_radius': 1.9,
                'electronegativity': 2.1,
                'molecular_weight': 46.07,
                'stability_factor': 0.7,
                'common_combinations': ['Pb', 'Sn'],
            },
            
            'Cs': {  # Cesium Cs⁺
                'name': 'Cesium',
                'formula': 'Cs⁺',
                'ionic_radius': 1.67,   # CN=8
                'electronegativity': 0.79,
                'molecular_weight': 132.91,
                'stability_factor': 0.9,  # Inorganic = more stable
                'common_combinations': ['Pb', 'Sn', 'Ge'],
            },
            
            'Rb': {  # Rubidium Rb⁺
                'name': 'Rubidium',
                'formula': 'Rb⁺',
                'ionic_radius': 1.52,
                'electronegativity': 0.82,
                'molecular_weight': 85.47,
                'stability_factor': 0.8,
                'common_combinations': ['Pb'],
            }
        }
        
        # B-site cations (metal cations in octahedral coordination)
        self.b_site_species = {
            'Pb': {  # Lead Pb²⁺
                'name': 'Lead',
                'formula': 'Pb²⁺',
                'ionic_radius': 1.19,   # CN=6
                'electronegativity': 2.33,
                'molecular_weight': 207.20,
                'oxidation_state': 2,
                'toxicity_concern': True,
                'band_character': 's-p',
                'common_combinations': ['I', 'Br', 'Cl'],
            },
            
            'Sn': {  # Tin Sn²⁺
                'name': 'Tin',
                'formula': 'Sn²⁺',
                'ionic_radius': 1.10,
                'electronegativity': 1.96,
                'molecular_weight': 118.71,
                'oxidation_state': 2,
                'toxicity_concern': False,
                'band_character': 's-p',
                'oxidation_stability': 'poor',
                'common_combinations': ['I', 'Br'],
            },
            
            'Ge': {  # Germanium Ge²⁺
                'name': 'Germanium',
                'formula': 'Ge²⁺',
                'ionic_radius': 0.87,
                'electronegativity': 2.01,
                'molecular_weight': 72.64,
                'oxidation_state': 2,
                'toxicity_concern': False,
                'band_character': 's-p',
                'stability_factor': 0.6,
                'common_combinations': ['I'],
            }
        }
        
        # X-site anions (halides)
        self.x_site_species = {
            'I': {  # Iodide I⁻
                'name': 'Iodide',
                'formula': 'I⁻',
                'ionic_radius': 2.20,   # CN=6
                'electronegativity': 2.66,
                'molecular_weight': 126.90,
                'band_character': 'p',
                'stability_air': 'moderate',
                'absorption_edge': 'red',
            },
            
            'Br': {  # Bromide Br⁻
                'name': 'Bromide',
                'formula': 'Br⁻',
                'ionic_radius': 1.96,
                'electronegativity': 2.96,
                'molecular_weight': 79.90,
                'band_character': 'p',
                'stability_air': 'good',
                'absorption_edge': 'yellow',
            },
            
            'Cl': {  # Chloride Cl⁻
                'name': 'Chloride',
                'formula': 'Cl⁻',
                'ionic_radius': 1.81,
                'electronegativity': 3.16,
                'molecular_weight': 35.45,
                'band_character': 'p',
                'stability_air': 'excellent',
                'absorption_edge': 'blue',
            }
        }
        
        # Bowing parameters for mixed compositions
        self.bowing_parameters = {
            ('I', 'Br'): 0.33, ('I', 'Cl'): 0.65, ('Br', 'Cl'): 0.23,
            ('Pb', 'Sn'): 0.25, ('Pb', 'Ge'): 0.40, ('Sn', 'Ge'): 0.20,
            ('MA', 'FA'): 0.05, ('MA', 'Cs'): 0.15, ('FA', 'Cs'): 0.12, ('Cs', 'Rb'): 0.03
        }
    
    def calculate_tolerance_factor(self, a_comp: Dict[str, float], 
                                 b_comp: Dict[str, float], 
                                 x_comp: Dict[str, float]) -> float:
        """Calculate Goldschmidt tolerance factor for mixed composition."""
        
        r_a = sum(frac * self.a_site_species[ion]['ionic_radius'] 
                 for ion, frac in a_comp.items() if frac > 0)
        r_b = sum(frac * self.b_site_species[ion]['ionic_radius'] 
                 for ion, frac in b_comp.items() if frac > 0)
        r_x = sum(frac * self.x_site_species[ion]['ionic_radius'] 
                 for ion, frac in x_comp.items() if frac > 0)
        
        if r_b == 0 or r_x == 0:
            return 0.0
        
        return (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))

# Global instance
ABX3_SPACE = ABX3_CompositionSpace()

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

# =============================================================================
# TRACK B: ABX₃ SOLID SOLUTION COMPONENTS  
# =============================================================================

# A-site cations with ionic radii (Shannon 1976) and properties
A_SITE_IONS = {
    'MA': {  # Methylammonium CH₃NH₃⁺
        'ionic_radius': 1.8,     # Å (estimated, effective)
        'mass': 32.04,           # g/mol
        'electronegativity': 2.2, # Pauling scale (estimated)
        'stability_score': 3.0,  # /10 (humidity stability)
        'cost_factor': 1.0,      # Relative cost multiplier
        'organic': True,
        'dipole_moment': 2.29    # Debye
    },
    'FA': {  # Formamidinium CH(NH₂)₂⁺
        'ionic_radius': 1.9,     # Å 
        'mass': 45.04,           # g/mol
        'electronegativity': 2.1,
        'stability_score': 4.0,  # Slightly more stable than MA
        'cost_factor': 1.2,
        'organic': True,
        'dipole_moment': 0.2     # Much smaller dipole
    },
    'Cs': {  # Cesium Cs⁺
        'ionic_radius': 1.67,    # Å (6-coordinate)
        'mass': 132.91,          # g/mol
        'electronegativity': 0.79,
        'stability_score': 8.0,  # High thermal stability
        'cost_factor': 2.5,      # Expensive
        'organic': False,
        'polarizability': 59.6   # Å³
    },
    'Rb': {  # Rubidium Rb⁺
        'ionic_radius': 1.52,    # Å (6-coordinate)
        'mass': 85.47,           # g/mol
        'electronegativity': 0.82,
        'stability_score': 7.5,  # Good stability
        'cost_factor': 3.0,      # More expensive than Cs
        'organic': False,
        'polarizability': 40.8   # Å³
    }
}

# B-site cations (metal centers)
B_SITE_IONS = {
    'Pb': {  # Lead Pb²⁺
        'ionic_radius': 1.19,    # Å (6-coordinate)
        'mass': 207.2,           # g/mol
        'electronegativity': 2.33,
        'stability_score': 7.0,  # Good for perovskites
        'cost_factor': 1.0,      # Reference
        'oxidation_state': 2,
        'electron_config': '[Xe] 4f¹⁴ 5d¹⁰ 6s²',
        'toxicity': 'high'       # Environmental concern
    },
    'Sn': {  # Tin Sn²⁺
        'ionic_radius': 1.10,    # Å (6-coordinate)
        'mass': 118.71,          # g/mol
        'electronegativity': 1.96,
        'stability_score': 4.0,  # Prone to oxidation Sn²⁺→Sn⁴⁺
        'cost_factor': 0.8,      # Cheaper than Pb
        'oxidation_state': 2,
        'electron_config': '[Kr] 4d¹⁰ 5s²',
        'toxicity': 'low'
    },
    'Ge': {  # Germanium Ge²⁺
        'ionic_radius': 0.87,    # Å (6-coordinate)
        'mass': 72.63,           # g/mol
        'electronegativity': 2.01,
        'stability_score': 5.0,  # Moderate stability
        'cost_factor': 4.0,      # Expensive semiconductor material
        'oxidation_state': 2,
        'electron_config': '[Ar] 3d¹⁰ 4s²',
        'toxicity': 'low'
    }
}

# X-site halide anions
X_SITE_IONS = {
    'I': {   # Iodide I⁻
        'ionic_radius': 2.20,    # Å (6-coordinate)
        'mass': 126.90,          # g/mol
        'electronegativity': 2.66,
        'stability_score': 6.0,  # Good for perovskites
        'cost_factor': 1.5,      # Moderate cost
        'polarizability': 7.1,   # Å³ (high)
        'bandgap_contribution': -0.5  # Lowers bandgap
    },
    'Br': {  # Bromide Br⁻
        'ionic_radius': 1.96,    # Å (6-coordinate)
        'mass': 79.90,           # g/mol
        'electronegativity': 2.96,
        'stability_score': 7.5,  # Better stability than I
        'cost_factor': 2.0,      # More expensive than I
        'polarizability': 4.2,   # Å³
        'bandgap_contribution': 0.0  # Reference
    },
    'Cl': {  # Chloride Cl⁻
        'ionic_radius': 1.81,    # Å (6-coordinate)
        'mass': 35.45,           # g/mol
        'electronegativity': 3.16,
        'stability_score': 8.5,  # Highest stability
        'cost_factor': 1.0,      # Cheapest
        'polarizability': 3.0,   # Å³ (lowest)
        'bandgap_contribution': 0.8  # Increases bandgap
    }
}

# NREL efficiency records for benchmarking (as of 2024)
NREL_RECORDS = {
    'single_junction': {
        'Si': {'PCE': 26.7, 'year': 2017, 'organization': 'Kaneka', 'area_cm2': 180.4},
        'GaAs': {'PCE': 29.1, 'year': 2019, 'organization': 'Alta Devices', 'area_cm2': 0.9927},
        'CIGS': {'PCE': 23.35, 'year': 2019, 'organization': 'Solar Frontier', 'area_cm2': 0.9927},
        'CdTe': {'PCE': 22.1, 'year': 2016, 'organization': 'First Solar', 'area_cm2': 1.062},
        'Perovskite': {'PCE': 25.7, 'year': 2021, 'organization': 'UNIST', 'area_cm2': 0.0937},
        'OPV': {'PCE': 19.6, 'year': 2023, 'organization': 'Tsinghua', 'area_cm2': 0.0516}
    },
    'multi_junction': {
        'GaInP/GaAs/Ge_3J': {'PCE': 39.2, 'year': 2013, 'organization': 'Sharp', 'area_cm2': 30.28, 'concentration': '302x'},
        'GaInP/GaAs/InGaAs_3J': {'PCE': 37.9, 'year': 2013, 'organization': 'Solar Junction', 'area_cm2': 5.54, 'concentration': '418x'},
        'GaInP/Si_2J': {'PCE': 32.8, 'year': 2017, 'organization': 'NREL', 'area_cm2': 1.0, 'concentration': '1x'},
        'Perovskite/Si_2J': {'PCE': 31.3, 'year': 2020, 'organization': 'EPFL', 'area_cm2': 1.43, 'concentration': '1x'},
        'CIGS/Perovskite_2J': {'PCE': 24.2, 'year': 2021, 'organization': 'HZB', 'area_cm2': 1.0, 'concentration': '1x'}
    },
    'emerging': {
        'Quantum_Dots': {'PCE': 16.6, 'year': 2021, 'organization': 'U. Toronto', 'area_cm2': 1.0},
        'Organic_Tandem': {'PCE': 17.3, 'year': 2019, 'organization': 'U. Michigan', 'area_cm2': 0.041},
        'All_Perovskite_2J': {'PCE': 26.4, 'year': 2022, 'organization': 'KAUST', 'area_cm2': 0.049}
    }
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