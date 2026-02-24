#!/usr/bin/env python3
"""
Interface Energy and Stability Analysis
=======================================

Module for calculating interface energies and thermodynamic stability
of multi-layer perovskite tandem devices.

Features:
- Lattice mismatch strain energy calculation
- Goldschmidt tolerance factor analysis
- Surface/interface energy modeling (Dupré adhesion)
- Halide interdiffusion barriers
- Thermodynamic stability assessment

References:
- Shannon (1976) Acta Crystallogr. A32:751 - Ionic radii database
- Goldschmidt (1926) tolerance factor t = (r_A + r_X)/[√2(r_B + r_X)]
- Dupré adhesion: W = γ₁ + γ₂ - γ₁₂
- Butler-Volmer interdiffusion kinetics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class InterfaceStability:
    """Results of interface stability analysis"""
    is_stable: bool
    delta_g: float  # kJ/mol
    strain_energy: float  # meV/atom
    tolerance_mismatch: float  # Δt
    adhesion_energy: float  # mJ/m²
    interdiffusion_barrier: float  # eV
    risk_factors: List[str]
    stability_score: float  # 0-10 (10 = most stable)

@dataclass
class LayerProperties:
    """Properties of a single layer for interface calculations"""
    composition: Dict[str, float]  # ABX₃ fractions
    lattice_parameter: float  # Å
    surface_energy: float  # mJ/m²
    tolerance_factor: float
    thermal_expansion: float  # /K
    elastic_modulus: float  # GPa
    thickness: float  # nm

class InterfaceEnergyCalculator:
    """
    Calculator for interface energies and stability in perovskite tandems.
    """
    
    def __init__(self):
        # Shannon ionic radii database [Å] - coordination number VI for octahedral B-site
        self.ionic_radii = {
            'A_site': {
                'MA': 1.8,   # CH₃NH₃⁺ (estimated molecular radius)
                'FA': 1.9,   # CH(NH₂)₂⁺ (estimated)
                'Cs': 1.67,  # Cs⁺ (VIII coordination)
                'Rb': 1.52,  # Rb⁺ (VIII coordination)
                'K': 1.37,   # K⁺ (VIII coordination)  
                'Na': 1.18   # Na⁺ (VIII coordination)
            },
            'B_site': {
                'Pb': 1.19,  # Pb²⁺ (VI coordination)
                'Sn': 1.10,  # Sn²⁺ (VI coordination)
                'Ge': 0.87,  # Ge²⁺ (VI coordination)
                'Ti': 0.605, # Ti⁴⁺ (VI coordination)
                'Zr': 0.72,  # Zr⁴⁺ (VI coordination)
                'Ca': 1.00,  # Ca²⁺ (VI coordination)
                'Sr': 1.18   # Sr²⁺ (VI coordination)
            },
            'X_site': {
                'I': 2.20,   # I⁻ (VI coordination)
                'Br': 1.96,  # Br⁻ (VI coordination)  
                'Cl': 1.81,  # Cl⁻ (VI coordination)
                'F': 1.33,   # F⁻ (VI coordination)
                'O': 1.40    # O²⁻ (VI coordination)
            }
        }
        
        # Lattice parameters for pure perovskites [Å] (cubic phase)
        self.lattice_parameters = {
            ('MA', 'Pb', 'I'): 6.27,   # MAPbI₃
            ('MA', 'Pb', 'Br'): 5.93,  # MAPbBr₃  
            ('MA', 'Pb', 'Cl'): 5.69,  # MAPbCl₃
            ('FA', 'Pb', 'I'): 6.32,   # FAPbI₃
            ('FA', 'Pb', 'Br'): 5.99,  # FAPbBr₃
            ('Cs', 'Pb', 'I'): 6.18,   # CsPbI₃ (cubic)
            ('Cs', 'Pb', 'Br'): 5.87,  # CsPbBr₃
            ('Cs', 'Pb', 'Cl'): 5.61,  # CsPbCl₃
            ('MA', 'Sn', 'I'): 6.24,   # MASnI₃
            ('FA', 'Sn', 'I'): 6.28,   # FASnI₃
            ('Cs', 'Sn', 'I'): 6.22,   # CsSnI₃
        }
        
        # Surface energies for perovskite facets [mJ/m²]
        # Estimated from DFT calculations and analogy to oxide perovskites
        self.surface_energies = {
            ('MA', 'Pb', 'I'): 35,    # Low surface energy
            ('MA', 'Pb', 'Br'): 42,   # Higher due to Br
            ('MA', 'Pb', 'Cl'): 48,   # Highest halide
            ('FA', 'Pb', 'I'): 32,    # Slightly lower than MA
            ('Cs', 'Pb', 'I'): 38,    # Inorganic, moderate
            ('Cs', 'Pb', 'Br'): 45,   # Inorganic Br
            ('MA', 'Sn', 'I'): 28,    # Sn lower surface energy
            ('Cs', 'Sn', 'I'): 30,    # Cs-Sn combination
        }
        
        # Elastic moduli [GPa] - approximate values
        self.elastic_moduli = {
            ('MA', 'Pb', 'I'): 15,    # Soft organic-inorganic
            ('MA', 'Pb', 'Br'): 20,
            ('MA', 'Pb', 'Cl'): 25,
            ('FA', 'Pb', 'I'): 18,
            ('Cs', 'Pb', 'I'): 22,    # Inorganic harder
            ('Cs', 'Pb', 'Br'): 28,
            ('MA', 'Sn', 'I'): 12,    # Sn softer than Pb
            ('Cs', 'Sn', 'I'): 16,
        }
        
        # Thermal expansion coefficients [/K]
        self.thermal_expansion = {
            ('MA', 'Pb', 'I'): 4.2e-5,   # High due to organic cation
            ('MA', 'Pb', 'Br'): 3.8e-5,
            ('MA', 'Pb', 'Cl'): 3.5e-5,
            ('FA', 'Pb', 'I'): 4.5e-5,   # Slightly higher than MA
            ('Cs', 'Pb', 'I'): 2.8e-5,   # Lower (inorganic)
            ('Cs', 'Pb', 'Br'): 2.5e-5,
            ('MA', 'Sn', 'I'): 4.8e-5,   # Sn expansion
            ('Cs', 'Sn', 'I'): 3.2e-5,
        }
        
        # Activation energies for halide migration [eV]
        self.migration_barriers = {
            'I-Br': 0.58,   # I/Br interdiffusion barrier
            'I-Cl': 0.92,   # I/Cl interdiffusion 
            'Br-Cl': 0.74,  # Br/Cl interdiffusion
            'I-I': 0.05,    # Same halide (negligible)
            'Br-Br': 0.05,
            'Cl-Cl': 0.05,
        }
    
    def calculate_tolerance_factor(self, composition: Dict[str, float]) -> float:
        """
        Calculate Goldschmidt tolerance factor for mixed composition.
        
        t = (r_A + r_X) / [√2 × (r_B + r_X)]
        
        Stable perovskite: 0.8 < t < 1.1
        """
        
        # Calculate effective ionic radii
        r_A = self._effective_radius('A_site', composition)
        r_B = self._effective_radius('B_site', composition) 
        r_X = self._effective_radius('X_site', composition)
        
        if r_B == 0 or r_X == 0:
            return 0.0
        
        tolerance_factor = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
        
        return tolerance_factor
    
    def _effective_radius(self, site: str, composition: Dict[str, float]) -> float:
        """Calculate effective ionic radius for mixed site"""
        
        total_radius = 0.0
        total_fraction = 0.0
        
        site_prefix = site.split('_')[0]  # A, B, or X
        
        for ion, radius in self.ionic_radii[site].items():
            fraction = composition.get(f'{site_prefix}_{ion}', 0.0)
            total_radius += fraction * radius
            total_fraction += fraction
        
        return total_radius / max(total_fraction, 1e-10)
    
    def calculate_lattice_parameter(self, composition: Dict[str, float]) -> float:
        """
        Calculate lattice parameter using Vegard's law for mixed compositions.
        """
        
        total_parameter = 0.0
        total_weight = 0.0
        
        # Get composition fractions
        a_ions = {ion: composition.get(f'A_{ion}', 0) for ion in ['MA', 'FA', 'Cs', 'Rb']}
        b_ions = {ion: composition.get(f'B_{ion}', 0) for ion in ['Pb', 'Sn', 'Ge']}
        x_ions = {ion: composition.get(f'X_{ion}', 0) for ion in ['I', 'Br', 'Cl']}
        
        # Weighted average over all possible combinations
        for a_ion, a_frac in a_ions.items():
            for b_ion, b_frac in b_ions.items():
                for x_ion, x_frac in x_ions.items():
                    
                    weight = a_frac * b_frac * x_frac
                    if weight > 1e-6:  # Significant contribution
                        
                        # Get reference lattice parameter
                        key = (a_ion, b_ion, x_ion)
                        if key in self.lattice_parameters:
                            lattice_param = self.lattice_parameters[key]
                        else:
                            # Estimate using ionic radii (rough approximation)
                            r_A = self.ionic_radii['A_site'].get(a_ion, 1.5)
                            r_B = self.ionic_radii['B_site'].get(b_ion, 1.0)  
                            r_X = self.ionic_radii['X_site'].get(x_ion, 2.0)
                            lattice_param = 2 * (r_B + r_X) * np.sqrt(2)  # Cubic approximation
                        
                        total_parameter += weight * lattice_param
                        total_weight += weight
        
        return total_parameter / max(total_weight, 1e-10)
    
    def calculate_strain_energy(self, layer1: LayerProperties, layer2: LayerProperties) -> float:
        """
        Calculate elastic strain energy at interface using proper biaxial strain theory.
        
        FIXED: Use proper biaxial strain energy: U = 2μ(1+ν)/(1-ν) × ε² × t
        where μ = shear modulus, ν = Poisson ratio, ε = lattice mismatch strain, t = film thickness.
        Include both elastic strain and misfit dislocation energy above critical thickness 
        (Matthews-Blakeslee criterion).
        """
        
        # Lattice mismatch strain
        a1 = layer1.lattice_parameter
        a2 = layer2.lattice_parameter
        mismatch_strain = abs(a1 - a2) / max(a1, a2)  # ε = Δa/a
        
        if mismatch_strain == 0:
            return 0.0
        
        # Material properties - convert Young's modulus to shear modulus
        # G = E / (2(1 + ν)) where G is shear modulus, E is Young's modulus
        poisson1 = 0.25  # Default Poisson ratio for perovskites
        poisson2 = 0.25  
        shear_modulus1 = layer1.elastic_modulus / (2 * (1 + poisson1))  # Pa
        shear_modulus2 = layer2.elastic_modulus / (2 * (1 + poisson2))  # Pa
        
        # Use properties of the strained layer (typically the thinner one)
        if layer1.thickness <= layer2.thickness:
            # Layer1 is strained to match layer2
            mu = shear_modulus1
            nu = poisson1
            t = layer1.thickness
            strained_layer = layer1
        else:
            # Layer2 is strained to match layer1
            mu = shear_modulus2
            nu = poisson2
            t = layer2.thickness
            strained_layer = layer2
        
        # Critical thickness for misfit dislocation formation (Matthews-Blakeslee)
        # h_c = (b/8π) × (1-ν cos²α)/(1+ν) × ln(h_c/b) × (1/ε)
        # Simplified approximation: h_c ≈ b/(8πε) where b is Burgers vector
        burgers_vector = a1 * 1e-10 / np.sqrt(2)  # Approximate: b ≈ a/√2 for <110> dislocations
        if mismatch_strain > 1e-6:
            critical_thickness = burgers_vector / (8 * np.pi * mismatch_strain)
        else:
            critical_thickness = 1e-3  # Very thick for small strain
        
        if t <= critical_thickness:
            # Below critical thickness: coherent strain (no dislocations)
            # Biaxial strain energy density: U = 2μ(1+ν)/(1-ν) × ε²
            biaxial_factor = 2 * mu * (1 + nu) / (1 - nu)  # Pa
            strain_energy_density = biaxial_factor * mismatch_strain**2  # J/m³
            
            # Total strain energy per unit area
            strain_energy_per_area = strain_energy_density * t  # J/m²
            
        else:
            # Above critical thickness: partial relaxation via dislocations
            # Residual strain after dislocation formation
            residual_strain = mismatch_strain * (critical_thickness / t)**0.5  # Empirical scaling
            
            # Elastic energy from residual strain
            biaxial_factor = 2 * mu * (1 + nu) / (1 - nu)
            elastic_energy_density = biaxial_factor * residual_strain**2
            elastic_energy = elastic_energy_density * t
            
            # Dislocation formation energy (line energy per unit length)
            # E_dislocation ≈ μb²/4π × ln(h/b) per unit length
            dislocation_spacing = burgers_vector / mismatch_strain  # Approximate spacing
            dislocation_line_energy = mu * burgers_vector**2 / (4 * np.pi) * np.log(t / burgers_vector)
            dislocation_energy_per_area = dislocation_line_energy / dislocation_spacing
            
            # Total energy is elastic + dislocation
            strain_energy_per_area = elastic_energy + dislocation_energy_per_area
        
        # Convert to more convenient units [meV/atom]
        # 1 J/m² ≈ 6.24e12 meV/Ų ≈ 62.4 meV/Ų for typical atomic densities
        # Approximate atomic density: ~1e15 atoms/cm² for perovskites
        atoms_per_m2 = 1e19  # atoms/m² (rough estimate)
        if atoms_per_m2 > 0:
            strain_energy_meV_per_atom = strain_energy_per_area / (Q * 1e-3) * 1000 / atoms_per_m2
        else:
            strain_energy_meV_per_atom = 0.0
        
        return strain_energy_meV_per_atom
    
    def calculate_adhesion_energy(self, layer1: LayerProperties, layer2: LayerProperties) -> float:
        """
        Calculate interface adhesion energy using Dupré equation.
        
        W_adhesion = γ₁ + γ₂ - γ₁₂
        
        where γ₁₂ is interface energy (estimated from geometric mean)
        """
        
        γ1 = self._get_surface_energy(layer1.composition)
        γ2 = self._get_surface_energy(layer2.composition)
        
        # Estimate interface energy as geometric mean minus attractive term
        γ12 = np.sqrt(γ1 * γ2) * 0.7  # 30% reduction due to interfacial bonding
        
        adhesion_energy = γ1 + γ2 - γ12  # mJ/m²
        
        return max(adhesion_energy, 0)  # Positive adhesion
    
    def _get_surface_energy(self, composition: Dict[str, float]) -> float:
        """Get surface energy for mixed composition"""
        
        total_energy = 0.0
        total_weight = 0.0
        
        # Composition fractions 
        a_ions = {ion: composition.get(f'A_{ion}', 0) for ion in ['MA', 'FA', 'Cs']}
        b_ions = {ion: composition.get(f'B_{ion}', 0) for ion in ['Pb', 'Sn']}
        x_ions = {ion: composition.get(f'X_{ion}', 0) for ion in ['I', 'Br', 'Cl']}
        
        for a_ion, a_frac in a_ions.items():
            for b_ion, b_frac in b_ions.items():
                for x_ion, x_frac in x_ions.items():
                    
                    weight = a_frac * b_frac * x_frac
                    if weight > 1e-6:
                        
                        key = (a_ion, b_ion, x_ion)
                        surface_energy = self.surface_energies.get(key, 40)  # Default value
                        
                        total_energy += weight * surface_energy
                        total_weight += weight
        
        return total_energy / max(total_weight, 1e-10)
    
    def calculate_interdiffusion_barrier(self, layer1: LayerProperties, layer2: LayerProperties) -> float:
        """
        Calculate activation energy for halide interdiffusion between layers.
        """
        
        # Get dominant halides in each layer
        x_ions = ['I', 'Br', 'Cl']
        
        halide1 = max(x_ions, key=lambda x: layer1.composition.get(f'X_{x}', 0))
        halide2 = max(x_ions, key=lambda x: layer2.composition.get(f'X_{x}', 0))
        
        # Get barrier for this halide pair
        barrier_key = f"{halide1}-{halide2}" if halide1 <= halide2 else f"{halide2}-{halide1}"
        
        barrier = self.migration_barriers.get(barrier_key, 0.5)  # Default moderate barrier
        
        return barrier
    
    def assess_interface_stability(self, layer1: LayerProperties, layer2: LayerProperties, 
                                  temperature: float = 298.15) -> InterfaceStability:
        """
        Comprehensive interface stability analysis.
        
        Args:
            layer1, layer2: Adjacent layers
            temperature: Operating temperature [K]
            
        Returns:
            InterfaceStability with all calculated properties
        """
        
        # Calculate individual contributions
        tolerance1 = self.calculate_tolerance_factor(layer1.composition)
        tolerance2 = self.calculate_tolerance_factor(layer2.composition)
        tolerance_mismatch = abs(tolerance1 - tolerance2)
        
        strain_energy = self.calculate_strain_energy(layer1, layer2)
        adhesion_energy = self.calculate_adhesion_energy(layer1, layer2)
        interdiffusion_barrier = self.calculate_interdiffusion_barrier(layer1, layer2)
        
        # Risk factors assessment
        risk_factors = []
        
        if tolerance_mismatch > 0.15:
            risk_factors.append("Large tolerance factor mismatch")
        
        if strain_energy > 50:  # meV/atom
            risk_factors.append("High strain energy")
        
        if adhesion_energy < 10:  # mJ/m²
            risk_factors.append("Weak adhesion")
            
        if interdiffusion_barrier < 0.3:  # eV
            risk_factors.append("Low interdiffusion barrier")
        
        # Thermal expansion mismatch
        te1 = self._get_thermal_expansion(layer1.composition)
        te2 = self._get_thermal_expansion(layer2.composition)
        te_mismatch = abs(te1 - te2)
        
        if te_mismatch > 1e-5:  # /K
            risk_factors.append("Thermal expansion mismatch")
        
        # Overall thermodynamic stability assessment
        # ΔG includes strain, interface, and mixing contributions
        delta_g = (strain_energy * 0.001 -  # Convert meV → kJ/mol (approximate)
                  adhesion_energy * 0.001 +  # Interface contribution
                  8.314 * temperature * 0.1 * len(risk_factors))  # Entropy penalty for risks
        
        # Stability score (0-10)
        score = 10.0
        score -= min(tolerance_mismatch * 20, 3)  # Tolerance penalty
        score -= min(strain_energy / 10, 2)       # Strain penalty  
        score -= min(len(risk_factors) * 1.5, 3)  # Risk penalty
        score += min(interdiffusion_barrier * 2, 2)  # Barrier bonus
        
        stability_score = max(0, min(score, 10))
        
        # Overall stability decision
        is_stable = (delta_g < 10 and  # kJ/mol
                    len(risk_factors) <= 2 and
                    strain_energy < 100)  # meV/atom
        
        return InterfaceStability(
            is_stable=is_stable,
            delta_g=delta_g,
            strain_energy=strain_energy,
            tolerance_mismatch=tolerance_mismatch,
            adhesion_energy=adhesion_energy,
            interdiffusion_barrier=interdiffusion_barrier,
            risk_factors=risk_factors,
            stability_score=stability_score
        )
    
    def _get_thermal_expansion(self, composition: Dict[str, float]) -> float:
        """Get thermal expansion coefficient for mixed composition"""
        
        total_expansion = 0.0
        total_weight = 0.0
        
        # Similar to surface energy calculation
        a_ions = {ion: composition.get(f'A_{ion}', 0) for ion in ['MA', 'FA', 'Cs']}
        b_ions = {ion: composition.get(f'B_{ion}', 0) for ion in ['Pb', 'Sn']}
        x_ions = {ion: composition.get(f'X_{ion}', 0) for ion in ['I', 'Br', 'Cl']}
        
        for a_ion, a_frac in a_ions.items():
            for b_ion, b_frac in b_ions.items():
                for x_ion, x_frac in x_ions.items():
                    
                    weight = a_frac * b_frac * x_frac
                    if weight > 1e-6:
                        
                        key = (a_ion, b_ion, x_ion)
                        expansion = self.thermal_expansion.get(key, 3e-5)
                        
                        total_expansion += weight * expansion
                        total_weight += weight
        
        return total_expansion / max(total_weight, 1e-10)
    
    def optimize_interface_stack(self, target_bandgaps: List[float], 
                                max_layers: int = 6) -> Tuple[List[Dict], float]:
        """
        Optimize layer compositions for stable interface stack.
        
        Args:
            target_bandgaps: Desired bandgaps for each layer [eV]
            max_layers: Maximum number of layers to consider
            
        Returns:
            (optimal_compositions, total_stability_score)
        """
        
        from .ml_bandgap import ML_BANDGAP_PREDICTOR
        
        if not ML_BANDGAP_PREDICTOR.is_fitted:
            ML_BANDGAP_PREDICTOR.fit()
        
        # Generate candidate compositions for each bandgap
        candidates_per_layer = []
        
        for target_bg in target_bandgaps[:max_layers]:
            layer_candidates = []
            
            # Sample composition space
            n_samples = 50
            np.random.seed(42)
            
            for _ in range(n_samples):
                # Generate random composition
                a_fracs = np.random.dirichlet([2, 2, 1, 0.2])  # MA, FA, Cs, Rb
                b_fracs = np.random.dirichlet([3, 1, 0.2])    # Pb, Sn, Ge
                x_fracs = np.random.dirichlet([2, 1.5, 0.5])  # I, Br, Cl
                
                composition = {
                    'A_MA': a_fracs[0], 'A_FA': a_fracs[1], 'A_Cs': a_fracs[2], 'A_Rb': a_fracs[3],
                    'B_Pb': b_fracs[0], 'B_Sn': b_fracs[1], 'B_Ge': b_fracs[2],
                    'X_I': x_fracs[0], 'X_Br': x_fracs[1], 'X_Cl': x_fracs[2]
                }
                
                # Predict bandgap
                pred_bg, uncertainty = ML_BANDGAP_PREDICTOR.predict(composition)
                
                # Check if close to target (within ±0.1 eV)
                if abs(pred_bg - target_bg) < 0.1:
                    layer_candidates.append({
                        'composition': composition,
                        'bandgap': pred_bg,
                        'uncertainty': uncertainty
                    })
            
            candidates_per_layer.append(layer_candidates)
        
        # Find optimal combination with best interface stability
        best_combination = None
        best_score = -1e6
        
        # Try different combinations (limit to prevent combinatorial explosion)
        max_combinations = 1000
        combinations_tried = 0
        
        def generate_combinations(layer_idx, current_combo, current_score):
            nonlocal best_combination, best_score, combinations_tried
            
            if combinations_tried > max_combinations:
                return
                
            if layer_idx >= len(candidates_per_layer):
                if current_score > best_score:
                    best_score = current_score
                    best_combination = current_combo.copy()
                combinations_tried += 1
                return
            
            # Try candidates for current layer
            for candidate in candidates_per_layer[layer_idx][:10]:  # Limit candidates
                
                # Create layer properties
                layer_props = LayerProperties(
                    composition=candidate['composition'],
                    lattice_parameter=self.calculate_lattice_parameter(candidate['composition']),
                    surface_energy=self._get_surface_energy(candidate['composition']),
                    tolerance_factor=self.calculate_tolerance_factor(candidate['composition']),
                    thermal_expansion=self._get_thermal_expansion(candidate['composition']),
                    elastic_modulus=self._get_elastic_modulus(candidate['composition']),
                    thickness=100  # nm, default
                )
                
                # Calculate interface score with previous layer
                interface_score = 10  # Perfect score for first layer
                if current_combo:
                    prev_layer = current_combo[-1]
                    stability = self.assess_interface_stability(prev_layer, layer_props)
                    interface_score = stability.stability_score
                
                # Combined score includes bandgap accuracy and interface stability
                bg_accuracy_score = 10 * np.exp(-0.5 * (candidate['bandgap'] - target_bandgaps[layer_idx])**2 / 0.01)
                total_score = current_score + 0.5 * bg_accuracy_score + 0.5 * interface_score
                
                current_combo.append(layer_props)
                generate_combinations(layer_idx + 1, current_combo, total_score)
                current_combo.pop()
        
        # Start recursive search
        generate_combinations(0, [], 0)
        
        if best_combination is None:
            return [], 0.0
        
        # Extract compositions from best combination
        optimal_compositions = [layer.composition for layer in best_combination]
        
        return optimal_compositions, best_score
    
    def _get_elastic_modulus(self, composition: Dict[str, float]) -> float:
        """Get elastic modulus for mixed composition"""
        
        total_modulus = 0.0
        total_weight = 0.0
        
        a_ions = {ion: composition.get(f'A_{ion}', 0) for ion in ['MA', 'FA', 'Cs']}
        b_ions = {ion: composition.get(f'B_{ion}', 0) for ion in ['Pb', 'Sn']}
        x_ions = {ion: composition.get(f'X_{ion}', 0) for ion in ['I', 'Br', 'Cl']}
        
        for a_ion, a_frac in a_ions.items():
            for b_ion, b_frac in b_ions.items():
                for x_ion, x_frac in x_ions.items():
                    
                    weight = a_frac * b_frac * x_frac
                    if weight > 1e-6:
                        
                        key = (a_ion, b_ion, x_ion)
                        modulus = self.elastic_moduli.get(key, 20)  # Default GPa
                        
                        total_modulus += weight * modulus
                        total_weight += weight
        
        return total_modulus / max(total_weight, 1e-10)

# Global instance
INTERFACE_CALCULATOR = InterfaceEnergyCalculator()

if __name__ == "__main__":
    print("Interface Energy Calculator Test")
    print("=" * 40)
    
    calc = InterfaceEnergyCalculator()
    
    # Test layers
    layer1_comp = {'A_MA': 1.0, 'A_FA': 0.0, 'A_Cs': 0.0, 'A_Rb': 0.0,
                   'B_Pb': 1.0, 'B_Sn': 0.0, 'B_Ge': 0.0,
                   'X_I': 0.8, 'X_Br': 0.2, 'X_Cl': 0.0}
    
    layer2_comp = {'A_MA': 0.5, 'A_FA': 0.5, 'A_Cs': 0.0, 'A_Rb': 0.0,
                   'B_Pb': 1.0, 'B_Sn': 0.0, 'B_Ge': 0.0,
                   'X_I': 0.0, 'X_Br': 1.0, 'X_Cl': 0.0}
    
    layer1 = LayerProperties(
        composition=layer1_comp,
        lattice_parameter=calc.calculate_lattice_parameter(layer1_comp),
        surface_energy=calc._get_surface_energy(layer1_comp),
        tolerance_factor=calc.calculate_tolerance_factor(layer1_comp),
        thermal_expansion=calc._get_thermal_expansion(layer1_comp),
        elastic_modulus=calc._get_elastic_modulus(layer1_comp),
        thickness=200
    )
    
    layer2 = LayerProperties(
        composition=layer2_comp,
        lattice_parameter=calc.calculate_lattice_parameter(layer2_comp),
        surface_energy=calc._get_surface_energy(layer2_comp),
        tolerance_factor=calc.calculate_tolerance_factor(layer2_comp),
        thermal_expansion=calc._get_thermal_expansion(layer2_comp),
        elastic_modulus=calc._get_elastic_modulus(layer2_comp),
        thickness=200
    )
    
    # Test interface stability
    stability = calc.assess_interface_stability(layer1, layer2)
    
    print(f"Interface stability: {stability.is_stable}")
    print(f"ΔG: {stability.delta_g:.2f} kJ/mol")
    print(f"Strain energy: {stability.strain_energy:.1f} meV/atom")
    print(f"Stability score: {stability.stability_score:.1f}/10")
    print(f"Risk factors: {stability.risk_factors}")