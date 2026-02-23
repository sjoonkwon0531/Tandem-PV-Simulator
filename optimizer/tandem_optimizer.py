"""
Tandem Optimizer for Multi-Junction Photovoltaic Cells

Integrates all analysis engines to optimize tandem solar cell designs
for maximum PCE, minimum LCOE, or optimal PCE/cost ratio.

References:
- Storn, R. & Price, K. "Differential Evolution - a simple and efficient 
  heuristic for global optimization" J. Glob. Optim. 11, 341 (1997)
- Lau, C.F.J. et al. "Perovskite-silicon tandem solar cells: Progress and challenges" 
  Adv. Energy Mater. 12, 2101662 (2022)
- Essig, S. et al. "Raising the one-sun conversion efficiency of III-V/Si solar cells 
  to 32.8% for two junctions and 35.9% for three junctions" Nat. Energy 2, 17144 (2017)
"""

import warnings
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from scipy.optimize import differential_evolution, minimize
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MATERIAL_DB, TRACK_A_MATERIALS, TRACK_B_MATERIALS
from engines.optical_tmm import TransferMatrixCalculator
from engines.band_alignment import DetailedBalanceCalculator, BandgapOptimizer
from engines.interface_loss import InterfaceLossCalculator, TunnelJunctionCalculator
from engines.thermal_model import ThermalExpansionModel
from engines.stability import StabilityEngine
from engines.economics import EconomicsEngine

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TandemOptimizer:
    """
    Multi-objective optimizer for tandem photovoltaic cells.
    
    Integrates all analysis engines to find optimal designs considering:
    - Power conversion efficiency (optical + electrical)
    - Manufacturing cost and LCOE
    - Thermal expansion compatibility
    - Long-term stability
    - Manufacturability constraints
    """
    
    def __init__(self,
                 track: str = 'A',
                 max_junctions: int = 4,
                 optimization_target: str = 'pce'):
        """
        Initialize tandem optimizer.
        
        Args:
            track: 'A' for established materials, 'B' for perovskite tuning
            max_junctions: Maximum number of junctions to consider
            optimization_target: 'pce', 'lcoe', or 'pce_per_cost'
        """
        self.track = track
        self.max_junctions = max_junctions
        self.optimization_target = optimization_target
        
        # Initialize all engines
        self.optical_engine = TransferMatrixCalculator()
        self.band_engine = DetailedBalanceCalculator()
        self.bandgap_optimizer = BandgapOptimizer()
        self.interface_engine = InterfaceLossCalculator()
        self.tunnel_engine = TunnelJunctionCalculator()
        self.thermal_engine = ThermalExpansionModel()
        self.stability_engine = StabilityEngine()
        self.economics_engine = EconomicsEngine()
        
        # Material databases
        self.material_db = MATERIAL_DB
        self.available_materials = TRACK_A_MATERIALS if track == 'A' else TRACK_B_MATERIALS
        
        # Optimization constraints
        self.constraints = {
            'min_thickness_nm': 50,       # Minimum layer thickness
            'max_thickness_nm': 2000,     # Maximum layer thickness  
            'max_cte_mismatch': 5e-6,     # Maximum CTE mismatch (K⁻¹)
            'min_stability_score': 0.4,   # Minimum stability score
            'current_matching_tolerance': 0.05,  # ±5% current matching
            'min_bandgap_separation': 0.2,  # Minimum ΔEg between junctions
        }
        
        # Weighting factors for multi-objective optimization
        self.weights = {
            'pce': 0.4,
            'cost': 0.2,
            'stability': 0.2,
            'thermal': 0.1,
            'manufacturability': 0.1
        }
    
    def generate_layer_stack(self, 
                           design_vector: np.ndarray,
                           n_junctions: int) -> List[Tuple[str, float]]:
        """
        Convert optimization vector to layer stack.
        
        Args:
            design_vector: Optimization parameters
            n_junctions: Number of junctions
            
        Returns:
            List of (material_name, thickness_nm) tuples
        """
        stack = []
        vector_idx = 0
        
        # Substrate (always present)
        stack.append(('glass', 3000000))  # 3mm glass
        
        # Front contact
        stack.append(('ITO', 100))
        
        for junction in range(n_junctions):
            # HTL
            if self.track == 'A':
                htl_material = 'PEDOT'
            else:
                htl_material = 'PEDOT'  # Could be optimized
            
            htl_thickness = self._decode_thickness(design_vector[vector_idx])
            stack.append((htl_material, htl_thickness))
            vector_idx += 1
            
            # Active layer
            if self.track == 'A':
                # Select from predefined materials
                material_idx = int(design_vector[vector_idx] * len(self.available_materials))
                material_idx = min(material_idx, len(self.available_materials) - 1)
                active_material = self.available_materials[material_idx]
            else:
                # Tune perovskite composition
                br_ratio = design_vector[vector_idx]
                active_material = f'perovskite_MAPb(Br{br_ratio:.2f}I{1-br_ratio:.2f})3'
            
            vector_idx += 1
            
            active_thickness = self._decode_thickness(design_vector[vector_idx])
            stack.append((active_material, active_thickness))
            vector_idx += 1
            
            # ETL
            etl_material = 'C60'
            etl_thickness = self._decode_thickness(design_vector[vector_idx])
            stack.append((etl_material, etl_thickness))
            vector_idx += 1
            
            # Tunnel junction (if not the last junction)
            if junction < n_junctions - 1:
                stack.append(('ITO', 50))  # Recombination layer
        
        # Back contact
        stack.append(('Au', 80))
        
        return stack
    
    def _decode_thickness(self, encoded_value: float) -> float:
        """Decode normalized thickness value to nanometers."""
        min_thick = self.constraints['min_thickness_nm']
        max_thick = self.constraints['max_thickness_nm']
        return min_thick + encoded_value * (max_thick - min_thick)
    
    def evaluate_design(self, 
                       design_vector: np.ndarray,
                       n_junctions: int,
                       environment: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a tandem design.
        
        Args:
            design_vector: Optimization parameters
            n_junctions: Number of junctions
            environment: Environmental conditions for stability analysis
            
        Returns:
            Dict with all performance metrics
        """
        if environment is None:
            environment = {
                'temperature': 25.0,
                'humidity': 0.6,
                'illumination': 1.0,
                'encapsulation_factor': 0.1
            }
        
        # Generate layer stack
        stack = self.generate_layer_stack(design_vector, n_junctions)
        
        # Extract active layers and bandgaps
        active_layers = []
        bandgaps = []
        
        for material_name, thickness in stack:
            if ('perovskite' in material_name.lower() or 
                material_name in self.available_materials):
                
                active_layers.append((material_name, thickness))
                
                # Get or estimate bandgap
                if material_name in self.material_db:
                    bandgap = self.material_db[material_name].get('bandgap', 1.5)
                else:
                    # For custom perovskites, estimate from composition
                    if 'Br' in material_name and 'I' in material_name:
                        # Extract Br ratio and interpolate bandgap
                        import re
                        br_match = re.search(r'Br(\d+\.?\d*)', material_name)
                        if br_match:
                            br_ratio = float(br_match.group(1))
                            # Linear interpolation: MAPbI3 = 1.55 eV, MAPbBr3 = 2.25 eV
                            bandgap = 1.55 + br_ratio * (2.25 - 1.55)
                        else:
                            bandgap = 1.5
                    else:
                        bandgap = 1.5
                
                bandgaps.append(bandgap)
        
        try:
            # 1. Optical analysis
            optical_result = self.optical_engine.calculate_stack_absorption(stack)
            
            # 2. Band alignment and current matching
            if len(bandgaps) > 0:
                current_match = self.band_engine.calculate_current_matching(
                    bandgaps, subcell_thicknesses=[layer[1] for layer in active_layers]
                )
                
                # Detailed balance limit
                sq_result = self.bandgap_optimizer.optimize_bandgaps(len(bandgaps))
                theoretical_pce = sq_result['optimal_pce']
            else:
                current_match = {'jsc_matched': 0.0, 'matching_loss': 1.0}
                theoretical_pce = 0.0
            
            # 3. Interface losses
            interface_result = self.interface_engine.calculate_total_interface_loss(stack)
            
            # 4. Thermal analysis
            thermal_result = self.thermal_engine.evaluate_stack_thermal_reliability(stack)
            
            # 5. Stability analysis
            stability_result = self.stability_engine.evaluate_stack_stability(
                stack, environment, initial_pce=theoretical_pce * 0.8  # Account for practical losses
            )
            
            # 6. Economic analysis
            manufacturing_cost = self.economics_engine.calculate_stack_manufacturing_cost(stack)
            
            # Calculate practical PCE (accounting for all losses)
            practical_pce = (theoretical_pce * 
                           current_match.get('current_matching_efficiency', 0.8) *
                           (1 - interface_result['total_loss_fraction']) *
                           optical_result.get('absorption_efficiency', 0.8))
            
            # LCOE calculation
            lcoe_result = self.economics_engine.calculate_lcoe(
                manufacturing_cost['cost_per_m2'],
                practical_pce
            )
            
            # Combine results
            evaluation = {
                'layer_stack': stack,
                'n_junctions': n_junctions,
                'bandgaps': bandgaps,
                'theoretical_pce': theoretical_pce,
                'practical_pce': practical_pce,
                'optical_analysis': optical_result,
                'current_matching': current_match,
                'interface_analysis': interface_result,
                'thermal_analysis': thermal_result,
                'stability_analysis': stability_result,
                'manufacturing_cost': manufacturing_cost,
                'lcoe_analysis': lcoe_result,
                'constraint_violations': self._check_constraints(
                    stack, thermal_result, stability_result, current_match
                )
            }
            
            return evaluation
            
        except Exception as e:
            # Return penalty values for failed designs
            return {
                'layer_stack': stack,
                'n_junctions': n_junctions,
                'practical_pce': 0.0,
                'lcoe_analysis': {'lcoe_cents_per_kwh': 1000.0},
                'manufacturing_cost': {'cost_per_m2': 10000.0},
                'constraint_violations': ['evaluation_failed'],
                'error': str(e)
            }
    
    def _check_constraints(self,
                          stack: List[Tuple[str, float]],
                          thermal_result: Dict,
                          stability_result: Dict,
                          current_match: Dict) -> List[str]:
        """Check constraint violations."""
        violations = []
        
        # Thermal constraint
        if thermal_result['stress_analysis']['mismatch_score'] > self.constraints['max_cte_mismatch']:
            violations.append('cte_mismatch_too_high')
        
        # Stability constraint
        if stability_result['stability_score'] < self.constraints['min_stability_score']:
            violations.append('stability_too_low')
        
        # Current matching constraint
        matching_error = abs(1.0 - current_match.get('current_matching_efficiency', 0.5))
        if matching_error > self.constraints['current_matching_tolerance']:
            violations.append('current_matching_poor')
        
        # Layer thickness constraints
        for material, thickness in stack:
            if (thickness < self.constraints['min_thickness_nm'] or 
                thickness > self.constraints['max_thickness_nm']):
                if thickness > 1000000:  # Skip substrate
                    continue
                violations.append(f'thickness_violation_{material}')
        
        return violations
    
    def objective_function(self, 
                          design_vector: np.ndarray,
                          n_junctions: int) -> float:
        """
        Multi-objective optimization function.
        
        Args:
            design_vector: Design parameters
            n_junctions: Number of junctions
            
        Returns:
            Objective value (lower is better for minimization)
        """
        evaluation = self.evaluate_design(design_vector, n_junctions)
        
        # Apply penalty for constraint violations
        penalty = len(evaluation['constraint_violations']) * 100.0
        
        if self.optimization_target == 'pce':
            # Maximize PCE (minimize negative PCE)
            objective = -evaluation['practical_pce'] + penalty
            
        elif self.optimization_target == 'lcoe':
            # Minimize LCOE
            objective = evaluation['lcoe_analysis']['lcoe_cents_per_kwh'] + penalty
            
        elif self.optimization_target == 'pce_per_cost':
            # Maximize PCE per unit cost
            pce = evaluation['practical_pce']
            cost = evaluation['manufacturing_cost']['cost_per_m2']
            objective = -pce / max(cost, 1.0) + penalty
            
        else:
            # Multi-objective weighted sum
            pce_score = evaluation['practical_pce'] / 40.0  # Normalize to ~40% max
            cost_score = 1.0 - evaluation['lcoe_analysis']['lcoe_cents_per_kwh'] / 100.0
            stability_score = evaluation['stability_analysis']['stability_score']
            thermal_score = evaluation['thermal_analysis']['thermal_score']
            manuf_score = evaluation['manufacturing_cost']['manufacturability_score'] / 100.0
            
            weighted_score = (self.weights['pce'] * pce_score +
                            self.weights['cost'] * max(0, cost_score) +
                            self.weights['stability'] * stability_score +
                            self.weights['thermal'] * thermal_score +
                            self.weights['manufacturability'] * manuf_score)
            
            objective = -weighted_score + penalty
        
        return objective
    
    def optimize_design(self,
                       n_junctions: int,
                       population_size: int = 50,
                       max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize tandem design for specified number of junctions.
        
        Args:
            n_junctions: Number of junctions to optimize
            population_size: DE population size
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dict with optimization results
        """
        # Define bounds for design variables
        if self.track == 'A':
            # Track A: [htl_thickness, material_selection, active_thickness, etl_thickness] per junction
            n_vars = n_junctions * 4
            bounds = []
            
            for junction in range(n_junctions):
                bounds.append((0.0, 1.0))  # HTL thickness (normalized)
                bounds.append((0.0, 0.99))  # Material selection (index)
                bounds.append((0.0, 1.0))  # Active thickness (normalized)
                bounds.append((0.0, 1.0))  # ETL thickness (normalized)
        
        else:
            # Track B: [htl_thickness, br_ratio, active_thickness, etl_thickness] per junction
            n_vars = n_junctions * 4
            bounds = []
            
            for junction in range(n_junctions):
                bounds.append((0.0, 1.0))  # HTL thickness (normalized)
                bounds.append((0.0, 1.0))  # Br ratio
                bounds.append((0.0, 1.0))  # Active thickness (normalized)
                bounds.append((0.0, 1.0))  # ETL thickness (normalized)
        
        # Run differential evolution optimization
        print(f"Optimizing {n_junctions}-junction design (Track {self.track})...")
        
        result = differential_evolution(
            func=lambda x: self.objective_function(x, n_junctions),
            bounds=bounds,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=42,
            disp=False,
            workers=1  # Single-threaded for stability
        )
        
        # Evaluate best design
        best_design = self.evaluate_design(result.x, n_junctions)
        
        return {
            'optimization_result': result,
            'best_design': best_design,
            'optimization_target': self.optimization_target,
            'n_junctions': n_junctions,
            'track': self.track,
            'success': result.success,
            'n_evaluations': result.nfev,
            'final_objective': result.fun
        }
    
    def compare_junction_numbers(self,
                               junction_range: List[int] = None,
                               quick_optimization: bool = True) -> Dict[str, Any]:
        """
        Compare optimal designs for different numbers of junctions.
        
        Args:
            junction_range: List of junction numbers to compare
            quick_optimization: Use reduced parameters for faster results
            
        Returns:
            Dict with comparison results
        """
        if junction_range is None:
            junction_range = list(range(1, self.max_junctions + 1))
        
        # Optimization parameters
        if quick_optimization:
            pop_size, max_iter = 20, 30
        else:
            pop_size, max_iter = 50, 100
        
        results = {}
        
        for n_junctions in junction_range:
            print(f"Optimizing {n_junctions}-junction design...")
            
            try:
                result = self.optimize_design(n_junctions, pop_size, max_iter)
                results[n_junctions] = result
                
                print(f"  PCE: {result['best_design']['practical_pce']:.2f}%")
                print(f"  LCOE: {result['best_design']['lcoe_analysis']['lcoe_cents_per_kwh']:.1f} ¢/kWh")
                print(f"  Cost: ${result['best_design']['manufacturing_cost']['cost_per_m2']:.1f}/m²")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                continue
        
        # Find sweet spot
        if results:
            if self.optimization_target == 'pce':
                best_n = max(results.keys(), 
                           key=lambda k: results[k]['best_design']['practical_pce'])
            elif self.optimization_target == 'lcoe':
                best_n = min(results.keys(),
                           key=lambda k: results[k]['best_design']['lcoe_analysis']['lcoe_cents_per_kwh'])
            else:
                # Multi-objective: best objective value
                best_n = min(results.keys(),
                           key=lambda k: results[k]['final_objective'])
            
            return {
                'results_by_junctions': results,
                'sweet_spot_junctions': best_n,
                'sweet_spot_design': results[best_n],
                'optimization_target': self.optimization_target,
                'track': self.track
            }
        else:
            return {'error': 'No successful optimizations'}


def demo_tandem_optimizer():
    """Demonstrate tandem optimizer capabilities."""
    
    print("=== Tandem Optimizer Demo ===")
    
    # Track A optimization (established materials)
    print("\n--- Track A: Established Materials ---")
    optimizer_a = TandemOptimizer(track='A', optimization_target='pce')
    
    # Quick comparison of 1-3 junctions
    comparison_a = optimizer_a.compare_junction_numbers([1, 2], quick_optimization=True)
    
    if 'sweet_spot_junctions' in comparison_a:
        best_a = comparison_a['sweet_spot_design']['best_design']
        print(f"Best Track A Design ({comparison_a['sweet_spot_junctions']} junctions):")
        print(f"  PCE: {best_a['practical_pce']:.2f}%")
        print(f"  LCOE: {best_a['lcoe_analysis']['lcoe_cents_per_kwh']:.1f} ¢/kWh")
        print(f"  Thermal Score: {best_a['thermal_analysis']['thermal_score']:.3f}")
        print(f"  Stability Score: {best_a['stability_analysis']['stability_score']:.3f}")
    
    # Track B optimization (perovskite tuning)
    print("\n--- Track B: Perovskite Composition Tuning ---")
    optimizer_b = TandemOptimizer(track='B', optimization_target='lcoe')
    
    # Single 2-junction optimization
    result_b = optimizer_b.optimize_design(2, population_size=20, max_iterations=20)
    
    if result_b['success']:
        best_b = result_b['best_design']
        print(f"Best Track B Design (2 junctions):")
        print(f"  PCE: {best_b['practical_pce']:.2f}%")
        print(f"  LCOE: {best_b['lcoe_analysis']['lcoe_cents_per_kwh']:.1f} ¢/kWh")
        print(f"  Manufacturing Cost: ${best_b['manufacturing_cost']['cost_per_m2']:.1f}/m²")
        
        # Show layer stack
        print("  Layer Stack:")
        for i, (material, thickness) in enumerate(best_b['layer_stack']):
            if thickness > 100000:  # Skip substrate details
                print(f"    {i+1}. {material} ({thickness/1e6:.1f} mm)")
            else:
                print(f"    {i+1}. {material} ({thickness:.0f} nm)")
    
    return {'track_a': comparison_a, 'track_b': result_b}


if __name__ == "__main__":
    demo_tandem_optimizer()