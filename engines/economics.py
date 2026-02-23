"""
Economics Engine for Tandem Photovoltaic Cells

Calculates manufacturing costs, LCOE, and manufacturability metrics
for multi-junction solar cell stacks.

References:
- Woodhouse, M. et al. "Crystalline Silicon Photovoltaic Module Manufacturing Costs" 
  NREL Technical Report (2019)
- Horowitz, K.A.W. et al. "A bottom-up cost analysis of a high concentration 
  PV module" AIP Conference Proceedings 1556, 22 (2013) 
- Vartiainen, E. et al. "True Cost of Solar Hydrogen" Sol. RRL 6, 2100487 (2022)
- Fu, R. et al. "U.S. Solar Photovoltaic System and Energy Storage Cost Benchmarks" 
  NREL Technical Report (2021)
"""

import warnings
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MATERIAL_DB

warnings.filterwarnings("ignore", category=RuntimeWarning)


class EconomicsEngine:
    """
    Economic analysis engine for tandem solar cell manufacturing.
    
    Includes:
    - Layer-by-layer manufacturing cost modeling
    - Deposition method cost comparison
    - LCOE (Levelized Cost of Energy) calculation
    - Manufacturability assessment
    - Sweet spot analysis for junction number optimization
    """
    
    def __init__(self):
        """Initialize economics engine with cost parameters."""
        self.material_db = MATERIAL_DB
        
        # Deposition method costs ($/m²)
        self.deposition_costs = {
            'spin_coat': 2.0,      # Solution processing
            'slot_die': 5.0,       # Industrial solution coating
            'inkjet': 8.0,         # Digital printing
            'thermal_evap': 15.0,  # Thermal evaporation
            'sputtering': 20.0,    # Sputter deposition
            'cvd': 35.0,           # Chemical vapor deposition
            'mocvd': 50.0,         # Metal-organic CVD
            'mbe': 100.0,          # Molecular beam epitaxy
            'alds': 80.0           # Atomic layer deposition
        }
        
        # Material cost per gram ($/g)
        self.material_unit_costs = {
            # Substrates
            'glass': 0.01,
            'pet': 0.05,
            'silicon': 0.20,
            
            # Transparent conductors
            'ito': 2.50,
            'fto': 1.80,
            'ato': 2.00,
            'graphene': 50.0,
            
            # Perovskites
            'perovskite': 5.0,
            'mapbi3': 8.0,
            'fapbi3': 12.0,
            'cspbi3': 15.0,
            
            # Organic materials
            'pedot': 25.0,
            'ptaa': 80.0,
            'spiro': 120.0,
            'p3ht': 35.0,
            'pcbm': 45.0,
            
            # Inorganic materials
            'tio2': 3.0,
            'sno2': 4.0,
            'zno': 2.5,
            'nio': 8.0,
            'c60': 200.0,
            
            # Metals
            'au': 60000.0,
            'ag': 800.0,
            'al': 2.0,
            'cu': 8.0,
            'cr': 15.0,
            
            # Default for unknown materials
            'unknown': 10.0
        }
        
        # Manufacturing yield factors
        self.yield_factors = {
            'spin_coat': 0.85,
            'slot_die': 0.92,
            'inkjet': 0.88,
            'thermal_evap': 0.95,
            'sputtering': 0.93,
            'cvd': 0.90,
            'mocvd': 0.87,
            'mbe': 0.98,
            'alds': 0.89
        }
        
        # Throughput (m²/hour)
        self.throughput_rates = {
            'spin_coat': 20.0,
            'slot_die': 100.0,
            'inkjet': 15.0,
            'thermal_evap': 5.0,
            'sputtering': 8.0,
            'cvd': 12.0,
            'mocvd': 3.0,
            'mbe': 1.0,
            'alds': 2.0
        }
        
        # LCOE parameters
        self.lcoe_params = {
            'discount_rate': 0.07,        # 7% WACC
            'system_lifetime': 25,         # years
            'capacity_factor': 0.20,       # Average solar CF
            'system_efficiency': 0.85,     # System losses
            'om_cost_per_kw': 15.0,       # $/kW-year O&M
            'installation_cost_per_w': 0.30  # $/W balance of system
        }
    
    def calculate_material_cost(self,
                              material_name: str,
                              thickness_nm: float,
                              area_m2: float = 1.0) -> Dict[str, float]:
        """
        Calculate material cost for a given layer.
        
        Args:
            material_name: Name of material
            thickness_nm: Thickness in nanometers
            area_m2: Device area in m²
            
        Returns:
            Dict with cost breakdown
        """
        # Get material density (kg/m³)
        try:
            material_props = self.material_db.get_material_properties(material_name)
            density = material_props.get('density', 2000)  # kg/m³
        except:
            density = 2000  # Default density
        
        # Calculate volume and mass
        thickness_m = thickness_nm * 1e-9
        volume_m3 = thickness_m * area_m2
        mass_kg = volume_m3 * density
        mass_g = mass_kg * 1000
        
        # Get unit cost
        unit_cost = self.material_unit_costs.get(material_name, 
                                                self.material_unit_costs['unknown'])
        
        # Calculate total material cost
        material_cost = mass_g * unit_cost
        
        # Add 20% waste factor
        waste_factor = 1.2
        total_material_cost = material_cost * waste_factor
        
        return {
            'thickness_nm': thickness_nm,
            'area_m2': area_m2,
            'volume_m3': volume_m3,
            'mass_kg': mass_kg,
            'unit_cost_per_g': unit_cost,
            'raw_material_cost': material_cost,
            'waste_factor': waste_factor,
            'total_material_cost': total_material_cost
        }
    
    def calculate_processing_cost(self,
                                material_name: str,
                                deposition_method: str,
                                area_m2: float = 1.0) -> Dict[str, float]:
        """
        Calculate processing cost for layer deposition.
        
        Args:
            material_name: Name of material
            deposition_method: Deposition method key
            area_m2: Device area in m²
            
        Returns:
            Dict with processing cost breakdown
        """
        if deposition_method not in self.deposition_costs:
            deposition_method = 'spin_coat'  # Default
        
        # Base processing cost
        base_cost_per_m2 = self.deposition_costs[deposition_method]
        base_cost = base_cost_per_m2 * area_m2
        
        # Yield adjustment
        yield_factor = self.yield_factors[deposition_method]
        yield_adjusted_cost = base_cost / yield_factor
        
        # Throughput (affects labor cost)
        throughput = self.throughput_rates[deposition_method]
        processing_time_hours = area_m2 / throughput
        
        # Labor cost (assuming $30/hour fully loaded)
        labor_cost_per_hour = 30.0
        labor_cost = processing_time_hours * labor_cost_per_hour
        
        # Equipment depreciation (simplified)
        equipment_cost_per_hour = base_cost_per_m2 * 0.1  # 10% of process cost
        equipment_cost = processing_time_hours * equipment_cost_per_hour
        
        total_processing_cost = yield_adjusted_cost + labor_cost + equipment_cost
        
        return {
            'deposition_method': deposition_method,
            'base_cost': base_cost,
            'yield_factor': yield_factor,
            'yield_adjusted_cost': yield_adjusted_cost,
            'processing_time_hours': processing_time_hours,
            'labor_cost': labor_cost,
            'equipment_cost': equipment_cost,
            'total_processing_cost': total_processing_cost
        }
    
    def calculate_layer_cost(self,
                           material_name: str,
                           thickness_nm: float,
                           deposition_method: str = None,
                           area_m2: float = 1.0) -> Dict[str, Any]:
        """
        Calculate total cost for a single layer.
        
        Args:
            material_name: Name of material
            thickness_nm: Thickness in nanometers
            deposition_method: Deposition method (auto-selected if None)
            area_m2: Device area in m²
            
        Returns:
            Dict with complete layer cost analysis
        """
        # Auto-select deposition method if not specified
        if deposition_method is None:
            deposition_method = self._select_deposition_method(material_name, thickness_nm)
        
        # Calculate material and processing costs
        material_cost = self.calculate_material_cost(material_name, thickness_nm, area_m2)
        processing_cost = self.calculate_processing_cost(material_name, deposition_method, area_m2)
        
        # Total layer cost
        total_cost = (material_cost['total_material_cost'] + 
                     processing_cost['total_processing_cost'])
        
        return {
            'material_name': material_name,
            'thickness_nm': thickness_nm,
            'deposition_method': deposition_method,
            'material_cost_analysis': material_cost,
            'processing_cost_analysis': processing_cost,
            'total_layer_cost': total_cost,
            'cost_per_m2': total_cost / area_m2
        }
    
    def calculate_stack_manufacturing_cost(self,
                                         layer_stack: List[Tuple[str, float]],
                                         deposition_methods: List[str] = None,
                                         area_m2: float = 1.0) -> Dict[str, Any]:
        """
        Calculate total manufacturing cost for a complete stack.
        
        Args:
            layer_stack: List of (material_name, thickness_nm) tuples
            deposition_methods: Optional list of deposition methods
            area_m2: Device area in m²
            
        Returns:
            Dict with complete stack cost analysis
        """
        if deposition_methods is None:
            deposition_methods = [None] * len(layer_stack)
        elif len(deposition_methods) != len(layer_stack):
            raise ValueError("deposition_methods must match layer_stack length")
        
        layer_costs = []
        total_material_cost = 0.0
        total_processing_cost = 0.0
        total_manufacturing_cost = 0.0
        
        for i, ((material, thickness), method) in enumerate(zip(layer_stack, deposition_methods)):
            layer_cost = self.calculate_layer_cost(material, thickness, method, area_m2)
            layer_costs.append(layer_cost)
            
            total_material_cost += layer_cost['material_cost_analysis']['total_material_cost']
            total_processing_cost += layer_cost['processing_cost_analysis']['total_processing_cost']
            total_manufacturing_cost += layer_cost['total_layer_cost']
        
        # Add overhead costs (facilities, utilities, management)
        overhead_factor = 1.15  # 15% overhead
        total_with_overhead = total_manufacturing_cost * overhead_factor
        
        # Manufacturability score based on complexity and yields
        manufacturability_score = self._calculate_manufacturability_score(layer_costs)
        
        return {
            'layer_stack': layer_stack,
            'layer_costs': layer_costs,
            'total_material_cost': total_material_cost,
            'total_processing_cost': total_processing_cost,
            'total_manufacturing_cost': total_manufacturing_cost,
            'overhead_cost': total_with_overhead - total_manufacturing_cost,
            'total_with_overhead': total_with_overhead,
            'cost_per_m2': total_with_overhead / area_m2,
            'manufacturability_score': manufacturability_score,
            'n_layers': len(layer_stack)
        }
    
    def calculate_lcoe(self,
                      manufacturing_cost_per_m2: float,
                      pce_percent: float,
                      degradation_rate_per_year: float = 0.005) -> Dict[str, float]:
        """
        Calculate Levelized Cost of Energy (LCOE).
        
        Args:
            manufacturing_cost_per_m2: Module manufacturing cost in $/m²
            pce_percent: Power conversion efficiency in %
            degradation_rate_per_year: Annual degradation rate (default 0.5%/year)
            
        Returns:
            Dict with LCOE analysis
        """
        # Convert PCE to fraction
        pce_fraction = pce_percent / 100.0
        
        # Solar irradiance (kWh/m²/year for reference location)
        annual_irradiance = 1800.0  # kWh/m²/year (good solar resource)
        
        # Module power output (kW/m²)
        module_power_density = pce_fraction * 1.0  # kW/m² (1 kW/m² at STC)
        
        # System efficiency and capacity factor
        system_efficiency = self.lcoe_params['system_efficiency']
        capacity_factor = self.lcoe_params['capacity_factor']
        
        # Annual energy production (kWh/m²/year)
        annual_energy_per_m2 = (annual_irradiance * pce_fraction * 
                               system_efficiency * capacity_factor)
        
        # Installation cost ($/m²)
        installation_cost_per_m2 = (self.lcoe_params['installation_cost_per_w'] * 
                                   module_power_density * 1000)  # Convert kW to W
        
        # Total CAPEX ($/m²)
        total_capex_per_m2 = manufacturing_cost_per_m2 + installation_cost_per_m2
        
        # O&M cost ($/m²/year)
        om_cost_per_m2_per_year = (self.lcoe_params['om_cost_per_kw'] * 
                                  module_power_density)
        
        # Capital recovery factor
        discount_rate = self.lcoe_params['discount_rate']
        lifetime = self.lcoe_params['system_lifetime']
        
        crf = (discount_rate * (1 + discount_rate)**lifetime / 
               ((1 + discount_rate)**lifetime - 1))
        
        # Present value of energy with degradation
        pv_energy = 0.0
        for year in range(1, lifetime + 1):
            degradation_factor = (1 - degradation_rate_per_year) ** (year - 1)
            annual_energy_degraded = annual_energy_per_m2 * degradation_factor
            pv_energy += annual_energy_degraded / (1 + discount_rate)**(year - 1)
        
        # Present value of O&M costs
        pv_om = 0.0
        for year in range(1, lifetime + 1):
            pv_om += om_cost_per_m2_per_year / (1 + discount_rate)**(year - 1)
        
        # LCOE calculation ($/kWh)
        lcoe = (total_capex_per_m2 * crf + pv_om / lifetime) / (pv_energy / lifetime)
        
        return {
            'lcoe_dollar_per_kwh': lcoe,
            'lcoe_cents_per_kwh': lcoe * 100,
            'manufacturing_cost_per_m2': manufacturing_cost_per_m2,
            'installation_cost_per_m2': installation_cost_per_m2,
            'total_capex_per_m2': total_capex_per_m2,
            'annual_energy_per_m2': annual_energy_per_m2,
            'om_cost_per_m2_per_year': om_cost_per_m2_per_year,
            'pce_percent': pce_percent,
            'degradation_rate_per_year': degradation_rate_per_year,
            'system_lifetime_years': lifetime
        }
    
    def analyze_junction_sweet_spot(self,
                                  track_a_designs: List[List[Tuple[str, float]]],
                                  track_b_designs: List[List[Tuple[str, float]]],
                                  pce_values: List[float]) -> Dict[str, Any]:
        """
        Analyze sweet spot for number of junctions (marginal cost vs. PCE).
        
        Args:
            track_a_designs: List of Track A layer stacks
            track_b_designs: List of Track B layer stacks  
            pce_values: Corresponding PCE values
            
        Returns:
            Dict with sweet spot analysis
        """
        if len(track_a_designs) != len(pce_values) or len(track_b_designs) != len(pce_values):
            raise ValueError("Design lists and PCE values must have same length")
        
        # Calculate costs for both tracks
        track_a_costs = []
        track_b_costs = []
        track_a_lcoe = []
        track_b_lcoe = []
        
        for i, (design_a, design_b, pce) in enumerate(zip(track_a_designs, track_b_designs, pce_values)):
            # Track A analysis
            cost_a = self.calculate_stack_manufacturing_cost(design_a)
            lcoe_a = self.calculate_lcoe(cost_a['cost_per_m2'], pce)
            track_a_costs.append(cost_a['cost_per_m2'])
            track_a_lcoe.append(lcoe_a['lcoe_cents_per_kwh'])
            
            # Track B analysis
            cost_b = self.calculate_stack_manufacturing_cost(design_b)
            lcoe_b = self.calculate_lcoe(cost_b['cost_per_m2'], pce)
            track_b_costs.append(cost_b['cost_per_m2'])
            track_b_lcoe.append(lcoe_b['lcoe_cents_per_kwh'])
        
        # Calculate marginal gains
        track_a_marginal_pce = np.diff(pce_values)
        track_a_marginal_cost = np.diff(track_a_costs)
        track_b_marginal_pce = np.diff(pce_values)
        track_b_marginal_cost = np.diff(track_b_costs)
        
        # Marginal cost per PCE point
        track_a_marginal_cost_per_pce = track_a_marginal_cost / np.maximum(track_a_marginal_pce, 0.01)
        track_b_marginal_cost_per_pce = track_b_marginal_cost / np.maximum(track_b_marginal_pce, 0.01)
        
        # Find sweet spots (minimum LCOE)
        track_a_sweet_spot = np.argmin(track_a_lcoe)
        track_b_sweet_spot = np.argmin(track_b_lcoe)
        
        return {
            'n_junctions': list(range(1, len(pce_values) + 1)),
            'pce_values': pce_values,
            'track_a': {
                'costs_per_m2': track_a_costs,
                'lcoe_cents_per_kwh': track_a_lcoe,
                'marginal_cost_per_pce': list(track_a_marginal_cost_per_pce),
                'sweet_spot_junction': track_a_sweet_spot + 1,
                'sweet_spot_lcoe': track_a_lcoe[track_a_sweet_spot]
            },
            'track_b': {
                'costs_per_m2': track_b_costs,
                'lcoe_cents_per_kwh': track_b_lcoe,
                'marginal_cost_per_pce': list(track_b_marginal_cost_per_pce),
                'sweet_spot_junction': track_b_sweet_spot + 1,
                'sweet_spot_lcoe': track_b_lcoe[track_b_sweet_spot]
            },
            'winner': 'track_a' if track_a_lcoe[track_a_sweet_spot] < track_b_lcoe[track_b_sweet_spot] else 'track_b'
        }
    
    def _select_deposition_method(self, material_name: str, thickness_nm: float) -> str:
        """Auto-select appropriate deposition method based on material and thickness."""
        name = material_name.lower()
        
        # Thick substrates
        if thickness_nm > 100000:  # > 100 μm
            return 'slot_die'
        
        # Transparent conductors
        elif any(tc in name for tc in ['ito', 'fto', 'ato']):
            return 'sputtering'
        
        # Perovskites
        elif 'perovskite' in name or any(pv in name for pv in ['mapb', 'fapb', 'cspb']):
            if thickness_nm < 300:
                return 'spin_coat'
            else:
                return 'slot_die'
        
        # Organic materials
        elif any(org in name for org in ['pedot', 'ptaa', 'spiro', 'p3ht', 'pcbm']):
            return 'spin_coat'
        
        # Inorganic materials
        elif any(inorg in name for inorg in ['tio2', 'sno2', 'zno', 'nio']):
            if thickness_nm < 100:
                return 'alds'
            else:
                return 'cvd'
        
        # Metals
        elif any(metal in name for metal in ['au', 'ag', 'al', 'cu', 'cr']):
            if thickness_nm < 200:
                return 'thermal_evap'
            else:
                return 'sputtering'
        
        # Default
        else:
            return 'spin_coat'
    
    def _calculate_manufacturability_score(self, layer_costs: List[Dict]) -> float:
        """Calculate overall manufacturability score (0-100)."""
        if not layer_costs:
            return 50.0
        
        # Average yield
        yields = [cost['processing_cost_analysis']['yield_factor'] for cost in layer_costs]
        avg_yield = np.mean(yields)
        
        # Throughput score (higher is better)
        throughputs = [self.throughput_rates[cost['deposition_method']] for cost in layer_costs]
        avg_throughput = np.mean(throughputs)
        throughput_score = min(100, avg_throughput / 50.0 * 100)
        
        # Complexity penalty (more layers = lower score)
        complexity_score = max(0, 100 - len(layer_costs) * 5)
        
        # Overall score
        manufacturability = (avg_yield * 100 * 0.4 + 
                           throughput_score * 0.3 + 
                           complexity_score * 0.3)
        
        return min(100, max(0, manufacturability))


def demo_economics_engine():
    """Demonstrate economics engine capabilities."""
    engine = EconomicsEngine()
    
    # Example stacks for comparison
    single_junction = [
        ('glass', 3000000),      # 3mm substrate
        ('ITO', 100),           # Transparent conductor
        ('PEDOT', 50),          # HTL
        ('perovskite', 500),    # Active layer 1
        ('C60', 100),           # ETL
        ('Au', 80)              # Back contact
    ]
    
    tandem_junction = [
        ('glass', 3000000),
        ('ITO', 100),
        ('PEDOT', 50),
        ('perovskite', 300),  # Wide gap top
        ('C60', 50),
        ('ITO', 50),          # Recombination layer
        ('PEDOT', 30),
        ('perovskite', 600),  # Narrow gap bottom
        ('C60', 100),
        ('Au', 80)
    ]
    
    print("=== Economics Engine Demo ===")
    
    # Single junction analysis
    cost_1j = engine.calculate_stack_manufacturing_cost(single_junction)
    lcoe_1j = engine.calculate_lcoe(cost_1j['cost_per_m2'], 22.0)  # 22% PCE
    
    print(f"Single Junction:")
    print(f"  Manufacturing Cost: ${cost_1j['cost_per_m2']:.2f}/m²")
    print(f"  LCOE: {lcoe_1j['lcoe_cents_per_kwh']:.2f} ¢/kWh")
    print(f"  Manufacturability: {cost_1j['manufacturability_score']:.1f}/100")
    
    # Tandem analysis
    cost_2j = engine.calculate_stack_manufacturing_cost(tandem_junction)
    lcoe_2j = engine.calculate_lcoe(cost_2j['cost_per_m2'], 28.0)  # 28% PCE
    
    print(f"\nTandem Junction:")
    print(f"  Manufacturing Cost: ${cost_2j['cost_per_m2']:.2f}/m²")
    print(f"  LCOE: {lcoe_2j['lcoe_cents_per_kwh']:.2f} ¢/kWh")
    print(f"  Manufacturability: {cost_2j['manufacturability_score']:.1f}/100")
    
    print(f"\nCost Increase: {(cost_2j['cost_per_m2']/cost_1j['cost_per_m2']-1)*100:.1f}%")
    print(f"LCOE Improvement: {(1-lcoe_2j['lcoe_cents_per_kwh']/lcoe_1j['lcoe_cents_per_kwh'])*100:.1f}%")
    
    return {
        'single_junction': {'cost': cost_1j, 'lcoe': lcoe_1j},
        'tandem_junction': {'cost': cost_2j, 'lcoe': lcoe_2j}
    }


if __name__ == "__main__":
    demo_economics_engine()