"""
Test module for economics engine and manufacturing cost calculations.

Tests cost modeling, LCOE calculations, and manufacturability analysis
for photovoltaic device stacks.
"""

import pytest
import numpy as np
import warnings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.economics import EconomicsEngine

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestEconomics:
    """Test economics and manufacturing cost calculations."""
    
    @pytest.fixture
    def economics(self):
        """Create economics engine instance."""
        return EconomicsEngine()
    
    def test_material_cost_calculation(self, economics):
        """Test material cost calculation for individual layers."""
        # Test a typical perovskite layer
        result = economics.calculate_material_cost(
            material_name='perovskite',
            thickness_nm=500,
            area_m2=1.0
        )
        
        assert 'total_material_cost' in result
        assert 'mass_kg' in result
        assert 'unit_cost_per_g' in result
        
        # Cost should be positive
        assert result['total_material_cost'] > 0
        
        # Thicker layers should cost more
        thick_result = economics.calculate_material_cost(
            'perovskite', 1000, 1.0  # Double thickness
        )
        
        assert thick_result['total_material_cost'] > result['total_material_cost']
    
    def test_processing_cost_calculation(self, economics):
        """Test processing cost calculation."""
        # Test different deposition methods
        methods = ['spin_coat', 'slot_die', 'thermal_evap']
        
        costs = {}
        for method in methods:
            result = economics.calculate_processing_cost(
                material_name='perovskite',
                deposition_method=method,
                area_m2=1.0
            )
            
            assert 'total_processing_cost' in result
            assert 'yield_factor' in result
            assert 'processing_time_hours' in result
            
            costs[method] = result['total_processing_cost']
        
        # Different methods should have different costs
        assert len(set(costs.values())) > 1  # Not all the same
        
        # Thermal evaporation should be more expensive than spin coating
        assert costs['thermal_evap'] > costs['spin_coat']
    
    def test_layer_cost_integration(self, economics):
        """Test integration of material and processing costs."""
        result = economics.calculate_layer_cost(
            material_name='perovskite',
            thickness_nm=500,
            deposition_method='spin_coat'
        )
        
        assert 'total_layer_cost' in result
        assert 'material_cost_analysis' in result
        assert 'processing_cost_analysis' in result
        
        # Total cost should be sum of material and processing
        material_cost = result['material_cost_analysis']['total_material_cost']
        processing_cost = result['processing_cost_analysis']['total_processing_cost']
        
        expected_total = material_cost + processing_cost
        assert abs(result['total_layer_cost'] - expected_total) < 0.01
    
    def test_stack_manufacturing_cost(self, economics):
        """Test complete stack manufacturing cost."""
        # Simple single junction stack
        stack = [
            ('glass', 3000000),   # Substrate
            ('ITO', 100),         # Transparent conductor
            ('PEDOT', 50),        # HTL
            ('perovskite', 500),  # Active layer
            ('C60', 100),         # ETL
            ('Au', 80)            # Back contact
        ]
        
        result = economics.calculate_stack_manufacturing_cost(stack)
        
        assert 'total_manufacturing_cost' in result
        assert 'total_with_overhead' in result
        assert 'manufacturability_score' in result
        assert 'layer_costs' in result
        
        # Should have overhead
        assert result['total_with_overhead'] > result['total_manufacturing_cost']
        
        # Manufacturability should be 0-100
        assert 0 <= result['manufacturability_score'] <= 100
        
        # Should have cost for each layer
        assert len(result['layer_costs']) == len(stack)
    
    def test_lcoe_calculation(self, economics):
        """Test Levelized Cost of Energy calculation."""
        # Typical values
        manufacturing_cost = 200.0  # $/m²
        pce = 20.0                 # %
        
        result = economics.calculate_lcoe(manufacturing_cost, pce)
        
        assert 'lcoe_dollar_per_kwh' in result
        assert 'lcoe_cents_per_kwh' in result
        assert 'total_capex_per_m2' in result
        assert 'annual_energy_per_m2' in result
        
        # LCOE should be positive
        assert result['lcoe_dollar_per_kwh'] > 0
        
        # Cents should be 100x dollars
        cents_calculated = result['lcoe_dollar_per_kwh'] * 100
        assert abs(result['lcoe_cents_per_kwh'] - cents_calculated) < 0.01
    
    def test_lcoe_efficiency_dependence(self, economics):
        """Test LCOE dependence on efficiency."""
        manufacturing_cost = 200.0  # $/m²
        
        # Different efficiencies
        efficiencies = [15.0, 20.0, 25.0]  # %
        lcoes = []
        
        for eff in efficiencies:
            result = economics.calculate_lcoe(manufacturing_cost, eff)
            lcoes.append(result['lcoe_cents_per_kwh'])
        
        # Higher efficiency should give lower LCOE
        assert lcoes[1] < lcoes[0]  # 20% < 15%
        assert lcoes[2] < lcoes[1]  # 25% < 20%
    
    def test_lcoe_cost_dependence(self, economics):
        """Test LCOE dependence on manufacturing cost."""
        pce = 20.0  # %
        
        # Different manufacturing costs
        costs = [150.0, 200.0, 250.0]  # $/m²
        lcoes = []
        
        for cost in costs:
            result = economics.calculate_lcoe(cost, pce)
            lcoes.append(result['lcoe_cents_per_kwh'])
        
        # Higher cost should give higher LCOE
        assert lcoes[1] > lcoes[0]
        assert lcoes[2] > lcoes[1]
    
    def test_cost_scaling_with_junctions(self, economics):
        """Test that cost scales with number of junctions."""
        # Single junction
        single_j = [
            ('glass', 3000000), ('ITO', 100), ('PEDOT', 50),
            ('perovskite', 500), ('C60', 100), ('Au', 80)
        ]
        
        # Two junction (tandem)
        tandem_j = [
            ('glass', 3000000), ('ITO', 100),
            ('PEDOT', 50), ('perovskite', 300), ('C60', 50),   # Top cell
            ('ITO', 50),                                        # Tunnel junction
            ('PEDOT', 30), ('perovskite', 600), ('C60', 100),  # Bottom cell
            ('Au', 80)
        ]
        
        cost_1j = economics.calculate_stack_manufacturing_cost(single_j)
        cost_2j = economics.calculate_stack_manufacturing_cost(tandem_j)
        
        # Tandem should be more expensive
        assert cost_2j['cost_per_m2'] > cost_1j['cost_per_m2']
        
        # But not excessively more (reasonable scaling)
        cost_ratio = cost_2j['cost_per_m2'] / cost_1j['cost_per_m2']
        assert 1.3 < cost_ratio < 3.0  # 30-200% increase is reasonable
    
    def test_sweet_spot_analysis(self, economics):
        """Test sweet spot analysis for junction optimization."""
        # Mock designs for different junction numbers
        designs_a = [
            # 1J
            [('glass', 3000000), ('ITO', 100), ('PEDOT', 50), ('perovskite', 500), ('C60', 100), ('Au', 80)],
            # 2J
            [('glass', 3000000), ('ITO', 100), ('PEDOT', 50), ('perovskite', 300), ('C60', 50),
             ('ITO', 50), ('PEDOT', 30), ('perovskite', 600), ('C60', 100), ('Au', 80)],
            # 3J
            [('glass', 3000000), ('ITO', 100), ('PEDOT', 30), ('perovskite', 200), ('C60', 30),
             ('ITO', 40), ('PEDOT', 30), ('perovskite', 300), ('C60', 30),
             ('ITO', 40), ('PEDOT', 30), ('perovskite', 500), ('C60', 100), ('Au', 80)]
        ]
        
        # Same designs for Track B (simplified)
        designs_b = designs_a.copy()
        
        # Mock PCE values (should increase with junctions but with diminishing returns)
        pce_values = [22.0, 28.0, 32.0]
        
        result = economics.analyze_junction_sweet_spot(designs_a, designs_b, pce_values)
        
        assert 'track_a' in result
        assert 'track_b' in result
        assert 'winner' in result
        
        # Should identify sweet spots
        assert 'sweet_spot_junction' in result['track_a']
        assert 'sweet_spot_junction' in result['track_b']
        
        # LCOE should be calculated for each design
        lcoe_a = result['track_a']['lcoe_cents_per_kwh']
        assert len(lcoe_a) == len(pce_values)
        assert all(lcoe > 0 for lcoe in lcoe_a)
    
    def test_manufacturability_scoring(self, economics):
        """Test manufacturability score calculation."""
        # High-throughput, high-yield process
        easy_stack = [
            ('glass', 3000000),
            ('ITO', 100),      # Sputtered (high yield)
            ('perovskite', 500) # Spin coated (fast)
        ]
        
        # Complex, low-throughput process
        complex_stack = [
            ('glass', 3000000),
            ('ITO', 100),
            ('TiO2', 50),      # ALD (slow)
            ('perovskite', 300),
            ('C60', 100),      # Evaporated
            ('MoO3', 20),      # Evaporated
            ('Au', 80)         # Evaporated
        ]
        
        result_easy = economics.calculate_stack_manufacturing_cost(easy_stack)
        result_complex = economics.calculate_stack_manufacturing_cost(complex_stack)
        
        # Simple stack should have higher manufacturability
        assert (result_easy['manufacturability_score'] >= 
                result_complex['manufacturability_score'])
    
    def test_deposition_method_selection(self, economics):
        """Test automatic deposition method selection."""
        # Test different materials
        materials = ['perovskite', 'ITO', 'Au', 'TiO2']
        
        for material in materials:
            result = economics.calculate_layer_cost(material, 500)
            
            # Should auto-select appropriate method
            assert 'deposition_method' in result
            method = result['deposition_method']
            assert method in economics.deposition_costs.keys()
            
            # Method should be reasonable for material type
            if material == 'perovskite':
                assert method in ['spin_coat', 'slot_die']
            elif material == 'ITO':
                assert method == 'sputtering'
            elif material == 'Au':
                assert method in ['thermal_evap', 'sputtering']
    
    def test_area_scaling(self, economics):
        """Test cost scaling with device area."""
        stack = [('perovskite', 500)]
        
        areas = [0.1, 1.0, 10.0]  # m²
        costs = []
        
        for area in areas:
            result = economics.calculate_stack_manufacturing_cost(stack, area_m2=area)
            costs.append(result['total_with_overhead'])
        
        # Total cost should scale with area
        for i in range(1, len(costs)):
            ratio = costs[i] / costs[i-1]
            area_ratio = areas[i] / areas[i-1]
            assert abs(ratio - area_ratio) < 0.1  # Should be approximately linear
    
    def test_cost_breakdown_analysis(self, economics):
        """Test detailed cost breakdown."""
        stack = [
            ('glass', 3000000),
            ('ITO', 100),
            ('perovskite', 500),
            ('Au', 80)
        ]
        
        result = economics.calculate_stack_manufacturing_cost(stack)
        
        # Check that material and processing costs are separately tracked
        total_material = result['total_material_cost']
        total_processing = result['total_processing_cost']
        total_manufacturing = result['total_manufacturing_cost']
        
        # Manufacturing should be sum of material and processing
        assert abs(total_manufacturing - (total_material + total_processing)) < 0.01
        
        # Should be able to identify dominant cost drivers
        layer_costs = [layer['total_layer_cost'] for layer in result['layer_costs']]
        max_cost_layer = np.argmax(layer_costs)
        
        # Substrate (glass) is usually dominant
        assert max_cost_layer == 0  # First layer (glass)


class TestCostOptimization:
    """Test cost optimization strategies."""
    
    def test_thickness_optimization(self):
        """Test finding optimal layer thicknesses for cost."""
        economics = EconomicsEngine()
        
        # Test different thicknesses of expensive material
        material = 'Au'  # Expensive
        thicknesses = [50, 80, 100, 150]  # nm
        
        costs = []
        for thickness in thicknesses:
            result = economics.calculate_layer_cost(material, thickness)
            costs.append(result['total_layer_cost'])
        
        # Cost should increase with thickness
        for i in range(1, len(costs)):
            assert costs[i] > costs[i-1]
        
        # But may not be linear due to processing costs
    
    def test_material_substitution_analysis(self):
        """Test cost impact of material substitution."""
        economics = EconomicsEngine()
        
        # Compare expensive vs. cheap alternatives
        expensive_stack = [
            ('glass', 3000000),
            ('ITO', 100),      # Expensive TC
            ('spiro', 100),    # Expensive HTL
            ('perovskite', 500),
            ('Au', 100)        # Expensive contact
        ]
        
        cheap_stack = [
            ('glass', 3000000),
            ('FTO', 100),      # Cheaper TC
            ('PEDOT', 100),    # Cheaper HTL
            ('perovskite', 500),
            ('Ag', 100)        # Cheaper contact
        ]
        
        result_exp = economics.calculate_stack_manufacturing_cost(expensive_stack)
        result_cheap = economics.calculate_stack_manufacturing_cost(cheap_stack)
        
        # Cheap stack should cost less
        assert result_cheap['cost_per_m2'] < result_exp['cost_per_m2']


def test_realistic_economic_analysis():
    """Test realistic economic analysis scenarios."""
    economics = EconomicsEngine()
    
    print("=== Economics Engine Tests ===")
    
    # Single junction analysis
    single_j = [
        ('glass', 3000000), ('ITO', 100), ('PEDOT', 50),
        ('perovskite', 500), ('C60', 100), ('Au', 80)
    ]
    
    cost_1j = economics.calculate_stack_manufacturing_cost(single_j)
    lcoe_1j = economics.calculate_lcoe(cost_1j['cost_per_m2'], 20.0)
    
    print(f"Single Junction:")
    print(f"  Cost: ${cost_1j['cost_per_m2']:.1f}/m²")
    print(f"  LCOE: {lcoe_1j['lcoe_cents_per_kwh']:.1f} ¢/kWh")
    print(f"  Manufacturability: {cost_1j['manufacturability_score']:.1f}/100")
    
    # Tandem junction analysis
    tandem_j = [
        ('glass', 3000000), ('ITO', 100),
        ('PEDOT', 50), ('perovskite', 300), ('C60', 50),
        ('ITO', 50), ('PEDOT', 30), ('perovskite', 600),
        ('C60', 100), ('Au', 80)
    ]
    
    cost_2j = economics.calculate_stack_manufacturing_cost(tandem_j)
    lcoe_2j = economics.calculate_lcoe(cost_2j['cost_per_m2'], 28.0)
    
    print(f"\nTandem Junction:")
    print(f"  Cost: ${cost_2j['cost_per_m2']:.1f}/m²")
    print(f"  LCOE: {lcoe_2j['lcoe_cents_per_kwh']:.1f} ¢/kWh")
    print(f"  Manufacturability: {cost_2j['manufacturability_score']:.1f}/100")
    
    # Analysis
    cost_increase = (cost_2j['cost_per_m2'] / cost_1j['cost_per_m2'] - 1) * 100
    lcoe_improvement = (1 - lcoe_2j['lcoe_cents_per_kwh'] / lcoe_1j['lcoe_cents_per_kwh']) * 100
    
    print(f"\nCost increase: {cost_increase:.1f}%")
    print(f"LCOE improvement: {lcoe_improvement:.1f}%")
    
    # Basic checks
    assert cost_1j['cost_per_m2'] > 0, "Single junction should have positive cost"
    assert cost_2j['cost_per_m2'] > cost_1j['cost_per_m2'], "Tandem should cost more"
    assert lcoe_1j['lcoe_cents_per_kwh'] > 0, "LCOE should be positive"
    assert cost_increase < 200, "Cost increase should be reasonable"
    
    # Sweet spot indication
    if lcoe_improvement > 0:
        print("Tandem shows LCOE advantage")
    else:
        print("Single junction has lower LCOE")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_realistic_economic_analysis()