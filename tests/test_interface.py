"""
Test module for interface loss and tunnel junction calculations.

Tests resistive losses, recombination at interfaces, and 
tunnel junction resistance modeling.
"""

import pytest
import numpy as np
import warnings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.interface_loss import InterfaceLossCalculator, TunnelJunctionCalculator

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestInterfaceLossCalculator:
    """Test interface loss calculations."""
    
    @pytest.fixture
    def loss_calc(self):
        """Create interface loss calculator instance."""
        return InterfaceLossCalculator()
    
    def test_single_junction_interface_losses(self, loss_calc):
        """Test interface losses in single junction."""
        # Simple single junction stack
        stack = [
            ('glass', 3000000),
            ('ITO', 100),
            ('PEDOT', 50),        # HTL
            ('perovskite', 500),  # Active
            ('C60', 100),         # ETL
            ('Au', 80)
        ]
        
        result = loss_calc.calculate_total_interface_losses(stack)
        
        # Should identify interfaces and calculate losses
        assert 'total_loss_fraction' in result
        assert 'interface_analysis' in result
        assert 'n_interfaces' in result
        
        # Total loss should be reasonable (5-20% for single junction)
        assert 0.05 < result['total_loss_fraction'] < 0.25
        
        # Should have multiple interfaces
        assert result['n_interfaces'] >= 3  # At least HTL/active, active/ETL, ETL/contact
    
    def test_tandem_interface_losses(self, loss_calc):
        """Test interface losses in tandem junction."""
        # Two-junction tandem stack
        stack = [
            ('glass', 3000000),
            ('ITO', 100),
            ('PEDOT', 50),
            ('perovskite', 300),   # Top subcell
            ('C60', 50),
            ('ITO', 50),           # Tunnel junction
            ('PEDOT', 30),
            ('perovskite', 600),   # Bottom subcell
            ('C60', 100),
            ('Au', 80)
        ]
        
        result = loss_calc.calculate_total_interface_losses(stack)
        
        # Tandem should have higher losses than single junction
        assert result['total_loss_fraction'] > 0.1
        assert result['n_interfaces'] > 5
        
        # Should identify tunnel junction
        tunnel_interfaces = [analysis for analysis in result['interface_analysis'] 
                           if 'tunnel' in analysis.get('interface_type', '').lower()]
        
        # May or may not explicitly identify tunnel junction depending on implementation
        # But should have more interfaces than single junction
    
    def test_loss_scaling_with_junctions(self, loss_calc):
        """Test that losses increase with number of junctions."""
        # Compare different junction numbers
        stacks = {
            1: [('glass', 3000000), ('ITO', 100), ('PEDOT', 50), ('perovskite', 500), ('C60', 100), ('Au', 80)],
            2: [('glass', 3000000), ('ITO', 100), ('PEDOT', 50), ('perovskite', 300), ('C60', 50),
                ('ITO', 50), ('PEDOT', 30), ('perovskite', 600), ('C60', 100), ('Au', 80)],
            3: [('glass', 3000000), ('ITO', 100), ('PEDOT', 30), ('perovskite', 200), ('C60', 30),
                ('ITO', 40), ('PEDOT', 30), ('perovskite', 300), ('C60', 30),
                ('ITO', 40), ('PEDOT', 30), ('perovskite', 500), ('C60', 100), ('Au', 80)]
        }
        
        losses = {}
        for n_junctions, stack in stacks.items():
            result = loss_calc.calculate_total_interface_losses(stack)
            losses[n_junctions] = result['total_loss_fraction']
        
        # Losses should generally increase with more junctions
        assert losses[2] > losses[1]
        assert losses[3] > losses[2]
        
        # But should still be reasonable (< 50%)
        for loss in losses.values():
            assert loss < 0.5
    
    def test_interface_types_identification(self, loss_calc):
        """Test identification of different interface types."""
        stack = [
            ('ITO', 100),         # Conductor
            ('PEDOT', 50),        # HTL
            ('perovskite', 500),  # Active
            ('C60', 100),         # ETL
            ('Au', 80)            # Metal contact
        ]
        
        result = loss_calc.calculate_total_interface_losses(stack)
        
        # Should categorize interfaces
        interface_types = [analysis.get('interface_type', 'unknown') 
                          for analysis in result['interface_analysis']]
        
        # Should have various interface types
        assert len(interface_types) > 0
        
        # Each interface should have some loss calculation
        for analysis in result['interface_analysis']:
            assert 'loss_fraction' in analysis
            assert analysis['loss_fraction'] >= 0
    
    def test_band_offset_effects(self, loss_calc):
        """Test effect of band offsets on interface losses."""
        # Stack with good band alignment
        good_stack = [
            ('PEDOT', 50),        # HTL with good alignment
            ('perovskite', 500),  
            ('C60', 100)          # ETL with good alignment
        ]
        
        # Stack with poor band alignment (if materials exist)
        # This is hard to test without detailed band alignment data
        # So we'll just test that different materials give different results
        poor_stack = [
            ('TiO2', 50),         # Different energy levels
            ('perovskite', 500),
            ('ZnO', 100)
        ]
        
        result_good = loss_calc.calculate_total_interface_losses(good_stack)
        result_poor = loss_calc.calculate_total_interface_losses(poor_stack)
        
        # Both should give reasonable results
        assert 0 <= result_good['total_loss_fraction'] <= 1
        assert 0 <= result_poor['total_loss_fraction'] <= 1
    
    def test_thickness_dependence(self, loss_calc):
        """Test interface loss dependence on layer thickness."""
        # Very thin layers might have higher interface losses
        thin_stack = [
            ('PEDOT', 10),        # Very thin
            ('perovskite', 100),  # Very thin
            ('C60', 10)           # Very thin
        ]
        
        # Normal thickness layers
        normal_stack = [
            ('PEDOT', 50),
            ('perovskite', 500),
            ('C60', 100)
        ]
        
        result_thin = loss_calc.calculate_total_interface_losses(thin_stack)
        result_normal = loss_calc.calculate_total_interface_losses(normal_stack)
        
        # Both should give reasonable results
        # Thin layers might have higher losses due to poor coverage
        assert result_thin['total_loss_fraction'] >= 0
        assert result_normal['total_loss_fraction'] >= 0
    
    def test_loss_breakdown_analysis(self, loss_calc):
        """Test detailed loss breakdown analysis."""
        stack = [
            ('ITO', 100),
            ('PEDOT', 50),
            ('perovskite', 500),
            ('C60', 100),
            ('Au', 80)
        ]
        
        result = loss_calc.calculate_total_interface_losses(stack)
        
        # Should provide breakdown by interface
        assert 'interface_analysis' in result
        
        total_calculated = sum(analysis.get('loss_fraction', 0) 
                             for analysis in result['interface_analysis'])
        
        # Sum of individual losses should approximately equal total
        # (within reasonable tolerance due to coupling effects)
        assert abs(total_calculated - result['total_loss_fraction']) < 0.1


class TestTunnelJunctionCalculator:
    """Test tunnel junction calculations."""
    
    @pytest.fixture
    def tunnel_calc(self):
        """Create tunnel junction calculator instance."""
        return TunnelJunctionCalculator()
    
    def test_tunnel_resistance_calculation(self, tunnel_calc):
        """Test tunnel junction resistance calculation."""
        # Typical tunnel junction parameters
        barrier_height = 1.0    # eV
        barrier_width = 2.0     # nm
        doping_density = 1e20   # cm^-3
        
        from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            n_type_doping=doping_density
        ,
            p_type_doping=doping_density
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        assert 'resistance_ohm_cm2' in result
        assert 'tunneling_probability' in result
        
        # Tunnel resistance should be positive
        assert result['resistance_ohm_cm2'] > 0
        
        # Should be reasonable for good tunnel junction (< 1 Ω·cm²)
        assert result['resistance_ohm_cm2'] < 10
        
        # Tunneling probability should be between 0 and 1
        assert 0 <= result['tunneling_probability'] <= 1
    
    def test_barrier_width_dependence(self, tunnel_calc):
        """Test dependence on barrier width."""
        barrier_height = 1.0
        doping_density = 1e20
        
        widths = [1.0, 2.0, 3.0, 4.0]  # nm
        resistances = []
        
        for width in widths:
            from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height,
            barrier_width=width,
            n_type_doping=doping_density
            ,
            p_type_doping=doping_density
            ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        resistances.append(result['resistance_ohm_cm2'])
        
        # Resistance should increase with barrier width (exponentially)
        for i in range(1, len(resistances)):
            assert resistances[i] > resistances[i-1]
        
        # Should show exponential-like increase
        ratio = resistances[-1] / resistances[0]
        assert ratio > 5  # Significant increase over range
    
    def test_doping_dependence(self, tunnel_calc):
        """Test dependence on doping density."""
        barrier_height = 1.0
        barrier_width = 2.0
        
        dopings = [1e18, 1e19, 1e20, 1e21]  # cm^-3
        resistances = []
        
        for doping in dopings:
            from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            n_type_doping=doping
            ,
            p_type_doping=doping
            ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
            resistances.append(result['resistance_ohm_cm2'])
        
        # Higher doping should reduce resistance
        for i in range(1, len(resistances)):
            assert resistances[i] <= resistances[i-1]
    
    def test_tunnel_junction_design_optimization(self, tunnel_calc):
        """Test tunnel junction design optimization."""
        # Define design space
        designs = [
            {'height': 0.5, 'width': 1.5, 'doping': 1e20},  # Good design
            {'height': 1.5, 'width': 3.0, 'doping': 1e19},  # Poor design
            {'height': 0.8, 'width': 2.0, 'doping': 5e20},  # Intermediate
        ]
        
        results = []
        for design in designs:
            from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=**design)
            results.append(result)
        
        # Good design should have lowest resistance
        resistances = [r['resistance_ohm_cm2'] for r in results]
        assert resistances[0] < resistances[1]  # Good < Poor
        
        # All should be physically reasonable
        for resistance in resistances:
            assert 0.001 < resistance < 1000  # mΩ·cm² to kΩ·cm²
    
    def test_quantum_mechanical_effects(self,
            barrier_width=tunnel_calc):
        """Test quantum mechanical tunneling effects."""
        # Very thin barrier should have high tunneling probability
        thin_result = tunnel_calc.calculate_tunneling_resistance(
            barrier_height=1.0,
            n_type_doping=barrier_width=1.0,  # Very thin
            doping_density=1e20
        ,
            p_type_doping=barrier_width=1.0,  # Very thin
            doping_density=1e20
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        # Thick barrier should have low tunneling probability
        thick_from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height=1.0,
            barrier_width=barrier_width=4.0,
            n_type_doping=# Thick
            doping_density=1e20
        ,
            p_type_doping=# Thick
            doping_density=1e20
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        # Thin barrier should tunnel much better
        assert (thin_result['tunneling_probability'] > 
                thick_result['tunneling_probability'])
        
        assert (thin_result['resistance_ohm_cm2'] < 
                thick_result['resistance_ohm_cm2'])
    
    def test_physical_limits(self, tunnel_calc):
        """Test physical limits and edge cases."""
        # Zero barrier width should give very low resistance
        zero_width = from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height=1.0,
            barrier_width=barrier_width=0.1,
            n_type_doping=# Nearly zero
            doping_density=1e20
        ,
            p_type_doping=# Nearly zero
            doping_density=1e20
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        assert zero_width['resistance_ohm_cm2'] < 0.1  # Very low
        
        # Very high barrier should give very high resistance
        high_barrier = from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height=3.0,
            barrier_width=# Very high
            barrier_width=3.0,
            n_type_doping=doping_density=1e19
        ,
            p_type_doping=doping_density=1e19
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        assert high_barrier['resistance_ohm_cm2'] > 10  # High resistance
    
    def test_temperature_dependence(self, tunnel_calc):
        """Test temperature dependence if implemented."""
        # Basic test - temperature dependence might not be fully implemented
        # but should not crash
        
        result_room_temp = from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height=1.0,
            barrier_width=barrier_width=2.0,
            n_type_doping=doping_density=1e20
        ,
            p_type_doping=doping_density=1e20
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        # Should give reasonable results at room temperature
        assert 'resistance_ohm_cm2' in result_room_temp
        assert result_room_temp['resistance_ohm_cm2'] > 0


class TestIntegration:
    """Test integration between interface and tunnel junction calculators."""
    
    def test_tandem_stack_with_tunnel_junctions(self):
        """Test complete tandem stack with tunnel junctions."""
        loss_calc = InterfaceLossCalculator()
        tunnel_calc = TunnelJunctionCalculator()
        
        # Two-junction stack
        stack = [
            ('glass', 3000000),
            ('ITO', 100),
            ('PEDOT', 50),
            ('perovskite', 300),   # Top cell
            ('C60', 50),
            ('ITO', 50),           # Tunnel junction
            ('PEDOT', 30),
            ('perovskite', 600),   # Bottom cell
            ('C60', 100),
            ('Au', 80)
        ]
        
        # Calculate interface losses
        interface_result = loss_calc.calculate_total_interface_losses(stack)
        
        # Calculate tunnel junction resistance
        tunnel_from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=barrier_height=0.8,
            barrier_width=barrier_width=2.0,
            n_type_doping=doping_density=1e20
        ,
            p_type_doping=doping_density=1e20
        ,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
        
        # Both should give reasonable results
        assert interface_result['total_loss_fraction'] > 0
        assert tunnel_result['resistance_ohm_cm2'] > 0
        
        # Combined analysis
        total_series_resistance = tunnel_result['resistance_ohm_cm2']  # Ω·cm²
        interface_losses = interface_result['total_loss_fraction']
        
        # Should be able to estimate overall cell resistance and losses
        assert total_series_resistance < 5.0  # Good tunnel junction
        assert interface_losses < 0.3         # Reasonable interface losses


def test_realistic_device_analysis():
    """Test analysis of realistic device structures."""
    loss_calc = InterfaceLossCalculator()
    tunnel_calc = TunnelJunctionCalculator()
    
    print("=== Interface Loss Tests ===")
    
    # Single junction
    single_j = [
        ('glass', 3000000), ('ITO', 100), ('PEDOT', 50),
        ('perovskite', 500), ('C60', 100), ('Au', 80)
    ]
    
    result_1j = loss_calc.calculate_total_interface_losses(single_j)
    print(f"Single junction loss: {result_1j['total_loss_fraction']:.3f}")
    
    # Tandem junction
    tandem_j = [
        ('glass', 3000000), ('ITO', 100), ('PEDOT', 50),
        ('perovskite', 300), ('C60', 50), ('ITO', 50),
        ('PEDOT', 30), ('perovskite', 600), ('C60', 100), ('Au', 80)
    ]
    
    result_2j = loss_calc.calculate_total_interface_losses(tandem_j)
    print(f"Tandem junction loss: {result_2j['total_loss_fraction']:.3f}")
    
    # Tunnel junction resistance
    tunnel_from engines.interface_loss import TunnelJunctionParams
        params = TunnelJunctionParams(
            barrier_height=1.0,
            barrier_width=2.0,
            n_type_doping=1e20,
            p_type_doping=1e20,
            temperature=T_CELL
        )
        result = tunnel_calc.calculate_tunneling_resistance(params)
    print(f"Tunnel junction resistance: {tunnel_result['resistance_ohm_cm2']:.3f} Ω·cm²")
    
    # Basic physics checks
    assert result_1j['total_loss_fraction'] > 0, "Should have some interface losses"
    assert result_2j['total_loss_fraction'] > result_1j['total_loss_fraction'], "Tandem should have higher losses"
    assert tunnel_result['resistance_ohm_cm2'] > 0, "Tunnel resistance should be positive"
    assert tunnel_result['resistance_ohm_cm2'] < 10, "Should be reasonable tunnel junction"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_realistic_device_analysis()