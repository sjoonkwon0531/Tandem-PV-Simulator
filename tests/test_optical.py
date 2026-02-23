"""
Test module for optical transfer matrix method (TMM) calculations.

Tests light absorption, reflection, and transmission through
multi-layer photovoltaic stacks.
"""

import pytest
import numpy as np
import warnings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.optical_tmm import TransferMatrixCalculator

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestTransferMatrixCalculator:
    """Test transfer matrix optical calculations."""
    
    @pytest.fixture
    def tmm_calc(self):
        """Create TMM calculator instance."""
        return TransferMatrixCalculator()
    
    def test_single_layer_absorption(self, tmm_calc):
        """Test absorption in a single active layer."""
        # Simple stack: air / active_layer / air
        stack = [
            ('glass', 1000000),  # Thick substrate (transparent)
            ('perovskite', 500)  # Active layer
        ]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Should have reasonable absorption
        assert 'layer_absorption' in result
        assert len(result['layer_absorption']) == 2
        assert result['total_absorption'] > 0.3  # At least 30% absorbed
        assert result['total_absorption'] <= 1.0
        
        # Perovskite layer should absorb most light
        perovskite_absorption = result['layer_absorption'][1]  # Second layer
        assert perovskite_absorption > 0.3
    
    def test_single_layer_reflectance(self, tmm_calc):
        """Test reflectance calculation."""
        # Glass substrate only
        stack = [('glass', 1000000)]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Glass should have some reflection (4-8% typical)
        assert 'total_reflection' in result
        assert 0.03 < result['total_reflection'] < 0.12
    
    def test_beer_lambert_consistency(self, tmm_calc):
        """Test consistency with Beer-Lambert law for thick absorbing layers."""
        # Single absorbing layer with different thicknesses
        thicknesses = [100, 200, 400]  # nm
        absorptions = []
        
        for thickness in thicknesses:
            stack = [('perovskite', thickness)]
            result = tmm_calc.calculate_stack_absorption(stack)
            absorptions.append(result['total_absorption'])
        
        # Thicker layers should absorb more (monotonic increase)
        assert absorptions[1] > absorptions[0]
        assert absorptions[2] > absorptions[1]
        
        # Should approach Beer-Lambert behavior
        # For same material, A = 1 - exp(-αd), so more absorption with thickness
        # But not exactly exponential due to interference effects
    
    def test_anti_reflection_coating(self, tmm_calc):
        """Test effect of anti-reflection coating."""
        # Without AR coating
        stack_no_ar = [
            ('glass', 1000000),
            ('perovskite', 500)
        ]
        
        # With AR coating (TiO2 layer)
        stack_with_ar = [
            ('glass', 1000000),
            ('TiO2', 80),      # Quarter-wave AR coating
            ('perovskite', 500)
        ]
        
        result_no_ar = tmm_calc.calculate_stack_absorption(stack_no_ar)
        result_with_ar = tmm_calc.calculate_stack_absorption(stack_with_ar)
        
        # AR coating should reduce reflection and increase absorption
        assert result_with_ar['total_reflection'] < result_no_ar['total_reflection']
        # Note: This might not always hold due to wavelength averaging
    
    def test_multilayer_photocurrent_conservation(self, tmm_calc):
        """Test photocurrent conservation in multi-layer stacks."""
        # Complete tandem stack
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
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Energy conservation: R + A + T = 1
        total = (result['total_reflection'] + 
                result['total_absorption'] + 
                result['total_transmission'])
        
        assert abs(total - 1.0) < 0.05  # Within 5% (numerical tolerance)
        
        # Should have multiple active layers with absorption
        active_layers = [i for i, (name, _) in enumerate(stack) 
                        if 'perovskite' in name.lower()]
        
        assert len(active_layers) == 2
        for layer_idx in active_layers:
            assert result['layer_absorption'][layer_idx] > 0.1
    
    def test_wavelength_integration(self, tmm_calc):
        """Test wavelength integration with solar spectrum."""
        stack = [('perovskite', 500)]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Should integrate over AM1.5G spectrum
        assert 'weighted_absorption' in result or 'total_absorption' in result
        
        # Absorption should be reasonable for perovskite bandgap
        assert result['total_absorption'] > 0.5
        assert result['total_absorption'] < 0.95
    
    def test_complex_refractive_index_handling(self, tmm_calc):
        """Test handling of complex refractive indices."""
        # Metal layer should have high extinction coefficient
        stack = [
            ('perovskite', 500),
            ('Au', 100)  # Metallic layer
        ]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Gold should absorb light (parasitic absorption)
        au_layer_idx = 1
        assert result['layer_absorption'][au_layer_idx] > 0.01
        
        # Should have low transmission due to metal
        assert result['total_transmission'] < 0.1
    
    def test_interference_effects(self, tmm_calc):
        """Test optical interference in thin film stacks."""
        # Two identical stacks with different spacer thickness
        base_stack = [
            ('glass', 1000000),
            ('perovskite', 400)
        ]
        
        stack_thin_spacer = [
            ('glass', 1000000),
            ('TiO2', 50),      # Thin spacer
            ('perovskite', 400)
        ]
        
        stack_thick_spacer = [
            ('glass', 1000000),
            ('TiO2', 150),     # Thick spacer
            ('perovskite', 400)
        ]
        
        result_base = tmm_calc.calculate_stack_absorption(base_stack)
        result_thin = tmm_calc.calculate_stack_absorption(stack_thin_spacer)
        result_thick = tmm_calc.calculate_stack_absorption(stack_thick_spacer)
        
        # Different spacer thicknesses should give different absorption
        # (due to interference effects)
        abs_diff_thin = abs(result_thin['total_absorption'] - result_base['total_absorption'])
        abs_diff_thick = abs(result_thick['total_absorption'] - result_base['total_absorption'])
        
        # Should see some difference due to interference
        assert abs_diff_thin > 0.01 or abs_diff_thick > 0.01
    
    def test_transparent_conductor_optimization(self, tmm_calc):
        """Test optimization of transparent conductor thickness."""
        thicknesses = [80, 100, 120, 150]  # nm
        absorptions = []
        reflections = []
        
        for thickness in thicknesses:
            stack = [
                ('glass', 1000000),
                ('ITO', thickness),
                ('perovskite', 500)
            ]
            
            result = tmm_calc.calculate_stack_absorption(stack)
            absorptions.append(result['total_absorption'])
            reflections.append(result['total_reflection'])
        
        # Should see variation with ITO thickness
        absorption_range = max(absorptions) - min(absorptions)
        assert absorption_range > 0.02  # At least 2% variation
    
    def test_stack_absorption_spectral_response(self, tmm_calc):
        """Test spectral response of absorption."""
        stack = [
            ('perovskite', 500),  # Single active layer
        ]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Should have wavelength-dependent information
        # Even if integrated, should capture key physics
        assert result['total_absorption'] > 0.4
        
        # Perovskite should absorb strongly above bandgap
        # (This is implicit in the spectrum-averaged result)
    
    def test_error_handling(self, tmm_calc):
        """Test error handling for invalid inputs."""
        # Empty stack
        with pytest.raises((ValueError, IndexError)):
            tmm_calc.calculate_stack_absorption([])
        
        # Invalid thickness
        result = tmm_calc.calculate_stack_absorption([('perovskite', -100)])
        # Should handle gracefully (might clamp to positive)
        
        # Unknown material (should use defaults)
        result = tmm_calc.calculate_stack_absorption([('unknown_material', 500)])
        assert 'total_absorption' in result
    
    def test_photocurrent_calculation(self, tmm_calc):
        """Test photocurrent density calculation."""
        stack = [('perovskite', 500)]
        
        result = tmm_calc.calculate_stack_absorption(stack)
        
        # Should be able to calculate photocurrent from absorption
        if 'photocurrent_density' in result:
            # Reasonable photocurrent for perovskite
            assert result['photocurrent_density'] > 10  # mA/cm²
            assert result['photocurrent_density'] < 30  # mA/cm²
        
        # Total absorption should convert to reasonable current
        # This is approximated based on absorbed photon flux


class TestOpticalConstants:
    """Test optical constants database and interpolation."""
    
    def test_refractive_index_retrieval(self):
        """Test retrieval of refractive index data."""
        from config import MATERIAL_DB
        
        tmm_calc = TransferMatrixCalculator()
        
        # Common materials should have optical data
        test_materials = ['perovskite', 'ITO', 'glass']
        
        for material in test_materials:
            if material in MATERIAL_DB:
                props = MATERIAL_DB[material]
                
                # Should have refractive index information
                assert 'refractive_index' in props or 'n' in props
                
                if 'refractive_index' in props:
                    n = props['refractive_index']
                    assert isinstance(n, (int, float, complex))
                    assert n.real > 0  # Physical requirement


def test_integration_with_material_database():
    """Test integration with material database."""
    from config import MATERIAL_DB
    
    tmm_calc = TransferMatrixCalculator()
    
    # Build stack from database materials
    available_materials = list(MATERIAL_DB.keys())[:5]  # First 5 materials
    
    stack = []
    for material in available_materials:
        if MATERIAL_DB[material].get('thickness_range'):
            thickness = MATERIAL_DB[material]['thickness_range'][0]
        else:
            thickness = 500  # Default
        
        stack.append((material, thickness))
    
    # Should be able to calculate absorption
    result = tmm_calc.calculate_stack_absorption(stack)
    assert 'total_absorption' in result
    assert 0 <= result['total_absorption'] <= 1


if __name__ == "__main__":
    # Run basic tests
    tmm = TransferMatrixCalculator()
    
    print("=== Optical TMM Tests ===")
    
    # Test single layer
    stack_1 = [('perovskite', 500)]
    result_1 = tmm.calculate_stack_absorption(stack_1)
    print(f"Single layer absorption: {result_1['total_absorption']:.3f}")
    print(f"Single layer reflection: {result_1['total_reflection']:.3f}")
    
    # Test bilayer
    stack_2 = [('glass', 1000000), ('perovskite', 500)]
    result_2 = tmm.calculate_stack_absorption(stack_2)
    print(f"Bilayer absorption: {result_2['total_absorption']:.3f}")
    print(f"Bilayer reflection: {result_2['total_reflection']:.3f}")
    
    # Conservation check
    total = result_2['total_reflection'] + result_2['total_absorption'] + result_2['total_transmission']
    print(f"Energy conservation (R+A+T): {total:.3f}")
    
    # Basic physics checks
    assert result_1['total_absorption'] > 0.3, "Perovskite should absorb strongly"
    assert result_2['total_reflection'] > result_1.get('total_reflection', 0), "Glass increases reflection"
    assert abs(total - 1.0) < 0.1, "Energy conservation"
    
    print("All tests passed!")