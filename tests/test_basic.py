#!/usr/bin/env python3
"""
Basic tests for Tandem PV Simulator v2.0
========================================

Simple tests to verify core functionality works.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_import():
    """Test that configuration module imports correctly"""
    from config import MATERIAL_DB, ABX3_SPACE
    
    assert MATERIAL_DB is not None
    assert ABX3_SPACE is not None

def test_material_database():
    """Test material database basic functionality"""
    from config import MATERIAL_DB
    
    # Test Track A materials
    track_a_materials = MATERIAL_DB.list_materials('A')
    assert len(track_a_materials) > 0
    assert 'c-Si' in track_a_materials
    assert 'GaAs' in track_a_materials
    
    # Test Track B materials
    track_b_materials = MATERIAL_DB.list_materials('B')
    assert len(track_b_materials) > 0
    assert 'MAPbI3' in track_b_materials
    
    # Test material properties
    si_props = MATERIAL_DB.get_material('c-Si', 'A')
    assert si_props['bandgap'] == 1.12
    assert si_props['type'] == 'indirect'

def test_abx3_composition_space():
    """Test ABX₃ composition space functionality"""
    from config import ABX3_SPACE
    
    # Test tolerance factor calculation
    a_comp = {'MA': 1.0, 'FA': 0.0, 'Cs': 0.0, 'Rb': 0.0}
    b_comp = {'Pb': 1.0, 'Sn': 0.0, 'Ge': 0.0}
    x_comp = {'I': 1.0, 'Br': 0.0, 'Cl': 0.0}
    
    tolerance_factor = ABX3_SPACE.calculate_tolerance_factor(a_comp, b_comp, x_comp)
    assert 0.8 < tolerance_factor < 1.2  # Reasonable range for perovskites

def test_ml_bandgap_predictor():
    """Test ML bandgap predictor basic functionality"""
    try:
        from engines.ml_bandgap import ML_BANDGAP_PREDICTOR
        
        # Test that predictor can be initialized
        assert ML_BANDGAP_PREDICTOR is not None
        
        # Test composition creation
        composition = {
            'A_MA': 1.0, 'A_FA': 0.0, 'A_Cs': 0.0, 'A_Rb': 0.0,
            'B_Pb': 1.0, 'B_Sn': 0.0, 'B_Ge': 0.0,
            'X_I': 1.0, 'X_Br': 0.0, 'X_Cl': 0.0
        }
        
        # Test feature extraction
        features = ML_BANDGAP_PREDICTOR.extract_features(composition)
        assert len(features) == 20  # Should extract 20 features
        
    except ImportError:
        pytest.skip("ML bandgap predictor not available")

def test_interface_calculator():
    """Test interface energy calculator basic functionality"""
    try:
        from engines.interface_energy import INTERFACE_CALCULATOR
        
        # Test tolerance factor calculation (single dict with site-prefixed keys)
        composition = {
            'A_MA': 1.0, 'A_FA': 0.0, 'A_Cs': 0.0, 'A_Rb': 0.0,
            'B_Pb': 1.0, 'B_Sn': 0.0, 'B_Ge': 0.0,
            'X_I': 1.0, 'X_Br': 0.0, 'X_Cl': 0.0,
        }
        
        tolerance_factor = INTERFACE_CALCULATOR.calculate_tolerance_factor(composition)
        assert isinstance(tolerance_factor, (int, float))
        assert tolerance_factor >= 0
        
    except ImportError:
        pytest.skip("Interface calculator not available")

def test_solar_spectrum_calculator():
    """Test solar spectrum calculator basic functionality"""
    try:
        from engines.solar_spectrum import SOLAR_SPECTRUM_CALCULATOR
        
        # Test spectrum calculation
        spectrum = SOLAR_SPECTRUM_CALCULATOR.calculate_spectrum(air_mass=1.5)
        
        assert spectrum.total_irradiance > 0
        assert len(spectrum.wavelengths) > 0
        assert len(spectrum.irradiance) == len(spectrum.wavelengths)
        
        # Test solar position calculation
        zenith, azimuth = SOLAR_SPECTRUM_CALCULATOR.solar_position(
            latitude=37.5, day_of_year=172, hour=12
        )
        
        assert 0 <= zenith <= 90  # Zenith angle should be reasonable at noon
        assert 0 <= azimuth <= 360  # Azimuth in valid range
        
    except ImportError:
        pytest.skip("Solar spectrum calculator not available")

def test_am15g_spectrum():
    """Test AM1.5G spectrum generation"""
    from config import get_am15g_spectrum
    
    wavelengths = np.linspace(300, 1200, 100)
    spectrum = get_am15g_spectrum(wavelengths)
    
    assert len(spectrum) == len(wavelengths)
    assert np.all(spectrum >= 0)  # All values should be non-negative
    
    # Total power should be reasonable
    total_power = np.trapezoid(spectrum, wavelengths)
    assert 800 < total_power < 1200  # Rough range for integrated spectrum

def test_mixed_halide_properties():
    """Test mixed halide perovskite property calculation"""
    from config import MATERIAL_DB
    
    # Test 50:50 I:Br mixture
    composition = {'I': 0.5, 'Br': 0.5}
    mixed_props = MATERIAL_DB.get_mixed_halide_properties(composition)
    
    assert 'bandgap' in mixed_props
    assert 'bowing_correction' in mixed_props
    assert mixed_props['bandgap'] > 0
    
    # Should be between pure I and pure Br bandgaps
    assert 1.5 < mixed_props['bandgap'] < 2.5

if __name__ == "__main__":
    pytest.main([__file__])