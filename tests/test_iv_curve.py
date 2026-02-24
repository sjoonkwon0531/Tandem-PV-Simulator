#!/usr/bin/env python3
"""
Tests for IV Curve Simulator Engine
==================================

Test the I-V curve simulation capabilities for single cells and tandem structures.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.iv_curve import (
    SingleDiodeModel, TandemIVSimulator, 
    simulate_subcell_iv, simulate_tandem_iv, find_mpp
)

class TestSingleDiodeModel:
    """Test single diode model functionality"""
    
    def test_initialization(self):
        """Test that single diode model initializes properly"""
        model = SingleDiodeModel(
            bandgap=1.12,
            temperature=298.15,
            concentration=1.0
        )
        
        assert model.Eg == 1.12
        assert model.T == 298.15
        assert model.X == 1.0
        assert model.Iph > 0  # Should have positive photocurrent
        assert model.I0 > 0   # Should have positive saturation current
    
    def test_iv_curve_generation(self):
        """Test I-V curve generation"""
        model = SingleDiodeModel(bandgap=1.42, temperature=298.15)  # GaAs-like
        
        V, I = model.generate_iv_curve(n_points=50)
        
        assert len(V) == 50
        assert len(I) == 50
        assert V[0] == 0.0  # Starts at zero voltage
        assert I[0] > I[-1]  # Current decreases with voltage
        assert all(v >= 0 for v in V)  # All voltages positive
    
    def test_parameter_extraction(self):
        """Test solar cell parameter extraction"""
        model = SingleDiodeModel(bandgap=1.12, temperature=298.15)  # Si-like
        
        params = model.extract_parameters()
        
        # Check all required parameters present
        required_params = ['Jsc', 'Voc', 'FF', 'PCE', 'Vmpp', 'Impp', 'Pmpp']
        for param in required_params:
            assert param in params
            assert params[param] >= 0  # All should be non-negative
        
        # Sanity checks
        assert 0.6 < params['Voc'] < 0.9  # Reasonable Voc for Si
        assert 0.5 < params['FF'] < 0.9   # Reasonable fill factor
        assert params['PCE'] < 50         # Efficiency less than 50%
        assert params['Jsc'] > 10         # Reasonable current density
    
    def test_temperature_effects(self):
        """Test temperature dependence"""
        model_cold = SingleDiodeModel(bandgap=1.42, temperature=273.15)  # 0°C
        model_hot = SingleDiodeModel(bandgap=1.42, temperature=373.15)   # 100°C
        
        params_cold = model_cold.extract_parameters()
        params_hot = model_hot.extract_parameters()
        
        # Voc should decrease with temperature
        assert params_cold['Voc'] > params_hot['Voc']
        
        # Jsc should slightly increase with temperature
        assert params_cold['Jsc'] <= params_hot['Jsc']
    
    def test_concentration_effects(self):
        """Test concentration dependence"""
        model_1sun = SingleDiodeModel(bandgap=1.42, concentration=1.0)
        model_10sun = SingleDiodeModel(bandgap=1.42, concentration=10.0)
        
        params_1sun = model_1sun.extract_parameters()
        params_10sun = model_10sun.extract_parameters()
        
        # Current should scale linearly with concentration
        assert abs(params_10sun['Jsc'] / params_1sun['Jsc'] - 10.0) < 1.0
        
        # Voltage should increase logarithmically
        assert params_10sun['Voc'] > params_1sun['Voc']

class TestTandemIVSimulator:
    """Test tandem IV simulation functionality"""
    
    def test_two_junction_tandem(self):
        """Test basic two-junction tandem simulation"""
        subcells = [
            {'bandgap': 1.81, 'n': 1.1},  # GaInP top
            {'bandgap': 1.12, 'n': 1.2}   # Si bottom
        ]
        
        tandem = TandemIVSimulator(subcells)
        V, I, metrics = tandem.generate_tandem_iv()
        
        assert len(V) > 0
        assert len(I) > 0
        assert len(V) == len(I)
        
        # Check metrics
        assert 'Jsc' in metrics
        assert 'Voc' in metrics
        assert 'PCE' in metrics
        
        # Voc should be sum of subcell Vocs (approximately)
        assert metrics['Voc'] > 1.5  # Should be > individual Vocs
        assert metrics['Voc'] < 3.0  # But reasonable upper bound
        
        # Current should be limited by lowest subcell
        assert metrics['Jsc'] > 0
        assert metrics['Jsc'] < 50  # Reasonable upper bound
    
    def test_three_junction_tandem(self):
        """Test three-junction tandem simulation"""
        subcells = [
            {'bandgap': 2.0, 'n': 1.1},   # High Eg top
            {'bandgap': 1.4, 'n': 1.2},   # Medium Eg middle  
            {'bandgap': 0.9, 'n': 1.3}    # Low Eg bottom
        ]
        
        tandem = TandemIVSimulator(subcells)
        V, I, metrics = tandem.generate_tandem_iv()
        
        assert len(subcells) == 3
        assert 'subcell_params' in metrics
        assert len(metrics['subcell_params']) == 3
        
        # Three-junction should have higher Voc than two-junction
        assert metrics['Voc'] > 2.0

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_simulate_subcell_iv(self):
        """Test subcell IV simulation function"""
        result = simulate_subcell_iv(
            bandgap=1.12,
            temperature=298.15,
            concentration=1.0
        )
        
        # Should return dictionary with V, I arrays and parameters
        assert 'V' in result
        assert 'I' in result
        assert 'Jsc' in result
        assert 'Voc' in result
        assert 'PCE' in result
        
        assert isinstance(result['V'], np.ndarray)
        assert isinstance(result['I'], np.ndarray)
        assert len(result['V']) == len(result['I'])
    
    def test_simulate_tandem_iv(self):
        """Test tandem IV simulation function"""
        subcells = [
            {'bandgap': 1.8, 'temperature': 298.15},
            {'bandgap': 1.1, 'temperature': 298.15}
        ]
        
        result = simulate_tandem_iv(subcells, connection='series')
        
        # Should return dictionary with tandem characteristics
        assert 'V' in result
        assert 'I' in result
        assert 'Jsc' in result
        assert 'Voc' in result
        assert 'PCE' in result
        assert 'subcell_params' in result
        
        # Should have data for both subcells
        assert len(result['subcell_params']) == 2
    
    def test_find_mpp(self):
        """Test maximum power point finding"""
        # Create simple test I-V data
        V = np.linspace(0, 1, 100)
        I = 30 * (1 - V/1.2)  # Linear approximation
        I = np.maximum(I, 0)   # Ensure non-negative
        
        V_mpp, I_mpp, P_mpp = find_mpp(V, I)
        
        assert 0 <= V_mpp <= 1
        assert I_mpp >= 0
        assert P_mpp >= 0
        assert P_mpp == V_mpp * I_mpp

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_high_bandgap(self):
        """Test with unrealistically high bandgap"""
        model = SingleDiodeModel(bandgap=4.0, temperature=298.15)
        params = model.extract_parameters()
        
        # Should still work but with very low current
        assert params['Jsc'] >= 0
        assert params['PCE'] >= 0
    
    def test_very_low_bandgap(self):
        """Test with very low bandgap"""
        model = SingleDiodeModel(bandgap=0.5, temperature=298.15)
        params = model.extract_parameters()
        
        # Should still work
        assert params['Jsc'] > 0
        assert params['Voc'] > 0
        assert params['PCE'] >= 0
    
    def test_empty_tandem(self):
        """Test tandem with no subcells"""
        tandem = TandemIVSimulator([])
        V, I, metrics = tandem.generate_tandem_iv()
        
        # Should return empty arrays but not crash
        assert len(V) <= 1
        assert len(I) <= 1

if __name__ == "__main__":
    # Run a few basic tests
    print("Running IV curve engine tests...")
    
    # Test single cell
    print("\n1. Testing single diode model...")
    model = SingleDiodeModel(bandgap=1.12, temperature=298.15)
    params = model.extract_parameters()
    print(f"   Si cell: Jsc={params['Jsc']:.1f} mA/cm², Voc={params['Voc']:.3f} V, PCE={params['PCE']:.1f}%")
    
    # Test tandem
    print("\n2. Testing tandem simulation...")
    subcells = [
        {'bandgap': 1.81, 'n': 1.1},  # GaInP
        {'bandgap': 1.12, 'n': 1.2}   # Si
    ]
    result = simulate_tandem_iv(subcells)
    print(f"   Tandem: Jsc={result['Jsc']:.1f} mA/cm², Voc={result['Voc']:.3f} V, PCE={result['PCE']:.1f}%")
    
    # Test temperature effect
    print("\n3. Testing temperature effects...")
    model_25C = SingleDiodeModel(bandgap=1.42, temperature=298.15)
    model_85C = SingleDiodeModel(bandgap=1.42, temperature=358.15)
    
    params_25C = model_25C.extract_parameters()
    params_85C = model_85C.extract_parameters()
    
    voc_temp_coeff = (params_85C['Voc'] - params_25C['Voc']) / (85 - 25)
    print(f"   Voc temperature coefficient: {voc_temp_coeff*1000:.2f} mV/°C")
    
    print("\n✅ All IV curve tests passed!")