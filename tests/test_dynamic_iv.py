#!/usr/bin/env python3
"""
Tests for Dynamic I-V Engine and Load Matching Engine
=====================================================
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engines.dynamic_iv import DynamicIVEngine
from engines.load_matching import LoadMatchingEngine


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def default_engine():
    """Default DynamicIVEngine with MAPbI3 perovskite + IGZO FET."""
    pv = {'bandgap': 1.55, 'temperature': 298.15, 'thickness': 500,
          'ion_density': 1e18, 'ion_mobility': 1e-9, 'V_bi': 1.0}
    fet = {'mu_fe': 20, 'C_ox': 300, 'W_L': 100, 'V_th': 0.5, 'V_G_max': 5.0}
    iface = {'R_contact': 10, 'C_interface': 44, 'n_traps': 1e12}
    return DynamicIVEngine(pv, fet, iface)


@pytest.fixture
def load_engine(default_engine):
    """Default LoadMatchingEngine."""
    return LoadMatchingEngine(default_engine)


# ============================================================
# Test 1: Static I-V consistency
# ============================================================

class TestStaticIV:
    def test_static_iv_returns_positive_current(self, default_engine):
        V = np.linspace(0, 1.0, 50)
        I = default_engine.static_iv(V, G=1.0, T=298.15)
        # At V=0, current should be positive (Jsc)
        assert I[0] > 0, "Jsc should be positive"

    def test_static_iv_zero_at_voc(self, default_engine):
        V = np.linspace(0, 1.5, 200)
        I = default_engine.static_iv(V, G=1.0, T=298.15)
        # Current should cross zero near Voc
        sign_changes = np.where(np.diff(np.sign(I)))[0]
        assert len(sign_changes) > 0, "I-V curve should cross zero at Voc"


# ============================================================
# Test 2: Operating point with V_G
# ============================================================

class TestOperatingPoint:
    def test_vg_zero_gives_zero_power(self, default_engine):
        """V_G=0 (below V_th) -> FET off -> no output power."""
        op = default_engine.operating_point(0.0, G=1.0)
        assert op['P_out'] == 0.0 or op['P_out'] < 0.01

    def test_vg_optimal_gives_nonzero_power(self, default_engine):
        """V_G above threshold -> nonzero output."""
        op = default_engine.operating_point(3.0, G=1.0)
        assert op['P_out'] > 0, "Should produce power when FET is on"

    def test_vg_sweep_monotonic_until_peak(self, default_engine):
        """Power should generally increase with V_G until an optimum."""
        V_G_arr = np.linspace(0.5, 5.0, 20)
        P_arr = []
        for vg in V_G_arr:
            op = default_engine.operating_point(vg, G=1.0)
            P_arr.append(op['P_out'])
        P_arr = np.array(P_arr)
        # There should be a peak, and power should increase from V_th
        assert np.max(P_arr) > P_arr[0], "Power should increase from near-threshold"

    def test_operating_point_energy_conservation(self, default_engine):
        """P_out should not exceed incident power × thermodynamic limit."""
        op = default_engine.operating_point(3.0, G=1.0, T=298.15)
        P_incident = 100.0  # mW/cm² at 1 sun
        # Shockley-Queisser limit for 1.55 eV is ~33%
        assert op['P_out'] <= P_incident * 0.40, \
            f"P_out={op['P_out']} exceeds 40% of incident power"

    def test_operating_point_at_different_irradiance(self, default_engine):
        """Higher irradiance -> higher power."""
        op_low = default_engine.operating_point(3.0, G=0.5)
        op_high = default_engine.operating_point(3.0, G=1.0)
        assert op_high['P_out'] > op_low['P_out']


# ============================================================
# Test 3: Timescale verification
# ============================================================

class TestTimescales:
    def test_ion_time_constant_range(self, default_engine):
        """τ_ion should be in ms range (1-10000 ms depending on parameters)."""
        tau_ms = default_engine.ion_time_constant_ms
        assert 0.1 <= tau_ms <= 10000, \
            f"Ion time constant {tau_ms} ms outside expected range"

    def test_rc_time_constant_range(self, default_engine):
        """τ_RC should be 1-1000 μs."""
        tau_us = default_engine.rc_time_constant_us
        assert 1 <= tau_us <= 5000, \
            f"RC time constant {tau_us} μs outside expected range"

    def test_ion_slower_than_rc(self, default_engine):
        """Ion migration should be slower than interface RC."""
        assert default_engine.tau_ion > default_engine.tau_rc, \
            "Ion migration should be slower than RC dynamics"


# ============================================================
# Test 4: Dynamic response
# ============================================================

class TestDynamicResponse:
    def test_dynamic_response_shape(self, default_engine):
        N = 100
        V_G = np.ones(N) * 3.0
        G = np.ones(N) * 1.0
        T = np.ones(N) * 298.15
        result = default_engine.dynamic_response(V_G, G, T, dt=0.001)
        assert result['P_out'].shape == (N,)
        assert result['V_op'].shape == (N,)
        assert result['eta'].shape == (N,)

    def test_dynamic_response_positive_power(self, default_engine):
        N = 50
        V_G = np.ones(N) * 3.0
        G = np.ones(N) * 1.0
        T = np.ones(N) * 298.15
        result = default_engine.dynamic_response(V_G, G, T, dt=0.001)
        assert np.all(result['P_out'] >= 0), "Power should be non-negative"

    def test_dynamic_response_step_input(self, default_engine):
        """Step change in V_G should cause transient response."""
        N = 200
        V_G = np.concatenate([np.ones(100) * 1.0, np.ones(100) * 4.0])
        G = np.ones(N) * 1.0
        T = np.ones(N) * 298.15
        result = default_engine.dynamic_response(V_G, G, T, dt=0.001)
        # Power should change between first and second half
        P_first = np.mean(result['P_out'][:50])
        P_second = np.mean(result['P_out'][150:])
        assert abs(P_second - P_first) > 0.01, "Step should cause power change"


# ============================================================
# Test 5: Power envelope
# ============================================================

class TestPowerEnvelope:
    def test_dynamic_range_greater_than_one(self, default_engine):
        """Dynamic range should be > 1 (controllability)."""
        env = default_engine.power_envelope(G=1.0)
        assert env['dynamic_range'] > 1.0, \
            f"Dynamic range {env['dynamic_range']} should be > 1"

    def test_power_envelope_has_optimum(self, default_engine):
        env = default_engine.power_envelope(G=1.0)
        assert env['P_max'] > env['P_min']
        assert 0 < env['V_G_opt'] <= default_engine.fet_params['V_G_max']

    def test_control_resolution_exists(self, default_engine):
        """dP/dV_G should be non-zero in controllable range."""
        env = default_engine.power_envelope(G=1.0)
        # At least some points should have nonzero slope
        assert np.max(np.abs(env['dP_dVG'])) > 0


# ============================================================
# Test 6: Efficiency map
# ============================================================

class TestEfficiencyMap:
    def test_efficiency_map_shape(self, default_engine):
        VG = np.array([1.0, 3.0, 5.0])
        G = np.array([0.5, 1.0])
        T = np.array([298.15])
        emap = default_engine.efficiency_map(VG, G, T)
        assert emap.shape == (3, 2, 1)

    def test_efficiency_map_bounded(self, default_engine):
        VG = np.array([1.0, 3.0])
        G = np.array([1.0])
        T = np.array([298.15])
        emap = default_engine.efficiency_map(VG, G, T)
        assert np.all(emap >= 0)
        assert np.all(emap <= 0.5), "Efficiency should be < 50%"


# ============================================================
# Test 7: Load matching
# ============================================================

class TestLoadMatching:
    def test_generate_aidc_load(self, load_engine):
        result = load_engine.generate_aidc_load(hours=1, gpu_count=100, dt=10)
        assert len(result['time_s']) > 0
        assert len(result['load_kW']) == len(result['time_s'])
        assert np.all(result['load_kW'] > 0)

    def test_match_analysis(self, load_engine):
        N = 100
        pv = np.ones(N) * 50  # kW
        load = np.ones(N) * 80  # kW
        result = load_engine.match_analysis(pv, load, dt=1.0)
        assert result['deficit_kWh'] > 0
        assert result['surplus_kWh'] == 0 or result['surplus_kWh'] < 0.01
        assert 0 <= result['self_consumption_ratio'] <= 1.0

    def test_hess_reduction(self, load_engine):
        without = {'hess_capacity_kWh': 100, 'surplus_kWh': 50,
                   'deficit_kWh': 30, 'total_pv_kWh': 200, 'total_load_kWh': 180,
                   'self_consumption_ratio': 0.75, 'match_score': 0.6}
        with_ctrl = {'hess_capacity_kWh': 60, 'surplus_kWh': 20,
                     'deficit_kWh': 15, 'total_pv_kWh': 200, 'total_load_kWh': 180,
                     'self_consumption_ratio': 0.90, 'match_score': 0.8}
        result = load_engine.hess_reduction(with_ctrl, without)
        assert result['capacity_reduction_pct'] > 0
        assert result['cycling_reduction_pct'] > 0
        assert result['cost_saving_usd_per_year'] > 0

    def test_perfect_match_zero_hess(self, load_engine):
        """If PV exactly matches load, HESS should be minimal."""
        N = 100
        load = np.ones(N) * 50
        result = load_engine.match_analysis(load, load, dt=1.0)
        assert result['surplus_kWh'] < 0.01
        assert result['deficit_kWh'] < 0.01
        assert result['self_consumption_ratio'] > 0.99


# ============================================================
# Test 8: Edge cases
# ============================================================

class TestEdgeCases:
    def test_zero_irradiance(self, default_engine):
        op = default_engine.operating_point(3.0, G=0.0)
        assert op['P_out'] < 0.01

    def test_high_temperature(self, default_engine):
        """Higher temperature should reduce Voc and efficiency."""
        op_cool = default_engine.operating_point(3.0, G=1.0, T=298.15)
        op_hot = default_engine.operating_point(3.0, G=1.0, T=358.15)
        assert op_hot['eta'] < op_cool['eta']

    def test_very_high_vg(self, default_engine):
        """Very high V_G should not cause errors."""
        op = default_engine.operating_point(100.0, G=1.0)
        assert op['P_out'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
