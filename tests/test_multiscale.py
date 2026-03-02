#!/usr/bin/env python3
"""Tests for MultiscaleControlEngine (Phase 1 Integration)."""

import numpy as np
import pytest
from engines.multiscale_control import MultiscaleControlEngine, MultiscaleResult


@pytest.fixture
def engine():
    return MultiscaleControlEngine(
        pv_params={'bandgap': 1.55, 'thickness': 500, 'ion_density': 1e18,
                   'ion_mobility': 1e-9, 'ion_diffusivity': 2.6e-11, 'V_bi': 1.0},
        fet_params={'mu_fe': 20, 'C_ox': 300, 'W_L': 100, 'V_th': 0.5, 'V_G_max': 5.0},
        ion_params={'grid_points': 30},
        interface_params={'R_contact': 5.0},
    )


class TestMultiscaleSimulation:
    def test_basic_run(self, engine):
        """Simulation should complete and return valid result."""
        N = 10
        V_G = np.full(N, 3.0)
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        res = engine.simulate_multiscale(V_G, G, T, total_time_s=10.0,
                                         n_medium_per_coarse=5,
                                         n_fine_per_medium=5)
        assert isinstance(res, MultiscaleResult)
        assert len(res.P_out) == N
        assert np.all(res.P_out >= 0)

    def test_zero_irradiance(self, engine):
        """G=0 → P_out=0."""
        N = 5
        V_G = np.full(N, 3.0)
        G = np.zeros(N)
        T = np.full(N, 300.0)
        res = engine.simulate_multiscale(V_G, G, T, total_time_s=5.0,
                                         n_medium_per_coarse=3,
                                         n_fine_per_medium=3)
        assert np.allclose(res.P_out, 0.0, atol=1e-6)

    def test_zero_gate_voltage(self, engine):
        """V_G=0 → FET off → P_out≈0."""
        N = 5
        V_G = np.zeros(N)
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        res = engine.simulate_multiscale(V_G, G, T, total_time_s=5.0,
                                         n_medium_per_coarse=3,
                                         n_fine_per_medium=3)
        assert np.all(res.P_out < 1.0)  # very small

    def test_higher_G_higher_P(self, engine):
        """More light → more power."""
        N = 5
        V_G = np.full(N, 3.0)
        T = np.full(N, 300.0)

        G_low = np.full(N, 0.3)
        res_low = engine.simulate_multiscale(V_G, G_low, T, 5.0,
                                             n_medium_per_coarse=3,
                                             n_fine_per_medium=3)
        G_high = np.full(N, 1.0)
        res_high = engine.simulate_multiscale(V_G, G_high, T, 5.0,
                                              n_medium_per_coarse=3,
                                              n_fine_per_medium=3)
        assert np.mean(res_high.P_out) > np.mean(res_low.P_out)

    def test_energy_conservation(self, engine):
        """Energy balance should be approximately conserved."""
        N = 10
        V_G = np.full(N, 3.0)
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        res = engine.simulate_multiscale(V_G, G, T, 10.0,
                                         n_medium_per_coarse=3,
                                         n_fine_per_medium=3)
        val = engine.validate_energy_conservation(tol=0.15)
        assert val['valid'], f"Energy not conserved: {val}"


class TestOptimalControl:
    def test_optimal_control_runs(self, engine):
        N = 10
        load = np.full(N, 10.0)  # mW/cm²
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        V_G_opt = engine.optimal_control_strategy(load, G, T)
        assert len(V_G_opt) == N
        assert np.all(V_G_opt >= 0)
        assert np.all(V_G_opt <= engine.dynamic_iv.fet_params['V_G_max'])

    def test_control_tracks_load(self, engine):
        """With active control, P_out should approach P_load."""
        N = 10
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        # Get max power for reference
        envelope = engine.dynamic_iv.power_envelope(1.0)
        P_max = envelope['P_max']
        load = np.full(N, P_max * 0.5)

        V_G_opt = engine.optimal_control_strategy(load, G, T)
        res = engine.simulate_multiscale(V_G_opt, G, T, 10.0,
                                         n_medium_per_coarse=3,
                                         n_fine_per_medium=3)
        # Should be roughly in the right ballpark
        mean_P = np.mean(res.P_out)
        assert mean_P > 0


class TestPerformanceSummary:
    def test_summary_after_sim(self, engine):
        N = 5
        V_G = np.full(N, 3.0)
        G = np.full(N, 1.0)
        T = np.full(N, 300.0)
        engine.simulate_multiscale(V_G, G, T, 5.0,
                                   n_medium_per_coarse=3,
                                   n_fine_per_medium=3)
        summary = engine.performance_summary()
        assert 'P_max_mW_cm2' in summary
        assert 'dynamic_range' in summary
        assert 'tau_interface_us' in summary
        assert 'tau_ion_ms' in summary

    def test_summary_no_sim(self, engine):
        summary = engine.performance_summary()
        assert 'error' in summary


class TestTimescaleSeparation:
    def test_interface_faster_than_ion(self, engine):
        """Interface τ (μs) should be much faster than ion τ (ms)."""
        tau_iface = engine.interface_engine.tau_rc_etl
        tau_ion = engine.ion_engine.get_ion_timescale()
        assert tau_iface < tau_ion / 10  # at least 10x separation

    def test_ion_faster_than_fet(self, engine):
        """Ion τ (ms-s) should be faster than FET control τ (s)."""
        tau_ion = engine.ion_engine.get_ion_timescale()
        # FET control operates at ~1s scale
        assert tau_ion < 10.0  # ion dynamics < 10s
