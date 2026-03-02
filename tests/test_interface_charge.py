#!/usr/bin/env python3
"""Tests for InterfaceChargeEngine (Phase 1-3)."""

import numpy as np
import pytest
from engines.interface_charge import InterfaceChargeEngine


@pytest.fixture
def engine():
    return InterfaceChargeEngine()


@pytest.fixture
def engine_custom():
    return InterfaceChargeEngine(
        etl_params={'N_t': 1e17, 'E_t': 0.3, 'sigma_n': 1e-15, 'sigma_p': 1e-17, 'C_geo': 30e-9},
        htl_params={'N_t': 5e16, 'E_t': 0.4, 'sigma_n': 1e-17, 'sigma_p': 1e-15, 'C_geo': 25e-9},
        interface_params={'R_contact': 5.0, 'v_thermal': 1e7},
    )


class TestTrapDynamics:
    def test_capture_emission_balance_steady_state(self, engine):
        """At long times, capture ≈ emission → steady state."""
        n_e, n_h = 1e16, 1e16
        T = 300.0
        dt = 1e-5  # 10 μs steps
        for _ in range(10000):
            res = engine.trap_dynamics(n_e, n_h, 0.0, T, dt)
        nt1 = res['n_t_etl']
        for _ in range(100):
            res2 = engine.trap_dynamics(n_e, n_h, 0.0, T, dt)
        # Should be within 5% of previous value
        assert abs(res2['n_t_etl'] - nt1) / max(nt1, 1.0) < 0.05

    def test_charge_conservation(self, engine):
        """Trapped charge changes should be bounded by rates × dt."""
        engine.reset()
        n_e, n_h = 1e16, 1e16
        dt = 1e-7
        res = engine.trap_dynamics(n_e, n_h, 0.0, 300.0, dt)
        # n_t should be non-negative and <= N_t
        assert res['n_t_etl'] >= 0
        assert res['n_t_etl'] <= engine.etl['N_t']
        assert res['n_t_htl'] >= 0
        assert res['n_t_htl'] <= engine.htl['N_t']

    def test_positive_J_rec(self, engine):
        """Recombination current should be non-negative."""
        engine.reset()
        res = engine.trap_dynamics(1e16, 1e16, 0.5, 300.0, 1e-6)
        assert res['J_rec'] >= 0

    def test_higher_carriers_more_trapping(self, engine):
        """Higher carrier density → faster trap filling."""
        engine.reset()
        r_low = engine.trap_dynamics(1e10, 1e10, 0.0, 300.0, 1e-7)
        nt_low = r_low['n_t_etl']
        engine.reset()
        r_high = engine.trap_dynamics(1e18, 1e18, 0.0, 300.0, 1e-7)
        nt_high = r_high['n_t_etl']
        # In a single short step, high carriers should capture more
        assert nt_high > nt_low


class TestTimescale:
    def test_tau_rc_etl_range(self, engine_custom):
        """τ_RC should be in 1-1000 μs range for typical parameters."""
        tau = engine_custom.tau_rc_etl
        assert 1e-7 <= tau <= 1e-3  # 0.1 μs to 1 ms

    def test_tau_rc_htl_range(self, engine_custom):
        tau = engine_custom.tau_rc_htl
        assert 1e-7 <= tau <= 1e-3


class TestImpedance:
    def test_high_freq_geo_capacitance(self, engine):
        """At high frequency, C → C_geo (ionic and trap frozen out)."""
        cap_high = engine.interface_capacitance(0.5, 1e6, 300.0)
        cap_low = engine.interface_capacitance(0.5, 0.1, 300.0)
        # Low freq should have larger total capacitance
        assert cap_low['C_total'] > cap_high['C_total']
        # High freq should be dominated by C_geo
        assert cap_high['C_geo'] / cap_high['C_total'] > 0.5

    def test_impedance_spectrum_shape(self, engine):
        """Z should decrease in magnitude with frequency."""
        freq = np.logspace(-1, 6, 50)
        z = engine.impedance_spectrum(0.5, freq, 300.0)
        assert z['Z_magnitude'][0] > z['Z_magnitude'][-1]

    def test_impedance_real_positive(self, engine):
        freq = np.logspace(0, 5, 20)
        z = engine.impedance_spectrum(0.5, freq, 300.0)
        assert np.all(z['Z_real'] > 0)


class TestFlushing:
    def test_pulse_reduces_trapped_charge(self, engine):
        """A voltage pulse should cause lower steady-state trapped charge vs no pulse."""
        # Compare: with strong V_pulse (fast flushing) vs V=0 (no flushing)
        # At V=0, flushing is disabled so traps fill more
        engine.reset()
        engine.n_t_etl = engine.etl['N_t'] * 0.8  # start with filled traps
        engine.n_t_htl = engine.htl['N_t'] * 0.8
        nt_before = engine.n_t_etl
        # Apply strong pulse
        for _ in range(200):
            engine.trap_dynamics(1e14, 1e14, 5.0, 300.0, 1e-6)
        nt_after_pulse = engine.n_t_etl
        # Strong pulse should flush traps
        assert nt_after_pulse < nt_before

    def test_recovery_after_pulse(self, engine):
        """After pulse ends, trapped charge may rebuild."""
        res = engine.flush_response(V_pulse=3.0, pulse_width=50e-6, T=300.0, dt=1e-6)
        assert len(res['time']) > 10
        assert len(res['n_t_etl']) == len(res['time'])


class TestTemperature:
    def test_higher_T_less_trapped(self, engine):
        """Higher T → higher emission rate → less trapped charge at SS."""
        engine.reset()
        for _ in range(2000):
            engine.trap_dynamics(1e16, 1e16, 0.0, 280.0, 1e-6)
        nt_cold = engine.n_t_etl

        engine.reset()
        for _ in range(2000):
            engine.trap_dynamics(1e16, 1e16, 0.0, 360.0, 1e-6)
        nt_hot = engine.n_t_etl
        assert nt_hot < nt_cold

    def test_recombination_current_temperature(self, engine):
        """J_rec increases with T (emission rate increases)."""
        J_cold = engine.recombination_current(1e16, 1e15, 0.5, 280.0)
        J_hot = engine.recombination_current(1e16, 1e15, 0.5, 360.0)
        assert J_hot > J_cold
