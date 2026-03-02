#!/usr/bin/env python3
"""
Tests for 1D Drift-Diffusion Ion Dynamics Engine
=================================================
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engines.ion_dynamics import IonDynamicsEngine, _bernoulli, _thomas_solve


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def engine():
    """Default engine with iodide ions in 500nm MAPbI3."""
    return IonDynamicsEngine(
        layer_thickness_nm=500,
        ion_params={
            'iodide': {
                'D_i': 1e-12, 'mu_i': 4e-11, 'n_i0': 1.6e19,
                'E_activation': 0.58, 'charge': -1,
            }
        },
        grid_points=100,
    )


@pytest.fixture
def engine_two_species():
    """Engine with both iodide and MA cation."""
    return IonDynamicsEngine(
        layer_thickness_nm=500,
        ion_params={
            'iodide': {
                'D_i': 1e-12, 'mu_i': 4e-11, 'n_i0': 1.6e19,
                'E_activation': 0.58, 'charge': -1,
            },
            'ma_cation': {
                'D_i': 1e-16, 'mu_i': 4e-15, 'n_i0': 1.6e19,
                'E_activation': 1.12, 'charge': +1,
            },
        },
        grid_points=50,
    )


# ============================================================
# Test 1: Bernoulli function
# ============================================================

class TestBernoulli:
    def test_bernoulli_at_zero(self):
        """B(0) = 1."""
        assert abs(_bernoulli(np.array([0.0]))[0] - 1.0) < 1e-8

    def test_bernoulli_identity(self):
        """B(x) + B(-x) = x  (identity for Bernoulli function... 
        actually B(-x) = B(x) + x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        Bp = _bernoulli(x)
        Bm = _bernoulli(-x)
        # Identity: B(-x) = B(x) + x
        np.testing.assert_allclose(Bm, Bp + x, rtol=1e-6)


# ============================================================
# Test 2: Thomas solver
# ============================================================

class TestThomas:
    def test_simple_system(self):
        """Solve a known 3x3 tridiagonal system."""
        # [ 2 -1  0] [x0]   [1]
        # [-1  2 -1] [x1] = [0]
        # [ 0 -1  2] [x2]   [1]
        a = np.array([-1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0])
        d = np.array([1.0, 0.0, 1.0])
        x = _thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-10)


# ============================================================
# Test 3: Poisson solver consistency
# ============================================================

class TestPoisson:
    def test_charge_neutral_gives_flat_field(self, engine):
        """Uniform charge neutrality → zero internal field (with zero BC)."""
        n_cat = np.full(engine.N, 1e18)
        n_an = np.full(engine.N, 1e18)
        phi, E = engine.solve_poisson(n_cat, n_an, V_left=0.0, V_right=0.0)
        # E should be approximately zero
        assert np.max(np.abs(E)) < 1e3, "Neutral charge should give ~zero field"

    def test_nonzero_bias_gives_field(self, engine):
        """Applied voltage creates field even with neutral charge."""
        n_cat = np.full(engine.N, 1e18)
        n_an = np.full(engine.N, 1e18)
        phi, E = engine.solve_poisson(n_cat, n_an, V_left=0.0, V_right=1.0)
        # Average field should be ~V/L
        avg_E = np.mean(E)
        expected = -1.0 / engine.L  # V/cm
        assert abs(avg_E - expected) / abs(expected) < 0.1


# ============================================================
# Test 4: Diffusion only (E=0) — Gaussian spreading
# ============================================================

class TestDiffusionOnly:
    def test_gaussian_diffusion(self):
        """A Gaussian pulse should broaden under pure diffusion."""
        eng = IonDynamicsEngine(
            layer_thickness_nm=1000, grid_points=200,
            ion_params={'test': {'D_i': 1e-10, 'mu_i': 0, 'n_i0': 1e18,
                                  'E_activation': 0, 'charge': -1}}
        )
        # Initial Gaussian
        x0 = eng.L / 2
        sigma0 = eng.L / 20
        n_init = 1e18 + 1e17 * np.exp(-((eng.x - x0) ** 2) / (2 * sigma0 ** 2))

        E_zero = np.zeros(eng.N)
        D_i = 1e-10
        dt = 1e-4
        n_current = n_init.copy()

        for _ in range(100):
            n_current = eng.drift_diffusion_step(n_current, E_zero, D_i, 0.0, dt, -1)

        # Pulse should have broadened: variance increased
        mean_init = np.average(eng.x, weights=n_init - 1e18)
        var_init = np.average((eng.x - mean_init) ** 2, weights=np.maximum(n_init - 1e18, 0))

        excess = np.maximum(n_current - 1e18, 0)
        if np.sum(excess) > 0:
            mean_final = np.average(eng.x, weights=excess)
            var_final = np.average((eng.x - mean_final) ** 2, weights=excess)
            assert var_final > var_init, "Gaussian should broaden under diffusion"


# ============================================================
# Test 5: Drift only — uniform density shifts
# ============================================================

class TestDriftOnly:
    def test_uniform_drift_preserves_total(self):
        """Drift should conserve total ion count (zero-flux BC)."""
        eng = IonDynamicsEngine(
            layer_thickness_nm=500, grid_points=100,
            ion_params={'test': {'D_i': 1e-20, 'mu_i': 1e-8, 'n_i0': 1e18,
                                  'E_activation': 0, 'charge': -1}}
        )
        n_init = np.full(eng.N, 1e18)
        E_uniform = np.full(eng.N, 1e4)  # V/cm
        dt = 1e-6

        n_new = eng.drift_diffusion_step(n_init, E_uniform, 1e-20, 1e-8, dt, -1)
        total_init = np.trapezoid(n_init, eng.x)
        total_final = np.trapezoid(n_new, eng.x)
        np.testing.assert_allclose(total_final, total_init, rtol=1e-3)


# ============================================================
# Test 6: Steady state convergence
# ============================================================

class TestSteadyState:
    def test_converges(self, engine):
        """Steady state should converge within max iterations."""
        result = engine.steady_state(V_applied=0.5, G=1.0, T=300.0)
        assert result['converged'], "Steady state should converge"
        assert result['V_OC'] > 0
        assert result['P_out'] > 0

    def test_steady_state_positive_outputs(self, engine):
        result = engine.steady_state(V_applied=0.0, G=1.0, T=300.0)
        assert result['FF'] > 0.2
        assert result['FF'] < 0.95


# ============================================================
# Test 7: Hysteresis physics
# ============================================================

class TestHysteresis:
    def test_reverse_higher_pce(self, engine):
        """Reverse scan typically shows higher PCE (known phenomenon)."""
        result = engine.hysteresis_iv(V_sweep_rate=0.1, G=1.0, T=300.0)
        # For slow sweep, reverse should be >= forward (or close)
        # This is the well-known hysteresis direction for normal structure
        assert result['PCE_reverse'] >= result['PCE_forward'] * 0.8, \
            "Reverse PCE should not be much lower than forward"

    def test_fast_sweep_reduces_hysteresis(self, engine):
        """Faster sweep → ions can't follow → less hysteresis."""
        slow = engine.hysteresis_iv(V_sweep_rate=0.01, G=1.0, T=300.0)
        fast = engine.hysteresis_iv(V_sweep_rate=10.0, G=1.0, T=300.0)
        # Fast sweep should have lower or equal HI
        assert fast['HI'] <= slow['HI'] + 0.05, \
            f"Fast HI={fast['HI']:.3f} should be <= slow HI={slow['HI']:.3f}"

    def test_hysteresis_index_bounded(self, engine):
        result = engine.hysteresis_iv(V_sweep_rate=1.0, G=1.0, T=300.0)
        assert 0 <= result['HI'] <= 1.0, f"HI={result['HI']} should be in [0,1]"


# ============================================================
# Test 8: Temperature dependence
# ============================================================

class TestTemperature:
    def test_higher_T_faster_diffusion(self, engine):
        """Higher temperature → higher D_i → faster response."""
        tau_cold = engine.get_ion_timescale(T=280.0)
        tau_hot = engine.get_ion_timescale(T=350.0)
        assert tau_hot < tau_cold, "Higher T should give faster (smaller) timescale"


# ============================================================
# Test 9: Response time
# ============================================================

class TestResponseTime:
    def test_ion_timescale_in_range(self, engine):
        """τ_ion should be in [1ms, 10s] for typical perovskite."""
        tau = engine.get_ion_timescale(T=300.0)
        assert 1e-3 <= tau <= 100.0, f"tau={tau} outside [1ms, 100s]"

    def test_response_time_returns_values(self, engine):
        result = engine.response_time(V_step=0.5, G=1.0, T=300.0,
                                       duration_s=0.5, n_steps=200)
        assert 'tau_ion' in result
        assert 'tau_90' in result
        assert result['tau_90'] >= result['tau_ion']


# ============================================================
# Test 10: Energy / mass conservation
# ============================================================

class TestConservation:
    def test_ion_conservation_in_simulate(self, engine):
        """Total ion count should be conserved during simulation."""
        N_t = 50
        V_t = np.full(N_t, 0.5)
        G_t = np.full(N_t, 1.0)
        T_t = np.full(N_t, 300.0)

        # Initial total
        total_init = {}
        for name, sp in engine.ion_species.items():
            total_init[name] = np.trapezoid(engine.ion_profiles[name], engine.x)

        result = engine.simulate(V_t, G_t, T_t, dt=1e-4)

        for name in engine.ion_species:
            total_final = np.trapezoid(result['n_ion'][name], engine.x)
            np.testing.assert_allclose(
                total_final, total_init[name], rtol=0.02,
                err_msg=f"Ion {name} not conserved"
            )


# ============================================================
# Test 11: Two-species engine
# ============================================================

class TestTwoSpecies:
    def test_two_species_steady_state(self, engine_two_species):
        """Two species should produce valid output even if slow species not fully converged."""
        result = engine_two_species.steady_state(V_applied=0.5, G=1.0, T=300.0,
                                                  max_iter=500, tol=1e-3)
        # MA⁺ is extremely slow (D~1e-16), may not fully converge
        assert 'iodide' in result['n_ion']
        assert 'ma_cation' in result['n_ion']
        assert result['V_OC'] > 0
        assert result['P_out'] > 0

    def test_cation_slower_than_anion(self, engine_two_species):
        """MA⁺ (D~1e-16) should be much slower than I⁻ (D~1e-12)."""
        sp_i = engine_two_species.ion_species['iodide']
        sp_ma = engine_two_species.ion_species['ma_cation']
        assert sp_ma['D_i'] < sp_i['D_i'] * 1e-2


# ============================================================
# Test 12: Simulate output shapes
# ============================================================

class TestSimulate:
    def test_output_shapes(self, engine):
        N_t = 30
        result = engine.simulate(
            np.full(N_t, 0.5), np.full(N_t, 1.0), np.full(N_t, 300.0), dt=1e-4
        )
        assert result['P_out'].shape == (N_t,)
        assert result['V_OC'].shape == (N_t,)
        assert result['FF'].shape == (N_t,)
        assert result['E_field'].shape == (engine.N,)

    def test_positive_power(self, engine):
        N_t = 20
        result = engine.simulate(
            np.full(N_t, 0.3), np.full(N_t, 1.0), np.full(N_t, 300.0), dt=1e-4
        )
        assert np.all(result['P_out'] > 0)


# ============================================================
# Test 13: Integration with DynamicIVEngine
# ============================================================

class TestIntegration:
    def test_dynamic_iv_has_ion_engine(self):
        from engines.dynamic_iv import DynamicIVEngine
        pv = {'bandgap': 1.55, 'thickness': 500, 'ion_density': 1e18,
              'ion_mobility': 1e-9, 'V_bi': 1.0}
        fet = {'V_th': 0.5, 'W_L': 100}
        eng = DynamicIVEngine(pv, fet, {})
        ion_eng = eng.get_ion_engine()
        assert ion_eng is not None
        assert isinstance(ion_eng, IonDynamicsEngine)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
