#!/usr/bin/env python3
"""Tests for Array Scaleup Engine (Phase 3-2)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engines.device_simulator import PVFETDeviceSimulator
from engines.array_scaleup import ArrayScaleupEngine


@pytest.fixture
def cell():
    return PVFETDeviceSimulator()


@pytest.fixture
def array(cell):
    return ArrayScaleupEngine(cell, n_series=6, n_parallel=3)


class TestModuleIV:
    def test_module_power_positive(self, array):
        ns, np_ = array.n_series, array.n_parallel
        G = np.ones((ns, np_))
        T = np.full((ns, np_), 298.15)
        res = array.module_iv(G, T)
        assert res['P_max'] > 0

    def test_module_power_le_sum_cells(self, array, cell):
        """Module power ≤ n_cells × single cell power (mismatch)."""
        ns, np_ = array.n_series, array.n_parallel
        G = np.ones((ns, np_))
        T = np.full((ns, np_), 298.15)
        res = array.module_iv(G, T)

        V = np.linspace(0, 1.2, 200)
        cell_res = cell.jv_characteristics(V, G=1.0, T=298.15, V_G=2.0)
        single_Pmax = float(np.max(cell_res['P']))
        total_ideal = single_Pmax * ns * np_
        assert res['P_max'] <= total_ideal * 1.01  # 1% tolerance

    def test_shape_mismatch_raises(self, array):
        with pytest.raises(ValueError):
            array.module_iv(np.ones((2, 2)), np.ones((2, 2)))


class TestPartialShading:
    def test_individual_ge_uniform(self, array):
        """Individual V_G control ≥ uniform under shading."""
        ns, np_ = array.n_series, array.n_parallel
        shade = np.zeros((ns, np_))
        shade[0, :] = 0.8  # Top row heavily shaded
        shade[1, :] = 0.3
        res = array.partial_shading_analysis(shade)
        assert res['P_individual'] >= res['P_uniform'] * 0.99

    def test_row_ge_uniform(self, array):
        ns, np_ = array.n_series, array.n_parallel
        shade = np.zeros((ns, np_))
        shade[0, :] = 0.7
        res = array.partial_shading_analysis(shade)
        assert res['P_row'] >= res['P_uniform'] * 0.99

    def test_shading_reduces_power(self, array):
        ns, np_ = array.n_series, array.n_parallel
        shade = np.zeros((ns, np_))
        shade[:2, :] = 0.5
        res = array.partial_shading_analysis(shade)
        assert res['P_uniform'] < res['P_no_shading']

    def test_mismatch_losses_positive(self, array):
        ns, np_ = array.n_series, array.n_parallel
        shade = np.zeros((ns, np_))
        shade[0, :] = 0.6
        res = array.partial_shading_analysis(shade)
        assert res['mismatch_loss_uniform'] >= 0


class TestMismatchAnalysis:
    def test_mc_distributions(self, array):
        res = array.mismatch_analysis(cell_variation_pct=5, n_trials=20, seed=0)
        assert len(res['loss_no_control']) == 20
        assert res['mean_loss_no_control'] >= 0
        assert res['mean_loss_no_control'] < 0.5

    def test_control_reduces_mismatch(self, array):
        res = array.mismatch_analysis(cell_variation_pct=8, n_trials=20, seed=0)
        assert res['mean_loss_with_control'] <= res['mean_loss_no_control'] * 1.05


class TestArrayToGrid:
    def test_array_output_positive(self, array):
        res = array.array_to_grid(n_modules=10)
        assert res['P_ac_kW_mppt'] > 0
        assert res['annual_MWh_mppt'] > 0

    def test_inverter_efficiency_applied(self, array):
        res = array.array_to_grid(n_modules=10, inverter_efficiency=0.95)
        assert res['P_ac_kW_mppt'] < res['P_array_kW_mppt']


class TestAnnualYield:
    def test_pr_in_range(self, array):
        res = array.annual_yield(location_lat=35)
        assert 0 < res['PR_mppt'] < 1.0
        assert 0 < res['PR_active'] < 1.0

    def test_active_pr_ge_mppt(self, array):
        res = array.annual_yield()
        assert res['PR_active'] >= res['PR_mppt']

    def test_annual_yield_reasonable(self, array):
        res = array.annual_yield(location_lat=35)
        # Should be in 800-2200 kWh/kWp range
        assert 800 <= res['annual_kWh_per_kWp_mppt'] <= 2200
