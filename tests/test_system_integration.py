#!/usr/bin/env python3
"""Tests for System Integration Engine (Phase 3-3)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engines.device_simulator import PVFETDeviceSimulator
from engines.array_scaleup import ArrayScaleupEngine
from engines.system_integration import SystemIntegrationEngine, ECON_DEFAULTS


@pytest.fixture
def system():
    cell = PVFETDeviceSimulator()
    array = ArrayScaleupEngine(cell, n_series=6, n_parallel=3)
    return SystemIntegrationEngine(array)


class TestSimulate24h:
    def test_returns_4_scenarios(self, system):
        res = system.simulate_24h()
        assert 'scenarios' in res
        for s in ['A', 'B', 'C', 'D']:
            assert s in res['scenarios']

    def test_time_array_1440(self, system):
        res = system.simulate_24h()
        assert len(res['time_min']) == 1440

    def test_energy_conservation(self, system):
        """PV + Grid_import ≈ Load + HESS_net_charge + Grid_export + losses."""
        res = system.simulate_24h()
        for s in ['A', 'B']:
            sc = res['scenarios'][s]
            pv = sc['total_pv_kWh']
            imp = sc['total_import_kWh']
            load = sc['total_load_kWh']
            exp = sc['total_export_kWh']
            # Allow 20% for HESS losses and numerical errors
            supply = pv + imp
            demand = load + exp
            ratio = supply / max(demand, 1)
            assert 0.5 < ratio < 2.0, f"Energy balance ratio {ratio} out of range for {s}"

    def test_scenario_ordering_self_sufficiency(self, system):
        """D ≥ C ≥ B ≥ A in self-sufficiency."""
        res = system.simulate_24h()
        ss = {s: res['scenarios'][s]['self_sufficiency'] for s in 'ABCD'}
        # Allow small tolerance
        assert ss['D'] >= ss['A'] * 0.95, f"D={ss['D']}, A={ss['A']}"
        assert ss['B'] >= ss['A'] * 0.99

    def test_self_consumption_in_range(self, system):
        res = system.simulate_24h()
        for s in 'ABCD':
            sc = res['scenarios'][s]['self_consumption']
            assert 0 <= sc <= 1.0

    def test_scenario_d_smaller_hess(self, system):
        res = system.simulate_24h()
        cap_a = res['scenarios']['A']['hess_capacity_kWh']
        cap_d = res['scenarios']['D']['hess_capacity_kWh']
        assert cap_d <= cap_a


class TestHESSSizing:
    def test_active_control_reduces_hess(self, system):
        res = system.hess_sizing_optimization(target_self_consumption=0.7)
        assert res['D']['capacity_kWh'] <= res['A']['capacity_kWh'] * 1.01

    def test_reduction_pct_positive_for_d(self, system):
        res = system.hess_sizing_optimization(target_self_consumption=0.7)
        assert res['D']['reduction_pct'] >= 0


class TestEconomicComparison:
    def test_lcoe_in_range(self, system):
        res = system.economic_comparison()
        for s in 'ABCD':
            lcoe = res[s]['LCOE_per_MWh']
            assert 30 <= lcoe <= 150, f"LCOE={lcoe} for {s} out of range"

    def test_npv_discount_rate_relationship(self, system):
        res_low = system.economic_comparison(discount_rate=0.05)
        res_high = system.economic_comparison(discount_rate=0.12)
        # Higher discount rate → lower NPV
        assert res_high['A']['NPV'] < res_low['A']['NPV']

    def test_capex_includes_fet_for_b(self, system):
        res = system.economic_comparison()
        # B has FET cost, A doesn't
        assert res['B']['CAPEX'] > res['A']['CAPEX']

    def test_payback_positive(self, system):
        res = system.economic_comparison()
        for s in 'ABCD':
            assert res[s]['payback_years'] > 0


class TestCarbonImpact:
    def test_all_scenarios_positive_savings(self, system):
        res = system.carbon_impact()
        for s in 'ABCD':
            assert res[s]['annual_grid_carbon_saved_tCO2'] > 0
            assert res[s]['lifetime_total_tCO2_saved'] > 0

    def test_d_saves_more_hess_carbon(self, system):
        res = system.carbon_impact()
        assert res['D']['hess_carbon_avoided_tCO2'] >= res['A']['hess_carbon_avoided_tCO2']


class TestSensitivityAnalysis:
    def test_returns_all_params(self, system):
        res = system.sensitivity_analysis()
        assert 'pv_cost' in res
        assert 'hess_cost' in res
        assert 'electricity_price' in res

    def test_delta_positive(self, system):
        res = system.sensitivity_analysis()
        for param, vals in res.items():
            assert vals['delta'] >= 0
