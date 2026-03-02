#!/usr/bin/env python3
"""Tests for PV-FET Device Simulator (Phase 3-1)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engines.device_simulator import PVFETDeviceSimulator


@pytest.fixture
def device():
    return PVFETDeviceSimulator()


@pytest.fixture
def device_fapbi3():
    d = PVFETDeviceSimulator()
    d.build_layer_stack(perovskite='FAPbI3', etl='TiO2', htl='PTAA')
    return d


class TestBuildLayerStack:
    def test_default_stack_has_7_layers(self, device):
        assert len(device.layer_stack) == 7

    def test_custom_materials(self, device_fapbi3):
        names = [l['name'] for l in device_fapbi3.layer_stack]
        assert 'perovskite' in names
        assert device_fapbi3.perovskite_config['Eg'] == 1.48


class TestBandDiagram:
    def test_band_diagram_keys(self, device):
        bd = device.band_diagram()
        for key in ['x_nm', 'CB', 'VB', 'E_Fn', 'E_Fp']:
            assert key in bd

    def test_band_continuity(self, device):
        """CB and VB should not have huge jumps (>10eV) between adjacent points.
        Heterointerface offsets up to ~9 eV are allowed (e.g., gate oxide)."""
        bd = device.band_diagram(V_applied=0.5)
        dCB = np.abs(np.diff(bd['CB']))
        dVB = np.abs(np.diff(bd['VB']))
        assert np.max(dCB) < 10.0, f"CB has discontinuity > 10 eV: {np.max(dCB)}"
        assert np.max(dVB) < 10.0, f"VB has discontinuity > 10 eV: {np.max(dVB)}"

    def test_cb_above_vb(self, device):
        bd = device.band_diagram()
        # In each layer, CB should be above VB (CB > VB in our convention)
        # Allow for sign conventions
        gap = bd['CB'] - bd['VB']
        assert np.all(gap > -0.5), "CB below VB by more than 0.5 eV"

    def test_illumination_splits_quasi_fermi(self, device):
        bd_dark = device.band_diagram(illumination=False)
        bd_light = device.band_diagram(illumination=True)
        split_dark = np.mean(np.abs(bd_dark['E_Fn'] - bd_dark['E_Fp']))
        split_light = np.mean(np.abs(bd_light['E_Fn'] - bd_light['E_Fp']))
        assert split_light > split_dark

    def test_vg_affects_igzo_band(self, device):
        bd0 = device.band_diagram(V_G=0)
        bd5 = device.band_diagram(V_G=5)
        # IGZO region should differ
        assert not np.allclose(bd0['CB'], bd5['CB'])


class TestJVCharacteristics:
    def test_basic_jv_params(self, device):
        V = np.linspace(0, 1.2, 200)
        res = device.jv_characteristics(V, G=1.0, T=298.15, V_G=2.0)
        assert res['V_OC'] > 0, "V_OC should be positive"
        assert res['J_SC'] > 0, "J_SC should be positive"
        assert 0 < res['FF'] < 1, f"FF={res['FF']} out of range"
        assert res['PCE'] > 0, "PCE should be positive"

    def test_pce_below_sq_limit(self, device):
        V = np.linspace(0, 1.2, 200)
        res = device.jv_characteristics(V, G=1.0, T=298.15, V_G=3.0)
        assert res['PCE'] < 0.34, f"PCE={res['PCE']} exceeds SQ limit"

    def test_vg_changes_pce(self, device):
        V = np.linspace(0, 1.2, 200)
        res0 = device.jv_characteristics(V, G=1.0, T=298.15, V_G=0)
        res3 = device.jv_characteristics(V, G=1.0, T=298.15, V_G=3.0)
        # V_G should change PCE (controllability)
        assert res0['PCE'] != res3['PCE']

    def test_higher_G_higher_Jsc(self, device):
        V = np.linspace(0, 1.2, 200)
        res1 = device.jv_characteristics(V, G=0.5, T=298.15)
        res2 = device.jv_characteristics(V, G=1.0, T=298.15)
        assert res2['J_SC'] > res1['J_SC']


class TestQuantumEfficiency:
    def test_eqe_bounds(self, device):
        wl = np.linspace(350, 900, 100)
        res = device.quantum_efficiency(wl)
        assert np.all(res['EQE'] >= 0)
        assert np.all(res['EQE'] <= 1.0)

    def test_bandgap_edge(self, device):
        """EQE should drop near bandgap wavelength."""
        wl = np.linspace(300, 1000, 200)
        res = device.quantum_efficiency(wl)
        bg_wl = 1240 / device._Eg  # ~800 nm for MAPbI3
        # EQE at 500nm should be > EQE at 900nm
        eqe_500 = np.interp(500, wl, res['EQE'])
        eqe_900 = np.interp(900, wl, res['EQE'])
        assert eqe_500 > eqe_900

    def test_vg_affects_eqe(self, device):
        wl = np.linspace(400, 800, 50)
        res0 = device.quantum_efficiency(wl, V_G=0)
        res5 = device.quantum_efficiency(wl, V_G=5)
        # Should differ due to collection efficiency change
        assert not np.allclose(res0['EQE'], res5['EQE'])


class TestParasiticLoss:
    def test_all_losses_positive(self, device):
        losses = device.parasitic_loss_analysis()
        for name, val in losses.items():
            assert val >= 0, f"Loss {name} is negative"

    def test_total_loss_reasonable(self, device):
        losses = device.parasitic_loss_analysis()
        total = sum(losses.values())
        assert total < 1.0, "Total loss > 100%"
        assert total > 0.01, "Total loss unrealistically low"


class TestTemperatureCoefficient:
    def test_voc_decreases_with_T(self, device):
        T_range = np.array([280, 300, 320, 340])
        res = device.temperature_coefficient(T_range, G=1000, V_G_opt=False)
        # V_OC should decrease with T (physical)
        assert res['V_OC'][0] > res['V_OC'][-1], "V_OC should decrease with T"

    def test_dvoc_dt_negative(self, device):
        T_range = np.linspace(280, 340, 5)
        res = device.temperature_coefficient(T_range, G=1000, V_G_opt=False)
        # Most dV_OC/dT should be negative
        assert np.mean(res['dVOC_dT']) < 0

    def test_active_control_mitigates_T_loss(self, device):
        T_range = np.array([298.15, 340])
        res_fixed = device.temperature_coefficient(T_range, V_G_opt=False)
        res_opt = device.temperature_coefficient(T_range, V_G_opt=True)
        # Optimized should have >= PCE at high T
        assert res_opt['PCE'][-1] >= res_fixed['PCE'][-1] * 0.99
