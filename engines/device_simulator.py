#!/usr/bin/env python3
"""
PV-FET Monolithic Device Simulator (Phase 3-1)
===============================================

1D device simulation of monolithic PV-FET structure:
Glass/ITO(top)/HTL/Perovskite/ETL/ITO(mid)/IGZO/Gate-oxide/Gate

Integrates with existing dynamic_iv.py FET model and config.py material DB.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import brentq

try:
    from config import (Q, KB, H, C, T_CELL, MATERIAL_DB,
                        get_am15g_spectrum, DEFAULT_CONFIG)
except ImportError:
    from ..config import (Q, KB, H, C, T_CELL, MATERIAL_DB,
                          get_am15g_spectrum, DEFAULT_CONFIG)


# ── Material property databases for device layers ──────────────────────────

_ELECTRODE_DB = {
    'ITO': {'work_function': 4.7, 'sheet_R_ohm_sq': 15, 'n_opt': 1.9,
            'k_opt': 0.01, 'thickness_nm': 150},
    'FTO': {'work_function': 4.4, 'sheet_R_ohm_sq': 8, 'n_opt': 2.0,
            'k_opt': 0.02, 'thickness_nm': 400},
    'AZO': {'work_function': 4.5, 'sheet_R_ohm_sq': 20, 'n_opt': 1.85,
            'k_opt': 0.015, 'thickness_nm': 200},
}

_ETL_DB = {
    'SnO2': {'CB': -4.0, 'VB': -7.6, 'Eg': 3.6, 'thickness_nm': 30,
             'mobility_e': 100, 'trap_density': 1e16},
    'TiO2': {'CB': -4.0, 'VB': -7.3, 'Eg': 3.3, 'thickness_nm': 50,
             'mobility_e': 1, 'trap_density': 5e16},
    'ZnO':  {'CB': -4.2, 'VB': -7.6, 'Eg': 3.4, 'thickness_nm': 40,
             'mobility_e': 50, 'trap_density': 2e16},
    'C60':  {'CB': -3.9, 'VB': -6.2, 'Eg': 2.3, 'thickness_nm': 20,
             'mobility_e': 0.1, 'trap_density': 1e17},
}

_HTL_DB = {
    'Spiro': {'HOMO': -5.2, 'LUMO': -2.2, 'Eg': 3.0, 'thickness_nm': 200,
              'mobility_h': 2e-4, 'trap_density': 5e15},
    'PTAA':  {'HOMO': -5.1, 'LUMO': -2.3, 'Eg': 2.8, 'thickness_nm': 60,
              'mobility_h': 1e-3, 'trap_density': 3e15},
    'NiOx':  {'HOMO': -5.4, 'LUMO': -1.8, 'Eg': 3.6, 'thickness_nm': 30,
              'mobility_h': 0.1, 'trap_density': 1e16},
    'P3HT':  {'HOMO': -5.0, 'LUMO': -3.0, 'Eg': 2.0, 'thickness_nm': 100,
              'mobility_h': 1e-2, 'trap_density': 1e16},
}

_PEROVSKITE_DB = {
    'MAPbI3':  {'Eg': 1.55, 'CB': -3.93, 'VB': -5.48, 'epsilon_r': 25,
                'alpha_cm': 1e5, 'thickness_nm': 500, 'mobility_e': 20,
                'mobility_h': 10, 'n_i': 1e10},
    'FAPbI3':  {'Eg': 1.48, 'CB': -3.90, 'VB': -5.38, 'epsilon_r': 28,
                'alpha_cm': 1.2e5, 'thickness_nm': 500, 'mobility_e': 25,
                'mobility_h': 12, 'n_i': 2e10},
    'CsPbI3':  {'Eg': 1.73, 'CB': -3.85, 'VB': -5.58, 'epsilon_r': 20,
                'alpha_cm': 9e4, 'thickness_nm': 400, 'mobility_e': 15,
                'mobility_h': 8, 'n_i': 5e9},
    'MAPbBr3': {'Eg': 2.30, 'CB': -3.60, 'VB': -5.90, 'epsilon_r': 22,
                'alpha_cm': 8e4, 'thickness_nm': 400, 'mobility_e': 30,
                'mobility_h': 15, 'n_i': 1e8},
}

_IGZO_DB = {
    'IGZO': {'mu_fe': 20, 'Eg': 3.0, 'CB': -4.2, 'VB': -7.2,
             'V_th': 0.5, 'default_thickness_nm': 30},
}

_GATE_OXIDE_DB = {
    'Al2O3': {'epsilon_r': 9.0, 'Eg': 6.5, 'thickness_nm': 20,
              'leakage_A_cm2': 1e-9},
    'SiO2':  {'epsilon_r': 3.9, 'Eg': 9.0, 'thickness_nm': 50,
              'leakage_A_cm2': 1e-10},
    'HfO2':  {'epsilon_r': 25, 'Eg': 5.5, 'thickness_nm': 10,
              'leakage_A_cm2': 1e-8},
}


class PVFETDeviceSimulator:
    """PV-FET monolithic device simulator."""

    def __init__(self, layer_stack: Optional[List[Dict]] = None,
                 perovskite_config: Optional[Dict] = None,
                 fet_config: Optional[Dict] = None):
        if layer_stack is not None:
            self.layer_stack = layer_stack
        else:
            self.layer_stack = []
        self.perovskite_config = perovskite_config or {}
        self.fet_config = fet_config or {}

        # Build default if no stack given
        if not self.layer_stack:
            self.build_layer_stack()

        self._cache_derived()

    # ── Layer stack builder ─────────────────────────────────────────────

    def build_layer_stack(self, perovskite='MAPbI3', etl='SnO2', htl='Spiro',
                          electrode_top='ITO', electrode_mid='ITO',
                          igzo_thickness_nm=30, gate_oxide='Al2O3'):
        """Build device from predefined materials."""
        pvk = _PEROVSKITE_DB.get(perovskite, _PEROVSKITE_DB['MAPbI3'])
        etl_m = _ETL_DB.get(etl, _ETL_DB['SnO2'])
        htl_m = _HTL_DB.get(htl, _HTL_DB['Spiro'])
        top_e = _ELECTRODE_DB.get(electrode_top, _ELECTRODE_DB['ITO'])
        mid_e = _ELECTRODE_DB.get(electrode_mid, _ELECTRODE_DB['ITO'])
        igzo = _IGZO_DB['IGZO']
        gox = _GATE_OXIDE_DB.get(gate_oxide, _GATE_OXIDE_DB['Al2O3'])

        self.layer_stack = [
            {'name': 'electrode_top', 'material': electrode_top,
             'thickness_nm': top_e['thickness_nm'], 'properties': top_e},
            {'name': 'HTL', 'material': htl,
             'thickness_nm': htl_m['thickness_nm'], 'properties': htl_m},
            {'name': 'perovskite', 'material': perovskite,
             'thickness_nm': pvk['thickness_nm'], 'properties': pvk},
            {'name': 'ETL', 'material': etl,
             'thickness_nm': etl_m['thickness_nm'], 'properties': etl_m},
            {'name': 'electrode_mid', 'material': electrode_mid,
             'thickness_nm': mid_e['thickness_nm'], 'properties': mid_e},
            {'name': 'IGZO', 'material': 'IGZO',
             'thickness_nm': igzo_thickness_nm, 'properties': igzo},
            {'name': 'gate_oxide', 'material': gate_oxide,
             'thickness_nm': gox['thickness_nm'], 'properties': gox},
        ]

        self.perovskite_config = pvk
        self.fet_config = {
            'mu_fe': igzo['mu_fe'],
            'V_th': igzo['V_th'],
            'igzo_thickness_nm': igzo_thickness_nm,
            'gate_oxide': gox,
        }

        self._cache_derived()
        return self.layer_stack

    def _cache_derived(self):
        """Pre-compute derived quantities."""
        pvk = self.perovskite_config
        self._Eg = pvk.get('Eg', 1.55)
        self._alpha = pvk.get('alpha_cm', 1e5)
        self._thickness_cm = pvk.get('thickness_nm', 500) * 1e-7

        # Gate oxide capacitance
        gox = self.fet_config.get('gate_oxide', _GATE_OXIDE_DB['Al2O3'])
        eps0 = 8.854e-14  # F/cm
        eps_r = gox.get('epsilon_r', 9.0)
        t_ox_cm = gox.get('thickness_nm', 20) * 1e-7
        self._C_ox = eps0 * eps_r / t_ox_cm  # F/cm²

        # Sheet resistances → series R contribution (Ω·cm²)
        top_e = self._get_layer('electrode_top')
        mid_e = self._get_layer('electrode_mid')
        R_top = top_e.get('properties', {}).get('sheet_R_ohm_sq', 15) * 0.01
        R_mid = mid_e.get('properties', {}).get('sheet_R_ohm_sq', 15) * 0.01
        self._Rs_electrode = R_top + R_mid  # Ω·cm²

    def _get_layer(self, name: str) -> Dict:
        for l in self.layer_stack:
            if l['name'] == name:
                return l
        return {}

    # ── Band diagram ────────────────────────────────────────────────────

    def band_diagram(self, V_applied: float = 0, V_G: float = 0,
                     illumination: bool = False) -> Dict[str, np.ndarray]:
        """Compute 1D energy band diagram across all layers.

        Returns dict with keys: x_nm, CB, VB, E_Fn, E_Fp
        """
        segments_x = []
        segments_CB = []
        segments_VB = []
        offset = 0.0  # running position in nm
        E_ref = 0.0  # vacuum reference at top electrode Fermi

        # Top electrode Fermi level
        top_wf = self._get_layer('electrode_top').get('properties', {}).get('work_function', 4.7)

        for layer in self.layer_stack:
            t = layer['thickness_nm']
            props = layer.get('properties', {})
            n_pts = max(int(t / 5), 5)
            x_local = np.linspace(0, t, n_pts)

            if layer['name'] in ('electrode_top', 'electrode_mid'):
                wf = props.get('work_function', 4.7)
                # Metal: CB ≈ VB ≈ -WF, but show thin band
                cb = np.full(n_pts, -wf + 0.1)
                vb = np.full(n_pts, -wf - 0.1)
            elif layer['name'] == 'HTL':
                lumo = props.get('LUMO', -2.2)  # higher energy
                homo = props.get('HOMO', -5.2)  # lower energy
                bend = V_applied * 0.1 * np.linspace(0, 1, n_pts)
                cb = np.full(n_pts, lumo) - bend  # LUMO > HOMO
                vb = np.full(n_pts, homo) - bend
            elif layer['name'] == 'perovskite':
                cb_edge = props.get('CB', -3.93)  # e.g., -3.93 eV
                vb_edge = props.get('VB', -5.48)  # e.g., -5.48 eV
                V_drop_pvk = V_applied * 0.7
                bend = V_drop_pvk * np.linspace(0, 1, n_pts)
                cb = np.full(n_pts, cb_edge) - bend  # CB > VB
                vb = np.full(n_pts, vb_edge) - bend
            elif layer['name'] == 'ETL':
                cb_edge = props.get('CB', -4.0)
                eg = props.get('Eg', 3.6)
                vb_edge = cb_edge - eg  # VB = CB - Eg
                bend = V_applied * 0.15 * np.linspace(0, 1, n_pts)
                cb = np.full(n_pts, cb_edge) - bend
                vb = np.full(n_pts, vb_edge) - bend
            elif layer['name'] == 'IGZO':
                # IGZO: CB ~ -4.2 eV, VB ~ -7.2 eV (electron energy convention)
                # In our plot: higher = higher electron energy
                V_eff = max(V_G - self.fet_config.get('V_th', 0.5), 0)
                bend = V_eff * 0.2 * np.linspace(1, 0, n_pts)
                cb = np.full(n_pts, 4.2) + bend  # CB above VB
                vb = np.full(n_pts, 1.2) + bend  # Eg=3.0 eV
            elif layer['name'] == 'gate_oxide':
                eg = props.get('Eg', 6.5)
                cb = np.full(n_pts, 7.5)  # wide bandgap insulator
                vb = np.full(n_pts, 1.0)
            else:
                cb = np.zeros(n_pts)
                vb = np.ones(n_pts) * (-2)

            segments_x.append(x_local + offset)
            segments_CB.append(cb)
            segments_VB.append(vb)
            offset += t

        x_nm = np.concatenate(segments_x)
        CB = np.concatenate(segments_CB)
        VB = np.concatenate(segments_VB)

        # Quasi-Fermi levels
        if illumination:
            # Splitting proportional to V_oc estimate
            kT = KB * T_CELL / Q
            V_oc_est = self._Eg - 0.4  # rough estimate
            E_Fn = CB - 0.1 + V_oc_est * 0.3
            E_Fp = VB + 0.1 - V_oc_est * 0.3
        else:
            E_Fn = (CB + VB) / 2
            E_Fp = E_Fn.copy()

        return {'x_nm': x_nm, 'CB': CB, 'VB': VB, 'E_Fn': E_Fn, 'E_Fp': E_Fp}

    # ── J-V characteristics ─────────────────────────────────────────────

    def jv_characteristics(self, V_range: np.ndarray, G: float = 1.0,
                           T: float = 298.15, V_G: float = 0
                           ) -> Dict[str, np.ndarray]:
        """Compute J-V characteristics with FET modulation.

        Returns dict with V, J (mA/cm²), P (mW/cm²), V_OC, J_SC, FF, PCE.
        """
        kT = KB * T / Q
        Eg = self._Eg

        # Temperature-dependent bandgap (Varshni)
        Eg_T = Eg - 4.5e-4 * T ** 2 / (T + 300)

        # Photocurrent from absorption
        J_ph = self._photocurrent(G, T)  # mA/cm²

        # FET active control model
        V_th = self.fet_config.get('V_th', 0.5)
        V_eff = max(V_G - V_th, 0)
        control_factor = np.tanh(V_eff / 2.0)  # 0 to ~1

        # Ideality factor: V_G modulates recombination quality
        n_id = 2.0 - 0.7 * control_factor  # 2.0 (off) → 1.3 (on)

        # Reverse saturation current: J0 = J00 * T^3 * exp(-Eg/(n*kT))
        # Calibrated so at T=298K, n_id=1.3: Voc ~ 1.1V for Jph ~ 22 mA/cm²
        J00 = 8.9e-14 / (298.15**3 * np.exp(-1.483 / (1.3 * KB * 298.15 / Q)))
        J0 = J00 * T**3 * np.exp(-Eg_T / (n_id * kT))
        J0 = max(J0, 1e-30)

        # Resistances (kΩ·cm² so that J[mA/cm²] × Rs → V)
        Rs_base = (self._Rs_electrode + 0.5) * 1e-3
        Rs = Rs_base * (1 + 0.5 * (1 - control_factor))

        # Shunt resistance (kΩ·cm²): better with V_G control
        Rsh = 0.5 + 9.5 * control_factor  # 0.5 (off) → 10 kΩ·cm² (on)

        J = np.zeros_like(V_range, dtype=float)
        for i, V in enumerate(V_range):
            # Implicit equation: J = Jph - J0*(exp((V+J*Rs)/(n*Vt))-1) - (V+J*Rs)/Rsh
            # Solve iteratively
            J_guess = J_ph - J0 * (np.exp(V / (n_id * kT)) - 1) - V / Rsh
            for _ in range(50):
                exp_term = np.exp(np.clip((V + J_guess * Rs) / (n_id * kT), -50, 50))
                f = J_ph - J0 * (exp_term - 1) - (V + J_guess * Rs) / Rsh - J_guess
                df = -J0 * Rs / (n_id * kT) * exp_term - Rs / Rsh - 1
                if abs(df) < 1e-30:
                    break
                dJ = f / (-df)
                J_guess += dJ
                if abs(dJ) < 1e-10:
                    break
            J[i] = J_guess

        P = V_range * J  # mW/cm²

        # Extract parameters
        # J_SC at V=0
        idx0 = np.argmin(np.abs(V_range))
        J_SC = float(J[idx0])

        # V_OC: J=0 crossing
        sign_changes = np.where(np.diff(np.sign(J)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation
            V_OC = V_range[idx] - J[idx] * (V_range[idx + 1] - V_range[idx]) / (J[idx + 1] - J[idx])
        else:
            V_OC = V_range[-1] if J[-1] > 0 else 0.0

        V_OC = max(V_OC, 0)
        J_SC = max(J_SC, 0)

        # FF and PCE
        P_max = float(np.max(P)) if np.any(P > 0) else 0.0
        if V_OC > 0 and J_SC > 0:
            FF = P_max / (V_OC * J_SC)
        else:
            FF = 0.0

        FF = np.clip(FF, 0, 0.92)  # Physical limit
        P_in = G * 100.0  # mW/cm²
        PCE = P_max / P_in if P_in > 0 else 0.0

        # Shockley-Queisser limit
        sq_limit = self._sq_limit(Eg_T)
        PCE = min(PCE, sq_limit)

        return {
            'V': V_range, 'J': J, 'P': P,
            'V_OC': float(V_OC), 'J_SC': float(J_SC),
            'FF': float(FF), 'PCE': float(PCE),
        }

    def _photocurrent(self, G: float, T: float) -> float:
        """Calculate photocurrent density [mA/cm²] from AM1.5G spectrum."""
        wl = np.linspace(300, 1240 / self._Eg, 200)  # up to bandgap wavelength
        spectrum = get_am15g_spectrum(wl)  # W/m²/nm

        # Photon flux: Φ = E_λ / E_photon  [photons/m²/s/nm]
        E_photon = H * C / (wl * 1e-9)  # J per photon
        photon_flux = spectrum / E_photon  # photons/m²/s/nm

        # Absorption: Beer-Lambert (wavelength-dependent)
        E_ph = 1240.0 / wl
        alpha_wl = np.where(E_ph > self._Eg,
                            self._alpha * np.sqrt(np.maximum(E_ph - self._Eg, 0) / 0.5),
                            self._alpha * 0.01)
        absorptance = 1 - np.exp(-alpha_wl * self._thickness_cm)
        # Collection efficiency (accounts for optical + transport + interface losses)
        # Calibrated: MAPbI3 Jsc ~ 22 mA/cm² at 1 sun
        collection = 0.52

        # Integrate
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        J_ph = Q * _trapz(photon_flux * absorptance * collection, wl) * 1e-1  # A/m² → mA/cm²
        return float(J_ph * G)

    def _sq_limit(self, Eg: float) -> float:
        """Shockley-Queisser limit for given bandgap."""
        # Tabulated approximate values
        if Eg < 0.5:
            return 0.10
        elif Eg < 1.0:
            return 0.25 + (Eg - 0.5) * 0.14
        elif Eg < 1.1:
            return 0.32 + (Eg - 1.0) * 0.01
        elif Eg < 1.4:
            return 0.33
        elif Eg < 1.6:
            return 0.33 - (Eg - 1.4) * 0.02
        elif Eg < 2.0:
            return 0.31 - (Eg - 1.6) * 0.05
        else:
            return max(0.05, 0.31 - (Eg - 1.6) * 0.05)

    # ── Quantum efficiency ──────────────────────────────────────────────

    def quantum_efficiency(self, wavelength_range: np.ndarray,
                           V_G: float = 0) -> Dict[str, np.ndarray]:
        """Compute EQE and IQE spectra.

        Returns dict with wavelength_nm, EQE, IQE.
        """
        wl = wavelength_range
        E_photon = 1240.0 / wl  # eV

        # Reflection loss from top electrode (~4% front surface)
        top_props = self._get_layer('electrode_top').get('properties', {})
        n_ito = top_props.get('n_opt', 1.9)
        R_front = ((n_ito - 1) / (n_ito + 1)) ** 2

        # Parasitic absorption in HTL, ETL (simplified)
        htl_props = self._get_layer('HTL').get('properties', {})
        etl_props = self._get_layer('ETL').get('properties', {})
        htl_t = htl_props.get('thickness_nm', 200) * 1e-7  # cm
        etl_t = etl_props.get('thickness_nm', 30) * 1e-7

        # Approximate parasitic alpha for transport layers
        alpha_htl = 100  # cm⁻¹ (weak absorption)
        alpha_etl = 50

        T_htl = np.exp(-alpha_htl * htl_t)
        T_etl = np.exp(-alpha_etl * etl_t)

        # Perovskite absorption
        Eg = self._Eg
        alpha_pvk = np.where(E_photon > Eg,
                             self._alpha * np.sqrt(np.maximum(E_photon - Eg, 0) / 0.5),
                             self._alpha * 0.01 * np.exp((E_photon - Eg) / 0.015))
        A_pvk = 1 - np.exp(-alpha_pvk * self._thickness_cm)

        # IQE: collection efficiency, V_G affects it through FET channel
        V_eff = max(V_G - self.fet_config.get('V_th', 0.5), 0)
        collection_base = 0.95
        # Higher V_G → better carrier extraction (up to a point)
        collection_boost = 0.03 * np.tanh(V_eff / 2)
        IQE = np.clip(collection_base + collection_boost, 0, 1.0) * np.ones_like(wl)

        # Sub-bandgap: no collection
        IQE = np.where(E_photon > Eg * 0.95, IQE, IQE * np.exp(-((Eg - E_photon) / 0.05) ** 2))

        # EQE = (1-R) * T_parasitic * A_pvk * IQE
        EQE = (1 - R_front) * T_htl * A_pvk * IQE

        # Ensure bounds
        EQE = np.clip(EQE, 0, 1.0)
        IQE = np.clip(IQE, 0, 1.0)

        return {'wavelength_nm': wl, 'EQE': EQE, 'IQE': IQE}

    # ── Parasitic loss analysis ─────────────────────────────────────────

    def parasitic_loss_analysis(self) -> Dict[str, float]:
        """Quantify all parasitic losses.

        Returns dict mapping loss_name → loss fraction (0-1 relative to incident).
        """
        # Optical losses
        top_props = self._get_layer('electrode_top').get('properties', {})
        n_ito = top_props.get('n_opt', 1.9)
        reflection_loss = ((n_ito - 1) / (n_ito + 1)) ** 2

        htl_t = self._get_layer('HTL').get('properties', {}).get('thickness_nm', 200) * 1e-7
        etl_t = self._get_layer('ETL').get('properties', {}).get('thickness_nm', 30) * 1e-7
        parasitic_htl = 1 - np.exp(-100 * htl_t)  # ~2% for 200nm
        parasitic_etl = 1 - np.exp(-50 * etl_t)

        # Electrical losses
        Rs_loss = self._Rs_electrode / 1e4  # fraction (series R relative to Rsh)
        fet_on_R = 1.0 / (self.fet_config.get('mu_fe', 20) * self._C_ox * 2 * 1e3 + 1e-30)
        fet_loss = fet_on_R / 1e4

        # Recombination losses
        # SRH in bulk
        pvk_traps = 1e15  # typical trap density cm⁻³
        srh_loss = pvk_traps / 1e18 * 0.05  # rough scaling

        # Interface recombination
        etl_traps = self._get_layer('ETL').get('properties', {}).get('trap_density', 1e16)
        htl_traps = self._get_layer('HTL').get('properties', {}).get('trap_density', 5e15)
        interface_loss = (etl_traps + htl_traps) / 1e18 * 0.03

        # Auger (negligible for perovskites at 1 sun)
        auger_loss = 0.001

        return {
            'reflection': float(reflection_loss),
            'parasitic_absorption_HTL': float(parasitic_htl),
            'parasitic_absorption_ETL': float(parasitic_etl),
            'series_resistance': float(Rs_loss),
            'FET_on_resistance': float(fet_loss),
            'SRH_bulk': float(srh_loss),
            'interface_recombination': float(interface_loss),
            'Auger': float(auger_loss),
        }

    # ── Temperature coefficient ─────────────────────────────────────────

    def temperature_coefficient(self, T_range: np.ndarray, G: float = 1000,
                                V_G_opt: bool = True
                                ) -> Dict[str, np.ndarray]:
        """Compute temperature coefficients.

        G is in W/m² (converted to suns internally).
        Returns dict with T, PCE, V_OC, FF, dPCE_dT, dVOC_dT.
        """
        G_sun = G / 1000.0
        V_arr = np.linspace(0, 1.3, 200)

        PCE_list = []
        VOC_list = []
        FF_list = []

        for T in T_range:
            if V_G_opt:
                # Optimize V_G for each T
                best_pce = 0
                best_res = None
                for vg in np.linspace(0, 5, 11):
                    res = self.jv_characteristics(V_arr, G_sun, T, V_G=vg)
                    if res['PCE'] > best_pce:
                        best_pce = res['PCE']
                        best_res = res
                if best_res is None:
                    best_res = self.jv_characteristics(V_arr, G_sun, T, V_G=0)
            else:
                best_res = self.jv_characteristics(V_arr, G_sun, T, V_G=0)

            PCE_list.append(best_res['PCE'])
            VOC_list.append(best_res['V_OC'])
            FF_list.append(best_res['FF'])

        PCE_arr = np.array(PCE_list)
        VOC_arr = np.array(VOC_list)
        FF_arr = np.array(FF_list)

        # Numerical derivatives
        dT = np.gradient(T_range)
        dPCE_dT = np.gradient(PCE_arr, T_range)
        dVOC_dT = np.gradient(VOC_arr, T_range)
        dFF_dT = np.gradient(FF_arr, T_range)

        return {
            'T': T_range,
            'PCE': PCE_arr, 'V_OC': VOC_arr, 'FF': FF_arr,
            'dPCE_dT': dPCE_dT, 'dVOC_dT': dVOC_dT, 'dFF_dT': dFF_dT,
        }
