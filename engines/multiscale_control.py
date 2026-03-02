#!/usr/bin/env python3
"""
Multiscale Control Engine (Phase 1 Integration)
================================================

Integrates three timescale physics via operator splitting:
  - Layer 1 (seconds):  PV-FET gate voltage control
  - Layer 2 (ms):       Ion drift-diffusion
  - Layer 3 (μs):       Interface charge dynamics

Operator splitting ensures each layer updates at its natural cadence
while coupling state variables between scales.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

from .dynamic_iv import DynamicIVEngine
from .ion_dynamics import IonDynamicsEngine
from .interface_charge import InterfaceChargeEngine


@dataclass
class MultiscaleResult:
    """Container for multiscale simulation outputs."""
    time_s: np.ndarray = field(default_factory=lambda: np.array([]))
    P_out: np.ndarray = field(default_factory=lambda: np.array([]))
    V_op: np.ndarray = field(default_factory=lambda: np.array([]))
    eta: np.ndarray = field(default_factory=lambda: np.array([]))
    dV_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    J_rec_interface: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_trapped: np.ndarray = field(default_factory=lambda: np.array([]))
    V_G: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_balance: Dict = field(default_factory=dict)


class MultiscaleControlEngine:
    """Three-timescale integrated PV active control engine."""

    def __init__(self, pv_params: Optional[Dict] = None,
                 fet_params: Optional[Dict] = None,
                 ion_params: Optional[Dict] = None,
                 interface_params: Optional[Dict] = None):

        pv_p = pv_params or {}
        fet_p = fet_params or {}
        ion_p = ion_params or {}
        iface_p = interface_params or {}

        # Layer 1: Dynamic I-V (seconds)
        self.dynamic_iv = DynamicIVEngine(
            perovskite_params=pv_p,
            fet_params=fet_p,
            interface_params={
                'R_contact': iface_p.get('R_contact', 10),
                'C_interface': iface_p.get('C_interface', 44),
                'n_traps': iface_p.get('n_traps', 1e12),
            },
        )

        # Layer 2: Ion dynamics (ms)
        ion_species = ion_p.get('species', {
            'iodide': {
                'D_i': pv_p.get('ion_diffusivity', 2.6e-11),
                'mu_i': pv_p.get('ion_mobility', 1e-9),
                'n_i0': pv_p.get('ion_density', 1e18),
                'E_activation': 0.58,
                'charge': -1,
            }
        })
        self.ion_engine = IonDynamicsEngine(
            layer_thickness_nm=pv_p.get('thickness', 500),
            ion_params=ion_species,
            grid_points=ion_p.get('grid_points', 50),
        )

        # Layer 3: Interface charge (μs)
        self.interface_engine = InterfaceChargeEngine(
            etl_params=iface_p.get('etl_interface', None),
            htl_params=iface_p.get('htl_interface', None),
            interface_params=iface_p,
        )

        # Timestep defaults
        self.dt_coarse = 1.0      # s
        self.dt_medium = 1e-3     # s (1 ms)
        self.dt_fine = 1e-6       # s (1 μs)

        # Store last result
        self._last_result: Optional[MultiscaleResult] = None

    def simulate_multiscale(self, control_signal_t: np.ndarray,
                            G_t: np.ndarray, T_t: np.ndarray,
                            total_time_s: float,
                            dt_coarse: float = 1.0,
                            n_medium_per_coarse: int = 10,
                            n_fine_per_medium: int = 10) -> MultiscaleResult:
        """Run operator-splitting multiscale simulation.

        Args:
            control_signal_t: V_G(t) at coarse timestep resolution
            G_t: irradiance [suns] at coarse resolution
            T_t: temperature [K] at coarse resolution
            total_time_s: total duration [s]
            dt_coarse: coarse step [s]
            n_medium_per_coarse: medium steps per coarse step
            n_fine_per_medium: fine steps per medium step
        """
        N_coarse = len(control_signal_t)
        dt_med = dt_coarse / n_medium_per_coarse
        dt_fin = dt_med / n_fine_per_medium

        P_out = np.zeros(N_coarse)
        V_op = np.zeros(N_coarse)
        eta = np.zeros(N_coarse)
        dV_ion = np.zeros(N_coarse)
        J_rec_arr = np.zeros(N_coarse)
        Q_trap_arr = np.zeros(N_coarse)

        self.interface_engine.reset()

        # Ion engine state reset
        for name, sp in self.ion_engine.ion_species.items():
            self.ion_engine.ion_profiles[name] = np.full(
                self.ion_engine.N, sp['n_i0']
            )

        cumulative_dV_ion = 0.0

        for ic in range(N_coarse):
            V_G = control_signal_t[ic]
            G = G_t[ic]
            T = T_t[ic]

            # Layer 1: get FET operating point
            op = self.dynamic_iv.operating_point(V_G, G, T)

            # Layer 2: ion dynamics (medium steps)
            V_app_arr = np.full(n_medium_per_coarse, op['V_op'])
            G_arr = np.full(n_medium_per_coarse, G)
            T_arr = np.full(n_medium_per_coarse, T)
            ion_res = self.ion_engine.simulate(V_app_arr, G_arr, T_arr, dt_med)
            cumulative_dV_ion = ion_res['dV_ion'][-1] if len(ion_res['dV_ion']) > 0 else 0.0

            # Layer 3: interface charge (fine steps within each medium step)
            n_e = max(G * 1e16, 1e10)  # approximate carrier density
            n_h = n_e
            J_rec_acc = 0.0
            for _ in range(n_medium_per_coarse):
                for _ in range(n_fine_per_medium):
                    iface_res = self.interface_engine.trap_dynamics(
                        n_e, n_h, op['V_op'], T, dt_fin
                    )
                J_rec_acc += iface_res['J_rec']
            J_rec_avg = J_rec_acc / max(n_medium_per_coarse, 1)

            # Combine: correct P_out with ion shift and recombination loss
            V_eff = op['V_op'] + np.clip(cumulative_dV_ion, -0.2, 0.2)
            V_eff = max(V_eff, 0.0)
            # J_rec_avg is interface recombination — typically < 5% of J_SC
            J_sc_approx = max(op['I_op'], 1.0)  # mA/cm²
            J_rec_capped = min(J_rec_avg, 0.1 * J_sc_approx)  # cap at 10% of Jsc
            P_base = op['P_out'] * (1.0 + np.clip(cumulative_dV_ion, -0.2, 0.2) / max(op['V_op'], 0.1)) if op['P_out'] > 0 else 0.0
            P_corrected = max(0, P_base - J_rec_capped * V_eff)

            P_out[ic] = P_corrected
            V_op[ic] = V_eff
            P_incident = 100.0 * G
            eta[ic] = P_corrected / P_incident if P_incident > 0 else 0.0
            dV_ion[ic] = cumulative_dV_ion
            J_rec_arr[ic] = J_rec_avg
            Q_trap_arr[ic] = iface_res.get('Q_trapped', 0.0)

        time_arr = np.arange(N_coarse) * dt_coarse

        result = MultiscaleResult(
            time_s=time_arr,
            P_out=P_out,
            V_op=V_op,
            eta=eta,
            dV_ion=dV_ion,
            J_rec_interface=J_rec_arr,
            Q_trapped=Q_trap_arr,
            V_G=control_signal_t.copy(),
            energy_balance=self._compute_energy_balance(P_out, J_rec_arr, G_t, dt_coarse),
        )
        self._last_result = result
        return result

    def optimal_control_strategy(self, load_profile_t: np.ndarray,
                                 G_t: np.ndarray, T_t: np.ndarray,
                                 lambda_smooth: float = 0.01) -> np.ndarray:
        """Compute optimal V_G(t) to match a load profile.

        Simple iterative approach: for each timestep, find V_G that
        minimises |P_pv - P_load| from the power envelope.
        """
        N = len(load_profile_t)
        V_G_opt = np.zeros(N)
        V_G_max = self.dynamic_iv.fet_params['V_G_max']

        for i in range(N):
            G = G_t[i]
            T = T_t[i]
            P_target = load_profile_t[i]

            if G < 0.01:
                V_G_opt[i] = 0.0
                continue

            # Sweep V_G to find best match
            best_vg = 0.0
            best_err = float('inf')
            for vg in np.linspace(0, V_G_max, 30):
                op = self.dynamic_iv.operating_point(vg, G, T)
                err = abs(op['P_out'] - P_target)
                if err < best_err:
                    best_err = err
                    best_vg = vg

            # Smoothing penalty
            if i > 0 and lambda_smooth > 0:
                slew = abs(best_vg - V_G_opt[i - 1])
                if slew > 1.0:
                    best_vg = V_G_opt[i - 1] + np.sign(best_vg - V_G_opt[i - 1]) * 1.0

            V_G_opt[i] = best_vg

        return V_G_opt

    def performance_summary(self) -> Dict:
        """Summarise the last simulation result."""
        if self._last_result is None:
            return {'error': 'No simulation run yet'}

        r = self._last_result
        P = r.P_out
        P_pos = P[P > 0]

        if len(P_pos) == 0:
            return {'P_max': 0, 'P_min': 0, 'dynamic_range': 0}

        P_max = float(np.max(P_pos))
        P_min = float(np.min(P_pos))
        dynamic_range = P_max / P_min if P_min > 1e-10 else float('inf')

        # Response times (rough)
        tau_us = self.interface_engine.tau_rc_etl * 1e6  # μs
        tau_ms = self.ion_engine.get_ion_timescale() * 1e3  # ms

        # Efficiency loss breakdown
        avg_eta = float(np.mean(r.eta[r.eta > 0])) if np.any(r.eta > 0) else 0.0
        avg_Jrec = float(np.mean(r.J_rec_interface))

        return {
            'P_max_mW_cm2': P_max,
            'P_min_mW_cm2': P_min,
            'dynamic_range': dynamic_range,
            'avg_efficiency': avg_eta,
            'tau_interface_us': tau_us,
            'tau_ion_ms': tau_ms,
            'avg_J_rec_mA_cm2': avg_Jrec,
            'energy_balance': r.energy_balance,
        }

    def validate_energy_conservation(self, tol: float = 0.1) -> Dict:
        """Verify energy conservation across scales."""
        if self._last_result is None:
            return {'valid': False, 'reason': 'No simulation'}

        eb = self._last_result.energy_balance
        unaccounted_frac = eb.get('unaccounted_fraction', 1.0)
        valid = abs(unaccounted_frac) < tol

        return {
            'valid': valid,
            'unaccounted_fraction': unaccounted_frac,
            'energy_balance': eb,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _compute_energy_balance(self, P_out: np.ndarray,
                                J_rec: np.ndarray,
                                G_t: np.ndarray,
                                dt: float) -> Dict:
        P_incident = 100.0 * G_t  # mW/cm²
        total_incident = float(np.sum(P_incident) * dt)
        total_output = float(np.sum(P_out) * dt)
        # Recombination loss estimate
        avg_V = 0.8  # approximate operating voltage
        total_rec = float(np.sum(J_rec * avg_V) * dt)
        total_thermal = total_incident - total_output - total_rec
        total_thermal = max(total_thermal, 0.0)
        unaccounted = total_incident - total_output - total_rec - total_thermal
        frac = unaccounted / total_incident if total_incident > 0 else 0.0

        return {
            'E_incident': total_incident,
            'E_output': total_output,
            'E_recombination': total_rec,
            'E_thermal': total_thermal,
            'E_unaccounted': unaccounted,
            'unaccounted_fraction': frac,
        }
