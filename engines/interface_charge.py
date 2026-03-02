#!/usr/bin/env python3
"""
Interface Charge Flushing Engine (Phase 1-3)
=============================================

Models ETL/Perovskite and Perovskite/HTL interface charge accumulation
and flushing dynamics at the μs timescale.

Physics:
    - SRH trap capture/emission at interfaces
    - Frequency-dependent interface capacitance
    - Impedance spectroscopy simulation
    - Voltage-pulse flushing response

Timescale: τ_RC = R_contact × C_interface ~ 1-100 μs
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.constants import elementary_charge as q, Boltzmann as kB


class InterfaceChargeEngine:
    """ETL/Perovskite and Perovskite/HTL interface charge dynamics."""

    def __init__(self, etl_params: Optional[Dict] = None,
                 htl_params: Optional[Dict] = None,
                 interface_params: Optional[Dict] = None):
        etl = etl_params or {}
        htl = htl_params or {}
        iface = interface_params or {}

        # ETL/Perovskite interface (electron traps)
        self.etl = {
            'N_t': etl.get('N_t', 1e17),          # cm⁻³
            'E_t': etl.get('E_t', 0.3),            # eV trap depth from CB
            'sigma_n': etl.get('sigma_n', 1e-15),  # cm² electron capture
            'sigma_p': etl.get('sigma_p', 1e-17),  # cm² hole capture
            'C_geo': etl.get('C_geo', 30e-9),      # F/cm²
        }

        # Perovskite/HTL interface (hole traps)
        self.htl = {
            'N_t': htl.get('N_t', 5e16),
            'E_t': htl.get('E_t', 0.4),
            'sigma_n': htl.get('sigma_n', 1e-17),
            'sigma_p': htl.get('sigma_p', 1e-15),
            'C_geo': htl.get('C_geo', 25e-9),
        }

        self.R_contact = iface.get('R_contact', 5.0)      # Ω·cm²
        self.v_th = iface.get('v_thermal', 1e7)            # cm/s

        # State: trapped charge densities
        self.n_t_etl = 0.0   # cm⁻³ (trapped electrons at ETL interface)
        self.n_t_htl = 0.0   # cm⁻³ (trapped holes at HTL interface)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emission_rate(self, E_t: float, sigma: float, T: float) -> float:
        """Thermal emission rate e_n or e_p [s⁻¹].
        e = σ · v_th · N_c · exp(-E_t / kT)
        Using N_c ~ 1e19 cm⁻³ (effective DOS).
        """
        N_c = 1e19  # cm⁻³
        return sigma * self.v_th * N_c * np.exp(-E_t * q / (kB * T))

    def _flush_tau(self, V_ext: float) -> float:
        """Flushing time constant τ_flush(V_ext) [s].
        Flushing only occurs under strong forward/reverse bias pulse.
        At V_ext=0, τ_flush → ∞ (no flushing, only thermal emission).
        τ_flush = τ_base / max(|V_ext|/V_th_flush, ε)
        """
        tau_base = 100e-6  # 100 μs base flushing time
        V_th_flush = 1.0   # V threshold for significant flushing
        v_ratio = abs(V_ext) / V_th_flush
        if v_ratio < 0.01:
            return 1e6  # effectively infinite — no flushing at low V
        return tau_base / v_ratio

    @property
    def tau_rc_etl(self) -> float:
        """ETL interface RC time constant [s]."""
        return self.R_contact * self.etl['C_geo']

    @property
    def tau_rc_htl(self) -> float:
        """HTL interface RC time constant [s]."""
        return self.R_contact * self.htl['C_geo']

    # ------------------------------------------------------------------
    # Core: trap dynamics
    # ------------------------------------------------------------------
    def trap_dynamics(self, n_e: float, n_h: float, V_ext: float,
                      T: float, dt: float) -> Dict[str, float]:
        """Advance SRH trap dynamics by dt.

        Args:
            n_e: free electron density [cm⁻³]
            n_h: free hole density [cm⁻³]
            V_ext: external voltage [V]
            T: temperature [K]
            dt: time step [s]

        Returns:
            dict with n_t_etl, n_t_htl, J_rec [mA/cm²], Q_trapped [C/cm²]
        """
        # --- ETL/Perovskite (electron traps) ---
        N_t = self.etl['N_t']
        cap_e = self.etl['sigma_n'] * self.v_th * n_e * (N_t - self.n_t_etl)
        em_e = self._emission_rate(self.etl['E_t'], self.etl['sigma_n'], T) * self.n_t_etl
        tau_f = self._flush_tau(V_ext)
        flush_e = self.n_t_etl / tau_f if tau_f > 0 else 0.0

        dnt_etl = (cap_e - em_e - flush_e) * dt
        self.n_t_etl = np.clip(self.n_t_etl + dnt_etl, 0.0, N_t)

        # --- Perovskite/HTL (hole traps) ---
        N_t_h = self.htl['N_t']
        cap_h = self.htl['sigma_p'] * self.v_th * n_h * (N_t_h - self.n_t_htl)
        em_h = self._emission_rate(self.htl['E_t'], self.htl['sigma_p'], T) * self.n_t_htl
        v_ratio_h = abs(V_ext) / 1.0
        tau_f_h = 100e-6 / v_ratio_h if v_ratio_h > 0.01 else 1e6
        flush_h = self.n_t_htl / tau_f_h if tau_f_h > 0 else 0.0

        dnt_htl = (cap_h - em_h - flush_h) * dt
        self.n_t_htl = np.clip(self.n_t_htl + dnt_htl, 0.0, N_t_h)

        # Recombination current (mA/cm²)
        # SRH interface recombination: R = (n·p - ni²) / (τ_p·(n+n1) + τ_n·(p+p1))
        # Simplified: J_rec ∝ trapped charge × emission rate × d_interface
        d_iface = 10e-7  # cm (10 nm interface region)
        R_etl = em_e  # emission from traps = recombination rate at SS
        R_htl = em_h
        J_rec = q * (R_etl + R_htl) * d_iface * 1e3  # A/cm² → mA/cm²

        # Total trapped charge
        Q_trapped = q * (self.n_t_etl + self.n_t_htl) * d_iface  # C/cm²

        return {
            'n_t_etl': self.n_t_etl,
            'n_t_htl': self.n_t_htl,
            'J_rec': J_rec,
            'Q_trapped': Q_trapped,
            'cap_rate_etl': cap_e,
            'cap_rate_htl': cap_h,
            'em_rate_etl': em_e,
            'em_rate_htl': em_h,
        }

    # ------------------------------------------------------------------
    # Capacitance & Impedance
    # ------------------------------------------------------------------
    def interface_capacitance(self, V: float, frequency: float,
                              T: float = 300.0) -> Dict[str, float]:
        """Frequency-dependent interface capacitance.

        C(ω) = C_geo + C_ionic(ω) + C_trap(ω)
        """
        omega = 2 * np.pi * frequency

        C_geo = self.etl['C_geo'] + self.htl['C_geo']

        # Ionic capacitance (Debye relaxation, dominant at low freq)
        tau_ion = 0.01  # s (typical ion relaxation ~10 ms)
        C_ion_0 = 1e-6  # F/cm² (low-freq ionic capacitance)
        C_ionic = C_ion_0 / (1.0 + (omega * tau_ion) ** 2)

        # Trap capacitance (SRH response)
        tau_trap = self.tau_rc_etl  # μs scale
        C_trap_0 = q * (self.etl['N_t'] + self.htl['N_t']) * 10e-7  # rough
        C_trap_0 = min(C_trap_0, 1e-5)  # cap at reasonable value
        C_trap = C_trap_0 / (1.0 + (omega * tau_trap) ** 2)

        C_total = C_geo + C_ionic + C_trap

        return {
            'C_total': C_total,
            'C_geo': C_geo,
            'C_ionic': C_ionic,
            'C_trap': C_trap,
        }

    def impedance_spectrum(self, V_dc: float, freq_range: np.ndarray,
                           T: float = 300.0) -> Dict[str, np.ndarray]:
        """Impedance spectroscopy simulation.

        Z(ω) = R_s + R_rec/(1+jωR_rec·C_μ) + R_ion/(1+jωR_ion·C_ion)
        """
        R_s = 1.0   # Ω·cm² series resistance
        R_rec = 50.0  # Ω·cm² recombination resistance
        C_mu = self.etl['C_geo'] + self.htl['C_geo']  # chemical capacitance

        R_ion = 200.0  # Ω·cm² ionic resistance
        tau_ion = 0.01  # s
        C_ion = tau_ion / R_ion

        omega = 2 * np.pi * freq_range

        Z_rec = R_rec / (1.0 + 1j * omega * R_rec * C_mu)
        Z_ion = R_ion / (1.0 + 1j * omega * R_ion * C_ion)
        Z_total = R_s + Z_rec + Z_ion

        return {
            'frequency': freq_range,
            'Z_real': Z_total.real,
            'Z_imag': Z_total.imag,
            'Z_magnitude': np.abs(Z_total),
            'Z_phase': np.angle(Z_total, deg=True),
            'R_s': R_s,
            'R_rec': R_rec,
            'R_ion': R_ion,
        }

    # ------------------------------------------------------------------
    # Flushing response
    # ------------------------------------------------------------------
    def flush_response(self, V_pulse: float, pulse_width: float,
                       T: float = 300.0, n_e: float = 1e16,
                       n_h: float = 1e16,
                       dt: float = 1e-6) -> Dict[str, np.ndarray]:
        """Simulate trap flushing from a voltage pulse.

        Args:
            V_pulse: pulse voltage [V]
            pulse_width: pulse duration [s]
            T: temperature [K]
            n_e, n_h: carrier densities [cm⁻³]
            dt: simulation timestep [s]

        Returns:
            time, n_t_etl, n_t_htl, J_rec, Q_trapped arrays
        """
        # Pre-fill traps to steady state (V_ext=0)
        self.n_t_etl = self.etl['N_t'] * 0.5
        self.n_t_htl = self.htl['N_t'] * 0.5

        total_time = pulse_width * 5  # observe recovery
        N = max(int(total_time / dt), 100)
        t = np.linspace(0, total_time, N)
        actual_dt = t[1] - t[0] if N > 1 else dt

        nt_etl = np.zeros(N)
        nt_htl = np.zeros(N)
        J_rec = np.zeros(N)
        Q_trap = np.zeros(N)

        for i in range(N):
            V = V_pulse if t[i] < pulse_width else 0.0
            res = self.trap_dynamics(n_e, n_h, V, T, actual_dt)
            nt_etl[i] = res['n_t_etl']
            nt_htl[i] = res['n_t_htl']
            J_rec[i] = res['J_rec']
            Q_trap[i] = res['Q_trapped']

        return {
            'time': t,
            'n_t_etl': nt_etl,
            'n_t_htl': nt_htl,
            'J_rec': J_rec,
            'Q_trapped': Q_trap,
        }

    # ------------------------------------------------------------------
    # Recombination current
    # ------------------------------------------------------------------
    def recombination_current(self, n_trapped_etl: float,
                              n_trapped_htl: float,
                              V: float, T: float) -> float:
        """Interface recombination current [mA/cm²]."""
        d_iface = 10e-7  # cm
        # Recombination ~ emission from traps
        em_etl = self._emission_rate(self.etl['E_t'], self.etl['sigma_n'], T) * n_trapped_etl
        em_htl = self._emission_rate(self.htl['E_t'], self.htl['sigma_p'], T) * n_trapped_htl
        J_rec = q * (em_etl + em_htl) * d_iface * 1e3  # mA/cm²
        return J_rec

    def reset(self):
        """Reset trapped charge state to zero."""
        self.n_t_etl = 0.0
        self.n_t_htl = 0.0
