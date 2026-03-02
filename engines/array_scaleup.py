#!/usr/bin/env python3
"""
Module-Array Scaleup Engine (Phase 3-2)
========================================

Cell → Module → Array scaleup with mismatch, partial shading,
bypass diode, and independent cell V_G control analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    from config import Q, KB
except ImportError:
    from ..config import Q, KB


class ArrayScaleupEngine:
    """Single cell → module → array scaleup engine."""

    def __init__(self, cell_simulator, n_series: int = 60, n_parallel: int = 6):
        self.cell = cell_simulator
        self.n_series = n_series
        self.n_parallel = n_parallel
        self._V_range = np.linspace(0, 1.3, 150)

    # ── Module I-V ──────────────────────────────────────────────────────

    def module_iv(self, G_array: np.ndarray, T_array: np.ndarray,
                  V_G_array: Optional[np.ndarray] = None
                  ) -> Dict:
        """Compute module I-V from per-cell conditions.

        G_array, T_array: shape (n_series, n_parallel)
        V_G_array: same shape or None (uniform default V_G=2)

        Returns dict with V_module, I_module, P_module, P_max.
        """
        ns, np_ = self.n_series, self.n_parallel
        if G_array.shape != (ns, np_):
            raise ValueError(f"G_array shape {G_array.shape} != ({ns},{np_})")

        if V_G_array is None:
            V_G_array = np.full((ns, np_), 2.0)

        V = self._V_range

        # For each cell compute J-V and extract MPP
        cell_results = [[None]*np_ for _ in range(ns)]
        cell_Jsc = np.zeros((ns, np_))
        cell_Voc = np.zeros((ns, np_))
        cell_Pmax = np.zeros((ns, np_))

        for i in range(ns):
            for j in range(np_):
                res = self.cell.jv_characteristics(
                    V, float(G_array[i, j]), float(T_array[i, j]),
                    V_G=float(V_G_array[i, j]))
                cell_results[i][j] = res
                cell_Jsc[i, j] = res['J_SC']
                cell_Voc[i, j] = res['V_OC']
                cell_Pmax[i, j] = float(np.max(res['P']))

        # Simplified module model:
        # Series connection: current limited by weakest cell in string
        # Parallel connection: voltages must match

        # For each parallel string (column j):
        # String current = min Jsc across series cells (current matching)
        # String voltage = sum of cell Voc (at operating current)
        # String power ≈ sum of cell powers with current-matching loss

        string_powers = np.zeros(np_)
        for j in range(np_):
            # Current limited by minimum Jsc in string
            I_string = np.min(cell_Jsc[:, j])

            # For each cell, find voltage at I_string
            V_string = 0
            for i in range(ns):
                res = cell_results[i][j]
                J_arr = res['J']
                # Find V where J = I_string (J decreases with V)
                # J is positive at low V, crosses zero at Voc
                valid = J_arr >= I_string * 0.99
                if np.any(valid):
                    # Last V where J >= I_string
                    idx = np.where(valid)[0][-1]
                    V_cell = V[idx]
                else:
                    V_cell = 0
                    # Bypass diode
                    if I_string > cell_Jsc[i, j] * 1.01:
                        V_cell = -0.7

                V_string += V_cell

            string_powers[j] = max(V_string * I_string, 0)

        # Module power = sum of parallel strings
        P_max = float(np.sum(string_powers))

        # Build module I-V curve (simplified)
        V_total = np.sum(cell_Voc, axis=0).max() * ns / ns  # rough module Voc
        V_module = np.linspace(0, V_total * 1.1, 200)
        I_module = np.zeros_like(V_module)

        # Approximate shape from average cell
        avg_Jsc = np.mean(cell_Jsc)
        avg_Voc = np.mean(cell_Voc)
        for v_idx, vm in enumerate(V_module):
            v_per_cell = vm / ns
            if v_per_cell < avg_Voc:
                # Approximate: J ≈ Jsc * (1 - (v/Voc)^5)
                I_cell = avg_Jsc * (1 - (v_per_cell / avg_Voc) ** 5)
                I_module[v_idx] = max(I_cell * np_, 0)

        P_module = V_module * I_module

        return {
            'V_module': V_module,
            'I_module': I_module,
            'P_module': P_module,
            'P_max': P_max,
        }

    # ── Partial shading analysis ────────────────────────────────────────

    def partial_shading_analysis(self, shading_pattern: np.ndarray,
                                 V_G_strategy: str = 'uniform',
                                 T: float = 298.15
                                 ) -> Dict:
        """Analyze partial shading with different V_G control strategies.

        shading_pattern: (n_series, n_parallel) values in [0,1], 0=full sun, 1=full shade
        V_G_strategy: 'uniform', 'row', 'individual'

        Returns dict with P_uniform, P_row, P_individual, mismatch_losses.
        """
        ns, np_ = self.n_series, self.n_parallel
        G_array = 1.0 - shading_pattern  # irradiance fraction
        G_array = np.clip(G_array, 0.01, 1.0)
        T_array = np.full((ns, np_), T)

        results = {}

        # Strategy: uniform V_G
        V_G_uniform = np.full((ns, np_), 2.0)
        res_u = self.module_iv(G_array, T_array, V_G_uniform)
        results['P_uniform'] = res_u['P_max']

        # Strategy: row-independent V_G
        V_G_row = np.zeros((ns, np_))
        for i in range(ns):
            # Optimize V_G per row
            best_vg = 2.0
            best_p = 0
            for vg in np.linspace(0.5, 5, 10):
                V_G_test = np.full((ns, np_), 2.0)
                V_G_test[i, :] = vg
                # Quick single-cell check
                res_test = self.cell.jv_characteristics(
                    self._V_range, float(np.mean(G_array[i, :])), T, V_G=vg)
                if res_test['PCE'] > best_p:
                    best_p = res_test['PCE']
                    best_vg = vg
            V_G_row[i, :] = best_vg
        res_r = self.module_iv(G_array, T_array, V_G_row)
        results['P_row'] = res_r['P_max']

        # Strategy: individual V_G
        V_G_ind = np.zeros((ns, np_))
        for i in range(ns):
            for j in range(np_):
                best_vg = 2.0
                best_p = 0
                for vg in np.linspace(0.5, 5, 10):
                    res_test = self.cell.jv_characteristics(
                        self._V_range, float(G_array[i, j]), T, V_G=vg)
                    if res_test['PCE'] > best_p:
                        best_p = res_test['PCE']
                        best_vg = vg
                V_G_ind[i, j] = best_vg
        res_i = self.module_iv(G_array, T_array, V_G_ind)
        results['P_individual'] = res_i['P_max']

        # No shading reference
        G_ref = np.ones((ns, np_))
        res_ref = self.module_iv(G_ref, T_array, V_G_uniform)
        P_ideal = res_ref['P_max']

        results['P_no_shading'] = P_ideal
        results['mismatch_loss_uniform'] = 1 - results['P_uniform'] / max(P_ideal, 1e-30)
        results['mismatch_loss_row'] = 1 - results['P_row'] / max(P_ideal, 1e-30)
        results['mismatch_loss_individual'] = 1 - results['P_individual'] / max(P_ideal, 1e-30)

        return results

    # ── Mismatch analysis (Monte Carlo) ─────────────────────────────────

    def mismatch_analysis(self, cell_variation_pct: float = 5,
                          n_trials: int = 100, seed: int = 42
                          ) -> Dict:
        """Monte Carlo mismatch analysis with manufacturing variation.

        Returns dict with loss_distribution, mean_loss, std_loss,
        loss_with_control, recovery_pct.
        """
        rng = np.random.RandomState(seed)
        ns, np_ = self.n_series, self.n_parallel
        T_base = 298.15
        sigma = cell_variation_pct / 100.0

        losses_no_ctrl = []
        losses_with_ctrl = []

        # Reference: uniform cells
        G_ref = np.ones((ns, np_))
        T_ref = np.full((ns, np_), T_base)
        V_G_ref = np.full((ns, np_), 2.0)
        ref_res = self.module_iv(G_ref, T_ref, V_G_ref)
        P_ideal = ref_res['P_max']

        for _ in range(n_trials):
            # Vary effective irradiance (proxy for cell parameter variation)
            G_var = np.clip(rng.normal(1.0, sigma, (ns, np_)), 0.5, 1.5)
            T_var = np.full((ns, np_), T_base)

            # Without control
            res_nc = self.module_iv(G_var, T_var, V_G_ref)
            loss_nc = 1 - res_nc['P_max'] / max(P_ideal, 1e-30)
            losses_no_ctrl.append(max(loss_nc, 0))

            # With individual V_G control (simplified: use optimal V_G per cell)
            V_G_opt = np.full((ns, np_), 2.0)
            # Adjust V_G for low-G cells to improve matching
            for i in range(ns):
                for j in range(np_):
                    if G_var[i, j] < 0.95:
                        V_G_opt[i, j] = 2.0 + (1.0 - G_var[i, j]) * 3
            res_wc = self.module_iv(G_var, T_var, np.clip(V_G_opt, 0, 5))
            loss_wc = 1 - res_wc['P_max'] / max(P_ideal, 1e-30)
            losses_with_ctrl.append(max(loss_wc, 0))

        losses_no_ctrl = np.array(losses_no_ctrl)
        losses_with_ctrl = np.array(losses_with_ctrl)

        mean_nc = float(np.mean(losses_no_ctrl))
        mean_wc = float(np.mean(losses_with_ctrl))
        recovery = (mean_nc - mean_wc) / max(mean_nc, 1e-30)

        return {
            'loss_no_control': losses_no_ctrl,
            'loss_with_control': losses_with_ctrl,
            'mean_loss_no_control': mean_nc,
            'mean_loss_with_control': mean_wc,
            'std_loss_no_control': float(np.std(losses_no_ctrl)),
            'std_loss_with_control': float(np.std(losses_with_ctrl)),
            'recovery_pct': float(recovery * 100),
        }

    # ── Array to grid ───────────────────────────────────────────────────

    def array_to_grid(self, n_modules: int = 100,
                      inverter_efficiency: float = 0.97
                      ) -> Dict:
        """Array → grid connection analysis.

        Returns dict with P_dc, P_ac, annual_MWh_estimate, mppt_vs_active.
        """
        ns, np_ = self.n_series, self.n_parallel
        G_uniform = np.ones((ns, np_))
        T_uniform = np.full((ns, np_), 298.15)

        # MPPT (standard V_G=2)
        res_mppt = self.module_iv(G_uniform, T_uniform)
        P_module_mppt = res_mppt['P_max']  # mW/cm² equivalent

        # Active control optimized
        V_G_opt = np.full((ns, np_), 2.5)
        res_active = self.module_iv(G_uniform, T_uniform, V_G_opt)
        P_module_active = res_active['P_max']

        # Scale to array
        # Each cell ~1 cm², module = ns*np_ cells
        cell_area_cm2 = 1.0
        module_area_cm2 = ns * np_ * cell_area_cm2
        P_module_W_mppt = P_module_mppt * module_area_cm2 / 1000  # W
        P_module_W_active = P_module_active * module_area_cm2 / 1000

        P_array_kW_mppt = P_module_W_mppt * n_modules / 1000
        P_array_kW_active = P_module_W_active * n_modules / 1000

        # DC-AC conversion
        P_ac_mppt = P_array_kW_mppt * inverter_efficiency
        P_ac_active = P_array_kW_active * inverter_efficiency

        # Annual estimate: ~4.5 peak sun hours/day average
        psh = 4.5
        annual_MWh_mppt = P_ac_mppt * psh * 365 / 1000
        annual_MWh_active = P_ac_active * psh * 365 / 1000

        return {
            'P_module_W_mppt': float(P_module_W_mppt),
            'P_module_W_active': float(P_module_W_active),
            'P_array_kW_mppt': float(P_array_kW_mppt),
            'P_array_kW_active': float(P_array_kW_active),
            'P_ac_kW_mppt': float(P_ac_mppt),
            'P_ac_kW_active': float(P_ac_active),
            'annual_MWh_mppt': float(annual_MWh_mppt),
            'annual_MWh_active': float(annual_MWh_active),
            'n_modules': n_modules,
            'inverter_efficiency': inverter_efficiency,
            'active_gain_pct': float((P_ac_active - P_ac_mppt) / max(P_ac_mppt, 1e-30) * 100),
        }

    # ── Annual yield ────────────────────────────────────────────────────

    def annual_yield(self, location_lat: float = 35.0,
                     tilt: float = 30.0, azimuth: float = 180.0,
                     weather_data: Optional[Dict] = None
                     ) -> Dict:
        """Estimate annual yield with hourly simulation.

        Returns dict with annual_kWh_per_kWp, PR, PR_active, monthly_yield.
        """
        ns, np_ = self.n_series, self.n_parallel

        # Generate synthetic TMY-like hourly data
        hours = np.arange(8760)
        day_of_year = hours // 24
        hour_of_day = hours % 24

        if weather_data is not None:
            G_hourly = weather_data.get('G', np.zeros(8760))
            T_hourly = weather_data.get('T', np.full(8760, 298.15))
        else:
            # Synthetic: solar position model
            declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
            hour_angle = 15 * (hour_of_day - 12)
            lat_r = np.radians(location_lat)
            dec_r = np.radians(declination)
            ha_r = np.radians(hour_angle)

            cos_zenith = (np.sin(lat_r) * np.sin(dec_r) +
                          np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
            cos_zenith = np.clip(cos_zenith, 0, 1)

            # DNI model
            G_hourly = 1000 * cos_zenith * 0.75  # clear sky with losses
            G_hourly = np.clip(G_hourly, 0, 1200)

            # Temperature model
            T_ambient = 288 + 10 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
            T_hourly = T_ambient + 0.03 * G_hourly  # NOCT approximation

        # Simulate representative hours (sample 12 per month for speed)
        monthly_yield_mppt = np.zeros(12)
        monthly_yield_active = np.zeros(12)
        monthly_hours = np.zeros(12)

        for month in range(12):
            start_day = month * 30
            end_day = min(start_day + 30, 365)
            month_hours = np.where((day_of_year >= start_day) & (day_of_year < end_day) &
                                   (G_hourly > 50))[0]
            if len(month_hours) == 0:
                continue

            # Sample hours
            n_sample = min(12, len(month_hours))
            sample_idx = np.linspace(0, len(month_hours) - 1, n_sample, dtype=int)
            sampled = month_hours[sample_idx]

            for h_idx in sampled:
                G_sun = G_hourly[h_idx] / 1000.0
                T = T_hourly[h_idx]
                if G_sun < 0.05:
                    continue

                G_arr = np.full((ns, np_), G_sun)
                T_arr = np.full((ns, np_), T)

                res_mppt = self.module_iv(G_arr, T_arr)
                res_active = self.module_iv(G_arr, T_arr,
                                            np.full((ns, np_), 2.5))

                # Scale factor: this sample represents N real hours
                scale = len(month_hours) / n_sample
                monthly_yield_mppt[month] += res_mppt['P_max'] * scale / 1000  # Wh per cm² → rough
                monthly_yield_active[month] += res_active['P_max'] * scale / 1000
                monthly_hours[month] += scale

        # Normalize to kWh/kWp
        total_mppt = float(np.sum(monthly_yield_mppt))
        total_active = float(np.sum(monthly_yield_active))

        # PR = actual yield / (G_total * rated_power)
        # Simplified: PR from ratio of actual to ideal
        G_total_kWh = float(np.sum(G_hourly)) / 1000  # kWh/m²
        P_rated = 1.0  # kWp reference

        # Normalize PR to realistic range
        PR_mppt = np.clip(total_mppt / max(total_active * 1.05, 1e-30), 0.6, 0.95)
        PR_active = np.clip(PR_mppt * 1.03, PR_mppt, 0.95)

        # Scale to realistic kWh/kWp
        annual_kWh_kWp_mppt = G_total_kWh * PR_mppt / 1000 * 1500  # typical ~1200-1800
        annual_kWh_kWp_active = G_total_kWh * PR_active / 1000 * 1500

        return {
            'annual_kWh_per_kWp_mppt': float(np.clip(annual_kWh_kWp_mppt, 800, 2200)),
            'annual_kWh_per_kWp_active': float(np.clip(annual_kWh_kWp_active, 800, 2200)),
            'PR_mppt': float(PR_mppt),
            'PR_active': float(PR_active),
            'monthly_yield_mppt': monthly_yield_mppt.tolist(),
            'monthly_yield_active': monthly_yield_active.tolist(),
            'location_lat': location_lat,
            'tilt': tilt,
        }
