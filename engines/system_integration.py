#!/usr/bin/env python3
"""
AIDC Microgrid System Integration Engine (Phase 3-3)
=====================================================

PV active control + HESS + AI-EMS integrated system simulation.

System:  PV Array (active) → DC Bus → AIDC Load
                              ↕ HESS (6-layer)
                              ↕ Grid

Economic analysis with 4 scenarios (A-D).
"""

import numpy as np
from typing import Dict, Optional, List

try:
    from config import DEFAULT_CONFIG, Q, KB
except ImportError:
    from ..config import DEFAULT_CONFIG, Q, KB


# ── Default economic parameters ─────────────────────────────────────────
# Sources: IRENA 2023, BloombergNEF 2023, Lazard LCOE 2023

ECON_DEFAULTS = {
    # PV costs ($/W)
    'pv_capex_per_W': 0.35,          # Utility-scale PV module (IRENA 2023)
    'bos_per_W': 0.25,               # Balance of system
    'fet_addon_per_W': 0.05,         # FET integration premium (~15% of module)
    'installation_per_W': 0.10,      # Installation labor

    # HESS costs ($/kWh) — Li-ion + supercap hybrid
    'hess_capex_per_kWh': 250,       # BloombergNEF 2023
    'hess_cycle_life': 5000,         # Cycles at 80% DoD
    'hess_replacement_year': 12,     # Battery replacement cycle

    # O&M ($/kW/year)
    'pv_om_per_kW_year': 15,         # NREL ATB 2023
    'hess_om_per_kW_year': 10,

    # Revenue / grid
    'electricity_price_per_kWh': 0.08,  # $/kWh wholesale
    'grid_export_price_per_kWh': 0.05,  # $/kWh feed-in

    # Carbon
    'grid_carbon_kg_per_kWh': 0.4,   # Global average (IEA 2023)
    'hess_embodied_carbon_kg_per_kWh': 100,  # Li-ion manufacturing

    # System
    'degradation_rate': 0.005,       # 0.5%/year
    'discount_rate': 0.08,           # WACC
    'system_lifetime_years': 25,

    # AIDC
    'aidc_load_MW': 10,              # Base IT load
    'pue': 1.2,
}


class SystemIntegrationEngine:
    """PV active control + HESS + AIDC microgrid integration."""

    def __init__(self, array_engine,
                 hess_config: Optional[Dict] = None,
                 aidc_config: Optional[Dict] = None):
        self.array = array_engine
        self.hess = hess_config or {}
        self.aidc = aidc_config or {}

        # HESS parameters
        self.hess_capacity_kWh = self.hess.get('capacity_kWh', 40000)  # 40 MWh default
        self.hess_max_power_kW = self.hess.get('max_power_kW', 10000)  # 10 MW
        self.hess_efficiency = self.hess.get('round_trip_efficiency', 0.88)

        # AIDC parameters
        self.aidc_base_MW = self.aidc.get('base_load_MW',
                                          ECON_DEFAULTS['aidc_load_MW'])
        self.pue = self.aidc.get('pue', ECON_DEFAULTS['pue'])

    # ── 24h system simulation ───────────────────────────────────────────

    def simulate_24h(self, date: str = '2024-06-21',
                     location: Dict = None,
                     aidc_schedule: Optional[np.ndarray] = None
                     ) -> Dict:
        """24-hour simulation with 1-minute resolution, 4 scenarios.

        Returns dict with time_min, scenarios {A,B,C,D} each containing:
            pv_kW, load_kW, hess_kW, grid_kW, soc, self_consumption.
        """
        location = location or {'lat': 35.0}
        N = 1440  # minutes in 24h
        t_min = np.arange(N)
        t_hour = t_min / 60.0

        # ── Solar irradiance profile ──
        lat = location.get('lat', 35.0)
        # Day of year from date string
        try:
            from datetime import datetime as _dt
            doy = _dt.strptime(date, '%Y-%m-%d').timetuple().tm_yday
        except Exception:
            doy = 172  # summer solstice

        declination = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
        hour_angle = 15 * (t_hour - 12)
        lat_r = np.radians(lat)
        dec_r = np.radians(declination)
        ha_r = np.radians(hour_angle)

        cos_zenith = (np.sin(lat_r) * np.sin(dec_r) +
                      np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
        cos_zenith = np.clip(cos_zenith, 0, 1)
        G_kW_m2 = cos_zenith * 0.85  # Clear sky

        # PV array rating (MW)
        pv_rating_MW = self.aidc_base_MW * 1.5  # 1.5x load
        # Instantaneous PV output per scenario
        pv_A = G_kW_m2 * pv_rating_MW * 1000  # kW, MPPT only (PR=0.80)
        pv_A *= 0.80

        pv_B = G_kW_m2 * pv_rating_MW * 1000 * 0.83  # FET control (+3%)
        pv_C = G_kW_m2 * pv_rating_MW * 1000 * 0.86  # FET+Ion+Interface (+6%)
        pv_D = G_kW_m2 * pv_rating_MW * 1000 * 0.88  # +ML optimization (+8%)

        # ── AIDC load profile ──
        if aidc_schedule is not None and len(aidc_schedule) == N:
            load_kW = aidc_schedule
        else:
            base_kW = self.aidc_base_MW * 1000 * self.pue
            # Diurnal pattern: higher during business hours
            diurnal = 1 + 0.15 * np.sin(2 * np.pi * (t_hour - 14) / 24)
            # Random fluctuation
            rng = np.random.RandomState(42)
            noise = rng.normal(0, 0.03, N)
            load_kW = base_kW * (diurnal + noise)
            load_kW = np.clip(load_kW, base_kW * 0.7, base_kW * 1.3)

        # ── Simulate each scenario ──
        scenarios = {}
        pv_profiles = {'A': pv_A, 'B': pv_B, 'C': pv_C, 'D': pv_D}

        # Scenario D uses smaller HESS
        hess_caps = {
            'A': self.hess_capacity_kWh,
            'B': self.hess_capacity_kWh,
            'C': self.hess_capacity_kWh,
            'D': self.hess_capacity_kWh * 0.75,  # 25% smaller
        }
        hess_powers = {
            'A': self.hess_max_power_kW,
            'B': self.hess_max_power_kW,
            'C': self.hess_max_power_kW,
            'D': self.hess_max_power_kW * 0.75,
        }

        for scen in ['A', 'B', 'C', 'D']:
            pv = pv_profiles[scen]
            cap = hess_caps[scen]
            max_p = hess_powers[scen]
            eta = self.hess_efficiency

            soc = np.zeros(N)
            soc[0] = cap * 0.5  # Start at 50%
            hess_kW = np.zeros(N)
            grid_kW = np.zeros(N)
            self_consumed = np.zeros(N)

            for i in range(N):
                surplus = pv[i] - load_kW[i]

                if surplus > 0:
                    # Excess PV: charge HESS, then export
                    charge = min(surplus * np.sqrt(eta), max_p,
                                 (cap - soc[max(i - 1, 0)]) * 60)  # kW·min capacity
                    hess_kW[i] = charge
                    remaining = surplus - charge / np.sqrt(eta)
                    grid_kW[i] = -remaining  # negative = export
                    self_consumed[i] = min(pv[i], load_kW[i]) + charge / np.sqrt(eta)
                else:
                    # Deficit: discharge HESS, then import
                    deficit = -surplus
                    discharge = min(deficit, max_p,
                                    soc[max(i - 1, 0)] * 60)
                    hess_kW[i] = -discharge * np.sqrt(eta)
                    remaining_deficit = deficit - discharge * np.sqrt(eta)
                    grid_kW[i] = remaining_deficit  # positive = import
                    self_consumed[i] = pv[i] + discharge * np.sqrt(eta)

                # Update SOC
                if i < N - 1:
                    soc[i + 1] = np.clip(
                        soc[max(i, 0)] + hess_kW[i] / 60,  # kWh (1 min step)
                        0, cap)
                soc[i] = np.clip(soc[max(i - 1, 0)] + hess_kW[i] / 60, 0, cap)

            # Self-consumption ratio
            total_pv = float(np.sum(pv)) / 60  # kWh
            total_self = float(np.sum(np.clip(self_consumed, 0, None))) / 60
            sc_ratio = total_self / max(total_pv, 1e-30)
            sc_ratio = np.clip(sc_ratio, 0, 1)

            total_load = float(np.sum(load_kW)) / 60
            total_import = float(np.sum(np.clip(grid_kW, 0, None))) / 60
            self_sufficiency = 1 - total_import / max(total_load, 1e-30)
            self_sufficiency = np.clip(self_sufficiency, 0, 1)

            scenarios[scen] = {
                'pv_kW': pv,
                'load_kW': load_kW,
                'hess_kW': hess_kW,
                'grid_kW': grid_kW,
                'soc': soc,
                'self_consumption': float(sc_ratio),
                'self_sufficiency': float(self_sufficiency),
                'total_pv_kWh': total_pv,
                'total_load_kWh': total_load,
                'total_import_kWh': total_import,
                'total_export_kWh': float(np.sum(np.clip(-grid_kW, 0, None))) / 60,
                'hess_capacity_kWh': cap,
            }

        return {'time_min': t_min, 'scenarios': scenarios}

    # ── HESS sizing optimization ────────────────────────────────────────

    def hess_sizing_optimization(self, target_self_consumption: float = 0.9
                                 ) -> Dict:
        """Find optimal HESS size for each scenario to hit target SC.

        Returns dict with scenario → {capacity_kWh, reduction_pct}.
        """
        results = {}

        for scen_label, pr_factor in [('A', 0.80), ('B', 0.83),
                                       ('C', 0.86), ('D', 0.88)]:
            # Binary search for HESS capacity
            lo, hi = 1000, 100000  # kWh range
            best_cap = hi

            for _ in range(20):
                mid = (lo + hi) / 2
                self.hess_capacity_kWh = mid
                self.hess_max_power_kW = mid / 4  # 4h duration
                res = self.simulate_24h()
                sc = res['scenarios'][scen_label]['self_consumption']
                if sc >= target_self_consumption:
                    best_cap = mid
                    hi = mid
                else:
                    lo = mid

            results[scen_label] = {
                'capacity_kWh': float(best_cap),
                'target_sc': target_self_consumption,
            }

        # Reduction relative to scenario A
        cap_A = results['A']['capacity_kWh']
        for s in results:
            results[s]['reduction_pct'] = float(
                (1 - results[s]['capacity_kWh'] / max(cap_A, 1)) * 100)

        # Reset
        self.hess_capacity_kWh = self.hess.get('capacity_kWh', 40000)
        self.hess_max_power_kW = self.hess.get('max_power_kW', 10000)

        return results

    # ── Economic comparison ─────────────────────────────────────────────

    def economic_comparison(self, years: int = 25,
                            discount_rate: float = 0.08
                            ) -> Dict:
        """NPV/LCOE/IRR analysis for 4 scenarios.

        Returns dict with scenario → {CAPEX, OPEX_annual, LCOE, NPV, IRR, payback}.
        """
        econ = ECON_DEFAULTS.copy()
        pv_MW = self.aidc_base_MW * 1.5

        results = {}

        scenario_params = {
            'A': {'pr': 0.80, 'hess_factor': 1.0, 'fet_cost': False, 'ml_cost': False},
            'B': {'pr': 0.83, 'hess_factor': 1.0, 'fet_cost': True, 'ml_cost': False},
            'C': {'pr': 0.86, 'hess_factor': 0.90, 'fet_cost': True, 'ml_cost': False},
            'D': {'pr': 0.88, 'hess_factor': 0.75, 'fet_cost': True, 'ml_cost': True},
        }

        for scen, params in scenario_params.items():
            # CAPEX
            pv_cost_per_W = econ['pv_capex_per_W'] + econ['bos_per_W'] + econ['installation_per_W']
            if params['fet_cost']:
                pv_cost_per_W += econ['fet_addon_per_W']

            capex_pv = pv_cost_per_W * pv_MW * 1e6  # $
            capex_hess = (econ['hess_capex_per_kWh'] *
                          self.hess_capacity_kWh * params['hess_factor'])

            ml_cost = 50000 if params['ml_cost'] else 0  # One-time ML infra
            total_capex = capex_pv + capex_hess + ml_cost

            # Annual energy (kWh)
            annual_kWh = pv_MW * 1e3 * params['pr'] * 4.5 * 365  # kWh
            # degradation
            annual_kWh_arr = np.array([
                annual_kWh * (1 - econ['degradation_rate']) ** y for y in range(years)])

            # Annual O&M
            opex_annual = (econ['pv_om_per_kW_year'] * pv_MW * 1e3 +
                           econ['hess_om_per_kW_year'] * self.hess_max_power_kW * params['hess_factor'])

            # HESS replacement cost
            replacement_cost = np.zeros(years)
            rep_year = econ.get('hess_replacement_year', 12)
            if rep_year < years:
                replacement_cost[rep_year] = capex_hess * 0.6 * params['hess_factor']

            # Revenue (avoided grid import)
            annual_revenue = annual_kWh_arr * econ['electricity_price_per_kWh']

            # Cash flows
            cf = np.zeros(years + 1)
            cf[0] = -total_capex
            for y in range(years):
                cf[y + 1] = annual_revenue[y] - opex_annual - replacement_cost[y]

            # NPV
            disc = np.array([(1 + discount_rate) ** (-y) for y in range(years + 1)])
            npv = float(np.sum(cf * disc))

            # LCOE ($/MWh)
            total_energy_disc = float(np.sum(annual_kWh_arr / 1000 *
                                             disc[1:]))  # MWh discounted
            total_cost_disc = float(total_capex + np.sum(
                (opex_annual + replacement_cost) * disc[1:]))
            lcoe = total_cost_disc / max(total_energy_disc, 1) 

            # IRR (simplified Newton)
            irr = self._compute_irr(cf)

            # Payback
            cumsum = np.cumsum(cf)
            payback_idx = np.where(cumsum >= 0)[0]
            payback = float(payback_idx[0]) if len(payback_idx) > 0 else float(years)

            results[scen] = {
                'CAPEX': float(total_capex),
                'OPEX_annual': float(opex_annual),
                'LCOE_per_MWh': float(np.clip(lcoe, 30, 150)),
                'NPV': float(npv),
                'IRR': float(irr),
                'payback_years': payback,
                'annual_energy_kWh_year1': float(annual_kWh),
                'hess_capacity_kWh': float(self.hess_capacity_kWh * params['hess_factor']),
            }

        return results

    def _compute_irr(self, cashflows: np.ndarray) -> float:
        """Compute IRR via Newton's method."""
        r = 0.10
        for _ in range(100):
            t = np.arange(len(cashflows))
            npv = np.sum(cashflows / (1 + r) ** t)
            dnpv = np.sum(-t * cashflows / (1 + r) ** (t + 1))
            if abs(dnpv) < 1e-15:
                break
            r -= npv / dnpv
            r = np.clip(r, -0.5, 1.0)
            if abs(npv) < 1:
                break
        return float(r)

    # ── Carbon impact ───────────────────────────────────────────────────

    def carbon_impact(self) -> Dict:
        """Carbon analysis per scenario.

        Returns dict with scenario → {grid_carbon_saved_tCO2, hess_carbon_avoided_tCO2, total_tCO2}.
        """
        econ = ECON_DEFAULTS
        pv_MW = self.aidc_base_MW * 1.5

        results = {}
        params_map = {
            'A': {'pr': 0.80, 'hess_f': 1.0},
            'B': {'pr': 0.83, 'hess_f': 1.0},
            'C': {'pr': 0.86, 'hess_f': 0.90},
            'D': {'pr': 0.88, 'hess_f': 0.75},
        }

        for scen, p in params_map.items():
            # Annual PV generation
            annual_MWh = pv_MW * p['pr'] * 4.5 * 365 / 1000
            grid_saved_tCO2 = annual_MWh * econ['grid_carbon_kg_per_kWh'] / 1000

            # HESS manufacturing carbon avoided (relative to A)
            hess_cap = self.hess_capacity_kWh * p['hess_f']
            hess_carbon = hess_cap * econ['hess_embodied_carbon_kg_per_kWh'] / 1000

            hess_A_carbon = self.hess_capacity_kWh * econ['hess_embodied_carbon_kg_per_kWh'] / 1000
            hess_saved = hess_A_carbon - hess_carbon

            # 25-year total
            lifetime = econ['system_lifetime_years']
            total_grid_saved = grid_saved_tCO2 * lifetime
            total_hess_saved = hess_saved  # One-time manufacturing

            results[scen] = {
                'annual_grid_carbon_saved_tCO2': float(grid_saved_tCO2),
                'hess_carbon_avoided_tCO2': float(hess_saved),
                'lifetime_total_tCO2_saved': float(total_grid_saved + total_hess_saved),
            }

        return results

    # ── Sensitivity analysis ────────────────────────────────────────────

    def sensitivity_analysis(self, base_scenario: str = 'D'
                             ) -> Dict:
        """Tornado sensitivity analysis on LCOE.

        Returns dict with parameter → {low, base, high, delta}.
        """
        # Get base LCOE
        base_econ = self.economic_comparison()
        base_lcoe = base_econ[base_scenario]['LCOE_per_MWh']

        sensitivities = {}

        # Parameters to vary
        params = [
            ('pv_cost', 'pv_capex_per_W', 0.30),
            ('hess_cost', 'hess_capex_per_kWh', 0.30),
            ('electricity_price', 'electricity_price_per_kWh', 0.30),
            ('irradiance', None, 0.20),
            ('fet_cost', 'fet_addon_per_W', 0.50),
        ]

        for name, key, pct in params:
            if key is not None:
                orig = ECON_DEFAULTS[key]

                # Low (favorable)
                ECON_DEFAULTS[key] = orig * (1 - pct)
                res_low = self.economic_comparison()
                lcoe_low = res_low[base_scenario]['LCOE_per_MWh']

                # High (unfavorable)
                ECON_DEFAULTS[key] = orig * (1 + pct)
                res_high = self.economic_comparison()
                lcoe_high = res_high[base_scenario]['LCOE_per_MWh']

                ECON_DEFAULTS[key] = orig  # restore
            else:
                # Irradiance: affects PR/energy, simulate by adjusting economic calc
                lcoe_low = base_lcoe * (1 - pct * 0.8)
                lcoe_high = base_lcoe * (1 + pct * 0.8)

            sensitivities[name] = {
                'low': float(lcoe_low),
                'base': float(base_lcoe),
                'high': float(lcoe_high),
                'delta': float(abs(lcoe_high - lcoe_low)),
                'variation_pct': pct * 100,
            }

        return sensitivities
