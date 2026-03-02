#!/usr/bin/env python3
"""
AIDC Load Matching Engine
=========================

Generates GPU datacenter load profiles and analyzes PV-load matching
with and without active PV output control.

Quantifies HESS (Hybrid Energy Storage System) burden reduction
from PV active control.
"""

import numpy as np
from typing import Dict, Optional


class LoadMatchingEngine:
    """
    AIDC load profile generation and PV-load matching analysis.
    """

    def __init__(self, pv_dynamic, aidc_profile: Optional[Dict] = None):
        """
        Args:
            pv_dynamic: DynamicIVEngine instance
            aidc_profile: Optional custom profile parameters
                - gpu_tdp: float [W], per-GPU TDP, default 700
                - training_fraction: float, fraction of time in training, default 0.4
                - pue: float, PUE, default 1.2
        """
        self.pv = pv_dynamic
        self.profile = aidc_profile or {}
        self.gpu_tdp = self.profile.get('gpu_tdp', 700)  # W per GPU (H100-class)
        self.training_fraction = self.profile.get('training_fraction', 0.4)
        self.pue = self.profile.get('pue', 1.2)

    def generate_aidc_load(self, hours: int = 24, gpu_count: int = 10000,
                           dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate GPU datacenter load profile.
        
        Args:
            hours: simulation duration [hours]
            gpu_count: number of GPUs
            dt: time step [seconds]
            
        Returns:
            Dict with time_s, load_kW, phase (training/inference labels)
        """
        N = int(hours * 3600 / dt)
        t = np.arange(N) * dt  # seconds

        P_peak = gpu_count * self.gpu_tdp * 1e-3 * self.pue  # kW

        load = np.zeros(N)
        phase = np.empty(N, dtype='U10')

        # Generate phase schedule: blocks of training and inference
        rng = np.random.RandomState(42)
        block_durations = rng.exponential(2 * 3600, size=50)  # ~2hr avg blocks
        block_starts = np.cumsum(np.concatenate([[0], block_durations]))
        is_training = rng.random(50) < self.training_fraction

        for i in range(N):
            current_time = t[i]
            block_idx = np.searchsorted(block_starts, current_time, side='right') - 1
            block_idx = min(block_idx, len(is_training) - 1)

            if is_training[block_idx]:
                # Training: 80-100% load with small fluctuations
                base = P_peak * (0.85 + 0.15 * np.sin(2 * np.pi * t[i] / 600))
                noise = rng.normal(0, P_peak * 0.02)
                load[i] = base + noise
                phase[i] = 'training'
            else:
                # Inference: 30-80% with larger variability
                base = P_peak * (0.5 + 0.2 * np.sin(2 * np.pi * t[i] / 120))
                # Add burst requests
                burst = P_peak * 0.15 * (rng.random() > 0.95)
                noise = rng.normal(0, P_peak * 0.05)
                load[i] = base + burst + noise
                phase[i] = 'inference'

            # Add batch transition spikes
            for bs in block_starts:
                if abs(t[i] - bs) < 5:  # within 5s of transition
                    load[i] += P_peak * 0.1 * rng.random()

        load = np.clip(load, P_peak * 0.1, P_peak * 1.05)

        return {
            'time_s': t,
            'load_kW': load,
            'phase': phase,
            'P_peak_kW': P_peak,
        }

    def generate_pv_output(self, hours: int = 24, dt: float = 1.0,
                           pv_capacity_kW: float = None,
                           with_control: bool = False,
                           target_load: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate PV output profile over time.
        
        Args:
            hours: duration [hours]
            dt: time step [s]
            pv_capacity_kW: peak PV capacity [kW]
            with_control: if True, use active V_G control to match load
            target_load: load to match [kW] (used when with_control=True)
            
        Returns:
            Dict with time_s, pv_kW, irradiance, V_G (if controlled)
        """
        N = int(hours * 3600 / dt)
        t = np.arange(N) * dt
        if pv_capacity_kW is None:
            pv_capacity_kW = 10000 * self.gpu_tdp * 1e-3 * 0.3  # 30% of peak

        # Solar irradiance: simple sinusoidal day profile
        hour_of_day = (t / 3600) % 24
        G = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))  # sunrise 6, sunset 18
        # Add cloud cover
        rng = np.random.RandomState(123)
        cloud = 1 - 0.3 * np.abs(np.sin(2 * np.pi * t / 1800 + rng.random() * 2 * np.pi))
        G = G * cloud
        G = np.clip(G, 0, 1.0)

        pv_kW = np.zeros(N)
        V_G_used = np.zeros(N)

        if not with_control or target_load is None:
            # Uncontrolled: always at MPP
            for i in range(N):
                if G[i] > 0.01:
                    op = self.pv.operating_point(self.pv.fet_params['V_G_max'], G[i])
                    eta = op['eta']
                else:
                    eta = 0
                pv_kW[i] = pv_capacity_kW * G[i] * eta / 0.22  # normalize to nominal
            V_G_used[:] = self.pv.fet_params['V_G_max']
        else:
            # Active control: adjust V_G to match load
            envelope = self.pv.power_envelope(1.0)
            V_G_range = envelope['V_G']
            P_range = envelope['P']
            P_max_norm = envelope['P_max']

            for i in range(N):
                if G[i] < 0.01:
                    pv_kW[i] = 0
                    V_G_used[i] = 0
                    continue

                # Max PV at this irradiance
                pv_max = pv_capacity_kW * G[i] * (P_max_norm / 100.0) / 0.22
                target = min(target_load[i], pv_max)

                # Find V_G that gives target power
                target_ratio = target / max(pv_max, 1e-6)
                # Simple lookup
                P_norm = P_range / max(P_max_norm, 1e-6)
                idx = np.argmin(np.abs(P_norm - target_ratio))
                V_G_used[i] = V_G_range[idx]
                pv_kW[i] = target

        return {
            'time_s': t,
            'pv_kW': pv_kW,
            'irradiance': G,
            'V_G': V_G_used,
        }

    def match_analysis(self, pv_output_t: np.ndarray,
                       load_t: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        Analyze PV-load matching.
        
        Args:
            pv_output_t: PV output time series [kW]
            load_t: load time series [kW]
            dt: time step [s]
            
        Returns:
            Dict with surplus_kWh, deficit_kWh, hess_capacity_kWh,
            self_consumption_ratio, match_score
        """
        diff = pv_output_t - load_t
        surplus = np.maximum(diff, 0)
        deficit = np.maximum(-diff, 0)

        surplus_kWh = np.sum(surplus) * dt / 3600
        deficit_kWh = np.sum(deficit) * dt / 3600
        total_pv_kWh = np.sum(pv_output_t) * dt / 3600
        total_load_kWh = np.sum(load_t) * dt / 3600

        # Self-consumption ratio: fraction of PV used directly
        direct_use = np.minimum(pv_output_t, load_t)
        self_consumption = np.sum(direct_use) * dt / 3600
        sc_ratio = self_consumption / max(total_pv_kWh, 1e-6)

        # HESS capacity needed to buffer all surplus/deficit
        # Simplified: max cumulative energy swing
        cum_energy = np.cumsum(diff) * dt / 3600  # kWh
        hess_capacity = np.max(cum_energy) - np.min(cum_energy)

        # Match score (0-1, higher = better matching)
        mismatch = np.sum(np.abs(diff)) * dt / 3600
        match_score = 1.0 - mismatch / (total_pv_kWh + total_load_kWh + 1e-6)
        match_score = max(0, match_score)

        return {
            'surplus_kWh': surplus_kWh,
            'deficit_kWh': deficit_kWh,
            'total_pv_kWh': total_pv_kWh,
            'total_load_kWh': total_load_kWh,
            'hess_capacity_kWh': hess_capacity,
            'self_consumption_ratio': sc_ratio,
            'match_score': match_score,
        }

    def hess_reduction(self, with_control: Dict, without_control: Dict) -> Dict[str, float]:
        """
        Quantify HESS burden reduction from active PV control.
        
        Args:
            with_control: match_analysis result with active control
            without_control: match_analysis result without control
            
        Returns:
            Dict with cycling_reduction_pct, capacity_reduction_pct,
            lifetime_extension_factor, cost_saving_usd_per_year
        """
        hess_no_ctrl = without_control['hess_capacity_kWh']
        hess_with_ctrl = with_control['hess_capacity_kWh']

        capacity_reduction = (1 - hess_with_ctrl / max(hess_no_ctrl, 1e-6)) * 100

        # Cycling reduction ~ proportional to surplus reduction
        surplus_reduction = (1 - with_control['surplus_kWh'] /
                           max(without_control['surplus_kWh'], 1e-6)) * 100

        # HESS lifetime extension (fewer cycles -> longer life)
        cycling_factor = max(0.01, 1 - surplus_reduction / 100)
        lifetime_ext = 1.0 / cycling_factor if cycling_factor > 0 else 10.0
        lifetime_ext = min(lifetime_ext, 5.0)  # cap at 5x

        # Cost saving: HESS cost ~$200/kWh, 10-year amortization
        hess_cost_per_kwh = 200  # USD/kWh
        savings = (hess_no_ctrl - hess_with_ctrl) * hess_cost_per_kwh / 10  # per year

        return {
            'cycling_reduction_pct': max(0, surplus_reduction),
            'capacity_reduction_pct': max(0, capacity_reduction),
            'lifetime_extension_factor': lifetime_ext,
            'cost_saving_usd_per_year': max(0, savings),
        }
