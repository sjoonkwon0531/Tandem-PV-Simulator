#!/usr/bin/env python3
"""
Dynamic I-V Curve Engine for Active PV Output Control
=====================================================

Implements dynamic I-V modeling with gate-voltage-controlled operating point,
ion migration dynamics, and interface charge flushing for perovskite PV-FET
integrated devices.

Physics based on:
- PV-FET monolithic integration (Science Advances 2022)
- Ion migration in perovskites (drift-diffusion)
- Interface charge accumulation/flushing (RC dynamics)

References:
- report.md sections 2, 3, 4 for detailed physics
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from scipy.optimize import brentq
from scipy.constants import k as kB, elementary_charge as q

from .iv_curve import SingleDiodeModel

# Optional import for precision ion dynamics
try:
    from .ion_dynamics import IonDynamicsEngine as _IonEngine
    _HAS_ION_ENGINE = True
except ImportError:
    _HAS_ION_ENGINE = False


class DynamicIVEngine:
    """
    Dynamic I-V engine combining perovskite PV with IGZO FET control.
    
    Three timescale physics:
    - Seconds: PV-FET operating point (gate voltage control)
    - Milliseconds: Ion migration in perovskite (drift-diffusion)
    - Microseconds: Interface charge flushing (RC dynamics)
    """

    def __init__(self, perovskite_params: Dict, fet_params: Dict, interface_params: Dict):
        """
        Args:
            perovskite_params: PV cell parameters
                - bandgap: float [eV]
                - temperature: float [K], default 298.15
                - ideality_factor: float, default 1.3
                - series_resistance: float [Ω·cm²], default 1.0
                - shunt_resistance: float [Ω·cm²], default 1e4
                - area: float [cm²], default 1.0
                - thickness: float [nm], default 500
                - ion_density: float [cm⁻³], default 1e18
                - ion_mobility: float [cm²/V·s], default 1e-9
                - ion_diffusivity: float [cm²/s], default 2.6e-11
                - V_bi: float [V], default 1.0
            fet_params: IGZO FET parameters
                - mu_fe: float [cm²/V·s], field-effect mobility, default 20
                - C_ox: float [nF/cm²], gate oxide capacitance, default 300
                - W_L: float, width/length ratio, default 100
                - V_th: float [V], threshold voltage, default 0.5
                - V_G_max: float [V], max gate voltage, default 5.0
            interface_params: Interface charge parameters
                - R_contact: float [Ω], contact resistance, default 10
                - C_interface: float [μF/cm²], interface capacitance, default 44
                - n_traps: float [cm⁻²], trap density, default 1e12
        """
        # Perovskite PV parameters
        self.pv_params = {
            'bandgap': perovskite_params.get('bandgap', 1.55),
            'temperature': perovskite_params.get('temperature', 298.15),
            'ideality_factor': perovskite_params.get('ideality_factor', 1.3),
            'series_resistance': perovskite_params.get('series_resistance', 1.0),
            'shunt_resistance': perovskite_params.get('shunt_resistance', 1e4),
            'area': perovskite_params.get('area', 1.0),
            'thickness': perovskite_params.get('thickness', 500),  # nm
            'ion_density': perovskite_params.get('ion_density', 1e18),  # cm⁻³
            'ion_mobility': perovskite_params.get('ion_mobility', 1e-9),  # cm²/V·s
            'ion_diffusivity': perovskite_params.get('ion_diffusivity', 2.6e-11),  # cm²/s
            'V_bi': perovskite_params.get('V_bi', 1.0),  # V
        }

        # FET parameters
        self.fet_params = {
            'mu_fe': fet_params.get('mu_fe', 20),  # cm²/V·s
            'C_ox': fet_params.get('C_ox', 300),  # nF/cm²
            'W_L': fet_params.get('W_L', 100),  # W/L ratio
            'V_th': fet_params.get('V_th', 0.5),  # V
            'V_G_max': fet_params.get('V_G_max', 5.0),  # V
        }

        # Interface parameters
        self.iface_params = {
            'R_contact': interface_params.get('R_contact', 10),  # Ω
            'C_interface': interface_params.get('C_interface', 44),  # μF/cm²
            'n_traps': interface_params.get('n_traps', 1e12),  # cm⁻²
        }

        # Derived timescales
        L_cm = self.pv_params['thickness'] * 1e-7  # nm -> cm
        mu_i = self.pv_params['ion_mobility']
        V_bi = self.pv_params['V_bi']
        self.tau_ion = L_cm**2 / (mu_i * V_bi) if (mu_i * V_bi) > 0 else 0.01  # s

        R_c = self.iface_params['R_contact']  # Ω
        C_i = self.iface_params['C_interface'] * 1e-6  # μF -> F
        self.tau_rc = R_c * C_i  # s

        # Precision ion dynamics engine (lazy init)
        self._ion_engine: object = None

    def get_ion_engine(self) -> 'IonDynamicsEngine':
        """Return (and lazily create) the precision 1D ion dynamics engine."""
        if self._ion_engine is None and _HAS_ION_ENGINE:
            ion_params = {
                'iodide': {
                    'D_i': self.pv_params.get('ion_diffusivity', 2.6e-11),
                    'mu_i': self.pv_params.get('ion_mobility', 1e-9),
                    'n_i0': self.pv_params.get('ion_density', 1e18),
                    'E_activation': 0.58,
                    'charge': -1,
                }
            }
            self._ion_engine = _IonEngine(
                layer_thickness_nm=self.pv_params['thickness'],
                ion_params=ion_params,
                grid_points=100,
            )
        return self._ion_engine

    def _make_cell(self, G: float = 1.0, T: Optional[float] = None) -> SingleDiodeModel:
        """Create a SingleDiodeModel for given irradiance and temperature."""
        T = T or self.pv_params['temperature']
        cell = SingleDiodeModel(
            bandgap=self.pv_params['bandgap'],
            temperature=T,
            concentration=G,
            ideality_factor=self.pv_params['ideality_factor'],
            series_resistance=self.pv_params['series_resistance'],
            shunt_resistance=self.pv_params['shunt_resistance'],
        )
        return cell

    def static_iv(self, V: np.ndarray, G: float = 1.0, T: float = 298.15) -> np.ndarray:
        """
        Static I-V curve using existing SingleDiodeModel.
        
        Args:
            V: voltage array [V]
            G: irradiance [suns]
            T: temperature [K]
            
        Returns:
            Current density array [mA/cm²]
        """
        cell = self._make_cell(G, T)
        return cell.current_density(V)

    def _channel_conductance(self, V_G: float) -> float:
        """
        IGZO FET channel conductance g_ch [S/cm²].
        
        g_ch = μ_FE * C_ox * (W/L) * max(V_G - V_th, 0)
        
        Returns conductance in mA/V (to match mA/cm² current units).
        """
        mu = self.fet_params['mu_fe']  # cm²/V·s
        C_ox = self.fet_params['C_ox'] * 1e-9  # nF/cm² -> F/cm²
        W_L = self.fet_params['W_L']
        V_th = self.fet_params['V_th']
        V_eff = max(V_G - V_th, 0)

        # g_ch in A/V per unit width, we express as mA/V for our current units
        g_ch = mu * C_ox * W_L * V_eff * 1e3  # ×1e3 to convert A->mA
        return g_ch

    def operating_point(self, V_G: float, G: float = 1.0, T: float = 298.15) -> Dict[str, float]:
        """
        Calculate operating point for given gate voltage.
        
        The FET acts as a variable load. The operating point is found at the
        intersection of the PV I-V curve and the FET load line.
        
        Load line: I = g_ch * V  (FET in linear region as resistive load)
        
        Args:
            V_G: gate voltage [V]
            G: irradiance [suns]
            T: temperature [K]
            
        Returns:
            Dict with V_op, I_op, P_out, g_ch, R_load
        """
        g_ch = self._channel_conductance(V_G)

        if g_ch < 1e-12:
            # FET is off — open circuit condition
            cell = self._make_cell(G, T)
            params = cell.extract_parameters()
            return {
                'V_op': params['Voc'],
                'I_op': 0.0,
                'P_out': 0.0,
                'g_ch': g_ch,
                'R_load': float('inf'),
                'eta': 0.0,
            }

        cell = self._make_cell(G, T)

        # Find intersection: I_pv(V) = g_ch * V
        # i.e., I_pv(V) - g_ch * V = 0
        params = cell.extract_parameters()
        V_oc = params['Voc']

        def equation(V):
            I_pv = cell.current_density(V)
            return float(I_pv) - g_ch * V

        # Check bounds
        f_0 = equation(0.001)
        f_voc = equation(V_oc * 0.999)

        if f_0 * f_voc > 0:
            # No crossing in range — pick MPP or Voc
            if g_ch > params['Jsc'] / 0.001:
                # Very high conductance -> short circuit-like
                V_op = 0.001
            else:
                V_op = V_oc * 0.999
        else:
            V_op = brentq(equation, 0.001, V_oc * 0.999, xtol=1e-8)

        I_op = float(cell.current_density(V_op))
        P_out = V_op * I_op  # mW/cm²
        P_incident = 100.0 * G  # mW/cm² at 1 sun = 100 mW/cm²
        eta = P_out / P_incident if P_incident > 0 else 0.0

        return {
            'V_op': V_op,
            'I_op': I_op,
            'P_out': P_out,
            'g_ch': g_ch,
            'R_load': 1.0 / g_ch if g_ch > 0 else float('inf'),
            'eta': eta,
        }

    def dynamic_response(self, V_G_t: np.ndarray, G_t: np.ndarray,
                         T_t: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """
        Time-dependent response with ion migration and interface charge dynamics.
        
        Args:
            V_G_t: gate voltage time series [V], shape (N,)
            G_t: irradiance time series [suns], shape (N,)
            T_t: temperature time series [K], shape (N,)
            dt: time step [s]
            
        Returns:
            Dict with P_out, V_op, I_op, eta, delta_Voc_ion, Q_interface arrays
        """
        N = len(V_G_t)
        assert len(G_t) == N and len(T_t) == N

        P_out = np.zeros(N)
        V_op = np.zeros(N)
        I_op = np.zeros(N)
        eta = np.zeros(N)
        delta_Voc_ion = np.zeros(N)
        Q_iface = np.zeros(N)

        # Ion migration state (simplified as effective V_OC shift)
        dV_ion = 0.0  # current ion-induced Voc shift [V]
        # Max ion-induced shift based on ion density
        n_i = self.pv_params['ion_density']
        d_i = self.pv_params['thickness'] * 1e-7 * 0.2  # 20% of thickness as ion accumulation region
        eps_r = 25  # perovskite relative permittivity
        eps_0 = 8.854e-12  # F/m
        dV_ion_max = q * n_i * 1e6 * (d_i * 1e-2) / (eps_0 * eps_r)  # V
        dV_ion_max = min(dV_ion_max, 0.2)  # Cap at 200 mV (physical limit)

        # Interface charge state
        Q = 0.0  # Coulombs/cm²

        for i in range(N):
            # 1. Get static operating point
            op = self.operating_point(V_G_t[i], G_t[i], T_t[i])

            # 2. Ion dynamics (ms timescale)
            # Ion drift toward new equilibrium proportional to voltage change
            V_applied = op['V_op']
            # Target ion shift: proportional to deviation from equilibrium
            dV_ion_target = dV_ion_max * (1.0 - V_applied / max(op['V_op'] + 0.1, 0.5))
            # Exponential relaxation toward target
            alpha_ion = 1.0 - np.exp(-dt / self.tau_ion)
            dV_ion += alpha_ion * (dV_ion_target - dV_ion)
            dV_ion = np.clip(dV_ion, -dV_ion_max, dV_ion_max)

            # 3. Interface charge dynamics (μs timescale)
            R_c = self.iface_params['R_contact']
            C_i = self.iface_params['C_interface'] * 1e-6  # F/cm²
            V_interface = Q / C_i if C_i > 0 else 0
            dQdt = (V_applied - V_interface) / R_c - Q / (R_c * C_i) if R_c > 0 else 0
            Q += dQdt * dt
            dV_iface = Q / C_i if C_i > 0 else 0

            # 4. Apply corrections to operating point
            # Ion shift modifies effective Voc -> shifts operating point
            V_correction = dV_ion + dV_iface * 0.01  # interface effect is small
            corrected_P = max(0, op['P_out'] * (1.0 + V_correction / max(op['V_op'], 0.1)))

            P_out[i] = corrected_P
            V_op[i] = op['V_op'] + V_correction * 0.1
            I_op[i] = corrected_P / max(V_op[i], 1e-6)
            P_incident = 100.0 * G_t[i]
            eta[i] = corrected_P / P_incident if P_incident > 0 else 0
            delta_Voc_ion[i] = dV_ion
            Q_iface[i] = Q

        return {
            'P_out': P_out,
            'V_op': V_op,
            'I_op': I_op,
            'eta': eta,
            'delta_Voc_ion': delta_Voc_ion,
            'Q_interface': Q_iface,
        }

    def power_envelope(self, G: float = 1.0, T: float = 298.15,
                       n_points: int = 50) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute power envelope by sweeping V_G.
        
        Args:
            G: irradiance [suns]
            T: temperature [K]
            n_points: number of V_G points
            
        Returns:
            Dict with V_G_array, P_array, P_min, P_max, dynamic_range, dP_dVG
        """
        V_G_max = self.fet_params['V_G_max']
        V_G_arr = np.linspace(0, V_G_max, n_points)
        P_arr = np.zeros(n_points)

        for i, vg in enumerate(V_G_arr):
            op = self.operating_point(vg, G, T)
            P_arr[i] = op['P_out']

        P_min = P_arr[0]  # V_G=0, FET off
        P_max = np.max(P_arr)
        idx_max = np.argmax(P_arr)

        dynamic_range = P_max / P_min if P_min > 1e-10 else float('inf')

        # Control resolution dP/dV_G
        dP_dVG = np.gradient(P_arr, V_G_arr)

        return {
            'V_G': V_G_arr,
            'P': P_arr,
            'P_min': P_min,
            'P_max': P_max,
            'V_G_opt': V_G_arr[idx_max],
            'dynamic_range': dynamic_range,
            'dP_dVG': dP_dVG,
        }

    def efficiency_map(self, V_G_range: np.ndarray, G_range: np.ndarray,
                       T_range: np.ndarray) -> np.ndarray:
        """
        Generate 3D efficiency map (V_G, G, T) → η.
        
        Returns:
            3D array of shape (len(V_G_range), len(G_range), len(T_range))
        """
        nVG = len(V_G_range)
        nG = len(G_range)
        nT = len(T_range)
        eta_map = np.zeros((nVG, nG, nT))

        for i, vg in enumerate(V_G_range):
            for j, g in enumerate(G_range):
                for kk, t in enumerate(T_range):
                    op = self.operating_point(vg, g, t)
                    eta_map[i, j, kk] = op['eta']

        return eta_map

    @property
    def ion_time_constant_ms(self) -> float:
        """Ion migration time constant in milliseconds."""
        return self.tau_ion * 1e3

    @property
    def rc_time_constant_us(self) -> float:
        """Interface RC time constant in microseconds."""
        return self.tau_rc * 1e6
