#!/usr/bin/env python3
"""
1D Drift-Diffusion Ion Transport Engine for Perovskite PV
==========================================================

Solves the coupled Poisson + drift-diffusion equations for ionic species
(I⁻ vacancies, MA⁺/FA⁺ cations) inside perovskite absorber layers.

Physics:
    ∂n_i/∂t = D_i ∂²n_i/∂x² + μ_i ∂(n_i·E)/∂x - R(n_i)

    Poisson: ∂²φ/∂x² = -q/ε (p - n + N_cat - N_an + N_D - N_A)

Numerical methods:
    - Scharfetter-Gummel discretization (exponential fitting)
    - Thomas algorithm for tridiagonal Poisson solve
    - Implicit Euler time integration for stability
    - Adaptive dt via CFL condition

References:
    - Eames et al., Nature Communications 6, 7497 (2015)
    - Calado et al., Nature Communications 7, 13831 (2016)
    - Richardson et al., Energy Environ. Sci. 9, 1476 (2016)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.constants import (
    elementary_charge as q,
    Boltzmann as kB,
    epsilon_0,
)


def _bernoulli(x: np.ndarray) -> np.ndarray:
    """Bernoulli function B(x) = x / (exp(x) - 1), stable for small x."""
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    big = ~small
    out[small] = 1.0 - 0.5 * x[small]
    with np.errstate(over="ignore", invalid="ignore"):
        ex = np.exp(x[big])
        out[big] = x[big] / (ex - 1.0)
        # Handle overflow (large positive x → B→0, large negative → B→-x)
        overflow = ~np.isfinite(out[big])
        if np.any(overflow):
            xb = x[big]
            out_big = out[big]
            out_big[overflow & (xb[big if isinstance(big, slice) else None] > 0 if False else True)] = 0.0
            # For very negative x: B(x) ≈ -x
            neg_overflow = overflow & (xb < 0)
            out_big[neg_overflow] = -xb[neg_overflow]
            out[big] = out_big
    return out


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                  d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal system.
    a[i] is the sub-diagonal element at row i+1 (length N-1).
    b[i] is the main diagonal element at row i (length N).
    c[i] is the super-diagonal element at row i (length N-1).
    d[i] is the RHS at row i (length N).
    """
    N = len(b)
    cp = np.empty(N - 1, dtype=float)
    dp = np.empty(N, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, N):
        w = a[i - 1] / (b[i] - a[i - 1] * cp[i - 1])
        # Actually standard forward sweep:
        denom = b[i] - a[i - 1] * cp[i - 1]
        if i < N - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    x = np.empty(N, dtype=float)
    x[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class IonDynamicsEngine:
    """1D drift-diffusion simulator for perovskite ion transport.

    Solves coupled Poisson + continuity equations on a uniform 1D grid
    spanning the perovskite absorber layer.
    """

    def __init__(
        self,
        layer_thickness_nm: float = 500.0,
        ion_params: Optional[Dict] = None,
        grid_points: int = 100,
        layer_params: Optional[Dict] = None,
    ):
        """
        Args:
            layer_thickness_nm: perovskite thickness [nm]
            ion_params: dict with keys like 'iodide', 'ma_cation' each containing
                        D_i, mu_i, n_i0, E_activation
            layer_params: dict with epsilon_r, N_D, N_A
            grid_points: number of spatial grid points
        """
        self.L = layer_thickness_nm * 1e-7  # cm
        self.N = grid_points
        self.dx = self.L / (self.N - 1)
        self.x = np.linspace(0, self.L, self.N)  # cm

        # Layer properties
        lp = layer_params or {}
        self.eps_r = lp.get("epsilon_r", 25.0)
        self.N_D = lp.get("N_D", 1e16)  # cm⁻³
        self.N_A = lp.get("N_A", 1e16)  # cm⁻³

        # Ion species
        default_ions = {
            "iodide": {
                "D_i": 1e-12,
                "mu_i": 4e-11,
                "n_i0": 1.6e19,
                "E_activation": 0.58,
                "charge": -1,  # anion
            }
        }
        self.ion_species = ion_params if ion_params else default_ions
        # Ensure charge field
        for name, sp in self.ion_species.items():
            if "charge" not in sp:
                sp["charge"] = -1 if "iodide" in name or "anion" in name else +1

        # Thermal voltage at 300K
        self.T0 = 300.0
        self.VT = kB * self.T0 / q  # ~0.02585 V

        # Derived
        self.eps = epsilon_0 * self.eps_r * 1e-2  # F/cm (ε₀ in F/m → F/cm)
        # Actually: ε₀ = 8.854e-14 F/cm,  eps = eps_r * eps_0_cgs
        self.eps = self.eps_r * 8.854e-14  # F/cm

        # Initialize ion profiles (uniform)
        self.ion_profiles: Dict[str, np.ndarray] = {}
        for name, sp in self.ion_species.items():
            self.ion_profiles[name] = np.full(self.N, sp["n_i0"])

    def _update_VT(self, T: float):
        """Update thermal voltage for temperature T."""
        self.VT = kB * T / q

    def _effective_D(self, sp: Dict, T: float) -> float:
        """Temperature-dependent diffusion coefficient using Arrhenius."""
        D0 = sp["D_i"]
        Ea = sp.get("E_activation", 0.0)
        if Ea > 0 and T != self.T0:
            # D(T) = D(T0) * exp(-Ea/kB * (1/T - 1/T0))
            return D0 * np.exp(-Ea * q / kB * (1.0 / T - 1.0 / self.T0))
        return D0

    def _effective_mu(self, sp: Dict, T: float) -> float:
        """Temperature-dependent mobility via Einstein relation consistency."""
        D = self._effective_D(sp, T)
        VT = kB * T / q
        return D / VT  # Einstein: D = mu * kT/q

    def solve_poisson(
        self,
        n_cation: np.ndarray,
        n_anion: np.ndarray,
        n_electron: Optional[np.ndarray] = None,
        n_hole: Optional[np.ndarray] = None,
        V_left: float = 0.0,
        V_right: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Poisson equation for electrostatic potential and field.

        ∂²φ/∂x² = -q/ε (p - n + n_cat - n_an + N_D - N_A)

        Dirichlet BCs: φ(0) = V_left, φ(L) = V_right

        Args:
            n_cation: cation concentration profile [cm⁻³]
            n_anion: anion concentration profile [cm⁻³]
            n_electron: electron concentration (optional)
            n_hole: hole concentration (optional)
            V_left, V_right: boundary potentials [V]

        Returns:
            (phi, E_field) — potential [V] and electric field [V/cm]
        """
        N = self.N
        dx = self.dx

        if n_electron is None:
            n_electron = np.full(N, self.N_D)  # approximate equilibrium
        if n_hole is None:
            n_hole = np.full(N, self.N_A)

        # Net charge density
        rho = q * (n_hole - n_electron + n_cation - n_anion + self.N_D - self.N_A)

        # Build tridiagonal system for interior points
        # φ_{i-1} - 2φ_i + φ_{i+1} = -(rho_i / eps) * dx²
        N_int = N - 2  # interior points
        if N_int < 1:
            phi = np.linspace(V_left, V_right, N)
            E = -np.gradient(phi, dx)
            return phi, E

        rhs_coeff = -dx * dx / self.eps

        a = np.ones(N_int - 1)       # lower
        b = -2.0 * np.ones(N_int)    # main
        c = np.ones(N_int - 1)       # upper
        d = rho[1:-1] * rhs_coeff
        d[0] -= V_left
        d[-1] -= V_right

        phi_int = _thomas_solve(a, b, c, d)
        phi = np.empty(N)
        phi[0] = V_left
        phi[-1] = V_right
        phi[1:-1] = phi_int

        # Electric field E = -dφ/dx
        E = np.zeros(N)
        E[1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * dx)
        E[0] = -(phi[1] - phi[0]) / dx
        E[-1] = -(phi[-1] - phi[-2]) / dx

        return phi, E

    def drift_diffusion_step(
        self,
        n_i: np.ndarray,
        E: np.ndarray,
        D_i: float,
        mu_i: float,
        dt: float,
        charge_sign: int = -1,
    ) -> np.ndarray:
        """Single implicit Euler drift-diffusion step with Scharfetter-Gummel.

        The continuity equation for species with signed charge z:
            ∂n/∂t = -∂J/∂x
            J = z * (mu_i * n * E - D_i * dn/dx)   [particle flux, sign convention]

        For anions (z=-1): J = -mu_i * n * E + D_i * dn/dx  → drift opposite to E
        For cations (z=+1): J = mu_i * n * E - D_i * dn/dx  → drift along E

        Scharfetter-Gummel flux at interface i+1/2:
            J_{i+1/2} = (D/dx) * [B(-u)*n_{i+1} - B(u)*n_i]
        where u = z * mu * E * dx / D  (dimensionless field)

        Returns updated n_i.
        """
        N = len(n_i)
        dx = self.dx

        # Edge electric field (at i+1/2 interfaces)
        E_half = 0.5 * (E[:-1] + E[1:])  # N-1 values

        # Dimensionless field parameter
        u = charge_sign * mu_i * E_half * dx / D_i  # N-1

        Bp = _bernoulli(u)     # B(u)
        Bm = _bernoulli(-u)    # B(-u)

        # Build implicit Euler tridiagonal system:
        # n_i^{new} - dt * (flux divergence) = n_i^{old}
        # flux divergence at node i = (J_{i+1/2} - J_{i-1/2}) / dx
        # J_{i+1/2} = (D/dx) * [Bm_{i+1/2} * n_{i+1} - Bp_{i+1/2} * n_i]

        coeff = D_i * dt / (dx * dx)

        # Main diagonal
        main = np.ones(N)
        lower = np.zeros(N - 1)
        upper = np.zeros(N - 1)

        # Interior points (1 to N-2)
        for i in range(1, N - 1):
            # From J_{i+1/2}: contributes -Bp_{i} to n_i, +Bm_{i} to n_{i+1}
            # From J_{i-1/2}: contributes +Bp_{i-1} to n_{i-1}, -Bm_{i-1} to n_i
            # Divergence = (J_{i+1/2} - J_{i-1/2}) / dx
            # For implicit: n_new + coeff*(...) = n_old

            main[i] += coeff * (Bp[i] + Bm[i - 1])
            if i < N - 1:
                upper[i] = -coeff * Bm[i]
            if i > 0:
                lower[i - 1] = -coeff * Bp[i - 1]

        # Boundary conditions: zero flux (reflecting)
        # At x=0: J_{-1/2} = 0 → only J_{1/2} contributes
        main[0] += coeff * Bp[0]
        upper[0] = -coeff * Bm[0]

        # At x=L: J_{N-1/2} = 0 → only J_{N-3/2} contributes
        main[-1] += coeff * Bm[-1]
        lower[-1] = -coeff * Bp[-1]

        # Solve
        rhs = n_i.copy()
        n_new = _thomas_solve(lower, main, upper, rhs)

        # Enforce positivity
        n_new = np.maximum(n_new, 0.0)

        return n_new

    def simulate(
        self,
        V_applied_t: np.ndarray,
        G_t: np.ndarray,
        T_t: np.ndarray,
        dt: float,
        total_time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Full time-domain simulation.

        Args:
            V_applied_t: applied voltage time series [V]
            G_t: generation rate / irradiance time series [suns]
            T_t: temperature time series [K]
            dt: time step [s]
            total_time: ignored (inferred from array length)

        Returns:
            Dict with P_out, V_OC, FF, E_field (last snapshot), n_ion (last snapshot),
            V_OC_t, FF_t time series
        """
        N_t = len(V_applied_t)
        assert len(G_t) == N_t and len(T_t) == N_t

        V_OC_t = np.zeros(N_t)
        FF_t = np.zeros(N_t)
        P_out_t = np.zeros(N_t)
        dV_ion_t = np.zeros(N_t)

        # Initialize ion profiles
        profiles = {}
        for name, sp in self.ion_species.items():
            profiles[name] = self.ion_profiles[name].copy()

        E_last = np.zeros(self.N)

        for t_idx in range(N_t):
            T = T_t[t_idx]
            G = G_t[t_idx]
            V_app = V_applied_t[t_idx]
            self._update_VT(T)

            # Sum cation/anion profiles
            n_cat = np.zeros(self.N)
            n_an = np.zeros(self.N)
            for name, sp in self.ion_species.items():
                if sp["charge"] > 0:
                    n_cat += profiles[name]
                else:
                    n_an += profiles[name]

            # Solve Poisson
            _, E = self.solve_poisson(n_cat, n_an, V_left=0.0, V_right=V_app)
            E_last = E.copy()

            # Drift-diffusion step for each species
            for name, sp in self.ion_species.items():
                D_eff = self._effective_D(sp, T)
                mu_eff = self._effective_mu(sp, T)
                profiles[name] = self.drift_diffusion_step(
                    profiles[name], E, D_eff, mu_eff, dt, sp["charge"]
                )

            # Compute ion-induced voltage shift
            # ΔV_ion ≈ q * <Δn_ion * x> / (ε * L) — dipole moment of redistribution
            for name, sp in self.ion_species.items():
                dn = profiles[name] - sp["n_i0"]
                dipole = np.trapezoid(dn * self.x, self.x)
                dV = q * dipole / (self.eps * self.L) * sp["charge"]
                dV_ion_t[t_idx] += dV

            # Estimate V_OC and FF from ion shift
            # Base V_OC from irradiance (simplified)
            V_OC_base = 1.0 + self.VT * np.log(max(G, 1e-6))
            V_OC_eff = V_OC_base + np.clip(dV_ion_t[t_idx], -0.2, 0.2)
            V_OC_t[t_idx] = V_OC_eff

            # FF estimation (empirical Green formula)
            v_oc_norm = V_OC_eff / self.VT
            if v_oc_norm > 1:
                FF_t[t_idx] = (v_oc_norm - np.log(v_oc_norm + 0.72)) / (v_oc_norm + 1)
            else:
                FF_t[t_idx] = 0.25

            # J_SC scales with G
            J_SC = 20.0 * G  # mA/cm² (approximate for perovskite)
            P_out_t[t_idx] = J_SC * V_OC_eff * FF_t[t_idx]  # mW/cm²

        # Store final profiles
        self.ion_profiles = profiles

        return {
            "P_out": P_out_t,
            "V_OC": V_OC_t,
            "FF": FF_t,
            "E_field": E_last,
            "n_ion": {name: p.copy() for name, p in profiles.items()},
            "dV_ion": dV_ion_t,
        }

    def steady_state(
        self, V_applied: float, G: float, T: float, max_iter: int = 500, tol: float = 1e-6
    ) -> Dict:
        """Find steady-state ion distribution by iterating until convergence.

        Uses pseudo-time-stepping with large dt until |Δn/n| < tol.
        """
        self._update_VT(T)

        profiles = {}
        for name, sp in self.ion_species.items():
            profiles[name] = np.full(self.N, sp["n_i0"])

        dt_pseudo = 1e-3  # start with 1 ms steps, increase
        converged = False

        for iteration in range(max_iter):
            n_cat = np.zeros(self.N)
            n_an = np.zeros(self.N)
            for name, sp in self.ion_species.items():
                if sp["charge"] > 0:
                    n_cat += profiles[name]
                else:
                    n_an += profiles[name]

            _, E = self.solve_poisson(n_cat, n_an, V_left=0.0, V_right=V_applied)

            max_rel_change = 0.0
            for name, sp in self.ion_species.items():
                D_eff = self._effective_D(sp, T)
                mu_eff = self._effective_mu(sp, T)
                n_old = profiles[name].copy()
                profiles[name] = self.drift_diffusion_step(
                    profiles[name], E, D_eff, mu_eff, dt_pseudo, sp["charge"]
                )
                denom = np.maximum(np.abs(n_old), 1e10)
                rel_change = np.max(np.abs(profiles[name] - n_old) / denom)
                max_rel_change = max(max_rel_change, rel_change)

            if max_rel_change < tol:
                converged = True
                break

            # Adaptive dt increase
            if max_rel_change < 0.01:
                dt_pseudo = min(dt_pseudo * 2.0, 1.0)

        self.ion_profiles = profiles

        # Compute steady-state outputs
        dV_ion = 0.0
        for name, sp in self.ion_species.items():
            dn = profiles[name] - sp["n_i0"]
            dipole = np.trapezoid(dn * self.x, self.x)
            dV_ion += q * dipole / (self.eps * self.L) * sp["charge"]

        V_OC_base = 1.0 + self.VT * np.log(max(G, 1e-6))
        V_OC = V_OC_base + np.clip(dV_ion, -0.2, 0.2)

        v_oc_n = V_OC / self.VT
        FF = (v_oc_n - np.log(v_oc_n + 0.72)) / (v_oc_n + 1) if v_oc_n > 1 else 0.25
        J_SC = 20.0 * G
        P_out = J_SC * V_OC * FF

        return {
            "converged": converged,
            "iterations": iteration + 1 if not converged else iteration + 1,
            "V_OC": V_OC,
            "FF": FF,
            "P_out": P_out,
            "dV_ion": dV_ion,
            "E_field": E,
            "n_ion": {name: p.copy() for name, p in profiles.items()},
        }

    def hysteresis_iv(
        self,
        V_sweep_rate: float,
        G: float = 1.0,
        T: float = 300.0,
        V_max: float = 1.2,
        n_points: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Simulate I-V hysteresis from forward and reverse sweeps.

        Args:
            V_sweep_rate: voltage sweep rate [V/s]
            G: irradiance [suns]
            T: temperature [K]
            V_max: maximum voltage [V]
            n_points: points per sweep direction

        Returns:
            Dict with V_forward, I_forward, V_reverse, I_reverse,
            P_forward, P_reverse, HI (hysteresis index)
        """
        self._update_VT(T)

        # Time for sweep
        sweep_time = V_max / V_sweep_rate  # s
        dt = sweep_time / n_points

        V_fwd = np.linspace(0, V_max, n_points)
        V_rev = np.linspace(V_max, 0, n_points)

        # Reset ion profiles to uniform
        for name, sp in self.ion_species.items():
            self.ion_profiles[name] = np.full(self.N, sp["n_i0"])

        # Forward sweep
        I_fwd = np.zeros(n_points)
        profiles_fwd = {name: self.ion_profiles[name].copy() for name in self.ion_species}

        for i, V in enumerate(V_fwd):
            n_cat = np.zeros(self.N)
            n_an = np.zeros(self.N)
            for name, sp in self.ion_species.items():
                if sp["charge"] > 0:
                    n_cat += profiles_fwd[name]
                else:
                    n_an += profiles_fwd[name]

            _, E = self.solve_poisson(n_cat, n_an, V_left=0.0, V_right=V)

            for name, sp in self.ion_species.items():
                D_eff = self._effective_D(sp, T)
                mu_eff = self._effective_mu(sp, T)
                profiles_fwd[name] = self.drift_diffusion_step(
                    profiles_fwd[name], E, D_eff, mu_eff, dt, sp["charge"]
                )

            # Ion-induced V_OC shift
            dV = 0.0
            for name, sp in self.ion_species.items():
                dn = profiles_fwd[name] - sp["n_i0"]
                dipole = np.trapezoid(dn * self.x, self.x)
                dV += q * dipole / (self.eps * self.L) * sp["charge"]

            V_OC_eff = 1.0 + self.VT * np.log(max(G, 1e-6)) + np.clip(dV, -0.2, 0.2)
            J_SC = 20.0 * G
            # Simple diode: I = J_SC - J_SC * exp((V - V_OC_eff) / (n*VT))
            n_ideal = 1.3
            if V < V_OC_eff:
                I_fwd[i] = J_SC * (1.0 - np.exp((V - V_OC_eff) / (n_ideal * self.VT)))
            else:
                I_fwd[i] = 0.0

        # Reverse sweep (start from forward-sweep end state)
        I_rev = np.zeros(n_points)
        profiles_rev = {name: profiles_fwd[name].copy() for name in self.ion_species}

        for i, V in enumerate(V_rev):
            n_cat = np.zeros(self.N)
            n_an = np.zeros(self.N)
            for name, sp in self.ion_species.items():
                if sp["charge"] > 0:
                    n_cat += profiles_rev[name]
                else:
                    n_an += profiles_rev[name]

            _, E = self.solve_poisson(n_cat, n_an, V_left=0.0, V_right=V)

            for name, sp in self.ion_species.items():
                D_eff = self._effective_D(sp, T)
                mu_eff = self._effective_mu(sp, T)
                profiles_rev[name] = self.drift_diffusion_step(
                    profiles_rev[name], E, D_eff, mu_eff, dt, sp["charge"]
                )

            dV = 0.0
            for name, sp in self.ion_species.items():
                dn = profiles_rev[name] - sp["n_i0"]
                dipole = np.trapezoid(dn * self.x, self.x)
                dV += q * dipole / (self.eps * self.L) * sp["charge"]

            V_OC_eff = 1.0 + self.VT * np.log(max(G, 1e-6)) + np.clip(dV, -0.2, 0.2)
            J_SC = 20.0 * G
            n_ideal = 1.3
            if V < V_OC_eff:
                I_rev[i] = J_SC * (1.0 - np.exp((V - V_OC_eff) / (n_ideal * self.VT)))
            else:
                I_rev[i] = 0.0

        P_fwd = V_fwd * I_fwd
        P_rev = V_rev * I_rev

        PCE_fwd = np.max(P_fwd) / (100.0 * G) if G > 0 else 0
        PCE_rev = np.max(P_rev) / (100.0 * G) if G > 0 else 0

        # Hysteresis index
        HI = abs(PCE_rev - PCE_fwd) / max(PCE_rev, 1e-10) if PCE_rev > 0 else 0.0

        return {
            "V_forward": V_fwd,
            "I_forward": I_fwd,
            "V_reverse": V_rev,
            "I_reverse": I_rev,
            "P_forward": P_fwd,
            "P_reverse": P_rev,
            "PCE_forward": PCE_fwd * 100,
            "PCE_reverse": PCE_rev * 100,
            "HI": HI,
        }

    def response_time(
        self, V_step: float, G: float = 1.0, T: float = 300.0, duration_s: float = 1.0,
        n_steps: int = 1000,
    ) -> Dict[str, float]:
        """Extract characteristic response times from a voltage step.

        Applies a step in V_applied from 0 to V_step and monitors P_out(t).

        Returns:
            tau_ion: time to reach 63% of final change [s]
            tau_90: time to reach 90% of final change [s]
            P_initial, P_final
        """
        dt = duration_s / n_steps

        # Reset profiles
        for name, sp in self.ion_species.items():
            self.ion_profiles[name] = np.full(self.N, sp["n_i0"])

        V_t = np.full(n_steps, V_step)
        G_t = np.full(n_steps, G)
        T_t = np.full(n_steps, T)

        result = self.simulate(V_t, G_t, T_t, dt)
        P = result["P_out"]

        P_init = P[0]
        P_final = P[-1]
        delta_P = P_final - P_init

        if abs(delta_P) < 1e-10:
            return {
                "tau_ion": 0.0,
                "tau_90": 0.0,
                "P_initial": P_init,
                "P_final": P_final,
            }

        # Find 63% (1 - 1/e) time
        target_63 = P_init + 0.632 * delta_P
        target_90 = P_init + 0.9 * delta_P

        time_arr = np.arange(n_steps) * dt

        tau_63 = duration_s  # fallback
        tau_90 = duration_s

        if delta_P > 0:
            idx_63 = np.where(P >= target_63)[0]
            idx_90 = np.where(P >= target_90)[0]
        else:
            idx_63 = np.where(P <= target_63)[0]
            idx_90 = np.where(P <= target_90)[0]

        if len(idx_63) > 0:
            tau_63 = time_arr[idx_63[0]]
        if len(idx_90) > 0:
            tau_90 = time_arr[idx_90[0]]

        return {
            "tau_ion": tau_63,
            "tau_90": tau_90,
            "P_initial": P_init,
            "P_final": P_final,
        }

    def get_ion_timescale(self, T: float = 300.0) -> float:
        """Estimate ion migration timescale τ = L²/(μ·V_bi) [s]."""
        # Use the first (or fastest) species
        for name, sp in self.ion_species.items():
            mu = self._effective_mu(sp, T)
            V_bi = 1.0  # approximate built-in voltage
            tau = self.L ** 2 / (mu * V_bi) if mu * V_bi > 0 else 1.0
            return tau
        return 1.0
