#!/usr/bin/env python3
"""
Multi-Property Material Predictor for ABX₃ Perovskites
========================================================

Extends ml_bandgap.py: predicts Eg, μ_e, μ_h, τ_ion, C_interface, stability
from ABX₃ composition. Enables rapid screening for dynamic PV control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from .ml_bandgap import PerovskiteBandgapPredictor


class MaterialPredictor:
    """Multi-property predictor extending PerovskiteBandgapPredictor."""

    # Physical property reference values by composition
    _REF_PROPERTIES = {
        # (A, B, X) -> {mu_e, mu_h, tau_ion, C_interface, stability}
        ('MA', 'Pb', 'I'): {'mu_e': 25, 'mu_h': 10, 'tau_ion': 10, 'C_if': 44, 'stab': 3.0},
        ('MA', 'Pb', 'Br'): {'mu_e': 20, 'mu_h': 8, 'tau_ion': 15, 'C_if': 35, 'stab': 5.0},
        ('MA', 'Pb', 'Cl'): {'mu_e': 15, 'mu_h': 5, 'tau_ion': 20, 'C_if': 28, 'stab': 6.0},
        ('FA', 'Pb', 'I'): {'mu_e': 30, 'mu_h': 12, 'tau_ion': 8, 'C_if': 50, 'stab': 4.0},
        ('FA', 'Pb', 'Br'): {'mu_e': 22, 'mu_h': 9, 'tau_ion': 12, 'C_if': 38, 'stab': 5.5},
        ('Cs', 'Pb', 'I'): {'mu_e': 18, 'mu_h': 7, 'tau_ion': 25, 'C_if': 30, 'stab': 7.0},
        ('Cs', 'Pb', 'Br'): {'mu_e': 15, 'mu_h': 6, 'tau_ion': 30, 'C_if': 25, 'stab': 8.5},
        ('Cs', 'Pb', 'Cl'): {'mu_e': 12, 'mu_h': 4, 'tau_ion': 35, 'C_if': 20, 'stab': 9.0},
        ('MA', 'Sn', 'I'): {'mu_e': 40, 'mu_h': 20, 'tau_ion': 5, 'C_if': 55, 'stab': 1.5},
        ('FA', 'Sn', 'I'): {'mu_e': 35, 'mu_h': 18, 'tau_ion': 6, 'C_if': 48, 'stab': 2.0},
        ('Cs', 'Sn', 'I'): {'mu_e': 30, 'mu_h': 15, 'tau_ion': 8, 'C_if': 40, 'stab': 3.5},
        ('MA', 'Ge', 'I'): {'mu_e': 10, 'mu_h': 4, 'tau_ion': 40, 'C_if': 22, 'stab': 2.5},
        ('Rb', 'Pb', 'I'): {'mu_e': 12, 'mu_h': 5, 'tau_ion': 30, 'C_if': 28, 'stab': 5.0},
    }

    # Ionic radii (Å) and electronegativity (Pauling)
    IONIC_RADII = {
        'A': {'MA': 1.8, 'FA': 1.9, 'Cs': 1.67, 'Rb': 1.52},
        'B': {'Pb': 1.19, 'Sn': 1.10, 'Ge': 0.87},
        'X': {'I': 2.20, 'Br': 1.96, 'Cl': 1.81},
    }
    ELECTRONEG = {
        'A': {'MA': 2.2, 'FA': 2.1, 'Cs': 0.79, 'Rb': 0.82},
        'B': {'Pb': 2.33, 'Sn': 1.96, 'Ge': 2.01},
        'X': {'I': 2.66, 'Br': 2.96, 'Cl': 3.16},
    }

    def __init__(self):
        self.bandgap_predictor = PerovskiteBandgapPredictor()
        self._models: Dict[str, GaussianProcessRegressor] = {}
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._dataset = None

    def featurize(self, composition: Dict[str, float]) -> np.ndarray:
        """Extract physics-based feature vector from composition.

        Features (14-dim):
        0-9:   composition fractions (A_MA..X_Cl)
        10:    tolerance factor
        11:    octahedral factor
        12:    mixing entropy
        13:    Goldschmidt stability score
        """
        features = np.zeros(14)

        keys = ['A_MA', 'A_FA', 'A_Cs', 'A_Rb', 'B_Pb', 'B_Sn', 'B_Ge', 'X_I', 'X_Br', 'X_Cl']
        for i, k in enumerate(keys):
            features[i] = composition.get(k, 0.0)

        # Effective radii
        r_A = sum(composition.get(f'A_{s}', 0) * self.IONIC_RADII['A'][s] for s in ['MA', 'FA', 'Cs', 'Rb'])
        r_B = sum(composition.get(f'B_{s}', 0) * self.IONIC_RADII['B'][s] for s in ['Pb', 'Sn', 'Ge'])
        r_X = sum(composition.get(f'X_{s}', 0) * self.IONIC_RADII['X'][s] for s in ['I', 'Br', 'Cl'])

        # Tolerance factor
        if r_B > 0 and r_X > 0:
            features[10] = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))

        # Octahedral factor
        if r_X > 0:
            features[11] = r_B / r_X

        # Mixing entropy: -Σ x_i ln(x_i) for each site
        entropy = 0.0
        for site_keys in [['A_MA', 'A_FA', 'A_Cs', 'A_Rb'], ['B_Pb', 'B_Sn', 'B_Ge'], ['X_I', 'X_Br', 'X_Cl']]:
            for k in site_keys:
                x = composition.get(k, 0.0)
                if x > 1e-10:
                    entropy -= x * np.log(x)
        features[12] = entropy

        # Goldschmidt stability: 0.8 < t < 1.0 and μ > 0.414
        t = features[10]
        mu = features[11]
        stability_t = 1.0 - abs(t - 0.9) / 0.2 if 0.7 < t < 1.1 else 0.0
        stability_mu = 1.0 if mu > 0.414 else mu / 0.414
        features[13] = stability_t * stability_mu

        return features

    def _build_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Build multi-property dataset from reference values."""
        rng = np.random.default_rng(42)
        compositions = []
        targets = {p: [] for p in ['mu_e', 'mu_h', 'tau_ion', 'C_interface', 'stability']}

        a_sites = ['MA', 'FA', 'Cs', 'Rb']
        b_sites = ['Pb', 'Sn', 'Ge']
        x_sites = ['I', 'Br', 'Cl']

        # Generate ~100 compositions
        for _ in range(100):
            a_w = rng.dirichlet([1, 1, 1, 0.2])
            b_w = rng.dirichlet([2, 1, 0.3])
            x_w = rng.dirichlet([2, 1.5, 0.5])

            comp = {}
            for i, s in enumerate(a_sites): comp[f'A_{s}'] = a_w[i]
            for i, s in enumerate(b_sites): comp[f'B_{s}'] = b_w[i]
            for i, s in enumerate(x_sites): comp[f'X_{s}'] = x_w[i]

            # Interpolate properties
            props = {p: 0.0 for p in targets}
            total_w = 0.0
            for ia, a in enumerate(a_sites):
                for ib, b in enumerate(b_sites):
                    for ix, x in enumerate(x_sites):
                        w = a_w[ia] * b_w[ib] * x_w[ix]
                        ref = self._REF_PROPERTIES.get((a, b, x))
                        if ref and w > 0:
                            props['mu_e'] += w * ref['mu_e']
                            props['mu_h'] += w * ref['mu_h']
                            props['tau_ion'] += w * ref['tau_ion']
                            props['C_interface'] += w * ref['C_if']
                            props['stability'] += w * ref['stab']
                            total_w += w

            if total_w > 0.01:
                for p in props:
                    props[p] /= total_w
                    props[p] += rng.normal(0, props[p] * 0.05)  # 5% noise

                compositions.append(comp)
                for p in targets:
                    targets[p].append(props[p])

        X = np.array([self.featurize(c) for c in compositions])
        y_dict = {p: np.array(v) for p, v in targets.items()}

        return X, y_dict

    def fit(self):
        """Fit multi-property GP models."""
        if self._is_fitted:
            return

        X, y_dict = self._build_dataset()
        X_scaled = self._scaler.fit_transform(X)

        for prop_name, y in y_dict.items():
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05,
                                          n_restarts_optimizer=5, random_state=42)
            gp.fit(X_scaled, y)
            self._models[prop_name] = gp

        # Fit bandgap predictor
        if not self.bandgap_predictor.is_fitted:
            self.bandgap_predictor.fit()

        self._is_fitted = True

    def predict_multi(self, composition: Dict[str, float]) -> Dict:
        """Predict multiple properties simultaneously.

        Returns:
            Dict with Eg, mu_e, mu_h, tau_ion, C_interface, stability, confidence
        """
        if not self._is_fitted:
            self.fit()

        # Bandgap from existing predictor
        Eg, Eg_std = self.bandgap_predictor.predict(composition)

        # Other properties from GP models
        features = self.featurize(composition).reshape(1, -1)
        features_scaled = self._scaler.transform(features)

        result = {
            'Eg': float(np.clip(Eg, 0.5, 4.0)),
            'confidence': {'Eg': float(max(1 - Eg_std, 0))},
        }

        prop_ranges = {
            'mu_e': (1, 100),      # cm²/Vs
            'mu_h': (0.5, 50),
            'tau_ion': (0.1, 200),  # ms
            'C_interface': (5, 100),  # nF/cm²
            'stability': (0, 10),
        }

        for prop, (lo, hi) in prop_ranges.items():
            pred, std = self._models[prop].predict(features_scaled, return_std=True)
            val = float(np.clip(pred[0], lo, hi))
            result[prop] = val
            result['confidence'][prop] = float(max(1 - std[0] / max(abs(val), 1e-6), 0))

        return result

    def screen_for_dynamic_control(self, target_tau_range: Tuple[float, float] = (1, 100),
                                   stability_threshold: float = 4.0,
                                   Eg_range: Tuple[float, float] = (1.1, 1.8),
                                   n_candidates: int = 200,
                                   seed: int = 777) -> List[Dict]:
        """Screen compositions optimal for dynamic control.

        Criteria:
        - τ_ion in target_tau_range
        - stability > threshold
        - Eg in tandem-optimal range
        """
        if not self._is_fitted:
            self.fit()

        rng = np.random.default_rng(seed)
        results = []

        a_sites = ['MA', 'FA', 'Cs', 'Rb']
        b_sites = ['Pb', 'Sn', 'Ge']
        x_sites = ['I', 'Br', 'Cl']

        for _ in range(n_candidates):
            a_w = rng.dirichlet([1, 1, 1, 0.2])
            b_w = rng.dirichlet([2, 1, 0.3])
            x_w = rng.dirichlet([2, 1.5, 0.5])

            comp = {}
            for i, s in enumerate(a_sites): comp[f'A_{s}'] = float(a_w[i])
            for i, s in enumerate(b_sites): comp[f'B_{s}'] = float(b_w[i])
            for i, s in enumerate(x_sites): comp[f'X_{s}'] = float(x_w[i])

            props = self.predict_multi(comp)

            if (target_tau_range[0] <= props['tau_ion'] <= target_tau_range[1]
                    and props['stability'] > stability_threshold
                    and Eg_range[0] <= props['Eg'] <= Eg_range[1]):
                results.append({
                    'composition': comp,
                    'properties': props,
                })

        # Sort by stability descending
        results.sort(key=lambda x: x['properties']['stability'], reverse=True)
        return results
