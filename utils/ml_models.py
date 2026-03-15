"""
Lightweight ML Surrogate Models
================================

XGBoost-based bandgap prediction trained on DFT database.
Lightweight - no PyTorch, minimal memory footprint.

Features:
- Composition-based featurization
- Bandgap regression with uncertainty quantification
- Active learning acquisition functions

Author: OpenClaw Agent
Date: 2026-03-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re
import joblib
from pathlib import Path

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class CompositionFeaturizer:
    """
    Convert chemical formula to numerical features.
    Uses elemental properties from periodic table.
    """
    
    # Elemental properties (subset relevant for perovskites)
    ELEMENT_PROPS = {
        # A-site cations
        'Cs': {'radius': 1.67, 'electronegativity': 0.79, 'valence': 1},
        'Rb': {'radius': 1.52, 'electronegativity': 0.82, 'valence': 1},
        'K': {'radius': 1.38, 'electronegativity': 0.82, 'valence': 1},
        'MA': {'radius': 2.17, 'electronegativity': 0.0, 'valence': 1},  # Organic
        'FA': {'radius': 2.53, 'electronegativity': 0.0, 'valence': 1},  # Organic
        
        # B-site metals
        'Pb': {'radius': 1.19, 'electronegativity': 2.33, 'valence': 2},
        'Sn': {'radius': 1.10, 'electronegativity': 1.96, 'valence': 2},
        'Ge': {'radius': 0.73, 'electronegativity': 2.01, 'valence': 2},
        'Sr': {'radius': 1.18, 'electronegativity': 0.95, 'valence': 2},
        'Ca': {'radius': 1.00, 'electronegativity': 1.00, 'valence': 2},
        
        # X-site halides
        'I': {'radius': 2.20, 'electronegativity': 2.66, 'valence': -1},
        'Br': {'radius': 1.96, 'electronegativity': 2.96, 'valence': -1},
        'Cl': {'radius': 1.81, 'electronegativity': 3.16, 'valence': -1},
        'F': {'radius': 1.33, 'electronegativity': 3.98, 'valence': -1},
    }
    
    def __init__(self):
        self.feature_names = []
    
    def featurize(self, formula: str) -> np.ndarray:
        """
        Convert formula to feature vector.
        
        Features (18-dimensional):
        - A-site: avg radius, avg electronegativity, n_species
        - B-site: avg radius, avg electronegativity, n_species
        - X-site: avg radius, avg electronegativity, n_species, variance
        - Tolerance factor
        - Octahedral factor
        - A-site organic fraction
        - X-site mixing entropy
        - A-site mixing entropy
        - B-site mixing entropy
        - A-site radius variance
        - X-site radius variance
        - X-site electronegativity variance
        """
        comp = self._parse_composition(formula)
        
        if not comp:
            return np.zeros(18)
        
        features = []
        
        # A-site features
        a_species = comp.get('A', {})
        a_radius = self._weighted_avg(a_species, 'radius')
        a_en = self._weighted_avg(a_species, 'electronegativity')
        a_n_species = len(a_species)
        a_organic_frac = sum(frac for elem, frac in a_species.items() if elem in ['MA', 'FA'])
        a_entropy = self._mixing_entropy(a_species)
        a_radius_var = self._weighted_variance(a_species, 'radius')
        
        features.extend([a_radius, a_en, a_n_species, a_organic_frac, a_entropy, a_radius_var])
        
        # B-site features
        b_species = comp.get('B', {})
        b_radius = self._weighted_avg(b_species, 'radius')
        b_en = self._weighted_avg(b_species, 'electronegativity')
        b_n_species = len(b_species)
        b_entropy = self._mixing_entropy(b_species)
        
        features.extend([b_radius, b_en, b_n_species, b_entropy])
        
        # X-site features
        x_species = comp.get('X', {})
        x_radius = self._weighted_avg(x_species, 'radius')
        x_en = self._weighted_avg(x_species, 'electronegativity')
        x_n_species = len(x_species)
        x_entropy = self._mixing_entropy(x_species)
        x_radius_var = self._weighted_variance(x_species, 'radius')
        x_en_var = self._weighted_variance(x_species, 'electronegativity')
        
        features.extend([x_radius, x_en, x_n_species, x_entropy, x_radius_var, x_en_var])
        
        # Structural descriptors
        tolerance = self._tolerance_factor(a_radius, b_radius, x_radius)
        octahedral = self._octahedral_factor(b_radius, x_radius)
        
        features.extend([tolerance, octahedral])
        
        return np.array(features)
    
    def _parse_composition(self, formula: str) -> Dict[str, Dict[str, float]]:
        """
        Parse ABX3 formula into composition dictionary.
        Returns: {'A': {'Cs': 0.13, 'FA': 0.87}, 'B': {'Pb': 1.0}, 'X': {'I': 0.62, 'Br': 0.38}}
        """
        # Normalize subscripts
        formula = formula.replace('₀', '0').replace('₁', '1').replace('₂', '2')
        formula = formula.replace('₃', '3').replace('₄', '4').replace('₅', '5')
        formula = formula.replace('₆', '6').replace('₇', '7').replace('₈', '8')
        formula = formula.replace('₉', '9')
        
        comp = {'A': {}, 'B': {}, 'X': {}}
        
        # A-site
        for elem in ['MA', 'FA', 'Cs', 'Rb', 'K']:
            pattern = f'{elem}([0-9.]*)'
            match = re.search(pattern, formula)
            if match:
                frac = float(match.group(1)) if match.group(1) else 1.0
                comp['A'][elem] = frac
        
        # Normalize A-site
        total_a = sum(comp['A'].values())
        if total_a > 0:
            comp['A'] = {k: v/total_a for k, v in comp['A'].items()}
        
        # B-site
        for elem in ['Pb', 'Sn', 'Ge', 'Sr', 'Ca']:
            if elem in formula:
                pattern = f'{elem}([0-9.]*)'
                match = re.search(pattern, formula)
                if match:
                    frac = float(match.group(1)) if match.group(1) else 1.0
                    comp['B'][elem] = frac
        
        # Normalize B-site
        total_b = sum(comp['B'].values())
        if total_b > 0:
            comp['B'] = {k: v/total_b for k, v in comp['B'].items()}
        
        # X-site (look for parenthesized halides or simple I3/Br3/Cl3)
        halide_pattern = r'\(([IBrClF0-9.]+)\)|([IBrClF]+)3'
        match = re.search(halide_pattern, formula)
        
        if match:
            halides_str = match.group(1) if match.group(1) else match.group(2)
            
            # Parse halide composition
            for elem in ['I', 'Br', 'Cl', 'F']:
                pattern = f'{elem}([0-9.]*)'
                h_match = re.search(pattern, halides_str)
                if h_match:
                    frac = float(h_match.group(1)) if h_match.group(1) else 1.0
                    comp['X'][elem] = frac
        
        # Normalize X-site
        total_x = sum(comp['X'].values())
        if total_x > 0:
            comp['X'] = {k: v/total_x for k, v in comp['X'].items()}
        
        return comp
    
    def _weighted_avg(self, species: Dict[str, float], prop: str) -> float:
        """Calculate weighted average of elemental property"""
        if not species:
            return 0.0
        
        total = 0.0
        for elem, frac in species.items():
            if elem in self.ELEMENT_PROPS:
                total += frac * self.ELEMENT_PROPS[elem][prop]
        
        return total
    
    def _weighted_variance(self, species: Dict[str, float], prop: str) -> float:
        """Calculate weighted variance of elemental property"""
        if len(species) <= 1:
            return 0.0
        
        avg = self._weighted_avg(species, prop)
        
        variance = 0.0
        for elem, frac in species.items():
            if elem in self.ELEMENT_PROPS:
                variance += frac * (self.ELEMENT_PROPS[elem][prop] - avg) ** 2
        
        return variance
    
    def _mixing_entropy(self, species: Dict[str, float]) -> float:
        """Calculate configurational entropy: -Σ x_i ln(x_i)"""
        if len(species) <= 1:
            return 0.0
        
        entropy = 0.0
        for frac in species.values():
            if frac > 0:
                entropy -= frac * np.log(frac)
        
        return entropy
    
    def _tolerance_factor(self, r_a: float, r_b: float, r_x: float) -> float:
        """Goldschmidt tolerance factor: (r_A + r_X) / [sqrt(2) * (r_B + r_X)]"""
        if r_b == 0 or r_x == 0:
            return 0.0
        
        return (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))
    
    def _octahedral_factor(self, r_b: float, r_x: float) -> float:
        """Octahedral factor: r_B / r_X"""
        if r_x == 0:
            return 0.0
        
        return r_b / r_x
    
    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return [
            'A_radius', 'A_electronegativity', 'A_n_species', 'A_organic_frac', 
            'A_mixing_entropy', 'A_radius_var',
            'B_radius', 'B_electronegativity', 'B_n_species', 'B_mixing_entropy',
            'X_radius', 'X_electronegativity', 'X_n_species', 'X_mixing_entropy',
            'X_radius_var', 'X_en_var',
            'tolerance_factor', 'octahedral_factor'
        ]


class BandgapPredictor:
    """
    XGBoost-based bandgap prediction model with uncertainty quantification.
    """
    
    def __init__(self, use_xgboost: bool = True):
        self.featurizer = CompositionFeaturizer()
        self.scaler = StandardScaler()
        
        if use_xgboost and XGBOOST_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Fallback to RandomForest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.trained = False
        self.train_score = None
    
    def train(self, df: pd.DataFrame, formula_col: str = 'formula', 
              target_col: str = 'bandgap') -> Dict[str, float]:
        """
        Train model on database.
        
        Returns:
            Dict with training metrics
        """
        # Filter valid data
        df_clean = df[[formula_col, target_col]].dropna()
        
        # Featurize
        X = np.array([self.featurizer.featurize(f) for f in df_clean[formula_col]])
        y = df_clean[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, 
                                   scoring='neg_mean_absolute_error')
        
        self.trained = True
        self.train_score = {
            'n_samples': len(y),
            'cv_mae': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'train_r2': self.model.score(X_scaled, y)
        }
        
        return self.train_score
    
    def predict(self, formulas: List[str], return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict bandgaps for list of formulas.
        
        Returns:
            predictions, uncertainties (if return_uncertainty=True)
        """
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        # Featurize
        X = np.array([self.featurizer.featurize(f) for f in formulas])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        if return_uncertainty:
            # Uncertainty from ensemble variance (for RandomForest/XGBoost with trees)
            if hasattr(self.model, 'estimators_'):
                # RandomForest
                tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                uncertainties = tree_preds.std(axis=0)
            else:
                # XGBoost - use CV MAE as constant uncertainty
                uncertainties = np.ones(len(predictions)) * self.train_score['cv_mae']
            
            return predictions, uncertainties
        
        return predictions, None
    
    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'trained': self.trained,
            'train_score': self.train_score
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.trained = model_data['trained']
        self.train_score = model_data['train_score']
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances"""
        if not self.trained:
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        feature_names = self.featurizer.get_feature_names()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df


def train_default_model(df: pd.DataFrame, save_path: Optional[str] = None) -> BandgapPredictor:
    """
    Train default bandgap predictor on database.
    Convenience function for quick setup.
    """
    predictor = BandgapPredictor(use_xgboost=XGBOOST_AVAILABLE)
    
    metrics = predictor.train(df)
    
    print(f"Model trained on {metrics['n_samples']} samples")
    print(f"Cross-validation MAE: {metrics['cv_mae']:.3f} ± {metrics['cv_mae_std']:.3f} eV")
    print(f"Training R²: {metrics['train_r2']:.3f}")
    
    if save_path:
        predictor.save(save_path)
        print(f"Model saved to {save_path}")
    
    return predictor
