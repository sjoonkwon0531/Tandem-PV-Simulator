"""
Multi-Lab Data Simulator with Controlled Non-IID Distribution
==============================================================

Simulates N independent labs, each with:
- Private local dataset
- Different data distributions (non-IID, realistic)
- Varying data quality, size, and coverage

Addresses the data silo problem in materials science:
"Lab A has halides, Lab B has oxides, Lab C has both but noisy"

Author: OpenClaw Agent (V9)
Date: 2026-03-15
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import wasserstein_distance
from scipy.special import kl_div


@dataclass
class LabProfile:
    """Profile for a simulated lab."""
    lab_id: str
    name: str
    specialty: str  # "halides", "oxides", "mixed"
    data_size: int
    noise_level: float  # 0.0-1.0
    bandgap_bias: float  # Systematic bias in measurements
    coverage_min: float  # Min bandgap in dataset
    coverage_max: float  # Max bandgap in dataset
    data_quality: float  # 0.0-1.0


class LabDataSimulator:
    """
    Generate synthetic datasets for multiple labs with controlled heterogeneity.
    
    Key features:
    - Non-IID: Each lab has different data distribution
    - Realistic: Labs specialize in different materials
    - Controlled: Set heterogeneity level (low/medium/high)
    - Metrics: KL divergence, Earth Mover's Distance between labs
    """
    
    def __init__(self, n_labs: int = 5, heterogeneity: str = "medium", random_state: int = 42):
        """
        Args:
            n_labs: Number of labs to simulate (3-10)
            heterogeneity: "low", "medium", "high" (controls distribution overlap)
            random_state: Random seed for reproducibility
        """
        self.n_labs = n_labs
        self.heterogeneity = heterogeneity
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Lab specialties (round-robin)
        specialties = ["halides", "oxides", "mixed"]
        
        # Heterogeneity parameters
        heterogeneity_params = {
            "low": {"coverage_overlap": 0.8, "noise_range": (0.05, 0.15), "size_variance": 0.3},
            "medium": {"coverage_overlap": 0.5, "noise_range": (0.05, 0.25), "size_variance": 0.5},
            "high": {"coverage_overlap": 0.2, "noise_range": (0.1, 0.4), "size_variance": 0.8}
        }
        
        params = heterogeneity_params[heterogeneity]
        
        # Generate lab profiles
        self.labs = []
        base_size = 150
        
        for i in range(n_labs):
            specialty = specialties[i % len(specialties)]
            
            # Data size: varying significantly
            size_factor = 1.0 + self.rng.uniform(-params["size_variance"], params["size_variance"])
            data_size = int(base_size * size_factor)
            
            # Noise level: different measurement quality
            noise_level = self.rng.uniform(*params["noise_range"])
            
            # Bandgap coverage: different ranges based on specialty
            if specialty == "halides":
                # Halides: 1.0-3.0 eV typically
                coverage_min = 1.0 + self.rng.uniform(0, 0.3)
                coverage_max = 2.5 + self.rng.uniform(0, 0.5)
            elif specialty == "oxides":
                # Oxides: 2.0-5.0 eV typically (wider bandgaps)
                coverage_min = 2.0 + self.rng.uniform(0, 0.5)
                coverage_max = 4.5 + self.rng.uniform(0, 0.5)
            else:  # mixed
                # Mixed: broader coverage
                coverage_min = 1.2 + self.rng.uniform(0, 0.3)
                coverage_max = 4.0 + self.rng.uniform(0, 0.5)
            
            # Systematic bias (some labs measure consistently high/low)
            bandgap_bias = self.rng.uniform(-0.1, 0.1)
            
            # Data quality (inverse of noise)
            data_quality = 1.0 - noise_level
            
            lab = LabProfile(
                lab_id=f"lab_{i+1}",
                name=f"Lab {chr(65+i)}",  # Lab A, Lab B, ...
                specialty=specialty,
                data_size=data_size,
                noise_level=noise_level,
                bandgap_bias=bandgap_bias,
                coverage_min=coverage_min,
                coverage_max=coverage_max,
                data_quality=data_quality
            )
            self.labs.append(lab)
    
    def generate_lab_data(self, lab: LabProfile, global_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset for a specific lab.
        
        Args:
            lab: Lab profile
            global_features: Global feature pool (N x D)
        
        Returns:
            X_local: Local features (n_local x D)
            y_local: Local labels with noise and bias
        """
        n_global = len(global_features)
        
        # Sample indices based on specialty (non-IID!)
        # Each lab sees a biased subset of the global data
        if lab.specialty == "halides":
            # Prefer lower feature indices (halides cluster together)
            probs = np.exp(-np.arange(n_global) / (n_global * 0.3))
        elif lab.specialty == "oxides":
            # Prefer higher feature indices (oxides cluster together)
            probs = np.exp(-np.flip(np.arange(n_global)) / (n_global * 0.3))
        else:  # mixed
            # More uniform
            probs = np.ones(n_global)
        
        probs = probs / probs.sum()
        
        # Sample local data
        indices = self.rng.choice(n_global, size=lab.data_size, replace=False, p=probs)
        X_local = global_features[indices]
        
        # Generate true labels (from global function)
        # True function: linear combination + nonlinearity
        true_labels = self._true_function(X_local)
        
        # Filter by coverage range (labs can only measure certain bandgaps)
        mask = (true_labels >= lab.coverage_min) & (true_labels <= lab.coverage_max)
        X_local = X_local[mask]
        true_labels = true_labels[mask]
        
        # Add systematic bias
        y_local = true_labels + lab.bandgap_bias
        
        # Add measurement noise
        noise = self.rng.normal(0, lab.noise_level, size=len(y_local))
        y_local = y_local + noise
        
        # Ensure we have enough data
        if len(y_local) < 20:
            # Re-sample with replacement if too few after filtering
            indices = self.rng.choice(len(y_local), size=20, replace=True)
            X_local = X_local[indices]
            y_local = y_local[indices]
        
        return X_local, y_local
    
    def _true_function(self, X: np.ndarray) -> np.ndarray:
        """
        Ground truth bandgap function.
        
        Simulates physics: bandgap depends on ionic radius, electronegativity, etc.
        """
        # Simplified: linear combination + sine for nonlinearity
        # Features assumed: [ionic_radius_A, electronegativity_B, ...]
        
        # Base: weighted sum of features
        weights = np.array([0.8, -0.6, 0.4, -0.3, 0.5])[:X.shape[1]]
        base = X @ weights
        
        # Nonlinearity
        nonlinear = 0.3 * np.sin(X[:, 0] * 2) if X.shape[1] > 0 else 0
        
        # Scale to realistic bandgap range (1-5 eV)
        bandgap = 2.5 + base + nonlinear
        bandgap = np.clip(bandgap, 0.5, 6.0)
        
        return bandgap
    
    def generate_all_labs(self, n_features: int = 5) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate datasets for all labs.
        
        Args:
            n_features: Number of features per sample
        
        Returns:
            Dictionary mapping lab_id -> (X_local, y_local)
        """
        # Generate global feature pool (large)
        n_global = 1000
        global_features = self.rng.randn(n_global, n_features)
        
        # Normalize features to reasonable ranges
        # Feature 0: ionic radius (1.0-2.5 Å)
        global_features[:, 0] = 1.0 + (global_features[:, 0] - global_features[:, 0].min()) / \
                                (global_features[:, 0].max() - global_features[:, 0].min()) * 1.5
        
        # Feature 1: electronegativity (0.5-3.0)
        if n_features > 1:
            global_features[:, 1] = 0.5 + (global_features[:, 1] - global_features[:, 1].min()) / \
                                    (global_features[:, 1].max() - global_features[:, 1].min()) * 2.5
        
        # Other features: normalized to [-1, 1]
        for i in range(2, n_features):
            global_features[:, i] = (global_features[:, i] - global_features[:, i].mean()) / \
                                    global_features[:, i].std()
        
        # Generate data for each lab
        lab_datasets = {}
        for lab in self.labs:
            X_local, y_local = self.generate_lab_data(lab, global_features)
            lab_datasets[lab.lab_id] = (X_local, y_local)
        
        return lab_datasets
    
    def compute_heterogeneity_metrics(self, lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Quantify how different labs' data distributions are.
        
        Metrics:
        - KL divergence (distribution difference)
        - Earth Mover's Distance (Wasserstein)
        - Coverage overlap
        - Label distribution statistics
        """
        lab_ids = list(lab_datasets.keys())
        n = len(lab_ids)
        
        # Pairwise KL divergences (averaged)
        kl_divergences = []
        emd_distances = []
        
        for i in range(n):
            for j in range(i+1, n):
                X_i, y_i = lab_datasets[lab_ids[i]]
                X_j, y_j = lab_datasets[lab_ids[j]]
                
                # Histogram of labels (for distribution comparison)
                bins = np.linspace(0.5, 6.0, 30)
                hist_i, _ = np.histogram(y_i, bins=bins, density=True)
                hist_j, _ = np.histogram(y_j, bins=bins, density=True)
                
                # Avoid zeros in KL divergence
                hist_i = hist_i + 1e-10
                hist_j = hist_j + 1e-10
                
                # KL divergence (symmetrized)
                kl_ij = np.sum(kl_div(hist_i, hist_j))
                kl_ji = np.sum(kl_div(hist_j, hist_i))
                kl_sym = (kl_ij + kl_ji) / 2
                kl_divergences.append(kl_sym)
                
                # Earth Mover's Distance (Wasserstein)
                emd = wasserstein_distance(y_i, y_j)
                emd_distances.append(emd)
        
        # Label statistics per lab
        lab_stats = {}
        for lab_id, (X, y) in lab_datasets.items():
            lab_stats[lab_id] = {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "size": len(y)
            }
        
        # Overall metrics
        metrics = {
            "avg_kl_divergence": float(np.mean(kl_divergences)),
            "max_kl_divergence": float(np.max(kl_divergences)),
            "avg_emd": float(np.mean(emd_distances)),
            "max_emd": float(np.max(emd_distances)),
            "lab_stats": lab_stats,
            "heterogeneity_level": self.heterogeneity
        }
        
        return metrics
    
    def get_lab_profiles_df(self) -> pd.DataFrame:
        """Return lab profiles as DataFrame for display."""
        data = []
        for lab in self.labs:
            data.append({
                "Lab ID": lab.lab_id,
                "Name": lab.name,
                "Specialty": lab.specialty,
                "Data Size": lab.data_size,
                "Noise Level": f"{lab.noise_level:.2f}",
                "Bandgap Bias": f"{lab.bandgap_bias:+.2f} eV",
                "Coverage": f"{lab.coverage_min:.1f}-{lab.coverage_max:.1f} eV",
                "Quality": f"{lab.data_quality:.2%}"
            })
        return pd.DataFrame(data)
    
    def recommend_most_valuable_lab(self, lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
        """
        Recommend which lab has the most unique/valuable data.
        
        Strategy: Lab with most different distribution from others = highest marginal value
        """
        lab_ids = list(lab_datasets.keys())
        uniqueness_scores = {}
        
        for lab_id in lab_ids:
            # Compute average distance to all other labs
            distances = []
            for other_id in lab_ids:
                if other_id != lab_id:
                    y_i = lab_datasets[lab_id][1]
                    y_j = lab_datasets[other_id][1]
                    emd = wasserstein_distance(y_i, y_j)
                    distances.append(emd)
            
            uniqueness_scores[lab_id] = np.mean(distances)
        
        # Most unique lab
        most_unique = max(uniqueness_scores, key=uniqueness_scores.get)
        
        return most_unique


def generate_centralized_dataset(lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge all lab datasets into one centralized dataset.
    
    This is the "ideal" scenario (impossible in practice due to privacy/IP).
    Used as upper-bound baseline.
    """
    X_all = []
    y_all = []
    
    for X, y in lab_datasets.values():
        X_all.append(X)
        y_all.append(y)
    
    X_centralized = np.vstack(X_all)
    y_centralized = np.hstack(y_all)
    
    return X_centralized, y_centralized
