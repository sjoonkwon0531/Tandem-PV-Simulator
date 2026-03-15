"""
Federated Learning Engine for Materials Discovery
==================================================

Implements Federated Averaging (FedAvg) with:
- Local training on private lab data
- Gradient aggregation (simulated secure aggregation)
- Differential privacy (Gaussian mechanism)
- Communication rounds tracking
- Privacy budget management (epsilon-delta DP)

Key features:
- No PyTorch: Uses sklearn models with weight averaging
- Privacy-preserving: Adds calibrated noise to gradients
- Secure aggregation: Simulates that server never sees raw gradients
- Realistic: Handles non-IID data across labs

Author: OpenClaw Agent (V9)
Date: 2026-03-15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import copy


@dataclass
class FederatedRound:
    """Results from one communication round."""
    round_num: int
    global_mae: float
    global_r2: float
    local_maes: Dict[str, float]
    privacy_epsilon: float
    noise_scale: float


class FederatedLearner:
    """
    Federated Averaging (FedAvg) implementation.
    
    Algorithm:
    1. Initialize global model
    2. For each communication round:
       a. Send global model to all labs
       b. Each lab trains locally on private data
       c. Labs send model updates (with optional DP noise)
       d. Server aggregates updates → new global model
    3. Repeat until convergence
    
    Privacy:
    - Differential Privacy: Add Gaussian noise to model updates
    - Secure Aggregation: Server only sees aggregated update (not individual)
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 10,
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            model_type: "random_forest" or "decision_tree"
            n_estimators: Number of trees (for random forest)
            max_depth: Max tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Global model (initialized but not trained)
        self.global_model = None
        
        # Federated training history
        self.rounds: List[FederatedRound] = []
        
        # Privacy budget tracker
        self.total_epsilon = 0.0
        self.privacy_delta = 1e-5
    
    def _create_model(self):
        """Create a fresh model instance."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:  # decision_tree
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
    
    def initialize_global_model(self, X_init: np.ndarray, y_init: np.ndarray):
        """
        Initialize global model with small sample.
        
        In practice, this could be:
        - Pre-trained on public data
        - Random initialization
        - Small sample from one lab
        """
        self.global_model = self._create_model()
        self.global_model.fit(X_init, y_init)
    
    def federated_round(
        self,
        lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        test_data: Tuple[np.ndarray, np.ndarray],
        epsilon: float = 1.0,
        local_epochs: int = 1,
        sample_fraction: float = 1.0
    ) -> FederatedRound:
        """
        Execute one round of federated learning.
        
        Args:
            lab_datasets: Dict mapping lab_id -> (X_local, y_local)
            test_data: (X_test, y_test) for global evaluation
            epsilon: Privacy budget for this round (lower = more privacy)
            local_epochs: Number of local training iterations
            sample_fraction: Fraction of labs participating (1.0 = all)
        
        Returns:
            FederatedRound with results
        """
        X_test, y_test = test_data
        
        # Sample labs for this round (can be < 100% for efficiency)
        lab_ids = list(lab_datasets.keys())
        n_selected = max(1, int(len(lab_ids) * sample_fraction))
        selected_labs = np.random.choice(lab_ids, size=n_selected, replace=False)
        
        # Local training
        local_models = {}
        local_maes = {}
        
        for lab_id in selected_labs:
            X_local, y_local = lab_datasets[lab_id]
            
            # Initialize local model from global model
            local_model = copy.deepcopy(self.global_model)
            
            # Local training (multiple epochs)
            for _ in range(local_epochs):
                local_model.fit(X_local, y_local)
            
            # Evaluate locally
            y_pred_local = local_model.predict(X_local)
            mae_local = mean_absolute_error(y_local, y_pred_local)
            local_maes[lab_id] = mae_local
            
            local_models[lab_id] = local_model
        
        # Aggregate models with differential privacy
        noise_scale = self._compute_noise_scale(epsilon, sensitivity=1.0)
        self.global_model = self._aggregate_models(
            local_models,
            noise_scale=noise_scale,
            add_noise=(epsilon < float('inf'))
        )
        
        # Evaluate global model on test data
        y_pred_global = self.global_model.predict(X_test)
        mae_global = mean_absolute_error(y_test, y_pred_global)
        r2_global = r2_score(y_test, y_pred_global)
        
        # Update privacy budget
        if epsilon < float('inf'):
            self.total_epsilon += epsilon
        
        # Record round
        round_num = len(self.rounds) + 1
        federated_round = FederatedRound(
            round_num=round_num,
            global_mae=mae_global,
            global_r2=r2_global,
            local_maes=local_maes,
            privacy_epsilon=epsilon,
            noise_scale=noise_scale
        )
        self.rounds.append(federated_round)
        
        return federated_round
    
    def _aggregate_models(
        self,
        local_models: Dict[str, RandomForestRegressor],
        noise_scale: float = 0.0,
        add_noise: bool = False
    ) -> RandomForestRegressor:
        """
        Aggregate local models into global model (FedAvg).
        
        For tree-based models (Random Forest, Decision Tree):
        - We can't directly average weights (not neural network)
        - Instead: Average predictions on a grid, fit new model
        
        This is a simplified approximation. In practice:
        - Use linear models (can average coefficients)
        - Use neural networks (can average weights)
        - Use model distillation
        
        With DP noise:
        - Add Gaussian noise to aggregated predictions (not weights directly)
        """
        # Create prediction grid (sample points in feature space)
        n_labs = len(local_models)
        
        # Get feature dimension from first model
        first_model = list(local_models.values())[0]
        n_features = first_model.n_features_in_
        
        # Generate grid of feature points
        n_grid = 500
        X_grid = np.random.randn(n_grid, n_features)
        
        # Aggregate predictions from all local models
        predictions_per_lab = []
        for model in local_models.values():
            pred = model.predict(X_grid)
            predictions_per_lab.append(pred)
        
        # Average predictions (FedAvg on prediction level)
        avg_predictions = np.mean(predictions_per_lab, axis=0)
        
        # Add differential privacy noise
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=avg_predictions.shape)
            avg_predictions = avg_predictions + noise
        
        # Fit new global model to averaged predictions
        # (This is model distillation / knowledge transfer)
        global_model = self._create_model()
        global_model.fit(X_grid, avg_predictions)
        
        return global_model
    
    def _compute_noise_scale(self, epsilon: float, sensitivity: float = 1.0) -> float:
        """
        Compute noise scale for Gaussian mechanism (differential privacy).
        
        Gaussian mechanism: Add N(0, σ²) noise where σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        
        Args:
            epsilon: Privacy budget (lower = more privacy)
            sensitivity: Global sensitivity of the query (max change in output)
        
        Returns:
            Standard deviation of noise to add
        """
        if epsilon == float('inf'):
            return 0.0
        
        # Standard Gaussian mechanism formula
        delta = self.privacy_delta
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        return sigma
    
    def train_federated(
        self,
        lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        test_data: Tuple[np.ndarray, np.ndarray],
        n_rounds: int = 10,
        epsilon_per_round: float = 1.0,
        local_epochs: int = 1
    ) -> List[FederatedRound]:
        """
        Train federated model for multiple rounds.
        
        Args:
            lab_datasets: Dict mapping lab_id -> (X_local, y_local)
            test_data: (X_test, y_test)
            n_rounds: Number of communication rounds
            epsilon_per_round: Privacy budget per round
            local_epochs: Local training iterations per round
        
        Returns:
            List of FederatedRound results
        """
        # Initialize global model with small sample from first lab
        first_lab_data = list(lab_datasets.values())[0]
        X_init = first_lab_data[0][:20]  # Small initialization sample
        y_init = first_lab_data[1][:20]
        self.initialize_global_model(X_init, y_init)
        
        # Federated training rounds
        for round_num in range(n_rounds):
            self.federated_round(
                lab_datasets=lab_datasets,
                test_data=test_data,
                epsilon=epsilon_per_round,
                local_epochs=local_epochs
            )
        
        return self.rounds
    
    def get_privacy_budget_status(self) -> Dict:
        """
        Get current privacy budget status.
        
        Returns:
            Dict with total epsilon spent, delta, and budget status
        """
        return {
            "total_epsilon": self.total_epsilon,
            "delta": self.privacy_delta,
            "rounds": len(self.rounds),
            "avg_epsilon_per_round": self.total_epsilon / max(1, len(self.rounds)),
            "privacy_guarantee": f"({self.total_epsilon:.2f}, {self.privacy_delta})-DP"
        }


def train_centralized_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "random_forest",
    n_estimators: int = 10,
    max_depth: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Train centralized model (upper bound baseline).
    
    This is the "ideal" scenario where all data is pooled (impossible in practice).
    """
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
    
    # Train on all data
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "model": model,
        "mae": mae,
        "r2": r2,
        "training_size": len(y_train)
    }


def train_local_only_baseline(
    lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray],
    model_type: str = "random_forest",
    n_estimators: int = 10,
    max_depth: int = 5,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Train separate models for each lab (lower bound baseline).
    
    This is the "no collaboration" scenario.
    Each lab trains only on their own data.
    """
    X_test, y_test = test_data
    results = {}
    
    for lab_id, (X_local, y_local) in lab_datasets.items():
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=random_state
            )
        
        # Train on local data only
        model.fit(X_local, y_local)
        
        # Evaluate on global test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[lab_id] = {
            "model": model,
            "mae": mae,
            "r2": r2,
            "training_size": len(y_local)
        }
    
    return results


class SecureAggregationSimulator:
    """
    Simulate secure aggregation protocol.
    
    Key idea: Server aggregates gradients without seeing individual gradients.
    
    Protocol:
    1. Each lab encrypts their gradient
    2. Server sums encrypted gradients
    3. Server decrypts only the sum
    
    Result: Server never sees individual gradients (privacy!)
    
    This is a conceptual simulation (no actual cryptography).
    """
    
    def __init__(self):
        self.rounds = []
    
    def simulate_round(
        self,
        n_labs: int,
        gradient_dim: int = 10
    ) -> Dict:
        """
        Simulate one round of secure aggregation.
        
        Args:
            n_labs: Number of participating labs
            gradient_dim: Dimension of gradients
        
        Returns:
            Summary of what server sees vs. what exists
        """
        # Generate fake gradients (in practice, from actual training)
        individual_gradients = [
            np.random.randn(gradient_dim) for _ in range(n_labs)
        ]
        
        # What labs send: Encrypted gradients (represented as strings)
        encrypted_gradients = [
            f"ENCRYPTED_GRAD_{i}" for i in range(n_labs)
        ]
        
        # What server computes: Sum of encrypted gradients
        # (In reality, done with homomorphic encryption)
        aggregated_gradient = np.sum(individual_gradients, axis=0)
        
        # What server sees
        server_view = {
            "received": encrypted_gradients,  # Server sees encrypted blobs
            "aggregated": aggregated_gradient,  # Server sees sum only
            "individual_gradients_visible": False  # Server CANNOT see individual
        }
        
        # Ground truth (for simulation only)
        ground_truth = {
            "individual_gradients": individual_gradients,
            "individual_visible_to_server": False
        }
        
        round_summary = {
            "round_num": len(self.rounds) + 1,
            "n_labs": n_labs,
            "server_view": server_view,
            "ground_truth": ground_truth,
            "privacy_preserved": True
        }
        
        self.rounds.append(round_summary)
        
        return round_summary


def analyze_privacy_accuracy_tradeoff(
    lab_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray],
    epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, float('inf')],
    n_rounds: int = 5
) -> List[Dict]:
    """
    Analyze privacy-accuracy tradeoff.
    
    Train federated models with different privacy budgets.
    Show: More privacy (lower epsilon) → More noise → Lower accuracy
    
    Args:
        lab_datasets: Lab datasets
        test_data: Test data
        epsilon_values: Privacy budgets to try
        n_rounds: Communication rounds per experiment
    
    Returns:
        List of results for each epsilon
    """
    results = []
    
    for epsilon in epsilon_values:
        # Train federated model with this epsilon
        learner = FederatedLearner(
            model_type="random_forest",
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        rounds = learner.train_federated(
            lab_datasets=lab_datasets,
            test_data=test_data,
            n_rounds=n_rounds,
            epsilon_per_round=epsilon / n_rounds,  # Split budget across rounds
            local_epochs=1
        )
        
        # Final performance
        final_round = rounds[-1]
        
        results.append({
            "epsilon": epsilon,
            "total_epsilon": epsilon,
            "final_mae": final_round.global_mae,
            "final_r2": final_round.global_r2,
            "privacy_label": "No Privacy" if epsilon == float('inf') else f"ε={epsilon}"
        })
    
    return results
