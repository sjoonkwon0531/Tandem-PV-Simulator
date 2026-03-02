#!/usr/bin/env python3
"""
ML-based Optimal PV Controller
================================

Numpy-only MLP with Adam optimizer for real-time V_G control.

Input:  [G, T, Load, SOC_HESS, P_pv_current]  (5-dim)
Output: V_G_optimal (scalar)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class MLController:
    """MLP-based controller for optimal gate voltage prediction."""

    def __init__(self, model_type: str = 'mlp', input_dim: int = 5,
                 hidden_dims: Optional[List[int]] = None,
                 V_G_max: float = 5.0, seed: int = 42):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.V_G_max = V_G_max
        self.rng = np.random.default_rng(seed)

        # Build MLP weights
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        dims = [input_dim] + self.hidden_dims + [1]
        for i in range(len(dims) - 1):
            # He initialization
            std = np.sqrt(2.0 / dims[i])
            self.weights.append(self.rng.normal(0, std, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))

        # Input normalization
        self._input_mean = np.zeros(input_dim)
        self._input_std = np.ones(input_dim)

        # Adam state
        self._adam_m_w = [np.zeros_like(w) for w in self.weights]
        self._adam_v_w = [np.zeros_like(w) for w in self.weights]
        self._adam_m_b = [np.zeros_like(b) for b in self.biases]
        self._adam_v_b = [np.zeros_like(b) for b in self.biases]
        self._adam_t = 0

        self.training_history: List[float] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass. x: (batch, input_dim). Returns (output, activations)."""
        activations = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(z, 0)  # ReLU
            else:
                h = self.V_G_max * _sigmoid(z)  # Output in [0, V_G_max]
            activations.append(h)
        return h, activations

    def predict(self, state: np.ndarray) -> float:
        """Predict optimal V_G from state vector.

        Args:
            state: [G, T, Load, SOC_HESS, P_pv_current] shape (5,) or (N,5)

        Returns:
            V_G prediction (scalar or array)
        """
        x = np.atleast_2d(state)
        x_norm = (x - self._input_mean) / (self._input_std + 1e-10)
        out, _ = self._forward(x_norm)
        result = out.flatten()
        return float(result[0]) if result.size == 1 else result

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------
    def generate_training_episodes(self, surrogate, n_episodes: int = 200,
                                   steps_per_episode: int = 48,
                                   seed: int = 99) -> Dict:
        """Generate training data using surrogate for fast evaluation.

        Each episode = 24h with 30-min steps.
        """
        rng = np.random.default_rng(seed)
        all_X = []
        all_y = []

        for ep in range(n_episodes):
            # Random weather/load profile
            G_base = rng.uniform(0.2, 1.0)
            T_base = rng.uniform(270, 330)
            Load_base = rng.uniform(5, 20)  # mW/cm²

            for step in range(steps_per_episode):
                hour = step * 0.5  # half-hour steps
                # Diurnal irradiance
                G = G_base * max(0, np.sin(np.pi * hour / 24)) + 0.01
                T = T_base + 10 * np.sin(np.pi * (hour - 6) / 12)
                Load = Load_base * (1 + 0.3 * np.sin(2 * np.pi * hour / 24))
                SOC = rng.uniform(0.2, 0.8)
                P_current = rng.uniform(0, 20)

                state = np.array([G, T, Load, SOC, P_current])

                # Find optimal V_G by grid search on surrogate
                best_vg = 0.0
                best_err = float('inf')
                for vg in np.linspace(0, self.V_G_max, 30):
                    try:
                        pred = surrogate.predict_steady(vg, G, T)
                        err = abs(pred['P_out'] - Load)
                    except Exception:
                        err = float('inf')
                    if err < best_err:
                        best_err = err
                        best_vg = vg

                all_X.append(state)
                all_y.append(best_vg)

        X = np.array(all_X)
        y = np.array(all_y)

        # Compute normalization stats
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-10

        return {'X': X, 'y': y, 'n_samples': len(X)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, episodes: Dict, epochs: int = 100, lr: float = 0.001,
              batch_size: int = 64, lambda_smooth: float = 0.001) -> List[float]:
        """Train MLP with Adam optimizer.

        Args:
            episodes: dict with 'X', 'y' arrays
            epochs: training epochs
            lr: learning rate
            batch_size: mini-batch size
            lambda_smooth: smoothness penalty weight

        Returns:
            List of epoch losses
        """
        X = episodes['X']
        y = episodes['y'].reshape(-1, 1)
        N = len(X)

        # Normalize inputs
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-10
        X_norm = (X - self._input_mean) / self._input_std

        losses = []
        for epoch in range(epochs):
            # Shuffle
            perm = self.rng.permutation(N)
            X_shuf = X_norm[perm]
            y_shuf = y[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_b = X_shuf[start:end]
                y_b = y_shuf[start:end]

                # Forward
                y_pred, acts = self._forward(X_b)

                # Loss: MSE + smoothness penalty
                diff = y_pred - y_b
                loss = np.mean(diff ** 2)
                # Smoothness: penalize large weights in last layer
                loss += lambda_smooth * np.mean(self.weights[-1] ** 2)

                # Backward (manual backprop)
                grad_out = 2 * diff / len(X_b)
                # Through sigmoid output
                sig_val = y_pred / self.V_G_max
                grad_out = grad_out * self.V_G_max * sig_val * (1 - sig_val)

                grads_w = []
                grads_b = []

                delta = grad_out
                for i in range(len(self.weights) - 1, -1, -1):
                    grads_w.insert(0, acts[i].T @ delta)
                    grads_b.insert(0, delta.sum(axis=0))
                    if i > 0:
                        delta = delta @ self.weights[i].T
                        delta = delta * (acts[i] > 0).astype(float)  # ReLU grad

                # Smoothness gradient on last layer
                grads_w[-1] += 2 * lambda_smooth * self.weights[-1] / len(X_b)

                # Adam update
                self._adam_t += 1
                for i in range(len(self.weights)):
                    self._adam_step(self.weights, self._adam_m_w, self._adam_v_w,
                                    i, grads_w[i], lr)
                    self._adam_step(self.biases, self._adam_m_b, self._adam_v_b,
                                    i, grads_b[i], lr)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        self.training_history = losses
        return losses

    def _adam_step(self, params, m_list, v_list, idx, grad, lr,
                   beta1=0.9, beta2=0.999, eps=1e-8):
        """Single Adam optimizer step."""
        m_list[idx] = beta1 * m_list[idx] + (1 - beta1) * grad
        v_list[idx] = beta2 * v_list[idx] + (1 - beta2) * grad ** 2
        m_hat = m_list[idx] / (1 - beta1 ** self._adam_t)
        v_hat = v_list[idx] / (1 - beta2 ** self._adam_t)
        params[idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, test_scenarios: Dict) -> Dict:
        """Evaluate controller performance.

        Args:
            test_scenarios: dict with 'X' states and 'y' optimal V_G

        Returns:
            Performance metrics
        """
        X = test_scenarios['X']
        y_true = test_scenarios['y']

        y_pred = np.array([self.predict(X[i]) for i in range(len(X))])

        tracking_error = np.abs(y_pred - y_true)
        control_effort = np.mean(np.abs(np.diff(y_pred))) if len(y_pred) > 1 else 0.0

        return {
            'mae': float(np.mean(tracking_error)),
            'rmse': float(np.sqrt(np.mean(tracking_error ** 2))),
            'max_error': float(np.max(tracking_error)),
            'mean_control_effort': float(control_effort),
            'V_G_range': (float(y_pred.min()), float(y_pred.max())),
            'in_bounds': bool(np.all((y_pred >= 0) & (y_pred <= self.V_G_max))),
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500)))
    )
