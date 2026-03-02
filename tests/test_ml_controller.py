#!/usr/bin/env python3
"""Tests for MLController (8+ tests)."""

import numpy as np
import pytest
import time

from engines.ml_controller import MLController, _sigmoid


@pytest.fixture(scope="module")
def controller():
    return MLController(input_dim=5, hidden_dims=[32, 16, 8], V_G_max=5.0, seed=42)


class TestMLPArchitecture:
    def test_weight_dimensions(self, controller):
        dims = [5, 32, 16, 8, 1]
        for i, (W, b) in enumerate(zip(controller.weights, controller.biases)):
            assert W.shape == (dims[i], dims[i + 1])
            assert b.shape == (dims[i + 1],)

    def test_forward_pass_shape(self, controller):
        x = np.random.randn(10, 5)
        out, acts = controller._forward(x)
        assert out.shape == (10, 1)
        assert len(acts) == 5  # input + 4 layers

    def test_predict_scalar(self, controller):
        state = np.array([0.5, 300, 10, 0.5, 15])
        v = controller.predict(state)
        assert isinstance(v, float)
        assert 0 <= v <= 5.0

    def test_predict_in_bounds(self, controller):
        """All predictions must be in [0, V_G_max]."""
        for _ in range(50):
            state = np.random.randn(5)
            v = controller.predict(state)
            assert 0 <= v <= 5.0


class TestTraining:
    def test_generate_episodes(self):
        """Test with a mock surrogate."""
        class MockSurrogate:
            def predict_steady(self, V_G, G, T):
                return {'P_out': V_G * G * 5}

        ctrl = MLController(hidden_dims=[16, 8], seed=0)
        episodes = ctrl.generate_training_episodes(MockSurrogate(), n_episodes=10, steps_per_episode=12)
        assert 'X' in episodes and 'y' in episodes
        assert episodes['X'].shape[1] == 5
        assert len(episodes['y']) == episodes['n_samples']

    def test_training_convergence(self):
        """Loss should decrease over epochs."""
        ctrl = MLController(hidden_dims=[16, 8], seed=0)

        class MockSurrogate:
            def predict_steady(self, V_G, G, T):
                return {'P_out': V_G * G * 5}

        episodes = ctrl.generate_training_episodes(MockSurrogate(), n_episodes=20, steps_per_episode=24)
        losses = ctrl.train(episodes, epochs=50, lr=0.005, batch_size=32)

        assert len(losses) == 50
        # Average of last 10 should be less than first 10
        assert np.mean(losses[-10:]) < np.mean(losses[:10])

    def test_predict_after_training(self):
        ctrl = MLController(hidden_dims=[16, 8], seed=0)

        class MockSurrogate:
            def predict_steady(self, V_G, G, T):
                return {'P_out': V_G * G * 5}

        episodes = ctrl.generate_training_episodes(MockSurrogate(), n_episodes=10, steps_per_episode=12)
        ctrl.train(episodes, epochs=30, lr=0.005)

        v = ctrl.predict(np.array([0.5, 300, 10, 0.5, 15]))
        assert 0 <= v <= 5.0


class TestPerformance:
    def test_prediction_speed(self, controller):
        """Single prediction should be < 1ms."""
        state = np.array([0.5, 300, 10, 0.5, 15])
        # Warm up
        controller.predict(state)

        t0 = time.perf_counter()
        for _ in range(1000):
            controller.predict(state)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 0.001  # < 1ms per prediction

    def test_evaluate_metrics(self):
        ctrl = MLController(hidden_dims=[16, 8], seed=0)

        class MockSurrogate:
            def predict_steady(self, V_G, G, T):
                return {'P_out': V_G * G * 5}

        episodes = ctrl.generate_training_episodes(MockSurrogate(), n_episodes=10, steps_per_episode=12)
        ctrl.train(episodes, epochs=20, lr=0.005)
        metrics = ctrl.evaluate(episodes)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['in_bounds']


class TestSigmoid:
    def test_sigmoid_range(self):
        x = np.linspace(-100, 100, 200)
        y = _sigmoid(x)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_sigmoid_midpoint(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10
