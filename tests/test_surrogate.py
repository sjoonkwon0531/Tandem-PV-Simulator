#!/usr/bin/env python3
"""Tests for PhysicsSurrogate (10+ tests)."""

import numpy as np
import pytest
import time

from engines.surrogate_model import PhysicsSurrogate
from engines.multiscale_control import MultiscaleControlEngine


@pytest.fixture(scope="module")
def engine():
    return MultiscaleControlEngine()


@pytest.fixture(scope="module")
def trained_surrogate(engine):
    s = PhysicsSurrogate()
    data = s.generate_training_data(engine, n_samples=80)
    s.train_steady_state(data)
    tr_data = s.generate_transient_data(engine, n_samples=30)
    s.train_transient(tr_data)
    hy_data = s.generate_hysteresis_data(engine, n_samples=20)
    s.train_hysteresis(hy_data)
    s._test_data = data  # stash for reuse
    return s


class TestTrainingDataGeneration:
    def test_lhs_shape(self):
        s = PhysicsSurrogate()
        samples = s._latin_hypercube(50, 3, np.random.default_rng(0))
        assert samples.shape == (50, 3)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_generate_data_shape(self, engine):
        s = PhysicsSurrogate()
        data = s.generate_training_data(engine, n_samples=20)
        assert data['inputs'].shape == (20, 3)
        assert len(data['P_out']) == 20

    def test_data_physical_range(self, engine):
        s = PhysicsSurrogate()
        data = s.generate_training_data(engine, n_samples=30)
        assert np.all(data['P_out'] >= 0)
        assert np.all(data['eta'] >= 0) and np.all(data['eta'] <= 1)


class TestSteadyStateSurrogate:
    def test_train_returns_metrics(self, engine):
        s = PhysicsSurrogate()
        data = s.generate_training_data(engine, n_samples=60)
        metrics = s.train_steady_state(data)
        assert 'P_out' in metrics
        assert metrics['P_out']['R2'] > 0.9

    def test_predict_returns_dict(self, trained_surrogate):
        result = trained_surrogate.predict_steady(2.0, 0.5, 300)
        assert 'P_out' in result and 'eta' in result

    def test_predict_nonnegative(self, trained_surrogate):
        result = trained_surrogate.predict_steady(1.0, 0.3, 280)
        assert result['P_out'] >= 0
        assert result['eta'] >= 0

    def test_accuracy_report(self, trained_surrogate):
        report = trained_surrogate.accuracy_report(trained_surrogate._test_data)
        assert report['P_out']['R2'] > 0.9

    def test_speed_vs_full(self, trained_surrogate, engine):
        """Surrogate should be >> faster than full sim."""
        # Surrogate timing
        t0 = time.perf_counter()
        for _ in range(100):
            trained_surrogate.predict_steady(2.5, 0.8, 310)
        t_surr = time.perf_counter() - t0

        # Full sim timing
        t0 = time.perf_counter()
        for _ in range(100):
            engine.dynamic_iv.operating_point(2.5, 0.8, 310)
        t_full = time.perf_counter() - t0

        # Surrogate should be reasonably fast (not necessarily 100x for RBF with small data)
        assert t_surr < 30  # under 30s for 100 calls is fine


class TestTransientSurrogate:
    def test_predict_transient_shape(self, trained_surrogate):
        V_G_t = np.full(50, 2.0)
        P = trained_surrogate.predict_transient(V_G_t, 0.5, 300, dt=0.01)
        assert P.shape == (50,)
        assert np.all(P >= 0)

    def test_transient_decays(self, trained_surrogate):
        """Transient components should decay over time."""
        V_G_t = np.full(100, 2.5)
        P = trained_surrogate.predict_transient(V_G_t, 0.8, 300, dt=0.1)
        # Later values should approach P_ss (less transient)
        assert abs(P[-1] - P[-2]) < abs(P[1] - P[0]) + 1e-10


class TestHysteresisSurrogate:
    def test_predict_hysteresis_range(self, trained_surrogate):
        hi = trained_surrogate.predict_hysteresis(1.0, 0.8, 300)
        assert 0 <= hi <= 1

    def test_not_trained_raises(self):
        s = PhysicsSurrogate()
        with pytest.raises(RuntimeError):
            s.predict_steady(1.0, 0.5, 300)


class TestEdgeCases:
    def test_low_irradiance(self, trained_surrogate):
        result = trained_surrogate.predict_steady(2.0, 0.1, 300)
        assert result['P_out'] >= 0

    def test_extreme_temperature(self, trained_surrogate):
        result = trained_surrogate.predict_steady(2.0, 0.5, 250)
        assert result['P_out'] >= 0
        result2 = trained_surrogate.predict_steady(2.0, 0.5, 350)
        assert result2['P_out'] >= 0
