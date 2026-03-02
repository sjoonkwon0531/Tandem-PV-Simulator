#!/usr/bin/env python3
"""Tests for MaterialPredictor (8+ tests)."""

import numpy as np
import pytest

from engines.material_predictor import MaterialPredictor


@pytest.fixture(scope="module")
def predictor():
    p = MaterialPredictor()
    p.fit()
    return p


MAPBI3 = {'A_MA': 1.0, 'A_FA': 0.0, 'A_Cs': 0.0, 'A_Rb': 0.0,
           'B_Pb': 1.0, 'B_Sn': 0.0, 'B_Ge': 0.0,
           'X_I': 1.0, 'X_Br': 0.0, 'X_Cl': 0.0}


class TestFeaturization:
    def test_feature_shape(self):
        p = MaterialPredictor()
        f = p.featurize(MAPBI3)
        assert f.shape == (14,)

    def test_tolerance_factor_range(self):
        p = MaterialPredictor()
        f = p.featurize(MAPBI3)
        t = f[10]  # tolerance factor
        assert 0.7 < t < 1.1

    def test_octahedral_factor(self):
        p = MaterialPredictor()
        f = p.featurize(MAPBI3)
        mu = f[11]  # octahedral factor
        assert 0.3 < mu < 0.8

    def test_mixing_entropy_pure(self):
        """Pure composition should have zero mixing entropy."""
        p = MaterialPredictor()
        f = p.featurize(MAPBI3)
        assert abs(f[12]) < 1e-10  # no mixing


class TestMultiPropertyPrediction:
    def test_predict_all_properties(self, predictor):
        result = predictor.predict_multi(MAPBI3)
        for key in ['Eg', 'mu_e', 'mu_h', 'tau_ion', 'C_interface', 'stability']:
            assert key in result
            assert isinstance(result[key], float)

    def test_bandgap_consistency(self, predictor):
        """Eg from predict_multi should be close to ml_bandgap predictor."""
        result = predictor.predict_multi(MAPBI3)
        bg_direct, _ = predictor.bandgap_predictor.predict(MAPBI3)
        assert abs(result['Eg'] - bg_direct) < 0.01

    def test_property_ranges(self, predictor):
        result = predictor.predict_multi(MAPBI3)
        assert 0.5 < result['Eg'] < 4.0
        assert 1 <= result['mu_e'] <= 100
        assert 0.5 <= result['mu_h'] <= 50
        assert 0.1 <= result['tau_ion'] <= 200
        assert 5 <= result['C_interface'] <= 100
        assert 0 <= result['stability'] <= 10

    def test_confidence_present(self, predictor):
        result = predictor.predict_multi(MAPBI3)
        assert 'confidence' in result
        assert 'Eg' in result['confidence']


class TestScreening:
    def test_screening_returns_list(self, predictor):
        results = predictor.screen_for_dynamic_control(
            target_tau_range=(1, 100),
            stability_threshold=2.0,
            n_candidates=100,
        )
        assert isinstance(results, list)

    def test_screening_filters_correctly(self, predictor):
        results = predictor.screen_for_dynamic_control(
            target_tau_range=(5, 50),
            stability_threshold=3.0,
            Eg_range=(1.2, 1.7),
            n_candidates=200,
        )
        for r in results:
            props = r['properties']
            assert 5 <= props['tau_ion'] <= 50
            assert props['stability'] > 3.0
            assert 1.2 <= props['Eg'] <= 1.7

    def test_screening_sorted_by_stability(self, predictor):
        results = predictor.screen_for_dynamic_control(
            n_candidates=200,
            stability_threshold=2.0,
        )
        if len(results) >= 2:
            stabs = [r['properties']['stability'] for r in results]
            assert stabs == sorted(stabs, reverse=True)


class TestMixedCompositions:
    def test_mixed_composition(self, predictor):
        comp = {'A_MA': 0.5, 'A_FA': 0.5, 'A_Cs': 0.0, 'A_Rb': 0.0,
                'B_Pb': 0.8, 'B_Sn': 0.2, 'B_Ge': 0.0,
                'X_I': 0.7, 'X_Br': 0.3, 'X_Cl': 0.0}
        result = predictor.predict_multi(comp)
        assert result['Eg'] > 0
        assert result['stability'] >= 0
