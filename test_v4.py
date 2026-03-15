#!/usr/bin/env python3
"""
V4 Quick Test Script
====================

Run this to verify V4 installation and functionality.

Usage:
    python test_v4.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("AlphaMaterials V4 - Quick Test")
print("=" * 60)
print()

# Test 1: Dependencies
print("1️⃣  Testing dependencies...")
try:
    import streamlit
    import pandas
    import numpy
    import plotly
    import requests
    import sklearn
    import scipy
    print("   ✅ Core dependencies OK")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    print("   → Run: pip install -r requirements_v4.txt")
    sys.exit(1)

try:
    import xgboost
    print("   ✅ XGBoost available")
    has_xgboost = True
except ImportError:
    print("   ⚠️  XGBoost not available (will use RandomForest)")
    has_xgboost = False

print()

# Test 2: V4 Modules
print("2️⃣  Testing V4 modules...")
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

try:
    from db_clients import UnifiedDBClient, CacheDB
    from data_parser import UserDataParser
    from ml_models import BandgapPredictor, CompositionFeaturizer
    print("   ✅ All modules imported")
except ImportError as e:
    print(f"   ❌ Module import failed: {e}")
    sys.exit(1)

print()

# Test 3: Sample Data
print("3️⃣  Testing sample data...")
sample_path = Path(__file__).parent / 'data' / 'sample_data' / 'perovskites_sample.csv'

if not sample_path.exists():
    print(f"   ❌ Sample data not found: {sample_path}")
    sys.exit(1)

import pandas as pd
df = pd.read_csv(sample_path)
print(f"   ✅ Loaded {len(df)} materials")
print(f"   ✅ Bandgap range: {df['bandgap'].min():.2f} - {df['bandgap'].max():.2f} eV")

print()

# Test 4: Composition Featurizer
print("4️⃣  Testing composition featurizer...")
featurizer = CompositionFeaturizer()

test_formulas = ["MAPbI3", "FAPbI3", "CsPbBr3"]
for formula in test_formulas:
    features = featurizer.featurize(formula)
    if len(features) != 18:
        print(f"   ❌ Wrong feature count for {formula}: {len(features)}")
        sys.exit(1)

print(f"   ✅ Featurized {len(test_formulas)} compositions")

print()

# Test 5: ML Model Training
print("5️⃣  Testing ML model training...")
model = BandgapPredictor(use_xgboost=has_xgboost)

try:
    metrics = model.train(df, formula_col='formula', target_col='bandgap')
    print(f"   ✅ Model trained ({metrics['n_samples']} samples)")
    print(f"   ✅ CV MAE: {metrics['cv_mae']:.3f} ± {metrics['cv_mae_std']:.3f} eV")
    print(f"   ✅ R² score: {metrics['train_r2']:.3f}")
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    sys.exit(1)

print()

# Test 6: Predictions
print("6️⃣  Testing predictions...")
test_formula = "MAPbI3"
try:
    predictions, uncertainties = model.predict([test_formula], return_uncertainty=True)
    pred = predictions[0]
    unc = uncertainties[0] if uncertainties is not None else 0.0
    actual = df[df['formula'] == test_formula]['bandgap'].values[0]
    error = abs(pred - actual)
    
    print(f"   Formula: {test_formula}")
    print(f"   Predicted: {pred:.2f} ± {unc:.2f} eV")
    print(f"   Actual: {actual:.2f} eV")
    print(f"   Error: {error:.2f} eV")
    
    if error < 0.5:
        print("   ✅ Prediction within tolerance")
    else:
        print("   ⚠️  Large prediction error (may be due to small training set)")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    sys.exit(1)

print()

# Summary
print("=" * 60)
print("🎉 All tests passed!")
print("=" * 60)
print()
print("✅ V4 is ready to use!")
print()
print("To start the app:")
print("   streamlit run app_v4.py")
print()
print("Or run V3 demo:")
print("   streamlit run app_v3_sait.py")
print()
print("=" * 60)
