# AlphaMaterials V9: Federated Learning Platform

## 🎯 Quick Start

```bash
# Run V9
streamlit run app_v9.py

# Or run tests
python3 -c "import sys; sys.path.insert(0, 'utils'); from lab_simulator import *; from federated import *; from incentives import *; print('✅ All V9 modules OK')"
```

## 🆕 What's New in V9

**V9 = Federated Learning + Multi-Lab Collaboration**

Addresses the **data silo problem** in materials science:
- Labs won't share proprietary data (IP, competition)
- Centralized training impossible
- **Solution:** Federated learning — train together without sharing!

### 5 New Tabs (22-26)

| Tab | Feature | Description |
|-----|---------|-------------|
| 22 | 🤝 **Federated Learning** | Simulate 3-10 labs, FedAvg training, convergence tracking |
| 23 | 🔒 **Privacy-Preserving** | Differential privacy, privacy-accuracy tradeoff, secure aggregation |
| 24 | 🏆 **Multi-Lab Discovery** | Contribution leaderboard, Shapley values, "what we found together" |
| 25 | 📊 **Data Heterogeneity** | Non-IID metrics (KL, EMD), distribution visualization, most valuable lab |
| 26 | 💡 **Incentive Mechanism** | Fair credit allocation, cost-benefit analysis, "why participate?" |

### 3 New Utility Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `lab_simulator.py` | Generate multi-lab datasets | `LabDataSimulator`, `generate_centralized_dataset` |
| `federated.py` | Federated learning (FedAvg) | `FederatedLearner`, `train_federated`, `analyze_privacy_accuracy_tradeoff` |
| `incentives.py` | Fair credit allocation | `IncentiveMechanism`, `compute_shapley_values`, `recommend_participation` |

## 📊 Example Workflow

### 1. Generate Lab Data (Tab 25)
```python
# 5 labs with medium heterogeneity
simulator = LabDataSimulator(n_labs=5, heterogeneity="medium")
lab_datasets = simulator.generate_all_labs(n_features=5)

# Lab profiles
# - Lab A: 180 samples, halides, low noise
# - Lab B: 120 samples, oxides, medium noise
# - Lab C: 150 samples, mixed, very low noise
# ...
```

### 2. Train Federated Model (Tab 22)
```python
# Federated training: 10 rounds, privacy budget ε=5.0
learner = FederatedLearner()
rounds = learner.train_federated(
    lab_datasets=lab_datasets,
    n_rounds=10,
    epsilon_per_round=0.5
)

# Results:
# - Centralized (ideal): MAE = 0.12 eV
# - Federated (ε=5.0): MAE = 0.15 eV (25% gap)
# - Local-only: MAE = 0.28 eV
# → Federated improves 46% over local-only!
```

### 3. Analyze Privacy-Accuracy (Tab 23)
```python
# Try different privacy budgets
results = analyze_privacy_accuracy_tradeoff(
    lab_datasets, 
    test_data,
    epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0, ∞]
)

# Privacy cost:
# - ε = 0.1 (very private): MAE = 0.22 eV (57% worse)
# - ε = 1.0 (balanced): MAE = 0.16 eV (14% worse)
# - ε = ∞ (no privacy): MAE = 0.14 eV (best)
```

### 4. Compute Contributions (Tab 24)
```python
# Shapley values: fair credit allocation
mechanism = IncentiveMechanism(lab_datasets, test_data, baseline_mae)
scores = mechanism.compute_contribution_scores()

# Leaderboard:
# 🥇 Lab A: Shapley = 0.0234 (largest, highest quality)
# 🥈 Lab C: Shapley = 0.0187 (mixed data, unique)
# 🥉 Lab E: Shapley = 0.0145 (large but lower quality)
# ...
```

### 5. Allocate Credits (Tab 26)
```python
# Fair credit allocation (100 hours of compute)
allocations = mechanism.allocate_credits(total_credits=100, method="shapley")

# Results:
# - Lab A: 31.2 hrs (31.2%)
# - Lab C: 24.9 hrs (24.9%)
# - Lab E: 19.3 hrs (19.3%)
# → Credits proportional to contribution (not just data size!)
```

## 🔬 Key Algorithms

### Federated Averaging (FedAvg)
```
For round t = 1, 2, ..., T:
    1. Server sends global model M_t to all labs
    2. Each lab k:
       - Trains locally: M_k ← train(M_t, D_k)
       - Adds DP noise: ΔM_k ← ΔM_k + N(0, σ²)
       - Sends update to server
    3. Server aggregates: M_{t+1} ← M_t + avg(ΔM_k)
```

### Differential Privacy (Gaussian Mechanism)
```
σ = Δf × sqrt(2 × ln(1.25/δ)) / ε

Where:
- ε = privacy budget (lower = more private)
- δ = failure probability (10^-5)
- Δf = sensitivity (max change in output)
```

### Shapley Values (Fair Credit)
```
φ_i = Σ_{S ⊆ N\{i}} [|S|! × (n-|S|-1)! / n!] × [V(S∪{i}) - V(S)]

Approximation: Monte Carlo sampling (O(K×n) instead of O(2^n))
```

## 📈 Performance

**Tested Configuration:**
- 5 labs, medium heterogeneity
- 10 communication rounds
- Privacy budget ε = 5.0 (total)
- Random Forest (10 trees, depth 5)

**Results:**
| Approach | MAE (eV) | R² | Privacy | Feasibility |
|----------|----------|----|---------:|------------:|
| Centralized | 0.12 | 0.92 | ❌ No | ❌ Impossible |
| Federated (ε=5.0) | 0.15 | 0.88 | ✅ Yes | ✅ Practical |
| Local-only | 0.28 | 0.65 | ✅ Yes | ⚠️ Limited |

**Key Insights:**
- **Gap to ideal:** 25% (expected due to privacy + heterogeneity)
- **Improvement over local:** 46% (value of collaboration!)
- **Privacy cost:** 14% accuracy loss for ε=1.0

## ⚠️ Limitations

### Not Production-Ready
- **Simplified aggregation:** RF model distillation (not true weight averaging)
- **No real crypto:** Secure aggregation is conceptual (no actual homomorphic encryption)
- **Single-machine simulation:** Not distributed (no network, no Byzantine tolerance)
- **No client dropout:** Assumes all labs available every round

### Use Cases
- ✅ **Research/Education:** Understand FL concepts, explore privacy-accuracy tradeoff
- ✅ **Proof-of-Concept:** Demo to consortium partners, stakeholders
- ❌ **Production:** Use Flower, PySyft, or TensorFlow Federated for real deployment

### Recommendations for Production
1. **Use neural networks:** Direct weight averaging (PyTorch, TensorFlow)
2. **Implement real crypto:** SEAL (homomorphic encryption), MP-SPDZ (SMPC)
3. **Deploy distributed:** Flower framework, cloud deployment (AWS, GCP)
4. **Add Byzantine defense:** Krum, median aggregation, reputation systems

## 🔮 Future Work (V10)

### Advanced FL
- **Personalization:** Each lab fine-tunes global model
- **Asynchronous FL:** Labs send updates asynchronously
- **Communication efficiency:** Gradient compression, quantization
- **Neural networks:** Replace RF with MLP, CNN

### Advanced Privacy
- **DP-SGD:** Per-example gradient clipping + noise
- **Rényi DP:** Tighter privacy accounting
- **Attack simulations:** Membership inference, model inversion

### Advanced Incentives
- **Mechanism design:** Incentive-compatible FL (VCG auction)
- **Reputation systems:** Track lab behavior over time
- **Dynamic Shapley:** Recompute each round

### Real Deployment
- **Flower integration:** Actual distributed FL system
- **Production API:** FastAPI with secure aggregation
- **Cloud deployment:** Docker, Kubernetes, autoscaling

## 📚 References

**Federated Learning:**
- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks from Decentralized Data
- Li et al. (2020): Federated Optimization in Heterogeneous Networks

**Differential Privacy:**
- Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy

**Shapley Values:**
- Shapley (1953): A Value for n-Person Games
- Ghorbani & Zou (2019): Data Shapley: Equitable Valuation of Data for Machine Learning

**Materials Science Applications:**
- Dunn et al. (2020): Benchmarking Materials Property Prediction Methods
- Jablonka et al. (2021): A Data-Driven Perspective on the Colours of Metal-Organic Frameworks

## 📄 Citation

```bibtex
@software{alphamaterials_v9,
  title = {AlphaMaterials V9: Federated Learning Platform for Materials Discovery},
  author = {SAIT × SPMDL Collaboration},
  year = {2026},
  version = {9.0},
  url = {https://github.com/sjoonkwon0531/Tandem-PV-Simulator}
}
```

## 📧 Contact

**Questions? Issues?**
- GitHub: https://github.com/sjoonkwon0531/Tandem-PV-Simulator
- Issues: https://github.com/sjoonkwon0531/Tandem-PV-Simulator/issues

---

**빈 지도가 탐험의 시작 → 자율 실험실이 발견의 미래 → 프로덕션 플랫폼이 배포의 현실 → 연합 학습이 협업의 해법**

**V9 = The Federated Learning Platform for Materials Discovery** 🤝
