# AlphaMaterials — AI-Driven Materials Discovery Platform

> **"빈 지도가 탐험의 시작"** — The empty map is the start of exploration.

## Overview

AlphaMaterials is a comprehensive, lightweight AI platform for accelerated materials discovery.
Built entirely on scikit-learn and scipy (no PyTorch), it runs on any laptop with < 500 MB RAM.

**Developed by:** SPMDL, Sungkyunkwan University
**PI:** Prof. S. Joon Kwon | [sjoonkwon.com](https://sjoonkwon.com)

## Quick Start

```bash
pip install streamlit pandas numpy plotly scikit-learn scipy joblib openpyxl requests psutil
streamlit run app_v11.py
```

## Versions

| Version | Tabs | Focus | File |
|---------|------|-------|------|
| V3 | 6 | SAIT Demo (Why AI?) | `app_v3_sait.py` |
| V4 | 5 | Database Integration | `app_v4.py` |
| V5 | 7 | Bayesian Optimization + Pareto | `app_v5.py` |
| V6 | 12 | Inverse Design + Economics | `app_v6.py` |
| V7 | 17 | Digital Twin + Autonomous Lab | `app_v7.py` |
| V8 | 22 | Model Zoo + Benchmarks | `app_v8.py` |
| V9 | 27 | Federated Learning | `app_v9.py` |
| V10 | 32 | NL Query + Protocols | `app_v10.py` |
| **V11** | **36** | **Unified Platform (Final)** | **`app_v11.py`** |

## Features (36 Tabs)

### Core Discovery
- 📂 Database Explorer (Materials Project, AFLOW, JARVIS)
- 📤 User Data Upload (CSV/Excel)
- 🧠 ML Surrogate Model (XGBoost/RandomForest)
- 🎯 Bayesian Optimization (EI, UCB, Thompson Sampling)
- 🏆 Multi-Objective Pareto Optimization

### Design & Analysis
- 🧬 Generative Inverse Design
- 💰 Techno-Economic Analysis ($/Watt)
- ⚠️ Scale-Up Risk Assessment
- 🏭 Digital Twin (process simulation ODEs)
- 🌍 What-If Scenario Engine

### Collaboration & Privacy
- 🤝 Federated Learning Simulator
- 🔒 Differential Privacy
- 💡 Incentive Mechanism (Shapley values)

### Intelligence
- 💬 Natural Language Query Engine
- 🧪 Synthesis Protocol Generator
- 📝 Research Report Auto-Generator
- 🕸️ Knowledge Graph Visualization
- ⚖️ Decision Matrix (TOPSIS/AHP)

### Platform
- 🏛️ Model Zoo with versioning
- 🌐 API Mode (OpenAPI spec)
- 🏅 Benchmark Suite
- 🎓 Educational Mode (tutorials, quiz)
- 🔄 Unified Workflow Engine
- 💡 Smart Recommendations

## Citation

```bibtex
@software{alphamaterials2026,
  title     = {AlphaMaterials: AI-Driven Materials Discovery Platform},
  author    = {Kwon, S. Joon and SPMDL},
  year      = {2026},
  url       = {https://github.com/sjoonkwon0531/Tandem-PV-Simulator},
  version   = {11.0},
  institution = {Sungkyunkwan University}
}
```

## License

For academic and research use. Contact Prof. Kwon for commercial licensing.
