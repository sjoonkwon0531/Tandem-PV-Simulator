# V11 CHANGELOG — AlphaMaterials: The Unified Platform

## Overview
V11 is the **final deployment version** — the Grand Unification of V3-V10.

## New Features

### 🔄 Unified Workflow Engine (`utils/workflow_engine.py`)
- One-click full pipeline: DB → Screen → Model → BO → Inverse → Protocol → Report
- Configurable steps (toggle on/off)
- Progress tracking with estimated time
- Pipeline summary export (JSON)

### 💡 Smart Recommendations (`utils/smart_recommendations.py`)
- Context-aware suggestions based on user activity
- Priority-ranked (high/medium/low)
- Contextual tips per tab
- Discovery progress radar chart

### 📊 Performance Monitor
- System health: CPU, memory, Python version
- Module status: all 25+ modules checked
- Data quality indicators
- Feature usage analytics

### 🎨 Theme & Accessibility (`utils/themes.py`)
- Dark / Light / High Contrast themes
- Colorblind-safe palettes (Wong 2011)
- Adjustable font size (12-20px)
- CSS generation for Streamlit

### 📜 About & Credits
- Version history (V3-V11 with dates)
- Technology stack documentation
- BibTeX citation format
- SPMDL / Sungkyunkwan University credits
- Philosophy: "빈 지도가 탐험의 시작"

## Architecture

### Tab Count: 36 total
- V4-V5: 7 tabs (core ML + optimization)
- V6: 5 tabs (inverse design + economics)
- V7: 5 tabs (digital twin + autonomous)
- V8: 4 tabs (model zoo + education)
- V9: 5 tabs (federated learning)
- V10: 5 tabs (NL query + protocols)
- V11: 5 tabs (workflow + recommendations + about)

### Utility Modules: 25
- V4: db_clients, data_parser
- V5: ml_models, bayesian_opt, multi_objective, session
- V6: inverse_design, techno_economics, export
- V7: digital_twin, auto_scheduler, transfer_learning, scenario_engine
- V8: model_zoo, api_generator, benchmarks, education
- V9: lab_simulator, federated, incentives
- V10: nl_query, report_generator, protocol_generator, knowledge_graph, decision_matrix
- V11: workflow_engine, smart_recommendations, themes

## Technical Constraints Met
- ✅ No PyTorch/TensorFlow
- ✅ sklearn, scipy, numpy, pandas, plotly, streamlit only
- ✅ < 500 MB RAM
- ✅ CPU only
- ✅ All previous versions preserved (V3-V10 .py files untouched)
