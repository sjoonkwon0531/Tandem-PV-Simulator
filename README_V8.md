# AlphaMaterials V8: Foundation Model Hub + Deployment Platform

**The Complete Production Platform for Materials Discovery**

From data exploration to enterprise deployment — all in one place.

---

## 🚀 Quick Start

```bash
# Install dependencies (same as V7)
pip install streamlit pandas numpy plotly scikit-learn scipy xgboost joblib

# Run V8
streamlit run app_v8.py

# Run tests
python3 test_v8.py
```

Open browser at http://localhost:8501

---

## 🆕 What's New in V8

### 1. 🏛️ **Model Zoo / Foundation Model Hub**
- **Central registry** for all trained models
- **Model cards** with full metadata (training data, metrics, domain, version)
- **Version control** with changelog
- **Side-by-side comparison** (accuracy, speed, coverage)
- **Import/export** models (share with collaborators)

**Use case:** "Which model should I use? How does mine compare to the base model?"

### 2. 🌐 **API Mode**
- **OpenAPI 3.0 spec generation** (Swagger-compatible)
- **RESTful endpoints:** `/predict`, `/predict/batch`, `/models`, `/health`
- **Rate limiting** (simulated, in-memory)
- **Usage tracking** (requests, success rate, model usage)

**Use case:** "I want to deploy this as an API for my lab automation system."

### 3. 🏅 **Benchmark Suite**
- **Standard datasets:** Castelli Perovskites, JARVIS-DFT, Materials Project
- **Leaderboard:** Rank models by MAE, R², inference speed
- **Statistical tests:** Paired t-test, bootstrap CI, McNemar test
- **Reproducibility reports:** Full settings to reproduce results

**Use case:** "How does my model compare to state-of-the-art? Is the improvement statistically significant?"

### 4. 🎓 **Educational Mode**
- **Interactive tutorials:** Bandgap basics, Bayesian optimization, Pareto fronts
- **Glossary:** 15 technical terms explained simply
- **Quiz mode:** Test understanding (predict bandgaps, model reveals)
- **Explainability:** SHAP-like feature importance breakdown
- **Guided workflow:** 7-step discovery process

**Use case:** "I'm new to materials discovery. How do I learn the basics?"

### 5. 🚀 **Unified Landing Page**
- **Version evolution:** V3 → V8 feature comparison
- **Quick-start wizard:** "What do you want to do?" → Routes to right tab
- **System health dashboard:** Database, model, cache status
- **Recent activity feed:** Track recent actions
- **Feature matrix:** Which features in which version?

**Use case:** "What can this platform do? Where do I start?"

---

## 📊 Version Comparison

| Feature | V3 | V4 | V5 | V6 | V7 | V8 |
|---------|----|----|----|----|----|----|
| ML Surrogate | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Database Integration | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bayesian Optimization | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Multi-Objective | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Inverse Design | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Techno-Economics | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Digital Twin | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Autonomous Scheduler | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Transfer Learning | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Model Zoo** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |
| **API Generation** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |
| **Benchmarks** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |
| **Education** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |
| **Landing Page** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ NEW** |

---

## 🏗️ Architecture

### File Structure

```
tandem-pv/
├── app_v8.py                   # V8 main app (NEW)
├── V8_CHANGELOG.md             # V8 documentation (NEW)
├── test_v8.py                  # V8 tests (NEW)
├── utils/
│   ├── model_zoo.py            # V8: Model registry (NEW)
│   ├── api_generator.py        # V8: OpenAPI spec (NEW)
│   ├── benchmarks.py           # V8: Benchmark suite (NEW)
│   ├── education.py            # V8: Tutorials & quiz (NEW)
│   ├── digital_twin.py         # V7
│   ├── auto_scheduler.py       # V7
│   ├── transfer_learning.py    # V7
│   ├── scenario_engine.py      # V7
│   ├── inverse_design.py       # V6
│   ├── techno_economics.py     # V6
│   └── ... (V4-V5 modules)
├── models/registry/            # Model zoo storage
└── data/

```

### New Modules

#### `utils/model_zoo.py`
- `ModelRegistry`: Central registry for models
- `ModelCard`: Metadata container (dataclass)
- Functions: `register_model()`, `get_model()`, `compare_models()`, `export_model()`, `import_model()`

#### `utils/api_generator.py`
- `APISpecGenerator`: Generate OpenAPI 3.0 spec
- `RateLimiter`: In-memory rate limiting
- `UsageTracker`: Track API usage statistics

#### `utils/benchmarks.py`
- `StandardBenchmarks`: Pre-defined datasets (Castelli, JARVIS, MP)
- `BenchmarkSuite`: Run benchmarks, generate leaderboard
- `StatisticalTests`: Paired t-test, bootstrap CI, McNemar test
- `ReproducibilityReport`: Generate reproducibility documentation

#### `utils/education.py`
- `TutorialLibrary`: 3 interactive tutorials
- `Glossary`: 15 technical terms
- `QuizEngine`: Generate quizzes, explain predictions
- `GuidedWorkflow`: 7-step discovery process

---

## 📖 Usage Examples

### Model Zoo

```python
# Register a model
from utils.model_zoo import ModelRegistry

registry = ModelRegistry()

card = registry.register_model(
    model=trained_model,
    model_id='my-model-v1',
    name='My Perovskite Model',
    version='1.0.0',
    family='user-trained',
    training_data=df,
    features_used=['feat1', 'feat2'],
    target_property='bandgap',
    metrics={'mae': 0.18, 'r2': 0.82, 'rmse': 0.24, 'inference_speed_ms': 4.2},
    domain='halide_perovskites',
    author='User',
    description='My custom model'
)

# Compare models
comparison = registry.compare_models(['my-model-v1', 'halide-base-v1'])
print(comparison)

# Export model
registry.export_model('my-model-v1', './exports')
```

### API Generation

```python
# Generate OpenAPI spec
from utils.api_generator import APISpecGenerator

api_gen = APISpecGenerator(title="AlphaMaterials API", version="8.0.0")
spec = api_gen.generate_spec()

# Save spec
api_gen.export_json('openapi.json')

# Use spec to generate client
# openapi-generator generate -i openapi.json -g python
```

### Benchmarks

```python
# Run benchmark
from utils.benchmarks import BenchmarkSuite

suite = BenchmarkSuite()

result = suite.run_benchmark(
    model=my_model,
    featurizer=my_featurizer,
    benchmark_name='JARVIS-DFT',
    model_id='my-model-v1'
)

print(f"MAE: {result.mae:.4f}, R²: {result.r2:.4f}")

# Get leaderboard
leaderboard = suite.get_leaderboard(metric='mae')
print(leaderboard)
```

### Education

```python
# Get tutorial
from utils.education import TutorialLibrary

tutorial = TutorialLibrary.get_bandgap_tutorial()

for section in tutorial.sections:
    print(section['title'])
    print(section['content'])

# Search glossary
from utils.education import Glossary

matches = Glossary.search('bandgap')
for term, definition in matches:
    print(f"{term}: {definition}")
```

---

## 🎯 Workflows

### Workflow 1: Discover New Materials

1. **Load Data** (tab 1: Database)
   - Click "Load Database" → 500 materials loaded
2. **Train Model** (tab 3: ML Model)
   - Click "Train Model" → MAE: 0.15 eV, R²: 0.85
3. **Register Model** (tab 18: Model Zoo)
   - Model ID: `halide-base-v1`
   - Version: `1.0.0`
4. **Run Benchmark** (tab 20: Benchmarks)
   - Dataset: JARVIS-DFT
   - Result: MAE 0.18 eV (rank #3)
5. **Bayesian Optimization** (tab 5)
   - Target: 1.35 eV
   - Top candidates found
6. **Inverse Design** (tab 9)
   - Generate 100 candidates
7. **Analyze Costs** (tab 11: Techno-Economics)
   - Filter: cost <$0.30/W
8. **Decision** (tab 16: Dashboard)
   - Pick best from Pareto front

### Workflow 2: Deploy as API

1. **Train Model** (tab 3)
2. **Register Model** (tab 18)
   - Model ID: `production-v1`
3. **Generate API Spec** (tab 19: API Mode)
   - Download `openapi.json`
4. **Implement Server** (external)
   ```bash
   # Use FastAPI
   fastapi-code-generator --input openapi.json --output ./api_server
   cd api_server
   uvicorn main:app --reload
   ```
5. **Deploy** (external)
   - Docker: `docker build -t alphamaterials-api .`
   - Cloud: Deploy to AWS Lambda, GCP Cloud Run

### Workflow 3: Benchmark & Publish

1. **Train Model** (tab 3)
2. **Run All Benchmarks** (tab 20)
   - Castelli, JARVIS, Materials Project
3. **Statistical Tests**
   - Compare to baseline
   - Paired t-test: p=0.023 → Significant!
4. **Reproducibility Report**
   - Generate full report
   - Save for paper supplementary material
5. **Export Results** (tab 15: Publication Export)
   - LaTeX table, JSON data, figures
6. **Submit Paper** 📄

### Workflow 4: Learn & Explore (New User)

1. **Landing Page** (tab 0)
   - Read overview
   - Check system health
   - Choose wizard: "Learn the basics"
2. **Education** (tab 21)
   - Tutorial: "Understanding Bandgap" (10 min)
   - Quiz: 2/2 correct ✅
3. **Guided Workflow**
   - Follow 7 steps
   - Database → ML → BO → Decision
4. **First Discovery** 🎉
   - Found material with Eg=1.36 eV (error 0.01 eV!)

---

## 🔬 Example Use Cases

### Academic Research
- **Scenario:** PhD student exploring halide perovskites for tandems
- **Workflow:** Load MP data → Train model → Register in zoo → Run benchmarks → Compare to literature → Publish
- **Outcome:** Paper with reproducible results + shared model

### Industry R&D
- **Scenario:** Company developing low-cost solar cells
- **Workflow:** Load proprietary data → Train model → What-if scenarios (Pb ban) → TEA → Multi-objective (cost vs performance) → Pick candidate
- **Outcome:** Material recommendation for pilot line

### Lab Automation
- **Scenario:** Autonomous lab with robotic synthesis
- **Workflow:** Deploy API → Robotic system calls `/predict/batch` → Candidates ranked → Experiments scheduled → Results fed back
- **Outcome:** Closed-loop autonomous discovery

### Education
- **Scenario:** University course on materials informatics
- **Workflow:** Students complete tutorials → Take quizzes → Follow guided workflow → Discover material → Present findings
- **Outcome:** Hands-on learning with real tools

---

## ⚠️ Limitations

### Model Zoo
- **Local only** (not cloud-synced)
- **Manual export/import** for sharing
- For production: Deploy registry on shared server or use MLflow

### API Mode
- **Spec only** (no actual server)
- **In-memory rate limiting** (resets on restart)
- For production: Implement FastAPI server, use Redis for rate limiting

### Benchmarks
- **Simulated datasets** (realistic but not real)
- For publication: Download real datasets from sources

### Education
- **Text-based tutorials** (no videos)
- **Basic quizzes** (no adaptive difficulty)
- For advanced users: Add video tutorials, Jupyter notebooks

---

## 🔮 Roadmap (V9 Ideas)

- [ ] **Cloud Model Zoo:** Hugging Face Hub integration
- [ ] **Production API:** Full FastAPI server implementation
- [ ] **Real Benchmarks:** Download Castelli, JARVIS, MP via APIs
- [ ] **Video Tutorials:** Embed YouTube videos
- [ ] **Multi-tenancy:** Support multiple organizations
- [ ] **Advanced Explainability:** SHAP, LIME integration
- [ ] **Global Leaderboard:** Community-driven benchmark submissions

---

## 📄 Citation

```bibtex
@software{alphamaterials_v8,
  title={AlphaMaterials V8: Foundation Model Hub + Deployment Platform},
  author={SAIT × SPMDL Collaboration},
  year={2026},
  version={8.0.0},
  url={https://github.com/sjoonkwon0531/Tandem-PV-Simulator}
}
```

---

## 📞 Support

- **Documentation:** See `V8_CHANGELOG.md` for detailed documentation
- **Issues:** Report bugs on GitHub Issues
- **Questions:** Open GitHub Discussions

---

## 🏆 Credits

- **V3:** Core ML surrogate
- **V4:** Database integration
- **V5:** Bayesian optimization + multi-objective
- **V6:** Inverse design + techno-economics
- **V7:** Digital twin + autonomous scheduler + transfer learning
- **V8:** Model zoo + API + benchmarks + education + landing page

Built with ❤️ by the AlphaMaterials team.

**빈 지도가 탐험의 시작 → 프로덕션 플랫폼이 배포의 현실**

---

**Ready for enterprise deployment and public release! 🚀**
