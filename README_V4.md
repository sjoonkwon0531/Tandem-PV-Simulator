# AlphaMaterials V4: Connected Platform

**Transform your perovskite research with AI-driven design connected to global databases**

🔬 **SAIT × SPMDL Collaboration Platform**  
📅 **Version:** V4.0 — Connected Platform  
📆 **Date:** 2026-03-15

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_v4.txt

# 2. Run V4 app
streamlit run app_v4.py

# Or run original V3 demo
streamlit run app_v3_sait.py
```

**App opens at:** http://localhost:8501

---

## 🎯 What is V4?

**V4 = "Connected Platform"** — Evolution from V3 demo to real data-driven tool.

### V3 (Demo)
- 16 hardcoded compositions
- "Why AI?" demonstration
- Presentation-focused

### V4 (Connected)
- **1,500+ compositions** from global databases
- **User data upload** (CSV/Excel)
- **ML surrogate** for bandgap prediction
- **Property space mapping**
- Real research tool

**Philosophy preserved:** "빈 지도가 탐험의 시작" — The empty map is the start of exploration.

---

## ✨ Key Features

### 1️⃣ **Database Explorer** 🗄️
- Live connection to Materials Project, AFLOW, JARVIS-DFT
- SQLite cache for offline access
- Browse, filter, download thousands of perovskites
- Fallback to sample data if APIs unavailable

### 2️⃣ **Upload Your Data** 📤
- CSV/Excel support with auto-parsing
- Smart column detection
- Merge with global databases
- "Your data in context" visualization

### 3️⃣ **Property Space Mapping** 🗺️
- 18D composition features → 2D PCA projection
- Your materials highlighted as stars
- Identify unexplored regions
- Novelty analysis

### 4️⃣ **ML Surrogate** 🤖
- XGBoost bandgap predictor
- Train on database in ~1 second
- Uncertainty quantification
- Batch predictions

### 5️⃣ **Why AI?** 🔬
- V3 demo preserved
- Original "Why AI?" moment
- Link to full 6-tab demo

---

## 📁 File Structure

```
tandem-pv/
├── app_v3_sait.py              # V3 original (preserved)
├── app_v4.py                   # V4 main app ⭐
├── V4_CHANGELOG.md             # Detailed changelog
├── README_V4.md                # This file
├── requirements_v4.txt         # Dependencies
│
├── utils/                      # V4 modules
│   ├── db_clients.py           # API wrappers + cache
│   ├── data_parser.py          # CSV/Excel parser
│   └── ml_models.py            # XGBoost surrogate
│
├── data/
│   ├── cache.db                # SQLite cache (auto-generated)
│   └── sample_data/
│       └── perovskites_sample.csv  # Fallback dataset (59 compositions)
│
└── models/                     # Trained models (optional)
    └── bandgap_predictor.joblib
```

---

## 🔑 API Keys (Optional)

V4 works **without API keys** using bundled sample data.

For full database access:

### Materials Project
1. Register at [materialsproject.org](https://materialsproject.org)
2. Get free API key
3. In app sidebar: Settings → API Keys → Enter key

### AFLOW & JARVIS
- Public APIs, no key needed
- May have rate limits

---

## 📊 Usage Workflow

### Step 1: Load Database (Tab 1)
```
1. Click "Load Database"
2. Wait 5-10 seconds (first time)
3. Browse 1,500+ perovskites
4. Filter by bandgap, source
5. Download CSV
```

### Step 2: Upload Your Data (Tab 2)
```
1. Prepare CSV/Excel with columns:
   - formula (e.g., MAPbI3, FA0.87Cs0.13PbI3)
   - bandgap (eV)
   - Optional: voc, jsc, pce, stability, etc.

2. Upload file
3. Review parsed data
4. Click "Merge with Database"
```

**Example CSV:**
```csv
formula,bandgap,pce,notes
MAPbI3,1.59,21.3,Standard reference
FA0.87Cs0.13PbI3,1.55,23.1,Our champion cell
```

### Step 3: Property Space Map (Tab 3)
```
1. View PCA projection
2. Your data = stars ⭐
3. Database = dots •
4. Color = bandgap
5. Find unexplored regions (empty spaces)
```

### Step 4: Train ML Model (Tab 4)
```
1. Click "Train ML Model"
2. Wait ~1 second
3. Review metrics (MAE, R²)
4. Make predictions:
   - Single: Enter formula → Predict
   - Batch: Paste multiple formulas → Predict
```

### Step 5: Why AI? (Tab 5)
```
- Read V3 philosophy
- Access original demo
```

---

## 🎓 Tutorial: Your First Analysis

**Scenario:** You've measured bandgaps for 5 new compositions. Where do they fit in the global landscape?

```bash
# 1. Create your_data.csv
cat > your_data.csv << EOF
formula,bandgap,notes
Cs0.05FA0.95PbI3,1.53,Stable alpha phase
MA0.10FA0.90PbI3,1.54,Mixed cation
FAPb(I0.70Br0.30)3,1.63,Wide gap for tandem
Cs0.20Rb0.05FA0.75PbI3,1.56,Triple cation
FA0.90Cs0.10Pb(I0.85Br0.15)3,1.60,Optimized composition
EOF

# 2. Run app
streamlit run app_v4.py

# 3. In app:
#    - Tab 1: Load Database
#    - Tab 2: Upload your_data.csv
#    - Tab 3: See your stars on the map!
#    - Tab 4: Train model, predict variations
```

**Result:** You'll see exactly where your materials sit relative to 1,500+ known perovskites. Empty regions nearby = opportunities for next experiments.

---

## 🧪 Example: Bandgap Prediction

```python
# In app Tab 4, after training:

# Single prediction
Formula: FA0.87Cs0.13Pb(I0.62Br0.38)3
→ Predicted: 1.68 ± 0.15 eV

# Batch prediction
Formulas:
MAPbI3
FAPbI3
CsPbBr3
MA0.5FA0.5PbI3

→ Results downloadable as CSV
```

---

## ⚠️ Limitations (Honest Disclosure)

### Data Quality
- DFT has ~0.3-0.5 eV systematic error (GGA-PBE)
- ML adds ~0.2-0.3 eV error
- Total uncertainty: ~0.5-0.8 eV

### Model Scope
- ✅ Composition → Bandgap
- ❌ Device performance (Voc, Jsc, PCE)
- ❌ Stability/degradation
- ❌ Processing conditions

### Extrapolation Risk
- Model trained on available data
- Unknown chemistries = high uncertainty
- Sn oxidation not captured

### API Availability
- Materials Project: 100 requests/day (free)
- App works offline after first fetch
- Fallback to sample data if APIs fail

**Use V4 for screening, not final design. Always validate experimentally.**

---

## 🔧 Troubleshooting

### App won't start
```bash
# Check Python version (3.8+)
python --version

# Reinstall dependencies
pip install -r requirements_v4.txt --upgrade

# Clear Streamlit cache
streamlit cache clear
```

### Database won't load
- Check internet connection
- Verify API key (if using Materials Project)
- App will fallback to sample data automatically

### Upload fails
- Check CSV format (comma-separated, UTF-8)
- Required columns: `formula`, `bandgap`
- Column names are case-insensitive

### Model training fails
- Need ≥10 samples with valid bandgaps
- Remove rows with missing data
- Check formulas are parseable

---

## 📚 Resources

### Documentation
- `V4_CHANGELOG.md`: Detailed evolution from V3
- `app_v4.py`: Inline code comments
- `utils/*.py`: Module docstrings

### External Links
- [Materials Project](https://materialsproject.org): Database + API docs
- [AFLOW](http://aflowlib.duke.edu): DFT library
- [JARVIS-DFT](https://jarvis.nist.gov): NIST perovskite data
- [Streamlit](https://streamlit.io): Framework docs
- [XGBoost](https://xgboost.readthedocs.io): ML library

### Papers (Perovskite Background)
- Goldschmidt tolerance factor: *Z. Anorg. Allg. Chem.* 1926
- Halide segregation: Hoke et al., *Chem. Sci.* 2015
- All-perovskite tandems: recent reviews in *Nat. Energy*, *Joule*

---

## 🤝 Contributing

**Found a bug?** Open an issue.  
**Have data?** Share via upload feature.  
**Want a feature?** Suggest in roadmap discussion.

**Future ideas (V5?):**
- Multi-property prediction
- Active learning loop
- Cloud deployment
- Collaborative workspaces

---

## 📜 License & Citation

**Software:** MIT License

**Data:**
- Materials Project: CC BY 4.0
- AFLOW: Public domain
- JARVIS: NIST public data
- User uploads: User retains ownership

**Citation:**
```
AlphaMaterials V4: Connected Platform for AI-Driven Perovskite Design
SAIT × SPMDL Collaboration, 2026
https://github.com/[your-repo]/tandem-pv
```

---

## 🙏 Acknowledgements

- **SAIT** (Samsung Advanced Institute of Technology): Collaboration and funding
- **SPMDL Lab**: Material design expertise
- **Materials Project, AFLOW, JARVIS**: Open data infrastructure
- **V3 creators**: Original "Why AI?" philosophy

---

## 🎯 Philosophy

**빈 지도가 탐험의 시작**  
*The empty map is the start of exploration.*

V3 showed why AI matters.  
V4 gives you the tool to explore.

**From 16 → 1,500+ compositions.**  
**From demo → discovery.**

The journey continues.

---

**Questions?** Read `V4_CHANGELOG.md` for detailed technical docs.

**Ready to explore?** `streamlit run app_v4.py` 🚀

---

*AlphaMaterials V4 — Connected Platform*  
*2026-03-15*
