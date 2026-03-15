# AlphaMaterials V6 — Quick Start Guide

**Version:** V6.0  
**Date:** 2026-03-15  
**Mission:** Generative Inverse Design + Techno-Economics for Perovskite Solar Cells

---

## 🚀 Installation

```bash
# Navigate to project directory
cd /root/.openclaw/workspace/tandem-pv

# Install dependencies (if not already installed)
pip install streamlit pandas numpy scipy scikit-learn xgboost plotly openpyxl joblib

# Run V6
streamlit run app_v6.py
```

---

## 📋 What's New in V6?

### 🆕 Five Major Features

1. **🧬 Generative Inverse Design (Tab 7)**
   - Specify target properties → AI generates candidates
   - "I want Eg=1.35±0.05 eV, stability>0.85, cost<$100/kg"
   - Two methods: Rejection sampling (fast) or Genetic algorithm (thorough)

2. **💰 Techno-Economic Analysis (Tab 8)**
   - Calculate $/Watt for any composition
   - Compare vs silicon baseline ($0.25/W)
   - Cost waterfall + sensitivity analysis
   - Supply chain risk scoring

3. **⚠️ Scale-Up Risk Assessment (Tab 9)**
   - Toxicity score (Pb content, RoHS compliance)
   - Supply chain risk (element availability)
   - Technology Readiness Level (TRL 1-9)
   - Regulatory compliance (RoHS, REACH)
   - Spider/radar chart visualization

4. **📄 Publication Export (Tab 10)**
   - LaTeX tables (booktabs format)
   - High-DPI figures (300 DPI PNG, SVG)
   - Auto-generated methods section
   - BibTeX references
   - One-click supplementary information package

5. **📊 Dashboard Summary (Tab 11)**
   - Campaign overview (key metrics, timeline)
   - Top discoveries from all optimization methods
   - HTML report export

### ✅ All V5 Features Preserved

- Database integration (Materials Project, AFLOW, JARVIS)
- User data upload (CSV/Excel)
- ML surrogate model (XGBoost + fine-tuning)
- Bayesian Optimization (EI, UCB, TS)
- Multi-objective Pareto optimization
- Experiment planner
- Session save/load

---

## 🎯 Complete Workflow

### Phase 1: Setup (Tabs 1-3)

**Tab 1: Load Database**
1. Click "🚀 Load Database"
2. Wait ~10 seconds (cached after first load)
3. ✅ 500+ materials loaded

**Tab 2: Upload Your Data**
1. Prepare CSV with columns: `formula`, `bandgap`
2. Upload file
3. Click "💾 Save to Session"

**Tab 3: Train ML Model**
1. Click "🚀 Train Model" → XGBoost on database
2. (Optional) Click "🔥 Fine-tune" → Personalize on your data
3. ✅ Model ready (MAE ~0.2-0.3 eV)

---

### Phase 2A: Forward Optimization (V5 workflow)

**Tab 4: Bayesian Optimization**
1. Click "🚀 Fit BO" → Gaussian Process on your data
2. Click "🔮 Suggest" → Get 5-20 next experiments
3. Click "➕ Add Top 5 to Queue"

**Tab 5: Multi-Objective**
1. Set weights: Bandgap match, Stability, Synthesizability, Cost
2. Click "🎯 Evaluate Multi-Objective"
3. View Pareto front (2D, 3D plots)
4. Get top recommendations

**Tab 6: Experiment Planner**
1. Review queued experiments
2. Export as CSV
3. Synthesize in lab!

---

### Phase 2B: Inverse Design (V6 NEW workflow)

**Tab 7: Inverse Design**
1. **Define constraints:**
   - Target bandgap: `1.35` eV ± `0.05` eV
   - Min stability: `0.85`
   - Max cost: `$100/kg`

2. **Generate candidates:**
   - Candidates to screen: `1000`
   - Method: `Rejection sampling` (fast) or `Genetic` (thorough)
   - Click "🚀 Generate Candidates"

3. **Review results:**
   - Top candidates table (ranked by feasibility + confidence)
   - 3D visualization (target region + candidates)
   - Click "➕ Add Top 10 to Experiment Queue"

**Example Output:**
```
✅ Found 47 valid candidates!

Top 5:
1. MA0.3FA0.7PbI2.85Br0.15  | Eg=1.36 eV | Stab=0.92 | Cost=$45/kg | Score=0.94
2. Cs0.05FA0.95PbI3         | Eg=1.34 eV | Stab=0.89 | Cost=$52/kg | Score=0.91
3. MAPb0.7Sn0.3I3           | Eg=1.37 eV | Stab=0.87 | Cost=$38/kg | Score=0.88
...
```

---

### Phase 3: Economic Analysis (V6 NEW)

**Tab 8: Techno-Economics**
1. **Select compositions:**
   - Source: "Inverse Design Candidates" (or Pareto, BO, Manual)
   - Top N: `5`

2. **Calculate economics:**
   - Click "💰 Calculate Economics"
   - View $/Watt table
   - Compare vs silicon baseline

3. **Detailed analysis:**
   - Cost waterfall chart (Material → Process → Total)
   - Tornado sensitivity (which cost drivers matter?)
   - Key insight: "Efficiency is 3× more important than material cost"

**Example Output:**
```
Cost Analysis Results:

Formula              | Efficiency | $/Watt | vs Silicon | Competitive?
---------------------|------------|--------|------------|-------------
MA0.3FA0.7PbI2.85Br0.15 | 0.23    | $0.22  | 0.88×      | ✅ Yes
Cs0.05FA0.95PbI3     | 0.24       | $0.20  | 0.80×      | ✅ Yes
MAPb0.7Sn0.3I3       | 0.19       | $0.28  | 1.12×      | ❌ No
```

---

### Phase 4: Risk Assessment (V6 NEW)

**Tab 9: Scale-Up Risk**
1. Enter composition: `MAPbI3`
2. Has experimental data? ✅ (optional)
3. Click "⚠️ Assess Risks"

4. **View risk scores:**
   - Toxicity: High (Pb-rich, NOT RoHS compliant)
   - Supply risk: Low (Pb is commodity)
   - TRL: 7/9 (well-established)
   - Regulatory risk: High (requires mitigation)

5. **Spider chart:** Visual 5-dimensional risk profile

**Actionable Insights:**
- "High Pb toxicity → Requires encapsulation + recycling protocol"
- "TRL 7 → Ready for pilot-scale manufacturing"

---

### Phase 5: Publication Export (V6 NEW)

**Tab 10: Publication Export**
1. **Select export options:**
   - ✅ LaTeX tables
   - ✅ CSV tables
   - ✅ High-DPI figures (300 DPI PNG, SVG)
   - ✅ Methods section
   - ✅ BibTeX references

2. Click "📤 Generate Export Package"

3. **Output files** (in `./exports/supplementary_information/`):
   - `SI_Table_S1_candidates.csv` — Inverse design candidates
   - `SI_Table_S2_pareto.csv` — Pareto-optimal materials
   - `SI_Table_S3_cost_analysis.csv` — Techno-economic comparison
   - `SI_Figure_S1_*.png` — All figures @ 300 DPI
   - `SI_Methods.txt` — Auto-generated methods section
   - `references.bib` — BibTeX citations
   - `README.md` — SI overview

4. **Use in paper:**
   - Copy LaTeX tables into manuscript
   - Insert figures
   - Paste methods section
   - Add BibTeX to references

---

### Phase 6: Dashboard & Save (V6 NEW + V5)

**Tab 11: Dashboard Summary**
1. View campaign metrics:
   - Materials screened: 523
   - Experiments done: 47
   - Candidates generated: 47
   - Pareto optimal: 12

2. Review timeline:
   - Database → Upload → Train → BO → Inverse → MO → Export ✅

3. Export HTML report:
   - Click "📄 Export Full Report (HTML)"
   - Download → Share with team

**Tab 12: Session Manager**
1. **Save progress:**
   - Session name: `Project_X_Week3`
   - Description: "Low-Pb compositions, targeting 1.35 eV"
   - Click "💾 Save Session"

2. **Resume later:**
   - Select session from dropdown
   - Click "📂 Load Session"
   - All data + models + results restored

---

## 💡 Tips & Best Practices

### Inverse Design

**When to use rejection sampling:**
- Loose constraints (tolerance ≥ 0.1 eV, min_stability < 0.8)
- Need fast results (1000 candidates in ~5 seconds)
- Exploratory phase

**When to use genetic algorithm:**
- Tight constraints (tolerance ≤ 0.05 eV, min_stability > 0.9)
- Willing to wait (~30 seconds)
- Optimization phase

**Constraint tuning:**
- If NO candidates found → Relax constraints
- If TOO MANY candidates → Tighten constraints
- Start loose, then narrow

### Techno-Economics

**Efficiency estimates:**
- If you don't have experimental efficiency, the system estimates from bandgap
- Peak efficiency at Eg ≈ 1.34 eV (Shockley-Queisser limit)
- Manual override recommended if you have real data

**Cost sensitivity:**
- Tornado chart shows which parameters matter
- If efficiency dominates → Focus on device optimization, not material cost reduction
- If encapsulation dominates → Negotiate bulk pricing or simplify process

### Risk Assessment

**Toxicity mitigation:**
- Pb-based ≠ automatically rejected
- Encapsulation + recycling can enable deployment
- Pb-free (Sn, Ge) still under development (lower TRL)

**Supply chain:**
- High-risk elements (Cs, Ge) → Have backup sources or substitutes
- Diversification strategy: "50% Cs, 50% Rb" spreads risk

### Publication Export

**LaTeX tables:**
- Compatible with most physics/chemistry journals
- Use `\usepackage{booktabs}` in your LaTeX preamble
- Compile with pdflatex

**Methods section:**
- Auto-generated text is a STARTING POINT
- Edit for journal-specific requirements
- Add experimental details (synthesis conditions, characterization)

**Figures:**
- 300 DPI PNG suitable for most journals
- SVG (vector) for presentations/posters
- Check journal figure size requirements

---

## ⚠️ Troubleshooting

### "No candidates found" (Inverse Design)

**Cause:** Constraints too tight

**Solutions:**
- Increase bandgap tolerance (0.05 → 0.10 eV)
- Lower min_stability (0.9 → 0.85)
- Increase max_cost ($50 → $100/kg)
- Try genetic algorithm (may find solutions in narrow regions)

### "Cost per watt is negative or >$10/W" (Techno-Economics)

**Cause:** Missing or invalid efficiency data

**Solutions:**
- Check efficiency input (must be 0-1, not 0-100)
- Ensure bandgap is reasonable (0.5-3.0 eV)
- Verify composition is valid (contains A, B, X sites)

### "Model not trained" errors

**Cause:** Skipped Tab 3 (ML training)

**Solution:**
- Go to Tab 3
- Click "🚀 Train Model"
- Wait for training to complete

### Session won't load

**Cause:** Version incompatibility (V5 session loaded in V6)

**Solution:**
- V6 sessions are forward-compatible with V5 core data
- May lose V6-specific data (inverse candidates, cost analysis)
- Re-run analyses in V6 tabs

---

## 📚 Key Concepts

### Inverse Design vs Forward Optimization

| | Forward (BO, V5) | Inverse (V6) |
|---|---|---|
| **Input** | Composition | Target properties |
| **Output** | Properties | Compositions |
| **Method** | Predict → Rank → Suggest | Constrain → Generate → Filter |
| **Use Case** | "What's the best next experiment?" | "What compositions meet my targets?" |

**Analogy:**
- Forward = "Test this recipe, see how it tastes"
- Inverse = "I want spicy, sour, <500 calories → What recipes?"

### Feasibility vs Confidence

**Feasibility score (0-1):**
- How well does candidate meet constraints?
- 1.0 = perfectly meets all targets
- 0.0 = violates all constraints

**Confidence (0-1):**
- How certain is the GP prediction?
- High confidence = low uncertainty (GP has seen similar compositions)
- Low confidence = high uncertainty (extrapolation)

**Combined score:**
- Weighted average: 40% feasibility + 60% confidence
- Balances "meets targets" with "prediction is reliable"

### TRL Scale (1-9)

| TRL | Stage | Description | Example |
|-----|-------|-------------|---------|
| 1-2 | Basic research | Concept, theory | New perovskite structure proposed |
| 3-4 | Proof of concept | Lab demo, small samples | First 1 cm² cell made |
| 5-6 | Technology development | Optimization, upscaling | 10×10 cm module, efficiency stable |
| 7-8 | System demo | Pilot production | 100 modules produced, field-tested |
| 9 | Commercial | Full-scale manufacturing | GW-scale factory, selling to market |

**V6 TRL heuristics:**
- MAPbI3, FAPbI3 → TRL 7 (widely studied, proven)
- Mixed cation/halide → TRL 5-6 (optimization ongoing)
- Pb-free Sn/Ge → TRL 4-5 (early development)
- Novel compositions → TRL 3-4 (proof-of-concept)

---

## 🎓 Learning Resources

### Understanding the Methods

**Bayesian Optimization:**
- Paper: Shahriari et al. (2016) "Taking the Human Out of the Loop"
- Tutorial: https://distill.pub/2020/bayesian-optimization/

**Gaussian Processes:**
- Book: Rasmussen & Williams (2006) "Gaussian Processes for Machine Learning"
- Free PDF: http://www.gaussianprocess.org/gpml/

**Multi-Objective Optimization:**
- Paper: Deb et al. (2002) "NSGA-II: A fast elitist multi-objective genetic algorithm"
- Tutorial: https://pymoo.org/

**Inverse Design:**
- Paper: Sanchez-Lengeling & Aspuru-Guzik (2018) "Inverse molecular design"
- Review: https://doi.org/10.1126/science.aat2663

### Perovskite Solar Cells

**Fundamentals:**
- Review: Stranks et al. (2015) "Electron-hole diffusion lengths in perovskites"
- Stability: Saliba et al. (2018) "Cesium-containing triple cation perovskites"

**Techno-Economics:**
- Analysis: Cai et al. (2017) "Cost-performance analysis of perovskite solar modules"
- Comparison: Woodhouse et al. (2019) "Levelized cost of energy for perovskites"

---

## 📞 Support & Contribution

### Reporting Issues

**GitHub Issues:** https://github.com/sjoonkwon0531/Tandem-PV-Simulator/issues

**Include:**
- V6 version (check footer: "V6.0")
- Error message (full traceback)
- Steps to reproduce
- Input data (CSV file, constraints used)

### Contributing

**Pull requests welcome!**

**Areas for contribution:**
- New acquisition functions (qEI, qKG)
- Better cost models (regional pricing, learning curves)
- VAE/GAN generative models
- Cloud deployment (Streamlit Cloud)
- Additional export formats (Excel, TIFF)

### Citation

**If you use AlphaMaterials V6 in your research:**

```bibtex
@software{alphamaterials_v6,
  title = {AlphaMaterials V6: Generative Inverse Design and Techno-Economic Analysis for Perovskite Solar Cells},
  author = {SAIT and SPMDL Collaboration},
  year = {2026},
  url = {https://github.com/sjoonkwon0531/Tandem-PV-Simulator}
}
```

---

## 🏁 Ready to Start?

```bash
streamlit run app_v6.py
```

**First-time workflow:**
1. Tab 1: Load database ✅
2. Tab 2: Upload your data (or skip if just exploring) ✅
3. Tab 3: Train model ✅
4. **Tab 7: Try inverse design!** 🧬
   - Target: 1.35 eV ± 0.05 eV
   - Min stability: 0.85
   - Max cost: $100/kg
   - Generate → Marvel at AI-generated candidates ✨

**Have fun discovering materials! 🚀**

---

빈 지도가 탐험의 시작 — The empty map is the start of exploration

*End of Quick Start Guide*
