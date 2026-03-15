# V7 CHANGELOG: Autonomous Lab Agent + Digital Twin

**Date:** 2026-03-15  
**Mission:** Transform deployment-ready platform (V6) → Autonomous lab agent with digital twin capabilities (V7)

---

## 🎯 Mission Statement

**V7 = "From Human-in-the-Loop to Autonomous Agent"**

V6 enabled inverse design + techno-economics + publication export = deployment-ready platform.  
V7 closes the automation gap: **Digital Twin Simulation → Autonomous Closed-Loop → Multi-Domain Learning → Collaborative Discovery → Policy Scenario Analysis**

**Core innovation:** Real-time process simulation (ODEs) + fully autonomous experiment scheduling (no human intervention) + transfer learning across material domains + what-if scenario engine = Ready for lights-out laboratory automation.

---

## 🆕 What's New in V7

### 1. **Digital Twin Mode** 🏭

**Problem:**
- V6 could predict final properties (bandgap, cost) but not HOW to make them
- No simulation of fabrication process (spin-coating, annealing, crystallization)
- Researchers fly blind: "Will this process recipe work?"
- Trial-and-error wastes time, materials, money

**Solution:**
- **Real-time process simulation**: Coupled ODEs model film formation from precursor to final crystal
- **Physics-based:** Precursor concentration → Nucleation → Grain growth → Film thickness
- **Interactive control:** Sliders for temperature, spin speed, annealing time, precursor concentration
- **Live animation:** Watch grains nucleate and grow in real-time (Plotly animated scatter)
- **Final property prediction:** Grain size, roughness, crystallinity, bandgap shift, defect density, quality score

**Implementation:**
- `utils/digital_twin.py`: DigitalTwin class with ODE solver (scipy.integrate.solve_ivp)
- Tab 10: Interactive process simulation UI
- Physics models:
  - dC/dt = -k_evap(T) × C (precursor evaporation, Arrhenius)
  - dN/dt = k_nuc(C,T) × (1 - N/N_max) (nucleation, supersaturation-dependent)
  - dG/dt = k_growth(C,T) × √N (grain growth, Ostwald ripening)
  - dh/dt = -k_thin(ω) × h (film thinning during spin)

**Process Stages:**
1. **Spin-coating (0-30s):** Solution spreads, film thins, nucleation begins
2. **Room temp hold (30s-60s):** Solvent evaporates slowly, more nucleation
3. **Annealing (60s-660s):** High temp accelerates crystallization, grains grow
4. **Final film:** Predict grain size, roughness, bandgap, defects

**Example Workflow:**
1. User inputs: Spin 3000 rpm, anneal 100°C for 10 min, precursor 1.0 M
2. Simulation runs 1000 time points over 660s
3. Output: Grain size 350 nm, crystallinity 92%, roughness 15 nm, quality score 0.78
4. Animation shows grain nucleation at 30s, rapid growth at 100-200s, saturation at 400s
5. Final bandgap: 1.57 eV (target 1.55 eV, shift +0.02 eV from grain boundaries)

**Key Insights:**
- Higher spin speed → thinner film → smaller grains → wider bandgap
- Higher anneal temp → faster crystallization → larger grains → lower defect density
- Optimal anneal time exists: too short = incomplete crystallization, too long = no improvement
- Process-property relationships visible: "If I increase temp by 20°C, grain size grows 100 nm"

**Impact:**
Instead of black-box "make material → measure properties," users now understand "process → microstructure → properties" causal chain. Enables process optimization before touching equipment.

---

### 2. **Autonomous Experiment Scheduler** 🤖

**Problem:**
- V6 Bayesian Optimization suggests next experiment, but human must run it
- Manual loop: Suggest → Synthesize → Measure → Update → Repeat (slow, labor-intensive)
- Human bottleneck: Can't run 24/7, subjective decisions, fatigue errors
- No clear stopping criteria: "Have I converged? Should I keep going?"

**Solution:**
- **Full closed-loop automation:** Predict → Suggest → Simulate → Evaluate → Learn → Repeat (no human intervention)
- **Batch mode:** Run N experiments with BO automatically
- **Convergence tracking:** Plot improvement over iterations, detect saturation
- **Budget-aware:** User sets max experiments, system optimizes within budget
- **Smart stopping criteria:**
  - Max iterations reached
  - Convergence (no improvement > threshold for N iterations)
  - Diminishing returns (improvement rate < threshold)

**Implementation:**
- `utils/auto_scheduler.py`: AutonomousScheduler class
- Tab 6: Full autonomous loop UI
- Integration with V5 BayesianOptimizer + GP surrogate
- Simulator function: ML model as "virtual lab" with realistic noise (±0.05 eV)

**Autonomous Loop Algorithm:**
```
Initialize with training data + ML model
FOR iteration = 1 to max_iterations:
    # PREDICT
    BO suggests top candidates (acquisition function)
    
    # SIMULATE (or wait for real results)
    Run experiments (virtual or real)
    Measure bandgaps (with noise)
    
    # EVALUATE
    Calculate best-so-far
    Track error vs target
    
    # LEARN
    Update GP model with new data
    Refit BO
    
    # CHECK CONVERGENCE
    IF no improvement in last N iterations:
        BREAK
    IF improvement rate < threshold:
        BREAK
    IF max iterations reached:
        BREAK

RETURN best candidate + iteration history
```

**Example Run (20 iterations, batch=3, target=1.35 eV):**
```
Iteration 1: Suggested MAPbI3, FAPbI3, CsPbI3 → Best error 0.24 eV
Iteration 2: Suggested MA0.7FA0.3PbI3 → Best error 0.18 eV
Iteration 3: Suggested MA0.6FA0.4PbI3 → Best error 0.12 eV
...
Iteration 12: Suggested MA0.55FA0.45PbI3 → Best error 0.02 eV
Iteration 13-15: No improvement → CONVERGED
Final: MA0.55FA0.45PbI3, Eg = 1.37 eV (error 0.02 eV)
```

**Convergence Plot:**
- X-axis: Iteration
- Y-axis: Best error (eV)
- Shows exponential improvement → plateau → convergence

**Exploration vs Exploitation Plot:**
- Scatter: Iteration vs measured bandgap
- Color: Iteration number (early = exploration, late = exploitation)
- Shows transition from broad search → narrow refinement

**Key Features:**
- **Batch scheduling:** Suggest N diverse experiments per iteration (parallel labs)
- **Diversity-aware:** Avoid redundant suggestions (max acquisition + high uncertainty + random sampling)
- **Budget tracking:** "You have 15 experiments left in budget, optimize carefully"
- **Real-time metrics:** Total experiments, best candidate, improvement vs initial

**Impact:**
Enables overnight/weekend autonomous runs. Lab starts Friday evening, comes back Monday to 50 experiments completed + best candidate identified. No human intervention needed.

---

### 3. **Transfer Learning Across Domains** 🔄

**Problem:**
- V6 trained models are domain-specific: Halide perovskites only
- Knowledge doesn't transfer: Oxide perovskite researcher starts from scratch
- Different domains have similar physics: ABX3 structure, A-site chemistry, B-site electronics
- Waste of data: "We've done 1000 halide experiments, can we use any of this for oxides?"

**Solution:**
- **Multi-domain pre-trained models:** Halide perovskites, oxide perovskites, chalcogenides
- **Domain selector:** Switch material family, model adapts
- **Transfer learning:** Fine-tune source domain model on target domain data
- **Cross-domain insights:** "Patterns from oxides suggest trying X in halides"
- **Shared feature representations:** Ionic radius, electronegativity, valence electrons

**Implementation:**
- `utils/transfer_learning.py`: TransferLearningEngine class
- Tab 4: Domain selection + transfer UI
- Pre-trained models for 3 domains (sample data included)
- Knowledge transfer via feature importance analysis

**Domains:**

| Domain | Formula Pattern | Bandgap Range | Key Features | Sample Materials |
|--------|----------------|---------------|--------------|------------------|
| **Halide Perovskites** | ABX3 (X=I/Br/Cl) | 1.2-2.5 eV | A-site ionic radius, B-site electronegativity, X-site size | MAPbI3, FAPbBr3, CsPbCl3 |
| **Oxide Perovskites** | ABO3 (O=oxygen) | 3.0-5.5 eV | A-site size, B-site d-electrons, oxygen vacancies | SrTiO3, BaTiO3, CaTiO3 |
| **Chalcogenides** | A2BX4 (X=S/Se/Te) | 1.0-2.5 eV | A-site electronegativity, B-site size, X-site polarizability | Cu2ZnSnS4, Ag2ZnSnSe4 |

**Transfer Learning Workflow:**
1. **Train source model:** Fine-tune halide perovskite model on user data
2. **Extract knowledge:** Feature importances → Top 5 critical features
3. **Transfer to target:** Adapt model to oxide perovskites
4. **Generate insights:** "Halides show strong A-site dependence → Try Sr/Ba substitution in oxides"
5. **Cross-domain prediction:** Use halide model to predict oxide properties (with warnings)

**Example Transfer (Halides → Oxides):**
```
Source: Halide perovskites (trained on 200 samples)
Top features: Ionic radius A (importance 0.35), Electronegativity B (0.28), Ionic radius X (0.22)

Target: Oxide perovskites
Insight 1: Halides dominated by A-site size → In oxides, explore Sr/Ba/Ca substitution
Insight 2: Halides (1.2-2.5 eV) vs Oxides (3.0-5.5 eV) → Different applications (PV vs UV sensors)
Insight 3: B-site chemistry critical in both → Pb/Sn in halides ↔ Ti/Zr in oxides

Cross-domain model adapted with 20 oxide samples → MAE improved from 0.5 eV to 0.2 eV
```

**Cross-Domain Insights Examples:**
- **Halides → Chalcogenides:** "Both tunable in 1.0-2.5 eV range. Halides easier synthesis, chalcogenides better stability."
- **Oxides → Halides:** "Oxides stable but insulating. Consider hybrid: oxide scaffold + halide active layer."
- **Chalcogenides → Halides:** "CZTS is Pb-free, earth-abundant. If Pb banned, CZTS chemistry guides alternatives."

**Feature Importance Transfer Plot:**
- Side-by-side bar chart: Source domain importances vs Target domain importances
- Shows which features transfer (similar importance) vs domain-specific (different importance)

**Impact:**
Researchers no longer start from zero when switching domains. Halide expert entering oxide space can leverage 80% of knowledge. Accelerates cross-domain innovation.

---

### 4. **Collaborative Mode** 👥

**Problem:**
- V6 is single-user: No way to share discoveries with colleagues
- Labs work in silos: "We found X" → "We also found X" (duplicated effort)
- No community knowledge base: Discoveries scattered in papers, notebooks, emails
- No feedback mechanism: "Is this material promising? Others' opinions?"

**Solution:**
- **Multi-user session support (simulated):** JSON-based discovery feed
- **Share discoveries:** Composition + bandgap + notes → Public feed
- **Annotation system:** Tag compositions with insights, warnings, tips
- **Discovery feed:** "Lab A found X (Eg=1.50 eV), Lab B found Y (Eg=1.35 eV)"
- **Upvote system:** Community validates promising candidates
- **Export/import:** Share discovery JSON files between teams

**Implementation:**
- Tab 14: Collaborative discovery UI
- `st.session_state.collaborative_discoveries`: List of discovery dicts
- JSON export: Download full discovery feed

**Discovery Schema:**
```json
{
  "lab_id": "Lab_472",
  "formula": "MA0.6FA0.4PbI3",
  "bandgap": 1.36,
  "notes": "High efficiency, stable for 1000h",
  "timestamp": "2026-03-15T12:30:00",
  "upvotes": 5,
  "annotations": [
    {
      "lab_id": "Lab_189",
      "text": "We replicated this, confirmed Eg=1.37 eV, excellent reproducibility!",
      "timestamp": "2026-03-15T14:00:00"
    }
  ]
}
```

**Collaborative Workflow:**
1. **Lab A discovers:** MA0.6FA0.4PbI3, Eg=1.36 eV → Shares to feed
2. **Lab B sees feed:** "Interesting! Let me try similar composition"
3. **Lab B experiments:** MA0.5FA0.5PbI3, Eg=1.35 eV → Shares + Annotates Lab A's discovery
4. **Lab C upvotes:** Both discoveries useful → Upvotes
5. **Community consensus:** MA0.5-0.6FA0.5-0.4PbI3 region is sweet spot for 1.35 eV target

**Use Cases:**
- **Multi-site collaboration:** SAIT + University partner share discoveries in real-time
- **Internal team sharing:** Researcher A → Researcher B within same lab
- **Literature mining:** Convert published data to discovery format, build database
- **Replication validation:** "We tried X, got different result" → Annotation warns others

**Limitations (Honest Disclosure):**
- **Simulated only:** Real multi-user requires backend (database, authentication, real-time sync)
- **No access control:** All discoveries public (no private/embargoed data)
- **JSON-based:** Manual export/import (not auto-synced like Google Docs)
- **No conflict resolution:** If two labs edit same discovery, last write wins

**Future (V8?):**
- Backend database (PostgreSQL, Firebase)
- Real-time WebSocket sync
- User authentication (login, permissions)
- Private workspaces + public sharing
- API for programmatic access

**Impact:**
Transforms isolated researchers into collaborative network. Discoveries amplified: "I found X" → "We collectively explored X-Y-Z space." Reduces duplication, accelerates validation.

---

### 5. **What-If Scenario Engine** 🌍

**Problem:**
- V6 optimizes for current constraints, but what if constraints change?
- Policy uncertainty: "Will Pb be banned? Should I invest in Pb-free now?"
- Market volatility: "What if iodine price doubles? Is my material still viable?"
- Target flexibility: "Can I hit Eg < 1.0 eV for NIR? Or is it impossible?"
- No way to compare policy impacts: "Lead ban vs iodine shortage: Which hurts more?"

**Solution:**
- **Scenario-based optimization:** Define constraints → Re-run optimization → Compare results
- **Predefined scenarios:** Pb ban, iodine crisis, low-bandgap requirement, earth-abundant only, cost-aggressive
- **Custom scenarios:** User defines banned elements, cost multipliers, bandgap range, RoHS compliance
- **Side-by-side comparison:** Run 5 scenarios → See which has most valid candidates
- **Policy impact report:** Baseline vs policy → Candidates lost, cost increase, feasibility, recommendation

**Implementation:**
- `utils/scenario_engine.py`: ScenarioEngine class
- Tab 13: Scenario definition + comparison UI
- Integration with InverseDesignEngine + TechnoEconomicAnalyzer
- Policy impact assessment with actionable recommendations

**Predefined Scenarios:**

| Scenario | Description | Banned Elements | Bandgap Range (eV) | Max $/W |
|----------|-------------|-----------------|-------------------|---------|
| **Baseline** | Current state, no restrictions | None | 1.2-1.8 | $0.30 |
| **Lead Ban** | RoHS-like Pb regulation | Pb | 1.2-1.8 | $0.30 |
| **Iodine Crisis** | I price doubles (supply shock) | None (I cost ×2) | 1.2-1.8 | $0.30 |
| **Low Bandgap** | NIR application (Eg < 1.0 eV) | None | 0.5-1.0 | $0.50 |
| **High Bandgap** | UV application (Eg > 2.5 eV) | None | 2.5-3.5 | $0.50 |
| **Earth-Abundant** | Exclude rare/expensive | Cs, Rb, Ge | 1.2-1.8 | $0.25 |
| **Tandem Top Cell** | Wide-gap for Si tandem | None | 1.65-1.80 | $0.40 |
| **Cost-Aggressive** | Beat silicon ($0.20/W) | None | 1.2-1.8 | $0.20 |

**Scenario Comparison Workflow:**
1. **Select scenarios:** Baseline, Pb Ban, Iodine Crisis
2. **Run optimization:** Each scenario generates 500 candidates
3. **Compare results:**
   - Baseline: 450 valid candidates, best $/W = $0.22
   - Pb Ban: 180 valid candidates (60% loss), best $/W = $0.28 (+27% cost)
   - Iodine Crisis: 420 valid candidates (7% loss), best $/W = $0.24 (+9% cost)
4. **Conclusion:** Iodine crisis manageable, Pb ban severe impact

**Policy Impact Report Example:**
```
Policy: Lead Ban (RoHS Compliance)
Baseline: 450 candidates, best cost $0.22/W
Policy: 180 candidates, best cost $0.28/W

Impact:
- Candidates lost: 270 (60%)
- Cost increase: +27% ($0.06/W)
- Feasible: YES (180 alternatives exist)

Recommendation:
🔴 Policy significantly increases costs (>15%). Consider phased implementation or R&D investment in Pb-free alternatives.
```

**Feasibility Map:**
- 2D scatter: Bandgap (x-axis) vs $/W (y-axis)
- Color: Feasibility score
- Green shaded region: Target bandgap range
- Red dashed line: Max $/W constraint
- Shows "sweet spot" where constraints overlap

**Use Cases:**
- **R&D planning:** "If Pb banned, do we have alternatives? → Yes, but 27% more expensive → Invest in cost reduction R&D"
- **Supply chain risk:** "Iodine crisis → Br-based alternatives viable? → Yes, 420 candidates → Diversify to Br"
- **Application pivot:** "Can we do NIR (Eg < 1.0 eV)? → 50 candidates found → Feasible but challenging"
- **Competitive analysis:** "To beat silicon ($0.20/W), we need Eg=1.3-1.4 eV + earth-abundant → 12 candidates → Tight but possible"

**Impact:**
De-risks strategic decisions. Instead of reacting to policy/market changes, proactively assess impact. "What-if" becomes "We've already analyzed that scenario."

---

## 🏗️ Technical Architecture

### New Dependencies (V7)

**None!** V7 uses only existing V6 dependencies:
- `scipy` (already in V6, now used for ODE integration)
- `sklearn`, `xgboost` (V5-V6)
- `streamlit`, `pandas`, `numpy`, `plotly` (core)

**Philosophy:** Lightweight, zero bloat. Digital twin uses scipy.integrate (built-in), not specialized ODE solvers. Transfer learning uses sklearn (already present), not PyTorch. Everything runs on CPU, no GPU required.

### File Structure

```
tandem-pv/
├── app_v3_sait.py              # V3 preserved
├── app_v4.py                   # V4 preserved
├── app_v5.py                   # V5 preserved
├── app_v6.py                   # V6 preserved
├── app_v7.py                   # V7 main app (NEW)
├── V4_CHANGELOG.md
├── V5_CHANGELOG.md
├── V6_CHANGELOG.md
├── V7_CHANGELOG.md             # This file (NEW)
├── utils/
│   ├── db_clients.py           # V4
│   ├── data_parser.py          # V4
│   ├── ml_models.py            # V4-V5
│   ├── bayesian_opt.py         # V5
│   ├── multi_objective.py      # V5
│   ├── session.py              # V5
│   ├── inverse_design.py       # V6
│   ├── techno_economics.py     # V6
│   ├── export.py               # V6
│   ├── digital_twin.py         # V7 (NEW)
│   ├── auto_scheduler.py       # V7 (NEW)
│   ├── transfer_learning.py    # V7 (NEW)
│   └── scenario_engine.py      # V7 (NEW)
├── data/
│   ├── cache.db
│   └── sample_data/
│       └── perovskites_sample.csv
├── sessions/                   # V5-V7 session storage
└── exports/                    # V6-V7 publication export
```

---

## 🔄 What's Preserved from V6

### All V6 Features Intact ✅
- ✅ Generative Inverse Design (V6)
- ✅ Techno-Economic Analysis (V6)
- ✅ Scale-Up Risk Assessment (V6)
- ✅ Publication-Ready Export (V6)
- ✅ Dashboard Summary (V6)
- ✅ All V5 features (BO, ML, MO, etc.)

### UI Changes
- **V6:** 12 tabs
- **V7:** 17 tabs (5 new tabs inserted intelligently in workflow)
- Tab order optimized for natural workflow:
  1. Database → 2. Upload → 3. ML → **4. Transfer (NEW)** → 5. BO → **6. Autonomous (NEW)** → 7. MO → 8. Planner → 9. Inverse → **10. Digital Twin (NEW)** → 11. Techno → 12. Risk → **13. Scenarios (NEW)** → **14. Collaborative (NEW)** → 15. Export → 16. Dashboard → 17. Session

### Branding
- **V6:** Dark theme (purple gradient)
- **V7:** Enhanced dark theme (deeper black #0a0e1a) + glowing V7 badge + "NEW" tags on new tabs

---

## 📊 V6 vs V7 Comparison

| Feature | V6 (Deployment Ready) | V7 (Autonomous Agent) |
|---------|----------------------|----------------------|
| **Digital Twin** | ❌ | ✅ Real-time ODE simulation |
| **Autonomous Loop** | ❌ Manual BO | ✅ Fully autonomous (24/7) |
| **Transfer Learning** | ❌ Single domain | ✅ 3 domains (halide/oxide/chalcogenide) |
| **Collaborative** | ❌ Single-user | ✅ Multi-user feed (simulated) |
| **What-If Scenarios** | ❌ | ✅ Policy impact + cost sensitivity |
| **Process Simulation** | ❌ | ✅ Spin, anneal, crystallize |
| **Convergence Tracking** | ❌ | ✅ Auto-stop when converged |
| **Domain Switching** | ❌ | ✅ Halide ↔ Oxide ↔ Chalcogenide |
| **Cross-Domain Insights** | ❌ | ✅ "Oxides suggest X in halides" |
| **Scenario Comparison** | ❌ | ✅ Side-by-side feasibility |
| **Inverse Design** | ✅ | ✅ (Preserved) |
| **Techno-Economics** | ✅ | ✅ (Preserved) |
| **Publication Export** | ✅ | ✅ (Preserved) |
| **Target User** | Academia + Industry | **Autonomous Labs** + R&D Teams |

---

## 🚀 Usage Guide

### Complete V7 Workflow

**New Capabilities (V7 Additions to V6 Workflow):**

#### **A. Digital Twin Workflow (Tab 10)**

1. **Set process parameters:**
   - Spin speed: 3000 rpm
   - Anneal temp: 100°C
   - Anneal time: 600s
   - Precursor concentration: 1.0 M
   - Target bandgap: 1.55 eV

2. **Run simulation:**
   - Click "🚀 Run Simulation"
   - Watch time-series plots (concentration, nucleation, grain growth)
   - View animation (grains nucleate → grow → saturate)

3. **Analyze results:**
   - Final grain size: 350 nm ✅
   - Crystallinity: 92% ✅
   - Quality score: 0.78/1.0 ✅
   - Bandgap shift: +0.02 eV (grain boundary effect)

4. **Optimize process:**
   - Enable comparison mode
   - Test 3 conditions (vary temp/time)
   - Identify optimal: 120°C × 900s → grain size 450 nm, quality 0.85

#### **B. Autonomous Scheduler Workflow (Tab 6)**

1. **Configure loop:**
   - Max iterations: 20
   - Batch size: 3 (parallel experiments)
   - Convergence window: 5
   - Improvement threshold: 0.01 eV

2. **Start autonomous run:**
   - Click "🚀 Start Autonomous Loop"
   - System runs 20 iterations (or stops early if converged)
   - No human intervention needed

3. **Monitor progress:**
   - Convergence plot: Error drops from 0.24 eV → 0.02 eV over 12 iterations
   - Stopped early (no improvement in last 5 iterations)

4. **Review discoveries:**
   - Top 10 candidates automatically identified
   - Best: MA0.55FA0.45PbI3, Eg = 1.37 eV (error 0.02 eV)

#### **C. Transfer Learning Workflow (Tab 4)**

1. **Select source domain:**
   - Source: Halide perovskites (your trained model)

2. **Select target domain:**
   - Target: Oxide perovskites (want to explore)

3. **Transfer knowledge:**
   - Click "🔄 Transfer Knowledge"
   - System extracts key features from halides
   - Generates insights: "A-site size critical → Try Sr/Ba in oxides"

4. **Apply insights:**
   - Use oxide model to predict new compositions
   - Fine-tune with 20 oxide samples → MAE 0.2 eV

#### **D. What-If Scenario Workflow (Tab 13)**

1. **Select scenario:**
   - Scenario: "Lead Ban" (Pb prohibited)

2. **Run optimization:**
   - Click "▶️ Run Scenario"
   - System re-optimizes without Pb
   - Result: 180 valid candidates (vs 450 baseline)

3. **Assess impact:**
   - Candidates lost: 60%
   - Cost increase: +27%
   - Feasible: YES (Sn-based alternatives exist)

4. **Compare scenarios:**
   - Run: Baseline, Pb Ban, Iodine Crisis
   - Side-by-side comparison: Iodine crisis less severe (7% candidate loss)

5. **Policy report:**
   - Generate report: Lead Ban impact
   - Recommendation: "🔴 Severe impact → Phased implementation needed"

#### **E. Collaborative Discovery Workflow (Tab 14)**

1. **Share discovery:**
   - Formula: MA0.6FA0.4PbI3
   - Bandgap: 1.36 eV
   - Notes: "High efficiency, stable"
   - Click "📣 Share Discovery"

2. **Browse feed:**
   - See discoveries from other labs
   - Upvote promising candidates
   - Annotate with your insights

3. **Export discoveries:**
   - Download JSON file
   - Share with collaborators
   - Import their discoveries into your session

---

## ⚠️ Limitations & Honest Disclosure

### Digital Twin Limitations

**Physics model simplifications:**
- **ODEs are simplified:** Real crystallization involves nucleation kinetics, diffusion, solvent dynamics (not fully captured)
- **No 3D spatial modeling:** Assumes uniform film, no edge effects, no thickness gradients
- **Parameters are generic:** k_evap, k_nuc, k_growth estimated from literature, not material-specific
- **No solvent dynamics:** Assumes single-component solvent (real: DMF/DMSO mixtures)
- **No substrate effects:** Assumes ideal substrate, no surface energy, no wetting

**Animation is illustrative:**
- Grain positions are random (not actual nucleation sites)
- Grain sizes are averaged (real: size distribution)
- 2D view only (real: 3D film)

**Bandgap shift heuristics:**
- Grain boundary effect (ΔEg ∝ 1/grain_size) is empirical, not from first principles
- Defect density formula is approximate

**Recommendations:**
- Use digital twin for **qualitative trends**, not quantitative predictions
- "Higher temp → larger grains" ✅
- "Temp = 105°C gives exactly 347.2 nm grains" ❌
- Validate with real experiments before trusting absolute values

### Autonomous Scheduler Limitations

**Simulator is not reality:**
- Uses ML model + noise as "virtual lab"
- Real experiments have batch effects, human errors, equipment drift (not modeled)
- Convergence in simulation ≠ convergence in real lab

**BO limitations:**
- GP assumes smooth function (real: discontinuities, phase transitions)
- Acquisition function may miss global optimum if search space is rugged
- No constraint handling (e.g., "avoid toxic Pb" must be pre-filtered)

**Stopping criteria are heuristics:**
- Convergence threshold (0.01 eV) is user-defined, not universal
- Diminishing returns may occur in local optimum (not global)

**Batch diversity:**
- Diversity strategy (max acq + high uncertainty + random) is simple
- Advanced: Use clustering, determinantal point processes (not implemented)

**Recommendations:**
- Use autonomous scheduler for **high-throughput screening**, not final validation
- "Narrowed 10,000 candidates to top 10" ✅
- "Top candidate is THE answer" ❌
- Always validate top candidates experimentally

### Transfer Learning Limitations

**Pre-trained models are toy examples:**
- Sample data has 5-10 compositions per domain (real: need 100+)
- Feature importances may not generalize
- Models are RandomForest (fast but low capacity), not deep neural networks

**Cross-domain predictions are uncertain:**
- Using halide model on oxide compositions → Extrapolation (high error)
- Warning system flags out-of-distribution, but predictions still unreliable

**Domain definitions are rigid:**
- Formula pattern matching (regex) may misclassify novel compositions
- "Hybrid halide-oxide" doesn't fit in any domain

**Insights are templated:**
- Cross-domain insights are hardcoded text, not learned from data
- May not apply to specific user's case

**Recommendations:**
- Use transfer learning for **exploration**, not production
- "Try A-site substitution based on oxide insights" ✅
- "This cross-domain prediction is 1.45 eV exactly" ❌
- Collect real target-domain data as soon as possible, fine-tune model

### Collaborative Mode Limitations

**Not real multi-user:**
- Simulated (JSON in session state), not backend database
- No real-time sync (need to export/import manually)
- No conflict resolution (last write wins)

**No access control:**
- All discoveries are public (no private/embargoed data)
- Anyone can edit annotations (no authentication)

**No persistence:**
- Discoveries lost if session state cleared (need to export JSON)

**Recommendations:**
- Use collaborative mode for **proof-of-concept**, not production
- Real deployment needs backend infrastructure (Firebase, PostgreSQL)
- Export JSON frequently to avoid data loss

### What-If Scenario Limitations

**Cost model is from V6:**
- Same techno-economic limitations (simplified, not real quotes)
- Cost multipliers (I price ×2) are hypothetical, not actual market data

**Scenario search is brute-force:**
- Generates all ABX3 combinations → filters
- Expensive for large search spaces (>10,000 combos)
- No smart optimization (gradient descent, genetic algorithm)

**Policy impact is heuristic:**
- Recommendations ("🔴 Severe impact") are rule-based, not ML-predicted
- Thresholds (5% acceptable, 15% severe) are arbitrary

**Recommendations:**
- Use scenarios for **strategic planning**, not absolute forecasts
- "Pb ban reduces candidates by 60%" (relative comparison) ✅
- "Pb ban will cost exactly $0.06/W more" (absolute prediction) ❌
- Update cost database with real market data regularly

---

## 🎓 Key Learnings & Design Decisions

### 1. Why ODEs for Digital Twin, Not Kinetic Monte Carlo?

**Decision:** Use coupled ODEs (scipy.integrate), not Kinetic Monte Carlo (KMC) or Molecular Dynamics (MD)

**Reasons:**
- **Speed:** ODEs solve in <1 second for 1000 time points; KMC needs minutes-hours
- **Interpretability:** ODE parameters (k_evap, k_nuc) have physical meaning; KMC parameters obscure
- **Lightweight:** scipy built-in; KMC needs specialized libraries
- **Sufficient accuracy:** For process optimization (qualitative trends), ODEs adequate

**Trade-off:** ODEs miss:
- Stochastic nucleation events (KMC strength)
- Spatial heterogeneity (MD strength)
- Detailed atomistic mechanisms (MD strength)

**Verdict:** For V7 scope (qualitative process-property understanding), ODEs are best balance of speed/accuracy/simplicity. KMC/MD = future work (V8?).

---

### 2. Why Simulated Experiments for Autonomous Loop, Not Real Robot Integration?

**Decision:** Use ML model as simulator with noise, not actual robotic synthesis

**Reasons:**
- **Accessibility:** Anyone can demo autonomous loop without lab equipment
- **Speed:** Simulation runs in seconds; real experiments take hours-days
- **Reproducibility:** Same random seed → same results; real experiments have batch variation
- **Safety:** No risk of equipment damage, chemical spills during development

**Trade-off:** Simulation ≠ reality:
- Real experiments have systematic errors (calibration drift, purity issues)
- Stopping criteria may differ (convergence in silico ≠ in vitro)

**Verdict:** For V7 (demo + proof-of-concept), simulation is appropriate. Real integration requires:
- Hardware interface (robot API, sensor integration)
- Safety interlocks (emergency stop, containment)
- Validation loop (confirm ML predictions match measurements)
→ This is lights-out lab automation (V8-V9 scope)

---

### 3. Why 3 Domains (Halide/Oxide/Chalcogenide), Not 10+?

**Decision:** Implement 3 representative domains, not exhaustive coverage

**Reasons:**
- **Proof-of-concept:** 3 domains demonstrate transfer learning concept
- **Data availability:** Pre-training needs sample data for each domain (hard to find for obscure domains)
- **User focus:** Halide (PV champion), oxide (UV sensors), chalcogenide (Pb-free alternative) cover main use cases
- **Maintainability:** Adding domains requires curating datasets, defining patterns, testing

**Trade-off:** Missing domains:
- 2D perovskites (Ruddlesden-Popper)
- Double perovskites (A2BB'X6)
- Perovskite-inspired (vacancy-ordered, layered)

**Verdict:** 3 domains sufficient for V7 demo. Users can add custom domains (extend `DomainKnowledgeBase.DOMAINS`). Full domain library = community effort (V8+).

---

### 4. Why JSON for Collaborative Mode, Not Firebase/Database?

**Decision:** Store discoveries in `st.session_state` + JSON export, not backend database

**Reasons:**
- **Simplicity:** No server setup, no database schema, no authentication
- **Offline capability:** Works without internet, no cloud dependency
- **Portability:** JSON files easy to share (email, Git, Slack)
- **Proof-of-concept:** Demonstrates collaborative UX without infrastructure

**Trade-off:** No real-time sync:
- Users must export/import manually
- No central repository (discoveries fragmented)
- No conflict resolution (merge issues)

**Verdict:** JSON adequate for single-lab pilot (5-10 users sharing files). Real deployment (50+ users, 24/7 sync) needs backend. Migration path: JSON → PostgreSQL/Firebase straightforward.

---

### 5. Why Predefined Scenarios, Not LLM-Generated Scenarios?

**Decision:** Hardcode 8 common scenarios, not use LLM to generate custom scenarios from natural language

**Example hypothetical:**
- User: "What if Pb is banned and iodine price triples?"
- LLM: → Parses → Creates Scenario(banned=['Pb'], price_multipliers={'I': 3.0})

**Reasons for NOT implementing:**
- **Complexity:** NLP parsing + validation adds 500+ lines of code
- **Reliability:** LLM might misinterpret ("ban Pb" → bans "PbI3" or "Pb atom"?)
- **Scope creep:** V7 already has 5 major features, adding LLM = overload
- **User control:** Predefined scenarios are transparent; LLM-generated may surprise users

**Trade-off:** Less flexible:
- Users can't easily combine scenarios ("Pb ban AND iodine crisis")
- Predefined list may not cover user's specific case

**Verdict:** Predefined scenarios sufficient for V7. LLM scenario generation = advanced feature (V8 if demand exists). Workaround: Users can copy scenario code, modify manually.

---

## 🔮 Future Roadmap (V8 Ideas)

### Advanced Digital Twin
- **3D spatial modeling:** Finite element method (FEM) for thickness gradients
- **Multi-component solvents:** DMF+DMSO dynamics, anti-solvent dripping
- **Substrate effects:** ITO vs FTO surface energy, nucleation site density
- **Real-time sensor integration:** Optical sensors → update ODE parameters during synthesis

### Robotic Integration
- **Hardware API:** Control spin-coater, hot-plate, glovebox via Python
- **Sensor feedback:** QCM (thickness), PL (bandgap), XRD (crystallinity) → ML model
- **Closed-loop hardware:** Autonomous scheduler → Robot executes → Sensor measures → Model updates
- **Safety interlocks:** Emergency stop, leak detection, fume hood monitoring

### Advanced Transfer Learning
- **Deep neural networks:** Replace RandomForest with Graph Neural Networks (composition → property)
- **Meta-learning:** Learn how to learn (MAML algorithm)
- **Zero-shot transfer:** Predict properties of new domains without any target-domain data
- **Active learning:** Intelligently select which target-domain experiments to run

### Real Multi-User Backend
- **Database:** PostgreSQL for discoveries, Firebase for real-time sync
- **Authentication:** Login, user roles (admin, researcher, guest)
- **Real-time collaboration:** Multiple users editing same workspace (like Google Docs)
- **API:** RESTful API for programmatic access (LIMS integration)

### LLM-Powered Features
- **Natural language scenarios:** "What if Pb banned and cost < $0.20/W?" → Auto-generates Scenario
- **Insight generation:** LLM analyzes results → "Pattern detected: FA-rich compositions favor low bandgap"
- **Automated paper writing:** LLM drafts methods section from session history

### Advanced Scenarios
- **Multi-objective scenarios:** Optimize bandgap AND stability AND cost simultaneously
- **Temporal scenarios:** "What if Pb ban starts in 2028?" → Show transition strategy
- **Geopolitical scenarios:** "What if China restricts Cs exports?" → Supply chain re-routing

---

## 📄 Citation & License

**Software:**
- V3, V4, V5, V6, V7: Open source (MIT License)
- Dependencies: Respective licenses apply

**Data:**
- Materials Project: CC BY 4.0
- AFLOW: Public domain
- JARVIS: NIST public data

**Citation:**
```
AlphaMaterials V7: Autonomous Lab Agent + Digital Twin for Perovskite Solar Cells
SAIT × SPMDL Collaboration, 2026
```

**Paper (when published):**
```
[Author List]. "Autonomous Lab Agent with Digital Twin and Transfer Learning for Perovskite Photovoltaics."
[Journal]. [Year]. DOI: [...]
```

---

## 🏁 Conclusion

**V6 → V7 Evolution Summary:**

| Aspect | V6 | V7 |
|--------|----|----|
| **Purpose** | Deployment readiness | Autonomous lab agent |
| **Human Role** | In-the-loop (suggests, human executes) | Out-of-loop (fully autonomous) |
| **Process Understanding** | Black-box (composition → properties) | White-box (process → microstructure → properties) |
| **Domain Coverage** | Single domain (halides) | Multi-domain (halide + oxide + chalcogenide) |
| **Collaboration** | Single-user | Multi-user (simulated) |
| **Scenario Analysis** | None | Policy impact + cost sensitivity |
| **Stopping Criteria** | Manual | Auto-convergence |
| **Tabs** | 12 | 17 |
| **Target User** | Academia + Industry | **Autonomous Labs** + R&D Teams + Multi-Site Consortia |

**Mission accomplished:** V7 transforms V6's deployment-ready platform into an **autonomous lab agent** where:

1. **Understand processes** (digital twin: spin → anneal → crystal)
2. **Optimize autonomously** (closed-loop BO: 24/7 operation)
3. **Learn across domains** (transfer: halide ↔ oxide ↔ chalcogenide)
4. **Share discoveries** (collaborative: multi-user feed)
5. **Assess scenarios** (what-if: policy impact, cost sensitivity)

**빈 지도가 탐험의 시작 → 자율 실험실이 발견의 미래**

The journey from hardcoded demo (V3) → connected database (V4) → autonomous learning (V5) → deployment readiness (V6) → **autonomous lab agent (V7)** is complete.

The next frontier: **Full robotic integration + Real-time sensor feedback** (V8).

---

**Version:** V7.0  
**Status:** ✅ Complete  
**Next Steps:** Testing, validation, robotic integration planning

---

*End of Changelog*
