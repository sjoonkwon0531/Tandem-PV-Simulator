# AlphaMaterials V10: Autonomous Research Agent 🗣️

**The Ultimate Natural Language Interface for Materials Discovery**

---

## 🎯 What's New in V10?

V10 transforms the materials discovery platform into an **autonomous research agent** that speaks your language and automates your workflow:

### 5 New Revolutionary Features:

1. **🗣️ Natural Language Query Engine**
   - Ask questions in plain English: *"Find me a cheap, stable perovskite with bandgap 1.3 eV"*
   - System automatically maps to appropriate tools (search, design, optimize, predict, compare)
   - Query refinement: *"Now make it lead-free"*
   - No ML expertise required!

2. **📄 Automated Research Report Generator**
   - **One-click** publication-ready reports
   - 3 templates: Journal Paper | Internal Report | Presentation
   - Auto-include figures, tables, statistics
   - Markdown + HTML export

3. **🧪 Synthesis Protocol Generator**
   - AI suggests composition → **Printable lab protocol**
   - Step-by-step: precursor weighing, spin-coating, annealing
   - Safety warnings (Pb toxicity, Sn air-sensitivity)
   - Time & cost estimates

4. **🕸️ Knowledge Graph Visualization**
   - Interactive network: Compositions ↔ Properties ↔ Processes ↔ Applications
   - Discovery path tracking: *"How did we get to this candidate?"*
   - Path finding, similarity analysis
   - Export as JSON

5. **🎯 Decision Matrix (TOPSIS/AHP)**
   - *"Which candidate should I synthesize first?"* → **Systematic ranking**
   - Multi-criteria: Bandgap, stability, efficiency, cost
   - Radar charts, comparison tables
   - Decision rationale: *"Why this candidate?"*

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/sjoonkwon0531/Tandem-PV-Simulator.git
cd Tandem-PV-Simulator
pip install -r requirements.txt
```

### Run V10

```bash
# Full Streamlit app (32 tabs)
streamlit run app_v10.py

# Quick feature demo (no UI)
python3 demo_v10.py
```

---

## 📊 Complete Workflow Example

### From Question to Lab Bench in 6 Steps:

```
Step 1: Natural Language Query (Tab 27)
User: "Find me a perovskite with bandgap near 1.3 eV that's stable and cheap"
  → System searches database, returns 12 candidates

Step 2: Bayesian Optimization (Tab 5)
  → Runs 50 iterations
  → Best: Cs0.1FA0.9PbI2.8Br0.2 (Eg=1.35 eV, stability=0.85)

Step 3: Decision Matrix (Tab 31)
  → TOPSIS ranking of 5 candidates
  → Recommended: Cs0.1FA0.9PbI2.8Br0.2 (Score 0.782)
  → Rationale: "Best balance across criteria"

Step 4: Synthesis Protocol (Tab 29)
  → Generate step-by-step protocol
  → Time: 2h 50min | Cost: $78
  → Print PDF for lab bench

Step 5: Knowledge Graph (Tab 30)
  → Visualize discovery path: MAPbI3 → FAPbI3 → Cs0.1FA0.9PbI3 → Best
  → Insight: "Iterative improvement via FA + Cs substitution"

Step 6: Research Report (Tab 28)
  → Generate journal paper draft
  → Sections: Abstract, Intro, Methods, Results, Discussion, Conclusions
  → Download Markdown for editing
```

**Result:** AI prediction → Lab-ready protocol → Publication draft **in minutes, not weeks!**

---

## 🆕 V10 Features Deep Dive

### 1. Natural Language Query Engine

**Example Queries:**

| Query | Tool | Action |
|-------|------|--------|
| "Find stable materials with low cost" | search | Database query with filters |
| "Design a material with bandgap 1.5 eV" | design | Inverse design |
| "Optimize for efficiency and cost" | optimize | Multi-objective BO |
| "What's the bandgap of MAPbI3?" | predict | ML model inference |
| "Compare MAPbI3 and FAPbI3" | compare | Side-by-side analysis |

**Technical:** Regex + keyword matching (no external LLM API). Self-contained, explainable parsing.

---

### 2. Research Report Generator

**Templates:**

- **Journal Paper:** Abstract, Intro, Methods, Results, Discussion, Conclusions, References
- **Internal Report:** Executive summary, key findings, recommendations, next steps
- **Presentation:** Key findings, results, recommendations only

**Export:** Markdown (editable) + HTML (web-ready)

**Use Case:** Discovery campaign → One-click → Draft manuscript → Edit → Submit to journal

---

### 3. Synthesis Protocol Generator

**Example: MAPbI3**

```
Total Time: 3h 16min
Total Cost: $40 (10 substrates)

Safety Warnings:
  ☠️ LEAD HAZARD: Use PPE, fume hood, dispose as hazardous waste
  🧪 SOLVENT HAZARDS: DMF, DMSO toxic

Steps:
  1. Weigh precursors (MAI 159mg, PbI2 461mg) ⭐ CRITICAL
  2. Dissolve in DMF:DMSO (4:1) at 60°C (2 hours)
  3. Filter through 0.45 µm PTFE ⭐ CRITICAL
  4. Clean FTO substrates (30 min)
  5. Spin-coat at 4000 rpm + chlorobenzene drip ⭐ CRITICAL
  6. Anneal at 100°C (15 min) ⭐ CRITICAL
  7. Quality check (visual, XRD)
```

**Export:** Markdown → Print as PDF for lab bench

**Special Features:**
- Lead-free detection → Safety warnings
- Sn-based → Requires glovebox (inert atmosphere)
- Mixed halides → Phase segregation warnings

---

### 4. Knowledge Graph Visualization

**Nodes:**
- Composition (blue): MAPbI3, FAPbI3, etc.
- Property (green): Bandgap, stability, efficiency
- Process (red): Spin-coating, annealing
- Application (orange): Tandem solar cell
- Discovery (purple): BO iterations

**Edges:**
- `has_property`: Composition → Property (with value)
- `requires_process`: Composition → Process
- `enables_application`: Composition → Application
- `similar_to`: Composition ↔ Composition
- `derived_from`: Discovery path

**Interactive:** Zoom, pan, click nodes, find paths

**Use Case:** "How did we discover the best candidate?" → Graph shows iterative improvement path

---

### 5. Decision Matrix (TOPSIS)

**Example:**

Criteria: Bandgap (30%), Stability (35%), Efficiency (25%), Cost (10%)

Candidates: MAPbI3, FAPbI3, CsPbI3, Cs0.1FA0.9PbI3, Cs0.1FA0.9PbI2.8Br0.2

**TOPSIS Results:**

| Rank | Candidate | Score | Why? |
|------|-----------|-------|------|
| 🥇 1 | Cs0.1FA0.9PbI2.8Br0.2 | 0.782 | Ideal bandgap (1.35 eV), balanced |
| 🥈 2 | Cs0.1FA0.9PbI3 | 0.745 | Highest efficiency (22.8%) |
| 🥉 3 | FAPbI3 | 0.612 | Good balance |

**Decision Rationale:**
> "Cs0.1FA0.9PbI2.8Br0.2 recommended: Exactly 1.35 eV target for tandem cell top subcell, high stability (0.85), excellent efficiency (22.3%), reasonable cost ($0.47/W)."

**Visualizations:**
- Radar chart: Multi-dimensional comparison
- Bar chart: Overall scores
- Pie chart: Criteria weights
- Sensitivity analysis: How ranking changes with different weights

---

## 📦 What's Preserved from V9?

**All 27 V9 tabs + 5 new V10 tabs = 32 total**

✅ Federated Learning (multi-lab collaboration, FedAvg)  
✅ Differential Privacy (ε-δ DP, privacy-accuracy tradeoff)  
✅ Multi-Lab Discovery (contribution leaderboard, Shapley values)  
✅ Data Heterogeneity (KL, EMD metrics)  
✅ Incentive Mechanism (cost-benefit analysis)  
✅ All V8 features (Model Zoo, API, Benchmarks, Education)  
✅ All V7 features (Digital Twin, Autonomous, Transfer Learning)  
✅ All V6 features (Inverse Design, TEA, Export)

**V10 = V9 + Autonomous Research Agent**

---

## 🏗️ Architecture

### File Structure

```
tandem-pv/
├── app_v10.py                  # Main app (32 tabs)
├── demo_v10.py                 # Quick feature demo
├── V10_CHANGELOG.md            # Complete documentation
├── utils/
│   ├── nl_query.py             # Natural language parser
│   ├── report_generator.py     # Research reports
│   ├── protocol_generator.py   # Synthesis protocols
│   ├── knowledge_graph.py      # Knowledge graph
│   └── decision_matrix.py      # TOPSIS/AHP
└── [V4-V9 modules...]
```

### Dependencies

**No new dependencies!** V10 uses only existing libraries:
- `numpy`, `pandas`, `plotly`, `streamlit` (existing)
- `re` (regex for NL parsing)
- `scipy` (TOPSIS distance calculations)
- `networkx` (knowledge graph)

**Philosophy:** Self-contained, lightweight, no external APIs.

---

## 🎓 Key Innovations

1. **Accessibility:** Natural language → No ML expertise required
2. **Automation:** One-click reports + protocols → Hours saved
3. **Systematic Decisions:** TOPSIS → Not gut feeling
4. **Knowledge Integration:** Graph → See the big picture
5. **Lab-Ready Output:** Protocols → Print → Synthesize

**Bridges the gap** between AI predictions and experimental synthesis.

---

## 📊 V9 vs V10 Comparison

| Feature | V9 | V10 |
|---------|----|----|
| Natural Language Interface | ❌ | ✅ |
| Automated Reports | ❌ | ✅ |
| Synthesis Protocols | ❌ | ✅ |
| Knowledge Graph | ❌ | ✅ |
| Decision Matrix | ❌ | ✅ |
| Federated Learning | ✅ | ✅ |
| Differential Privacy | ✅ | ✅ |
| Target User | Multi-lab consortium | **Experimental researchers** |

---

## 🔮 Future Work (V11 Ideas)

- **Voice input:** Hands-free queries in lab
- **Auto-insert figures:** Plotly → Reports
- **Literature search:** Crossref API → Auto-cite
- **Equipment integration:** Push protocols to lab instruments
- **Graph reasoning:** "Find all lead-free materials similar to MAPbI3"
- **Experimental feedback loop:** Lab results → Update model → Re-rank

---

## 📄 Citation

```bibtex
@software{alphamaterials_v10,
  title={AlphaMaterials V10: Autonomous Research Agent + Natural Language Interface},
  author={SAIT × SPMDL × Autonomous Discovery},
  year={2026},
  url={https://github.com/sjoonkwon0531/Tandem-PV-Simulator}
}
```

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built on V3-V9 foundation. V10 adds autonomous research agent capabilities.

**Key insight:** The bottleneck isn't just data or algorithms—it's making AI tools **accessible and actionable** for experimental researchers.

---

## 🚀 Get Started Now!

```bash
# Clone repository
git clone https://github.com/sjoonkwon0531/Tandem-PV-Simulator.git
cd Tandem-PV-Simulator

# Run demo
python3 demo_v10.py

# Launch full app
streamlit run app_v10.py
```

**Start discovering materials in natural language today!** 🗣️

---

**Version:** V10.0  
**Status:** ✅ Complete  
**Date:** 2026-03-15

**V10 = From AI Predictions to Lab Bench in One Click** 🚀
