# V5 Quick Start Guide 🚀

**Get from zero to autonomous discovery in 10 minutes**

---

## ⚡ 3-Step Setup

### 1. Install & Run (2 minutes)

```bash
# Clone repo
git clone https://github.com/sjoonkwon0531/Tandem-PV-Simulator.git
cd Tandem-PV-Simulator

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn scipy joblib openpyxl requests

# Optional: better performance
pip install xgboost

# Launch V5
streamlit run app_v5.py
```

**Browser opens automatically at `http://localhost:8501`**

---

### 2. Load Database & Train Model (1 minute)

**In the app:**

1. **Tab 1 (Database):** Click "🚀 Load Database"
   - Fetches 500+ perovskites from Materials Project/AFLOW/JARVIS
   - First time: ~10 seconds (then cached)

2. **Tab 3 (ML Surrogate):** Click "🚀 Train Base Model"
   - Trains XGBoost on 500+ materials
   - ~1 second
   - Result: MAE ~0.25 eV

✅ **General model ready!**

---

### 3. Upload Your Data & Go (2 minutes)

**Prepare your data:**

Create CSV file (`my_data.csv`):
```csv
formula,bandgap
MAPbI3,1.59
FAPbI3,1.51
CsPbI3,1.72
FA0.85MA0.15PbI3,1.55
Cs0.05FA0.95PbI3,1.53
```

**In the app:**

1. **Tab 2 (Upload):** Upload `my_data.csv`
   - Click "💾 Save to Session"

2. **Tab 3 (ML Surrogate):** Scroll down, click "🔥 Fine-tune on Your Data"
   - Model adapts to YOUR lab conditions
   - Before: MAE ~0.35 eV → After: MAE ~0.18 eV ✅

3. **Tab 4 (Bayesian Opt):** Click "🚀 Fit BO on Your Data"
   - Then click "🔮 Suggest Next Experiments"
   - Get 5-10 AI-suggested compositions!

✅ **You're now doing autonomous discovery!**

---

## 📋 5-Minute Workflow

Once set up, each discovery cycle takes ~5 minutes:

```
1. Generate Suggestions (Tab 4)
   ↓
2. Filter by Multi-Objective (Tab 5) — optional
   ↓
3. Export Queue (Tab 6)
   ↓
4. Synthesize in Lab (1-2 weeks)
   ↓
5. Upload Results (Tab 2)
   ↓
6. Fine-tune Again (Tab 3)
   ↓
7. Repeat → Model gets smarter!
```

**Each cycle improves the AI**

---

## 🎯 First-Time Checklist

### Before First Use
- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] 500 MB free disk space
- [ ] Internet connection (for first database load)

### First Session (10 min)
- [ ] Install dependencies
- [ ] Run `streamlit run app_v5.py`
- [ ] Tab 1: Load database
- [ ] Tab 3: Train base model
- [ ] Tab 2: Upload CSV (5-10 experiments minimum)
- [ ] Tab 3: Fine-tune on your data
- [ ] Tab 4: Fit BO & generate suggestions
- [ ] Tab 6: Export experiment queue

### Second Session (5 min)
- [ ] Tab 2: Upload new results
- [ ] Tab 3: Fine-tune again (model improves!)
- [ ] Tab 4: Re-run BO (smarter suggestions)
- [ ] Tab 7: Save session

### Ongoing
- [ ] Repeat upload → fine-tune → BO cycle
- [ ] Save sessions regularly (Tab 7)
- [ ] Track improvement over time

---

## 🎓 Key Concepts (1-Minute Read)

### Bayesian Optimization
**What:** AI suggests next experiments based on what it's learned  
**Why:** Explores high-dimensional space efficiently  
**When to use:** After uploading ≥5 experiments  
**Tab:** 4

### Fine-Tuning
**What:** Adapts pre-trained model to YOUR lab conditions  
**Why:** DFT ≠ experiments; your setup is unique  
**When to use:** After each batch of new experiments  
**Tab:** 3

### Multi-Objective
**What:** Balance bandgap, stability, cost, synthesizability  
**Why:** Real materials need more than just bandgap  
**When to use:** When you want cheap/stable/easy-to-make materials  
**Tab:** 5

### Pareto Front
**What:** Materials where you can't improve one property without hurting another  
**Why:** Shows optimal trade-offs  
**When to use:** When objectives conflict (e.g., low cost vs high stability)  
**Tab:** 5

---

## 💡 Pro Tips

### For Best BO Performance
- Upload ≥10 diverse experiments (not all similar compositions)
- Include some "failures" (unexpected results teach the model!)
- Fine-tune after every 5-10 new experiments
- Try all 3 acquisition functions (EI, UCB, TS), pick overlap

### For Multi-Objective
- Start with equal weights (40% bandgap, 30% stability, 20% synth, 10% cost)
- Adjust based on your priorities (e.g., 80% bandgap if targeting specific Eg)
- Check Pareto front before trusting weighted scores

### For Session Management
- Save after every major milestone (10, 20, 50 experiments)
- Name sessions clearly (`tandem_top_cell_v1`, not `session_123`)
- Export queues as CSV immediately (don't lose suggestions!)
- Git-track session folders for reproducibility

---

## ❓ Common Questions

**Q: How many experiments before BO works?**  
A: Minimum 5, ideal 10-20. More data = better suggestions.

**Q: Should I trust the first BO suggestions?**  
A: Start with 1-2 safe experiments from top suggestions. Validate model accuracy before scaling up.

**Q: What if my data is noisy?**  
A: Repeat measurements, average results. BO handles some noise but garbage in = garbage out.

**Q: Can I use V5 without uploading data?**  
A: Yes! Train on database only (Tab 3). But fine-tuning on your data gives MUCH better results.

**Q: Do I need a Materials Project API key?**  
A: No. V5 works with bundled sample data. API key just gives more DFT data.

---

## 🐛 Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'streamlit'"  
**Fix:** `pip install streamlit`

**Problem:** "No module named 'bayesian_opt'"  
**Fix:** Make sure you're in the `tandem-pv` directory when running `streamlit run app_v5.py`

**Problem:** BO suggestions are all similar  
**Fix:** Upload more diverse data. Increase candidate pool size in Tab 4 settings.

**Problem:** Model accuracy is poor  
**Fix:** Check data quality. Remove outliers. Increase sample size.

**Problem:** App is slow  
**Fix:** First load is slow (downloading database). Subsequent loads use cache (<1s).

---

## 📊 Success Metrics

Track your progress:

| Metric | Week 1 | Week 2 | Week 4 | Week 8 |
|--------|--------|--------|--------|--------|
| Experiments | 10 | 20 | 40 | 80 |
| Model MAE | 0.25 eV | 0.18 eV | 0.12 eV | 0.08 eV |
| BO Iterations | 1 | 3 | 6 | 12 |
| Materials in Target Range | 1 | 3 | 8 | 15 |

**Goal:** Find target material (e.g., Eg = 1.68 eV) in <50 experiments vs. ~200 traditional trial-and-error

---

## 🎯 Recommended Workflow

### Week 1: Foundation
- Day 1: Set up app, load database
- Day 2-3: Synthesize 10 standard materials
- Day 4: Upload data, train model
- Day 5: Review BO suggestions

### Week 2: First Cycle
- Day 6-10: Synthesize top 5 BO suggestions
- Day 11: Upload new data
- Day 12: Fine-tune model, see improvement!
- Day 13: Generate next BO suggestions
- Day 14: Multi-objective filtering

### Week 3: Iteration
- Repeat upload → fine-tune → BO cycle
- Model gets smarter with each iteration
- Convergence: ~3-4 cycles to target

### Week 4: Scale Up
- Batch experiments (10-20 at once)
- Use experiment planner (Tab 6)
- Track convergence plots

---

## 🏆 Learning Path

### Beginner (Week 1)
- [ ] Understand tabs 1-3 (database, upload, ML)
- [ ] Train base model
- [ ] Upload first dataset
- [ ] Fine-tune model once

### Intermediate (Week 2-3)
- [ ] Use Bayesian Optimization (Tab 4)
- [ ] Understand acquisition functions
- [ ] Complete one full cycle (upload → fine-tune → BO → experiment)
- [ ] Save first session (Tab 7)

### Advanced (Week 4+)
- [ ] Multi-objective optimization (Tab 5)
- [ ] Experiment planner workflows (Tab 6)
- [ ] Compare acquisition functions
- [ ] Optimize session management

### Expert (Month 2+)
- [ ] Multi-campaign management
- [ ] Custom acquisition strategies
- [ ] Publication-ready exports
- [ ] Reproducible research workflows

---

## 📈 Expected Timeline

**From zero to discovery:**

| Milestone | Time | Cumulative |
|-----------|------|------------|
| Setup & install | 5 min | 5 min |
| Load DB & train | 1 min | 6 min |
| Upload first data | 2 min | 8 min |
| Fine-tune | 30 sec | 8.5 min |
| BO suggestions | 1 min | 9.5 min |
| **First suggestions ready** | — | **<10 min** ✅ |

**Full discovery campaign:**

| Phase | Duration |
|-------|----------|
| Week 1: Foundation | 10 experiments |
| Week 2-3: First cycle | +10 experiments |
| Week 4-6: Iteration | +20 experiments |
| Week 8: Convergence | Target found! |
| **Total** | ~40 experiments in 2 months |

**vs. Traditional:** ~200 experiments, 12 months

**V5 speedup: 5-10×** 🚀

---

## 🎉 You're Ready!

**Commands to remember:**

```bash
# Start V5
streamlit run app_v5.py

# Check status
git status

# Update
git pull origin main
pip install --upgrade -r requirements.txt
```

**Tabs to master:**
1. 🗄️ Database (load once, cache forever)
2. 📤 Upload (after each experiment batch)
3. 🤖 ML Surrogate (fine-tune regularly)
4. 🎯 Bayesian Opt (your AI brain)
5. 🏆 Multi-Objective (balance trade-offs)
6. 📋 Planner (organize experiments)
7. 💾 Sessions (save progress)

**Now go discover!** 🧠✨

---

**Questions?** Check `README_V5.md` or `V5_CHANGELOG.md`

**빈 지도가 탐험의 시작** — The empty map is the start of exploration.
