# Tandem PV Simulator V3 — Design Document

## Architecture: 2-Stage Simulation with Pre-computed DB

### Core Philosophy
- **95% lookup, 5% compute** — pre-compute everything possible
- **2-stage workflow** — quick preview → full analysis
- **Confidence scoring** — every value tagged with reliability

---

## A. Pre-computed Material Database

### ABX₃ Perovskite Property Table
Grid: A-site (MA/FA/Cs/Rb) × B-site (Pb/Sn/Ge) × X-site (I/Br/Cl) × mixing fractions
~5,000 compositions at 10% step, fine grid ~50,000 at 2% step in promising regions

**Properties per composition:**
| Property | Unit | Source Priority |
|----------|------|---------------|
| Bandgap (Eg) | eV | Expt > DFT > ML |
| Crystal phase (RT) | cubic/tetra/ortho | Expt > tolerance factor model |
| Phase transition temp | K | Expt > empirical |
| Surface energy | J/m² | DFT > model |
| Absorption spectrum α(λ) | cm⁻¹, 300-1550nm | Expt > model |
| Refractive index n(λ), k(λ) | -, 300-1550nm | Expt > model |
| Exciton generation rate | cm⁻³s⁻¹ | Calculated from α |
| Exciton binding energy | meV | Expt > DFT |
| Exciton lifetime | ns | Expt > model |
| Carrier mobility (e/h) | cm²/Vs | Expt > model |
| Defect tolerance score | 0-10 | ML from literature |
| Trap density (typical) | cm⁻³ | Expt range |
| CTE | ppm/K | Expt > model |
| Lattice parameter | Å | Vegard + bowing |
| Tolerance factor | - | Calculated |
| Goldschmidt stability | stable/marginal/unstable | Calculated |
| Deformation potential | eV | DFT > literature |
| Confidence score | ★/★★/★★★ | Auto-assigned |
| Reference | DOI | Tagged per value |

### Track A Materials (Non-perovskite)
Same property set for: c-Si, a-Si, GaAs, GaInP, InGaAs, CIGS, CdTe, organic PV, QD, etc.

### Electrode Database
| Material | Type | Sheet R (Ω/□) | Transmittance | Work Function | Stability | TRL | Ref |
|----------|------|--------------|---------------|---------------|-----------|-----|-----|
| ITO | TCO | 10-15 | >85% | 4.7eV | ★★★ | 9 | - |
| FTO | TCO | 7-15 | >80% | 4.4eV | ★★★ | 9 | - |
| AZO | TCO | 20-50 | >85% | 4.4eV | ★★ | 7 | - |
| Ag (evap) | Metal | <1 | opaque | 4.3eV | ★★ | 9 | - |
| Au (evap) | Metal | <1 | opaque | 5.1eV | ★★★ | 9 | - |
| Cu | Metal | <1 | opaque | 4.7eV | ★ | 7 | - |
| PEDOT:PSS | Polymer | 50-200 | >85% | 5.0eV | ★ | 8 | - |
| Graphene | Carbon | 30-300 | >90% | 4.5eV | ★★ | 5 | - |

### ETL Database (최근 3년 고성능 + 고안정성)
| Material | Eg (eV) | Mobility (cm²/Vs) | LUMO/CB (eV) | Stability | Deposition | TRL | Ref |
|----------|---------|-------------------|-------------|-----------|-----------|-----|-----|
| SnO₂ | 3.6 | 15-25 | -4.1 | ★★★ | ALD/spin | 9 | - |
| TiO₂ | 3.2 | 0.1-1 | -4.0 | ★★★ | ALD/spin | 9 | - |
| ZnO | 3.3 | 10-50 | -4.2 | ★★ | ALD/spin | 8 | - |
| C₆₀ | 1.7 | 1-5 | -4.5 | ★★ | evap | 8 | - |
| PCBM | 1.7 | 0.01-0.1 | -3.9 | ★ | spin | 7 | - |
| BCP | 3.5 | - | -3.5 (EBL) | ★★ | evap | 8 | - |

### HTL Database (최근 3년 고성능 + 고안정성)
| Material | Eg (eV) | Mobility (cm²/Vs) | HOMO/VB (eV) | Stability | Deposition | TRL | Ref |
|----------|---------|-------------------|-------------|-----------|-----------|-----|-----|
| Spiro-OMeTAD | 3.0 | 1e-4 | -5.2 | ★ | spin | 9 | - |
| PTAA | 3.0 | 0.01 | -5.2 | ★★ | spin | 8 | - |
| NiOₓ | 3.6 | 0.01 | -5.4 | ★★★ | sputter/ALD | 8 | - |
| Me-4PACz (SAM) | 3.5 | - | -5.5 | ★★★ | spin/dip | 9 | - |
| 2PACz (SAM) | 3.5 | - | -5.4 | ★★★ | spin/dip | 9 | - |
| PEDOT:PSS | 1.5 | 0.1-1 | -5.0 | ★ | spin | 8 | - |
| Cu₂O | 2.1 | 1-10 | -5.4 | ★★ | sputter | 6 | - |
| P3HT | 1.9 | 0.01 | -5.0 | ★ | spin | 7 | - |

---

## B. Sidebar Workflow (Redesigned)

```
🌞 탠덤 PV 시뮬레이터 V3
━━━━━━━━━━━━━━━━━━━━━━
📊 Step 1: 재료 트랙
   [A - Multi-material] [B - All-Perovskite ABX₃]

🔢 Step 2: 접합 수
   [2T] [3T] [4T] [5T] [6T] [8T] [10T] [∞]

⚡ Step 3: 전극
   Top: [ITO ▼]  Bottom: [Ag ▼]

🔼 Step 4: ETL
   [SnO₂ ▼]

🔽 Step 5: HTL
   [Me-4PACz ▼]

━━━━━━━━━━━━━━━━━━━━━━
🌡️ 동작 조건
   온도: [25°C]  RH: [50%]
   위도: [37.5°N Seoul ▼]
   면적: [1 cm² ▼]
━━━━━━━━━━━━━━━━━━━━━━
🔬 [1차 시뮬레이션 — 구조 프리뷰]
   ↓ (사용자 확인 후)
🚀 [2차 풀 시뮬레이션]
```

---

## C. Stage 1: Quick Preview (~2-5초)

**DB Lookup + Light Calculation:**
1. N-junction 최적 Eg 분포 (pre-computed Pareto front에서 조회)
2. 각 층 ABX₃ 조성 매칭 (DB에서 target Eg ± 0.05eV 검색)
3. 결정상 안정성 체크 (RT에서 stable phase인지)
4. Normal + Inverted 구조 동시 제안

**표시 내용:**
- Multilayer 단면도 (전극/ETL/Absorber×N/HTL/전극)
- Total absorption spectrum (300-1550nm)
- Layer별 absorption spectrum
- Exciton generation rate profile (depth vs rate)
- E-field distribution (TMM, pre-computed에서 보간)
- 각 층 요약 테이블: 조성, Eg, 두께, 결정상, confidence

---

## D. Stage 2: Full Simulation (사용자 OK 후, ~30초)

**Delta 계산 (1차 결과 기반, 변경분만):**
1. I-V 곡선 (각 subcell + tandem)
2. 계면 안정성 (lattice mismatch, strain energy, interdiffusion)
3. Strain → Eg shift (deformation potential)
4. 비복사 재결합 (SRH + Auger → realistic PCE)
5. 환경 안정성 (RH + 온도 + UV)
6. 24시간 발전량 (위도 + 날짜 + 면적)
7. 장기 열화: 1주, 1달, 6개월, 1년, 2년, 5년, 10년, 20년
8. 제어 전략 (능동 TRL 표시 / 수동)
9. 경제성: LCOE, $/Wp, EPBT, 제조 비용 분해
10. NREL 벤치마크 비교
11. 민감도 분석 (Sobol → tornado chart)
12. 공정 레시피 (증착법, 온도, 어닐링)
13. 정책적 함의 (RE100, K-ETS, CBAM, IRA)
14. Bifacial gain (해당 시)

---

## E. Crystal Phase Modeling

### Phase Stability Rules
- tolerance factor t: 0.8-1.0 → cubic (ideal)
- t: 0.71-0.8 → orthorhombic
- t > 1.0 → hexagonal/non-perovskite
- Phase transition temperatures from literature DB

### RT Phase Map
- MAPbI₃: tetragonal (RT), cubic (>327K)
- FAPbI₃: α-phase cubic (desired, metastable at RT), δ-phase hexagonal (stable)
- CsPbI₃: orthorhombic (RT stable), cubic (>583K) → needs additives for RT stabilization
- Mixed compositions: phase stability depends on mixing ratios

### Warning System
- 🟢 Stable at RT (confirmed cubic/tetragonal)
- 🟡 Marginal (phase transition near RT, ±30K)
- 🔴 Unstable (non-perovskite phase at RT without additives)

---

## F. Hierarchical DB Strategy

### Level 1: Coarse Grid (~5,000 points)
- 10% composition steps
- Loaded at app startup (parquet, ~2MB)
- Used for ternary phase diagram coloring
- Instant lookup

### Level 2: Fine Grid (~50,000 points in promising regions)
- 2% steps where Level 1 shows interesting properties
- Loaded on demand
- Used for optimization interpolation

### Level 3: On-demand Calculation
- Only for final 2-3 candidate structures
- Full TMM, I-V, degradation ODE
- ~10-30 seconds per structure

### Pre-computed Pareto Fronts
- For each (Track, N-junction) combination: optimal Eg distributions
- Stored as JSON: {(B, 2T): [{egs: [1.8, 1.2], pce: 0.28, ...}, ...]}
- ~100 solutions per combination, ~20 combinations = ~2,000 pre-computed optima

---

## G. Confidence Scoring System

| Level | Symbol | Meaning | Example |
|-------|--------|---------|---------|
| 3 | ★★★ | Experimental, peer-reviewed | MAPbI₃ Eg = 1.55 eV |
| 2 | ★★ | DFT/computational, validated | CsSnGeI₃ Eg from DFT |
| 1 | ★ | ML prediction, interpolated | Mixed composition ML |
| 0 | ⚠️ | Extrapolated, low confidence | Far from training data |

---

## H. Implementation Phases

### Phase 1: Pre-computed DB Generation (로컬 서버)
- `scripts/generate_db.py` — ABX₃ 조성 그리드 생성 + 물성 계산
- `scripts/generate_pareto.py` — N-junction 최적해 사전 계산
- `data/perovskite_db.parquet` — Level 1+2 DB
- `data/pareto_fronts.json` — Pre-computed optima
- `data/electrodes.json`, `data/etl.json`, `data/htl.json`

### Phase 2: App Rebuild (app.py V3)
- Sidebar workflow redesign
- Stage 1 quick preview
- Stage 2 full simulation
- Crystal phase warnings
- Confidence badges

### Phase 3: Advanced Physics
- Non-radiative recombination model
- Strain → Eg shift
- Optical interference pre-computation
- Sobol sensitivity analysis
- Process recipe database

### Phase 4: Polish
- Bifacial mode
- Policy implications module
- Auto literature update pipeline
- Export: PDF report, recipe card
