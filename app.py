#!/usr/bin/env python3
"""
Tandem PV Simulator V3 - Pre-computed DB + 2-Stage Workflow
=========================================================

V3 Architecture: 95% lookup, 5% compute
- Stage 1: Quick preview (2-5 seconds) with DB lookup
- Stage 2: Full simulation (30 seconds) with detailed calculations
- Pre-computed ABX₃ database with 47K+ compositions
- Korean UI with English annotations

Author: AI Assistant (Subagent)  
Date: 2024-02-24
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure Streamlit
st.set_page_config(
    page_title="탠덤 PV 시뮬레이터 V3",
    page_icon="🌞", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# White theme CSS
st.markdown("""
<style>
    .main { background-color: white; }
    .stSelectbox > div > div { background-color: white; }
    .stSlider > div > div > div { background-color: #2E86AB; }
    .metric-card { 
        background: white; 
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .confidence-high { color: #16A085; font-weight: bold; }
    .confidence-med { color: #F39C12; font-weight: bold; }
    .confidence-low { color: #E74C3C; font-weight: bold; }
    .phase-stable { background-color: #D5FFDE; padding: 4px 8px; border-radius: 12px; }
    .phase-transition { background-color: #FFF4E6; padding: 4px 8px; border-radius: 12px; }
    .phase-unstable { background-color: #FFE6E6; padding: 4px 8px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING WITH CACHING
# =============================================================================

@st.cache_data
def load_perovskite_db():
    """Load pre-computed ABX₃ database"""
    try:
        df = pd.read_parquet('data/perovskite_db.parquet')
        return df
    except FileNotFoundError:
        st.error("❌ Perovskite database not found. Please run scripts/generate_db.py first.")
        return pd.DataFrame()

@st.cache_data
def load_pareto_fronts():
    """Load pre-computed Pareto front solutions"""
    try:
        with open('data/pareto_fronts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Pareto fronts not found. Please run scripts/generate_pareto.py first.")
        return {}

@st.cache_data
def load_electrodes():
    """Load electrode database"""
    try:
        with open('data/electrodes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Electrode database not found.")
        return []

@st.cache_data
def load_etl():
    """Load ETL database"""
    try:
        with open('data/etl.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ ETL database not found.")
        return []

@st.cache_data
def load_htl():
    """Load HTL database"""
    try:
        with open('data/htl.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ HTL database not found.")
        return []

@st.cache_data  
def load_track_a_materials():
    """Load Track A materials"""
    try:
        with open('data/track_a_materials.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Track A materials not found.")
        return []

# Load all databases at startup
perovskite_db = load_perovskite_db()
pareto_fronts = load_pareto_fronts()
electrodes_db = load_electrodes()
etl_db = load_etl()
htl_db = load_htl()
track_a_materials = load_track_a_materials()

# =============================================================================
# SIDEBAR - STEP-BY-STEP WORKFLOW
# =============================================================================

st.sidebar.markdown("# 🌞 탠덤 PV 시뮬레이터 V3")
st.sidebar.markdown("━━━━━━━━━━━━━━━━━━━━━━")

# Initialize session state
if 'stage1_complete' not in st.session_state:
    st.session_state.stage1_complete = False
if 'stage2_complete' not in st.session_state:
    st.session_state.stage2_complete = False

# Step 1: Material Track
st.sidebar.markdown("### 📊 Step 1: 재료 트랙 (Material Track)")
track = st.sidebar.radio(
    "Choose approach",
    ["A - Multi-material", "B - All-Perovskite ABX₃"],
    help="Track A: Mixed materials (Si, III-V, chalcogenides, perovskites)\nTrack B: Only ABX₃ perovskites with composition tuning"
)
track_code = 'A' if 'Multi' in track else 'B'

# Step 2: Number of Junctions
st.sidebar.markdown("### 🔢 Step 2: 접합 수 (Number of Junctions)")
n_junctions = st.sidebar.select_slider(
    "Select junctions",
    options=[2, 3, 4, 5, 6], 
    value=2,
    help="More junctions → higher theoretical efficiency but increased complexity"
)

# Step 3: Top Electrode
st.sidebar.markdown("### ⚡ Step 3: 전극 (Electrodes)")
electrode_names = [e['name'] for e in electrodes_db if e['transmittance'] > 0.7]
electrode_top = st.sidebar.selectbox(
    "Top electrode (transparent)",
    electrode_names,
    index=0 if electrode_names else None,
    help="Front contact - needs high transparency + conductivity"
)

electrode_back_names = [e['name'] for e in electrodes_db]
electrode_bottom = st.sidebar.selectbox(
    "Bottom electrode", 
    electrode_back_names,
    index=3 if len(electrode_back_names) > 3 else 0,
    help="Back contact - optimized for conductivity + work function"
)

# Step 4: ETL
st.sidebar.markdown("### 🔼 Step 4: ETL (Electron Transport)")
etl_names = [e['name'] for e in etl_db]
etl = st.sidebar.selectbox(
    "ETL material",
    etl_names,
    index=0 if etl_names else None,
    help="Electron extraction layer"
)

# Step 5: HTL  
st.sidebar.markdown("### 🔽 Step 5: HTL (Hole Transport)")
htl_names = [h['name'] for h in htl_db]
htl = st.sidebar.selectbox(
    "HTL material", 
    htl_names,
    index=0 if htl_names else None,
    help="Hole extraction layer"
)

st.sidebar.markdown("━━━━━━━━━━━━━━━━━━━━━━")

# Operating Conditions
st.sidebar.markdown("### 🌡️ 동작 조건 (Operating Conditions)")
temperature = st.sidebar.slider("온도 (Temperature) [°C]", 15, 45, 25)
humidity = st.sidebar.slider("상대습도 (RH) [%]", 30, 90, 50)
latitude = st.sidebar.selectbox(
    "위도 (Latitude)",
    ["37.5°N Seoul", "35.1°N Daejeon", "33.5°N Busan", "0° Equator", "52.5°N London"],
    index=0
)
area_cm2 = st.sidebar.selectbox(
    "면적 (Area)",
    ["1 cm² (lab)", "100 cm² (sub-module)", "1 m² (module)", "1000 m² (array)"],
    index=0
)

st.sidebar.markdown("━━━━━━━━━━━━━━━━━━━━━━")

# Stage 1 Button
stage1_button = st.sidebar.button(
    "🔬 1차 시뮬레이션 — 구조 프리뷰",
    help="Quick preview using pre-computed database (~5 seconds)",
    type="primary",
    use_container_width=True
)

# Stage 2 Button (disabled until Stage 1 complete)
stage2_button = st.sidebar.button(
    "🚀 2차 풀 시뮬레이션", 
    help="Full detailed simulation with all physics (~30 seconds)",
    disabled=not st.session_state.stage1_complete,
    type="secondary" if st.session_state.stage1_complete else "secondary",
    use_container_width=True
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_confidence_badge(confidence: int) -> str:
    """Return confidence badge HTML"""
    if confidence == 3:
        return "★★★ <span class='confidence-high'>High</span>"
    elif confidence == 2:
        return "★★ <span class='confidence-med'>Medium</span>"
    else:
        return "★ <span class='confidence-low'>Low</span>"

def get_phase_badge(phase: str, temp: float) -> str:
    """Return crystal phase badge with stability warning"""
    if phase == "cubic" and temp < 350:
        return "<span class='phase-stable'>🟢 cubic (stable)</span>"
    elif phase == "cubic" and temp < 400:
        return f"<span class='phase-transition'>🟡 cubic (phase transition at {temp:.0f}K — 근접 주의)</span>"
    elif phase == "orthorhombic":
        return "<span class='phase-unstable'>🔴 orthorhombic (불안정 — 첨가제 필요)</span>"
    else:
        return f"<span class='phase-unstable'>🔴 {phase} (not perovskite)</span>"

def find_best_composition(target_eg: float, db: pd.DataFrame) -> pd.Series:
    """Find best ABX₃ composition for target bandgap"""
    
    # Filter for stable compositions near target
    candidates = db[
        (abs(db['Eg'] - target_eg) < 0.05) & 
        (db['phase_stable_RT'] == True) &
        (db['stability_score'] > 5)
    ].copy()
    
    if len(candidates) == 0:
        # Relax constraints
        candidates = db[abs(db['Eg'] - target_eg) < 0.1].copy()
    
    if len(candidates) == 0:
        # Emergency fallback
        return db.loc[db['Eg'].sub(target_eg).abs().idxmin()]
    
    # Score by stability, confidence, and bandgap accuracy
    candidates['score'] = (
        candidates['stability_score'] * 0.4 +
        candidates['confidence'] * 0.3 + 
        candidates['defect_tolerance'] * 0.2 +
        (5 - abs(candidates['Eg'] - target_eg)) * 0.1
    )
    
    return candidates.loc[candidates['score'].idxmax()]

def run_stage1(track_code: str, n_junctions: int, electrode_top: str, 
               electrode_bottom: str, etl: str, htl: str, conditions: Dict) -> Dict:
    """Stage 1: Quick preview using pre-computed data"""
    
    st.info("🔄 Stage 1: Loading pre-computed solutions...")
    progress_bar = st.progress(0)
    
    # 1. Load optimal bandgap distribution
    progress_bar.progress(20)
    pareto_key = f"{track_code}_{n_junctions}T"
    if pareto_key in pareto_fronts:
        optimal_solution = pareto_fronts[pareto_key][0]  # Best solution
        optimal_bandgaps = optimal_solution['bandgaps']
    else:
        st.warning(f"⚠️ No pre-computed solution for {pareto_key}. Using fallback.")
        # Simple fallback distribution
        optimal_bandgaps = list(np.linspace(2.4, 1.1, n_junctions))
    
    progress_bar.progress(40)
    
    # 2. Find best ABX₃ compositions for each bandgap (Track B) or materials (Track A)
    layers = []
    if track_code == 'B':
        # All-perovskite: find compositions from database
        for i, target_eg in enumerate(optimal_bandgaps):
            best_comp = find_best_composition(target_eg, perovskite_db)
            layers.append({
                'layer_type': 'absorber',
                'material': f"ABX₃ Layer {i+1}",
                'composition': best_comp,
                'bandgap': best_comp['Eg'],
                'thickness_nm': 300 + i * 100,  # Typical thicknesses
                'confidence': best_comp['confidence']
            })
    else:
        # Multi-material: use Track A materials  
        for i, target_eg in enumerate(optimal_bandgaps):
            # Find best Track A material near target bandgap
            track_a_df = pd.DataFrame(track_a_materials)
            best_match_idx = abs(track_a_df['bandgap'] - target_eg).idxmin()
            best_material = track_a_materials[best_match_idx]
            
            layers.append({
                'layer_type': 'absorber',
                'material': best_material['name'],
                'bandgap': best_material['bandgap'],
                'thickness_nm': best_material.get('thickness_typical_um', 1.0) * 1000,
                'confidence': 3,  # High confidence for established materials
                'properties': best_material
            })
    
    progress_bar.progress(60)
    
    # 3. Build full device stack
    stack_layers = []
    
    # Top electrode
    top_electrode_data = next(e for e in electrodes_db if e['name'] == electrode_top)
    stack_layers.append({
        'name': electrode_top,
        'type': 'electrode',
        'thickness_nm': top_electrode_data['thickness_typical_nm'],
        'properties': top_electrode_data
    })
    
    # ETL
    etl_data = next(e for e in etl_db if e['name'] == etl)
    stack_layers.append({
        'name': etl,
        'type': 'etl', 
        'thickness_nm': etl_data['thickness_typical_nm'],
        'properties': etl_data
    })
    
    # Absorber layers (ensure 'type' key is set for optical/display functions)
    for layer in layers:
        layer['type'] = 'absorber'  # Normalize key name
        stack_layers.append(layer)
    
    # HTL
    htl_data = next(h for h in htl_db if h['name'] == htl)
    stack_layers.append({
        'name': htl,
        'type': 'htl',
        'thickness_nm': htl_data['thickness_typical_nm'], 
        'properties': htl_data
    })
    
    # Bottom electrode
    bottom_electrode_data = next(e for e in electrodes_db if e['name'] == electrode_bottom)
    stack_layers.append({
        'name': electrode_bottom,
        'type': 'electrode',
        'thickness_nm': bottom_electrode_data['thickness_typical_nm'],
        'properties': bottom_electrode_data
    })
    
    progress_bar.progress(80)
    
    # 4. Quick optical calculation from database
    # Simplified - just estimate total absorption
    total_thickness = sum(layer.get('thickness_nm', 0) for layer in stack_layers)
    
    # Estimate performance metrics from database interpolation
    if track_code == 'B':
        avg_absorption = np.mean([layer['composition']['absorption_coeff_500nm'] 
                                for layer in layers])
        avg_refractive_index = np.mean([layer['composition']['n_550'] 
                                      for layer in layers])
    else:
        avg_absorption = np.mean([layer['properties'].get('absorption_coeff_500nm', 50000) 
                                for layer in layers])
        avg_refractive_index = np.mean([layer['properties'].get('n_550', 3.5) 
                                      for layer in layers])
    
    progress_bar.progress(100)
    time.sleep(0.5)  # Brief pause for UI
    progress_bar.empty()
    
    return {
        'stack': stack_layers,
        'optimal_bandgaps': optimal_bandgaps,
        'layers': layers,
        'performance': {
            'estimated_jsc': 15.2,  # mA/cm²
            'estimated_voc': sum(optimal_bandgaps) * 0.4,  # V
            'estimated_ff': 0.85,
            'estimated_pce': optimal_solution.get('pce', 28.0)
        },
        'optics': {
            'total_thickness_um': total_thickness / 1000,
            'avg_absorption_coeff': avg_absorption,
            'avg_refractive_index': avg_refractive_index
        }
    }

def calculate_stack_absorption_from_db(stack: List, wavelengths=None) -> np.ndarray:
    """
    Estimate absorption spectrum from database properties.
    Uses Elliott model for direct-gap perovskite absorption:
      - Continuum: α ∝ sqrt(E - Eg) for E > Eg
      - Excitonic enhancement near band edge
      - Urbach tail below Eg (Eu ~ 15-25 meV for perovskite)
      - High-energy saturation (α plateaus above ~2*Eg)
    """
    if wavelengths is None:
        wavelengths = np.linspace(300, 1200, 200)  # Higher resolution
    
    photon_energies = 1240.0 / wavelengths  # eV
    
    # Track remaining light (Beer-Lambert through stack)
    transmitted = np.ones_like(wavelengths, dtype=float)
    total_absorption = np.zeros_like(wavelengths)
    
    for layer in stack:
        if layer.get('type', 'other') == 'absorber':
            # Get material properties
            if 'composition' in layer:
                alpha_ref = layer['composition']['absorption_coeff_500nm']  # cm⁻¹ at 500nm
                bandgap = layer['composition']['Eg']
            else:
                alpha_ref = layer['properties'].get('absorption_coeff_500nm', 50000)
                bandgap = layer['properties']['bandgap']
            
            E_ref = 1240.0 / 500.0  # 2.48 eV (500nm reference)
            thickness_cm = layer.get('thickness_nm', 300) * 1e-7
            
            # Urbach energy (perovskite: 15-25 meV; wider gap → slightly larger)
            Eu = 0.015 + 0.005 * max(0, bandgap - 1.5)  # 15-25 meV
            
            # Build absorption coefficient spectrum
            alpha = np.zeros_like(photon_energies)
            
            above_gap = photon_energies > bandgap
            below_gap = ~above_gap
            
            # --- Above bandgap: Elliott model (continuum + excitonic) ---
            dE = np.maximum(photon_energies[above_gap] - bandgap, 1e-6)
            dE_ref = max(E_ref - bandgap, 0.1)
            
            # Continuum: α ∝ √(E - Eg), normalized to α_ref at 500nm
            alpha_continuum = alpha_ref * np.sqrt(dE / dE_ref)
            
            # Excitonic enhancement near band edge (Lorentzian peak)
            # Perovskite exciton binding energy ~10-50 meV
            Eb = 0.025  # 25 meV typical for lead halide perovskite
            exciton_enhancement = 1.0 + 2.0 * Eb / (dE + Eb)  # Sommerfeld factor approx
            
            alpha_above = alpha_continuum * exciton_enhancement
            
            # High-energy saturation: α plateaus (interband transitions saturate)
            saturation_energy = bandgap + 1.5  # ~1.5 eV above Eg
            sat_factor = np.where(
                photon_energies[above_gap] > saturation_energy,
                1.0 + 0.1 * np.log(1 + photon_energies[above_gap] - saturation_energy),
                1.0
            )
            alpha_above *= sat_factor
            
            alpha[above_gap] = alpha_above
            
            # --- Below bandgap: Urbach tail ---
            alpha[below_gap] = alpha_ref * np.sqrt(0.01 / max(dE_ref, 0.1)) * \
                               np.exp((photon_energies[below_gap] - bandgap) / Eu)
            
            # Beer-Lambert with remaining transmitted light
            layer_absorption = transmitted * (1 - np.exp(-alpha * thickness_cm))
            total_absorption += layer_absorption
            transmitted *= np.exp(-alpha * thickness_cm)
        
        elif layer.get('type', 'other') in ('etl', 'htl'):
            # Transport layers: slight parasitic absorption at short wavelengths
            thickness_cm = layer.get('thickness_nm', 50) * 1e-7
            # Approximate: weak absorption below 350nm for TiO2/SnO2, Spiro etc.
            parasitic_alpha = np.where(wavelengths < 380, 5000 * np.exp(-(wavelengths - 300) / 30), 100)
            parasitic = transmitted * (1 - np.exp(-parasitic_alpha * thickness_cm))
            total_absorption += parasitic * 0.0  # Don't count as useful absorption
            transmitted *= np.exp(-parasitic_alpha * thickness_cm)
        
        elif layer.get('type', 'other') == 'electrode':
            # Front electrode (ITO/FTO): ~10% reflection + parasitic absorption
            thickness_cm = layer.get('thickness_nm', 100) * 1e-7
            # ITO: high transparency 400-900nm, absorbs in UV and NIR
            ito_alpha = np.where(wavelengths < 350, 30000, 
                        np.where(wavelengths > 900, 5000 * ((wavelengths - 900) / 300)**2, 200))
            parasitic = transmitted * (1 - np.exp(-ito_alpha * thickness_cm))
            total_absorption += parasitic * 0.0
            transmitted *= np.exp(-ito_alpha * thickness_cm)
    
    return np.clip(total_absorption, 0, 1)

# =============================================================================
# TAB STRUCTURE 
# =============================================================================

# Always visible tabs (after step 1-5 selection)
if all([track, n_junctions, electrode_top, electrode_bottom, etl, htl]):
    
    tab_list = ["📋 재료 DB 탐색", "🧪 ABX₃ 조성 설계"]
    
    # Add Stage 1 tabs if completed
    if st.session_state.stage1_complete:
        tab_list.extend(["🏗️ 디바이스 구조", "📊 광학 프리뷰", "🎯 밴드갭 최적화"])
    
    # Add Stage 2 tabs if completed  
    if st.session_state.stage2_complete:
        tab_list.extend([
            "⚡ I-V 곡선", "🔗 계면 안정성", "🌡️ 환경 & 열화", 
            "⚡ 24시간 발전", "🎮 제어 전략", "💰 경제성",
            "📊 민감도 분석", "🏭 공정 레시피"
        ])
    
    tabs = st.tabs(tab_list)
    
    # =============================================================================
    # TAB 1: 재료 DB 탐색 (Material DB Explorer) 
    # =============================================================================
    
    with tabs[0]:
        st.header("📋 재료 데이터베이스 탐색")
        
        if track_code == 'B' and not perovskite_db.empty:
            st.subheader("🔬 ABX₃ 페로브스카이트 데이터베이스")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("총 조성 수", f"{len(perovskite_db):,}")
                st.metric("밴드갭 범위", f"{perovskite_db['Eg'].min():.2f} - {perovskite_db['Eg'].max():.2f} eV")
                st.metric("RT 안정상", f"{(perovskite_db['phase_stable_RT']).sum():,}")
                
            with col2:
                # Bandgap distribution
                fig_eg = px.histogram(
                    perovskite_db, x='Eg', nbins=50,
                    title="Bandgap Distribution",
                    template='plotly_white'
                )
                fig_eg.update_layout(height=300)
                st.plotly_chart(fig_eg, use_container_width=True)
            
            # Filters
            st.subheader("🎯 필터링")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                eg_range = st.slider(
                    "밴드갭 범위 [eV]",
                    float(perovskite_db['Eg'].min()), 
                    float(perovskite_db['Eg'].max()),
                    (1.2, 2.5)
                )
            
            with col2:
                min_stability = st.slider("최소 안정성 점수", 0.0, 10.0, 5.0)
                
            with col3:
                only_stable = st.checkbox("RT 안정상만", value=True)
            
            # Apply filters
            filtered_db = perovskite_db[
                (perovskite_db['Eg'] >= eg_range[0]) & 
                (perovskite_db['Eg'] <= eg_range[1]) &
                (perovskite_db['stability_score'] >= min_stability)
            ]
            
            if only_stable:
                filtered_db = filtered_db[filtered_db['phase_stable_RT'] == True]
            
            st.write(f"🔍 필터링 결과: {len(filtered_db):,} / {len(perovskite_db):,} 조성")
            
            # Display filtered results
            if len(filtered_db) > 0:
                display_cols = [
                    'A_MA', 'A_FA', 'A_Cs', 'B_Pb', 'B_Sn', 'X_I', 'X_Br', 'X_Cl',
                    'Eg', 'crystal_phase', 'stability_score', 'confidence', 'phase_stable_RT'
                ]
                st.dataframe(
                    filtered_db[display_cols].head(100),
                    use_container_width=True,
                    height=300
                )
                
        else:  # Track A
            st.subheader("🔬 Track A 재료 데이터베이스")
            
            track_a_df = pd.DataFrame(track_a_materials)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("총 재료 수", len(track_a_materials))
                st.metric("밴드갭 범위", f"{track_a_df['bandgap'].min():.2f} - {track_a_df['bandgap'].max():.2f} eV")
                
            with col2:
                fig_track_a = px.scatter(
                    track_a_df, x='bandgap', y='cost_per_cm2_usd',
                    size='stability_score', hover_name='name',
                    title="Bandgap vs Cost",
                    template='plotly_white'
                )
                fig_track_a.update_layout(height=300)
                st.plotly_chart(fig_track_a, use_container_width=True)
            
            st.dataframe(track_a_df, use_container_width=True)
    
    # =============================================================================
    # TAB 2: ABX₃ 조성 설계 (ABX₃ Composition Design)
    # =============================================================================
    
    with tabs[1]:
        st.header("🧪 ABX₃ 조성 설계")
        
        if track_code == 'B':
            # === N-Junction Optimal Composition Map ===
            st.subheader(f"🏗️ {n_junctions}-Junction 최적 조성 맵")
            
            # Get optimal bandgaps from pareto fronts
            pareto_key = f"B_{n_junctions}T"
            if pareto_key in pareto_fronts:
                opt_bgs = pareto_fronts[pareto_key][0]['bandgaps']
            else:
                opt_bgs = list(np.linspace(2.4, 1.1, n_junctions))
            
            st.write(f"**SQ 이론 기반 최적 밴드갭**: {' → '.join([f'{bg:.2f} eV' for bg in opt_bgs])}")
            
            # Find best composition for each layer
            layer_data = []
            for i, target_eg in enumerate(opt_bgs):
                best = find_best_composition(target_eg, perovskite_db)
                
                # Build composition formula
                a_parts = [f"{'MA' if k=='A_MA' else 'FA' if k=='A_FA' else 'Cs'}{best[k]:.0%}" 
                           for k in ['A_MA','A_FA','A_Cs'] if best.get(k, 0) > 0.05]
                b_parts = [f"{'Pb' if k=='B_Pb' else 'Sn'}{best[k]:.0%}" 
                           for k in ['B_Pb','B_Sn'] if best.get(k, 0) > 0.05]
                x_parts = [f"{'I' if k=='X_I' else 'Br' if k=='X_Br' else 'Cl'}{best[k]:.0%}" 
                           for k in ['X_I','X_Br','X_Cl'] if best.get(k, 0) > 0.05]
                formula = f"({'/'.join(a_parts)})({'/'.join(b_parts)})({'/'.join(x_parts)})₃"
                
                # Confidence badge
                conf = best.get('confidence', 1)
                badge = '★★★' if conf >= 3 else '★★' if conf >= 2 else '★'
                
                layer_data.append({
                    'Layer': f"Layer {i+1} ({'Top' if i==0 else 'Bottom' if i==n_junctions-1 else 'Mid'})",
                    'Target Eg [eV]': f"{target_eg:.2f}",
                    'Matched Eg [eV]': f"{best['Eg']:.3f}",
                    'ABX₃ 조성': formula,
                    '안정성': f"{best.get('stability_score', 0):.1f}/10",
                    '신뢰도': badge,
                    '결정상': best.get('crystal_phase', 'N/A'),
                })
            
            st.dataframe(pd.DataFrame(layer_data), use_container_width=True, hide_index=True)
            
            # Bandgap cascade bar chart
            fig_cascade = go.Figure()
            matched_egs = [float(d['Matched Eg [eV]']) for d in layer_data]
            fig_cascade.add_trace(go.Bar(
                x=[d['Layer'] for d in layer_data],
                y=matched_egs,
                marker_color=[f'hsl({240 - i*40}, 70%, 55%)' for i in range(n_junctions)],
                text=[d['ABX₃ 조성'] for d in layer_data],
                textposition='outside',
                textfont=dict(size=9),
            ))
            fig_cascade.update_layout(
                title=f"{n_junctions}-Junction 밴드갭 캐스케이드",
                yaxis_title="Bandgap [eV]",
                template='plotly_white',
                height=400,
                yaxis_range=[0, max(matched_egs) * 1.3],
            )
            st.plotly_chart(fig_cascade, use_container_width=True)
            
            st.divider()
            
            # === Single Target Bandgap Search (existing) ===
            st.subheader("🎯 Target Bandgap → Composition")
            
            target_eg = st.slider("목표 밴드갭 [eV]", 1.0, 3.0, 1.6, 0.05)
            
            # Find best compositions near target
            candidates = perovskite_db[
                (abs(perovskite_db['Eg'] - target_eg) < 0.1) &
                (perovskite_db['phase_stable_RT'] == True)
            ].copy()
            
            if len(candidates) > 0:
                # Score and rank: weighted by accuracy + stability + confidence
                candidates['accuracy'] = abs(candidates['Eg'] - target_eg)
                candidates['composite_score'] = (
                    (1 - candidates['accuracy'] / 0.1) * 0.3 +  # Eg accuracy
                    candidates['stability_score'] / 10 * 0.35 +  # Stability
                    candidates['confidence'] / 3 * 0.2 +         # Confidence
                    candidates.get('defect_tolerance', 0.5) * 0.15  # Defect tolerance
                )
                candidates = candidates.sort_values('composite_score', ascending=False)
                
                st.write(f"🎯 Found {len(candidates)} stable compositions within ±0.1 eV")
                
                # Diverse Top 3: pick candidates with different B-site or X-site compositions
                selected = []
                for _, row in candidates.iterrows():
                    if len(selected) >= 3:
                        break
                    # Check diversity: at least one major composition difference
                    is_diverse = True
                    for prev in selected:
                        # Require meaningful difference in at least one site
                        b_diff = abs(row['B_Pb'] - prev['B_Pb']) + abs(row['B_Sn'] - prev['B_Sn'])
                        x_diff = abs(row['X_I'] - prev['X_I']) + abs(row['X_Br'] - prev['X_Br']) + abs(row['X_Cl'] - prev['X_Cl'])
                        a_diff = abs(row['A_MA'] - prev['A_MA']) + abs(row['A_FA'] - prev['A_FA']) + abs(row['A_Cs'] - prev['A_Cs'])
                        if b_diff < 0.2 and x_diff < 0.2 and a_diff < 0.2:
                            is_diverse = False
                            break
                    if is_diverse:
                        selected.append(row)
                
                # Fallback: if diversity filter too strict, fill with top remaining
                if len(selected) < 3:
                    for _, row in candidates.iterrows():
                        if len(selected) >= 3:
                            break
                        if not any(row.name == s.name for s in selected):
                            selected.append(row)
                
                # Display
                for i, comp in enumerate(selected):
                    
                    # Build compact composition label
                    a_parts = [f"{'MA' if k=='A_MA' else 'FA' if k=='A_FA' else 'Cs' if k=='A_Cs' else 'Rb'}{comp[k]:.0%}" 
                               for k in ['A_MA','A_FA','A_Cs'] if comp.get(k, 0) > 0.05]
                    b_parts = [f"{'Pb' if k=='B_Pb' else 'Sn' if k=='B_Sn' else 'Ge'}{comp[k]:.0%}" 
                               for k in ['B_Pb','B_Sn'] if comp.get(k, 0) > 0.05]
                    x_parts = [f"{'I' if k=='X_I' else 'Br' if k=='X_Br' else 'Cl'}{comp[k]:.0%}" 
                               for k in ['X_I','X_Br','X_Cl'] if comp.get(k, 0) > 0.05]
                    comp_label = f"({'/'.join(a_parts)})({'/'.join(b_parts)})({'/'.join(x_parts)})₃"
                    
                    with st.expander(f"💎 Candidate {i+1}: Eg = {comp['Eg']:.3f} eV — {comp_label}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**A-site:**")
                            if comp['A_MA'] > 0: st.write(f"MA: {comp['A_MA']:.1%}")
                            if comp['A_FA'] > 0: st.write(f"FA: {comp['A_FA']:.1%}")
                            if comp['A_Cs'] > 0: st.write(f"Cs: {comp['A_Cs']:.1%}")
                            
                        with col2:
                            st.write("**B-site:**")
                            if comp['B_Pb'] > 0: st.write(f"Pb: {comp['B_Pb']:.1%}")
                            if comp['B_Sn'] > 0: st.write(f"Sn: {comp['B_Sn']:.1%}")
                            
                        with col3:
                            st.write("**X-site:**")
                            if comp['X_I'] > 0: st.write(f"I: {comp['X_I']:.1%}")
                            if comp['X_Br'] > 0: st.write(f"Br: {comp['X_Br']:.1%}")
                            if comp['X_Cl'] > 0: st.write(f"Cl: {comp['X_Cl']:.1%}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Stability", f"{comp['stability_score']:.1f}/10")
                        with col2:
                            phase_html = get_phase_badge(comp['crystal_phase'], 
                                                       comp['phase_transition_temp'])
                            st.markdown(phase_html, unsafe_allow_html=True)
                        with col3:
                            conf_html = get_confidence_badge(comp['confidence'])
                            st.markdown(conf_html, unsafe_allow_html=True)
                            
            else:
                st.warning("⚠️ No stable compositions found near target bandgap")
            
            # =============================================================================
            # 🔺 삼각 상태도 (TERNARY PHASE DIAGRAMS)
            # =============================================================================
            
            st.subheader("🔺 삼각 상태도 (Ternary Phase Diagrams)")
            st.markdown("*Explore bandgap landscape across composition space*")
            
            # Check if B_Ge column exists in the database
            has_ge = 'B_Ge' in perovskite_db.columns and (perovskite_db['B_Ge'] > 0).any()
            
            # Sidebar controls for filtering
            with st.sidebar.expander("🔺 Ternary Plot Controls"):
                st.markdown("**Fixed site compositions:**")
                
                # X-site ternary controls
                st.markdown("*X-site ternary (I-Br-Cl):*")
                fixed_a_for_x = st.selectbox("Fixed A-site", ["FA", "MA", "Cs"], index=0, key="x_fixed_a")
                fixed_b_for_x = st.selectbox("Fixed B-site", ["Pb", "Sn"], index=0, key="x_fixed_b")
                
                # A-site ternary controls  
                st.markdown("*A-site ternary (MA-FA-Cs):*")
                fixed_b_for_a = st.selectbox("Fixed B-site", ["Pb", "Sn"], index=0, key="a_fixed_b")
                fixed_x_for_a = st.selectbox("Fixed X-site", ["I", "Br", "Cl"], index=0, key="a_fixed_x")
                
                # B-site ternary controls
                st.markdown("*B-site ternary:*")
                fixed_a_for_b = st.selectbox("Fixed A-site", ["FA", "MA", "Cs"], index=0, key="b_fixed_a") 
                fixed_x_for_b = st.selectbox("Fixed X-site", ["I", "Br", "Cl"], index=0, key="b_fixed_x")
                
                # Plotting controls
                max_points = st.slider("Max points per plot", 500, 3000, 1500)
                colorscale = st.selectbox("Colorscale", ["RdYlBu_r", "Viridis", "Plasma"], index=0)
            
            # Helper function to filter and prepare ternary data
            def prepare_ternary_data(df, site_type, fixed_conditions, tolerance=0.1):
                """Filter database and prepare ternary plot data"""
                
                # Apply fixed site conditions
                filtered_df = df.copy()
                for col, target_val in fixed_conditions.items():
                    if col in filtered_df.columns:
                        filtered_df = filtered_df[abs(filtered_df[col] - target_val) <= tolerance]
                
                # Remove rows where the varying components sum to less than 0.8 (incomplete)
                if site_type == 'X':
                    valid_mask = (filtered_df['X_I'] + filtered_df['X_Br'] + filtered_df['X_Cl']) >= 0.8
                    filtered_df = filtered_df[valid_mask]
                elif site_type == 'A':
                    valid_mask = (filtered_df['A_MA'] + filtered_df['A_FA'] + filtered_df['A_Cs']) >= 0.8
                    filtered_df = filtered_df[valid_mask]
                elif site_type == 'B':
                    if has_ge:
                        valid_mask = (filtered_df['B_Pb'] + filtered_df['B_Sn'] + filtered_df['B_Ge']) >= 0.8
                    else:
                        valid_mask = (filtered_df['B_Pb'] + filtered_df['B_Sn']) >= 0.8
                    filtered_df = filtered_df[valid_mask]
                
                # Sample if too many points
                if len(filtered_df) > max_points:
                    filtered_df = filtered_df.sample(n=max_points, random_state=42)
                
                return filtered_df
            
            # Create the three ternary plots
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. X-site ternary (I-Br-Cl)
                st.markdown("**🟦 X-site: I - Br - Cl**")
                
                # Define fixed conditions
                fixed_a_val = 1.0 if fixed_a_for_x == "FA" else 0.0
                fixed_b_val = 1.0 if fixed_b_for_x == "Pb" else 0.0
                
                x_conditions = {
                    f'A_{fixed_a_for_x}': fixed_a_val,
                    f'B_{fixed_b_for_x}': fixed_b_val
                }
                
                x_data = prepare_ternary_data(perovskite_db, 'X', x_conditions)
                
                if len(x_data) > 0:
                    # Create hover text
                    hover_text = [
                        f"I:{row['X_I']:.1%}, Br:{row['X_Br']:.1%}, Cl:{row['X_Cl']:.1%}<br>" +
                        f"Eg: {row['Eg']:.3f} eV<br>" +
                        f"Stability: {row['stability_score']:.1f}/10<br>" +
                        f"Phase: {row['crystal_phase']}"
                        for _, row in x_data.iterrows()
                    ]
                    
                    # Map stability to marker symbols
                    symbols = ['circle' if stable else 'x' for stable in x_data['phase_stable_RT']]
                    
                    fig_x = go.Figure(go.Scatterternary(
                        a=x_data['X_I'],
                        b=x_data['X_Br'], 
                        c=x_data['X_Cl'],
                        mode='markers',
                        marker=dict(
                            color=x_data['Eg'],
                            colorscale=colorscale,
                            colorbar=dict(title='Eg [eV]', x=1.1),
                            size=6,
                            symbol=symbols,
                            line=dict(color='black', width=0.5)
                        ),
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        name=f"Fixed: {fixed_a_for_x}100%/{fixed_b_for_x}100%"
                    ))
                    
                    fig_x.update_layout(
                        ternary=dict(
                            aaxis=dict(title='I [%]'),
                            baxis=dict(title='Br [%]'),
                            caxis=dict(title='Cl [%]'),
                        ),
                        title=f'X-site Ternary: {fixed_a_for_x}/{fixed_b_for_x} Fixed<br><sub>● stable, × unstable</sub>',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_x, use_container_width=True)
                    st.caption(f"Showing {len(x_data)} compositions")
                else:
                    st.warning(f"No data found for {fixed_a_for_x}/{fixed_b_for_x} combination")
                
                # 2. A-site ternary (MA-FA-Cs)  
                st.markdown("**🟩 A-site: MA - FA - Cs**")
                
                fixed_b_val_a = 1.0 if fixed_b_for_a == "Pb" else 0.0
                fixed_x_val_a = 1.0 if fixed_x_for_a == "I" else 0.0 if fixed_x_for_a == "Br" else 0.0
                
                a_conditions = {
                    f'B_{fixed_b_for_a}': fixed_b_val_a,
                    f'X_{fixed_x_for_a}': fixed_x_val_a
                }
                
                a_data = prepare_ternary_data(perovskite_db, 'A', a_conditions)
                
                if len(a_data) > 0:
                    hover_text_a = [
                        f"MA:{row['A_MA']:.1%}, FA:{row['A_FA']:.1%}, Cs:{row['A_Cs']:.1%}<br>" +
                        f"Eg: {row['Eg']:.3f} eV<br>" +
                        f"Stability: {row['stability_score']:.1f}/10<br>" +
                        f"Phase: {row['crystal_phase']}"
                        for _, row in a_data.iterrows()
                    ]
                    
                    symbols_a = ['circle' if stable else 'x' for stable in a_data['phase_stable_RT']]
                    
                    fig_a = go.Figure(go.Scatterternary(
                        a=a_data['A_MA'],
                        b=a_data['A_FA'],
                        c=a_data['A_Cs'], 
                        mode='markers',
                        marker=dict(
                            color=a_data['Eg'],
                            colorscale=colorscale,
                            colorbar=dict(title='Eg [eV]', x=1.1),
                            size=6,
                            symbol=symbols_a,
                            line=dict(color='black', width=0.5)
                        ),
                        text=hover_text_a,
                        hovertemplate='%{text}<extra></extra>',
                        name=f"Fixed: {fixed_b_for_a}100%/{fixed_x_for_a}100%"
                    ))
                    
                    fig_a.update_layout(
                        ternary=dict(
                            aaxis=dict(title='MA [%]'),
                            baxis=dict(title='FA [%]'), 
                            caxis=dict(title='Cs [%]'),
                        ),
                        title=f'A-site Ternary: {fixed_b_for_a}/{fixed_x_for_a} Fixed<br><sub>● stable, × unstable</sub>',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_a, use_container_width=True)
                    st.caption(f"Showing {len(a_data)} compositions")
                else:
                    st.warning(f"No data found for {fixed_b_for_a}/{fixed_x_for_a} combination")
            
            with col2:
                # 3. B-site ternary (Pb-Sn-Ge or Pb-Sn binary)
                if has_ge:
                    st.markdown("**🟪 B-site: Pb - Sn - Ge**")
                    
                    fixed_a_val_b = 1.0 if fixed_a_for_b == "FA" else 0.0 if fixed_a_for_b == "MA" else 0.0
                    fixed_x_val_b = 1.0 if fixed_x_for_b == "I" else 0.0 if fixed_x_for_b == "Br" else 0.0
                    
                    b_conditions = {
                        f'A_{fixed_a_for_b}': fixed_a_val_b,
                        f'X_{fixed_x_for_b}': fixed_x_val_b
                    }
                    
                    b_data = prepare_ternary_data(perovskite_db, 'B', b_conditions)
                    
                    if len(b_data) > 0:
                        hover_text_b = [
                            f"Pb:{row['B_Pb']:.1%}, Sn:{row['B_Sn']:.1%}, Ge:{row['B_Ge']:.1%}<br>" +
                            f"Eg: {row['Eg']:.3f} eV<br>" +
                            f"Stability: {row['stability_score']:.1f}/10<br>" +
                            f"Phase: {row['crystal_phase']}"
                            for _, row in b_data.iterrows()
                        ]
                        
                        symbols_b = ['circle' if stable else 'x' for stable in b_data['phase_stable_RT']]
                        
                        fig_b = go.Figure(go.Scatterternary(
                            a=b_data['B_Pb'],
                            b=b_data['B_Sn'],
                            c=b_data['B_Ge'],
                            mode='markers',
                            marker=dict(
                                color=b_data['Eg'],
                                colorscale=colorscale,
                                colorbar=dict(title='Eg [eV]', x=1.1),
                                size=6,
                                symbol=symbols_b,
                                line=dict(color='black', width=0.5)
                            ),
                            text=hover_text_b,
                            hovertemplate='%{text}<extra></extra>',
                            name=f"Fixed: {fixed_a_for_b}100%/{fixed_x_for_b}100%"
                        ))
                        
                        fig_b.update_layout(
                            ternary=dict(
                                aaxis=dict(title='Pb [%]'),
                                baxis=dict(title='Sn [%]'),
                                caxis=dict(title='Ge [%]'),
                            ),
                            title=f'B-site Ternary: {fixed_a_for_b}/{fixed_x_for_b} Fixed<br><sub>● stable, × unstable</sub>',
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_b, use_container_width=True)
                        st.caption(f"Showing {len(b_data)} compositions")
                    else:
                        st.warning(f"No Ge data found for {fixed_a_for_b}/{fixed_x_for_b} combination")
                
                else:
                    # Binary Pb-Sn plot as backup
                    st.markdown("**🟪 B-site: Pb - Sn (Binary)**")
                    st.info("ℹ️ Ge data not available in database")
                    
                    fixed_a_val_b = 1.0 if fixed_a_for_b == "FA" else 0.0 if fixed_a_for_b == "MA" else 0.0
                    fixed_x_val_b = 1.0 if fixed_x_for_b == "I" else 0.0 if fixed_x_for_b == "Br" else 0.0
                    
                    b_conditions = {
                        f'A_{fixed_a_for_b}': fixed_a_val_b,
                        f'X_{fixed_x_for_b}': fixed_x_val_b
                    }
                    
                    # Filter for binary Pb-Sn
                    b_data_binary = prepare_ternary_data(perovskite_db, 'B', b_conditions)
                    b_data_binary = b_data_binary[
                        (b_data_binary['B_Pb'] + b_data_binary['B_Sn']) >= 0.95  # Nearly pure Pb-Sn
                    ]
                    
                    if len(b_data_binary) > 0:
                        # Create scatter plot for binary case
                        fig_b_binary = px.scatter(
                            b_data_binary, 
                            x='B_Pb', 
                            y='B_Sn',
                            color='Eg',
                            symbol='phase_stable_RT',
                            color_continuous_scale=colorscale,
                            title=f'B-site Binary: {fixed_a_for_b}/{fixed_x_for_b} Fixed',
                            labels={'B_Pb': 'Pb fraction', 'B_Sn': 'Sn fraction'},
                            template='plotly_white',
                            height=400
                        )
                        fig_b_binary.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
                        
                        st.plotly_chart(fig_b_binary, use_container_width=True)
                        st.caption(f"Showing {len(b_data_binary)} Pb-Sn compositions")
                    else:
                        st.warning(f"No Pb-Sn data found for {fixed_a_for_b}/{fixed_x_for_b} combination")
                
                # Bandgap statistics table
                st.markdown("**📊 Bandgap Statistics**")
                stats_data = []
                
                if len(x_data) > 0:
                    stats_data.append({
                        'Site': 'X (I-Br-Cl)',
                        'Points': len(x_data),
                        'Eg Range [eV]': f"{x_data['Eg'].min():.2f} - {x_data['Eg'].max():.2f}",
                        'Stable %': f"{(x_data['phase_stable_RT']).mean():.1%}"
                    })
                
                if len(a_data) > 0:
                    stats_data.append({
                        'Site': 'A (MA-FA-Cs)', 
                        'Points': len(a_data),
                        'Eg Range [eV]': f"{a_data['Eg'].min():.2f} - {a_data['Eg'].max():.2f}",
                        'Stable %': f"{(a_data['phase_stable_RT']).mean():.1%}"
                    })
                
                if has_ge and len(b_data) > 0:
                    stats_data.append({
                        'Site': 'B (Pb-Sn-Ge)',
                        'Points': len(b_data),
                        'Eg Range [eV]': f"{b_data['Eg'].min():.2f} - {b_data['Eg'].max():.2f}",
                        'Stable %': f"{(b_data['phase_stable_RT']).mean():.1%}"
                    })
                elif 'b_data_binary' in locals() and len(b_data_binary) > 0:
                    stats_data.append({
                        'Site': 'B (Pb-Sn)',
                        'Points': len(b_data_binary),
                        'Eg Range [eV]': f"{b_data_binary['Eg'].min():.2f} - {b_data_binary['Eg'].max():.2f}",
                        'Stable %': f"{(b_data_binary['phase_stable_RT']).mean():.1%}"
                    })
                
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
                
        else:
            st.info("📌 Track A uses established materials. See Tab 1 for available options.")
    
    # =============================================================================
    # STAGE 1 SIMULATION
    # =============================================================================
    
    if stage1_button:
        with st.spinner("🔄 Running Stage 1 simulation..."):
            result = run_stage1(
                track_code, n_junctions, electrode_top, electrode_bottom, 
                etl, htl, {
                    'temperature': temperature,
                    'humidity': humidity, 
                    'latitude': latitude,
                    'area': area_cm2
                }
            )
            
            # Store results in session state
            st.session_state.stage1_result = result
            st.session_state.stage1_complete = True
            
            st.success("✅ Stage 1 complete! New tabs are now available.")
            st.rerun()
    
    # =============================================================================
    # STAGE 1 RESULT TABS
    # =============================================================================
    
    if st.session_state.stage1_complete and len(tabs) >= 3:
        
        # TAB 3: 디바이스 구조 (Device Structure)
        with tabs[2]:
            st.header("🏗️ 디바이스 구조")
            
            if 'stage1_result' in st.session_state:
                result = st.session_state.stage1_result
                
                # Device cross-section visualization
                st.subheader("📐 단면도 (Cross-section)")
                
                # Create layer diagram
                layers_data = []
                y_pos = 0
                
                for layer in result['stack']:
                    thickness = layer.get('thickness_nm', 0) / 1000  # Convert to μm
                    layers_data.append({
                        'name': layer.get('name', layer.get('material', 'Unknown')),
                        'type': layer.get('type', 'other'),
                        'y_start': y_pos,
                        'y_end': y_pos + thickness,
                        'thickness': thickness
                    })
                    y_pos += thickness
                
                # Plot cross-section
                fig_cross = go.Figure()
                
                colors = {
                    'electrode': '#BDC3C7',
                    'etl': '#3498DB', 
                    'absorber': '#E74C3C',
                    'htl': '#9B59B6',
                    'other': '#95A5A6'
                }
                
                for layer in layers_data:
                    fig_cross.add_shape(
                        type="rect",
                        x0=0, x1=1,
                        y0=layer['y_start'], y1=layer['y_end'],
                        fillcolor=colors.get(layer.get('type', 'other'), colors['other']),
                        opacity=0.7,
                        line=dict(color="black", width=1)
                    )
                    
                    # Add label
                    fig_cross.add_annotation(
                        x=0.5, y=(layer['y_start'] + layer['y_end'])/2,
                        text=f"{layer.get('name', layer.get('material', 'Unknown'))}<br>{layer['thickness']:.1f} μm",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                fig_cross.update_layout(
                    title="Device Layer Stack",
                    xaxis_title="Width",
                    yaxis_title="Thickness [μm]",
                    template='plotly_white',
                    height=400,
                    xaxis_range=[0, 1],
                    yaxis_range=[0, y_pos]
                )
                
                st.plotly_chart(fig_cross, use_container_width=True)
                
                # Layer details table
                st.subheader("📋 레이어 상세 정보")
                
                layer_df = []
                for layer in result['stack']:
                    if layer.get('type') == 'absorber':
                        if 'composition' in layer:
                            # Perovskite layer
                            comp = layer['composition']
                            layer_df.append({
                                'Layer': layer.get('material', layer.get('name', layer.get('material', 'Unknown'))),
                                'Type': layer.get('type', 'absorber'),
                                'Thickness [nm]': layer.get('thickness_nm', 0),
                                'Bandgap [eV]': comp.get('Eg', 'N/A'),
                                'Phase': comp.get('crystal_phase', 'N/A'), 
                                'Stability': f"{comp.get('stability_score', 0):.1f}/10",
                                'Confidence': get_confidence_badge(comp.get('confidence', 1))
                            })
                        else:
                            # Track A material
                            props = layer.get('properties', {})
                            layer_df.append({
                                'Layer': layer.get('material', layer.get('name', layer.get('material', 'Unknown'))),
                                'Type': layer.get('type', 'absorber'),
                                'Thickness [nm]': layer.get('thickness_nm', 0),
                                'Bandgap [eV]': props.get('bandgap', layer.get('bandgap', 'N/A')),
                                'Phase': props.get('type', 'N/A'),
                                'Stability': f"{props.get('stability_score', 8):.1f}/10",
                                'Confidence': get_confidence_badge(3)
                            })
                    else:
                        # Contact layers
                        props = layer.get('properties', {})
                        layer_df.append({
                            'Layer': layer.get('name', layer.get('material', 'Unknown')),
                            'Type': layer.get('type', 'Contact'),
                            'Thickness [nm]': layer.get('thickness_nm', 0),
                            'Bandgap [eV]': props.get('eg', 'N/A'),
                            'Phase': 'Contact',
                            'Stability': f"{props.get('stability', 3)}/3",
                            'Confidence': '★★★'
                        })
                
                st.dataframe(pd.DataFrame(layer_df), use_container_width=True)
        
        # TAB 4: 광학 프리뷰 (Optical Preview)
        with tabs[3]:
            st.header("📊 광학 프리뷰")
            
            if 'stage1_result' in st.session_state:
                result = st.session_state.stage1_result
                
                # Generate absorption spectrum
                wavelengths = np.linspace(300, 1200, 100)
                absorption = calculate_stack_absorption_from_db(result['stack'], wavelengths)
                
                # Plot absorption spectrum
                fig_abs = go.Figure()
                
                fig_abs.add_trace(go.Scatter(
                    x=wavelengths,
                    y=absorption * 100,
                    mode='lines',
                    name='Total Absorption',
                    line=dict(color='#E74C3C', width=3)
                ))
                
                fig_abs.update_layout(
                    title="Absorption Spectrum (Stage 1 Estimate)",
                    xaxis_title="Wavelength [nm]",
                    yaxis_title="Absorption [%]",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_abs, use_container_width=True)
                
                # Performance metrics
                st.subheader("⚡ 성능 추정 (Performance Estimate)")
                
                col1, col2, col3, col4 = st.columns(4)
                perf = result['performance']
                
                with col1:
                    st.metric("Jsc [mA/cm²]", f"{perf['estimated_jsc']:.1f}")
                with col2:
                    st.metric("Voc [V]", f"{perf['estimated_voc']:.2f}")
                with col3:
                    st.metric("FF", f"{perf['estimated_ff']:.3f}")
                with col4:
                    st.metric("PCE [%]", f"{perf['estimated_pce']:.1f}")
                
                st.info("💡 이는 1차 추정치입니다. 정확한 값은 2차 시뮬레이션에서 제공됩니다.")
        
        # TAB 5: 밴드갭 최적화 (Bandgap Optimization)
        with tabs[4]:
            st.header("🎯 밴드갭 최적화")
            
            if 'stage1_result' in st.session_state:
                result = st.session_state.stage1_result
                
                st.subheader("🔋 최적 밴드갭 분포")
                
                # Bandgap cascade plot
                bandgaps = result['optimal_bandgaps']
                
                fig_cascade = go.Figure()
                
                x_pos = range(len(bandgaps))
                fig_cascade.add_trace(go.Bar(
                    x=[f"Layer {i+1}" for i in x_pos],
                    y=bandgaps,
                    marker_color='#3498DB',
                    name='Bandgap'
                ))
                
                fig_cascade.update_layout(
                    title=f"Optimized Bandgap Cascade ({n_junctions} Junctions)",
                    xaxis_title="Layer (Top → Bottom)",
                    yaxis_title="Bandgap [eV]",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_cascade, use_container_width=True)
                
                # Current matching analysis
                st.subheader("⚡ 전류 매칭 분석")
                
                # Estimate current density for each junction
                # Simplified calculation based on bandgap
                estimated_currents = []
                for bg in bandgaps:
                    # Rough approximation: higher bandgap → lower current
                    jsc = 25 - (bg - 1.0) * 8  # Empirical scaling
                    estimated_currents.append(max(jsc, 5))
                
                current_df = pd.DataFrame({
                    'Junction': [f"J{i+1}" for i in range(len(bandgaps))],
                    'Bandgap [eV]': bandgaps,
                    'Estimated Jsc [mA/cm²]': estimated_currents
                })
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(current_df, use_container_width=True)
                    
                with col2:
                    current_mismatch = (max(estimated_currents) - min(estimated_currents)) / np.mean(estimated_currents)
                    st.metric("Current Mismatch", f"{current_mismatch:.1%}")
                    
                    if current_mismatch < 0.05:
                        st.success("✅ 우수한 전류 매칭")
                    elif current_mismatch < 0.10:
                        st.warning("⚠️ 보통 전류 매칭")
                    else:
                        st.error("❌ 전류 매칭 개선 필요")

# Run Stage 2 simulation
if stage2_button:
    st.info("🚀 Stage 2 simulation would run here with full physics calculations...")
    st.info("Implementation: detailed I-V, stability, economics, etc.")
    st.session_state.stage2_complete = True

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("━━━━━━━━━━━━━━━━━━━━━━")
st.sidebar.markdown("**V3 Features:**")
st.sidebar.markdown("✅ Pre-computed 47K+ ABX₃ DB")
st.sidebar.markdown("✅ 2-stage workflow") 
st.sidebar.markdown("✅ Korean + English UI")
st.sidebar.markdown("✅ Crystal phase warnings")
st.sidebar.markdown("✅ Confidence scoring")

if not perovskite_db.empty:
    st.sidebar.success(f"🔬 DB: {len(perovskite_db):,} compositions loaded")
else:
    st.sidebar.error("❌ Database not found")