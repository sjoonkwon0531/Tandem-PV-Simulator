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
    
    # Absorber layers
    for layer in layers:
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
    total_thickness = sum(layer['thickness_nm'] for layer in stack_layers)
    
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
    """Estimate absorption spectrum from database properties"""
    if wavelengths is None:
        wavelengths = np.linspace(300, 1200, 100)
    
    total_absorption = np.zeros_like(wavelengths)
    
    for layer in stack:
        if layer['type'] == 'absorber':
            # Get absorption coefficient from database
            if 'composition' in layer:
                alpha_500 = layer['composition']['absorption_coeff_500nm']
                bandgap = layer['composition']['Eg']
            else:
                alpha_500 = layer['properties'].get('absorption_coeff_500nm', 50000)
                bandgap = layer['properties']['bandgap']
            
            # Estimate spectrum assuming Urbach tail + direct gap
            photon_energies = 1240 / wavelengths  # eV
            
            # Above bandgap: direct absorption
            alpha = np.where(
                photon_energies > bandgap,
                alpha_500 * np.sqrt(np.maximum(photon_energies - bandgap, 0) / 0.48),  # Scale to 500nm
                alpha_500 * 0.001 * np.exp((photon_energies - bandgap) / 0.015)  # Urbach tail
            )
            
            # Beer-Lambert absorption
            thickness_cm = layer['thickness_nm'] * 1e-7
            layer_absorption = 1 - np.exp(-alpha * thickness_cm)
            
            total_absorption += layer_absorption * 0.9**len(stack)  # Rough interference
    
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
    
    # Always add Dynamic Control tab
    tab_list.append("🔄 Dynamic Control")
    tab_list.append("🤖 AI/ML Control")
    
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
            st.subheader("🎯 Target Bandgap → Composition")
            
            target_eg = st.slider("목표 밴드갭 [eV]", 1.0, 3.0, 1.6, 0.05)
            
            # Find best compositions near target
            candidates = perovskite_db[
                (abs(perovskite_db['Eg'] - target_eg) < 0.1) &
                (perovskite_db['phase_stable_RT'] == True)
            ].copy()
            
            if len(candidates) > 0:
                # Score and rank
                candidates['accuracy'] = abs(candidates['Eg'] - target_eg)
                candidates = candidates.sort_values(['accuracy', 'stability_score'], 
                                                   ascending=[True, False])
                
                st.write(f"🎯 Found {len(candidates)} stable compositions within ±0.1 eV")
                
                # Top 3 candidates
                for i in range(min(3, len(candidates))):
                    comp = candidates.iloc[i]
                    
                    with st.expander(f"💎 Candidate {i+1}: Eg = {comp['Eg']:.3f} eV"):
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
            st.experimental_rerun()
    
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
                    thickness = layer['thickness_nm'] / 1000  # Convert to μm
                    layers_data.append({
                        'name': layer['name'],
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
                        fillcolor=colors.get(layer['type'], colors['other']),
                        opacity=0.7,
                        line=dict(color="black", width=1)
                    )
                    
                    # Add label
                    fig_cross.add_annotation(
                        x=0.5, y=(layer['y_start'] + layer['y_end'])/2,
                        text=f"{layer['name']}<br>{layer['thickness']:.1f} μm",
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
                                'Layer': layer.get('material', layer['name']),
                                'Type': layer['type'],
                                'Thickness [nm]': layer['thickness_nm'],
                                'Bandgap [eV]': comp.get('Eg', 'N/A'),
                                'Phase': comp.get('crystal_phase', 'N/A'), 
                                'Stability': f"{comp.get('stability_score', 0):.1f}/10",
                                'Confidence': get_confidence_badge(comp.get('confidence', 1))
                            })
                        else:
                            # Track A material
                            props = layer.get('properties', {})
                            layer_df.append({
                                'Layer': layer.get('material', layer['name']),
                                'Type': layer['type'],
                                'Thickness [nm]': layer['thickness_nm'],
                                'Bandgap [eV]': props.get('bandgap', layer.get('bandgap', 'N/A')),
                                'Phase': props.get('type', 'N/A'),
                                'Stability': f"{props.get('stability_score', 8):.1f}/10",
                                'Confidence': get_confidence_badge(3)
                            })
                    else:
                        # Contact layers
                        props = layer.get('properties', {})
                        layer_df.append({
                            'Layer': layer['name'],
                            'Type': layer['type'],
                            'Thickness [nm]': layer['thickness_nm'],
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
# TAB: Dynamic Control (🔄)
# =============================================================================

with tabs[-1]:
    st.header("🔄 Dynamic PV Output Control")
    st.markdown("""
    PV-FET 일체형 능동 출력 제어 시뮬레이션.
    게이트 전압(V_G)으로 태양전지의 운영점을 실시간 제어합니다.
    """)
    
    try:
        from engines.dynamic_iv import DynamicIVEngine
        from engines.load_matching import LoadMatchingEngine
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Perovskite PV")
            pv_bg = st.slider("Bandgap (eV)", 1.2, 2.0, 1.55, 0.05, key="dyn_bg")
            pv_thick = st.slider("Thickness (nm)", 200, 1000, 500, 50, key="dyn_thick")
        with col2:
            st.subheader("IGZO FET")
            fet_vth = st.slider("V_th (V)", 0.1, 2.0, 0.5, 0.1, key="dyn_vth")
            fet_wl = st.slider("W/L ratio", 10, 500, 100, 10, key="dyn_wl")
        with col3:
            st.subheader("Control")
            vg_control = st.slider("V_G (V)", 0.0, 5.0, 3.0, 0.1, key="dyn_vg")
            irradiance = st.slider("Irradiance (suns)", 0.1, 1.5, 1.0, 0.1, key="dyn_irr")
        
        # Create engine
        engine = DynamicIVEngine(
            {'bandgap': pv_bg, 'thickness': pv_thick},
            {'V_th': fet_vth, 'W_L': fet_wl},
            {}
        )
        
        # Operating point
        op = engine.operating_point(vg_control, G=irradiance)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("V_op", f"{op['V_op']:.3f} V")
        m2.metric("I_op", f"{op['I_op']:.2f} mA/cm²")
        m3.metric("P_out", f"{op['P_out']:.2f} mW/cm²")
        m4.metric("η", f"{op['eta']*100:.1f}%")
        
        # I-V curve with operating point
        V_arr = np.linspace(0, 1.3, 100)
        I_arr = engine.static_iv(V_arr, G=irradiance)
        
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(x=V_arr, y=I_arr, mode='lines', name='I-V Curve'))
        # Load line
        g_ch = engine._channel_conductance(vg_control)
        if g_ch > 0:
            V_load = np.linspace(0, 1.3, 100)
            I_load = g_ch * V_load
            fig_iv.add_trace(go.Scatter(x=V_load, y=I_load, mode='lines',
                                       name=f'Load Line (V_G={vg_control:.1f}V)',
                                       line=dict(dash='dash', color='red')))
        fig_iv.add_trace(go.Scatter(x=[op['V_op']], y=[op['I_op']],
                                   mode='markers', name='Operating Point',
                                   marker=dict(size=12, color='green', symbol='star')))
        fig_iv.update_layout(title="I-V Curve with Operating Point",
                           xaxis_title="Voltage (V)", yaxis_title="Current (mA/cm²)",
                           height=400)
        st.plotly_chart(fig_iv, use_container_width=True)
        
        # Power envelope
        st.subheader("⚡ Power Envelope (V_G Sweep)")
        env = engine.power_envelope(G=irradiance)
        
        col_env1, col_env2 = st.columns(2)
        with col_env1:
            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(x=env['V_G'], y=env['P'], mode='lines',
                                        name='Power', line=dict(color='orange')))
            fig_env.add_vline(x=vg_control, line_dash="dash", line_color="green",
                            annotation_text=f"V_G={vg_control:.1f}V")
            fig_env.update_layout(title="Power vs Gate Voltage",
                                xaxis_title="V_G (V)", yaxis_title="P_out (mW/cm²)",
                                height=350)
            st.plotly_chart(fig_env, use_container_width=True)
        
        with col_env2:
            st.metric("P_min", f"{env['P_min']:.2f} mW/cm²")
            st.metric("P_max", f"{env['P_max']:.2f} mW/cm²")
            st.metric("Dynamic Range", f"{env['dynamic_range']:.1f}x")
            st.metric("V_G optimal", f"{env['V_G_opt']:.2f} V")
            st.metric("τ_ion", f"{engine.ion_time_constant_ms:.1f} ms")
            st.metric("τ_RC", f"{engine.rc_time_constant_us:.1f} μs")
        
        # 24h simulation
        st.subheader("📊 24시간 PV-AIDC 매칭 시뮬레이션")
        if st.button("Run 24h Simulation", key="run_24h"):
            with st.spinner("Simulating..."):
                lm = LoadMatchingEngine(engine)
                load_data = lm.generate_aidc_load(hours=24, gpu_count=100, dt=60)
                pv_no_ctrl = lm.generate_pv_output(hours=24, dt=60, pv_capacity_kW=load_data['P_peak_kW']*0.3)
                pv_with_ctrl = lm.generate_pv_output(hours=24, dt=60, pv_capacity_kW=load_data['P_peak_kW']*0.3,
                                                     with_control=True, target_load=load_data['load_kW'])
                
                match_no = lm.match_analysis(pv_no_ctrl['pv_kW'], load_data['load_kW'], dt=60)
                match_yes = lm.match_analysis(pv_with_ctrl['pv_kW'], load_data['load_kW'], dt=60)
                hess = lm.hess_reduction(match_yes, match_no)
                
                hours_arr = load_data['time_s'] / 3600
                fig_24h = go.Figure()
                fig_24h.add_trace(go.Scatter(x=hours_arr, y=load_data['load_kW'],
                                           name='AIDC Load', line=dict(color='red')))
                fig_24h.add_trace(go.Scatter(x=hours_arr, y=pv_no_ctrl['pv_kW'],
                                           name='PV (no control)', line=dict(color='blue', dash='dot')))
                fig_24h.add_trace(go.Scatter(x=hours_arr, y=pv_with_ctrl['pv_kW'],
                                           name='PV (active control)', line=dict(color='green')))
                fig_24h.update_layout(title="24h PV Output vs AIDC Load",
                                    xaxis_title="Hour", yaxis_title="Power (kW)", height=400)
                st.plotly_chart(fig_24h, use_container_width=True)
                
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("HESS Capacity Reduction", f"{hess['capacity_reduction_pct']:.1f}%")
                h2.metric("Cycling Reduction", f"{hess['cycling_reduction_pct']:.1f}%")
                h3.metric("Lifetime Extension", f"{hess['lifetime_extension_factor']:.1f}x")
                h4.metric("Cost Saving", f"${hess['cost_saving_usd_per_year']:.0f}/yr")
    
        # ── Ion Dynamics Sub-section ──────────────────────────────
        st.markdown("---")
        st.subheader("🔬 이온 Drift-Diffusion (1D 정밀 모델)")
        st.markdown("""
        페로브스카이트 내부 I⁻ 이온의 1D drift-diffusion을 풀어
        I-V 히스테리시스와 이온 분포를 시각화합니다.
        """)

        try:
            from engines.ion_dynamics import IonDynamicsEngine

            col_ion1, col_ion2 = st.columns(2)
            with col_ion1:
                sweep_rate = st.slider("Sweep rate (V/s)", 0.01, 10.0, 1.0,
                                       key="ion_sweep")
                ion_temp = st.slider("Temperature (K)", 260, 360, 300, 5,
                                      key="ion_temp")
            with col_ion2:
                ion_thick = st.slider("Layer thickness (nm)", 200, 1000, 500, 50,
                                       key="ion_thick")
                ion_grid = st.slider("Grid points", 30, 200, 80, 10,
                                      key="ion_grid")

            if st.button("Run Ion Dynamics", key="run_ion"):
                with st.spinner("Solving drift-diffusion..."):
                    ion_eng = IonDynamicsEngine(
                        layer_thickness_nm=ion_thick,
                        grid_points=ion_grid,
                    )
                    hyst = ion_eng.hysteresis_iv(
                        V_sweep_rate=sweep_rate, G=irradiance, T=ion_temp
                    )

                    # Hysteresis I-V
                    fig_hyst = go.Figure()
                    fig_hyst.add_trace(go.Scatter(
                        x=hyst['V_forward'], y=hyst['I_forward'],
                        mode='lines', name='Forward'))
                    fig_hyst.add_trace(go.Scatter(
                        x=hyst['V_reverse'], y=hyst['I_reverse'],
                        mode='lines', name='Reverse',
                        line=dict(dash='dash')))
                    fig_hyst.update_layout(
                        title=f"I-V Hysteresis (sweep {sweep_rate} V/s, HI={hyst['HI']:.3f})",
                        xaxis_title="Voltage (V)",
                        yaxis_title="Current (mA/cm²)",
                        height=400)
                    st.plotly_chart(fig_hyst, use_container_width=True)

                    h1, h2, h3 = st.columns(3)
                    h1.metric("PCE Forward", f"{hyst['PCE_forward']:.1f}%")
                    h2.metric("PCE Reverse", f"{hyst['PCE_reverse']:.1f}%")
                    h3.metric("Hysteresis Index", f"{hyst['HI']:.3f}")

                    # Steady-state ion profile
                    ss = ion_eng.steady_state(V_applied=0.8, G=irradiance, T=ion_temp)
                    fig_ion = go.Figure()
                    for name, prof in ss['n_ion'].items():
                        fig_ion.add_trace(go.Scatter(
                            x=ion_eng.x * 1e7,  # cm → nm
                            y=prof,
                            mode='lines', name=name))
                    fig_ion.update_layout(
                        title="Steady-State Ion Distribution",
                        xaxis_title="Position (nm)",
                        yaxis_title="Concentration (cm⁻³)",
                        height=350)
                    st.plotly_chart(fig_ion, use_container_width=True)

                    # Response time
                    resp = ion_eng.response_time(V_step=0.5, G=irradiance,
                                                  T=ion_temp, duration_s=0.5,
                                                  n_steps=200)
                    r1, r2 = st.columns(2)
                    r1.metric("τ_ion (63%)", f"{resp['tau_ion']*1e3:.1f} ms")
                    r2.metric("τ_90", f"{resp['tau_90']*1e3:.1f} ms")

        except Exception as e:
            st.warning(f"Ion dynamics not available: {e}")

        # ── Interface Charge (Phase 1-3) ──────────────────────────
        st.markdown("---")
        st.subheader("🔬 계면 전하 플러싱 (μs 스케일)")
        st.markdown("ETL/Perovskite 및 Perovskite/HTL 계면 트랩 동역학과 임피던스 분광법 시뮬레이션.")

        try:
            from engines.interface_charge import InterfaceChargeEngine

            col_if1, col_if2 = st.columns(2)
            with col_if1:
                if_r_contact = st.slider("R_contact (Ω·cm²)", 1.0, 20.0, 5.0, 0.5, key="if_rc")
                if_v_pulse = st.slider("Flush pulse V (V)", 0.5, 5.0, 2.0, 0.5, key="if_vp")
            with col_if2:
                if_pulse_w = st.slider("Pulse width (μs)", 10, 500, 100, 10, key="if_pw")
                if_temp = st.slider("Temperature (K)", 260, 360, 300, 5, key="if_temp")

            if st.button("Run Interface Charge Analysis", key="run_iface"):
                with st.spinner("Computing interface dynamics..."):
                    iface_eng = InterfaceChargeEngine(
                        interface_params={'R_contact': if_r_contact, 'v_thermal': 1e7}
                    )

                    # 1. Impedance spectrum (Nyquist plot)
                    freq = np.logspace(-1, 6, 200)
                    z_data = iface_eng.impedance_spectrum(0.5, freq, if_temp)

                    col_z1, col_z2 = st.columns(2)
                    with col_z1:
                        fig_nyq = go.Figure()
                        fig_nyq.add_trace(go.Scatter(
                            x=z_data['Z_real'], y=-z_data['Z_imag'],
                            mode='lines+markers', marker=dict(size=3),
                            name='Nyquist'))
                        fig_nyq.update_layout(
                            title="Impedance Nyquist Plot",
                            xaxis_title="Z' (Ω·cm²)",
                            yaxis_title="-Z'' (Ω·cm²)",
                            height=350,
                            yaxis=dict(scaleanchor="x", scaleratio=1))
                        st.plotly_chart(fig_nyq, use_container_width=True)

                    with col_z2:
                        fig_bode = go.Figure()
                        fig_bode.add_trace(go.Scatter(
                            x=freq, y=z_data['Z_magnitude'],
                            mode='lines', name='|Z|'))
                        fig_bode.update_layout(
                            title="Bode Plot (Magnitude)",
                            xaxis_title="Frequency (Hz)",
                            yaxis_title="|Z| (Ω·cm²)",
                            xaxis_type="log", yaxis_type="log",
                            height=350)
                        st.plotly_chart(fig_bode, use_container_width=True)

                    # 2. Flush response
                    flush = iface_eng.flush_response(
                        V_pulse=if_v_pulse, pulse_width=if_pulse_w * 1e-6,
                        T=if_temp, dt=1e-6)

                    fig_flush = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                             subplot_titles=["Trapped Charge", "Recombination Current"])
                    fig_flush.add_trace(go.Scatter(
                        x=flush['time'] * 1e6, y=flush['n_t_etl'],
                        name='ETL traps'), row=1, col=1)
                    fig_flush.add_trace(go.Scatter(
                        x=flush['time'] * 1e6, y=flush['n_t_htl'],
                        name='HTL traps'), row=1, col=1)
                    fig_flush.add_trace(go.Scatter(
                        x=flush['time'] * 1e6, y=flush['J_rec'],
                        name='J_rec'), row=2, col=1)
                    fig_flush.add_vrect(x0=0, x1=if_pulse_w,
                                       fillcolor="yellow", opacity=0.2,
                                       annotation_text="Pulse", row=1, col=1)
                    fig_flush.update_layout(height=500, title="Flush Pulse Response")
                    fig_flush.update_xaxes(title_text="Time (μs)", row=2, col=1)
                    st.plotly_chart(fig_flush, use_container_width=True)

                    st.metric("τ_RC (ETL)", f"{iface_eng.tau_rc_etl*1e6:.1f} μs")

        except Exception as e:
            st.warning(f"Interface charge engine not available: {e}")

        # ── Multiscale Integration ────────────────────────────────
        st.markdown("---")
        st.subheader("🌐 멀티스케일 통합 시뮬레이션")
        st.markdown("""
        3개 시간 스케일 (초/ms/μs) 통합 Operator Splitting 시뮬레이션.
        """)

        try:
            from engines.multiscale_control import MultiscaleControlEngine

            col_ms1, col_ms2 = st.columns(2)
            with col_ms1:
                ms_duration = st.slider("Duration (s)", 5, 60, 20, 5, key="ms_dur")
                ms_vg = st.slider("V_G (V)", 0.0, 5.0, 3.0, 0.5, key="ms_vg")
            with col_ms2:
                ms_irr = st.slider("Irradiance (suns)", 0.1, 1.5, 1.0, 0.1, key="ms_irr")

            if st.button("Run Multiscale Simulation", key="run_ms"):
                with st.spinner("Running 3-scale simulation..."):
                    ms_eng = MultiscaleControlEngine(
                        pv_params={'bandgap': pv_bg, 'thickness': pv_thick},
                        fet_params={'V_th': fet_vth, 'W_L': fet_wl},
                        interface_params={'R_contact': 5.0},
                    )
                    N_ms = ms_duration
                    V_G_ms = np.full(N_ms, ms_vg)
                    G_ms = np.full(N_ms, ms_irr)
                    T_ms = np.full(N_ms, 300.0)

                    res_ms = ms_eng.simulate_multiscale(
                        V_G_ms, G_ms, T_ms, ms_duration,
                        n_medium_per_coarse=5, n_fine_per_medium=5)

                    # Stacked timeseries
                    fig_ms = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                          subplot_titles=["P_out (mW/cm²)", "ΔV_ion (V)", "J_rec (mA/cm²)"])
                    fig_ms.add_trace(go.Scatter(x=res_ms.time_s, y=res_ms.P_out,
                                               name='P_out'), row=1, col=1)
                    fig_ms.add_trace(go.Scatter(x=res_ms.time_s, y=res_ms.dV_ion,
                                               name='ΔV_ion'), row=2, col=1)
                    fig_ms.add_trace(go.Scatter(x=res_ms.time_s, y=res_ms.J_rec_interface,
                                               name='J_rec'), row=3, col=1)
                    fig_ms.update_layout(height=600, title="Multiscale Simulation Results")
                    fig_ms.update_xaxes(title_text="Time (s)", row=3, col=1)
                    st.plotly_chart(fig_ms, use_container_width=True)

                    # Performance summary
                    summary = ms_eng.performance_summary()
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("P_max", f"{summary['P_max_mW_cm2']:.2f} mW/cm²")
                    s2.metric("Dynamic Range", f"{summary['dynamic_range']:.1f}x")
                    s3.metric("τ_interface", f"{summary['tau_interface_us']:.1f} μs")
                    s4.metric("τ_ion", f"{summary['tau_ion_ms']:.1f} ms")

                    # Energy balance
                    eb = summary['energy_balance']
                    st.markdown("**에너지 밸런스:**")
                    st.json(eb)

            # ── Comparative Analysis ──────────────────────────────
            st.markdown("---")
            st.subheader("📊 비교 분석: 제어 단계별 누적 효과")

            if st.button("Run Comparative Analysis", key="run_compare"):
                with st.spinner("Comparing control strategies..."):
                    N_cmp = 20
                    G_cmp = np.full(N_cmp, 1.0)
                    T_cmp = np.full(N_cmp, 300.0)

                    # Scenario 1: No control (V_G max, no ion/interface)
                    eng_base = DynamicIVEngine(
                        {'bandgap': pv_bg, 'thickness': pv_thick},
                        {'V_th': fet_vth, 'W_L': fet_wl}, {})
                    P_nocontrol = np.array([eng_base.operating_point(5.0, g, 300.0)['P_out'] for g in G_cmp])

                    # Scenario 2: FET only
                    P_fet = np.array([eng_base.operating_point(3.0, g, 300.0)['P_out'] for g in G_cmp])

                    # Scenario 3: FET + Ion
                    ms_eng2 = MultiscaleControlEngine(
                        pv_params={'bandgap': pv_bg, 'thickness': pv_thick},
                        fet_params={'V_th': fet_vth, 'W_L': fet_wl},
                        interface_params={'R_contact': 5.0},
                    )
                    res_fi = ms_eng2.simulate_multiscale(
                        np.full(N_cmp, 3.0), G_cmp, T_cmp, 20.0,
                        n_medium_per_coarse=5, n_fine_per_medium=3)
                    P_fet_ion = res_fi.P_out

                    # Scenario 4: FET + Ion + Interface (full)
                    res_full = ms_eng2.simulate_multiscale(
                        np.full(N_cmp, 3.0), G_cmp, T_cmp, 20.0,
                        n_medium_per_coarse=5, n_fine_per_medium=5)
                    P_full = res_full.P_out

                    t_cmp = np.arange(N_cmp)
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(x=t_cmp, y=P_nocontrol, name='No Control (MPPT)', line=dict(dash='dot')))
                    fig_cmp.add_trace(go.Scatter(x=t_cmp, y=P_fet, name='FET Only'))
                    fig_cmp.add_trace(go.Scatter(x=t_cmp, y=P_fet_ion, name='FET + Ion'))
                    fig_cmp.add_trace(go.Scatter(x=t_cmp, y=P_full, name='FET + Ion + Interface'))
                    fig_cmp.update_layout(title="Control Strategy Comparison",
                                        xaxis_title="Time step", yaxis_title="P_out (mW/cm²)",
                                        height=400)
                    st.plotly_chart(fig_cmp, use_container_width=True)

        except Exception as e:
            st.warning(f"Multiscale engine not available: {e}")

    except Exception as e:
        st.error(f"Dynamic Control engine error: {e}")
        import traceback
        st.code(traceback.format_exc())

    # =============================================================================
    # TAB: 🤖 AI/ML Control
    # =============================================================================
    ai_tab_idx = len(tab_list) - 1  # last tab
    with tabs[ai_tab_idx]:
        st.header("🤖 AI/ML 기반 제어 (Phase 2)")
        st.markdown("물리 모델의 빠른 근사(Surrogate) + ML 기반 최적 제어")

        ai_sub = st.selectbox("분석 모드", [
            "📈 Surrogate 성능 비교",
            "🎮 ML Controller 시뮬레이션",
            "🔬 소재 스크리닝"
        ])

        if ai_sub == "📈 Surrogate 성능 비교":
            st.subheader("Surrogate vs Full Simulation")
            n_test = st.slider("테스트 샘플 수", 20, 200, 60)

            if st.button("🚀 Surrogate 학습 & 비교", key="train_surrogate"):
                with st.spinner("Phase 1 엔진으로 학습 데이터 생성 중..."):
                    try:
                        from engines.surrogate_model import PhysicsSurrogate
                        from engines.multiscale_control import MultiscaleControlEngine

                        engine = MultiscaleControlEngine()
                        surrogate = PhysicsSurrogate()

                        data = surrogate.generate_training_data(engine, n_samples=n_test)
                        metrics = surrogate.train_steady_state(data)
                        report = surrogate.accuracy_report(data)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("P_out R²", f"{report['P_out']['R2']:.4f}")
                        col2.metric("η R²", f"{report['eta']['R2']:.4f}")
                        col3.metric("V_op R²", f"{report['V_op']['R2']:.4f}")

                        # Scatter plot: true vs predicted
                        for target in ['P_out', 'eta']:
                            y_true = data[target]
                            y_pred = np.array([
                                surrogate.predict_steady(data['inputs'][i, 0],
                                                         data['inputs'][i, 1],
                                                         data['inputs'][i, 2])[target]
                                for i in range(len(y_true))
                            ])

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                                                     name=target, marker=dict(size=5)))
                            rng = [min(y_true.min(), y_pred.min()),
                                   max(y_true.max(), y_pred.max())]
                            fig.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                                     name='Perfect', line=dict(dash='dash')))
                            fig.update_layout(title=f"{target}: Full Sim vs Surrogate",
                                              xaxis_title="Full Simulation",
                                              yaxis_title="Surrogate Prediction")
                            st.plotly_chart(fig, use_container_width=True)

                        st.success("✅ Surrogate 학습 완료!")

                    except Exception as e:
                        st.error(f"오류: {e}")

        elif ai_sub == "🎮 ML Controller 시뮬레이션":
            st.subheader("ML 기반 실시간 V_G 제어")

            if st.button("🧠 Controller 학습 & 시뮬레이션", key="train_controller"):
                with st.spinner("Surrogate 학습 → Controller 학습 중..."):
                    try:
                        from engines.surrogate_model import PhysicsSurrogate
                        from engines.ml_controller import MLController
                        from engines.multiscale_control import MultiscaleControlEngine

                        engine = MultiscaleControlEngine()
                        surrogate = PhysicsSurrogate()
                        data = surrogate.generate_training_data(engine, n_samples=60)
                        surrogate.train_steady_state(data)

                        ctrl = MLController(hidden_dims=[64, 32, 16])
                        episodes = ctrl.generate_training_episodes(surrogate, n_episodes=50, steps_per_episode=48)
                        losses = ctrl.train(episodes, epochs=80, lr=0.003)

                        # Loss curve
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
                        fig_loss.update_layout(title="학습 손실 곡선", xaxis_title="Epoch", yaxis_title="MSE Loss")
                        st.plotly_chart(fig_loss, use_container_width=True)

                        # 24h simulation
                        hours = np.linspace(0, 24, 48)
                        G_profile = np.maximum(0, np.sin(np.pi * hours / 24))
                        T_profile = 300 + 10 * np.sin(np.pi * (hours - 6) / 12)
                        Load_profile = 10 + 5 * np.sin(2 * np.pi * hours / 24)

                        V_G_pred = []
                        P_pred = []
                        for i in range(48):
                            state = np.array([G_profile[i], T_profile[i], Load_profile[i], 0.5, 10])
                            vg = ctrl.predict(state)
                            V_G_pred.append(vg)
                            p = surrogate.predict_steady(vg, max(G_profile[i], 0.01), T_profile[i])
                            P_pred.append(p['P_out'])

                        fig_ctrl = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                 subplot_titles=("V_G 제어 출력", "발전량 vs 부하"))
                        fig_ctrl.add_trace(go.Scatter(x=hours, y=V_G_pred, name='V_G (ML)'), row=1, col=1)
                        fig_ctrl.add_trace(go.Scatter(x=hours, y=P_pred, name='P_pv'), row=2, col=1)
                        fig_ctrl.add_trace(go.Scatter(x=hours, y=Load_profile, name='Load',
                                                      line=dict(dash='dash')), row=2, col=1)
                        fig_ctrl.update_layout(height=500, title="24시간 ML 제어 시뮬레이션")
                        st.plotly_chart(fig_ctrl, use_container_width=True)

                        st.success("✅ ML Controller 시뮬레이션 완료!")
                    except Exception as e:
                        st.error(f"오류: {e}")

        elif ai_sub == "🔬 소재 스크리닝":
            st.subheader("동적 제어 최적 ABX₃ 소재 검색")

            col1, col2 = st.columns(2)
            tau_min = col1.number_input("τ_ion 최소 (ms)", value=1.0)
            tau_max = col2.number_input("τ_ion 최대 (ms)", value=100.0)
            stab_thresh = st.slider("최소 안정성 점수", 0.0, 10.0, 4.0)
            eg_range = st.slider("밴드갭 범위 (eV)", 0.5, 3.5, (1.1, 1.8))
            n_cand = st.slider("후보 수", 50, 500, 200)

            if st.button("🔍 스크리닝 실행", key="screen_materials"):
                with st.spinner("소재 스크리닝 중..."):
                    try:
                        from engines.material_predictor import MaterialPredictor

                        mp = MaterialPredictor()
                        results = mp.screen_for_dynamic_control(
                            target_tau_range=(tau_min, tau_max),
                            stability_threshold=stab_thresh,
                            Eg_range=eg_range,
                            n_candidates=n_cand,
                        )

                        if results:
                            rows = []
                            for r in results[:20]:
                                p = r['properties']
                                c = r['composition']
                                name = " / ".join(f"{k.split('_')[1]}:{v:.2f}"
                                                  for k, v in c.items() if v > 0.05)
                                rows.append({
                                    '조성': name,
                                    'Eg (eV)': f"{p['Eg']:.2f}",
                                    'μ_e': f"{p['mu_e']:.1f}",
                                    'τ_ion (ms)': f"{p['tau_ion']:.1f}",
                                    '안정성': f"{p['stability']:.1f}",
                                })

                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                            st.success(f"✅ {len(results)}개 후보 발견 (상위 20개 표시)")
                        else:
                            st.warning("조건에 맞는 후보가 없습니다. 조건을 완화해보세요.")
                    except Exception as e:
                        st.error(f"오류: {e}")

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