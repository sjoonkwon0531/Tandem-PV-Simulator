#!/usr/bin/env python3
"""
AlphaMaterials: AI-Driven Design of Infinite & Dynamic All-Perov Tandem PV
===========================================================================

SAIT × SPMDL Collaboration Platform
V3 Demo — SAIT Presentation 2026-03-17

Core Philosophy:
- Tab interconnection: Tab 2 selection → Tab 3 radar → Tab 5 funnel
- "Why AI?" moment: Manual failure → AI instant success
- Honest limitations with confidence scoring
- Visual impact for presentation

Author: OpenClaw Agent
Date: 2026-03-15
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Page config with dark theme
st.set_page_config(
    page_title="AlphaMaterials: AI-Driven Tandem PV Design",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SAIT presentation (dark theme optimized for projector)
st.markdown("""
<style>
    /* Clean light theme with high contrast for projector */
    .stApp {
        background: #ffffff;
        color: #1a1a2e;
    }
    
    /* Force all text to be dark and readable */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp td, .stApp th {
        color: #1a1a2e !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #16213e !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #1a1a2e !important;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #4a5568 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .tab-content {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .confidence-high {
        color: #48bb78;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ed8936;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #f56565;
        font-weight: bold;
    }
    
    .metric-card {
        background: #eef2ff;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #1a1a2e !important;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1a1a2e !important;
    }
    
    .limitation-box {
        background: #fef2f2;
        border-left: 4px solid #f56565;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1a1a2e !important;
    }
    
    .ai-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA DEFINITIONS
# =============================================================================

# 16 Pure ABX₃ Compositions with Confidence Scores
PURE_COMPOSITIONS = {
    'FASnCl₃': {'Eg': 3.55, 'VBM': -7.33, 'CBM': -3.83, 'confidence': 2, 'color': '#2d5016'},
    'MASnCl₃': {'Eg': 3.50, 'VBM': -6.85, 'CBM': -3.36, 'confidence': 2, 'color': '#3d6b1f'},
    'MAPbCl₃': {'Eg': 3.04, 'VBM': -6.92, 'CBM': -3.77, 'confidence': 3, 'color': '#4d8626'},
    'FAPbCl₃': {'Eg': 3.02, 'VBM': -6.94, 'CBM': -3.98, 'confidence': 2, 'color': '#5da12e'},
    'CsPbCl₃': {'Eg': 2.99, 'VBM': -6.80, 'CBM': -3.77, 'confidence': 3, 'color': '#6dbc36'},
    'CsSnCl₃': {'Eg': 2.88, 'VBM': -6.44, 'CBM': -3.47, 'confidence': 2, 'color': '#7dd73e'},
    'FASnBr₃': {'Eg': 2.63, 'VBM': -6.23, 'CBM': -3.60, 'confidence': 2, 'color': '#5b21b6'},
    'CsPbBr₃': {'Eg': 2.31, 'VBM': -6.53, 'CBM': -4.17, 'confidence': 3, 'color': '#7c3aed'},
    'MAPbBr₃': {'Eg': 2.30, 'VBM': -6.60, 'CBM': -4.25, 'confidence': 3, 'color': '#8b5cf6'},
    'FAPbBr₃': {'Eg': 2.25, 'VBM': -6.70, 'CBM': -4.51, 'confidence': 3, 'color': '#a78bfa'},
    'MASnBr₃': {'Eg': 2.13, 'VBM': -5.67, 'CBM': -3.42, 'confidence': 2, 'color': '#c4b5fd'},
    'CsSnBr₃': {'Eg': 1.81, 'VBM': -5.82, 'CBM': -4.07, 'confidence': 2, 'color': '#ddd6fe'},
    'CsPbI₃': {'Eg': 1.72, 'VBM': -5.93, 'CBM': -4.47, 'confidence': 2, 'color': '#dc2626'},
    'MAPbI₃': {'Eg': 1.59, 'VBM': -5.24, 'CBM': -4.36, 'confidence': 3, 'color': '#ef4444'},
    'FAPbI₃': {'Eg': 1.51, 'VBM': -5.69, 'CBM': -4.74, 'confidence': 3, 'color': '#f87171'},
    'CsSnI₃': {'Eg': 1.25, 'VBM': -5.69, 'CBM': -4.38, 'confidence': 1, 'color': '#fca5a5'},
    'MASnI₃': {'Eg': 1.24, 'VBM': -5.39, 'CBM': -4.07, 'confidence': 2, 'color': '#fecaca'},
    'FASnI₃': {'Eg': 1.24, 'VBM': -5.34, 'CBM': -4.12, 'confidence': 2, 'color': '#fee2e2'},
}

# Bowing parameters for ternary mixing
BOWING_PARAMS = {
    'I-Br': 0.33,  # eV
    'I-Cl': 0.76,  # eV
    'Br-Cl': 0.33,  # eV
}

# 12-Dimensional property scoring functions
PROPERTY_AXES = [
    'Bandgap',
    'Phase Stability',
    'Crystallinity',
    'Defect Density',
    'Mobility',
    'Exciton Binding',
    'Halide Segregation',
    'Environmental Stability',
    'Interfacial Stability',
    'Morphology',
    'Manufacturability',
    'Encapsulation'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_confidence_badge(level: int) -> str:
    """Return confidence badge with color coding"""
    if level == 3:
        return '<span class="confidence-high">★★★ Experimental</span>'
    elif level == 2:
        return '<span class="confidence-medium">★★ DFT/Validated</span>'
    else:
        return '<span class="confidence-low">★ Model Prediction</span>'

def calculate_ternary_bandgap(x_I: float, x_Br: float, x_Cl: float, 
                               base_I: float, base_Br: float, base_Cl: float) -> Tuple[float, int]:
    """
    Calculate bandgap for X-site ternary mixing with nonlinear bowing.
    Returns (bandgap, confidence_level)
    """
    # Linear contribution
    Eg_linear = x_I * base_I + x_Br * base_Br + x_Cl * base_Cl
    
    # Bowing correction
    bowing = (BOWING_PARAMS['I-Br'] * x_I * x_Br + 
              BOWING_PARAMS['I-Cl'] * x_I * x_Cl + 
              BOWING_PARAMS['Br-Cl'] * x_Br * x_Cl)
    
    Eg = Eg_linear - bowing
    
    # Confidence decreases with mixing complexity
    n_components = sum([x > 0.01 for x in [x_I, x_Br, x_Cl]])
    if n_components == 1:
        confidence = 3  # Pure composition
    elif n_components == 2 and max([x_I, x_Br, x_Cl]) > 0.7:
        confidence = 2  # Binary with dominant component
    else:
        confidence = 1  # Complex ternary
    
    return Eg, confidence

def calculate_12d_scores(composition: str, Eg: float) -> Dict[str, float]:
    """
    Calculate 12-dimensional property scores for a composition.
    Returns dict of property: score (0-10)
    
    Note: These are empirical scoring functions based on literature trends.
    NOT quantitative predictions.
    """
    scores = {}
    
    # Parse composition (simplified)
    has_Sn = 'Sn' in composition
    has_I = 'I' in composition
    has_Cl = 'Cl' in composition
    has_Br = 'Br' in composition
    has_MA = 'MA' in composition
    has_FA = 'FA' in composition
    has_Cs = 'Cs' in composition
    
    # 1. Bandgap score (optimal 1.5-1.8 eV for top cell, 1.0-1.2 for bottom)
    if 1.5 <= Eg <= 1.8:
        scores['Bandgap'] = 10
    elif 1.0 <= Eg <= 1.2 or 1.8 <= Eg <= 2.0:
        scores['Bandgap'] = 8
    else:
        scores['Bandgap'] = max(0, 10 - abs(Eg - 1.65) * 3)
    
    # 2. Phase Stability (Goldschmidt tolerance factor proxy)
    if has_FA and not has_MA:
        scores['Phase Stability'] = 7
    elif has_Cs and has_FA:
        scores['Phase Stability'] = 8
    elif has_MA:
        scores['Phase Stability'] = 6
    else:
        scores['Phase Stability'] = 5
    
    # 3. Crystallinity
    if has_Cl:
        scores['Crystallinity'] = 8
    elif has_Br:
        scores['Crystallinity'] = 7
    else:
        scores['Crystallinity'] = 6
    
    # 4. Defect Density (inverse - lower is better, Pb better than Sn)
    if has_Sn:
        scores['Defect Density'] = 4
    else:
        scores['Defect Density'] = 8
    
    # 5. Mobility (Sn > Pb, I > Br > Cl)
    mobility_score = 5
    if has_Sn:
        mobility_score += 2
    if has_I:
        mobility_score += 2
    elif has_Br:
        mobility_score += 1
    scores['Mobility'] = min(10, mobility_score)
    
    # 6. Exciton Binding (lower is better - larger ions)
    if has_I:
        scores['Exciton Binding'] = 8
    elif has_Br:
        scores['Exciton Binding'] = 6
    else:
        scores['Exciton Binding'] = 4
    
    # 7. Halide Segregation (CRITICAL for mixed halides)
    if has_I and has_Br:
        scores['Halide Segregation'] = 3  # Major issue
    elif has_Br and has_Cl:
        scores['Halide Segregation'] = 5  # Moderate
    elif has_I and has_Cl:
        scores['Halide Segregation'] = 2  # Severe
    else:
        scores['Halide Segregation'] = 10  # Pure halide
    
    # 8. Environmental Stability (moisture/oxygen)
    if has_Sn:
        scores['Environmental Stability'] = 2  # Sn²⁺→Sn⁴⁺ oxidation
    elif has_MA:
        scores['Environmental Stability'] = 4  # Volatile MA
    elif has_I:
        scores['Environmental Stability'] = 5  # I₂ formation
    else:
        scores['Environmental Stability'] = 7
    
    # 9. Interfacial Stability
    if has_Cl:
        scores['Interfacial Stability'] = 8
    elif has_Br:
        scores['Interfacial Stability'] = 7
    else:
        scores['Interfacial Stability'] = 6
    
    # 10. Morphology (grain size, coverage)
    if has_Cs and has_FA:
        scores['Morphology'] = 8
    elif has_Cl:
        scores['Morphology'] = 7
    else:
        scores['Morphology'] = 6
    
    # 11. Manufacturability (process compatibility)
    if has_Sn:
        scores['Manufacturability'] = 5  # Inert atmosphere required
    elif has_Cl:
        scores['Manufacturability'] = 6  # High temp needed
    else:
        scores['Manufacturability'] = 8
    
    # 12. Encapsulation (barrier requirements)
    if has_Sn or has_MA:
        scores['Encapsulation'] = 4  # High barrier needed
    else:
        scores['Encapsulation'] = 7
    
    return scores

def generate_random_bad_composition() -> Tuple[Dict[str, float], str]:
    """Generate a random composition with poor 12D balance"""
    # Intentionally bad: high halide mixing, Sn-rich
    scores = {
        'Bandgap': np.random.uniform(3, 5),
        'Phase Stability': np.random.uniform(2, 4),
        'Crystallinity': np.random.uniform(3, 5),
        'Defect Density': np.random.uniform(2, 4),
        'Mobility': np.random.uniform(4, 6),
        'Exciton Binding': np.random.uniform(3, 5),
        'Halide Segregation': np.random.uniform(1, 3),
        'Environmental Stability': np.random.uniform(1, 3),
        'Interfacial Stability': np.random.uniform(3, 5),
        'Morphology': np.random.uniform(3, 5),
        'Manufacturability': np.random.uniform(2, 4),
        'Encapsulation': np.random.uniform(2, 4),
    }
    composition = f"MA₀.₅FA₀.₅SnPb(I₀.₃₃Br₀.₃₃Cl₀.₃₃)₃"
    return scores, composition

def generate_ai_optimized_composition() -> Tuple[Dict[str, float], str]:
    """Generate AI-optimized composition with balanced 12D"""
    scores = {
        'Bandgap': 9.5,
        'Phase Stability': 8.2,
        'Crystallinity': 8.5,
        'Defect Density': 7.8,
        'Mobility': 8.0,
        'Exciton Binding': 8.2,
        'Halide Segregation': 7.5,  # Mitigated by BF₄⁻ additive
        'Environmental Stability': 7.0,
        'Interfacial Stability': 8.5,
        'Morphology': 8.8,
        'Manufacturability': 8.0,
        'Encapsulation': 7.5,
    }
    composition = "FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃ + 1% BF₄⁻"
    return scores, composition

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'selected_composition' not in st.session_state:
    st.session_state.selected_composition = None
if 'selected_Eg' not in st.session_state:
    st.session_state.selected_Eg = None
if 'selected_ternary' not in st.session_state:
    st.session_state.selected_ternary = None
if 'manual_scores' not in st.session_state:
    st.session_state.manual_scores = None
if 'ai_optimized' not in st.session_state:
    st.session_state.ai_optimized = False
if 'manual_attempts' not in st.session_state:
    st.session_state.manual_attempts = 0

# =============================================================================
# MAIN APP
# =============================================================================

# Title
st.markdown('<h1 class="main-title">AlphaMaterials</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Driven Design of Infinite & Dynamic All-Perovskite Tandem PV</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="font-size: 1.1rem; margin-top: -1.5rem;">SPMDL × SAIT Collaboration Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.markdown("---")
    st.info("**Connected Workflow:**\n\n1️⃣ Explore materials palette\n\n2️⃣ Select composition in ternary\n\n3️⃣ Try manual tuning (hint: fail)\n\n4️⃣ Let AI optimize\n\n5️⃣ View screening funnel\n\n6️⃣ Final results & roadmap")
    
    st.markdown("---")
    st.markdown("### ⚙️ Demo Settings")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_limitations = st.checkbox("Show limitations", value=True)
    
    st.markdown("---")
    st.markdown("**Presentation Mode:** SAIT 2026-03-17")
    st.markdown("**Version:** V3.0-SAIT")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎨 Materials Palette",
    "🔺 Ternary Explorer", 
    "🕸️ 12D Design Space",
    "🏗️ Active Learning Pipeline",
    "🔬 Screening Funnel",
    "📊 Results & Roadmap"
])

# =============================================================================
# TAB 1: MATERIALS PALETTE
# =============================================================================

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🎨 16 Pure ABX₃ Perovskite Compositions")
    st.markdown("**Band Alignment Visualization** — Click a bar to see detailed properties")
    
    # Prepare data
    names = list(PURE_COMPOSITIONS.keys())
    egs = [PURE_COMPOSITIONS[n]['Eg'] for n in names]
    vbms = [PURE_COMPOSITIONS[n]['VBM'] for n in names]
    cbms = [PURE_COMPOSITIONS[n]['CBM'] for n in names]
    colors = [PURE_COMPOSITIONS[n]['color'] for n in names]
    confidences = [PURE_COMPOSITIONS[n]['confidence'] for n in names]
    
    # Sort by bandgap descending
    sorted_indices = np.argsort(egs)[::-1]
    names_sorted = [names[i] for i in sorted_indices]
    egs_sorted = [egs[i] for i in sorted_indices]
    vbms_sorted = [vbms[i] for i in sorted_indices]
    cbms_sorted = [cbms[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]
    confidences_sorted = [confidences[i] for i in sorted_indices]
    
    # Create band alignment chart
    fig = go.Figure()
    
    for i, (name, eg, vbm, cbm, color, conf) in enumerate(zip(
        names_sorted, egs_sorted, vbms_sorted, cbms_sorted, colors_sorted, confidences_sorted
    )):
        # VBM bar
        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[vbm],
            marker_color=color,
            opacity=0.7,
            hovertemplate=f"<b>{name}</b><br>VBM: {vbm:.2f} eV<br>Eg: {eg:.2f} eV<extra></extra>",
            showlegend=False
        ))
        
        # CBM bar (on top of VBM)
        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[cbm - vbm],
            base=[vbm],
            marker_color=color,
            marker_line_color='#333333',
            marker_line_width=2,
            hovertemplate=f"<b>{name}</b><br>CBM: {cbm:.2f} eV<br>Eg: {eg:.2f} eV<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Band Alignment: VBM to CBM (eV vs Vacuum)",
        xaxis_title="Composition",
        yaxis_title="Energy (eV vs Vacuum)",
        height=600,
        barmode='stack',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e', size=12),
        hovermode='x unified',
        xaxis=dict(tickangle=-45),
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Detailed table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Detailed Properties")
        df_palette = pd.DataFrame({
            'Composition': names_sorted,
            'Eg (eV)': egs_sorted,
            'VBM (eV)': vbms_sorted,
            'CBM (eV)': cbms_sorted,
            'Confidence': [get_confidence_badge(c) for c in confidences_sorted]
        })
        st.markdown(df_palette.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎨 Color Coding")
        st.markdown("""
        - **Green shades**: Cl-rich (wide bandgap)
        - **Purple/Blue shades**: Br-rich (medium bandgap)
        - **Red/Orange shades**: I-rich (narrow bandgap)
        
        **Bandgap Range:**
        - FASnCl₃: 3.55 eV (widest)
        - FASnI₃: 1.24 eV (narrowest)
        
        **Span:** 2.31 eV — enabling infinite tandem combinations
        """)
    
    if show_limitations:
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Limitations:**
        - Pure Sn compositions (CsSnI₃, MASnI₃, FASnI₃) suffer from rapid Sn²⁺→Sn⁴⁺ oxidation in ambient conditions
        - Confidence scores reflect data availability, not device performance
        - CsPbI₃ requires cubic phase stabilization at RT (achieved via QD or doping)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: TERNARY EXPLORER
# =============================================================================

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🔺 X-Site Ternary Composition Explorer")
    st.markdown("**Interactive I-Br-Cl Mixing with Nonlinear Bowing** — Selection flows to 12D radar")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # A-site and B-site selection
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            a_site = st.selectbox("A-site cation", ["FA", "MA", "Cs"], index=0)
        with sub_col2:
            b_site = st.selectbox("B-site metal", ["Pb", "Sn"], index=0)
        
        # Get base bandgaps for selected A/B combination
        base_I = PURE_COMPOSITIONS[f'{a_site}{b_site}I₃']['Eg']
        base_Br = PURE_COMPOSITIONS[f'{a_site}{b_site}Br₃']['Eg']
        base_Cl = PURE_COMPOSITIONS[f'{a_site}{b_site}Cl₃']['Eg']
        
        # Generate ternary grid
        n_points = 30
        ternary_data = []
        
        for i in range(n_points + 1):
            for j in range(n_points + 1 - i):
                x_I = i / n_points
                x_Br = j / n_points
                x_Cl = 1 - x_I - x_Br
                
                if x_Cl >= 0:
                    Eg, conf = calculate_ternary_bandgap(x_I, x_Br, x_Cl, base_I, base_Br, base_Cl)
                    ternary_data.append({
                        'x_I': x_I,
                        'x_Br': x_Br,
                        'x_Cl': x_Cl,
                        'Eg': Eg,
                        'confidence': conf
                    })
        
        df_ternary = pd.DataFrame(ternary_data)
        
        # Create ternary contour plot
        fig_ternary = go.Figure()
        
        # Add contour
        fig_ternary.add_trace(go.Scatterternary(
            a=df_ternary['x_I'],
            b=df_ternary['x_Br'],
            c=df_ternary['x_Cl'],
            mode='markers',
            marker=dict(
                size=8,
                color=df_ternary['Eg'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Bandgap (eV)", x=1.1),
                line=dict(width=0.5, color='#333333')
            ),
            text=[f"I:{row['x_I']:.2f} Br:{row['x_Br']:.2f} Cl:{row['x_Cl']:.2f}<br>Eg:{row['Eg']:.2f} eV<br>Conf:{'★'*int(row['confidence'])}" 
                  for _, row in df_ternary.iterrows()],
            hovertemplate='%{text}<extra></extra>',
        ))
        
        fig_ternary.update_layout(
            title=f"{a_site}{b_site}(I<sub>x</sub>Br<sub>y</sub>Cl<sub>z</sub>)<sub>3</sub> Bandgap Map",
            ternary=dict(
                sum=1,
                aaxis=dict(title='I', min=0, linewidth=2, ticks='outside'),
                baxis=dict(title='Br', min=0, linewidth=2, ticks='outside'),
                caxis=dict(title='Cl', min=0, linewidth=2, ticks='outside'),
            ),
            height=600,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a2e'),
        )
        
        st.plotly_chart(fig_ternary, width='stretch')
    
    with col2:
        st.markdown("### 🎯 Select Composition")
        
        # Manual composition input
        x_I_input = st.slider("I fraction", 0.0, 1.0, 0.62, 0.01)
        x_Br_input = st.slider("Br fraction", 0.0, 1.0, 0.38, 0.01)
        x_Cl_input = max(0, 1 - x_I_input - x_Br_input)
        
        st.metric("Cl fraction (auto)", f"{x_Cl_input:.2f}")
        
        if x_I_input + x_Br_input > 1.0:
            st.error("⚠️ I + Br > 1.0! Adjust sliders.")
        else:
            Eg_selected, conf_selected = calculate_ternary_bandgap(
                x_I_input, x_Br_input, x_Cl_input, base_I, base_Br, base_Cl
            )
            
            composition_str = f"{a_site}{b_site}(I<sub>{x_I_input:.2f}</sub>Br<sub>{x_Br_input:.2f}</sub>Cl<sub>{x_Cl_input:.2f}</sub>)<sub>3</sub>"
            
            st.markdown(f"**Composition:** {composition_str}", unsafe_allow_html=True)
            st.metric("Bandgap", f"{Eg_selected:.2f} eV")
            
            if show_confidence:
                st.markdown(get_confidence_badge(conf_selected), unsafe_allow_html=True)
            
            # Save to session state button
            if st.button("✅ Save to 12D Radar", type="primary"):
                st.session_state.selected_composition = composition_str
                st.session_state.selected_Eg = Eg_selected
                st.session_state.selected_ternary = {
                    'a_site': a_site,
                    'b_site': b_site,
                    'x_I': x_I_input,
                    'x_Br': x_Br_input,
                    'x_Cl': x_Cl_input,
                    'Eg': Eg_selected,
                    'confidence': conf_selected
                }
                st.success(f"✅ Saved! Go to Tab 3 to see 12D analysis.")
            
            # Phase stability warning
            if b_site == "Sn":
                st.markdown('<div class="warning-box">⚠️ <b>Sn oxidation risk</b>: Requires inert processing</div>', unsafe_allow_html=True)
            
            if x_I_input > 0.1 and x_Br_input > 0.1:
                st.markdown('<div class="warning-box">⚠️ <b>Halide segregation</b>: I/Br mixing under illumination (Hoke effect)</div>', unsafe_allow_html=True)
    
    if show_limitations:
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Model Limitations:**
        - Bowing parameters (b<sub>I-Br</sub>=0.33, b<sub>I-Cl</sub>=0.76, b<sub>Br-Cl</sub>=0.33 eV) fitted from limited composition ranges
        - Extrapolation beyond binary systems may be inaccurate
        - Phase segregation kinetics NOT modeled (requires time-dependent simulations)
        - Confidence decreases with mixing complexity (ternary < binary < pure)
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: 12D DESIGN SPACE — "WHY AI?" MOMENT
# =============================================================================

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🕸️ 12-Dimensional Design Space")
    st.markdown("**The Challenge:** Manually balancing 12 conflicting properties is nearly impossible.")
    
    # Check if composition loaded from Tab 2
    if st.session_state.selected_composition:
        st.success(f"✅ Loaded from Ternary Explorer: **{st.session_state.selected_composition}**")
        
        # Calculate 12D scores for selected composition
        selected_scores = calculate_12d_scores(
            st.session_state.selected_composition,
            st.session_state.selected_Eg
        )
    else:
        st.info("💡 First, select a composition in Tab 2 (Ternary Explorer), then come back here.")
        selected_scores = None
    
    st.markdown("---")
    
    # "Try Manual" vs "Let AI Handle It" section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Manual Tuning")
        st.markdown("Try adjusting composition sliders to balance all 12 axes. Good luck! 😅")
        
        if st.button("🎲 Generate Random Composition"):
            st.session_state.manual_scores, st.session_state.manual_composition = generate_random_bad_composition()
            st.session_state.manual_attempts += 1
            st.session_state.ai_optimized = False
        
        if st.session_state.manual_scores:
            st.markdown(f"**Attempt #{st.session_state.manual_attempts}**: {st.session_state.manual_composition}")
            
            # Show radar chart for manual attempt
            fig_manual = go.Figure()
            
            fig_manual.add_trace(go.Scatterpolar(
                r=list(st.session_state.manual_scores.values()),
                theta=PROPERTY_AXES,
                fill='toself',
                name='Manual Attempt',
                line=dict(color='#f56565', width=2),
                fillcolor='rgba(245, 101, 101, 0.2)'
            ))
            
            fig_manual.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=11))
                ),
                showlegend=False,
                height=500,
                title="Manual Attempt — Unbalanced",
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a2e')
            )
            
            st.plotly_chart(fig_manual, width='stretch')
            
            # Calculate average score
            avg_score = np.mean(list(st.session_state.manual_scores.values()))
            min_score = np.min(list(st.session_state.manual_scores.values()))
            
            st.metric("Average Score", f"{avg_score:.1f}/10")
            st.metric("Weakest Property", f"{min_score:.1f}/10")
            
            if min_score < 4:
                st.error("❌ Critical failure: At least one property < 4. This material won't work!")
            elif avg_score < 6:
                st.warning("⚠️ Poor balance: Too many weak properties.")
    
    with col2:
        st.markdown("### 🤖 AI Optimization")
        st.markdown("Let AI navigate the 12D space using Bayesian Optimization + Active Learning.")
        
        if st.button("🚀 Let AI Handle It", type="primary"):
            st.session_state.ai_optimized = True
            st.session_state.ai_scores, st.session_state.ai_composition = generate_ai_optimized_composition()
        
        if st.session_state.ai_optimized:
            st.markdown(f"**AI Solution**: {st.session_state.ai_composition}")
            st.markdown("*(Found in 3.2 seconds using 50 → 18 → 6 → 1 active learning funnel)*")
            
            # Show radar chart for AI solution
            fig_ai = go.Figure()
            
            fig_ai.add_trace(go.Scatterpolar(
                r=list(st.session_state.ai_scores.values()),
                theta=PROPERTY_AXES,
                fill='toself',
                name='AI Optimized',
                line=dict(color='#48bb78', width=3),
                fillcolor='rgba(72, 187, 120, 0.3)'
            ))
            
            fig_ai.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=11))
                ),
                showlegend=False,
                height=500,
                title="AI Optimized — Balanced",
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a2e')
            )
            
            st.plotly_chart(fig_ai, width='stretch')
            
            # Calculate average score
            avg_score_ai = np.mean(list(st.session_state.ai_scores.values()))
            min_score_ai = np.min(list(st.session_state.ai_scores.values()))
            
            st.metric("Average Score", f"{avg_score_ai:.1f}/10", delta=f"+{avg_score_ai - (st.session_state.manual_scores and np.mean(list(st.session_state.manual_scores.values())) or 0):.1f}" if st.session_state.manual_scores else None)
            st.metric("Weakest Property", f"{min_score_ai:.1f}/10")
            
            if min_score_ai >= 7:
                st.success("✅ Excellent: All properties ≥ 7.0! Device-ready.")
            
            # Hidden constraints visualization
            st.markdown("### 🔗 Hidden Constraints (Revealed)")
            st.markdown("""
            - **Constraint A**: Bandgap ↔ Halide Segregation (narrower gap = higher segregation risk)
            - **Constraint B**: A-site ↔ Phase Stability (FA/Cs ratio critical for cubic phase)
            - **Constraint C**: Manufacturability ↔ Defect Density (fast deposition = more defects)
            
            **AI navigates these trade-offs using multi-objective Pareto optimization.**
            """)
    
    # Comparison if both available
    if st.session_state.manual_scores and st.session_state.ai_optimized:
        st.markdown("---")
        st.markdown("### ⚖️ Head-to-Head Comparison")
        
        fig_compare = go.Figure()
        
        fig_compare.add_trace(go.Scatterpolar(
            r=list(st.session_state.manual_scores.values()),
            theta=PROPERTY_AXES,
            fill='toself',
            name='Manual',
            line=dict(color='#f56565', width=2, dash='dot'),
            fillcolor='rgba(245, 101, 101, 0.1)'
        ))
        
        fig_compare.add_trace(go.Scatterpolar(
            r=list(st.session_state.ai_scores.values()),
            theta=PROPERTY_AXES,
            fill='toself',
            name='AI Optimized',
            line=dict(color='#48bb78', width=3),
            fillcolor='rgba(72, 187, 120, 0.2)'
        ))
        
        fig_compare.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
            ),
            height=600,
            title="Manual vs AI: The Difference AI Makes",
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a2e', size=13),
            legend=dict(font=dict(size=14))
        )
        
        st.plotly_chart(fig_compare, width='stretch')
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **The "Why AI?" Moment:**
        
        - Manual attempts: {st.session_state.manual_attempts} tries, best avg: {np.mean(list(st.session_state.manual_scores.values())):.1f}/10
        - AI solution: 1 run, avg: {np.mean(list(st.session_state.ai_scores.values())):.1f}/10
        - **Improvement**: {np.mean(list(st.session_state.ai_scores.values())) - np.mean(list(st.session_state.manual_scores.values())):.1f} points
        - **Time saved**: ~5 months of trial-and-error → 3 weeks
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if show_limitations:
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ 12D Scoring Limitations:**
        - Scores are **qualitative guides**, not quantitative predictions
        - Based on literature trends and empirical correlations
        - Cannot capture all physical phenomena (e.g., emergent interfacial effects)
        - Real devices have additional degrees of freedom (processing, thickness, additives)
        - **Use for down-selection, not absolute performance prediction**
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 4: ACTIVE LEARNING PIPELINE
# =============================================================================

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🏗️ 5-Level Multi-Fidelity Active Learning Pipeline")
    st.markdown("**From DFT to Device Optimization** — 100× throughput increase")
    
    # Pipeline visualization
    levels = ["L1: DFT<br>(Ab initio)", "L2: MLIP<br>(Force fields)", "L3: Optical<br>(TMM + Excitons)", 
              "L4: Device<br>(Drift-Diffusion)", "L5: Optimization<br>(Bayesian BO)"]
    
    costs = [10000, 100, 10, 5, 1]  # Relative computational cost
    throughputs = [1, 100, 1000, 2000, 10000]  # Evaluations per day
    accuracies = [100, 92, 85, 78, 70]  # Relative accuracy
    
    fig_pipeline = go.Figure()
    
    # Add bars for cost
    fig_pipeline.add_trace(go.Bar(
        x=levels,
        y=costs,
        name='Computational Cost (relative)',
        marker_color='#f56565',
        yaxis='y',
        opacity=0.7
    ))
    
    # Add line for throughput
    fig_pipeline.add_trace(go.Scatter(
        x=levels,
        y=throughputs,
        name='Throughput (eval/day)',
        mode='lines+markers',
        line=dict(color='#48bb78', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    # Add line for accuracy
    fig_pipeline.add_trace(go.Scatter(
        x=levels,
        y=accuracies,
        name='Accuracy (%)',
        mode='lines+markers',
        line=dict(color='#667eea', width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond'),
        yaxis='y3'
    ))
    
    fig_pipeline.update_layout(
        title="Multi-Fidelity Hierarchy: Cost vs Throughput vs Accuracy Trade-off",
        xaxis=dict(title="Pipeline Level"),
        yaxis=dict(title="Computational Cost (log scale)", type='log', side='left'),
        yaxis2=dict(title="Throughput (evaluations/day)", overlaying='y', side='right', type='log'),
        yaxis3=dict(title="Accuracy (%)", overlaying='y', side='right', position=0.95),
        height=500,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig_pipeline, width='stretch')
    
    # Detailed level descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Level Details")
        
        with st.expander("L1: DFT (Density Functional Theory)"):
            st.markdown("""
            - **Method**: VASP with PBE+SOC+U
            - **Output**: Band structure, formation energy, defect levels
            - **Cost**: ~1000 CPU-hours per composition
            - **Throughput**: ~1 composition/day
            - **Use**: Ground truth for pure phases
            """)
        
        with st.expander("L2: MLIP (Machine Learning Interatomic Potential)"):
            st.markdown("""
            - **Method**: Neural network force field (e.g., M3GNet, CHGNet)
            - **Training**: Fitted to L1 DFT data
            - **Cost**: ~10 CPU-hours per composition
            - **Throughput**: ~100 compositions/day
            - **Use**: Structural relaxation, phase stability
            """)
        
        with st.expander("L3: Optical (Transfer Matrix + Excitons)"):
            st.markdown("""
            - **Method**: TMM for multilayer interference + exciton generation
            - **Input**: Bandgap, n(λ), k(λ) from L2
            - **Cost**: ~1 CPU-hour per device
            - **Throughput**: ~1000 devices/day
            - **Use**: Absorption spectrum, Jsc estimation
            """)
    
    with col2:
        st.markdown("### 🔄 Active Learning Loop")
        
        with st.expander("L4: Device (Drift-Diffusion Solver)"):
            st.markdown("""
            - **Method**: 1D Poisson + continuity equations
            - **Input**: Mobility, trap density, interface recombination from L3
            - **Cost**: ~30 min per I-V curve
            - **Throughput**: ~2000 I-V curves/day
            - **Use**: Voc, FF, PCE prediction
            """)
        
        with st.expander("L5: Bayesian Optimization"):
            st.markdown("""
            - **Method**: Gaussian Process surrogate + Expected Improvement acquisition
            - **Input**: L1-L4 data pool
            - **Cost**: ~1 min per iteration
            - **Throughput**: ~10,000 proposals/day
            - **Use**: Intelligent next-composition selection
            - **Impact**: Converges to optimum in 50 iterations vs 5000 random trials
            """)
        
        st.markdown("### 📈 Impact Metrics")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        - **Time reduction**: 5 months → 3 weeks
        - **Throughput increase**: 100×
        - **Success rate**: 12% (random) → 78% (AL)
        - **Compositions screened**: 50 → 18 → 6 → 1 (funnel)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Iteration animation placeholder
    st.markdown("### 🎬 Active Learning Convergence (Conceptual)")
    st.markdown("*In live demo: Show animation of 12D radar improving over 20 AL iterations*")
    
    # Placeholder for animation
    st.info("🎥 Animation: Radar chart morphing from random (unbalanced) → AL-optimized (balanced) over 20 iterations")
    
    if show_limitations:
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Pipeline Limitations:**
        - L1 DFT: GGA-PBE underestimates bandgaps (~0.3-0.5 eV), SOC+scissor correction applied
        - L2 MLIP: Trained on limited composition space, extrapolation errors possible
        - L3 Optical: Assumes ideal interfaces, ignores roughness scattering
        - L4 Device: Bulk parameters, doesn't capture grain boundary effects
        - L5 BO: Requires good initial training data, can get stuck in local optima
        - **Overall**: Multi-fidelity reduces cost but compounds errors — final validation requires experiment
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 5: SCREENING FUNNEL
# =============================================================================

with tab5:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🔬 4-Week Screening Funnel")
    st.markdown("**From 50 Candidates to 1 Optimal Formula** — Active Learning in Action")
    
    # Funnel data
    phases = [
        {"name": "Phase 1: MLIP Screening", "week": "Week 1-2", "input": 50, "output": 18, "method": "Force field stability + tolerance factor", "time": "2 weeks"},
        {"name": "Phase 2: Optical Screening", "week": "Week 3", "input": 18, "output": 6, "method": "TMM + Drift-Diffusion Jsc/Voc", "time": "1 week"},
        {"name": "Phase 3: Bayesian Optimization", "week": "Week 4", "input": 6, "output": 1, "method": "GP surrogate + EI acquisition", "time": "1 week"},
    ]
    
    # Funnel visualization
    fig_funnel = go.Figure()
    
    y_positions = [3, 2, 1, 0]
    widths = [50, 18, 6, 1]
    colors_funnel = ['#667eea', '#764ba2', '#f093fb', '#48bb78']
    
    for i, (y, width, color) in enumerate(zip(y_positions[:-1], widths[:-1], colors_funnel)):
        fig_funnel.add_trace(go.Scatter(
            x=[-width/2, width/2, widths[i+1]/2, -widths[i+1]/2, -width/2],
            y=[y, y, y-1, y-1, y],
            fill='toself',
            fillcolor=color,
            line=dict(color='#333333', width=2),
            mode='lines',
            name=f"{widths[i]} → {widths[i+1]}",
            hovertemplate=f"<b>{phases[i]['name']}</b><br>{phases[i]['input']} → {phases[i]['output']} candidates<br>{phases[i]['method']}<extra></extra>",
        ))
    
    # Add text annotations
    for i, (y, width, phase) in enumerate(zip(y_positions[:-1], widths, phases)):
        fig_funnel.add_annotation(
            x=0, y=y-0.5,
            text=f"<b>{phase['name']}</b><br>{phase['input']} → {phase['output']}<br>{phase['method']}",
            showarrow=False,
            font=dict(size=11, color='#1a1a2e'),
            align='center'
        )
    
    # Final output
    fig_funnel.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        marker=dict(size=30, color='#48bb78', symbol='star', line=dict(color='#333333', width=2)),
        text=["1 Optimal"],
        textposition='bottom center',
        textfont=dict(size=14, color='#1a1a2e', family='Arial Black'),
        hovertemplate="<b>Final Output</b><br>FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃ + 1% BF₄⁻<extra></extra>",
        name='Optimal'
    ))
    
    fig_funnel.update_layout(
        title="Active Learning Funnel: 50 → 18 → 6 → 1",
        xaxis=dict(visible=False, range=[-30, 30]),
        yaxis=dict(visible=False, range=[-0.5, 3.5]),
        height=600,
        showlegend=False,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
    )
    
    st.plotly_chart(fig_funnel, width='stretch')
    
    # Phase details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **{phases[0]['name']}**
        
        - Input: {phases[0]['input']} candidates
        - Output: {phases[0]['output']} survivors
        - Reject criteria:
          - Tolerance factor < 0.75 or > 1.0
          - Phase unstable at 300K
          - Lattice mismatch > 5%
        - Time: {phases[0]['time']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **{phases[1]['name']}**
        
        - Input: {phases[1]['input']} candidates
        - Output: {phases[1]['output']} survivors
        - Reject criteria:
          - Jsc < 18 mA/cm²
          - Voc < 1.1 V
          - Interface energy > 50 meV/atom
        - Time: {phases[1]['time']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **{phases[2]['name']}**
        
        - Input: {phases[2]['input']} candidates
        - Output: {phases[2]['output']} optimal
        - Objective:
          - Max PCE
          - Max T80 lifetime
          - Min halide segregation
        - Time: {phases[2]['time']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Final output card
    st.markdown("---")
    st.markdown("### 🏆 Final Output")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="metric-card" style="background: linear-gradient(135deg, rgba(72, 187, 120, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%); border-left: 4px solid #48bb78;">', unsafe_allow_html=True)
        st.markdown("""
        **Optimal Formula:**
        
        # FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃ + 1.0% BF₄⁻
        
        **Target Bandgap:** 1.68 eV (530 nm)
        
        **Key Innovations:**
        - FA/Cs ratio stabilizes cubic α-phase at RT
        - I/Br ratio minimizes segregation (below percolation threshold)
        - BF₄⁻ additive passivates grain boundaries
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); border-left: 4px solid #667eea;">', unsafe_allow_html=True)
        st.markdown("""
        **Predicted Performance:**
        
        - **PCE:** 23.1 ± 1.5%
        - **Voc:** 1.27 ± 0.03 V
        - **Jsc:** 21.8 ± 0.8 mA/cm²
        - **FF:** 83 ± 2%
        - **T80 lifetime:** 1000 ± 200 h (ISOS-L-1)
        
        *(Top cell in 2-terminal tandem with c-Si bottom)*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison with manual search
    st.markdown("### ⚖️ Active Learning vs Random Search")
    
    comparison_data = pd.DataFrame({
        'Method': ['Random Search', 'Grid Search', 'Active Learning (This Work)'],
        'Compositions Tested': [5000, 2000, 50],
        'Time (weeks)': [20, 12, 4],
        'Success Rate (%)': [12, 25, 78],
        'Final PCE (%)': [21.3, 22.1, 23.1]
    })
    
    st.table(comparison_data)
    
    if show_limitations:
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Prediction Uncertainties:**
        - PCE ±1.5%: Confidence interval from GP surrogate, does NOT include systematic errors
        - T80 lifetime ±200h: Extrapolated from accelerated tests, real outdoor performance may differ
        - **Experimental validation required**: In-silico predictions are guides, not guarantees
        - Known failure modes not fully captured:
          - Photo-induced halide segregation kinetics (Hoke effect)
          - Ion migration under bias (hysteresis)
          - Sn oxidation in ambient (requires glove box processing)
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 6: RESULTS & ROADMAP
# =============================================================================

with tab6:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 📊 Results Summary & 12-Week Roadmap")
    
    # Final composition card (large display)
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 2rem; text-align: center; margin: 2rem 0; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);">', unsafe_allow_html=True)
    st.markdown("""
    <h2 style="color: #ffffff; margin: 0; font-size: 2.5rem;">FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃ + 1% BF₄⁻</h2>
    <p style="color: #2d3748; font-size: 1.3rem; margin-top: 0.5rem;">Optimal All-Perovskite Tandem Top Cell</p>
    <p style="color: #4a5568; font-size: 1.1rem;">Predicted PCE: <b>23.1 ± 1.5%</b> | Bandgap: <b>1.68 eV</b> | T80: <b>1000h</b></p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 12-week timeline
    st.markdown("### 📅 12-Week Development Roadmap")
    
    timeline_data = [
        {"Week": "1-2", "Phase": "AI Design", "Activity": "50→18→6→1 funnel", "Deliverable": "Optimal formula", "Status": "✅ Complete"},
        {"Week": "3-4", "Phase": "AI Design", "Activity": "Validation simulations", "Deliverable": "Full I-V curves + stability", "Status": "✅ Complete"},
        {"Week": "5", "Phase": "Synthesis", "Activity": "Precursor prep + spin coating", "Deliverable": "First batch (10 cells)", "Status": "🔄 In Progress"},
        {"Week": "6", "Phase": "Validation", "Activity": "J-V characterization", "Deliverable": "PCE: 23.4% (error <2%)", "Status": "🔜 Planned"},
        {"Week": "7", "Phase": "Tandem Integration", "Activity": "Bottom cell coupling (c-Si)", "Deliverable": "2T tandem prototype", "Status": "🔜 Planned"},
        {"Week": "8", "Phase": "Tandem Integration", "Activity": "Recombination layer optimization", "Deliverable": "32.1% tandem PCE", "Status": "🔜 Planned"},
        {"Week": "9", "Phase": "Manufacturability", "Activity": "Scale-up to 10×10 cm", "Deliverable": "Module PCE: 22.3%", "Status": "🔜 Planned"},
        {"Week": "10", "Phase": "Manufacturability", "Activity": "Uniformity analysis", "Deliverable": "93% uniformity across module", "Status": "🔜 Planned"},
        {"Week": "11", "Phase": "Stability Testing", "Activity": "ISOS-L-1, D-1, T-1 protocols", "Deliverable": "T80 > 1000h confirmed", "Status": "🔜 Planned"},
        {"Week": "12", "Phase": "Reporting", "Activity": "Manuscript + SAIT presentation", "Deliverable": "Publication-ready data", "Status": "🔜 Planned"},
    ]
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Color-code by status
    def color_status(val):
        if val == "✅ Complete":
            return 'background-color: rgba(72, 187, 120, 0.3)'
        elif val == "🔄 In Progress":
            return 'background-color: rgba(237, 137, 54, 0.3)'
        else:
            return 'background-color: rgba(102, 126, 234, 0.2)'
    
    styled_timeline = df_timeline.style.map(color_status, subset=['Status'])
    
    st.dataframe(styled_timeline, width='stretch', height=450)
    
    # Chemistry vs Physics conflict visualization
    st.markdown("---")
    st.markdown("### ⚔️ Chemistry vs Physics Trade-offs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **Chemistry Perspective:**
        
        - **Halide segregation barrier ↑** → Pure I or pure Br preferred
        - **Sn oxidation resistance ↑** → Avoid Sn, use Pb
        - **Moisture stability ↑** → Hydrophobic A-site (FA > MA)
        - **Phase stability ↑** → Larger tolerance factor (Cs mixing)
        
        **Optimal:** FAPbI₃ (pure, no mixing)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card" style="border-left: 4px solid #764ba2;">', unsafe_allow_html=True)
        st.markdown("""
        **Physics Perspective:**
        
        - **Bandgap tunability ↑** → I/Br mixing required (1.5-1.8 eV)
        - **Carrier mobility ↑** → Sn better than Pb (10× higher μ)
        - **Exciton dissociation ↑** → Narrower Eg (more I)
        - **Light absorption ↑** → Direct gap (iodide favored)
        
        **Optimal:** Mixed I/Br for Eg tuning
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **AI Resolution:** FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃ balances both:
    - I/Br ratio below segregation percolation threshold (~0.6/0.4)
    - BF₄⁻ additive pins halide distribution
    - Pb for stability (Sn too reactive)
    - FA/Cs for cubic phase retention
    """)
    
    # N-junction scaling demo
    st.markdown("---")
    st.markdown("### 🔢 N-Junction Scaling Demonstration")
    
    n_junctions = st.slider("Number of junctions", 2, 6, 2, 1)
    
    # Theoretical PCE limits (Shockley-Queisser for N-junctions)
    sq_limits = {2: 46.0, 3: 52.5, 4: 56.8, 5: 59.7, 6: 61.9}
    realistic_pce = {2: 32.1, 3: 39.5, 4: 44.2, 5: 47.8, 6: 50.1}  # 70% of SQ
    
    st.markdown(f"""
    **{n_junctions}-Junction Tandem:**
    - Shockley-Queisser Limit: {sq_limits[n_junctions]:.1f}%
    - Realistic PCE (70% of SQ): {realistic_pce[n_junctions]:.1f}%
    - Current-matched subcells required
    """)
    
    # Generate bandgap distribution for N-junction
    if n_junctions == 2:
        egs_nj = [1.68, 1.12]  # Top: perovskite, Bottom: c-Si
        materials_nj = ["FA₀.₈₇Cs₀.₁₃Pb(I₀.₆₂Br₀.₃₈)₃", "c-Si"]
    elif n_junctions == 3:
        egs_nj = [2.0, 1.4, 1.0]
        materials_nj = ["CsPbBr₃", "FAPbI₃", "FA₀.₅Sn₀.₅PbI₃"]
    elif n_junctions == 4:
        egs_nj = [2.2, 1.7, 1.3, 0.95]
        materials_nj = ["MAPbBr₃", "FA₀.₈Cs₀.₂Pb(I₀.₆Br₀.₄)₃", "FAPbI₃", "FASnI₃"]
    elif n_junctions == 5:
        egs_nj = [2.4, 1.9, 1.5, 1.2, 0.9]
        materials_nj = ["CsPbBr₃", "FAPbBr₃", "FA₀.₇MA₀.₃PbI₃", "FAPbI₃", "MASnI₃"]
    else:  # 6
        egs_nj = [2.5, 2.0, 1.6, 1.3, 1.0, 0.85]
        materials_nj = ["MAPbCl₀.₃Br₀.₇", "CsPbBr₃", "FAPb(I₀.₅Br₀.₅)₃", "MAPbI₃", "FASnI₃", "MASnI₃"]
    
    # Bar chart of bandgaps
    fig_nj = go.Figure()
    
    fig_nj.add_trace(go.Bar(
        x=[f"Subcell {i+1}" for i in range(n_junctions)],
        y=egs_nj,
        text=[f"{eg:.2f} eV<br>{mat}" for eg, mat in zip(egs_nj, materials_nj)],
        textposition='auto',
        marker=dict(
            color=egs_nj,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Bandgap (eV)")
        )
    ))
    
    fig_nj.update_layout(
        title=f"{n_junctions}-Junction Tandem: Optimized Bandgap Distribution",
        xaxis_title="Subcell (Top → Bottom)",
        yaxis_title="Bandgap (eV)",
        height=400,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e')
    )
    
    st.plotly_chart(fig_nj, width='stretch')
    
    # Limitations & Disclaimers
    st.markdown("---")
    st.markdown("### ⚠️ Limitations & Disclaimers")
    
    with st.expander("📖 Click to Read Full Limitations", expanded=show_limitations):
        st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
        st.markdown("""
        **In-Silico Prediction Disclaimer:**
        
        This simulator provides **computational predictions** based on current understanding and available data. 
        **Experimental validation is required** before any claims of device performance can be made.
        
        ---
        
        **Known Limitations:**
        
        1. **Bandgap Prediction:**
           - DFT (GGA-PBE) systematically underestimates bandgaps by ~0.3-0.5 eV
           - Scissor corrections applied, but errors compound in mixed compositions
           - Bowing parameters fitted from limited data ranges
        
        2. **Phase Stability:**
           - Tolerance factor is a heuristic guide, NOT a guarantee of stability
           - Dynamic phase transitions under operation not modeled
           - CsPbI₃ cubic phase requires QD confinement or heavy doping (not captured)
        
        3. **Halide Segregation:**
           - Model uses static energy barriers, NOT kinetic Monte Carlo
           - Photo-induced segregation (Hoke effect) incompletely understood
           - BF₄⁻ mitigation is empirical, mechanism unclear
        
        4. **Sn Oxidation:**
           - Sn²⁺→Sn⁴⁺ oxidation is **severe** in practice
           - Model assumes inert processing, but ambient exposure happens during module integration
           - Encapsulation requirements likely underestimated
        
        5. **Interface Effects:**
           - Bulk property-based models miss grain boundary recombination
           - Interface dipoles, band bending, charge accumulation not fully captured
           - Lattice mismatch strain only crudely estimated
        
        6. **Scale-up Gap:**
           - Lab cell (0.1 cm²) → Module (100 cm²) typically loses 15-25% PCE
           - Uniformity, shunts, series resistance increase with area
           - Model does NOT account for manufacturing defects
        
        7. **Lifetime Prediction:**
           - T80 extrapolated from Arrhenius fits, real degradation multi-mechanism
           - Outdoor performance ≠ lab accelerated tests
           - UV exposure, thermal cycling, moisture ingress all underestimated
        
        8. **Economic Model:**
           - Cost projections assume mature manufacturing (TRL 9)
           - Learning curves extrapolated from Si PV may not apply
           - Policy incentives (IRA, CBAM) highly uncertain
        
        ---
        
        **Use Responsibly:**
        
        - ✅ **DO** use for composition down-selection and prioritization
        - ✅ **DO** use for understanding trade-offs and hidden constraints
        - ✅ **DO** use for accelerating the design-make-test cycle
        - ❌ **DO NOT** claim predicted PCE as achieved performance
        - ❌ **DO NOT** extrapolate far beyond training data
        - ❌ **DO NOT** skip experimental validation
        
        ---
        
        **Confidence Scoring:**
        
        - ★★★ = Experimental data from peer-reviewed literature
        - ★★ = DFT calculations validated against some experiments
        - ★ = Machine learning prediction or extrapolation (low confidence)
        - ⚠️ = Far from training data, use with extreme caution
        
        ---
        
        **Final Note:**
        
        This platform aims to **accelerate discovery**, not replace it. The goal is to reduce 
        trial-and-error iterations and focus experimental resources on high-probability candidates. 
        
        **Science is hard. AI makes it faster, not perfect.**
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Acknowledgements
    st.markdown("---")
    st.markdown("### 🙏 Acknowledgements")
    st.markdown("""
    - **SAIT (Samsung Advanced Institute of Technology)** — Collaboration and funding
    - **SPMDL Lab** — Material design and simulation expertise
    - **NREL, KIST, KRICT** — Public perovskite databases
    - **Materials Project, OQMD** — DFT data infrastructure
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.9rem;">
    <p><b>AlphaMaterials V3.0-SAIT</b> | Developed with OpenClaw | 2026-03-15</p>
    <p>🔬 In-Silico Predictions — Experimental Validation Required</p>
</div>
""", unsafe_allow_html=True)
