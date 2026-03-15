#!/usr/bin/env python3
"""
AlphaMaterials V6: Generative Inverse Design + Techno-Economics
================================================================

Evolution from V5 → V6: From autonomous discovery to industrial-scale deployment readiness

New in V6:
- Generative Inverse Design (target properties → candidate compositions)
- Techno-Economic Analysis ($/Watt, supply chain, manufacturing costs)
- Scale-Up Risk Assessment (toxicity, TRL, regulatory compliance)
- Publication-Ready Export (LaTeX tables, 300 DPI figures, methods text, BibTeX)
- Dashboard Summary (full campaign overview)

All V5 features preserved:
✅ Bayesian Optimization ✅ Model Fine-tuning ✅ Multi-objective
✅ Experiment Planner ✅ Session Management

SAIT × SPMDL Collaboration Platform
V6.0 - Generative Inverse Design + Techno-Economics

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
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

# Import V5 modules
try:
    from db_clients import UnifiedDBClient, CacheDB
    from data_parser import UserDataParser
    from ml_models import BandgapPredictor, CompositionFeaturizer
    from bayesian_opt import BayesianOptimizer
    from multi_objective import MultiObjectiveOptimizer, default_weights
    from session import SessionManager, create_default_session
    V5_AVAILABLE = True
except ImportError as e:
    st.error(f"V5 modules import failed: {e}")
    V5_AVAILABLE = False

# Import V6 modules
try:
    from inverse_design import InverseDesignEngine
    from techno_economics import TechnoEconomicAnalyzer, compare_to_silicon
    from export import PublicationExporter, format_property_table, create_graphical_abstract
    V6_AVAILABLE = True
except ImportError as e:
    st.error(f"V6 modules import failed: {e}")
    V6_AVAILABLE = False

MODULES_AVAILABLE = V5_AVAILABLE and V6_AVAILABLE

# Page config
st.set_page_config(
    page_title="AlphaMaterials V6: Inverse Design + Techno-Economics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (V6 branding - dark theme)
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
        color: #fafafa;
    }
    
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .v6-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.4rem 1.0rem;
        border-radius: 25px;
        font-size: 1.0rem;
        font-weight: bold;
        margin-left: 1rem;
    }
    
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #0d3d2d;
        border-left: 5px solid #48bb78;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #3d2d0d;
        border-left: 5px solid #f39c12;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #0d2d3d;
        border-left: 5px solid #3498db;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    .confidence-high { color: #48bb78; font-weight: bold; }
    .confidence-medium { color: #ed8936; font-weight: bold; }
    .confidence-low { color: #f56565; font-weight: bold; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables (V5 + V6)"""
    # V5 states (preserved)
    defaults = {
        'db_client': None,
        'db_data': None,
        'user_data': None,
        'combined_data': None,
        'ml_model': None,
        'model_trained': False,
        'db_loaded': False,
        'bo_optimizer': None,
        'bo_fitted': False,
        'bo_history': None,
        'mo_optimizer': None,
        'mo_weights': default_weights() if MODULES_AVAILABLE else {},
        'experiment_queue': None,
        'session_manager': SessionManager() if MODULES_AVAILABLE else None,
        'current_session': create_default_session() if MODULES_AVAILABLE else {},
        'training_history': [],
        'train_metrics': {},
        
        # V6 new states
        'inverse_engine': None,
        'inverse_candidates': None,
        'techno_analyzer': None,
        'cost_analysis_results': None,
        'publication_exporter': PublicationExporter() if MODULES_AVAILABLE else None,
        'campaign_summary': {},
        'pareto_optimal_materials': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_sample_data() -> pd.DataFrame:
    """Load bundled sample data as fallback"""
    sample_path = Path(__file__).parent / 'data' / 'sample_data' / 'perovskites_sample.csv'
    
    if sample_path.exists():
        return pd.read_csv(sample_path)
    else:
        # Hardcoded minimal fallback
        return pd.DataFrame({
            'formula': ['MAPbI3', 'FAPbI3', 'CsPbI3', 'MAPbBr3', 'FAPbBr3', 
                       'MA0.5FA0.5PbI3', 'Cs0.1MA0.9PbI3', 'MAPb0.5Sn0.5I3'],
            'bandgap': [1.59, 1.51, 1.72, 2.30, 2.25, 1.55, 1.62, 1.25],
            'source': ['fallback'] * 8
        })

def update_campaign_summary():
    """Update campaign summary statistics"""
    summary = {
        'total_materials_screened': len(st.session_state.combined_data) if st.session_state.combined_data is not None else 0,
        'user_experiments': len(st.session_state.user_data) if st.session_state.user_data is not None else 0,
        'model_trained': st.session_state.model_trained,
        'bo_active': st.session_state.bo_fitted,
        'inverse_candidates_generated': len(st.session_state.inverse_candidates) if st.session_state.inverse_candidates is not None else 0,
        'pareto_optimal_found': len(st.session_state.pareto_optimal_materials) if st.session_state.pareto_optimal_materials is not None else 0,
        'experiments_queued': len(st.session_state.experiment_queue) if st.session_state.experiment_queue is not None else 0,
        'last_updated': datetime.now().isoformat()
    }
    st.session_state.campaign_summary = summary

# =============================================================================
# MAIN APP
# =============================================================================

# Title
st.markdown('<h1 class="main-title">AlphaMaterials<span class="v6-badge">V6: Inverse Design + Techno-Economics</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Target Properties → AI Generates Candidates → Cost-Optimized → Publication-Ready</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🚀 V6 Navigation")
    st.markdown("---")
    
    st.markdown("**🆕 New in V6:**")
    st.success("✅ Generative Inverse Design\n\n✅ Techno-Economic Analysis\n\n✅ Scale-Up Risk Assessment\n\n✅ Publication Export\n\n✅ Campaign Dashboard")
    
    st.markdown("**✨ Preserved from V5:**")
    st.info("✅ Bayesian Optimization\n\n✅ Model Fine-tuning\n\n✅ Multi-Objective Pareto")
    
    st.markdown("---")
    st.markdown("### ⚙️ Global Settings")
    
    # Target bandgap
    target_bandgap = st.number_input(
        "🎯 Target Bandgap (eV)",
        min_value=0.5,
        max_value=3.0,
        value=1.35,
        step=0.01,
        help="Target bandgap for optimization & inverse design"
    )
    
    # Acquisition function
    acq_func = st.selectbox(
        "Acquisition Function",
        ['ei', 'ucb', 'ts'],
        format_func=lambda x: {'ei': 'Expected Improvement', 'ucb': 'Upper Confidence Bound', 'ts': 'Thompson Sampling'}[x],
    )
    
    st.markdown("---")
    
    # Status dashboard
    st.markdown("### 📊 Campaign Status")
    
    if st.session_state.db_loaded:
        n_total = len(st.session_state.combined_data) if st.session_state.combined_data is not None else 0
        n_user = len(st.session_state.user_data) if st.session_state.user_data is not None else 0
        st.metric("Materials DB", n_total)
        st.metric("Your Experiments", n_user)
    
    if st.session_state.model_trained:
        st.metric("ML Model", "✅ Trained")
    
    if st.session_state.bo_fitted:
        st.metric("Bayesian Opt", "✅ Active")
    
    if st.session_state.inverse_candidates is not None:
        st.metric("Inverse Design", f"✅ {len(st.session_state.inverse_candidates)} candidates")
    
    st.markdown("---")
    st.markdown("**Version:** V6.0")
    st.markdown("**Date:** 2026-03-15")

# =============================================================================
# TABS (V5 tabs 1-7 + V6 tabs 8-12)
# =============================================================================

tabs = st.tabs([
    "🗄️ Database",           # 1
    "📤 Upload Data",         # 2
    "🤖 ML Surrogate",        # 3
    "🎯 Bayesian Opt",        # 4
    "🏆 Multi-Objective",     # 5
    "📋 Experiment Planner",  # 6
    "🧬 Inverse Design",      # 7 (NEW V6)
    "💰 Techno-Economics",    # 8 (NEW V6)
    "⚠️ Scale-Up Risk",       # 9 (NEW V6)
    "📄 Publication Export",  # 10 (NEW V6)
    "📊 Dashboard",           # 11 (NEW V6)
    "💾 Session Manager"      # 12 (was tab 7)
])

# =============================================================================
# TAB 1: DATABASE (Same as V5, condensed)
# =============================================================================

with tabs[0]:
    st.markdown("## 🗄️ Database Explorer")
    st.markdown("**Load perovskite database from Materials Project, AFLOW, JARVIS**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Load Database", type="primary", key="load_db"):
            with st.spinner("Fetching from databases..."):
                try:
                    st.session_state.db_client = UnifiedDBClient()
                    db_data = st.session_state.db_client.get_all_perovskites(
                        max_per_source=200, use_cache=True
                    )
                    
                    if db_data.empty:
                        st.warning("No API data. Loading sample data...")
                        db_data = load_sample_data()
                    
                    st.session_state.db_data = db_data
                    st.session_state.combined_data = db_data.copy()
                    st.session_state.db_loaded = True
                    update_campaign_summary()
                    
                    st.success(f"✅ Loaded {len(db_data)} materials!")
                    
                except Exception as e:
                    st.error(f"Database load failed: {e}")
                    sample_data = load_sample_data()
                    st.session_state.db_data = sample_data
                    st.session_state.combined_data = sample_data.copy()
                    st.session_state.db_loaded = True
                    update_campaign_summary()
    
    with col2:
        st.info("**빈 지도가 탐험의 시작**")
    
    # Show database
    if st.session_state.db_loaded and st.session_state.db_data is not None:
        st.markdown("---")
        df = st.session_state.db_data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Materials", len(df))
        with col2:
            if 'bandgap' in df.columns:
                st.metric("Bandgap Range", f"{df['bandgap'].min():.2f} - {df['bandgap'].max():.2f} eV")
        with col3:
            if 'source' in df.columns:
                st.metric("Data Sources", df['source'].nunique())
        
        st.dataframe(df.head(100), use_container_width=True, height=300)

# =============================================================================
# TAB 2: UPLOAD DATA (Same as V5)
# =============================================================================

with tabs[1]:
    st.markdown("## 📤 Upload Your Experimental Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Must contain 'formula' and 'bandgap' columns",
        key="upload_data"
    )
    
    if uploaded_file is not None:
        try:
            parser = UserDataParser()
            file_content = uploaded_file.read()
            df_user = parser.parse(file_content, uploaded_file.name)
            
            if not df_user.empty:
                st.success(f"✅ Parsed {len(df_user)} materials")
                st.dataframe(df_user, use_container_width=True)
                
                if st.button("💾 Save to Session", type="primary", key="save_user_data"):
                    if st.session_state.db_loaded:
                        st.session_state.user_data = df_user
                        st.session_state.combined_data = parser.merge_with_db(
                            df_user, st.session_state.db_data
                        )
                        update_campaign_summary()
                        st.success("✅ Data saved!")
                    else:
                        st.error("Please load database first (Tab 1)")
        
        except Exception as e:
            st.error(f"Upload error: {e}")

# =============================================================================
# TAB 3: ML SURROGATE (V5 with fine-tuning)
# =============================================================================

with tabs[2]:
    st.markdown("## 🤖 ML Surrogate Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏋️ Train Base Model")
        
        if st.button("🚀 Train Model", type="primary", disabled=not st.session_state.db_loaded, key="train_model"):
            with st.spinner("Training XGBoost..."):
                try:
                    df_train = st.session_state.combined_data
                    df_train = df_train[df_train['bandgap'].notna() & (df_train['bandgap'] > 0)]
                    
                    model = BandgapPredictor(use_xgboost=True)
                    metrics = model.train(df_train, formula_col='formula', target_col='bandgap')
                    
                    st.session_state.ml_model = model
                    st.session_state.model_trained = True
                    st.session_state.train_metrics = metrics
                    update_campaign_summary()
                    
                    st.success("✅ Model trained!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Samples", metrics['n_samples'])
                    with col_b:
                        st.metric("CV MAE", f"{metrics['cv_mae']:.3f} eV")
                    with col_c:
                        st.metric("R²", f"{metrics['train_r2']:.3f}")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        # Fine-tuning
        if st.session_state.user_data is not None and st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### ⚡ Fine-tune on Your Data")
            
            learning_rate = st.slider("Fine-tuning Rate", 0.01, 0.5, 0.05, 0.01, key="ft_lr")
            
            if st.button("🔥 Fine-tune", type="primary", key="finetune"):
                with st.spinner("Fine-tuning..."):
                    try:
                        model = st.session_state.ml_model
                        ft_metrics = model.fine_tune(st.session_state.user_data, learning_rate=learning_rate)
                        
                        st.success("✅ Fine-tuned!")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("MAE Before", f"{ft_metrics['mae_before']:.3f} eV")
                        with col_b:
                            st.metric("MAE After", f"{ft_metrics['mae_after']:.3f} eV", 
                                     delta=f"{-ft_metrics['mae_improvement']:.3f} eV", delta_color="inverse")
                        
                    except Exception as e:
                        st.error(f"Fine-tuning failed: {e}")
    
    with col2:
        st.markdown("### ℹ️ Model Info")
        if st.session_state.model_trained:
            metrics = st.session_state.train_metrics
            st.markdown(f"""
            **Model Ready!**
            
            - Samples: {metrics.get('n_samples', 'N/A')}
            - MAE: {metrics.get('cv_mae', 0):.3f} eV
            - R²: {metrics.get('train_r2', 0):.3f}
            """)

# =============================================================================
# TAB 4: BAYESIAN OPTIMIZATION (V5)
# =============================================================================

with tabs[3]:
    st.markdown("## 🎯 Bayesian Optimization")
    
    if not st.session_state.model_trained or st.session_state.user_data is None:
        st.info("💡 Train model and upload data first (Tabs 2 & 3)")
    else:
        if st.button("🚀 Fit BO", type="primary", key="fit_bo"):
            with st.spinner("Fitting Gaussian Process..."):
                try:
                    bo = BayesianOptimizer(target_bandgap=target_bandgap, acq_function=acq_func)
                    bo.fit(st.session_state.user_data, formula_col='formula', target_col='bandgap')
                    
                    st.session_state.bo_optimizer = bo
                    st.session_state.bo_fitted = True
                    update_campaign_summary()
                    
                    st.success("✅ BO ready!")
                except Exception as e:
                    st.error(f"BO fitting failed: {e}")
        
        # Generate suggestions
        if st.session_state.bo_fitted:
            st.markdown("---")
            st.markdown("### 🎲 Generate Suggestions")
            
            n_suggestions = st.slider("Number of suggestions", 1, 20, 5, key="bo_n_sugg")
            
            if st.button("🔮 Suggest", key="bo_suggest"):
                with st.spinner("Evaluating candidates..."):
                    try:
                        bo = st.session_state.bo_optimizer
                        search_space = {'A': ['MA', 'FA', 'Cs'], 'B': ['Pb', 'Sn'], 'X': ['I', 'Br', 'Cl']}
                        suggestions = bo.optimize_composition(search_space=search_space, n_samples=1000)
                        
                        st.session_state.bo_history = suggestions
                        
                        st.dataframe(suggestions[['rank', 'formula', 'predicted_bandgap', 'uncertainty', 'acquisition_value']].head(n_suggestions),
                                   use_container_width=True)
                        
                        if st.button("➕ Add Top 5 to Queue", key="bo_add_queue"):
                            if st.session_state.experiment_queue is None:
                                st.session_state.experiment_queue = suggestions.head(5).copy()
                            else:
                                st.session_state.experiment_queue = pd.concat([
                                    st.session_state.experiment_queue, suggestions.head(5)
                                ], ignore_index=True)
                            st.success("✅ Added to queue")
                        
                    except Exception as e:
                        st.error(f"Suggestion failed: {e}")

# =============================================================================
# TAB 5: MULTI-OBJECTIVE (V5)
# =============================================================================

with tabs[4]:
    st.markdown("## 🏆 Multi-Objective Optimization")
    
    if not st.session_state.model_trained:
        st.info("💡 Train model first (Tab 3)")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            w_bandgap = st.slider("Bandgap Match", 0.0, 1.0, 0.4, 0.05, key="mo_w_bg")
        with col2:
            w_stability = st.slider("Stability", 0.0, 1.0, 0.3, 0.05, key="mo_w_stab")
        with col3:
            w_synth = st.slider("Synthesizability", 0.0, 1.0, 0.2, 0.05, key="mo_w_synth")
        with col4:
            w_cost = st.slider("Low Cost", 0.0, 1.0, 0.1, 0.05, key="mo_w_cost")
        
        if st.button("🎯 Evaluate Multi-Objective", type="primary", key="mo_eval"):
            with st.spinner("Evaluating objectives..."):
                try:
                    mo = MultiObjectiveOptimizer(target_bandgap=target_bandgap)
                    
                    # Get candidates
                    if st.session_state.bo_history is not None:
                        candidates = st.session_state.bo_history['formula'].head(100).tolist()
                        bandgaps = st.session_state.bo_history['predicted_bandgap'].head(100).values
                    else:
                        # Generate simple candidates
                        candidates = ['MAPbI3', 'FAPbI3', 'CsPbI3', 'MAPbBr3', 'MA0.5FA0.5PbI3']
                        bandgaps, _ = st.session_state.ml_model.predict(candidates)
                    
                    obj_df = mo.evaluate_objectives(candidates, bandgaps)
                    pareto_df = mo.calculate_pareto_front(obj_df, ['obj_bandgap_match', 'obj_stability', 'obj_synthesizability', 'obj_cost'])
                    
                    st.session_state.pareto_optimal_materials = pareto_df
                    update_campaign_summary()
                    
                    st.markdown(f"### 🌟 Pareto-Optimal: {len(pareto_df)} / {len(obj_df)} materials")
                    st.dataframe(pareto_df[['formula', 'bandgap', 'obj_bandgap_match', 'obj_stability', 'obj_cost']].head(10),
                               use_container_width=True)
                    
                    # 2D Pareto plots
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig1 = mo.plot_pareto_front_2d(obj_df, 'obj_bandgap_match', 'obj_stability', pareto_df)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col_b:
                        fig2 = mo.plot_pareto_front_2d(obj_df, 'obj_cost', 'obj_synthesizability', pareto_df)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"MO evaluation failed: {e}")

# =============================================================================
# TAB 6: EXPERIMENT PLANNER (V5)
# =============================================================================

with tabs[5]:
    st.markdown("## 📋 Experiment Planner")
    
    if st.session_state.experiment_queue is None or st.session_state.experiment_queue.empty:
        st.info("💡 No experiments queued. Generate suggestions in Tab 4 (Bayesian Opt)")
    else:
        queue = st.session_state.experiment_queue
        st.markdown(f"### 🧪 Queue: {len(queue)} experiments")
        
        st.dataframe(queue[['rank', 'formula', 'predicted_bandgap', 'uncertainty', 'acquisition_value']].head(20),
                   use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = queue.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download CSV", csv, f"queue_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", key="exp_download")
        with col2:
            if st.button("🗑️ Clear Queue", key="exp_clear"):
                st.session_state.experiment_queue = None
                st.rerun()

# =============================================================================
# TAB 7: INVERSE DESIGN (NEW V6)
# =============================================================================

with tabs[6]:
    st.markdown("## 🧬 Generative Inverse Design")
    st.markdown("**Specify target properties → AI generates candidate compositions**")
    
    if not st.session_state.model_trained:
        st.info("💡 Train ML model first (Tab 3) for better inverse design")
    
    st.markdown("### 🎯 Define Constraints")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        id_target_bg = st.number_input("Target Bandgap (eV)", 0.5, 3.0, 1.35, 0.01, key="id_target_bg")
    with col2:
        id_tol_bg = st.number_input("Bandgap Tolerance (±eV)", 0.01, 0.5, 0.05, 0.01, key="id_tol_bg")
    with col3:
        id_min_stab = st.slider("Min Stability", 0.0, 1.0, 0.85, 0.05, key="id_min_stab")
    with col4:
        id_max_cost = st.number_input("Max Cost ($/kg)", 10, 500, 100, 10, key="id_max_cost")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        id_n_candidates = st.slider("Candidates to Screen", 100, 5000, 1000, 100, key="id_n_cand")
    with col_b:
        id_method = st.selectbox("Method", ['rejection', 'genetic'], 
                                format_func=lambda x: 'Rejection Sampling (fast)' if x == 'rejection' else 'Genetic Algorithm (thorough)',
                                key="id_method")
    
    if st.button("🚀 Generate Candidates", type="primary", key="id_generate"):
        with st.spinner(f"Generating candidates via {id_method}..."):
            try:
                # Initialize engine
                gp_model = st.session_state.bo_optimizer.gp if st.session_state.bo_fitted else None
                featurizer = CompositionFeaturizer()
                
                engine = InverseDesignEngine(gp_model, featurizer)
                st.session_state.inverse_engine = engine
                
                # Generate
                candidates_df = engine.generate_candidates(
                    target_bandgap=id_target_bg,
                    bandgap_tolerance=id_tol_bg,
                    min_stability=id_min_stab,
                    max_cost=id_max_cost,
                    n_candidates=id_n_candidates,
                    method=id_method
                )
                
                st.session_state.inverse_candidates = candidates_df
                update_campaign_summary()
                
                if not candidates_df.empty:
                    st.success(f"✅ Found {len(candidates_df)} valid candidates!")
                    
                    # Display top candidates
                    st.markdown("### 🏆 Top Candidates")
                    display_cols = ['rank', 'formula', 'predicted_bandgap', 'stability_score', 'cost_per_kg', 'feasibility_score', 'confidence']
                    st.dataframe(candidates_df[display_cols].head(20), use_container_width=True)
                    
                    # Visualization
                    st.markdown("### 🎨 Target Region Visualization")
                    fig_viz = engine.visualize_target_region(
                        candidates_df, id_target_bg, id_tol_bg, id_min_stab, id_max_cost
                    )
                    st.plotly_chart(fig_viz, use_container_width=True)
                    
                    # Export to experiment queue
                    if st.button("➕ Add Top 10 to Experiment Queue", key="id_to_queue"):
                        top_10 = candidates_df.head(10).copy()
                        top_10['rank'] = range(1, len(top_10) + 1)
                        
                        if st.session_state.experiment_queue is None:
                            st.session_state.experiment_queue = top_10
                        else:
                            st.session_state.experiment_queue = pd.concat([
                                st.session_state.experiment_queue, top_10
                            ], ignore_index=True)
                        st.success("✅ Added to experiment queue!")
                
                else:
                    st.warning("⚠️ No candidates found satisfying all constraints. Try relaxing constraints.")
                    
            except Exception as e:
                st.error(f"Inverse design failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# =============================================================================
# TAB 8: TECHNO-ECONOMICS (NEW V6)
# =============================================================================

with tabs[7]:
    st.markdown("## 💰 Techno-Economic Analysis")
    st.markdown("**Manufacturing cost, $/Watt, supply chain risk, cost sensitivity**")
    
    # Initialize analyzer
    if st.session_state.techno_analyzer is None:
        st.session_state.techno_analyzer = TechnoEconomicAnalyzer()
    
    analyzer = st.session_state.techno_analyzer
    
    # Select compositions to analyze
    st.markdown("### 📋 Select Compositions to Analyze")
    
    # Source selection
    source_options = []
    if st.session_state.inverse_candidates is not None and not st.session_state.inverse_candidates.empty:
        source_options.append("Inverse Design Candidates")
    if st.session_state.pareto_optimal_materials is not None and not st.session_state.pareto_optimal_materials.empty:
        source_options.append("Pareto-Optimal Materials")
    if st.session_state.bo_history is not None and not st.session_state.bo_history.empty:
        source_options.append("BO Suggestions")
    
    source_options.append("Manual Entry")
    
    analysis_source = st.selectbox("Data Source", source_options, key="te_source")
    
    formulas_to_analyze = []
    efficiencies = []
    
    if analysis_source == "Manual Entry":
        manual_input = st.text_area(
            "Enter compositions (one per line, format: formula,efficiency)",
            value="MAPbI3,0.20\nFAPbI3,0.22\nMA0.5FA0.5PbI3,0.23",
            key="te_manual"
        )
        
        for line in manual_input.strip().split('\n'):
            if ',' in line:
                formula, eff = line.split(',')
                formulas_to_analyze.append(formula.strip())
                efficiencies.append(float(eff.strip()))
    
    elif analysis_source == "Inverse Design Candidates":
        top_n = st.slider("Top N candidates", 1, 20, 5, key="te_top_inverse")
        df_source = st.session_state.inverse_candidates.head(top_n)
        formulas_to_analyze = df_source['formula'].tolist()
        
        # Estimate efficiency from bandgap (Shockley-Queisser limit approximation)
        for bg in df_source['predicted_bandgap']:
            if 1.1 <= bg <= 1.4:
                eff = 0.30 - 0.05 * abs(bg - 1.34)  # Peak at ~1.34 eV
            else:
                eff = 0.20
            efficiencies.append(eff)
    
    elif analysis_source == "Pareto-Optimal Materials":
        top_n = st.slider("Top N Pareto", 1, 20, 5, key="te_top_pareto")
        df_source = st.session_state.pareto_optimal_materials.head(top_n)
        formulas_to_analyze = df_source['formula'].tolist()
        
        for bg in df_source['bandgap']:
            if 1.1 <= bg <= 1.4:
                eff = 0.30 - 0.05 * abs(bg - 1.34)
            else:
                eff = 0.20
            efficiencies.append(eff)
    
    elif analysis_source == "BO Suggestions":
        top_n = st.slider("Top N BO", 1, 20, 5, key="te_top_bo")
        df_source = st.session_state.bo_history.head(top_n)
        formulas_to_analyze = df_source['formula'].tolist()
        
        for bg in df_source['predicted_bandgap']:
            if 1.1 <= bg <= 1.4:
                eff = 0.30 - 0.05 * abs(bg - 1.34)
            else:
                eff = 0.20
            efficiencies.append(eff)
    
    # Run analysis
    if st.button("💰 Calculate Economics", type="primary", key="te_calc"):
        if formulas_to_analyze:
            with st.spinner("Analyzing techno-economics..."):
                try:
                    results = []
                    
                    for formula, eff in zip(formulas_to_analyze, efficiencies):
                        cost_data = analyzer.calculate_cost_per_watt(formula, eff)
                        mat_cost = analyzer.calculate_material_cost(formula)
                        supply_risk = analyzer.calculate_supply_risk(formula)
                        
                        results.append({
                            'formula': formula,
                            'efficiency': eff,
                            'cost_per_watt': cost_data['cost_per_watt'],
                            'material_cost_per_kg': mat_cost['cost_per_kg'],
                            'vs_silicon_ratio': cost_data['vs_silicon_ratio'],
                            'competitive': cost_data['competitive'],
                            'supply_risk': supply_risk['overall_risk_score']
                        })
                    
                    df_results = pd.DataFrame(results)
                    st.session_state.cost_analysis_results = df_results
                    
                    # Display results
                    st.markdown("### 📊 Cost Analysis Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Comparison chart
                    st.markdown("### 📈 $/Watt Comparison vs Silicon")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=df_results['formula'],
                        y=df_results['cost_per_watt'],
                        marker_color=['green' if c else 'red' for c in df_results['competitive']],
                        text=[f"${v:.3f}/W" for v in df_results['cost_per_watt']],
                        textposition='outside'
                    ))
                    
                    # Silicon baseline
                    fig.add_hline(y=0.25, line_dash="dash", line_color="blue", 
                                 annotation_text="Silicon Baseline ($0.25/W)")
                    
                    fig.update_layout(
                        title="Cost per Watt Comparison",
                        xaxis_title="Composition",
                        yaxis_title="$/Watt",
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed breakdown for top candidate
                    st.markdown("---")
                    st.markdown("### 🔍 Detailed Cost Breakdown (Top Candidate)")
                    
                    top_formula = df_results.iloc[0]['formula']
                    top_eff = df_results.iloc[0]['efficiency']
                    
                    fig_waterfall = analyzer.plot_cost_waterfall(top_formula, top_eff)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.markdown("### 📉 Sensitivity Analysis")
                    
                    df_sens, df_tornado = analyzer.sensitivity_analysis(top_formula, top_eff)
                    
                    fig_tornado = analyzer.plot_tornado_sensitivity(df_tornado, top_formula)
                    st.plotly_chart(fig_tornado, use_container_width=True)
                    
                    st.markdown("**Key Insight:** The parameters with highest sensitivity are the main cost drivers.")
                    
                except Exception as e:
                    st.error(f"Techno-economic analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("No compositions selected for analysis")

# =============================================================================
# TAB 9: SCALE-UP RISK (NEW V6)
# =============================================================================

with tabs[8]:
    st.markdown("## ⚠️ Scale-Up Risk Assessment")
    st.markdown("**Toxicity, supply chain, TRL, regulatory compliance**")
    
    if st.session_state.techno_analyzer is None:
        st.session_state.techno_analyzer = TechnoEconomicAnalyzer()
    
    analyzer = st.session_state.techno_analyzer
    
    # Select composition
    formula_to_assess = st.text_input("Composition to Assess", "MAPbI3", key="risk_formula")
    has_exp_data = st.checkbox("Has experimental validation?", value=False, key="risk_has_exp")
    
    if st.button("⚠️ Assess Risks", type="primary", key="risk_assess"):
        with st.spinner("Calculating risk scores..."):
            try:
                # Calculate all risk dimensions
                tox = analyzer.calculate_toxicity_score(formula_to_assess)
                supply = analyzer.calculate_supply_risk(formula_to_assess)
                trl = analyzer.calculate_trl(formula_to_assess, has_exp_data)
                reg = analyzer.calculate_regulatory_compliance(formula_to_assess)
                
                # Display results
                st.markdown(f"## Risk Assessment: {formula_to_assess}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Toxicity Score", f"{tox['toxicity_score']:.2f}")
                    st.caption(tox['toxicity_level'])
                
                with col2:
                    st.metric("Supply Risk", f"{supply['overall_risk_score']:.2f}")
                    st.caption(supply['risk_level'])
                
                with col3:
                    st.metric("TRL", f"{trl['trl']}/9")
                    st.caption(trl['description'][:30])
                
                with col4:
                    risk_color = "🟢" if reg['rohs_compliant'] else "🔴"
                    st.metric("RoHS Compliant", risk_color)
                    st.caption(reg['regulatory_risk'])
                
                # Detailed breakdowns
                st.markdown("---")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("### 🧪 Toxicity Details")
                    st.markdown(f"""
                    - **Pb Mass Fraction:** {tox['pb_mass_fraction']:.1%}
                    - **Pb-Free:** {tox['pb_free']}
                    - **Level:** {tox['toxicity_level']}
                    
                    ⚠️ Pb-based perovskites require encapsulation and recycling protocols.
                    """)
                
                with col_b:
                    st.markdown("### 🚢 Supply Chain Details")
                    st.markdown(f"""
                    - **Overall Risk:** {supply['overall_risk_score']:.2f} ({supply['risk_level']})
                    - **High-Risk Elements:** {', '.join(supply['high_risk_elements']) if supply['high_risk_elements'] else 'None'}
                    
                    💡 Diversify suppliers for high-risk elements.
                    """)
                
                # TRL & Regulatory
                st.markdown("---")
                
                col_c, col_d = st.columns(2)
                
                with col_c:
                    st.markdown("### 🔬 Technology Readiness")
                    st.markdown(f"""
                    - **TRL:** {trl['trl']}/9
                    - **Description:** {trl['description']}
                    - **Pilot-Ready:** {trl['ready_for_pilot']}
                    - **Commercial-Ready:** {trl['ready_for_commercialization']}
                    """)
                
                with col_d:
                    st.markdown("### 📜 Regulatory Compliance")
                    st.markdown(f"""
                    - **RoHS Compliant:** {reg['rohs_compliant']}
                    - **REACH Complexity:** {reg['reach_complexity']}
                    - **Overall Risk:** {reg['regulatory_risk']}
                    
                    {reg['notes']}
                    """)
                
                # Spider chart
                st.markdown("---")
                st.markdown("### 🕸️ Risk Radar Chart")
                
                fig_radar = analyzer.plot_scale_up_risk_radar(formula_to_assess, has_exp_data)
                st.plotly_chart(fig_radar, use_container_width=True)
                
                st.markdown("""
                **Interpretation:**
                - **Larger area = Lower risk** (better for scale-up)
                - Address low-scoring dimensions before commercialization
                """)
                
            except Exception as e:
                st.error(f"Risk assessment failed: {e}")

# =============================================================================
# TAB 10: PUBLICATION EXPORT (NEW V6)
# =============================================================================

with tabs[9]:
    st.markdown("## 📄 Publication-Ready Export")
    st.markdown("**LaTeX tables, 300 DPI figures, methods text, BibTeX**")
    
    if st.session_state.publication_exporter is None:
        st.session_state.publication_exporter = PublicationExporter()
    
    exporter = st.session_state.publication_exporter
    
    st.markdown("### 📦 Export Package Contents")
    
    # Check what's available
    has_inverse = st.session_state.inverse_candidates is not None and not st.session_state.inverse_candidates.empty
    has_pareto = st.session_state.pareto_optimal_materials is not None and not st.session_state.pareto_optimal_materials.empty
    has_cost = st.session_state.cost_analysis_results is not None and not st.session_state.cost_analysis_results.empty
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Data:**")
        st.markdown(f"- {'✅' if has_inverse else '❌'} Inverse Design Candidates")
        st.markdown(f"- {'✅' if has_pareto else '❌'} Pareto-Optimal Materials")
        st.markdown(f"- {'✅' if has_cost else '❌'} Cost Analysis Results")
    
    with col2:
        st.markdown("**Export Options:**")
        export_latex = st.checkbox("LaTeX Tables", value=True, key="exp_latex")
        export_csv = st.checkbox("CSV Tables", value=True, key="exp_csv")
        export_figs = st.checkbox("High-DPI Figures", value=True, key="exp_figs")
        export_methods = st.checkbox("Methods Section", value=True, key="exp_methods")
        export_bib = st.checkbox("BibTeX References", value=True, key="exp_bib")
    
    if st.button("📤 Generate Export Package", type="primary", key="pub_export"):
        with st.spinner("Generating publication package..."):
            try:
                figures = {}
                
                # Prepare candidates table
                if has_inverse:
                    candidates_export = st.session_state.inverse_candidates.head(20)
                    
                    if export_latex:
                        latex_table = exporter.export_table_latex(
                            candidates_export[['formula', 'predicted_bandgap', 'stability_score', 'cost_per_kg']],
                            caption="Top 20 candidates from generative inverse design",
                            label="tab:inverse_candidates",
                            filename="table_inverse_candidates.tex"
                        )
                    
                    if export_csv:
                        exporter.export_table_csv(candidates_export, "table_inverse_candidates.csv")
                
                # Prepare Pareto table
                if has_pareto:
                    pareto_export = st.session_state.pareto_optimal_materials.head(10)
                    
                    if export_latex:
                        exporter.export_table_latex(
                            pareto_export[['formula', 'bandgap', 'obj_bandgap_match', 'obj_stability', 'obj_cost']],
                            caption="Pareto-optimal materials from multi-objective optimization",
                            label="tab:pareto_optimal",
                            filename="table_pareto_optimal.tex"
                        )
                    
                    if export_csv:
                        exporter.export_table_csv(pareto_export, "table_pareto_optimal.csv")
                
                # Prepare cost table
                if has_cost:
                    if export_latex:
                        exporter.export_table_latex(
                            st.session_state.cost_analysis_results,
                            caption="Techno-economic analysis: cost per watt comparison",
                            label="tab:cost_analysis",
                            filename="table_cost_analysis.tex"
                        )
                    
                    if export_csv:
                        exporter.export_table_csv(st.session_state.cost_analysis_results, "table_cost_analysis.csv")
                
                # Generate figures (dummy - would use real plots)
                if export_figs:
                    # Graphical abstract
                    if has_inverse:
                        top_candidate = st.session_state.inverse_candidates.iloc[0]
                        fig_abstract = create_graphical_abstract(
                            top_candidate['formula'],
                            top_candidate['predicted_bandgap'],
                            0.25,  # placeholder
                            1
                        )
                        figures['graphical_abstract'] = fig_abstract
                        exporter.export_figure_png(fig_abstract, "graphical_abstract", width=1200, height=400, dpi=300)
                
                # Methods section
                if export_methods:
                    methods_text = exporter.generate_methods_section(
                        used_databases=['Materials Project', 'AFLOW', 'JARVIS'],
                        used_ml_models=['XGBoost', 'Gaussian Process'],
                        used_bo=st.session_state.bo_fitted,
                        used_mo=has_pareto,
                        n_experiments=len(st.session_state.user_data) if st.session_state.user_data is not None else 0
                    )
                    
                    with open(exporter.output_dir / 'methods_section.txt', 'w') as f:
                        f.write(methods_text)
                
                # BibTeX
                if export_bib:
                    exporter.generate_bibtex_file(
                        used_tools=['materials_project', 'xgboost', 'sklearn', 'gaussian_process', 'bayesian_optimization']
                    )
                
                # Create supplementary package
                if has_inverse and has_pareto:
                    si_dir = exporter.create_supplementary_package(
                        candidates_df=st.session_state.inverse_candidates.head(50),
                        pareto_df=st.session_state.pareto_optimal_materials,
                        cost_analysis_df=st.session_state.cost_analysis_results if has_cost else pd.DataFrame(),
                        figures=figures,
                        methods_text=methods_text if export_methods else ""
                    )
                    
                    st.success(f"✅ Publication package created at: {si_dir}")
                    st.info(f"📂 Files saved to: `{exporter.output_dir}`")
                else:
                    st.warning("⚠️ Need inverse design candidates and Pareto data for full package. Partial export completed.")
                    st.info(f"📂 Files saved to: `{exporter.output_dir}`")
                
                # Preview methods section
                if export_methods:
                    st.markdown("---")
                    st.markdown("### 📝 Methods Section Preview")
                    st.code(methods_text, language='markdown')
                
            except Exception as e:
                st.error(f"Export failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# =============================================================================
# TAB 11: DASHBOARD SUMMARY (NEW V6)
# =============================================================================

with tabs[10]:
    st.markdown("## 📊 Discovery Campaign Dashboard")
    st.markdown("**Complete overview of your discovery journey**")
    
    update_campaign_summary()
    summary = st.session_state.campaign_summary
    
    # Key metrics
    st.markdown("### 🎯 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Materials Screened", summary.get('total_materials_screened', 0))
        st.metric("Your Experiments", summary.get('user_experiments', 0))
    
    with col2:
        st.metric("Inverse Candidates", summary.get('inverse_candidates_generated', 0))
        st.metric("Pareto Optimal", summary.get('pareto_optimal_found', 0))
    
    with col3:
        model_status = "✅ Trained" if summary.get('model_trained', False) else "❌ Not trained"
        st.metric("ML Model", model_status)
        bo_status = "✅ Active" if summary.get('bo_active', False) else "❌ Inactive"
        st.metric("Bayesian Opt", bo_status)
    
    with col4:
        st.metric("Experiments Queued", summary.get('experiments_queued', 0))
        st.metric("Last Updated", summary.get('last_updated', 'N/A')[:19])
    
    # Campaign timeline
    st.markdown("---")
    st.markdown("### 📅 Campaign Timeline")
    
    timeline_data = []
    
    if st.session_state.db_loaded:
        timeline_data.append({'step': 'Database Loaded', 'status': '✅', 'count': summary.get('total_materials_screened', 0)})
    
    if st.session_state.user_data is not None:
        timeline_data.append({'step': 'User Data Uploaded', 'status': '✅', 'count': summary.get('user_experiments', 0)})
    
    if st.session_state.model_trained:
        timeline_data.append({'step': 'ML Model Trained', 'status': '✅', 'count': 1})
    
    if st.session_state.bo_fitted:
        timeline_data.append({'step': 'Bayesian Opt Fitted', 'status': '✅', 'count': 1})
    
    if st.session_state.inverse_candidates is not None:
        timeline_data.append({'step': 'Inverse Design Run', 'status': '✅', 'count': len(st.session_state.inverse_candidates)})
    
    if st.session_state.pareto_optimal_materials is not None:
        timeline_data.append({'step': 'Multi-Objective Analysis', 'status': '✅', 'count': len(st.session_state.pareto_optimal_materials)})
    
    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    else:
        st.info("Campaign not started yet. Load database (Tab 1) to begin.")
    
    # Best candidates summary
    st.markdown("---")
    st.markdown("### 🏆 Top Discoveries")
    
    if st.session_state.inverse_candidates is not None and not st.session_state.inverse_candidates.empty:
        st.markdown("#### Inverse Design Top 5")
        top_inverse = st.session_state.inverse_candidates.head(5)[['formula', 'predicted_bandgap', 'stability_score', 'cost_per_kg', 'combined_score']]
        st.dataframe(top_inverse, use_container_width=True, hide_index=True)
    
    if st.session_state.pareto_optimal_materials is not None and not st.session_state.pareto_optimal_materials.empty:
        st.markdown("#### Pareto-Optimal Top 5")
        top_pareto = st.session_state.pareto_optimal_materials.head(5)[['formula', 'bandgap', 'obj_bandgap_match', 'obj_stability', 'obj_cost']]
        st.dataframe(top_pareto, use_container_width=True, hide_index=True)
    
    # Export campaign report
    st.markdown("---")
    st.markdown("### 📥 Export Campaign Report")
    
    if st.button("📄 Export Full Report (HTML)", type="primary", key="dash_export"):
        with st.spinner("Generating HTML report..."):
            try:
                # Generate HTML report
                html_report = f"""
                <html>
                <head>
                    <title>AlphaMaterials V6 Campaign Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #667eea; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #667eea; color: white; }}
                    </style>
                </head>
                <body>
                    <h1>AlphaMaterials V6 Discovery Campaign Report</h1>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Campaign Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Materials Screened</td><td>{summary.get('total_materials_screened', 0)}</td></tr>
                        <tr><td>User Experiments</td><td>{summary.get('user_experiments', 0)}</td></tr>
                        <tr><td>Inverse Candidates Generated</td><td>{summary.get('inverse_candidates_generated', 0)}</td></tr>
                        <tr><td>Pareto Optimal Found</td><td>{summary.get('pareto_optimal_found', 0)}</td></tr>
                        <tr><td>Experiments Queued</td><td>{summary.get('experiments_queued', 0)}</td></tr>
                    </table>
                    
                    <h2>Top Inverse Design Candidates</h2>
                    {st.session_state.inverse_candidates.head(10).to_html() if st.session_state.inverse_candidates is not None else '<p>No data</p>'}
                    
                    <h2>Pareto-Optimal Materials</h2>
                    {st.session_state.pareto_optimal_materials.head(10).to_html() if st.session_state.pareto_optimal_materials is not None else '<p>No data</p>'}
                    
                    <hr>
                    <p><em>Generated by AlphaMaterials V6.0 | SAIT × SPMDL</em></p>
                </body>
                </html>
                """
                
                # Save to file
                report_path = Path('./exports') / f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                report_path.parent.mkdir(exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write(html_report)
                
                st.success(f"✅ Report exported to: {report_path}")
                
                # Download button
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_report,
                    file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    key="dash_download"
                )
                
            except Exception as e:
                st.error(f"Report export failed: {e}")

# =============================================================================
# TAB 12: SESSION MANAGER (V5, moved to last)
# =============================================================================

with tabs[11]:
    st.markdown("## 💾 Session Manager")
    st.markdown("**Save your discovery journey and resume later**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💾 Save Current Session")
        
        session_name = st.text_input(
            "Session Name",
            value=f"session_{datetime.now().strftime('%Y%m%d')}",
            key="sess_name"
        )
        
        session_desc = st.text_area(
            "Description (optional)",
            value="",
            key="sess_desc"
        )
        
        if st.button("💾 Save Session", type="primary", key="sess_save"):
            try:
                session_data = {
                    'description': session_desc,
                    'user_data': st.session_state.user_data,
                    'ml_model': st.session_state.ml_model,
                    'bo_state': {
                        'bo_optimizer': st.session_state.bo_optimizer,
                        'bo_history': st.session_state.bo_history,
                    },
                    'inverse_candidates': st.session_state.inverse_candidates,
                    'pareto_optimal_materials': st.session_state.pareto_optimal_materials,
                    'cost_analysis_results': st.session_state.cost_analysis_results,
                    'campaign_summary': st.session_state.campaign_summary,
                }
                
                session_path = st.session_state.session_manager.save_session(session_data, session_name)
                st.success(f"✅ Session saved to: {session_path}")
                
            except Exception as e:
                st.error(f"Save failed: {e}")
    
    with col2:
        st.markdown("### 📂 Load Saved Session")
        
        sessions_df = st.session_state.session_manager.list_sessions()
        
        if not sessions_df.empty:
            session_options = sessions_df['session_name'].tolist()
            selected_session = st.selectbox("Select session", session_options, key="sess_select")
            
            if st.button("📂 Load Session", key="sess_load"):
                try:
                    session_data = st.session_state.session_manager.load_session(selected_session)
                    
                    # Restore state
                    if 'user_data' in session_data:
                        st.session_state.user_data = session_data['user_data']
                    
                    if 'ml_model' in session_data:
                        st.session_state.ml_model = session_data['ml_model']
                        st.session_state.model_trained = True
                    
                    if 'inverse_candidates' in session_data:
                        st.session_state.inverse_candidates = session_data['inverse_candidates']
                    
                    if 'pareto_optimal_materials' in session_data:
                        st.session_state.pareto_optimal_materials = session_data['pareto_optimal_materials']
                    
                    if 'cost_analysis_results' in session_data:
                        st.session_state.cost_analysis_results = session_data['cost_analysis_results']
                    
                    st.success("✅ Session loaded!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Load failed: {e}")
        else:
            st.info("No saved sessions yet")
    
    # Session browser
    st.markdown("---")
    st.markdown("### 📊 All Sessions")
    
    if not sessions_df.empty:
        st.dataframe(sessions_df[['session_name', 'created_at', 'version']], use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">
    <p><b>AlphaMaterials V6.0 — Generative Inverse Design + Techno-Economics</b></p>
    <p>SAIT × SPMDL Collaboration | 2026-03-15</p>
    <p>🚀 Target → Generate → Optimize → Validate → Commercialize</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">빈 지도가 탐험의 시작 — The empty map is the start of exploration</p>
</div>
""", unsafe_allow_html=True)
