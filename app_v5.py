#!/usr/bin/env python3
"""
AlphaMaterials V5: Personalized Learning Platform
==================================================

Evolution from V4 → V5: Transform connected platform into autonomous discovery engine

New in V5:
- Bayesian Optimization (suggest next experiments)
- Surrogate model fine-tuning (your data makes it smarter)
- Multi-objective optimization (bandgap + stability + cost + synthesizability)
- Experiment Planner (prioritized experiment queue)
- Session Persistence (save/load your discovery journey)

SAIT × SPMDL Collaboration Platform
V5.0 - Personalized Learning Platform

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
    from data_parser import UserDataParser, example_csv, example_excel_description
    from ml_models import BandgapPredictor, CompositionFeaturizer, train_default_model
    from bayesian_opt import BayesianOptimizer, compare_acquisition_functions
    from multi_objective import MultiObjectiveOptimizer, default_weights
    from session import SessionManager, create_default_session
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import V5 modules: {e}")
    MODULES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AlphaMaterials V5: Personalized Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (V5 branding)
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #1a1a2e;
    }
    
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp td, .stApp th {
        color: #1a1a2e !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #16213e !important;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
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
    
    .v5-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 1rem;
    }
    
    .metric-card {
        background: #eef2ff;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #1a1a2e !important;
    }
    
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1a1a2e !important;
    }
    
    .learning-box {
        background: #fef5e7;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1a1a2e !important;
    }
    
    .confidence-high { color: #48bb78; font-weight: bold; }
    .confidence-medium { color: #ed8936; font-weight: bold; }
    .confidence-low { color: #f56565; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables"""
    # V4 states (preserved)
    if 'db_client' not in st.session_state:
        st.session_state.db_client = None
    if 'db_data' not in st.session_state:
        st.session_state.db_data = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'combined_data' not in st.session_state:
        st.session_state.combined_data = None
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'db_loaded' not in st.session_state:
        st.session_state.db_loaded = False
    
    # V5 new states
    if 'bo_optimizer' not in st.session_state:
        st.session_state.bo_optimizer = None
    if 'bo_fitted' not in st.session_state:
        st.session_state.bo_fitted = False
    if 'bo_history' not in st.session_state:
        st.session_state.bo_history = None
    if 'mo_optimizer' not in st.session_state:
        st.session_state.mo_optimizer = None
    if 'mo_weights' not in st.session_state:
        st.session_state.mo_weights = default_weights()
    if 'experiment_queue' not in st.session_state:
        st.session_state.experiment_queue = None
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if 'current_session' not in st.session_state:
        st.session_state.current_session = create_default_session()
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []

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
            'formula': ['MAPbI3', 'FAPbI3', 'CsPbI3', 'MAPbBr3', 'FAPbBr3'],
            'bandgap': [1.59, 1.51, 1.72, 2.30, 2.25],
            'source': ['fallback'] * 5
        })

# =============================================================================
# MAIN APP
# =============================================================================

# Title
st.markdown('<h1 class="main-title">AlphaMaterials<span class="v5-badge">V5: Personalized Learning</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">DFT Expert + MLIP + Bayesian Optimization = Autonomous Discovery Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="font-size: 1.0rem; margin-top: -1.5rem;">Upload Data → Model Learns → BO Suggests → Experiment → Repeat</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 V5 Navigation")
    st.markdown("---")
    
    st.markdown("**New in V5:**")
    st.success("✅ Bayesian Optimization\n\n✅ Model fine-tuning\n\n✅ Multi-objective Pareto\n\n✅ Experiment planner\n\n✅ Session save/load")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Target bandgap
    target_bandgap = st.number_input(
        "🎯 Target Bandgap (eV)",
        min_value=0.5,
        max_value=3.0,
        value=1.68,
        step=0.01,
        help="Target bandgap for Bayesian Optimization"
    )
    st.session_state.current_session['bo_state']['target_bandgap'] = target_bandgap
    
    # Acquisition function
    acq_func = st.selectbox(
        "Acquisition Function",
        ['ei', 'ucb', 'ts'],
        format_func=lambda x: {'ei': 'Expected Improvement', 'ucb': 'Upper Confidence Bound', 'ts': 'Thompson Sampling'}[x],
        help="Strategy for suggesting next experiments"
    )
    st.session_state.current_session['bo_state']['acq_function'] = acq_func
    
    st.markdown("---")
    
    # Status dashboard
    st.markdown("### 📊 Status Dashboard")
    
    if st.session_state.db_loaded:
        n_total = len(st.session_state.combined_data) if st.session_state.combined_data is not None else 0
        n_user = len(st.session_state.user_data) if st.session_state.user_data is not None else 0
        st.metric("Total Materials", n_total)
        st.metric("Your Data", n_user)
    
    if st.session_state.model_trained:
        st.metric("Model Status", "✅ Trained")
    else:
        st.metric("Model Status", "⏸️ Not trained")
    
    if st.session_state.bo_fitted:
        st.metric("BO Status", "✅ Active")
    else:
        st.metric("BO Status", "⏸️ Inactive")
    
    st.markdown("---")
    st.markdown("**Version:** V5.0-Learning")
    st.markdown("**Date:** 2026-03-15")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗄️ Database",
    "📤 Upload Data",
    "🤖 ML Surrogate",
    "🎯 Bayesian Optimization",
    "🏆 Multi-Objective",
    "📋 Experiment Planner",
    "💾 Session Manager"
])

# =============================================================================
# TAB 1: DATABASE (Same as V4, condensed)
# =============================================================================

with tab1:
    st.markdown("## 🗄️ Database Explorer")
    st.markdown("**Load perovskite database from Materials Project, AFLOW, JARVIS**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Load Database", type="primary"):
            with st.spinner("Fetching from databases..."):
                try:
                    api_key = st.session_state.get('mp_api_key', None)
                    st.session_state.db_client = UnifiedDBClient(mp_api_key=api_key)
                    
                    db_data = st.session_state.db_client.get_all_perovskites(
                        max_per_source=200,
                        use_cache=True
                    )
                    
                    if db_data.empty:
                        st.warning("No API data. Loading sample data...")
                        db_data = load_sample_data()
                    
                    st.session_state.db_data = db_data
                    st.session_state.combined_data = db_data.copy()
                    st.session_state.db_loaded = True
                    
                    st.success(f"✅ Loaded {len(db_data)} materials!")
                    
                except Exception as e:
                    st.error(f"Database load failed: {e}")
                    sample_data = load_sample_data()
                    st.session_state.db_data = sample_data
                    st.session_state.combined_data = sample_data.copy()
                    st.session_state.db_loaded = True
    
    with col2:
        st.info("**빈 지도가 탐험의 시작**\n\nThe empty map is the start of exploration")
    
    # Show database table if loaded
    if st.session_state.db_loaded and st.session_state.db_data is not None:
        st.markdown("---")
        st.markdown("### 📋 Database Contents")
        
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
# TAB 2: UPLOAD DATA (Same as V4)
# =============================================================================

with tab2:
    st.markdown("## 📤 Upload Your Experimental Data")
    st.markdown("**This is where the personalized learning begins!**")
    
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Must contain 'formula' and 'bandgap' columns"
    )
    
    if uploaded_file is not None:
        try:
            parser = UserDataParser()
            file_content = uploaded_file.read()
            df_user = parser.parse(file_content, uploaded_file.name)
            
            if not df_user.empty:
                st.success(f"✅ Parsed {len(df_user)} materials")
                
                # Show parsed data
                st.dataframe(df_user, use_container_width=True)
                
                # Summary
                summary = parser.get_summary()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Materials", summary.get('n_materials', 0))
                with col_b:
                    bg_range = summary.get('bandgap_range', (None, None))
                    if bg_range[0] is not None:
                        st.metric("Bandgap Range", f"{bg_range[0]:.2f} - {bg_range[1]:.2f} eV")
                
                # Merge button
                if st.button("💾 Save to Session", type="primary"):
                    if st.session_state.db_loaded:
                        st.session_state.user_data = df_user
                        st.session_state.combined_data = parser.merge_with_db(
                            df_user,
                            st.session_state.db_data
                        )
                        st.session_state.current_session['user_data'] = df_user
                        st.success("✅ Data saved! Go to Tab 3 to train/fine-tune model.")
                    else:
                        st.error("Please load database first (Tab 1)")
        
        except Exception as e:
            st.error(f"Upload error: {e}")

# =============================================================================
# TAB 3: ML SURROGATE (V4 + Fine-tuning)
# =============================================================================

with tab3:
    st.markdown("## 🤖 ML Surrogate Model")
    st.markdown("**Train on database, then fine-tune on your data**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏋️ Initial Training (Database)")
        
        if st.button("🚀 Train Base Model", type="primary", disabled=not st.session_state.db_loaded):
            with st.spinner("Training XGBoost model on database..."):
                try:
                    df_train = st.session_state.combined_data
                    df_train = df_train[
                        df_train['bandgap'].notna() & 
                        (df_train['bandgap'] > 0) &
                        (df_train['bandgap'] < 10)
                    ]
                    
                    model = BandgapPredictor(use_xgboost=True)
                    metrics = model.train(df_train, formula_col='formula', target_col='bandgap')
                    
                    st.session_state.ml_model = model
                    st.session_state.model_trained = True
                    st.session_state.train_metrics = metrics
                    st.session_state.current_session['ml_model'] = model
                    
                    # Log training
                    st.session_state.training_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'initial_training',
                        'n_samples': metrics['n_samples'],
                        'cv_mae': metrics['cv_mae'],
                        'r2': metrics['train_r2']
                    })
                    
                    st.success("✅ Base model trained!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Training Samples", metrics['n_samples'])
                    with col_b:
                        st.metric("CV MAE", f"{metrics['cv_mae']:.3f} eV")
                    with col_c:
                        st.metric("R² Score", f"{metrics['train_r2']:.3f}")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        st.markdown("---")
        st.markdown("### ⚡ Fine-tuning (Your Data)")
        
        if st.session_state.user_data is not None and st.session_state.model_trained:
            st.markdown("**Your data makes the model smarter!**")
            
            learning_rate = st.slider(
                "Fine-tuning Aggressiveness",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Lower = more conservative (preserve database knowledge)"
            )
            
            if st.button("🔥 Fine-tune on Your Data", type="primary"):
                with st.spinner("Fine-tuning model..."):
                    try:
                        model = st.session_state.ml_model
                        ft_metrics = model.fine_tune(
                            st.session_state.user_data,
                            learning_rate=learning_rate
                        )
                        
                        st.success("✅ Model fine-tuned!")
                        
                        # Log fine-tuning
                        st.session_state.training_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'fine_tuning',
                            'n_new_samples': ft_metrics['n_new_samples'],
                            'mae_before': ft_metrics['mae_before'],
                            'mae_after': ft_metrics['mae_after'],
                            'mae_improvement': ft_metrics['mae_improvement'],
                            'r2_improvement': ft_metrics['r2_improvement']
                        })
                        
                        # Show before/after
                        st.markdown("### 📈 Fine-tuning Results")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric(
                                "MAE on Your Data (Before)",
                                f"{ft_metrics['mae_before']:.3f} eV"
                            )
                        with col_b:
                            st.metric(
                                "MAE on Your Data (After)",
                                f"{ft_metrics['mae_after']:.3f} eV",
                                delta=f"{-ft_metrics['mae_improvement']:.3f} eV",
                                delta_color="inverse"
                            )
                        with col_c:
                            st.metric(
                                "R² Improvement",
                                f"+{ft_metrics['r2_improvement']:.3f}"
                            )
                        
                        # Visualization
                        fig = go.Figure()
                        
                        metrics_df = pd.DataFrame([
                            {'Stage': 'Before Fine-tuning', 'MAE': ft_metrics['mae_before']},
                            {'Stage': 'After Fine-tuning', 'MAE': ft_metrics['mae_after']}
                        ])
                        
                        fig.add_trace(go.Bar(
                            x=metrics_df['Stage'],
                            y=metrics_df['MAE'],
                            marker_color=['#f56565', '#48bb78']
                        ))
                        
                        fig.update_layout(
                            title="Model Accuracy Improvement on Your Data",
                            yaxis_title="Mean Absolute Error (eV)",
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('<div class="learning-box">', unsafe_allow_html=True)
                        st.markdown(f"""
                        **🎓 Your data made the model {ft_metrics['mae_improvement']/ft_metrics['mae_before']*100:.1f}% more accurate!**
                        
                        The model now understands your specific experimental conditions better.
                        This personalized model will give better BO suggestions.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Fine-tuning failed: {e}")
        else:
            st.info("💡 Upload your data (Tab 2) and train base model first")
    
    with col2:
        st.markdown("### ℹ️ Model Info")
        
        if st.session_state.model_trained:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            metrics = st.session_state.train_metrics
            st.markdown(f"""
            **Model Ready!**
            
            - Samples: {metrics['n_samples']}
            - MAE: {metrics['cv_mae']:.3f} eV
            - R²: {metrics['train_r2']:.3f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Training history
        if st.session_state.training_history:
            st.markdown("### 📊 Training History")
            
            history_df = pd.DataFrame(st.session_state.training_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            st.dataframe(history_df[['timestamp', 'type']], use_container_width=True)

# =============================================================================
# TAB 4: BAYESIAN OPTIMIZATION
# =============================================================================

with tab4:
    st.markdown("## 🎯 Bayesian Optimization")
    st.markdown("**Let AI suggest your next experiments**")
    
    if not st.session_state.model_trained or st.session_state.user_data is None:
        st.info("💡 Please train model and upload your data first (Tabs 2 & 3)")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🧠 Fit Bayesian Optimizer")
            
            st.markdown("""
            **How it works:**
            1. Gaussian Process learns from your experimental data
            2. Acquisition function balances exploration vs exploitation
            3. Suggests compositions with highest potential for discovery
            """)
            
            if st.button("🚀 Fit BO on Your Data", type="primary"):
                with st.spinner("Fitting Gaussian Process..."):
                    try:
                        bo = BayesianOptimizer(
                            target_bandgap=target_bandgap,
                            acq_function=acq_func
                        )
                        
                        bo_metrics = bo.fit(
                            st.session_state.user_data,
                            formula_col='formula',
                            target_col='bandgap'
                        )
                        
                        st.session_state.bo_optimizer = bo
                        st.session_state.bo_fitted = True
                        st.session_state.current_session['bo_state']['bo_optimizer'] = bo
                        
                        st.success("✅ Bayesian Optimizer ready!")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Training Samples", bo_metrics['n_samples'])
                        with col_b:
                            st.metric("Target Bandgap", f"{target_bandgap} eV")
                        
                        # Convergence plot
                        fig = bo.get_convergence_plot()
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"BO fitting failed: {e}")
            
            # Generate suggestions
            if st.session_state.bo_fitted:
                st.markdown("---")
                st.markdown("### 🎲 Generate Experiment Suggestions")
                
                n_suggestions = st.slider("Number of suggestions", 1, 20, 5)
                n_candidates = st.slider("Candidate pool size", 100, 5000, 1000)
                
                if st.button("🔮 Suggest Next Experiments"):
                    with st.spinner(f"Evaluating {n_candidates} candidates..."):
                        try:
                            bo = st.session_state.bo_optimizer
                            
                            # Generate candidate compositions
                            search_space = {
                                'A': ['MA', 'FA', 'Cs'],
                                'B': ['Pb', 'Sn'],
                                'X': ['I', 'Br', 'Cl']
                            }
                            
                            suggestions = bo.optimize_composition(
                                search_space=search_space,
                                n_samples=n_candidates
                            )
                            
                            st.session_state.bo_history = suggestions
                            
                            st.markdown("### 🏆 Top Experiment Suggestions")
                            
                            # Display suggestions
                            display_cols = [
                                'rank', 'formula', 'predicted_bandgap', 
                                'uncertainty', 'acquisition_value', 'distance_to_target'
                            ]
                            
                            st.dataframe(
                                suggestions[display_cols].head(n_suggestions),
                                use_container_width=True
                            )
                            
                            # Save to experiment queue
                            if st.button("➕ Add Top 5 to Experiment Queue"):
                                if st.session_state.experiment_queue is None:
                                    st.session_state.experiment_queue = suggestions.head(5).copy()
                                else:
                                    st.session_state.experiment_queue = pd.concat([
                                        st.session_state.experiment_queue,
                                        suggestions.head(5)
                                    ], ignore_index=True)
                                
                                st.success("✅ Added to experiment queue (see Tab 6)")
                            
                            # Acquisition landscape
                            st.markdown("### 🗺️ Acquisition Function Landscape")
                            
                            # Sample for visualization
                            sample_candidates = suggestions['formula'].head(200).tolist()
                            fig_acq = bo.plot_acquisition_landscape(sample_candidates)
                            st.plotly_chart(fig_acq, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Suggestion failed: {e}")
        
        with col2:
            st.markdown("### 📊 Acquisition Function")
            
            st.markdown(f"""
            **Current:** {acq_func.upper()}
            
            - **EI:** Expected Improvement (balanced)
            - **UCB:** Upper Confidence Bound (exploration)
            - **TS:** Thompson Sampling (stochastic)
            """)
            
            if st.session_state.bo_fitted:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("**BO Active ✅**\n\nReady to suggest experiments")
                st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 5: MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================

with tab5:
    st.markdown("## 🏆 Multi-Objective Optimization")
    st.markdown("**Optimize bandgap + stability + cost + synthesizability simultaneously**")
    
    if not st.session_state.model_trained:
        st.info("💡 Please train model first (Tab 3)")
    else:
        st.markdown("### ⚖️ Set Objective Weights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            w_bandgap = st.slider("Bandgap Match", 0.0, 1.0, 0.4, 0.05)
        with col2:
            w_stability = st.slider("Stability", 0.0, 1.0, 0.3, 0.05)
        with col3:
            w_synth = st.slider("Synthesizability", 0.0, 1.0, 0.2, 0.05)
        with col4:
            w_cost = st.slider("Low Cost", 0.0, 1.0, 0.1, 0.05)
        
        # Normalize weights
        total_weight = w_bandgap + w_stability + w_synth + w_cost
        if total_weight > 0:
            weights = {
                'obj_bandgap_match': w_bandgap / total_weight,
                'obj_stability': w_stability / total_weight,
                'obj_synthesizability': w_synth / total_weight,
                'obj_cost': w_cost / total_weight
            }
            st.session_state.mo_weights = weights
        
        # Evaluate candidates
        if st.button("🎯 Evaluate Multi-Objective", type="primary"):
            with st.spinner("Evaluating objectives..."):
                try:
                    mo = MultiObjectiveOptimizer(target_bandgap=target_bandgap)
                    st.session_state.mo_optimizer = mo
                    
                    # Generate candidates
                    if st.session_state.bo_history is not None:
                        candidates = st.session_state.bo_history['formula'].head(100).tolist()
                        bandgaps = st.session_state.bo_history['predicted_bandgap'].head(100).values
                    else:
                        # Generate random candidates
                        bo_temp = BayesianOptimizer(target_bandgap=target_bandgap)
                        bo_temp.featurizer = CompositionFeaturizer()
                        candidates = bo_temp._generate_candidates({
                            'A': ['MA', 'FA', 'Cs'],
                            'B': ['Pb', 'Sn'],
                            'X': ['I', 'Br', 'Cl']
                        }, 100)
                        
                        # Predict bandgaps if model available
                        if st.session_state.ml_model:
                            bandgaps, _ = st.session_state.ml_model.predict(candidates)
                        else:
                            bandgaps = None
                    
                    # Evaluate all objectives
                    obj_df = mo.evaluate_objectives(candidates, bandgaps)
                    
                    # Calculate Pareto front
                    pareto_df = mo.calculate_pareto_front(
                        obj_df,
                        ['obj_bandgap_match', 'obj_stability', 'obj_synthesizability', 'obj_cost']
                    )
                    
                    st.markdown("### 🌟 Pareto-Optimal Materials")
                    st.markdown(f"**{len(pareto_df)} / {len(obj_df)} materials are Pareto-optimal**")
                    
                    st.dataframe(pareto_df[
                        ['formula', 'bandgap', 'tolerance_factor', 
                         'obj_bandgap_match', 'obj_stability', 'obj_synthesizability', 'obj_cost']
                    ], use_container_width=True)
                    
                    # 2D Pareto fronts
                    st.markdown("### 📊 2D Pareto Fronts")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        fig1 = mo.plot_pareto_front_2d(
                            obj_df, 'obj_bandgap_match', 'obj_stability', pareto_df
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_b:
                        fig2 = mo.plot_pareto_front_2d(
                            obj_df, 'obj_cost', 'obj_synthesizability', pareto_df
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 3D Pareto front
                    st.markdown("### 🎨 3D Pareto Front")
                    
                    fig3 = mo.plot_pareto_front_3d(
                        obj_df, 'obj_bandgap_match', 'obj_stability', 'obj_cost', pareto_df
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Weighted recommendations
                    st.markdown("### 🎯 Top Recommendations (Weighted Score)")
                    
                    recommendations = mo.get_recommendations(obj_df, weights, n_top=10)
                    st.dataframe(recommendations[
                        ['rank', 'formula', 'bandgap', 'weighted_score']
                    ], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Multi-objective evaluation failed: {e}")

# =============================================================================
# TAB 6: EXPERIMENT PLANNER
# =============================================================================

with tab6:
    st.markdown("## 📋 Experiment Planner")
    st.markdown("**Prioritized experiment queue based on BO + multi-objective results**")
    
    if st.session_state.experiment_queue is None or st.session_state.experiment_queue.empty:
        st.info("💡 No experiments queued yet. Generate suggestions in Tab 4 (Bayesian Optimization)")
    else:
        queue = st.session_state.experiment_queue
        
        st.markdown(f"### 🧪 Experiment Queue ({len(queue)} experiments)")
        
        # Enhanced queue with synthesis difficulty
        queue_display = queue.copy()
        
        # Add synthesis difficulty estimate (based on complexity)
        def estimate_difficulty(formula):
            # Simple heuristic: more mixing = harder
            n_elements = formula.count('0.') + formula.count('0,')
            if n_elements >= 3:
                return "Hard"
            elif n_elements >= 1:
                return "Medium"
            else:
                return "Easy"
        
        queue_display['synthesis_difficulty'] = queue_display['formula'].apply(estimate_difficulty)
        
        # Display queue
        st.dataframe(queue_display[
            ['rank', 'formula', 'predicted_bandgap', 'uncertainty', 
             'acquisition_value', 'synthesis_difficulty']
        ], use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Queue")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = queue_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"experiment_queue_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Clear queue
            if st.button("🗑️ Clear Queue"):
                st.session_state.experiment_queue = None
                st.rerun()
        
        # Prioritization advice
        st.markdown("### 💡 Prioritization Advice")
        
        st.markdown("""
        **How to prioritize:**
        
        1. **High acquisition value** = Highest learning potential
        2. **Low uncertainty** = More confident predictions
        3. **Easy synthesis** = Faster turnaround
        4. **Low cost** = Budget-friendly
        
        **Recommended strategy:** Start with top 3 easy/medium experiments, 
        then tackle harder ones as you validate the model.
        """)

# =============================================================================
# TAB 7: SESSION MANAGER
# =============================================================================

with tab7:
    st.markdown("## 💾 Session Manager")
    st.markdown("**Save your discovery journey and resume later**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💾 Save Current Session")
        
        session_name = st.text_input(
            "Session Name",
            value=f"session_{datetime.now().strftime('%Y%m%d')}",
            help="Give your session a memorable name"
        )
        
        session_desc = st.text_area(
            "Description (optional)",
            value="",
            help="What were you exploring?"
        )
        
        if st.button("💾 Save Session", type="primary"):
            try:
                # Prepare session data
                session_data = {
                    'description': session_desc,
                    'user_data': st.session_state.user_data,
                    'ml_model': st.session_state.ml_model,
                    'bo_state': {
                        'bo_optimizer': st.session_state.bo_optimizer,
                        'bo_history': st.session_state.bo_history,
                        'target_bandgap': target_bandgap,
                        'acq_function': acq_func,
                        'n_iterations': len(st.session_state.bo_history) if st.session_state.bo_history is not None else 0
                    },
                    'mo_weights': st.session_state.mo_weights,
                    'experiment_queue': st.session_state.experiment_queue,
                    'training_history': st.session_state.training_history
                }
                
                # Save
                session_path = st.session_state.session_manager.save_session(
                    session_data,
                    session_name
                )
                
                st.success(f"✅ Session saved to: {session_path}")
                
            except Exception as e:
                st.error(f"Save failed: {e}")
    
    with col2:
        st.markdown("### 📂 Load Saved Session")
        
        # List sessions
        sessions_df = st.session_state.session_manager.list_sessions()
        
        if not sessions_df.empty:
            session_options = sessions_df['session_name'].tolist()
            selected_session = st.selectbox("Select session", session_options)
            
            if st.button("📂 Load Session"):
                try:
                    session_data = st.session_state.session_manager.load_session(selected_session)
                    
                    # Restore state
                    if 'user_data' in session_data:
                        st.session_state.user_data = session_data['user_data']
                    
                    if 'ml_model' in session_data:
                        st.session_state.ml_model = session_data['ml_model']
                        st.session_state.model_trained = True
                    
                    if 'bo_state' in session_data:
                        bo_state = session_data['bo_state']
                        if 'bo_optimizer' in bo_state:
                            st.session_state.bo_optimizer = bo_state['bo_optimizer']
                            st.session_state.bo_fitted = True
                        if 'bo_history' in bo_state:
                            st.session_state.bo_history = bo_state['bo_history']
                    
                    if 'mo_weights' in session_data:
                        st.session_state.mo_weights = session_data['mo_weights']
                    
                    if 'experiment_queue' in session_data:
                        st.session_state.experiment_queue = session_data['experiment_queue']
                    
                    if 'training_history' in session_data:
                        st.session_state.training_history = session_data['training_history']
                    
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
        st.dataframe(sessions_df[
            ['session_name', 'created_at', 'version', 'size_mb']
        ], use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.9rem;">
    <p><b>AlphaMaterials V5.0 — Personalized Learning Platform</b> | SAIT × SPMDL | 2026-03-15</p>
    <p>🧠 Upload Data → Model Learns → BO Suggests → Experiment → Repeat → Discovery</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">빈 지도가 탐험의 시작 — The empty map is the start of exploration</p>
</div>
""", unsafe_allow_html=True)
