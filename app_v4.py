#!/usr/bin/env python3
"""
AlphaMaterials V4: Connected Platform
=====================================

Evolution from V3 → V4: Transform hardcoded demo into real data-driven tool

New in V4:
- Real database integration (Materials Project, AFLOW, JARVIS-DFT)
- User data upload (CSV/Excel)
- Property space mapping (your data in context)
- Expanded composition space (16 → thousands)
- Lightweight ML surrogate (XGBoost bandgap predictor)

SAIT × SPMDL Collaboration Platform
V4.0 - Connected Platform

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

# Import V4 modules
try:
    from db_clients import UnifiedDBClient, CacheDB
    from data_parser import UserDataParser, example_csv, example_excel_description
    from ml_models import BandgapPredictor, CompositionFeaturizer, train_default_model
    DB_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import V4 modules: {e}")
    DB_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AlphaMaterials V4: Connected Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (preserved from V3 with V4 branding)
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
    
    .v4-badge {
        display: inline-block;
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
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
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
    
    .limitation-box {
        background: #fef2f2;
        border-left: 4px solid #f56565;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
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
    if 'selected_material' not in st.session_state:
        st.session_state.selected_material = None
    if 'db_loaded' not in st.session_state:
        st.session_state.db_loaded = False

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

def get_confidence_badge(source: str) -> str:
    """Return confidence badge based on data source"""
    if source in ['literature', 'experimental', 'materials_project']:
        return '<span class="confidence-high">★★★ Experimental</span>'
    elif source in ['dft', 'aflow', 'jarvis']:
        return '<span class="confidence-medium">★★ DFT</span>'
    elif source == 'user_upload':
        return '<span class="confidence-high">★★★ Your Data</span>'
    else:
        return '<span class="confidence-low">★ Prediction</span>'

# =============================================================================
# MAIN APP
# =============================================================================

# Title
st.markdown('<h1 class="main-title">AlphaMaterials<span class="v4-badge">V4: Connected Platform</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Driven Design of Infinite & Dynamic All-Perovskite Tandem PV</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="font-size: 1.0rem; margin-top: -1.5rem;">From Hardcoded Demo → Real Data-Driven Discovery Tool</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 V4 Navigation")
    st.markdown("---")
    
    st.markdown("**New in V4:**")
    st.success("✅ Live database integration\n\n✅ User data upload\n\n✅ Property space mapping\n\n✅ ML surrogate model")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # API key input (optional)
    with st.expander("🔑 API Keys (Optional)"):
        mp_api_key = st.text_input(
            "Materials Project API Key", 
            type="password",
            help="Get free key at materialsproject.org"
        )
        if mp_api_key:
            st.session_state.mp_api_key = mp_api_key
        
        st.info("💡 App works without API keys using bundled sample data")
    
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_limitations = st.checkbox("Show limitations", value=True)
    
    st.markdown("---")
    
    # Database status
    st.markdown("### 📊 Database Status")
    
    if st.session_state.db_loaded and st.session_state.combined_data is not None:
        n_total = len(st.session_state.combined_data)
        n_db = len(st.session_state.combined_data[st.session_state.combined_data['source'] != 'user_upload'])
        n_user = len(st.session_state.combined_data[st.session_state.combined_data['source'] == 'user_upload'])
        
        st.metric("Total Materials", n_total)
        st.metric("From Databases", n_db)
        st.metric("User Uploaded", n_user)
    else:
        st.info("Click 'Load Database' in Tab 1")
    
    st.markdown("---")
    st.markdown("**Version:** V4.0-Connected")
    st.markdown("**Date:** 2026-03-15")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗄️ Database Explorer",
    "📤 Upload Your Data",
    "🗺️ Property Space Map",
    "🤖 ML Surrogate",
    "🔬 Why AI? (V3 Demo)"
])

# =============================================================================
# TAB 1: DATABASE EXPLORER
# =============================================================================

with tab1:
    st.markdown("## 🗄️ Connected Database Explorer")
    st.markdown("**Real-time access to Materials Project, AFLOW, JARVIS-DFT perovskite databases**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📡 Data Sources")
        
        st.markdown("""
        **V4 connects to public materials databases:**
        
        - **Materials Project** (mp-api): 150k+ DFT calculations, RESTful API
        - **AFLOW** (aflowlib.duke.edu): High-throughput DFT library
        - **JARVIS-DFT** (NIST): Specialized perovskite database
        - **Local cache**: SQLite for offline access after first fetch
        - **Fallback**: Bundled sample data if APIs unavailable
        """)
        
        if st.button("🚀 Load Database", type="primary"):
            with st.spinner("Fetching data from databases..."):
                try:
                    # Initialize DB client
                    api_key = st.session_state.get('mp_api_key', None)
                    st.session_state.db_client = UnifiedDBClient(mp_api_key=api_key)
                    
                    # Try to load from databases
                    db_data = st.session_state.db_client.get_all_perovskites(
                        max_per_source=200,
                        use_cache=True
                    )
                    
                    if db_data.empty:
                        st.warning("⚠️ No data from APIs. Loading sample data...")
                        db_data = load_sample_data()
                    
                    st.session_state.db_data = db_data
                    st.session_state.combined_data = db_data.copy()
                    st.session_state.db_loaded = True
                    
                    st.success(f"✅ Loaded {len(db_data)} materials!")
                    
                except Exception as e:
                    st.error(f"Database load failed: {e}")
                    st.info("Loading bundled sample data instead...")
                    sample_data = load_sample_data()
                    st.session_state.db_data = sample_data
                    st.session_state.combined_data = sample_data.copy()
                    st.session_state.db_loaded = True
    
    with col2:
        st.markdown("### ℹ️ Info")
        
        st.info("""
        **빈 지도가 탐험의 시작**
        
        V3 had 16 hardcoded compositions.
        V4 connects to thousands.
        
        The empty map is the start of exploration.
        """)
        
        if st.session_state.db_loaded:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**Database loaded!**\n\nProceed to explore data below.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display database if loaded
    if st.session_state.db_loaded and st.session_state.db_data is not None:
        st.markdown("---")
        st.markdown("### 📋 Database Contents")
        
        df = st.session_state.db_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Materials", len(df))
        with col2:
            if 'bandgap' in df.columns:
                st.metric("Bandgap Range", f"{df['bandgap'].min():.2f} - {df['bandgap'].max():.2f} eV")
        with col3:
            if 'source' in df.columns:
                st.metric("Data Sources", df['source'].nunique())
        with col4:
            if 'is_stable' in df.columns:
                n_stable = df['is_stable'].sum() if df['is_stable'].dtype == bool else 0
                st.metric("Stable Phases", n_stable)
        
        # Bandgap distribution
        st.markdown("### 📊 Bandgap Distribution")
        
        if 'bandgap' in df.columns and 'formula' in df.columns:
            fig = px.histogram(
                df, 
                x='bandgap',
                nbins=50,
                color='source' if 'source' in df.columns else None,
                title="Bandgap Distribution Across Database",
                labels={'bandgap': 'Bandgap (eV)', 'count': 'Number of Materials'}
            )
            
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a2e'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive table
        st.markdown("### 🔍 Browse Materials")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'bandgap' in df.columns:
                bg_min, bg_max = st.slider(
                    "Bandgap Range (eV)",
                    float(df['bandgap'].min()),
                    float(df['bandgap'].max()),
                    (float(df['bandgap'].min()), float(df['bandgap'].max())),
                    0.1
                )
        
        with col2:
            if 'source' in df.columns:
                sources = ['All'] + list(df['source'].unique())
                selected_source = st.selectbox("Source", sources)
        
        with col3:
            if 'formula' in df.columns:
                search_formula = st.text_input("Search Formula", "")
        
        # Apply filters
        df_filtered = df.copy()
        
        if 'bandgap' in df.columns:
            df_filtered = df_filtered[
                (df_filtered['bandgap'] >= bg_min) & 
                (df_filtered['bandgap'] <= bg_max)
            ]
        
        if 'source' in df.columns and selected_source != 'All':
            df_filtered = df_filtered[df_filtered['source'] == selected_source]
        
        if 'formula' in df.columns and search_formula:
            df_filtered = df_filtered[
                df_filtered['formula'].str.contains(search_formula, case=False, na=False)
            ]
        
        st.dataframe(df_filtered, use_container_width=True, height=400)
        
        # Download button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="perovskite_database.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 2: UPLOAD YOUR DATA
# =============================================================================

with tab2:
    st.markdown("## 📤 Upload Your Experimental Data")
    st.markdown("**Merge your lab results with the global database**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="File must contain 'formula' and 'bandgap' columns (case-insensitive)"
        )
        
        if uploaded_file is not None:
            try:
                parser = UserDataParser()
                
                # Read file
                file_content = uploaded_file.read()
                df_user = parser.parse(file_content, uploaded_file.name)
                
                if not df_user.empty:
                    st.success(f"✅ Parsed {len(df_user)} materials from {uploaded_file.name}")
                    
                    # Show validation errors/warnings
                    errors = parser.get_validation_errors()
                    if errors:
                        for err in errors:
                            st.warning(err)
                    
                    # Show parsed data
                    st.markdown("### 📋 Parsed Data Preview")
                    st.dataframe(df_user.head(20), use_container_width=True)
                    
                    # Summary
                    summary = parser.get_summary()
                    st.markdown("### 📊 Summary")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Materials", summary.get('n_materials', 0))
                    with col_b:
                        bg_range = summary.get('bandgap_range', (None, None))
                        if bg_range[0] is not None:
                            st.metric("Bandgap Range", f"{bg_range[0]:.2f} - {bg_range[1]:.2f} eV")
                    with col_c:
                        has_device = summary.get('has_device_data', False)
                        st.metric("Device Data", "Yes ✅" if has_device else "No")
                    
                    # Save to session state
                    if st.button("💾 Merge with Database", type="primary"):
                        if st.session_state.db_loaded:
                            st.session_state.user_data = df_user
                            st.session_state.combined_data = parser.merge_with_db(
                                df_user,
                                st.session_state.db_data
                            )
                            st.success("✅ Data merged! Go to Tab 3 to visualize.")
                        else:
                            st.error("Please load database first (Tab 1)")
                
                else:
                    st.error("Failed to parse file. Check format.")
            
            except Exception as e:
                st.error(f"Upload error: {e}")
    
    with col2:
        st.markdown("### 📝 Format Guide")
        
        with st.expander("📄 CSV Template"):
            st.markdown("**Required columns:**")
            st.code("formula, bandgap")
            
            st.markdown("**Optional columns:**")
            st.code("voc, jsc, ff, pce, stability, thickness, method, notes")
            
            st.markdown("**Example:**")
            st.code(example_csv())
            
            st.download_button(
                "📥 Download Template",
                example_csv(),
                "perovskite_template.csv",
                "text/csv"
            )
        
        with st.expander("📊 Excel Format"):
            st.markdown(example_excel_description())
        
        st.markdown("### 🎯 Use Cases")
        st.markdown("""
        - **Lab notebook data**: Quick upload for analysis
        - **Literature mining**: Extracted data from papers
        - **Computational**: Your own DFT calculations
        - **Collaborations**: Share and merge datasets
        """)

# =============================================================================
# TAB 3: PROPERTY SPACE MAP
# =============================================================================

with tab3:
    st.markdown("## 🗺️ Property Space Mapping")
    st.markdown("**Visualize where your data sits relative to the full database**")
    
    if not st.session_state.db_loaded:
        st.info("💡 Load database in Tab 1 first")
    else:
        df = st.session_state.combined_data
        
        if df is None or df.empty:
            st.warning("No data available")
        else:
            st.markdown("### 🎨 Bandgap vs. Composition Space")
            
            # Extract composition features
            featurizer = CompositionFeaturizer()
            
            if 'formula' in df.columns:
                # Featurize all materials
                with st.spinner("Featurizing compositions..."):
                    features = []
                    formulas = []
                    bandgaps = []
                    sources = []
                    
                    for idx, row in df.iterrows():
                        try:
                            feat = featurizer.featurize(row['formula'])
                            features.append(feat)
                            formulas.append(row['formula'])
                            bandgaps.append(row.get('bandgap', np.nan))
                            sources.append(row.get('source', 'unknown'))
                        except:
                            pass
                    
                    features_array = np.array(features)
                
                # PCA for 2D visualization
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features_array)
                
                # Create plot
                plot_df = pd.DataFrame({
                    'PC1': features_2d[:, 0],
                    'PC2': features_2d[:, 1],
                    'formula': formulas,
                    'bandgap': bandgaps,
                    'source': sources
                })
                
                # Highlight user data
                plot_df['is_user'] = plot_df['source'] == 'user_upload'
                
                fig = go.Figure()
                
                # Database points
                db_points = plot_df[~plot_df['is_user']]
                fig.add_trace(go.Scatter(
                    x=db_points['PC1'],
                    y=db_points['PC2'],
                    mode='markers',
                    name='Database',
                    marker=dict(
                        size=8,
                        color=db_points['bandgap'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Bandgap (eV)"),
                        line=dict(width=0.5, color='#333'),
                        opacity=0.6
                    ),
                    text=db_points['formula'],
                    hovertemplate='<b>%{text}</b><br>Eg: %{marker.color:.2f} eV<extra></extra>'
                ))
                
                # User points (if any)
                user_points = plot_df[plot_df['is_user']]
                if not user_points.empty:
                    fig.add_trace(go.Scatter(
                        x=user_points['PC1'],
                        y=user_points['PC2'],
                        mode='markers',
                        name='Your Data',
                        marker=dict(
                            size=15,
                            color=user_points['bandgap'],
                            colorscale='Plasma',
                            symbol='star',
                            line=dict(width=2, color='#000'),
                        ),
                        text=user_points['formula'],
                        hovertemplate='<b>YOUR DATA: %{text}</b><br>Eg: %{marker.color:.2f} eV<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Property Space: Your Data in Context (PCA Projection)",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
                    height=600,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color='#1a1a2e'),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔍 Interpretation")
                st.markdown("""
                - **Database materials** (dots): Known perovskites from global databases
                - **Your materials** (stars): Your experimental data
                - **Color**: Bandgap (blue=narrow, yellow=wide)
                - **Empty regions**: Unexplored composition space — opportunities for discovery!
                - **Clusters**: Similar materials group together in property space
                
                **빈 지도가 탐험의 시작** — The empty map is the start of exploration.
                """)
                
                # Novelty analysis
                if not user_points.empty:
                    st.markdown("### 🆕 Novelty Analysis")
                    
                    # Calculate distance to nearest DB point for each user point
                    from scipy.spatial.distance import cdist
                    
                    db_coords = features_2d[~plot_df['is_user'].values]
                    user_coords = features_2d[plot_df['is_user'].values]
                    
                    distances = cdist(user_coords, db_coords, metric='euclidean')
                    min_distances = distances.min(axis=1)
                    
                    novelty_df = pd.DataFrame({
                        'formula': user_points['formula'].values,
                        'distance_to_nearest': min_distances
                    }).sort_values('distance_to_nearest', ascending=False)
                    
                    st.markdown("**Most novel materials (farthest from database):**")
                    st.dataframe(novelty_df.head(10), use_container_width=True)
                    
                    st.info("💡 High distance = unexplored region. Potential for new discoveries!")

# =============================================================================
# TAB 4: ML SURROGATE
# =============================================================================

with tab4:
    st.markdown("## 🤖 Lightweight ML Surrogate Model")
    st.markdown("**XGBoost bandgap predictor trained on database — fast predictions with confidence**")
    
    if not st.session_state.db_loaded:
        st.info("💡 Load database in Tab 1 first")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏋️ Train Model")
            
            st.markdown("""
            **Composition → Bandgap predictor**
            
            - **Input**: Chemical formula (e.g., FA0.87Cs0.13PbI3)
            - **Output**: Predicted bandgap ± uncertainty
            - **Model**: XGBoost (lightweight, no PyTorch)
            - **Features**: 18D composition featurization (tolerance factor, mixing entropy, etc.)
            """)
            
            if st.button("🚀 Train ML Model", type="primary"):
                with st.spinner("Training XGBoost model..."):
                    try:
                        df_train = st.session_state.combined_data
                        
                        # Filter valid data
                        df_train = df_train[
                            df_train['bandgap'].notna() & 
                            (df_train['bandgap'] > 0) &
                            (df_train['bandgap'] < 10)
                        ]
                        
                        # Train model
                        model = BandgapPredictor(use_xgboost=True)
                        metrics = model.train(df_train, formula_col='formula', target_col='bandgap')
                        
                        st.session_state.ml_model = model
                        st.session_state.model_trained = True
                        st.session_state.train_metrics = metrics
                        
                        st.success("✅ Model trained!")
                        
                        # Show metrics
                        st.markdown("### 📊 Training Metrics")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Training Samples", metrics['n_samples'])
                        with col_b:
                            st.metric("CV MAE", f"{metrics['cv_mae']:.3f} eV")
                        with col_c:
                            st.metric("R² Score", f"{metrics['train_r2']:.3f}")
                        
                        # Feature importance
                        st.markdown("### 🔍 Feature Importance")
                        
                        importance_df = model.get_feature_importance()
                        
                        fig = px.bar(
                            importance_df.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        
                        fig.update_layout(
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font=dict(color='#1a1a2e'),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        
        with col2:
            st.markdown("### ℹ️ Model Info")
            
            if st.session_state.model_trained:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                metrics = st.session_state.train_metrics
                st.markdown(f"""
                **Model Ready!**
                
                - Samples: {metrics['n_samples']}
                - MAE: {metrics['cv_mae']:.3f} ± {metrics['cv_mae_std']:.3f} eV
                - R²: {metrics['train_r2']:.3f}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Train model to enable predictions")
            
            if show_limitations:
                st.markdown('<div class="limitation-box">', unsafe_allow_html=True)
                st.markdown("""
                **Limitations:**
                
                - Simple composition features only
                - No structural info (lattice, symmetry)
                - Trained on available data (biased)
                - Uncertainty = ensemble variance (rough estimate)
                - Use for screening, not final design
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction interface
        if st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### 🔮 Make Predictions")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                formula_input = st.text_input(
                    "Enter formula",
                    "FA0.87Cs0.13Pb(I0.62Br0.38)3",
                    help="Examples: MAPbI3, FA0.85Cs0.15PbI3, CsPb(I0.7Br0.3)3"
                )
                
                if st.button("🎯 Predict Bandgap"):
                    try:
                        model = st.session_state.ml_model
                        
                        predictions, uncertainties = model.predict([formula_input], return_uncertainty=True)
                        
                        pred_eg = predictions[0]
                        uncertainty = uncertainties[0] if uncertainties is not None else 0.0
                        
                        st.markdown("### 📊 Prediction Result")
                        
                        st.markdown(f"""
                        **Formula:** {formula_input}
                        
                        **Predicted Bandgap:** {pred_eg:.3f} ± {uncertainty:.3f} eV
                        """)
                        
                        # Confidence assessment
                        if uncertainty < 0.1:
                            st.success("✅ High confidence (low uncertainty)")
                        elif uncertainty < 0.3:
                            st.warning("⚠️ Medium confidence")
                        else:
                            st.error("❌ Low confidence (high uncertainty) — use with caution!")
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            
            with col2:
                st.markdown("### 📈 Batch Prediction")
                
                batch_formulas = st.text_area(
                    "Enter formulas (one per line)",
                    "MAPbI3\nFAPbI3\nCsPbBr3",
                    height=150
                )
                
                if st.button("🎯 Predict Batch"):
                    try:
                        formulas = [f.strip() for f in batch_formulas.split('\n') if f.strip()]
                        
                        model = st.session_state.ml_model
                        predictions, uncertainties = model.predict(formulas, return_uncertainty=True)
                        
                        results_df = pd.DataFrame({
                            'formula': formulas,
                            'predicted_bandgap': predictions,
                            'uncertainty': uncertainties if uncertainties is not None else [0.0] * len(formulas)
                        })
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Predictions",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

# =============================================================================
# TAB 5: WHY AI? (V3 DEMO PRESERVED)
# =============================================================================

with tab5:
    st.markdown("## 🔬 Why AI? — The V3 Demo (Preserved)")
    st.markdown("**Original 12D design space demonstration from V3**")
    
    st.info("""
    💡 **This tab preserves the V3 "Why AI?" moment**
    
    V3 demonstrated the impossibility of manual 12D optimization.
    V4 extends this with real data and ML, but the core insight remains:
    
    **Human intuition fails in high-dimensional spaces. AI navigates them.**
    """)
    
    # Link back to V3 for full experience
    st.markdown("""
    ### 🎭 The Original Demo
    
    V3 showed:
    1. **Manual tuning fails**: Random sliders → unbalanced radar chart
    2. **AI succeeds**: Bayesian optimization → balanced solution in seconds
    3. **Hidden constraints**: Chemistry vs. Physics trade-offs revealed
    
    **Key insight:** The "Why AI?" moment comes from experiencing failure first,
    then seeing AI solve it instantly.
    
    ---
    
    **V4 Evolution:**
    - V3: Hardcoded 16 compositions → demo
    - V4: Thousands from databases → real tool
    
    The philosophy remains: **honest limitations, real data, AI acceleration**.
    """)
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("""
    **Access full V3 demo:**
    
    Run `streamlit run app_v3_sait.py` to experience the original 6-tab interactive demo.
    
    V4 focuses on **data connectivity** rather than duplicating V3's UI brilliance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.9rem;">
    <p><b>AlphaMaterials V4.0 — Connected Platform</b> | SAIT × SPMDL | 2026-03-15</p>
    <p>🔬 Real Databases + User Data + ML Surrogate = Discovery Acceleration</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">빈 지도가 탐험의 시작 — The empty map is the start of exploration</p>
</div>
""", unsafe_allow_html=True)
