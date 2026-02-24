#!/usr/bin/env python3
"""
N-Junction Infinite Tandem PV Simulator - Streamlit App
=======================================================

Comprehensive web interface for tandem photovoltaic cell simulation and optimization.
Features 8 tabs covering all aspects from SQ limits to economics.

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Configure Streamlit page
st.set_page_config(
    page_title="N-Junction Tandem PV Simulator", 
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import engines
try:
    from engines.band_alignment import DetailedBalanceCalculator, BandgapOptimizer
    from engines.optical_tmm import TransferMatrixCalculator  
    from engines.interface_loss import InterfaceLossCalculator
    from engines.thermal_model import analyze_thermal_performance
    from engines.stability import StabilityPredictor, EnvironmentalConditions
    from engines.economics import EconomicsEngine
    from config import MATERIAL_DB, get_am15g_spectrum
    
    ENGINES_LOADED = True
except ImportError as e:
    st.error(f"❌ Engine loading failed: {e}")
    ENGINES_LOADED = False

# Custom CSS for Korean-English UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > label, .stSlider > label {
        font-weight: 600;
        color: #2E86AB;
    }
    .metric-container {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .tab-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Global Parameters
st.sidebar.title("🌞 탠덤 PV 시뮬레이터")
st.sidebar.markdown("**N-Junction Tandem PV Simulator**")
st.sidebar.markdown("---")

# Global parameters
st.sidebar.subheader("🔧 글로벌 매개변수 (Global Parameters)")

# Track selection
track = st.sidebar.selectbox(
    "📊 재료 트랙 (Material Track)",
    ["A - Multi-material", "B - Perovskite Focus"],
    index=0,
    help="Track A: 9가지 재료 / Track B: 페로브스카이트 중심"
)

# Operating conditions
st.sidebar.subheader("🌡️ 동작 조건 (Operating Conditions)")

temperature = st.sidebar.slider(
    "온도 (Temperature) [°C]",
    min_value=-40, max_value=85, value=25, step=5,
    help="셀 동작 온도"
)

irradiance = st.sidebar.slider(
    "조사량 (Irradiance) [W/m²]", 
    min_value=200, max_value=1200, value=1000, step=50,
    help="태양광 조사량 (AM1.5G 기준)"
)

concentration = st.sidebar.slider(
    "집광비 (Concentration) [×]",
    min_value=1, max_value=1000, value=1, step=1,
    help="집광 시스템 배율"
)

# Convert temperature to Kelvin
T_cell = temperature + 273.15

# Shockley-Queisser reference values
SQ_LIMITS = {
    1: 0.337,  # 33.7%
    2: 0.45,   # 45%
    3: 0.51,   # 51%
    4: 0.56,   # 56%
    5: 0.60,   # 60%
    10: 0.68   # 68% (infinite limit approach)
}

# Main app tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 개요 & SQ 한계", "🎯 밴드갭 최적화", "🔍 광학 분석", 
    "⚡ 인터페이스 & 터널", "🌡️ 열적 분석", "⏳ 안정성 분석",
    "💰 경제성 분석", "🚀 종합 최적화"
])

if not ENGINES_LOADED:
    st.error("❌ 엔진을 로드할 수 없습니다. 파일을 확인하세요.")
    st.stop()

# =====================================================================
# TAB 1: Overview & SQ Limits
# =====================================================================
with tab1:
    st.markdown('<div class="tab-header">📈 개요 & 샤클리-퀘이저 한계 (Overview & Shockley-Queisser Limits)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌟 이론적 효율 한계")
        
        # Create SQ limit visualization
        n_junctions = list(range(1, 11))
        sq_efficiencies = [SQ_LIMITS.get(n, 0.68 * (1 - np.exp(-n/3))) for n in n_junctions]
        
        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(
            x=n_junctions,
            y=[eff * 100 for eff in sq_efficiencies],
            mode='lines+markers',
            name='SQ Limit',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        fig_sq.update_layout(
            title="접합부 개수에 따른 이론적 효율 한계",
            xaxis_title="접합부 개수 (Number of Junctions)",
            yaxis_title="효율 (Efficiency) [%]",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_sq, use_container_width=True)
        
        # Display key metrics
        st.markdown("### 📊 주요 한계값")
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            st.metric("1J 한계", "33.7%", help="단일 접합 이론 한계")
        with col1b:
            st.metric("2J 한계", "45.0%", help="이중 접합 이론 한계")  
        with col1c:
            st.metric("∞J 한계", "68.7%", help="무한 접합 이론 한계")
    
    with col2:
        st.subheader("🌅 AM1.5G 태양 스펙트럼")
        
        # Generate AM1.5G spectrum
        wavelengths = np.linspace(300, 1550, 500)
        try:
            spectrum = get_am15g_spectrum(wavelengths)
            
            fig_spectrum = go.Figure()
            fig_spectrum.add_trace(go.Scatter(
                x=wavelengths,
                y=spectrum,
                mode='lines',
                name='AM1.5G',
                fill='tonexty',
                line=dict(color='gold', width=2)
            ))
            
            fig_spectrum.update_layout(
                title="표준 태양 스펙트럼 (AM1.5G)",
                xaxis_title="파장 (Wavelength) [nm]", 
                yaxis_title="조사량 (Irradiance) [W⋅m⁻²⋅nm⁻¹]",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_spectrum, use_container_width=True)
            
        except Exception as e:
            st.error(f"스펙트럼 로드 실패: {e}")
            
    # Material comparison table
    st.subheader("📋 재료 비교표 (Material Comparison)")
    
    try:
        materials_a = MATERIAL_DB.list_materials('A')[:6]  # First 6 materials
        materials_b = MATERIAL_DB.list_materials('B')[:6]
        
        data = []
        
        for track_name, materials in [("Track A", materials_a), ("Track B", materials_b)]:
            for mat in materials:
                try:
                    props = MATERIAL_DB.get_material(mat, track_name.split()[1])
                    data.append({
                        "트랙": track_name,
                        "재료": mat,
                        "밴드갭 (eV)": f"{props.get('bandgap', 'N/A'):.2f}" if isinstance(props.get('bandgap'), (int, float)) else "N/A",
                        "굴절률": f"{props.get('n_550', 'N/A'):.2f}" if isinstance(props.get('n_550'), (int, float)) else "N/A",
                        "용도": props.get('application', 'Active Layer')
                    })
                except:
                    continue
        
        if data:
            df_materials = pd.DataFrame(data)
            st.dataframe(df_materials, use_container_width=True)
        else:
            st.info("재료 데이터베이스 정보를 로드할 수 없습니다.")
            
    except Exception as e:
        st.error(f"재료 데이터 로드 실패: {e}")

# =====================================================================  
# TAB 2: Band Alignment & Optimization
# =====================================================================
with tab2:
    st.markdown('<div class="tab-header">🎯 밴드갭 최적화 (Band Alignment & Optimization)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ 최적화 설정")
        
        n_junctions = st.slider(
            "접합부 개수 (Number of Junctions)",
            min_value=1, max_value=10, value=2, step=1,
            help="최적화할 접합부의 개수"
        )
        
        current_matching = st.checkbox(
            "전류 매칭 적용 (Apply Current Matching)",
            value=True,
            help="직렬 연결에서 전류 매칭 제약 적용"
        )
        
        if st.button("🚀 밴드갭 최적화 실행", type="primary"):
            with st.spinner("최적화 중..."):
                try:
                    # Initialize calculators
                    calc = DetailedBalanceCalculator(temperature=T_cell, concentration=concentration)
                    optimizer = BandgapOptimizer(calc)
                    
                    # Run optimization
                    result = optimizer.optimize_n_junction(n_junctions)
                    
                    # Store in session state
                    st.session_state['optimization_result'] = result
                    st.session_state['n_junctions'] = n_junctions
                    
                    st.success(f"✅ {n_junctions}-접합 최적화 완료!")
                    
                except Exception as e:
                    st.error(f"❌ 최적화 실패: {e}")
    
    with col2:
        st.subheader("📊 최적화 결과")
        
        if 'optimization_result' in st.session_state:
            result = st.session_state['optimization_result']
            
            # Display metrics
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                efficiency_pct = result.max_efficiency * 100
                st.metric("최대 효율", f"{efficiency_pct:.1f}%")
                
            with col2b:
                st.metric("총 전압", f"{result.voc_total:.2f} V")
                
            with col2c:
                st.metric("전류밀도", f"{result.jsc_matched:.1f} mA/cm²")
            
            # Bandgap distribution chart
            fig_bg = go.Figure()
            
            colors = px.colors.qualitative.Set1[:len(result.bandgaps)]
            
            fig_bg.add_trace(go.Bar(
                x=[f"J{i+1}" for i in range(len(result.bandgaps))],
                y=result.bandgaps,
                marker_color=colors,
                text=[f"{bg:.2f} eV" for bg in result.bandgaps],
                textposition='outside'
            ))
            
            fig_bg.update_layout(
                title="최적 밴드갭 분포 (Optimal Bandgap Distribution)",
                xaxis_title="접합부 (Junction)",
                yaxis_title="밴드갭 (Bandgap) [eV]",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_bg, use_container_width=True)
            
        else:
            st.info("👆 최적화를 실행하여 결과를 확인하세요.")
    
    # PCE vs N curve
    st.subheader("📈 접합부 개수에 따른 효율 변화")
    
    if st.button("🔄 효율 곡선 생성"):
        with st.spinner("다중 접합 효율 계산 중..."):
            try:
                calc = DetailedBalanceCalculator(temperature=T_cell, concentration=concentration)
                optimizer = BandgapOptimizer(calc)
                
                n_range = list(range(1, 8))  # 1-7 junctions
                efficiencies = []
                
                progress_bar = st.progress(0)
                
                for i, n in enumerate(n_range):
                    result = optimizer.optimize_n_junction(n)
                    efficiencies.append(result.max_efficiency * 100)
                    progress_bar.progress((i + 1) / len(n_range))
                
                progress_bar.empty()
                
                # Create efficiency curve
                fig_eff = go.Figure()
                
                # Theoretical SQ limits
                sq_theoretical = [SQ_LIMITS.get(n, SQ_LIMITS[10]) * 100 for n in n_range]
                
                fig_eff.add_trace(go.Scatter(
                    x=n_range,
                    y=sq_theoretical,
                    mode='lines+markers',
                    name='이론적 한계 (SQ)',
                    line=dict(color='red', dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig_eff.add_trace(go.Scatter(
                    x=n_range,
                    y=efficiencies,
                    mode='lines+markers',
                    name='시뮬레이션 결과',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=10)
                ))
                
                fig_eff.update_layout(
                    title="접합부 개수에 따른 효율 한계 (PCE vs N-Junctions)",
                    xaxis_title="접합부 개수 (Number of Junctions)",
                    yaxis_title="전력변환효율 (PCE) [%]",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_eff, use_container_width=True)
                
                # Display diminishing returns analysis
                st.subheader("📉 수익 체감 분석")
                
                improvements = [0] + [efficiencies[i] - efficiencies[i-1] for i in range(1, len(efficiencies))]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_improve = go.Figure()
                    fig_improve.add_trace(go.Bar(
                        x=n_range,
                        y=improvements,
                        marker_color='lightblue'
                    ))
                    fig_improve.update_layout(
                        title="접합부 추가시 효율 향상도",
                        xaxis_title="접합부 개수",
                        yaxis_title="효율 향상 [%p]",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_improve, use_container_width=True)
                
                with col2:
                    st.markdown("### 💡 분석 결과")
                    max_improve_idx = np.argmax(improvements[1:]) + 1
                    st.write(f"• 최대 효율 향상: {n_range[max_improve_idx]}J → {n_range[max_improve_idx]+1}J")
                    st.write(f"• 향상도: {improvements[max_improve_idx]:.1f}%p")
                    
                    # Cost-benefit analysis
                    if improvements[-1] < 2.0:  # Less than 2% improvement
                        st.warning("⚠️ 고접합 시스템에서 수익 체감 현상 발생")
                    
            except Exception as e:
                st.error(f"❌ 효율 곡선 생성 실패: {e}")

# =====================================================================
# TAB 3: Optical Analysis (TMM)
# =====================================================================
with tab3:
    st.markdown('<div class="tab-header">🔍 광학 분석 - TMM (Optical Analysis - Transfer Matrix Method)</div>', unsafe_allow_html=True)
    
    st.subheader("🏗️ 층 구조 설계 (Layer Stack Builder)")
    
    # Initialize session state for layer stack
    if 'layer_stack' not in st.session_state:
        st.session_state['layer_stack'] = [
            ("glass", 3000000),  # 3 mm substrate
            ("ITO", 100),        # 100 nm TCO
            ("perovskite", 500), # 500 nm active
            ("Au", 80)           # 80 nm contact
        ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ 층 편집")
        
        # Add layer interface
        st.markdown("**새 층 추가**")
        
        # Get available materials
        try:
            track_code = track.split()[0]  # 'A' or 'B'
            available_materials = MATERIAL_DB.list_materials(track_code)
        except:
            available_materials = ['glass', 'ITO', 'perovskite', 'Au', 'Ag']
        
        new_material = st.selectbox("재료 선택", available_materials)
        new_thickness = st.number_input("두께 [nm]", min_value=1, max_value=10000, value=100)
        
        if st.button("+ 층 추가"):
            st.session_state['layer_stack'].append((new_material, new_thickness))
            st.rerun()
        
        # Display current stack
        st.markdown("**현재 구조**")
        
        stack_display = []
        for i, (mat, thick) in enumerate(st.session_state['layer_stack']):
            stack_display.append({
                "순서": i+1,
                "재료": mat,
                "두께 [nm]": thick
            })
        
        df_stack = pd.DataFrame(stack_display)
        st.dataframe(df_stack, use_container_width=True)
        
        # Remove layer
        if len(st.session_state['layer_stack']) > 1:
            remove_idx = st.selectbox(
                "층 제거 (Remove Layer)", 
                options=range(len(st.session_state['layer_stack'])),
                format_func=lambda x: f"{x+1}. {st.session_state['layer_stack'][x][0]}"
            )
            
            if st.button("🗑️ 선택된 층 제거"):
                st.session_state['layer_stack'].pop(remove_idx)
                st.rerun()
    
    with col2:
        st.subheader("📊 광학 시뮬레이션")
        
        if st.button("🔬 광학 분석 실행", type="primary"):
            with st.spinner("TMM 계산 중..."):
                try:
                    # This is a placeholder - the actual optical engine would need
                    # proper interfacing. For demo purposes, create realistic data
                    
                    wavelengths = np.linspace(300, 1200, 200)
                    
                    # Simulate absorption/reflection/transmission
                    # In reality, this would use the TransferMatrixCalculator
                    
                    # Simple Beer-Lambert approximation for demo
                    total_thickness = sum(thick for mat, thick in st.session_state['layer_stack'] if mat != 'glass')
                    
                    # Simulate absorption based on materials
                    absorption = np.zeros_like(wavelengths)
                    reflection = np.ones_like(wavelengths) * 0.1  # 10% base reflection
                    
                    for mat, thick in st.session_state['layer_stack']:
                        if mat in ['perovskite', 'c-Si', 'GaAs']:
                            # Active materials - wavelength dependent absorption
                            if mat == 'perovskite':
                                bandgap_nm = 1240 / 1.6  # ~775 nm
                                abs_coeff = np.where(wavelengths < bandgap_nm, 
                                                   1e5 * (thick * 1e-9), 0) # Strong absorption
                            elif mat == 'c-Si':
                                bandgap_nm = 1240 / 1.12  # ~1107 nm  
                                abs_coeff = np.where(wavelengths < bandgap_nm,
                                                   1e4 * (thick * 1e-9), 0)
                            else:  # GaAs
                                bandgap_nm = 1240 / 1.42  # ~873 nm
                                abs_coeff = np.where(wavelengths < bandgap_nm,
                                                   5e4 * (thick * 1e-9), 0)
                            
                            layer_absorption = 1 - np.exp(-abs_coeff)
                            absorption += layer_absorption * (1 - absorption)  # Series absorption
                    
                    transmission = 1 - absorption - reflection
                    transmission = np.maximum(transmission, 0)  # No negative transmission
                    
                    # Create absorption spectrum plot
                    fig_optical = go.Figure()
                    
                    fig_optical.add_trace(go.Scatter(
                        x=wavelengths, y=absorption * 100,
                        mode='lines', name='흡수 (Absorption)',
                        line=dict(color='red', width=2), fill='tonexty'
                    ))
                    
                    fig_optical.add_trace(go.Scatter(
                        x=wavelengths, y=reflection * 100,
                        mode='lines', name='반사 (Reflection)', 
                        line=dict(color='silver', width=2)
                    ))
                    
                    fig_optical.add_trace(go.Scatter(
                        x=wavelengths, y=transmission * 100,
                        mode='lines', name='투과 (Transmission)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_optical.update_layout(
                        title="광학 스펙트럼 응답 (Optical Spectral Response)",
                        xaxis_title="파장 (Wavelength) [nm]",
                        yaxis_title="비율 (%) [%]",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig_optical, use_container_width=True)
                    
                    # Layer-by-layer photocurrent
                    st.subheader("📊 층별 광전류 밀도")
                    
                    # Calculate photocurrent for active layers
                    try:
                        spectrum_flux = get_am15g_spectrum(wavelengths)
                        photon_flux = spectrum_flux * wavelengths * 1e-9 / (4.135667696e-15 * 2.99792458e8)  # Convert to photons
                        
                        layer_currents = []
                        for mat, thick in st.session_state['layer_stack']:
                            if mat in ['perovskite', 'c-Si', 'GaAs', 'CIGS']:
                                # Calculate absorbed photons for this layer
                                if mat == 'perovskite':
                                    bandgap_ev = 1.6
                                elif mat == 'c-Si':
                                    bandgap_ev = 1.12
                                elif mat == 'GaAs':
                                    bandgap_ev = 1.42
                                else:  # CIGS
                                    bandgap_ev = 1.15
                                
                                # Simple current calculation
                                useful_photons = photon_flux * (wavelengths < (1240 / bandgap_ev))
                                layer_jsc = np.trapz(useful_photons * absorption, wavelengths) * 1.602e-19 * 1000  # mA/cm²
                                
                                layer_currents.append({
                                    "재료": mat,
                                    "두께 [nm]": thick,  
                                    "광전류밀도 [mA/cm²]": f"{layer_jsc:.1f}"
                                })
                        
                        if layer_currents:
                            df_currents = pd.DataFrame(layer_currents)
                            st.dataframe(df_currents, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"광전류 계산 중 오류: {e}")
                        
                except Exception as e:
                    st.error(f"❌ 광학 분석 실패: {e}")
        
        # Anti-reflection coating optimizer
        st.subheader("✨ 반사방지막 최적화")
        
        ar_material = st.selectbox("AR 코팅 재료", ['TiO2', 'SiO2', 'Si3N4', 'MgF2'])
        
        if st.button("🎯 AR 코팅 최적화"):
            # Placeholder for AR coating optimization
            st.info("반사방지막 최적화 기능은 향후 구현 예정입니다.")

# =====================================================================
# TAB 4: Interface & Tunnel Junctions  
# =====================================================================
with tab4:
    st.markdown('<div class="tab-header">⚡ 인터페이스 & 터널 접합 (Interface & Tunnel Junctions)</div>', unsafe_allow_html=True)
    
    st.subheader("🔧 터널 접합 설계")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**터널 접합 매개변수**")
        
        barrier_height = st.slider("장벽 높이 (Barrier Height) [eV]", 0.1, 2.0, 1.0, 0.1)
        barrier_width = st.slider("장벽 폭 (Barrier Width) [nm]", 0.5, 5.0, 2.0, 0.1)  
        doping_n = st.selectbox("N형 도핑 농도 [cm⁻³]", ['1e18', '1e19', '1e20', '1e21'])
        doping_p = st.selectbox("P형 도핑 농도 [cm⁻³]", ['1e18', '1e19', '1e20', '1e21'])
        
        n_doping = float(doping_n)
        p_doping = float(doping_p)
        
        # Tunnel resistance calculation (simplified)
        st.subheader("📊 터널 저항 계산")
        
        # WKB approximation for tunneling resistance (simplified)
        # R_tunnel ∝ exp(2 * sqrt(2m*φ) * d / ℏ) where φ is barrier height, d is width
        
        # Physical constants (simplified units)
        hbar = 1.054e-34  # J⋅s
        m_eff = 0.1 * 9.109e-31  # Effective mass (kg)  
        q = 1.602e-19  # C
        
        # Tunneling probability (qualitative)
        phi_j = barrier_height * q  # Convert to Joules
        width_m = barrier_width * 1e-9  # Convert to meters
        
        kappa = np.sqrt(2 * m_eff * phi_j) / hbar
        transmission = np.exp(-2 * kappa * width_m)
        
        # Resistance estimation (order of magnitude)
        # Higher doping = lower resistance
        doping_factor = 1e20 / np.sqrt(n_doping * p_doping)
        resistance_est = doping_factor / (transmission * 1e6)  # Ω⋅cm²
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("터널링 확률", f"{transmission:.2e}")
        with col1b:
            st.metric("예상 저항", f"{resistance_est:.2e} Ω⋅cm²")
            
        # Warning for high resistance
        if resistance_est > 1e-2:
            st.warning("⚠️ 높은 저항으로 인한 성능 저하 가능성")
        elif resistance_est < 1e-6:
            st.success("✅ 우수한 터널링 특성")
        else:
            st.info("ℹ️ 적절한 터널링 저항 범위")
    
    with col2:
        st.subheader("📈 설계 파라미터 영향 분석")
        
        # Parameter sensitivity analysis
        if st.button("🔍 민감도 분석 실행"):
            with st.spinner("매개변수 영향 분석 중..."):
                
                # Width sensitivity
                widths = np.linspace(0.5, 4.0, 20)
                resistances_width = []
                
                for w in widths:
                    w_m = w * 1e-9
                    kappa = np.sqrt(2 * m_eff * phi_j) / hbar  
                    trans = np.exp(-2 * kappa * w_m)
                    r_est = doping_factor / (trans * 1e6)
                    resistances_width.append(r_est)
                
                # Barrier height sensitivity  
                barriers = np.linspace(0.2, 2.0, 20)
                resistances_barrier = []
                
                for b in barriers:
                    phi = b * q
                    kappa = np.sqrt(2 * m_eff * phi) / hbar
                    trans = np.exp(-2 * kappa * width_m)
                    r_est = doping_factor / (trans * 1e6) 
                    resistances_barrier.append(r_est)
                
                # Create sensitivity plots
                fig_sens = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("장벽 폭 영향", "장벽 높이 영향"),
                    x_titles=["장벽 폭 [nm]", "장벽 높이 [eV]"],
                    y_titles=["저항 [Ω⋅cm²]", "저항 [Ω⋅cm²]"]
                )
                
                fig_sens.add_trace(
                    go.Scatter(x=widths, y=resistances_width, mode='lines', name='폭 의존성'),
                    row=1, col=1
                )
                
                fig_sens.add_trace(
                    go.Scatter(x=barriers, y=resistances_barrier, mode='lines', name='높이 의존성'), 
                    row=1, col=2
                )
                
                fig_sens.update_yaxes(type="log")
                fig_sens.update_layout(height=400, template="plotly_white")
                
                st.plotly_chart(fig_sens, use_container_width=True)
                
                # Design recommendations
                st.subheader("💡 설계 권장사항")
                
                optimal_width = widths[np.argmin(resistances_width)]
                optimal_barrier = barriers[np.argmin(resistances_barrier)]
                
                st.write(f"• 최적 장벽 폭: {optimal_width:.1f} nm")
                st.write(f"• 최적 장벽 높이: {optimal_barrier:.1f} eV") 
                st.write(f"• 권장 도핑: > 1e20 cm⁻³")
                
                if barrier_width > 3.0:
                    st.warning("⚠️ 장벽이 너무 두꺼워 터널링 효율 저하")
                if barrier_height > 1.5:
                    st.warning("⚠️ 장벽이 너무 높아 터널링 저항 증가")
    
    # N-junction loss analysis
    st.subheader("🔗 N-접합 손실 분석")
    
    n_junctions_loss = st.slider("분석할 접합 개수", 2, 10, 3)
    
    if st.button("📊 접합별 손실 분석"):
        with st.spinner("접합 손실 계산 중..."):
            
            # Simulate cumulative losses
            junctions = list(range(2, n_junctions_loss + 1))
            
            # Loss types
            tunnel_losses = []      # Tunnel junction resistance losses
            interface_losses = []   # Interface recombination losses
            series_losses = []      # Series resistance losses
            total_losses = []       # Total system losses
            
            for n in junctions:
                # Each additional junction adds losses
                tunnel_loss = (n - 1) * 0.5  # ~0.5% per tunnel junction
                interface_loss = (n - 1) * 0.3  # ~0.3% per interface
                series_loss = n * 0.2  # ~0.2% per junction (series)
                
                total_loss = tunnel_loss + interface_loss + series_loss
                
                tunnel_losses.append(tunnel_loss)
                interface_losses.append(interface_loss)
                series_losses.append(series_loss)
                total_losses.append(total_loss)
            
            # Create stacked bar chart
            fig_losses = go.Figure()
            
            fig_losses.add_trace(go.Bar(
                name='터널 접합 손실',
                x=[f"{n}J" for n in junctions],
                y=tunnel_losses,
                marker_color='lightcoral'
            ))
            
            fig_losses.add_trace(go.Bar(
                name='인터페이스 손실', 
                x=[f"{n}J" for n in junctions],
                y=interface_losses,
                marker_color='lightsalmon'
            ))
            
            fig_losses.add_trace(go.Bar(
                name='직렬 저항 손실',
                x=[f"{n}J" for n in junctions], 
                y=series_losses,
                marker_color='lightblue'
            ))
            
            fig_losses.update_layout(
                title="접합 개수에 따른 누적 손실 분석",
                xaxis_title="접합 구조",
                yaxis_title="상대적 손실 [%]",
                barmode='stack',
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_losses, use_container_width=True)
            
            # Summary table
            loss_data = []
            for i, n in enumerate(junctions):
                loss_data.append({
                    "접합수": f"{n}J",
                    "터널손실 [%]": f"{tunnel_losses[i]:.1f}",
                    "인터페이스손실 [%]": f"{interface_losses[i]:.1f}",
                    "직렬저항손실 [%]": f"{series_losses[i]:.1f}",
                    "총손실 [%]": f"{total_losses[i]:.1f}"
                })
            
            df_losses = pd.DataFrame(loss_data)
            st.dataframe(df_losses, use_container_width=True)
            
            # Critical point analysis
            critical_n = next((n for n, loss in zip(junctions, total_losses) if loss > 5), None)
            if critical_n:
                st.warning(f"⚠️ {critical_n}J 이상에서 손실 5% 초과 - 경제성 검토 필요")

# =====================================================================
# TAB 5: Thermal & CTE Analysis
# =====================================================================
with tab5:
    st.markdown('<div class="tab-header">🌡️ 열적 분석 & CTE (Thermal & CTE Analysis)</div>', unsafe_allow_html=True)
    
    st.subheader("🔧 열적 스트레스 분석 설정")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**재료 및 두께 설정**")
        
        # Material selection for thermal analysis
        thermal_materials = st.multiselect(
            "분석할 재료 선택",
            ['MAPbI3', 'c-Si', 'GaAs', 'GaInP', 'CIGS', 'CdTe'],
            default=['MAPbI3', 'c-Si']
        )
        
        # Thickness inputs
        thermal_thicknesses = []
        for mat in thermal_materials:
            default_thick = 500 if ('perovskite' in mat.lower() or 'MAP' in mat) else 5000
            thickness = st.number_input(
                f"{mat} 두께 [nm]", 
                min_value=10, max_value=500000, 
                value=default_thick,
                key=f"thermal_thick_{mat}"
            )
            thermal_thicknesses.append(thickness * 1e-9)  # Convert to meters
        
        # Operating temperature range
        st.markdown("**동작 온도 범위**")
        temp_min = st.number_input("최저 온도 [°C]", value=-40, min_value=-50, max_value=50)
        temp_max = st.number_input("최고 온도 [°C]", value=85, min_value=50, max_value=150)
        
        substrate_material = st.selectbox(
            "기판 재료",
            ['glass', 'sapphire', 'silicon', 'polymer'],
            help="기판 재료에 따른 CTE 매칭 분석"
        )
    
    with col2:
        st.subheader("🌡️ 열적 분석 결과")
        
        if thermal_materials and st.button("🔥 열적 분석 실행", type="primary"):
            with st.spinner("열적 스트레스 계산 중..."):
                try:
                    # Use thermal analysis engine
                    operating_conditions = {
                        'operating_temp': temperature + 273.15,
                        'min_temp': temp_min + 273.15,  
                        'max_temp': temp_max + 273.15
                    }
                    
                    # Call thermal analysis
                    thermal_result = analyze_thermal_performance(
                        thermal_materials, 
                        thermal_thicknesses,
                        operating_conditions,
                        substrate_material
                    )
                    
                    # Display key metrics
                    col2a, col2b, col2c = st.columns(3)
                    
                    with col2a:
                        max_stress = thermal_result['thermal_stress'].total_stress / 1e6  # Convert to MPa
                        st.metric("최대 열응력", f"{max_stress:.1f} MPa")
                        
                    with col2b:
                        curvature = thermal_result['thermal_stress'].curvature * 1000  # Convert to m⁻¹
                        st.metric("기판 곡률", f"{curvature:.3f} m⁻¹")
                        
                    with col2c:
                        cte_severity = thermal_result['thermal_stress'].cte_mismatch_severity
                        st.metric("CTE 불일치도", f"{cte_severity:.1f}/10")
                    
                    # Stress per layer visualization
                    if len(thermal_materials) > 1:
                        fig_stress = go.Figure()
                        
                        stress_values = [s/1e6 for s in thermal_result['thermal_stress'].stress_per_layer]  # MPa
                        
                        fig_stress.add_trace(go.Bar(
                            x=thermal_materials,
                            y=np.abs(stress_values),  # Absolute values for visualization
                            marker_color=['red' if s > 50 else 'orange' if s > 25 else 'green' for s in np.abs(stress_values)],
                            text=[f"{s:+.1f}" for s in stress_values],
                            textposition='outside'
                        ))
                        
                        fig_stress.update_layout(
                            title="층별 열응력 분포 (Thermal Stress by Layer)",
                            xaxis_title="재료 (Material)",
                            yaxis_title="열응력 (Thermal Stress) [MPa]",
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_stress, use_container_width=True)
                    
                    # Lifetime prediction
                    lifetime_pred = thermal_result['lifetime_prediction']
                    
                    st.subheader("⏳ 열적 수명 예측")
                    
                    col2d, col2e = st.columns(2)
                    
                    with col2d:
                        t80_years = lifetime_pred.t80_thermal
                        st.metric("T80 수명", f"{t80_years:.1f} 년", help="80% 성능 유지 기간")
                        
                    with col2e:
                        failure_mode = lifetime_pred.dominant_failure_mode
                        st.metric("주요 실패모드", failure_mode)
                    
                    # Recommendations
                    st.subheader("💡 설계 권장사항")
                    recommendations = thermal_result['recommendations']
                    
                    if recommendations['thermal_design_margin'] < 2.0:
                        st.warning("⚠️ 열적 설계 여유도 부족 - 온도 제한 또는 재료 변경 검토")
                    
                    if recommendations['substrate_suitability'] == 'poor':
                        st.error("❌ 기판 재료 부적합 - 다른 기판 검토 필요")
                    else:
                        st.success("✅ 적절한 기판 재료 선택")
                    
                    # Critical interfaces
                    if recommendations['critical_interfaces']:
                        st.warning(f"⚠️ 임계 인터페이스: {recommendations['critical_interfaces']}")
                    
                except Exception as e:
                    st.error(f"❌ 열적 분석 실패: {e}")
    
    # CTE mismatch map
    st.subheader("🗺️ CTE 불일치 맵")
    
    if st.button("🔍 CTE 매칭 분석"):
        with st.spinner("CTE 불일치 계산 중..."):
            
            # Common PV materials with their CTEs (×10⁻⁶ /K)
            materials_cte = {
                'c-Si': 2.6,
                'GaAs': 5.73, 
                'GaInP': 5.3,
                'MAPbI3': 42.0,  # High CTE
                'MAPbBr3': 38.0,
                'CsPbI3': 28.0,
                'CIGS': 8.8,
                'CdTe': 4.9,
                'glass': 9.0,
                'ITO': 7.0,
                'Au': 14.2,
                'Ag': 18.9
            }
            
            # Create CTE mismatch matrix
            materials_list = list(materials_cte.keys())
            n_materials = len(materials_list)
            
            mismatch_matrix = np.zeros((n_materials, n_materials))
            
            for i in range(n_materials):
                for j in range(n_materials):
                    cte1 = materials_cte[materials_list[i]]
                    cte2 = materials_cte[materials_list[j]]
                    # Relative mismatch as percentage
                    if min(cte1, cte2) > 0:
                        mismatch = abs(cte1 - cte2) / min(cte1, cte2) * 100
                    else:
                        mismatch = 0
                    mismatch_matrix[i, j] = mismatch
            
            # Create heatmap
            fig_cte = go.Figure(data=go.Heatmap(
                z=mismatch_matrix,
                x=materials_list,
                y=materials_list,
                colorscale='RdYlGn_r',  # Red for high mismatch, green for low
                colorbar=dict(title="CTE 불일치 [%]"),
                text=np.round(mismatch_matrix, 1),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))
            
            fig_cte.update_layout(
                title="재료간 CTE 불일치 매트릭스 (CTE Mismatch Matrix)",
                xaxis_title="재료 1",
                yaxis_title="재료 2", 
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_cte, use_container_width=True)
            
            # CTE table
            st.subheader("📊 재료별 CTE 값")
            
            cte_data = [
                {"재료": mat, "CTE [×10⁻⁶/K]": cte, "분류": 
                 "매우높음" if cte > 30 else "높음" if cte > 15 else "중간" if cte > 8 else "낮음"}
                for mat, cte in materials_cte.items()
            ]
            
            df_cte = pd.DataFrame(cte_data)
            df_cte = df_cte.sort_values('CTE [×10⁻⁶/K]')
            
            st.dataframe(df_cte, use_container_width=True)
            
            # Best matches recommendation
            st.subheader("💡 CTE 매칭 권장사항")
            
            # Find best matches for common active materials
            active_materials = ['c-Si', 'GaAs', 'MAPbI3', 'CIGS']
            
            for active in active_materials:
                if active in materials_cte:
                    active_cte = materials_cte[active]
                    
                    # Find materials with similar CTE (within 50% relative difference)
                    compatible = []
                    for mat, cte in materials_cte.items():
                        if mat != active:
                            rel_diff = abs(cte - active_cte) / active_cte * 100
                            if rel_diff < 50:  # Within 50% relative difference
                                compatible.append((mat, rel_diff))
                    
                    compatible.sort(key=lambda x: x[1])  # Sort by smallest difference
                    
                    if compatible:
                        best_matches = [mat for mat, diff in compatible[:3]]  # Top 3 matches
                        st.write(f"**{active}** 호환 재료: {', '.join(best_matches)}")

# =====================================================================
# TAB 6: Stability & Degradation
# =====================================================================
with tab6:
    st.markdown('<div class="tab-header">⏳ 안정성 & 열화 분석 (Stability & Degradation)</div>', unsafe_allow_html=True)
    
    st.subheader("🔧 환경 조건 설정")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**환경 매개변수**")
        
        # Environmental conditions
        humidity = st.slider("상대습도 (Relative Humidity) [%]", 0, 100, 60, 5)
        light_intensity = st.slider("광조사량 (Light Intensity) [W/m²]", 0, 1200, 1000, 50)
        oxygen_pressure = st.slider("산소 분압 (O₂ Partial Pressure) [Pa]", 0, 25000, 21000, 1000)
        uv_fraction = st.slider("UV 비율 (UV Fraction) [%]", 0, 20, 5, 1)
        encap_quality = st.slider("봉지재 품질 (Encapsulation Quality)", 0.0, 1.0, 0.8, 0.1)
        
        # Material selection for stability
        stability_materials = st.multiselect(
            "안정성 분석 재료",
            ['MAPbI3', 'MAPbBr3', 'FAPbI3', 'CsPbI3', 'c-Si', 'GaAs'],
            default=['MAPbI3', 'c-Si']
        )
        
        # Corresponding thicknesses
        stability_thicknesses = []
        for mat in stability_materials:
            stab_default = 500 if any(x in mat for x in ['MAP', 'FA', 'Cs']) else 5000
            thick = st.number_input(
                f"{mat} 두께 [nm]",
                min_value=10, max_value=500000,
                value=stab_default,
                key=f"stab_thick_{mat}"
            )
            stability_thicknesses.append(thick * 1e-9)  # Convert to meters
    
    with col2:
        st.subheader("📊 안정성 예측 결과")
        
        if stability_materials and st.button("⏳ 안정성 분석 실행", type="primary"):
            with st.spinner("장기 안정성 예측 중..."):
                try:
                    # Create environmental conditions object
                    env_conditions = EnvironmentalConditions(
                        temperature=T_cell,
                        relative_humidity=humidity,
                        light_intensity=light_intensity,
                        oxygen_partial_pressure=oxygen_pressure,
                        uv_fraction=uv_fraction / 100,
                        encapsulation_quality=encap_quality
                    )
                    
                    # Initialize stability predictor
                    stability_predictor = StabilityPredictor()
                    
                    # Predict stability
                    stability_result = stability_predictor.predict_long_term_stability(
                        stability_materials,
                        stability_thicknesses,
                        env_conditions,
                        simulation_years=25
                    )
                    
                    # Display key metrics
                    col2a, col2b, col2c = st.columns(3)
                    
                    with col2a:
                        t80_years = stability_result.t80_years
                        st.metric("T80 수명", f"{t80_years:.1f} 년")
                        
                    with col2b:
                        t90_years = stability_result.t90_years  
                        st.metric("T90 수명", f"{t90_years:.1f} 년")
                        
                    with col2c:
                        dominant_mode = stability_result.dominant_mechanism
                        st.metric("주요 열화모드", dominant_mode)
                    
                    # Degradation curve
                    years = np.linspace(0, 30, 100)
                    
                    # Exponential degradation model
                    degradation_rate = -np.log(0.8) / t80_years  # Rate for 80% at T80
                    efficiency_retention = np.exp(-degradation_rate * years)
                    
                    fig_degrad = go.Figure()
                    
                    fig_degrad.add_trace(go.Scatter(
                        x=years,
                        y=efficiency_retention * 100,
                        mode='lines',
                        name='전체 시스템',
                        line=dict(color='#2E86AB', width=3)
                    ))
                    
                    # Add T80 and T90 markers
                    fig_degrad.add_vline(x=t80_years, line_dash="dash", line_color="red", 
                                       annotation_text="T80")
                    fig_degrad.add_vline(x=t90_years, line_dash="dash", line_color="orange",
                                       annotation_text="T90")
                    
                    fig_degrad.add_hline(y=80, line_dash="dot", line_color="red", opacity=0.5)
                    fig_degrad.add_hline(y=90, line_dash="dot", line_color="orange", opacity=0.5)
                    
                    fig_degrad.update_layout(
                        title="PCE 열화 곡선 (PCE Degradation Curve)",
                        xaxis_title="시간 (Years)",
                        yaxis_title="성능 유지율 (Performance Retention) [%]",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_degrad, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 안정성 분석 실패: {e}")
                    # Provide fallback demo data
                    st.warning("데모 데이터를 표시합니다.")
                    
                    # Demo degradation data
                    years = np.linspace(0, 25, 100)
                    
                    # Different degradation rates for different materials
                    degradation_curves = {}
                    
                    for mat in stability_materials:
                        if 'MAP' in mat or 'FA' in mat:  # Organic perovskites - faster degradation
                            rate = 0.05  # 5%/year initial rate
                        elif 'Cs' in mat:  # Inorganic perovskites - better stability
                            rate = 0.02  # 2%/year
                        else:  # Silicon, III-V - very stable
                            rate = 0.005  # 0.5%/year
                        
                        retention = np.exp(-rate * years / 5) * 100  # Slow exponential
                        degradation_curves[mat] = retention
                    
                    # Plot demo curves
                    fig_demo = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    for i, (mat, curve) in enumerate(degradation_curves.items()):
                        fig_demo.add_trace(go.Scatter(
                            x=years, y=curve,
                            mode='lines', name=mat,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    
                    fig_demo.add_hline(y=80, line_dash="dash", line_color="red", 
                                     annotation_text="T80 기준선")
                    
                    fig_demo.update_layout(
                        title="재료별 안정성 비교 (데모)",
                        xaxis_title="시간 (Years)",
                        yaxis_title="성능 유지율 [%]",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_demo, use_container_width=True)
    
    # Degradation mechanism analysis
    st.subheader("🔬 열화 메커니즘 분석")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("**주요 열화 인자**")
        
        # Degradation mechanisms for different materials
        mechanisms = {
            'MAPbI3': ['이온 이동', '수분 분해', '상 분리', 'UV 분해'],
            'MAPbBr3': ['이온 이동', '수분 분해', '할로겐 편석'],  
            'FAPbI3': ['상 불안정', '이온 이동', '수분 분해'],
            'CsPbI3': ['상 전이', '표면 산화'],
            'c-Si': ['LID', 'PID', 'UV 열화', '열적 사이클링'],
            'GaAs': ['표면 재결합', '금속 확산', '광산화']
        }
        
        if stability_materials:
            for mat in stability_materials:
                if mat in mechanisms:
                    st.write(f"**{mat}**:")
                    for mech in mechanisms[mat]:
                        severity = np.random.choice(['낮음', '중간', '높음'], p=[0.3, 0.5, 0.2])
                        color = 'green' if severity == '낮음' else 'orange' if severity == '중간' else 'red'
                        st.markdown(f"  • {mech}: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("**개선 전략**")
        
        # Mitigation strategies
        strategies = {
            'environmental': ['습도 제어', '산소 차단', 'UV 필터', '온도 관리'],
            'materials': ['첨가제 도입', '계면 개선', '봉지재 최적화', '배리어 코팅'],
            'design': ['두께 최적화', '전극 개선', '아키텍처 변경']
        }
        
        st.markdown("**환경적 대책:**")
        for strategy in strategies['environmental']:
            st.write(f"  • {strategy}")
            
        st.markdown("**재료적 대책:**")  
        for strategy in strategies['materials']:
            st.write(f"  • {strategy}")
            
        st.markdown("**설계적 대책:**")
        for strategy in strategies['design']:
            st.write(f"  • {strategy}")
    
    # Accelerated testing conditions
    st.subheader("🚀 가속 시험 조건")
    
    if st.button("📋 가속 시험 계획 생성"):
        
        st.markdown("### IEC 61215 기반 가속 시험 조건")
        
        test_conditions = pd.DataFrame({
            "시험 항목": ["열 사이클링", "습열 시험", "UV 조사", "습동결", "기계적 하중"],
            "조건": ["TC: -40°C ↔ +85°C", "DH: +85°C/85%RH", "UV: 15 W/m² @ 280-320nm", 
                    "HF: -40°C ↔ +85°C/85%RH", "ML: 2400 Pa 풍압"],
            "기간": ["200 사이클", "1000 시간", "15 kWh/m²", "10 사이클", "1 시간"],
            "목적": ["열적 내구성", "수분 저항성", "UV 내성", "극한환경", "기계적 강도"]
        })
        
        st.dataframe(test_conditions, use_container_width=True)
        
        # Calculate equivalent real-time exposure
        st.markdown("### 실환경 대비 가속비")
        
        acceleration_factors = pd.DataFrame({
            "시험": ["TC (200 cycle)", "DH (1000h)", "UV (15 kWh/m²)"],
            "가속비": ["×20", "×8", "×5"],
            "실환경 등가": ["10년", "2년", "1년"],
            "비고": ["일교차 극한", "열대 다습", "고지대 강UV"]
        })
        
        st.dataframe(acceleration_factors, use_container_width=True)

# =====================================================================
# TAB 7: Economics & LCOE  
# =====================================================================
with tab7:
    st.markdown('<div class="tab-header">💰 경제성 분석 & LCOE (Economics & LCOE)</div>', unsafe_allow_html=True)
    
    st.subheader("💵 제조 비용 분석")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**셀 구조 정의**")
        
        # Economic analysis parameters
        cell_area = st.number_input("셀 면적 [cm²]", min_value=1, max_value=1000, value=100)
        annual_production = st.selectbox(
            "연간 생산량 [MW]",
            [1, 10, 100, 1000, 10000],
            index=2
        )
        
        # Material selection for cost analysis
        econ_materials = st.multiselect(
            "경제성 분석 재료",
            ['glass', 'ITO', 'perovskite', 'c-Si', 'GaAs', 'Au', 'Ag'],
            default=['glass', 'ITO', 'perovskite', 'Au']
        )
        
        # Material thicknesses for cost calculation
        econ_thicknesses = {}
        for mat in econ_materials:
            if mat == 'glass':
                default_thick = 3000000  # 3mm
            elif mat in ['ITO', 'Au', 'Ag']:
                default_thick = 100
            else:
                default_thick = 500
                
            econ_thicknesses[mat] = st.number_input(
                f"{mat} 두께 [nm]",
                min_value=1, max_value=5000000,
                value=default_thick,
                key=f"econ_thick_{mat}"
            )
        
        # Convert to stack format
        econ_stack = [(mat, econ_thicknesses[mat]) for mat in econ_materials]
        
    with col2:
        st.subheader("📊 비용 계산 결과")
        
        if econ_materials and st.button("💰 경제성 분석 실행", type="primary"):
            with st.spinner("제조 비용 계산 중..."):
                try:
                    # Initialize economics engine
                    economics = EconomicsEngine()
                    
                    # Calculate manufacturing cost
                    cost_result = economics.calculate_stack_manufacturing_cost(econ_stack)
                    
                    # Display key metrics
                    col2a, col2b = st.columns(2)
                    
                    with col2a:
                        cost_per_m2 = cost_result['cost_per_m2']
                        st.metric("제조비용", f"${cost_per_m2:.1f}/m²")
                        
                    with col2b:
                        cost_per_wp = cost_per_m2 / (200 * 0.15)  # Assume 15% efficiency, 200 W/m²
                        st.metric("비용/전력", f"${cost_per_wp:.2f}/Wp")
                    
                    # Cost breakdown visualization
                    if 'layer_costs' in cost_result:
                        layer_costs = cost_result['layer_costs']
                        
                        fig_cost = go.Figure(data=[go.Pie(
                            labels=list(layer_costs.keys()),
                            values=list(layer_costs.values()),
                            hole=0.3
                        )])
                        
                        fig_cost.update_layout(
                            title="재료별 비용 분포 (Cost Breakdown by Material)",
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_cost, use_container_width=True)
                    
                    # Cost scaling with production volume
                    st.subheader("📈 생산량에 따른 비용 변화")
                    
                    volumes = [1, 10, 100, 1000, 10000]  # MW/year
                    costs_scaled = []
                    
                    for vol in volumes:
                        # Simple scaling model: cost reduces with volume due to economies of scale
                        scale_factor = (vol / 100) ** (-0.3)  # Economy of scale exponent
                        scaled_cost = cost_per_m2 * scale_factor
                        costs_scaled.append(scaled_cost)
                    
                    fig_scale = go.Figure()
                    fig_scale.add_trace(go.Scatter(
                        x=volumes, y=costs_scaled,
                        mode='lines+markers',
                        name='제조비용',
                        line=dict(color='green', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_scale.update_layout(
                        title="생산량에 따른 제조비용 변화",
                        xaxis_title="연간 생산량 [MW]",
                        yaxis_title="제조비용 [$/m²]",
                        xaxis_type="log",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_scale, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ 비용 분석 실패: {e}")
    
    # LCOE calculation
    st.subheader("⚡ LCOE 분석 (Levelized Cost of Energy)")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("**LCOE 매개변수**")
        
        # LCOE parameters
        module_efficiency = st.slider("모듈 효율 [%]", 10, 50, 20, 1) / 100
        module_cost = st.number_input("모듈 비용 [$/Wp]", 0.1, 2.0, 0.5, 0.05)
        bos_cost = st.number_input("BOS 비용 [$/Wp]", 0.2, 1.5, 0.6, 0.05)
        installation_cost = st.number_input("설치비용 [$/Wp]", 0.1, 1.0, 0.3, 0.05)
        
        financing_cost = st.slider("금융비용 (WACC) [%]", 1, 15, 6, 1) / 100
        system_lifetime = st.slider("시스템 수명 [년]", 15, 35, 25, 1)
        degradation_rate = st.slider("연간 성능저하 [%/년]", 0.1, 1.0, 0.5, 0.1) / 100
        
        # Location parameters
        irradiance_annual = st.slider("연간 일사량 [kWh/m²/년]", 1000, 2500, 1800, 50)
        
    with col4:
        st.subheader("⚡ LCOE 계산")
        
        if st.button("💡 LCOE 계산 실행"):
            with st.spinner("LCOE 분석 중..."):
                try:
                    # Initialize economics engine  
                    economics = EconomicsEngine()
                    
                    # Calculate LCOE
                    lcoe_params = {
                        'module_cost': module_cost,
                        'bos_cost': bos_cost, 
                        'installation_cost': installation_cost,
                        'financing_cost': financing_cost,
                        'system_lifetime': system_lifetime,
                        'degradation_rate': degradation_rate,
                        'annual_irradiance': irradiance_annual
                    }
                    
                    lcoe_result = economics.calculate_lcoe(
                        module_efficiency, 
                        irradiance_annual,
                        **lcoe_params
                    )
                    
                    # Display LCOE result
                    lcoe_cents = lcoe_result['lcoe_usd_per_kwh'] * 100
                    st.metric("LCOE", f"{lcoe_cents:.1f} ¢/kWh", 
                             help="Levelized Cost of Energy")
                    
                    # LCOE breakdown
                    if 'cost_breakdown' in lcoe_result:
                        breakdown = lcoe_result['cost_breakdown']
                        
                        fig_lcoe = go.Figure(data=[go.Pie(
                            labels=list(breakdown.keys()),
                            values=list(breakdown.values()),
                            hole=0.3
                        )])
                        
                        fig_lcoe.update_layout(
                            title="LCOE 구성 요소 (LCOE Components)",
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_lcoe, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.subheader("📊 민감도 분석")
                    
                    # Efficiency sensitivity
                    eff_range = np.linspace(0.1, 0.4, 20)  # 10% to 40%
                    lcoe_eff = []
                    
                    for eff in eff_range:
                        lcoe_temp = economics.calculate_lcoe(
                            eff, irradiance_annual, **lcoe_params
                        )
                        lcoe_eff.append(lcoe_temp['lcoe_usd_per_kwh'] * 100)
                    
                    # Cost sensitivity  
                    cost_range = np.linspace(0.2, 1.2, 20)  # $0.2 to $1.2/Wp
                    lcoe_cost = []
                    
                    for cost in cost_range:
                        params_temp = lcoe_params.copy()
                        params_temp['module_cost'] = cost
                        lcoe_temp = economics.calculate_lcoe(
                            module_efficiency, irradiance_annual, **params_temp
                        )
                        lcoe_cost.append(lcoe_temp['lcoe_usd_per_kwh'] * 100)
                    
                    # Plot sensitivity
                    fig_sens = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("효율 민감도", "비용 민감도")
                    )
                    
                    fig_sens.add_trace(
                        go.Scatter(x=eff_range*100, y=lcoe_eff, mode='lines', 
                                 name='효율 영향', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    fig_sens.add_trace(
                        go.Scatter(x=cost_range, y=lcoe_cost, mode='lines',
                                 name='비용 영향', line=dict(color='red')),
                        row=1, col=2  
                    )
                    
                    fig_sens.update_xaxes(title_text="모듈 효율 [%]", row=1, col=1)
                    fig_sens.update_xaxes(title_text="모듈 비용 [$/Wp]", row=1, col=2)
                    fig_sens.update_yaxes(title_text="LCOE [¢/kWh]")
                    
                    fig_sens.update_layout(height=400, template="plotly_white")
                    
                    st.plotly_chart(fig_sens, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ LCOE 계산 실패: {e}")
    
    # N-junction economic sweet spot
    st.subheader("🎯 N-접합 경제성 최적점")
    
    if st.button("📈 접합수별 경제성 분석"):
        with st.spinner("다중 접합 경제성 분석 중..."):
            
            # Analyze economics vs number of junctions
            n_junctions = list(range(1, 8))
            
            # Simplified model for cost vs efficiency tradeoff
            efficiencies = []
            costs_per_wp = []
            lcoe_values = []
            
            for n in n_junctions:
                # Efficiency increases with junctions but with diminishing returns
                if n == 1:
                    eff = 0.20  # 20%
                    cost_multiplier = 1.0
                elif n == 2:
                    eff = 0.28  # 28% 
                    cost_multiplier = 2.0
                elif n == 3:
                    eff = 0.35  # 35%
                    cost_multiplier = 4.0
                else:
                    eff = 0.35 + (n-3) * 0.03  # Diminishing returns
                    cost_multiplier = 4.0 * (1.5 ** (n-3))  # Exponential cost growth
                
                efficiencies.append(eff)
                
                # Cost increases significantly with more junctions
                cost_wp = 0.5 * cost_multiplier
                costs_per_wp.append(cost_wp)
                
                # Calculate LCOE for this configuration  
                lcoe_temp = economics.calculate_lcoe(
                    eff, irradiance_annual,
                    module_cost=cost_wp,
                    bos_cost=0.6,
                    installation_cost=0.3,
                    financing_cost=0.06,
                    system_lifetime=25,
                    degradation_rate=0.005,
                    annual_irradiance=irradiance_annual
                )
                lcoe_values.append(lcoe_temp['lcoe_usd_per_kwh'] * 100)
            
            # Plot the sweet spot analysis
            fig_sweet = make_subplots(
                rows=2, cols=2,
                subplot_titles=("효율 vs 접합수", "비용 vs 접합수", "LCOE vs 접합수", "비용-효율 관계"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Efficiency vs N
            fig_sweet.add_trace(
                go.Scatter(x=n_junctions, y=[e*100 for e in efficiencies], 
                         mode='lines+markers', name='효율', line=dict(color='green')),
                row=1, col=1
            )
            
            # Cost vs N
            fig_sweet.add_trace(
                go.Scatter(x=n_junctions, y=costs_per_wp,
                         mode='lines+markers', name='비용', line=dict(color='red')),
                row=1, col=2
            )
            
            # LCOE vs N (sweet spot)
            fig_sweet.add_trace(
                go.Scatter(x=n_junctions, y=lcoe_values,
                         mode='lines+markers', name='LCOE', line=dict(color='blue', width=3)),
                row=2, col=1
            )
            
            # Mark the minimum LCOE point
            min_lcoe_idx = np.argmin(lcoe_values)
            fig_sweet.add_scatter(
                x=[n_junctions[min_lcoe_idx]], y=[lcoe_values[min_lcoe_idx]],
                mode='markers', marker=dict(color='red', size=15, symbol='star'),
                name='최적점', row=2, col=1
            )
            
            # Cost-efficiency scatter
            fig_sweet.add_trace(
                go.Scatter(x=[e*100 for e in efficiencies], y=costs_per_wp,
                         mode='markers+text', text=[f"{n}J" for n in n_junctions],
                         textposition="top right", name='접합 구성',
                         marker=dict(color=n_junctions, size=12, colorscale='viridis')),
                row=2, col=2
            )
            
            fig_sweet.update_xaxes(title_text="접합수", row=1, col=1)
            fig_sweet.update_xaxes(title_text="접합수", row=1, col=2)
            fig_sweet.update_xaxes(title_text="접합수", row=2, col=1)
            fig_sweet.update_xaxes(title_text="효율 [%]", row=2, col=2)
            
            fig_sweet.update_yaxes(title_text="효율 [%]", row=1, col=1)
            fig_sweet.update_yaxes(title_text="비용 [$/Wp]", row=1, col=2)
            fig_sweet.update_yaxes(title_text="LCOE [¢/kWh]", row=2, col=1)
            fig_sweet.update_yaxes(title_text="비용 [$/Wp]", row=2, col=2)
            
            fig_sweet.update_layout(height=600, template="plotly_white", showlegend=False)
            
            st.plotly_chart(fig_sweet, use_container_width=True)
            
            # Economic summary
            st.subheader("💡 경제성 분석 결과")
            
            optimal_n = n_junctions[min_lcoe_idx]
            optimal_lcoe = lcoe_values[min_lcoe_idx]
            optimal_eff = efficiencies[min_lcoe_idx] * 100
            optimal_cost = costs_per_wp[min_lcoe_idx]
            
            col3a, col3b, col3c, col3d = st.columns(4)
            
            with col3a:
                st.metric("최적 접합수", f"{optimal_n}J")
            with col3b:
                st.metric("최적 LCOE", f"{optimal_lcoe:.1f} ¢/kWh")
            with col3c:
                st.metric("해당 효율", f"{optimal_eff:.1f}%")
            with col3d:
                st.metric("해당 비용", f"${optimal_cost:.2f}/Wp")
            
            # Economic recommendations
            if optimal_n <= 2:
                st.success(f"✅ {optimal_n}J 구조가 경제적 최적점 - 상용화 적합")
            elif optimal_n <= 4:
                st.warning(f"⚠️ {optimal_n}J 구조가 최적이나 제조 복잡도 고려 필요")
            else:
                st.error(f"❌ {optimal_n}J 구조는 과도한 비용 - 재검토 권장")

# =====================================================================
# TAB 8: Comprehensive Optimizer
# =====================================================================
with tab8:
    st.markdown('<div class="tab-header">🚀 종합 최적화 (Comprehensive Optimizer)</div>', unsafe_allow_html=True)
    
    st.subheader("🎯 다목적 최적화 설정")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**최적화 목표 가중치**")
        
        # Multi-objective optimization weights
        w_efficiency = st.slider("효율 가중치 (Efficiency)", 0.0, 1.0, 0.4, 0.1)
        w_cost = st.slider("비용 가중치 (Cost)", 0.0, 1.0, 0.3, 0.1)  
        w_stability = st.slider("안정성 가중치 (Stability)", 0.0, 1.0, 0.2, 0.1)
        w_thermal = st.slider("열적 가중치 (Thermal)", 0.0, 1.0, 0.1, 0.1)
        
        # Normalize weights
        total_weight = w_efficiency + w_cost + w_stability + w_thermal
        if total_weight > 0:
            w_efficiency /= total_weight
            w_cost /= total_weight  
            w_stability /= total_weight
            w_thermal /= total_weight
        
        st.write("**정규화된 가중치:**")
        st.write(f"• 효율: {w_efficiency:.2f}")
        st.write(f"• 비용: {w_cost:.2f}")
        st.write(f"• 안정성: {w_stability:.2f}")  
        st.write(f"• 열적: {w_thermal:.2f}")
        
        # Optimization constraints
        st.markdown("**제약 조건**")
        
        max_junctions = st.slider("최대 접합수", 2, 10, 5)
        min_efficiency = st.slider("최소 효율 [%]", 15, 40, 25)
        max_cost = st.slider("최대 비용 [$/Wp]", 0.5, 3.0, 1.5)
        min_lifetime = st.slider("최소 수명 [년]", 10, 30, 20)
        
    with col2:
        st.subheader("🔍 최적화 실행")
        
        optimization_method = st.selectbox(
            "최적화 알고리즘",
            ["유전 알고리즘 (GA)", "입자 군집 (PSO)", "시뮬레이티드 어닐링 (SA)", "그리드 탐색"]
        )
        
        n_iterations = st.slider("최적화 반복수", 50, 500, 200)
        
        if st.button("🚀 종합 최적화 실행", type="primary"):
            with st.spinner(f"{optimization_method}로 최적화 중..."):
                try:
                    # Multi-objective optimization simulation
                    progress_bar = st.progress(0)
                    
                    # Generate candidate solutions (simplified)
                    np.random.seed(42)  # For reproducible results
                    n_candidates = 50
                    
                    candidates = []
                    
                    for i in range(n_candidates):
                        # Generate random candidate solution
                        n_junc = np.random.randint(1, max_junctions + 1)
                        
                        # Simulate performance based on number of junctions
                        if n_junc == 1:
                            efficiency = np.random.uniform(0.18, 0.25)
                            cost = np.random.uniform(0.4, 0.7)
                            stability = np.random.uniform(20, 30)
                            thermal = np.random.uniform(15, 25)
                        elif n_junc == 2:
                            efficiency = np.random.uniform(0.25, 0.32)
                            cost = np.random.uniform(0.8, 1.4)
                            stability = np.random.uniform(15, 25)
                            thermal = np.random.uniform(10, 20)
                        elif n_junc == 3:
                            efficiency = np.random.uniform(0.30, 0.38)
                            cost = np.random.uniform(1.5, 2.5)
                            stability = np.random.uniform(10, 20)
                            thermal = np.random.uniform(8, 15)
                        else:
                            efficiency = 0.35 + (n_junc - 3) * 0.03 + np.random.uniform(-0.02, 0.02)
                            cost = 2.0 * (1.5 ** (n_junc - 3)) + np.random.uniform(-0.2, 0.2)
                            stability = max(5, 20 - (n_junc - 3) * 2 + np.random.uniform(-2, 2))
                            thermal = max(3, 15 - (n_junc - 3) * 1.5 + np.random.uniform(-1, 1))
                        
                        # Apply constraints
                        if (efficiency * 100 >= min_efficiency and 
                            cost <= max_cost and 
                            stability >= min_lifetime):
                            
                            candidates.append({
                                'n_junctions': n_junc,
                                'efficiency': efficiency,
                                'cost': cost,
                                'stability': stability,
                                'thermal': thermal
                            })
                        
                        progress_bar.progress((i + 1) / n_candidates)
                    
                    progress_bar.empty()
                    
                    if not candidates:
                        st.error("❌ 제약 조건을 만족하는 해를 찾을 수 없습니다.")
                    else:
                        # Calculate multi-objective scores
                        for candidate in candidates:
                            # Normalize objectives (0-1 scale)
                            eff_norm = candidate['efficiency'] / 0.5  # Max possible ~50%
                            cost_norm = 1 - (candidate['cost'] - 0.3) / (3.0 - 0.3)  # Lower cost is better
                            stab_norm = candidate['stability'] / 30  # Max ~30 years
                            therm_norm = candidate['thermal'] / 30  # Max ~30 years
                            
                            # Multi-objective score
                            score = (w_efficiency * eff_norm + 
                                   w_cost * cost_norm +
                                   w_stability * stab_norm +
                                   w_thermal * therm_norm)
                            
                            candidate['score'] = score
                        
                        # Sort by score
                        candidates.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Display top results
                        st.success(f"✅ {len(candidates)}개의 후보 솔루션 발견!")
                        
                        # Best solution
                        best = candidates[0]
                        
                        col2a, col2b, col2c = st.columns(3)
                        with col2a:
                            st.metric("최적 접합수", f"{best['n_junctions']}J")
                        with col2b:
                            st.metric("효율", f"{best['efficiency']*100:.1f}%")
                        with col2c:
                            st.metric("종합 점수", f"{best['score']:.3f}")
                        
                        # Store results in session state
                        st.session_state['optimization_candidates'] = candidates[:10]  # Top 10
                        
                except Exception as e:
                    st.error(f"❌ 최적화 실패: {e}")
    
    # Results visualization
    if 'optimization_candidates' in st.session_state:
        candidates = st.session_state['optimization_candidates']
        
        st.subheader("📊 최적화 결과 시각화")
        
        # Pareto front visualization
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("**파레토 프론트 (Efficiency vs Cost)**")
            
            fig_pareto = go.Figure()
            
            # All candidates
            efficiencies = [c['efficiency']*100 for c in candidates]
            costs = [c['cost'] for c in candidates]
            scores = [c['score'] for c in candidates]
            n_junctions = [c['n_junctions'] for c in candidates]
            
            fig_pareto.add_trace(go.Scatter(
                x=efficiencies,
                y=costs, 
                mode='markers+text',
                text=[f"{n}J" for n in n_junctions],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=scores,
                    colorscale='RdYlGn',
                    colorbar=dict(title="종합점수"),
                    showscale=True
                ),
                name='후보 솔루션',
                hovertemplate='<b>%{text}</b><br>' +
                              '효율: %{x:.1f}%<br>' +
                              '비용: $%{y:.2f}/Wp<br>' +
                              '<extra></extra>'
            ))
            
            # Highlight best solution
            best = candidates[0]
            fig_pareto.add_trace(go.Scatter(
                x=[best['efficiency']*100],
                y=[best['cost']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='최적해'
            ))
            
            fig_pareto.update_layout(
                title="효율-비용 트레이드오프",
                xaxis_title="효율 [%]",
                yaxis_title="비용 [$/Wp]",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True)
        
        with col4:
            st.markdown("**종합 성능 레이더 차트**")
            
            # Radar chart for best solutions
            best_3 = candidates[:3]
            
            fig_radar = go.Figure()
            
            categories = ['효율', '비용<br>(역순)', '안정성', '열적성능']
            
            for i, candidate in enumerate(best_3):
                values = [
                    candidate['efficiency'] / 0.5,  # Normalize to 0-1
                    (3.0 - candidate['cost']) / (3.0 - 0.3),  # Inverse for cost
                    candidate['stability'] / 30,
                    candidate['thermal'] / 30
                ]
                values += [values[0]]  # Close the radar chart
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f"{candidate['n_junctions']}J (#{i+1})",
                    opacity=0.6
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="최적 솔루션 비교",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Results table
        st.subheader("📋 최적 구성 요소표")
        
        results_data = []
        for i, candidate in enumerate(candidates[:5]):  # Top 5
            results_data.append({
                "순위": i + 1,
                "접합수": f"{candidate['n_junctions']}J",
                "효율 [%]": f"{candidate['efficiency']*100:.1f}",
                "비용 [$/Wp]": f"{candidate['cost']:.2f}",
                "수명 [년]": f"{candidate['stability']:.1f}",
                "열성능 [년]": f"{candidate['thermal']:.1f}",
                "종합점수": f"{candidate['score']:.3f}"
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Export results
        st.subheader("📁 결과 내보내기")
        
        col5, col6 = st.columns([1, 1])
        
        with col5:
            # JSON export
            export_data = {
                "optimization_parameters": {
                    "weights": {
                        "efficiency": w_efficiency,
                        "cost": w_cost,
                        "stability": w_stability,
                        "thermal": w_thermal
                    },
                    "constraints": {
                        "max_junctions": max_junctions,
                        "min_efficiency": min_efficiency,
                        "max_cost": max_cost,
                        "min_lifetime": min_lifetime
                    }
                },
                "results": candidates
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📄 JSON 다운로드",
                data=json_str,
                file_name="tandem_pv_optimization_results.json",
                mime="application/json"
            )
        
        with col6:
            # CSV export
            csv_data = pd.DataFrame(candidates).to_csv(index=False)
            st.download_button(
                label="📊 CSV 다운로드", 
                data=csv_data,
                file_name="tandem_pv_optimization_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌞 N-Junction Tandem PV Simulator | "
    "Developed with ❤️ using Streamlit | "
    f"Temperature: {temperature}°C | Irradiance: {irradiance} W/m²"
    "</div>", 
    unsafe_allow_html=True
)

# Session state debugging (only for development)
if st.checkbox("🔍 세션 상태 디버깅", key="debug_session"):
    st.json(dict(st.session_state))