#!/usr/bin/env python3
"""
Pre-computed ABX₃ Perovskite Database Generator
============================================

Generates comprehensive ABX₃ property lookup table for V3 tandem simulator.
~2000-5000 compositions at 10% step grid for A, B, X sites.

Physics Models:
- Bandgap: Vegard's law + bowing parameters (literature)
- Crystal phase: Tolerance factor model
- Optical: Urbach tail + direct gap model
- Carrier transport: Pb/Sn composition-dependent

Output: data/perovskite_db.parquet + .csv
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, Tuple, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.ml_bandgap import ML_BANDGAP_PREDICTOR

print("🔬 Generating ABX₃ Perovskite Database...")
print("=" * 50)

# =============================================================================
# COMPOSITIONAL GRID GENERATION
# =============================================================================

def generate_composition_grid(step_size: float = 0.1) -> List[Dict]:
    """Generate ABX₃ composition grid with constraint: fractions sum to 1.0"""
    
    # A-site: MA, FA, Cs (skip Rb for simplicity)
    # B-site: Pb, Sn (skip Ge)  
    # X-site: I, Br, Cl
    
    compositions = []
    n_steps = int(1.0 / step_size) + 1
    steps = np.linspace(0, 1, n_steps)
    
    print(f"📊 Grid resolution: {step_size:.1f} ({n_steps} points per axis)")
    print(f"🔢 Target compositions: ~{n_steps**6:.0f} (before constraints)")
    
    count = 0
    for a_ma in steps:
        for a_fa in steps:
            for a_cs in steps:
                # A-site constraint: fractions sum to 1
                if abs(a_ma + a_fa + a_cs - 1.0) > 1e-6:
                    continue
                    
                for b_pb in steps:
                    for b_sn in steps:
                        # B-site constraint: fractions sum to 1
                        if abs(b_pb + b_sn - 1.0) > 1e-6:
                            continue
                            
                        for x_i in steps:
                            for x_br in steps:
                                for x_cl in steps:
                                    # X-site constraint: fractions sum to 1
                                    if abs(x_i + x_br + x_cl - 1.0) > 1e-6:
                                        continue
                                    
                                    # Valid composition
                                    composition = {
                                        'A_MA': a_ma,
                                        'A_FA': a_fa, 
                                        'A_Cs': a_cs,
                                        'A_Rb': 0.0,
                                        'B_Pb': b_pb,
                                        'B_Sn': b_sn,
                                        'B_Ge': 0.0,
                                        'X_I': x_i,
                                        'X_Br': x_br,
                                        'X_Cl': x_cl
                                    }
                                    compositions.append(composition)
                                    count += 1
    
    print(f"✅ Generated {count} valid compositions")
    return compositions

# =============================================================================
# PHYSICS MODELS
# =============================================================================

# Literature bandgaps [eV] - reference values
REFERENCE_BANDGAPS = {
    ('MA', 'Pb', 'I'): 1.55, ('MA', 'Pb', 'Br'): 2.30, ('MA', 'Pb', 'Cl'): 3.0,
    ('FA', 'Pb', 'I'): 1.48, ('FA', 'Pb', 'Br'): 2.23, ('FA', 'Pb', 'Cl'): 2.9,
    ('Cs', 'Pb', 'I'): 1.73, ('Cs', 'Pb', 'Br'): 2.36, ('Cs', 'Pb', 'Cl'): 3.0,
    ('MA', 'Sn', 'I'): 1.30, ('MA', 'Sn', 'Br'): 2.15, ('MA', 'Sn', 'Cl'): 2.8,
    ('FA', 'Sn', 'I'): 1.41, ('FA', 'Sn', 'Br'): 2.2, ('FA', 'Sn', 'Cl'): 2.85,
    ('Cs', 'Sn', 'I'): 1.30, ('Cs', 'Sn', 'Br'): 2.18, ('Cs', 'Sn', 'Cl'): 2.9,
}

# Bowing parameters [eV] - non-linear mixing
BOWING_PARAMETERS = {
    ('I', 'Br'): 0.33,   # I-Br mixing
    ('I', 'Cl'): 0.65,   # I-Cl mixing  
    ('Br', 'Cl'): 0.23,  # Br-Cl mixing
    ('Pb', 'Sn'): 0.30,  # Pb-Sn mixing
}

# Shannon ionic radii [Å]
IONIC_RADII = {
    'MA': 1.8,  'FA': 1.9,  'Cs': 1.67, 'Rb': 1.52,  # A-site
    'Pb': 1.19, 'Sn': 1.10, 'Ge': 0.87,             # B-site  
    'I': 2.20,  'Br': 1.96, 'Cl': 1.81              # X-site
}

# Phase transition temperatures [K] for known compositions
PHASE_TRANSITIONS = {
    ('MA', 'Pb', 'I'): 327,   # tetra→cubic
    ('FA', 'Pb', 'I'): 285,   # α-phase instability 
    ('Cs', 'Pb', 'I'): 593,   # black phase above this
    ('MA', 'Sn', 'I'): 250,   # oxidation onset
}

def calculate_bandgap(composition: Dict) -> Tuple[float, int]:
    """
    Calculate bandgap using Vegard's law + bowing + ML fallback
    Returns: (bandgap_eV, confidence_level)
    """
    
    # Extract site compositions
    a_sites = [('MA', composition['A_MA']), ('FA', composition['A_FA']), 
               ('Cs', composition['A_Cs'])]
    b_sites = [('Pb', composition['B_Pb']), ('Sn', composition['B_Sn'])]
    x_sites = [('I', composition['X_I']), ('Br', composition['X_Br']), 
               ('Cl', composition['X_Cl'])]
    
    # Vegard's law linear interpolation
    eg_linear = 0.0
    total_weight = 0.0
    confidence = 3  # Start with high confidence
    
    for (a_ion, a_frac) in a_sites:
        for (b_ion, b_frac) in b_sites:
            for (x_ion, x_frac) in x_sites:
                weight = a_frac * b_frac * x_frac
                if weight > 1e-6:
                    ref_key = (a_ion, b_ion, x_ion)
                    if ref_key in REFERENCE_BANDGAPS:
                        eg_linear += weight * REFERENCE_BANDGAPS[ref_key]
                        total_weight += weight
                    else:
                        # Missing reference - lower confidence
                        confidence = min(confidence, 2)
                        # Estimate using similar compositions
                        estimated_eg = estimate_missing_bandgap(a_ion, b_ion, x_ion)
                        eg_linear += weight * estimated_eg
                        total_weight += weight
    
    if total_weight == 0:
        # Fallback to ML predictor
        try:
            eg_ml, uncertainty = ML_BANDGAP_PREDICTOR.predict(composition)
            return eg_ml, 1 if uncertainty > 0.1 else 2
        except:
            return 2.0, 0  # Emergency fallback
    
    eg_linear /= total_weight
    
    # Apply bowing parameter corrections
    eg_bowed = eg_linear
    
    # X-site bowing
    for (x1, x2), bowing in BOWING_PARAMETERS.items():
        if x1 in ['I', 'Br', 'Cl'] and x2 in ['I', 'Br', 'Cl']:
            frac1 = composition[f'X_{x1}']
            frac2 = composition[f'X_{x2}']
            if frac1 > 0 and frac2 > 0:
                eg_bowed -= bowing * frac1 * frac2
    
    # B-site bowing  
    pb_frac = composition['B_Pb']
    sn_frac = composition['B_Sn']
    if pb_frac > 0 and sn_frac > 0:
        eg_bowed -= BOWING_PARAMETERS[('Pb', 'Sn')] * pb_frac * sn_frac
        confidence = min(confidence, 2)  # Mixed B-site less certain
    
    # Ensure physical range
    eg_bowed = np.clip(eg_bowed, 0.5, 4.0)
    
    return eg_bowed, confidence

def estimate_missing_bandgap(a_ion: str, b_ion: str, x_ion: str) -> float:
    """Estimate bandgap for missing reference using interpolation"""
    
    # Simple model based on electronegativity differences
    electronegativity = {
        'MA': 2.2, 'FA': 2.1, 'Cs': 0.79,
        'Pb': 2.33, 'Sn': 1.96, 'Ge': 2.01,
        'I': 2.66, 'Br': 2.96, 'Cl': 3.16
    }
    
    en_diff = abs(electronegativity[b_ion] - electronegativity[x_ion])
    base_eg = 1.0 + 0.8 * en_diff  # Empirical relation
    
    # A-site organic/inorganic correction
    if a_ion in ['MA', 'FA']:
        base_eg -= 0.1  # Organic cations lower Eg slightly
    
    return np.clip(base_eg, 0.8, 3.5)

def calculate_tolerance_factor(composition: Dict) -> float:
    """Calculate Goldschmidt tolerance factor"""
    
    # Effective ionic radii  
    r_A = (composition['A_MA'] * IONIC_RADII['MA'] + 
           composition['A_FA'] * IONIC_RADII['FA'] +
           composition['A_Cs'] * IONIC_RADII['Cs'])
    
    r_B = (composition['B_Pb'] * IONIC_RADII['Pb'] + 
           composition['B_Sn'] * IONIC_RADII['Sn'])
    
    r_X = (composition['X_I'] * IONIC_RADII['I'] + 
           composition['X_Br'] * IONIC_RADII['Br'] + 
           composition['X_Cl'] * IONIC_RADII['Cl'])
    
    if r_B + r_X > 0:
        tolerance_factor = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
    else:
        tolerance_factor = 0.9  # Default
    
    return tolerance_factor

def determine_crystal_phase(tolerance_factor: float) -> str:
    """Map tolerance factor to crystal phase"""
    
    if tolerance_factor < 0.8:
        return "orthorhombic"
    elif 0.8 <= tolerance_factor <= 1.0:
        return "cubic"  # or tetragonal - simplified
    else:
        return "hexagonal"  # non-perovskite

def calculate_phase_transition_temp(composition: Dict) -> float:
    """Estimate phase transition temperature [K]"""
    
    # Weight average of known transition temperatures
    total_temp = 0.0
    total_weight = 0.0
    
    for (a, b, x), temp in PHASE_TRANSITIONS.items():
        weight = composition[f'A_{a}'] * composition[f'B_{b}'] * composition[f'X_{x}']
        if weight > 1e-6:
            total_temp += weight * temp
            total_weight += weight
    
    if total_weight > 0:
        return total_temp / total_weight
    else:
        # Default based on A-site composition
        organic_frac = composition['A_MA'] + composition['A_FA']
        return 300 + 50 * (1 - organic_frac)  # Inorganics more stable

def calculate_lattice_parameter(composition: Dict) -> float:
    """Calculate lattice parameter using Vegard's law [Å]"""
    
    # Reference lattice parameters [Å] for cubic phase
    ref_lattice = {
        ('MA', 'Pb', 'I'): 6.28, ('MA', 'Pb', 'Br'): 5.93, ('MA', 'Pb', 'Cl'): 5.68,
        ('FA', 'Pb', 'I'): 6.36, ('Cs', 'Pb', 'I'): 6.29, ('Cs', 'Pb', 'Br'): 5.87,
        ('MA', 'Sn', 'I'): 6.24, ('FA', 'Sn', 'I'): 6.31, ('Cs', 'Sn', 'I'): 6.22
    }
    
    total_lattice = 0.0
    total_weight = 0.0
    
    for (a, b, x), lattice in ref_lattice.items():
        weight = composition[f'A_{a}'] * composition[f'B_{b}'] * composition[f'X_{x}']
        if weight > 1e-6:
            total_lattice += weight * lattice
            total_weight += weight
    
    if total_weight > 0:
        return total_lattice / total_weight
    else:
        return 6.0  # Default cubic perovskite

def calculate_properties(composition: Dict, bandgap: float) -> Dict:
    """Calculate all material properties from composition and bandgap"""
    
    # Tolerance factor and crystal phase
    tolerance_factor = calculate_tolerance_factor(composition)
    crystal_phase = determine_crystal_phase(tolerance_factor)
    phase_transition_temp = calculate_phase_transition_temp(composition)
    lattice_param = calculate_lattice_parameter(composition)
    
    # Surface energy model [J/m²]
    # Higher for smaller halides, lower for organic A-sites
    halide_factor = (composition['X_I'] * 0.5 + 
                    composition['X_Br'] * 1.0 + 
                    composition['X_Cl'] * 1.5)
    organic_factor = composition['A_MA'] + composition['A_FA']
    surface_energy = 0.8 + halide_factor - 0.3 * organic_factor
    surface_energy = np.clip(surface_energy, 0.5, 2.0)
    
    # Absorption coefficient at 500nm [cm⁻¹]
    # Direct bandgap: α = A * sqrt(E - Eg) for E > Eg
    # Urbach tail below bandgap
    E_500nm = 1240 / 500  # 2.48 eV
    if E_500nm > bandgap:
        absorption_coeff_500nm = 1e5 * np.sqrt(E_500nm - bandgap)
    else:
        # Urbach tail: α = α₀ * exp((E - Eg)/E_u)
        urbach_energy = 0.015  # eV, typical for perovskites
        absorption_coeff_500nm = 1e3 * np.exp((E_500nm - bandgap) / urbach_energy)
    
    absorption_coeff_500nm = np.clip(absorption_coeff_500nm, 1e2, 1e6)
    
    # Refractive index at 550nm (Moss relation approximation)
    n_550 = 1.5 + 1.0 / bandgap
    n_550 = np.clip(n_550, 2.0, 4.0)
    
    # Exciton binding energy [meV]
    # Organic A-sites → lower, inorganic → higher
    organic_frac = composition['A_MA'] + composition['A_FA']
    exciton_binding_energy = 15 - 10 * organic_frac + 20 * composition['A_Cs']
    exciton_binding_energy = np.clip(exciton_binding_energy, 5, 50)
    
    # Carrier mobility [cm²/Vs]
    # Pb → higher, Sn → lower (oxidation), composition-dependent
    pb_frac = composition['B_Pb']
    sn_frac = composition['B_Sn']
    electron_mobility = pb_frac * 20 + sn_frac * 3  # cm²/Vs
    hole_mobility = pb_frac * 15 + sn_frac * 2
    carrier_mobility = (electron_mobility + hole_mobility) / 2
    carrier_mobility = np.clip(carrier_mobility, 0.5, 50)
    
    # Exciton lifetime [ns]
    # I → longer, Br → shorter, mixed → intermediate
    i_frac = composition['X_I']
    br_frac = composition['X_Br']
    cl_frac = composition['X_Cl']
    exciton_lifetime = i_frac * 100 + br_frac * 10 + cl_frac * 5
    exciton_lifetime = np.clip(exciton_lifetime, 1, 1000)
    
    # Defect tolerance score [0-10]
    # Pb/Sn ratio, halide mixing effects
    pb_score = pb_frac * 8  # Pb more defect tolerant
    sn_score = sn_frac * 4  # Sn more sensitive
    halide_score = i_frac * 2 + br_frac * 1.5 + cl_frac * 1  # I best
    defect_tolerance = pb_score + sn_score + halide_score
    defect_tolerance = np.clip(defect_tolerance, 0, 10)
    
    # Coefficient of thermal expansion [ppm/K]
    # Organic A-sites → higher CTE, inorganic → lower
    organic_cte = (composition['A_MA'] + composition['A_FA']) * 40
    inorganic_cte = composition['A_Cs'] * 25
    cte = organic_cte + inorganic_cte + 10  # Base value
    cte = np.clip(cte, 10, 60)
    
    # Stability score [0-10]
    # Phase stability + moisture + thermal
    phase_stability = 10 if tolerance_factor > 0.8 else 5
    moisture_resistance = composition['A_Cs'] * 8 + (composition['A_MA'] + composition['A_FA']) * 2
    thermal_stability = 8 - 3 * sn_frac  # Sn oxidation issue
    stability_score = (phase_stability + moisture_resistance + thermal_stability) / 3
    stability_score = np.clip(stability_score, 0, 10)
    
    # Phase stable at room temperature
    phase_stable_RT = (crystal_phase in ["cubic", "tetragonal"] and 
                      tolerance_factor > 0.75 and 
                      phase_transition_temp < 350)
    
    return {
        'tolerance_factor': tolerance_factor,
        'crystal_phase': crystal_phase,
        'phase_transition_temp': phase_transition_temp,
        'lattice_param': lattice_param,
        'surface_energy': surface_energy,
        'absorption_coeff_500nm': absorption_coeff_500nm,
        'n_550': n_550,
        'exciton_binding_energy': exciton_binding_energy,
        'carrier_mobility': carrier_mobility,
        'exciton_lifetime': exciton_lifetime,
        'defect_tolerance': defect_tolerance,
        'CTE': cte,
        'stability_score': stability_score,
        'phase_stable_RT': phase_stable_RT
    }

# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

def generate_database(step_size: float = 0.1) -> pd.DataFrame:
    """Generate complete ABX₃ property database"""
    
    print("🔄 Generating composition grid...")
    compositions = generate_composition_grid(step_size)
    
    print("⚙️ Calculating properties...")
    data_rows = []
    
    # Fit ML model once at start
    try:
        ML_BANDGAP_PREDICTOR.fit()
        print("✅ ML bandgap predictor ready")
    except Exception as e:
        print(f"⚠️ ML predictor failed: {e}")
        print("📉 Falling back to Vegard's law only")
    
    for i, composition in enumerate(compositions):
        if i % 500 == 0:
            print(f"📊 Progress: {i}/{len(compositions)} ({100*i/len(compositions):.1f}%)")
        
        try:
            # Calculate bandgap and confidence
            bandgap, confidence = calculate_bandgap(composition)
            
            # Calculate all other properties
            properties = calculate_properties(composition, bandgap)
            
            # Create database row
            row = composition.copy()
            row.update({
                'Eg': bandgap,
                'confidence': confidence,
                **properties
            })
            
            data_rows.append(row)
            
        except Exception as e:
            print(f"⚠️ Error at composition {i}: {e}")
            continue
    
    print(f"✅ Generated {len(data_rows)} property records")
    return pd.DataFrame(data_rows)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Generate database
    df = generate_database(step_size=0.1)
    
    print("\n📈 Database Statistics:")
    print(f"📊 Total compositions: {len(df)}")
    print(f"🎯 Bandgap range: {df['Eg'].min():.2f} - {df['Eg'].max():.2f} eV")
    print(f"⭐ High confidence (3): {(df['confidence'] == 3).sum()}")
    print(f"⭐ Med confidence (2): {(df['confidence'] == 2).sum()}")
    print(f"⭐ Low confidence (1): {(df['confidence'] == 1).sum()}")
    print(f"🔬 Phase distribution:")
    print(df['crystal_phase'].value_counts())
    print(f"✅ RT stable compositions: {df['phase_stable_RT'].sum()}")
    
    # Save to both formats
    output_parquet = "data/perovskite_db.parquet"
    output_csv = "data/perovskite_db.csv"
    
    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)
    
    print(f"\n💾 Database saved:")
    print(f"📦 Parquet: {output_parquet} ({os.path.getsize(output_parquet)/1024:.1f} KB)")
    print(f"📄 CSV: {output_csv} ({os.path.getsize(output_csv)/1024:.1f} KB)")
    
    print("\n🎉 ABX₃ Database generation complete!")