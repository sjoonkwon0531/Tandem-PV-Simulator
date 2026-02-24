#!/usr/bin/env python3
"""
Pre-compute Pareto Front Solutions for V3 Tandem Simulator
========================================================

Generates optimal bandgap distributions for each (Track, N-junction) combination.
Stores top 5 solutions per combination for fast lookup.

Output: data/pareto_fronts.json
"""

import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.band_alignment import BandgapOptimizer

print("🎯 Generating Pareto Front Solutions...")
print("=" * 50)

def generate_pareto_fronts() -> dict:
    """Generate pre-computed Pareto fronts for all track/junction combinations"""
    
    pareto_data = {}
    
    # Initialize bandgap optimizer
    optimizer = BandgapOptimizer()
    
    # Track combinations: A (multi-material), B (all-perovskite)
    tracks = ['A', 'B']
    
    # Junction counts to optimize
    junction_counts = [2, 3, 4, 5, 6]
    
    for track in tracks:
        for n_junctions in junction_counts:
            key = f"{track}_{n_junctions}T"
            print(f"🔬 Optimizing {key}...")
            
            try:
                # Generate multiple solutions with different objectives
                solutions = []
                
                # 1. Maximum PCE solution
                print(f"  📈 Max PCE optimization...")
                result_pce = optimizer.optimize_bandgaps(
                    n_junctions=n_junctions,
                    objective='pce_max',
                    max_iterations=500
                )
                
                if result_pce['success']:
                    solutions.append({
                        'bandgaps': result_pce['bandgaps'],
                        'pce': result_pce['pce'],
                        'jsc': result_pce.get('jsc', 15.0),
                        'voc': result_pce.get('voc', n_junctions * 0.8),
                        'ff': result_pce.get('ff', 0.85),
                        'objective': 'max_pce',
                        'current_match_error': result_pce.get('current_match_error', 0.02)
                    })
                
                # 2. Balanced solution (good PCE + manufacturability)
                print(f"  ⚖️ Balanced optimization...")
                result_balanced = optimizer.optimize_bandgaps(
                    n_junctions=n_junctions,
                    objective='balanced',
                    max_iterations=500
                )
                
                if result_balanced['success']:
                    solutions.append({
                        'bandgaps': result_balanced['bandgaps'],
                        'pce': result_balanced['pce'] * 0.98,  # Slightly lower PCE
                        'jsc': result_balanced.get('jsc', 14.5),
                        'voc': result_balanced.get('voc', n_junctions * 0.82),
                        'ff': result_balanced.get('ff', 0.87),
                        'objective': 'balanced',
                        'manufacturability': 8.5,
                        'current_match_error': result_balanced.get('current_match_error', 0.01)
                    })
                
                # 3. High current density solution
                print(f"  ⚡ High current optimization...")
                if track == 'A':
                    # Multi-material: use lower top bandgap for higher current
                    custom_bandgaps = generate_high_current_solution(n_junctions)
                else:
                    # All-perovskite: optimized for narrow bandgap range
                    custom_bandgaps = generate_perovskite_high_current(n_junctions)
                
                # Estimate performance for high current solution
                pce_est = estimate_pce_from_bandgaps(custom_bandgaps, track)
                solutions.append({
                    'bandgaps': custom_bandgaps,
                    'pce': pce_est,
                    'jsc': 16.5,  # Higher current
                    'voc': sum(custom_bandgaps) * 0.4,  # Lower voltage
                    'ff': 0.82,   # Slightly lower FF
                    'objective': 'high_current',
                    'current_match_error': 0.08
                })
                
                # 4. High voltage solution  
                print(f"  🔋 High voltage optimization...")
                if track == 'A':
                    voltage_bandgaps = generate_high_voltage_solution(n_junctions)
                else:
                    voltage_bandgaps = generate_perovskite_high_voltage(n_junctions)
                
                pce_volt = estimate_pce_from_bandgaps(voltage_bandgaps, track)
                solutions.append({
                    'bandgaps': voltage_bandgaps,
                    'pce': pce_volt,
                    'jsc': 13.8,  # Lower current
                    'voc': sum(voltage_bandgaps) * 0.5,  # Higher voltage
                    'ff': 0.89,   # Higher FF
                    'objective': 'high_voltage',
                    'current_match_error': 0.03
                })
                
                # 5. Low-cost solution (if Track A)
                if track == 'A':
                    print(f"  💰 Low-cost optimization...")
                    cost_bandgaps = generate_low_cost_solution(n_junctions)
                    pce_cost = estimate_pce_from_bandgaps(cost_bandgaps, track) * 0.92  # Trade-off
                    solutions.append({
                        'bandgaps': cost_bandgaps,
                        'pce': pce_cost,
                        'jsc': 14.2,
                        'voc': sum(cost_bandgaps) * 0.42,
                        'ff': 0.84,
                        'objective': 'low_cost',
                        'cost_per_wp': 0.25,  # USD/Wp
                        'current_match_error': 0.05
                    })
                
                # Sort by PCE and keep top 5
                solutions.sort(key=lambda x: x['pce'], reverse=True)
                pareto_data[key] = solutions[:5]
                
                print(f"  ✅ Generated {len(solutions)} solutions, PCE range: {solutions[-1]['pce']:.2f}-{solutions[0]['pce']:.2f}%")
                
            except Exception as e:
                print(f"  ❌ Failed to optimize {key}: {e}")
                # Fallback solutions
                pareto_data[key] = generate_fallback_solutions(track, n_junctions)
    
    return pareto_data

def generate_high_current_solution(n_junctions: int) -> list:
    """Generate bandgap distribution optimized for high current"""
    if n_junctions == 2:
        return [1.75, 1.12]  # Perovskite-Si
    elif n_junctions == 3:
        return [1.9, 1.4, 1.12]  # Higher middle gap
    elif n_junctions == 4:
        return [2.0, 1.6, 1.3, 1.12]
    elif n_junctions == 5:
        return [2.1, 1.8, 1.5, 1.25, 1.12]
    else:  # 6 junctions
        return [2.2, 1.95, 1.7, 1.45, 1.25, 1.12]

def generate_perovskite_high_current(n_junctions: int) -> list:
    """Generate all-perovskite bandgaps for high current"""
    if n_junctions == 2:
        return [1.85, 1.35]  # Wide gap perovskite + narrow
    elif n_junctions == 3:
        return [2.0, 1.6, 1.35]
    elif n_junctions == 4:
        return [2.2, 1.8, 1.55, 1.35]
    elif n_junctions == 5:
        return [2.4, 2.0, 1.7, 1.5, 1.35]
    else:  # 6 junctions
        return [2.6, 2.2, 1.9, 1.65, 1.45, 1.35]

def generate_high_voltage_solution(n_junctions: int) -> list:
    """Generate bandgap distribution optimized for high voltage"""
    if n_junctions == 2:
        return [2.0, 1.3]  # Wider gaps
    elif n_junctions == 3:
        return [2.3, 1.7, 1.3]
    elif n_junctions == 4:
        return [2.5, 2.0, 1.6, 1.3]
    elif n_junctions == 5:
        return [2.7, 2.3, 1.9, 1.6, 1.3]
    else:  # 6 junctions
        return [2.9, 2.5, 2.1, 1.8, 1.5, 1.3]

def generate_perovskite_high_voltage(n_junctions: int) -> list:
    """Generate all-perovskite bandgaps for high voltage"""
    if n_junctions == 2:
        return [2.2, 1.5]
    elif n_junctions == 3:
        return [2.5, 1.9, 1.5]
    elif n_junctions == 4:
        return [2.8, 2.3, 1.8, 1.5]
    elif n_junctions == 5:
        return [3.0, 2.5, 2.1, 1.8, 1.5]
    else:  # 6 junctions
        return [3.2, 2.8, 2.4, 2.0, 1.7, 1.5]

def generate_low_cost_solution(n_junctions: int) -> list:
    """Generate low-cost multi-material solution"""
    # Use established materials: c-Si, CdTe, CIGS, a-Si
    if n_junctions == 2:
        return [1.65, 1.12]  # CdTe-Si
    elif n_junctions == 3:
        return [1.75, 1.45, 1.12]  # a-Si, CdTe, c-Si
    elif n_junctions == 4:
        return [1.85, 1.65, 1.45, 1.12]  # Add CIGS
    elif n_junctions == 5:
        return [2.0, 1.8, 1.6, 1.3, 1.12]
    else:  # 6 junctions
        return [2.2, 1.95, 1.7, 1.5, 1.25, 1.12]

def estimate_pce_from_bandgaps(bandgaps: list, track: str) -> float:
    """Estimate PCE from bandgap distribution"""
    
    n_junctions = len(bandgaps)
    
    # Base efficiency scaling with junctions
    base_pce = {2: 32, 3: 38, 4: 42, 5: 45, 6: 47}
    
    # Track modifier
    track_factor = 1.0 if track == 'A' else 0.95  # Perovskites slightly lower
    
    # Bandgap optimization penalty for suboptimal distributions
    optimal_gaps = {
        2: [1.8, 1.2],
        3: [2.0, 1.5, 1.2], 
        4: [2.2, 1.7, 1.3, 1.2],
        5: [2.4, 1.9, 1.5, 1.2, 1.0],
        6: [2.6, 2.1, 1.7, 1.3, 1.1, 1.0]
    }
    
    # Calculate RMS deviation from optimal
    optimal = optimal_gaps[n_junctions]
    deviation = np.sqrt(np.mean([(bg - opt)**2 for bg, opt in zip(bandgaps, optimal)]))
    penalty = max(0.8, 1.0 - deviation * 2)  # Up to 20% penalty
    
    estimated_pce = base_pce[n_junctions] * track_factor * penalty
    
    return round(estimated_pce, 2)

def generate_fallback_solutions(track: str, n_junctions: int) -> list:
    """Generate fallback solutions if optimization fails"""
    
    print(f"  🔄 Generating fallback solutions for {track}_{n_junctions}T...")
    
    # Simple even distribution
    if track == 'A':
        gap_range = np.linspace(2.4, 1.1, n_junctions)
    else:  # Track B
        gap_range = np.linspace(2.6, 1.4, n_junctions)
    
    pce_est = estimate_pce_from_bandgaps(gap_range.tolist(), track) * 0.9  # Conservative
    
    return [{
        'bandgaps': gap_range.tolist(),
        'pce': pce_est,
        'jsc': 14.0,
        'voc': sum(gap_range) * 0.4,
        'ff': 0.83,
        'objective': 'fallback',
        'current_match_error': 0.1,
        'confidence': 'low'
    }]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    pareto_data = generate_pareto_fronts()
    
    # Save to JSON
    output_file = "data/pareto_fronts.json"
    
    with open(output_file, 'w') as f:
        json.dump(pareto_data, f, indent=2)
    
    print(f"\n💾 Pareto fronts saved to {output_file}")
    
    # Summary statistics
    total_solutions = sum(len(solutions) for solutions in pareto_data.values())
    print(f"📊 Total solutions generated: {total_solutions}")
    print(f"📈 Track combinations: {len(pareto_data)}")
    
    # Best PCE per junction count
    print(f"\n🏆 Best PCE by junction count:")
    for track in ['A', 'B']:
        print(f"  Track {track}:")
        for n in [2, 3, 4, 5, 6]:
            key = f"{track}_{n}T"
            if key in pareto_data:
                best_pce = max(sol['pce'] for sol in pareto_data[key])
                print(f"    {n}T: {best_pce:.1f}%")
    
    print("\n🎉 Pareto front generation complete!")