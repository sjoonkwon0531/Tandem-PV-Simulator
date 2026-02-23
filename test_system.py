#!/usr/bin/env python3
"""Quick system functionality test"""

from engines.economics import EconomicsEngine
from engines.thermal_model import analyze_thermal_performance
from engines.stability import StabilityPredictor

print('=== System Functional Test ===')

# Test economics engine
economics = EconomicsEngine()
stack = [('glass', 3000000), ('ITO', 100), ('perovskite', 500), ('Au', 80)]
cost = economics.calculate_stack_manufacturing_cost(stack)
print(f'✓ Economics: Manufacturing cost ${cost["cost_per_m2"]:.1f}/m²')

# Test thermal engine
materials = ['MAPbI3', 'c-Si']
thicknesses = [500e-9, 200e-6]  # 500 nm perovskite, 200 μm silicon
result = analyze_thermal_performance(materials, thicknesses)
print(f'✓ Thermal: T80 {result["lifetime_prediction"].t80_thermal:.1f} years, max stress {result["thermal_stress"].total_stress/1e6:.1f} MPa')

# Test stability engine  
stability = StabilityPredictor()
materials_list = ['MAPbI3', 'c-Si']
thicknesses_list = [500e-9, 200e-6]

# Import EnvironmentalConditions to create proper object
from engines.stability import EnvironmentalConditions
env_conditions = EnvironmentalConditions(
    temperature=298.15,  # 25°C in Kelvin
    relative_humidity=60.0,  # 60% RH
    light_intensity=1000.0,  # 1 sun
    oxygen_partial_pressure=21000.0,  # Air level
    uv_fraction=0.05,  # 5% UV
    encapsulation_quality=0.8  # Good encapsulation
)

result = stability.predict_long_term_stability(materials_list, thicknesses_list, env_conditions)
print(f'✓ Stability: T80 {result.t80_years:.1f} years, dominant mode {result.dominant_mechanism}')

print('✓ All core engines functional!')