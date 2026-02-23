#!/usr/bin/env python3
"""
Tandem PV Optimizer Package
===========================

Multi-variable optimization for N-junction tandem photovoltaic devices.
Integrates all engine modules to find optimal designs that maximize efficiency
or minimize LCOE while satisfying engineering constraints.

Main optimization targets:
- Maximize PCE (power conversion efficiency)
- Minimize LCOE (levelized cost of electricity)
- Maximize performance/cost ratio
- Meet reliability constraints (thermal stress, degradation)

Available optimizers:
- tandem_optimizer: Primary multi-variable optimizer
- genetic_optimizer: Genetic algorithm for global optimization
- gradient_optimizer: Gradient-based local optimization
- constraint_optimizer: Constrained optimization with engineering limits
"""

from typing import List

__version__ = "1.0.0"
__all__ = [
    "tandem_optimizer",
]

# Optimizer availability check
AVAILABLE_OPTIMIZERS = []

try:
    from . import tandem_optimizer
    AVAILABLE_OPTIMIZERS.append("tandem_optimizer")
except ImportError:
    pass

def get_available_optimizers() -> List[str]:
    """Return list of successfully imported optimizers"""
    return AVAILABLE_OPTIMIZERS.copy()

def optimizer_status() -> dict:
    """Return detailed status of all optimizers"""
    expected = ["tandem_optimizer"]
    
    status = {}
    for optimizer in expected:
        status[optimizer] = optimizer in AVAILABLE_OPTIMIZERS
    
    return status

# Optimization targets
OPTIMIZATION_TARGETS = [
    'maximize_efficiency',      # Maximize PCE
    'minimize_lcoe',           # Minimize LCOE
    'maximize_efficiency_cost_ratio',  # Maximize PCE/cost
    'maximize_lifetime',       # Maximize T80 lifetime
    'minimize_thermal_stress', # Minimize thermal mismatch
    'custom'                   # Custom objective function
]

# Constraint types
CONSTRAINT_TYPES = [
    'current_matching',        # Jsc matching between subcells
    'thermal_stress_limit',    # Maximum CTE mismatch stress
    'cost_budget',            # Maximum manufacturing cost
    'stability_requirement',   # Minimum T80 lifetime
    'manufacturability',       # Minimum manufacturability score
    'material_availability',   # Use only available materials
    'thickness_limits',        # Physical thickness constraints
    'bandgap_spacing'         # Minimum bandgap differences
]