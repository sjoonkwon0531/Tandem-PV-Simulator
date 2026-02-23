#!/usr/bin/env python3
"""
N-Junction Tandem PV Simulator - Engine Package
=============================================

Core physics engines for tandem PV simulation:
- optical_tmm: Transfer Matrix Method for optical modeling
- band_alignment: Optimal bandgap distribution calculations  
- interface_loss: Recombination and tunneling junction models
- thermal_model: CTE mismatch and thermal stress analysis
- stability: Degradation mechanisms and lifetime prediction
- economics: Manufacturing cost and LCOE optimization

Each engine is standalone but shares the material database and configuration
from the parent config module.
"""

from typing import List

__version__ = "1.0.0"
__all__ = [
    "optical_tmm",
    "band_alignment", 
    "interface_loss",
    "thermal_model",
    "stability",
    "economics"
]

# Engine availability check
AVAILABLE_ENGINES = []

try:
    from . import optical_tmm
    AVAILABLE_ENGINES.append("optical_tmm")
except ImportError:
    pass

try:
    from . import band_alignment
    AVAILABLE_ENGINES.append("band_alignment")
except ImportError:
    pass

try:
    from . import interface_loss
    AVAILABLE_ENGINES.append("interface_loss")
except ImportError:
    pass

try:
    from . import thermal_model
    AVAILABLE_ENGINES.append("thermal_model")
except ImportError:
    pass

try:
    from . import stability
    AVAILABLE_ENGINES.append("stability")
except ImportError:
    pass

try:
    from . import economics
    AVAILABLE_ENGINES.append("economics")
except ImportError:
    pass

def get_available_engines() -> List[str]:
    """Return list of successfully imported engines"""
    return AVAILABLE_ENGINES.copy()

def engine_status() -> dict:
    """Return detailed status of all engines"""
    expected = ["optical_tmm", "band_alignment", "interface_loss", 
                "thermal_model", "stability", "economics"]
    
    status = {}
    for engine in expected:
        status[engine] = engine in AVAILABLE_ENGINES
    
    return status