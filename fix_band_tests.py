#!/usr/bin/env python3
"""Fix band alignment test API mismatches"""

import re

# Read the test file
with open('tests/test_band.py', 'r') as f:
    content = f.read()

# Fix all occurrences of optimize_bandgaps to optimize_n_junction
content = content.replace('optimizer.optimize_bandgaps(', 'optimizer.optimize_n_junction(')

# Fix all result access patterns for BandgapSolution objects
content = re.sub(r"result\['optimal_bandgaps'\]", "result.bandgaps", content)
content = re.sub(r"result\['optimal_pce'\]", "result.efficiency", content)

# Fix references to two_j_result and similar
content = re.sub(r"two_j_result\['optimal_pce'\]", "two_j_result.efficiency", content)

# Fix any remaining calculate_detailed_balance references
content = content.replace('calculator.calculate_detailed_balance(', 'calculator.calculate_single_junction(')

# Update assertions for the new API
content = re.sub(r"assert result\['optimal_pce'\] > ([0-9.]+)", r"assert result.efficiency > (\1/100.0)", content)

# Fix the material database integration test
content = re.sub(r"result = optimizer.optimize_bandgaps\(2\)", r"result = optimizer.optimize_n_junction(2)", content)

print("Fixed patterns applied. Content length:", len(content))

# Write the fixed file
with open('tests/test_band.py', 'w') as f:
    f.write(content)

print("✅ Band alignment tests fixed!")