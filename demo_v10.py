#!/usr/bin/env python3
"""
V10 Feature Demonstration Script
================================

Quick demo of all 5 new V10 features without running full Streamlit app.

Usage:
    python3 demo_v10.py

Author: OpenClaw Agent
Date: 2026-03-15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

import pandas as pd
from datetime import datetime

print("="*80)
print("AlphaMaterials V10 - Feature Demonstration")
print("="*80)
print()

# ============================================================================
# 1. Natural Language Query Engine
# ============================================================================

print("1️⃣  NATURAL LANGUAGE QUERY ENGINE")
print("-" * 80)

from nl_query import NaturalLanguageParser

parser = NaturalLanguageParser()

test_queries = [
    "Find me a perovskite with bandgap near 1.3 eV that's lead-free",
    "Design a material with bandgap 1.5 eV and stability > 0.8",
    "Optimize for efficiency and cost",
    "What's the bandgap of MAPbI3?",
    "Compare MAPbI3 and FAPbI3"
]

print("Parsing queries:\n")

for query in test_queries:
    intent = parser.parse(query)
    print(f"Query: \"{query}\"")
    print(f"  → Tool: {intent.tool}")
    print(f"  → Confidence: {intent.confidence:.1%}")
    print(f"  → Parameters: {intent.parameters}")
    print()

print("✅ Natural Language parsing successful!")
print()

# ============================================================================
# 2. Research Report Generator
# ============================================================================

print("2️⃣  RESEARCH REPORT GENERATOR")
print("-" * 80)

from report_generator import ResearchReportGenerator

campaign_data = {
    'discovery_method': 'Bayesian Optimization',
    'n_iterations': 50,
    'session_info': {
        'session_id': 'BO_2026_03_15_demo',
        'timestamp': datetime.now()
    },
    'best_candidate': {
        'composition': 'Cs0.1FA0.9PbI2.8Br0.2',
        'bandgap': 1.35,
        'stability': 0.85,
        'efficiency': 22.3,
        'cost': 0.45
    },
    'candidates': pd.DataFrame({
        'composition': ['Cs0.1FA0.9PbI3', 'MAPbI3', 'FAPbI3'],
        'bandgap': [1.35, 1.55, 1.48],
        'stability': [0.85, 0.72, 0.78],
        'efficiency': [22.3, 20.1, 21.5]
    })
}

generator = ResearchReportGenerator(template="journal_paper")
report = generator.generate_report(campaign_data, include_figures=True, include_tables=True)

print("Generated journal paper report:")
print()
print(report[:800])
print("\n[... truncated ...]")
print()
print(f"✅ Report generated! Total length: {len(report)} characters")
print()

# ============================================================================
# 3. Synthesis Protocol Generator
# ============================================================================

print("3️⃣  SYNTHESIS PROTOCOL GENERATOR")
print("-" * 80)

from protocol_generator import ProtocolGenerator

generator = ProtocolGenerator()
protocol = generator.generate_protocol("MAPbI3")

print(f"Generated synthesis protocol for {protocol.composition}:")
print(f"  → Title: {protocol.title}")
print(f"  → Total Time: {protocol.total_time}")
print(f"  → Total Cost: ${protocol.total_cost}")
print(f"  → Number of Steps: {len(protocol.steps)}")
print()
print("Steps:")
for step in protocol.steps:
    critical = "⭐" if step.critical else "  "
    print(f"  {critical} Step {step.step_number}: {step.action} ({step.duration})")
print()
print(f"Safety Warnings: {len(protocol.safety_warnings)} warnings issued")
print(f"Equipment Required: {len(protocol.equipment_list)} items")
print()
print("✅ Protocol generated successfully!")
print()

# ============================================================================
# 4. Knowledge Graph Visualization
# ============================================================================

print("4️⃣  KNOWLEDGE GRAPH VISUALIZATION")
print("-" * 80)

from knowledge_graph import KnowledgeGraph, build_graph_from_session

session_data = {
    'candidates': pd.DataFrame({
        'composition': ['MAPbI3', 'FAPbI3', 'CsPbI3', 'Cs0.1FA0.9PbI3'],
        'bandgap': [1.55, 1.48, 1.73, 1.50],
        'stability': [0.65, 0.72, 0.85, 0.88],
        'efficiency': [20.1, 21.5, 18.3, 22.8]
    }),
    'synthesis_methods': {
        'MAPbI3': 'one_step_spin_coating',
        'FAPbI3': 'one_step_spin_coating',
        'CsPbI3': 'one_step_spin_coating',
        'Cs0.1FA0.9PbI3': 'one_step_spin_coating'
    },
    'optimization_history': [
        {'composition': 'MAPbI3', 'iteration': 0, 'score': 0.65, 'bandgap': 1.55},
        {'composition': 'FAPbI3', 'iteration': 1, 'score': 0.72, 'bandgap': 1.48},
        {'composition': 'Cs0.1FA0.9PbI3', 'iteration': 2, 'score': 0.88, 'bandgap': 1.50}
    ]
}

kg = build_graph_from_session(session_data)
stats = kg.get_summary_statistics()

print("Knowledge Graph Statistics:")
print(f"  → Total Nodes: {stats['num_nodes']}")
print(f"  → Total Edges: {stats['num_edges']}")
print(f"  → Compositions: {stats['num_compositions']}")
print(f"  → Properties: {stats['num_properties']}")
print(f"  → Processes: {stats['num_processes']}")
print(f"  → Applications: {stats['num_applications']}")
print(f"  → Graph Density: {stats['density']:.3f}")
print(f"  → Avg Degree: {stats['avg_degree']:.1f}")
print()

# Find path
path = kg.find_path('comp_MAPbI3', 'comp_Cs0.1FA0.9PbI3')
if path:
    path_labels = [kg.nodes[nid].label for nid in path]
    print(f"Discovery Path: {' → '.join(path_labels)}")
else:
    print("No path found")

print()
print("✅ Knowledge graph built successfully!")
print()

# ============================================================================
# 5. Decision Matrix (TOPSIS)
# ============================================================================

print("5️⃣  DECISION MATRIX (TOPSIS)")
print("-" * 80)

from decision_matrix import DecisionMatrix, Criterion, Alternative

# Define criteria
criteria = [
    Criterion(name='bandgap', weight=0.30, direction='maximize', ideal_value=1.35),
    Criterion(name='stability', weight=0.35, direction='maximize'),
    Criterion(name='efficiency', weight=0.25, direction='maximize'),
    Criterion(name='cost', weight=0.10, direction='minimize')
]

# Define alternatives
alternatives = [
    Alternative(
        id='A1', name='MAPbI3',
        properties={'bandgap': 1.55, 'stability': 0.65, 'efficiency': 20.1, 'cost': 0.45}
    ),
    Alternative(
        id='A2', name='FAPbI3',
        properties={'bandgap': 1.48, 'stability': 0.72, 'efficiency': 21.5, 'cost': 0.48}
    ),
    Alternative(
        id='A3', name='CsPbI3',
        properties={'bandgap': 1.73, 'stability': 0.85, 'efficiency': 18.3, 'cost': 0.52}
    ),
    Alternative(
        id='A4', name='Cs0.1FA0.9PbI3',
        properties={'bandgap': 1.50, 'stability': 0.88, 'efficiency': 22.8, 'cost': 0.46}
    ),
    Alternative(
        id='A5', name='Cs0.1FA0.9PbI2.8Br0.2',
        properties={'bandgap': 1.35, 'stability': 0.85, 'efficiency': 22.3, 'cost': 0.47}
    )
]

# Create decision matrix
dm = DecisionMatrix(criteria, alternatives)

# Compute TOPSIS
results = dm.compute_topsis()

print("TOPSIS Analysis Results:")
print()
print(results.to_string(index=False))
print()

# Get top candidate
top_candidate = results.iloc[0]
print(f"🥇 Recommended Candidate: {top_candidate['Alternative']}")
print(f"   TOPSIS Score: {top_candidate['TOPSIS_Score']:.4f}")
print()

print("✅ TOPSIS analysis complete!")
print()

# ============================================================================
# Summary
# ============================================================================

print("="*80)
print("DEMONSTRATION COMPLETE!")
print("="*80)
print()
print("V10 Features Demonstrated:")
print("  ✅ 1. Natural Language Query Engine - Parse queries into tool calls")
print("  ✅ 2. Research Report Generator - Auto-generate publication drafts")
print("  ✅ 3. Synthesis Protocol Generator - Step-by-step lab procedures")
print("  ✅ 4. Knowledge Graph Visualization - Map composition-property relationships")
print("  ✅ 5. Decision Matrix (TOPSIS) - Systematic candidate ranking")
print()
print("All V10 modules working correctly!")
print()
print("To run the full Streamlit app:")
print("  streamlit run app_v10.py")
print()
print("="*80)
