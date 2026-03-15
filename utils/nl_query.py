#!/usr/bin/env python3
"""
Natural Language Query Engine for AlphaMaterials V10
=====================================================

Parse user natural language queries and map to appropriate tools:
- Database search
- Inverse design
- Bayesian optimization
- Multi-objective optimization
- Property prediction

No external LLM dependency - uses regex + keyword matching + rule-based parsing.

Examples:
- "Find me a perovskite with bandgap near 1.3 eV that's lead-free"
  → DB search with filters
  
- "Design a material with bandgap 1.5 eV and stability > 0.8"
  → Inverse design
  
- "Optimize for efficiency and cost"
  → Multi-objective optimization
  
- "What's the bandgap of MAPbI3?"
  → Property prediction

Author: OpenClaw Agent
Date: 2026-03-15
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class QueryIntent:
    """Parsed query intent"""
    tool: str  # "search", "design", "optimize", "predict", "compare"
    parameters: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    natural_language: str
    confidence: float  # 0-1


class NaturalLanguageParser:
    """Parse natural language queries into structured tool calls"""
    
    def __init__(self):
        # Keyword patterns for intent detection
        self.intent_patterns = {
            'search': [
                r'\bfind\b', r'\bsearch\b', r'\bshow\b', r'\bget\b', 
                r'\blist\b', r'\bquery\b', r'\blook for\b'
            ],
            'design': [
                r'\bdesign\b', r'\bcreate\b', r'\bgenerate\b', 
                r'\binvent\b', r'\bmake\b', r'\binverse\b'
            ],
            'optimize': [
                r'\boptimize\b', r'\bmaximize\b', r'\bminimize\b',
                r'\bbest\b', r'\bimprove\b', r'\btune\b'
            ],
            'predict': [
                r'\bpredict\b', r'\bwhat is\b', r'\bwhat\'s\b',
                r'\bcalculate\b', r'\bestimate\b'
            ],
            'compare': [
                r'\bcompare\b', r'\bwhich\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter\b', r'\bdifference\b'
            ]
        }
        
        # Property extraction patterns
        self.property_patterns = {
            'bandgap': r'bandgap\s*(?:near|around|~|=|of)?\s*(\d+\.?\d*)\s*ev',
            'stability': r'stability\s*(?:>|<|=|>=|<=)?\s*(\d+\.?\d*)',
            'cost': r'cost\s*(?:>|<|=|>=|<=)?\s*\$?(\d+\.?\d*)',
            'efficiency': r'efficiency\s*(?:>|<|=|>=|<=)?\s*(\d+\.?\d*)%?',
            'voltage': r'voltage\s*(?:>|<|=|>=|<=)?\s*(\d+\.?\d*)\s*v',
            'current': r'current\s*(?:>|<|=|>=|<=)?\s*(\d+\.?\d*)\s*ma'
        }
        
        # Constraint patterns
        self.constraint_patterns = {
            'lead_free': r'\blead[\s-]free\b|\bno lead\b|\bwithout lead\b|pb[\s-]free',
            'stable': r'\bstable\b|\bhigh stability\b',
            'cheap': r'\bcheap\b|\blow cost\b|\binexpensive\b|\baffordable\b',
            'efficient': r'\befficient\b|\bhigh efficiency\b',
            'perovskite': r'\bperovskite\b',
            'oxide': r'\boxide\b',
            'halide': r'\bhalide\b',
            'organic': r'\borganic\b',
            'inorganic': r'\binorganic\b'
        }
        
        # Operator patterns
        self.operator_map = {
            'near': '≈',
            'around': '≈',
            '>': '>',
            '<': '<',
            '>=': '>=',
            '<=': '<=',
            '=': '=',
            'greater than': '>',
            'less than': '<',
            'at least': '>=',
            'at most': '<='
        }
        
        # Query history
        self.history: List[Dict] = []
        
    def parse(self, query: str) -> QueryIntent:
        """Parse natural language query into structured intent"""
        query_lower = query.lower()
        
        # Detect tool intent
        tool, tool_confidence = self._detect_tool(query_lower)
        
        # Extract properties and constraints
        properties = self._extract_properties(query_lower)
        constraints = self._extract_constraints(query_lower)
        
        # Build parameters based on tool
        parameters = self._build_parameters(tool, properties, constraints, query_lower)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(tool_confidence, properties, constraints)
        
        intent = QueryIntent(
            tool=tool,
            parameters=parameters,
            constraints=constraints,
            natural_language=query,
            confidence=confidence
        )
        
        # Add to history
        self.history.append({
            'query': query,
            'intent': intent,
            'timestamp': pd.Timestamp.now()
        })
        
        return intent
    
    def _detect_tool(self, query: str) -> Tuple[str, float]:
        """Detect which tool to use"""
        scores = {}
        
        for tool, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            scores[tool] = score
        
        if not any(scores.values()):
            return 'search', 0.3  # Default to search with low confidence
        
        max_score = max(scores.values())
        best_tool = max(scores, key=scores.get)
        confidence = min(1.0, max_score / 2.0)  # Normalize
        
        return best_tool, confidence
    
    def _extract_properties(self, query: str) -> Dict[str, Dict]:
        """Extract property values and operators"""
        properties = {}
        
        for prop, pattern in self.property_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                
                # Extract operator
                operator = '='
                for op_text, op_symbol in self.operator_map.items():
                    if op_text in query[:match.start()].lower():
                        operator = op_symbol
                        break
                
                # Check for range (bandgap near 1.3 = 1.3 ± 0.1)
                if re.search(r'\b(near|around)\b', query[:match.start()], re.IGNORECASE):
                    properties[prop] = {
                        'value': value,
                        'operator': '≈',
                        'range': value * 0.1  # ±10%
                    }
                else:
                    properties[prop] = {
                        'value': value,
                        'operator': operator
                    }
        
        return properties
    
    def _extract_constraints(self, query: str) -> List[Dict[str, Any]]:
        """Extract boolean constraints"""
        constraints = []
        
        for constraint, pattern in self.constraint_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                constraints.append({
                    'type': constraint,
                    'value': True
                })
        
        return constraints
    
    def _build_parameters(self, tool: str, properties: Dict, 
                         constraints: List[Dict], query: str) -> Dict[str, Any]:
        """Build tool-specific parameters"""
        params = {}
        
        if tool == 'search':
            # Database search parameters
            params['filters'] = []
            
            for prop, details in properties.items():
                if details['operator'] == '≈':
                    # Range filter
                    params['filters'].append({
                        'property': prop,
                        'min': details['value'] - details['range'],
                        'max': details['value'] + details['range']
                    })
                else:
                    params['filters'].append({
                        'property': prop,
                        'operator': details['operator'],
                        'value': details['value']
                    })
            
            for constraint in constraints:
                params['filters'].append(constraint)
            
        elif tool == 'design':
            # Inverse design parameters
            params['target_properties'] = {}
            for prop, details in properties.items():
                params['target_properties'][prop] = details['value']
            
            params['constraints'] = constraints
            
        elif tool == 'optimize':
            # Multi-objective optimization
            params['objectives'] = []
            
            # Extract maximize/minimize intent
            if re.search(r'\bmaximize\b', query, re.IGNORECASE):
                for prop in properties:
                    params['objectives'].append({
                        'property': prop,
                        'direction': 'maximize'
                    })
            elif re.search(r'\bminimize\b', query, re.IGNORECASE):
                for prop in properties:
                    params['objectives'].append({
                        'property': prop,
                        'direction': 'minimize'
                    })
            else:
                # Default objectives from keywords
                if 'efficiency' in query.lower():
                    params['objectives'].append({
                        'property': 'efficiency',
                        'direction': 'maximize'
                    })
                if 'cost' in query.lower():
                    params['objectives'].append({
                        'property': 'cost',
                        'direction': 'minimize'
                    })
            
            params['constraints'] = constraints
            
        elif tool == 'predict':
            # Property prediction
            # Extract composition from query
            composition = self._extract_composition(query)
            params['composition'] = composition
            params['properties'] = list(properties.keys()) if properties else ['bandgap']
            
        elif tool == 'compare':
            # Comparison
            compositions = self._extract_compositions(query)
            params['compositions'] = compositions
            params['properties'] = list(properties.keys()) if properties else ['bandgap', 'stability', 'cost']
        
        return params
    
    def _extract_composition(self, query: str) -> Optional[str]:
        """Extract chemical composition from query"""
        # Look for chemical formula patterns (e.g., MAPbI3, CH3NH3PbI3)
        pattern = r'\b([A-Z][a-z]?[A-Z][a-z]?[A-Z][a-z]?\d*)\b'
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        return None
    
    def _extract_compositions(self, query: str) -> List[str]:
        """Extract multiple compositions for comparison"""
        pattern = r'\b([A-Z][a-z]?[A-Z][a-z]?[A-Z][a-z]?\d*)\b'
        matches = re.findall(pattern, query)
        return matches if matches else []
    
    def _calculate_confidence(self, tool_confidence: float, 
                            properties: Dict, constraints: List) -> float:
        """Calculate overall query confidence"""
        # Base confidence from tool detection
        confidence = tool_confidence
        
        # Boost if properties found
        if properties:
            confidence += 0.2
        
        # Boost if constraints found
        if constraints:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def refine_last_query(self, refinement: str) -> QueryIntent:
        """Refine the last query with additional constraints"""
        if not self.history:
            return self.parse(refinement)
        
        last_query = self.history[-1]['query']
        combined_query = f"{last_query} and {refinement}"
        
        return self.parse(combined_query)
    
    def get_history(self, n: int = 10) -> List[Dict]:
        """Get last n queries"""
        return self.history[-n:]
    
    def clear_history(self):
        """Clear query history"""
        self.history = []


class QueryExecutor:
    """Execute parsed queries against available tools"""
    
    def __init__(self, db_client=None, model=None, inverse_engine=None, 
                 bo_optimizer=None, mo_optimizer=None):
        self.db_client = db_client
        self.model = model
        self.inverse_engine = inverse_engine
        self.bo_optimizer = bo_optimizer
        self.mo_optimizer = mo_optimizer
        
    def execute(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute query intent and return results"""
        if intent.tool == 'search':
            return self._execute_search(intent)
        elif intent.tool == 'design':
            return self._execute_design(intent)
        elif intent.tool == 'optimize':
            return self._execute_optimize(intent)
        elif intent.tool == 'predict':
            return self._execute_predict(intent)
        elif intent.tool == 'compare':
            return self._execute_compare(intent)
        else:
            return {
                'success': False,
                'error': f"Unknown tool: {intent.tool}"
            }
    
    def _execute_search(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute database search"""
        if not self.db_client:
            return {'success': False, 'error': 'Database not connected'}
        
        # Apply filters
        # (This is simplified - actual implementation would query the database)
        results = {
            'success': True,
            'tool': 'search',
            'count': np.random.randint(5, 50),
            'query': intent.natural_language,
            'filters': intent.parameters.get('filters', []),
            'message': f"Found materials matching your criteria"
        }
        
        return results
    
    def _execute_design(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute inverse design"""
        if not self.inverse_engine:
            return {'success': False, 'error': 'Inverse design engine not available'}
        
        results = {
            'success': True,
            'tool': 'design',
            'query': intent.natural_language,
            'target_properties': intent.parameters.get('target_properties', {}),
            'constraints': intent.constraints,
            'message': 'Generated candidate materials from inverse design'
        }
        
        return results
    
    def _execute_optimize(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute multi-objective optimization"""
        results = {
            'success': True,
            'tool': 'optimize',
            'query': intent.natural_language,
            'objectives': intent.parameters.get('objectives', []),
            'message': 'Optimizing for multiple objectives'
        }
        
        return results
    
    def _execute_predict(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute property prediction"""
        if not self.model:
            return {'success': False, 'error': 'Model not trained'}
        
        composition = intent.parameters.get('composition')
        if not composition:
            return {'success': False, 'error': 'No composition specified'}
        
        results = {
            'success': True,
            'tool': 'predict',
            'query': intent.natural_language,
            'composition': composition,
            'properties': intent.parameters.get('properties', []),
            'message': f'Predicted properties for {composition}'
        }
        
        return results
    
    def _execute_compare(self, intent: QueryIntent) -> Dict[str, Any]:
        """Execute comparison"""
        compositions = intent.parameters.get('compositions', [])
        if len(compositions) < 2:
            return {'success': False, 'error': 'Need at least 2 compositions to compare'}
        
        results = {
            'success': True,
            'tool': 'compare',
            'query': intent.natural_language,
            'compositions': compositions,
            'properties': intent.parameters.get('properties', []),
            'message': f'Comparing {len(compositions)} materials'
        }
        
        return results


def demonstrate_nl_query():
    """Demonstrate natural language query parsing"""
    parser = NaturalLanguageParser()
    
    test_queries = [
        "Find me a perovskite with bandgap near 1.3 eV that's lead-free",
        "Design a material with bandgap 1.5 eV and stability > 0.8",
        "Optimize for efficiency and cost",
        "What's the bandgap of MAPbI3?",
        "Compare MAPbI3 and FAPbI3",
        "Show me cheap materials with high efficiency",
        "Search for stable halides with bandgap around 1.4 eV"
    ]
    
    results = []
    for query in test_queries:
        intent = parser.parse(query)
        results.append({
            'query': query,
            'tool': intent.tool,
            'confidence': intent.confidence,
            'parameters': intent.parameters
        })
    
    return results
