#!/usr/bin/env python3
"""
Knowledge Graph Visualization for AlphaMaterials V10
=====================================================

Map relationships between:
- Compositions ↔ Properties ↔ Processes ↔ Applications
- Interactive network graph (plotly)
- Discovery path tracking
- Auto-generated from user exploration history

Author: OpenClaw Agent
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Node:
    """Knowledge graph node"""
    id: str
    type: str  # "composition", "property", "process", "application"
    label: str
    properties: Dict
    timestamp: datetime = None


@dataclass
class Edge:
    """Knowledge graph edge"""
    source: str
    target: str
    relationship: str  # "has_property", "requires_process", "enables_application"
    weight: float = 1.0
    properties: Dict = None


class KnowledgeGraph:
    """Build and visualize knowledge graph for materials discovery"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.graph = nx.Graph()
        
        # Relationship types
        self.relationship_types = {
            'has_property': {'color': '#3498db', 'description': 'Exhibits property'},
            'requires_process': {'color': '#e74c3c', 'description': 'Synthesized via'},
            'enables_application': {'color': '#2ecc71', 'description': 'Used for'},
            'similar_to': {'color': '#f39c12', 'description': 'Similar composition'},
            'derived_from': {'color': '#9b59b6', 'description': 'Optimized from'},
            'competes_with': {'color': '#e67e22', 'description': 'Alternative to'}
        }
        
        # Node type colors
        self.node_colors = {
            'composition': '#3498db',
            'property': '#2ecc71',
            'process': '#e74c3c',
            'application': '#f39c12',
            'discovery': '#9b59b6'
        }
    
    def add_node(self, node: Node):
        """Add node to graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.properties)
    
    def add_edge(self, edge: Edge):
        """Add edge to graph"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source, 
            edge.target, 
            relationship=edge.relationship,
            weight=edge.weight
        )
    
    def add_composition(self, composition: str, properties: Dict):
        """Add composition node with property links"""
        node_id = f"comp_{composition}"
        
        # Add composition node
        node = Node(
            id=node_id,
            type="composition",
            label=composition,
            properties=properties,
            timestamp=datetime.now()
        )
        self.add_node(node)
        
        # Link to properties
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, (int, float)):
                prop_id = f"prop_{prop_name}"
                
                # Add property node if doesn't exist
                if prop_id not in self.nodes:
                    prop_node = Node(
                        id=prop_id,
                        type="property",
                        label=prop_name.replace('_', ' ').title(),
                        properties={'name': prop_name}
                    )
                    self.add_node(prop_node)
                
                # Link composition to property
                edge = Edge(
                    source=node_id,
                    target=prop_id,
                    relationship="has_property",
                    weight=abs(prop_value),
                    properties={'value': prop_value}
                )
                self.add_edge(edge)
    
    def add_process(self, composition: str, process: str, parameters: Dict):
        """Link composition to synthesis process"""
        comp_id = f"comp_{composition}"
        proc_id = f"proc_{process}"
        
        # Add process node if doesn't exist
        if proc_id not in self.nodes:
            proc_node = Node(
                id=proc_id,
                type="process",
                label=process.replace('_', ' ').title(),
                properties=parameters
            )
            self.add_node(proc_node)
        
        # Link composition to process
        edge = Edge(
            source=comp_id,
            target=proc_id,
            relationship="requires_process",
            properties=parameters
        )
        self.add_edge(edge)
    
    def add_application(self, composition: str, application: str, 
                       performance: Optional[float] = None):
        """Link composition to application"""
        comp_id = f"comp_{composition}"
        app_id = f"app_{application}"
        
        # Add application node if doesn't exist
        if app_id not in self.nodes:
            app_node = Node(
                id=app_id,
                type="application",
                label=application.replace('_', ' ').title(),
                properties={'name': application}
            )
            self.add_node(app_node)
        
        # Link composition to application
        weight = performance if performance else 1.0
        edge = Edge(
            source=comp_id,
            target=app_id,
            relationship="enables_application",
            weight=weight,
            properties={'performance': performance}
        )
        self.add_edge(edge)
    
    def add_similarity_link(self, comp1: str, comp2: str, similarity: float):
        """Link similar compositions"""
        edge = Edge(
            source=f"comp_{comp1}",
            target=f"comp_{comp2}",
            relationship="similar_to",
            weight=similarity,
            properties={'similarity': similarity}
        )
        self.add_edge(edge)
    
    def add_discovery_path(self, path: List[Dict]):
        """
        Add discovery path from optimization history
        
        Args:
            path: List of dicts with 'composition', 'iteration', 'score', etc.
        """
        for i, step in enumerate(path):
            comp = step['composition']
            self.add_composition(comp, step)
            
            # Add discovery node
            disc_id = f"disc_{i}"
            disc_node = Node(
                id=disc_id,
                type="discovery",
                label=f"Iteration {i}",
                properties={'iteration': i, 'score': step.get('score', 0)}
            )
            self.add_node(disc_node)
            
            # Link to composition
            edge = Edge(
                source=disc_id,
                target=f"comp_{comp}",
                relationship="derived_from",
                weight=step.get('score', 1.0)
            )
            self.add_edge(edge)
            
            # Link to previous iteration
            if i > 0:
                prev_disc_id = f"disc_{i-1}"
                edge = Edge(
                    source=prev_disc_id,
                    target=disc_id,
                    relationship="derived_from",
                    weight=1.0
                )
                self.add_edge(edge)
    
    def find_path(self, start: str, end: str) -> List[str]:
        """Find path between two nodes"""
        try:
            path = nx.shortest_path(self.graph, source=start, target=end)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[Node]:
        """Get neighboring nodes"""
        neighbors = []
        
        for edge in self.edges:
            if edge.source == node_id:
                if relationship is None or edge.relationship == relationship:
                    neighbors.append(self.nodes[edge.target])
            elif edge.target == node_id:
                if relationship is None or edge.relationship == relationship:
                    neighbors.append(self.nodes[edge.source])
        
        return neighbors
    
    def get_composition_properties(self, composition: str) -> Dict:
        """Get all properties of a composition"""
        comp_id = f"comp_{composition}"
        properties = {}
        
        for edge in self.edges:
            if edge.source == comp_id and edge.relationship == "has_property":
                prop_node = self.nodes[edge.target]
                properties[prop_node.label] = edge.properties.get('value')
        
        return properties
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute node centrality (importance)"""
        centrality = nx.betweenness_centrality(self.graph)
        return centrality
    
    def compute_communities(self) -> Dict[str, int]:
        """Detect communities (clusters)"""
        communities = nx.community.greedy_modularity_communities(self.graph)
        
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        
        return node_to_community
    
    def visualize_interactive(self, 
                            highlight_path: Optional[List[str]] = None,
                            filter_type: Optional[str] = None) -> go.Figure:
        """
        Create interactive network visualization
        
        Args:
            highlight_path: Optional list of node IDs to highlight
            filter_type: Optional node type to filter ("composition", "property", etc.)
        
        Returns:
            Plotly figure
        """
        # Filter nodes if requested
        if filter_type:
            nodes_to_show = {nid: n for nid, n in self.nodes.items() if n.type == filter_type}
            edges_to_show = [e for e in self.edges if e.source in nodes_to_show and e.target in nodes_to_show]
        else:
            nodes_to_show = self.nodes
            edges_to_show = self.edges
        
        # Create subgraph for layout
        subgraph = nx.Graph()
        for node_id in nodes_to_show:
            subgraph.add_node(node_id)
        for edge in edges_to_show:
            subgraph.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Compute layout
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        
        # Build edge traces
        edge_traces = []
        
        for rel_type, rel_info in self.relationship_types.items():
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in edges_to_show:
                if edge.relationship == rel_type:
                    x0, y0 = pos[edge.source]
                    x1, y1 = pos[edge.target]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_text.append(rel_info['description'])
            
            if edge_x:
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1, color=rel_info['color']),
                    hoverinfo='text',
                    hovertext=edge_text,
                    showlegend=True,
                    name=rel_type.replace('_', ' ').title(),
                    opacity=0.6
                )
                edge_traces.append(edge_trace)
        
        # Build node traces
        node_traces = []
        
        for node_type, color in self.node_colors.items():
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node_id, node in nodes_to_show.items():
                if node.type == node_type:
                    x, y = pos[node_id]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Build hover text
                    hover = f"<b>{node.label}</b><br>"
                    hover += f"Type: {node.type}<br>"
                    for key, value in node.properties.items():
                        if isinstance(value, float):
                            hover += f"{key}: {value:.3f}<br>"
                        else:
                            hover += f"{key}: {value}<br>"
                    node_text.append(hover)
                    
                    # Node size based on degree
                    degree = self.graph.degree[node_id]
                    node_size.append(10 + degree * 2)
            
            if node_x:
                # Check if in highlight path
                marker_line_width = 2
                marker_line_color = 'white'
                
                if highlight_path:
                    # Highlight nodes in path
                    marker_line_width = [4 if nid in highlight_path else 1 
                                        for nid in nodes_to_show if nodes_to_show[nid].type == node_type]
                    marker_line_color = ['yellow' if nid in highlight_path else 'white' 
                                        for nid in nodes_to_show if nodes_to_show[nid].type == node_type]
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=node_size,
                        color=color,
                        line=dict(width=marker_line_width, color=marker_line_color)
                    ),
                    text=[nodes_to_show[nid].label for nid in nodes_to_show if nodes_to_show[nid].type == node_type],
                    textposition="top center",
                    textfont=dict(size=8, color='white'),
                    hoverinfo='text',
                    hovertext=node_text,
                    showlegend=True,
                    name=node_type.title()
                )
                node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Materials Knowledge Graph",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#0a0e1a',
            paper_bgcolor='#0a0e1a',
            font=dict(color='white'),
            height=700
        )
        
        return fig
    
    def export_to_dict(self) -> Dict:
        """Export graph to dictionary"""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'label': node.label,
                    'properties': node.properties
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'relationship': edge.relationship,
                    'weight': edge.weight,
                    'properties': edge.properties
                }
                for edge in self.edges
            ]
        }
    
    def export_to_json(self, filename: str):
        """Export graph to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_dict(cls, data: Dict) -> 'KnowledgeGraph':
        """Load graph from dictionary"""
        kg = cls()
        
        for node_data in data['nodes']:
            node = Node(
                id=node_data['id'],
                type=node_data['type'],
                label=node_data['label'],
                properties=node_data['properties']
            )
            kg.add_node(node)
        
        for edge_data in data['edges']:
            edge = Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                relationship=edge_data['relationship'],
                weight=edge_data['weight'],
                properties=edge_data.get('properties')
            )
            kg.add_edge(edge)
        
        return kg
    
    @classmethod
    def load_from_json(cls, filename: str) -> 'KnowledgeGraph':
        """Load graph from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.load_from_dict(data)
    
    def get_summary_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_compositions': sum(1 for n in self.nodes.values() if n.type == 'composition'),
            'num_properties': sum(1 for n in self.nodes.values() if n.type == 'property'),
            'num_processes': sum(1 for n in self.nodes.values() if n.type == 'process'),
            'num_applications': sum(1 for n in self.nodes.values() if n.type == 'application'),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
        }


def build_graph_from_session(session_data: Dict) -> KnowledgeGraph:
    """
    Build knowledge graph from session data
    
    Args:
        session_data: Dictionary with:
            - candidates: pd.DataFrame with compositions and properties
            - optimization_history: List of optimization steps
            - synthesis_methods: Dict of composition -> synthesis method
    
    Returns:
        KnowledgeGraph
    """
    kg = KnowledgeGraph()
    
    # Add compositions and properties
    candidates = session_data.get('candidates', pd.DataFrame())
    if not candidates.empty:
        for _, row in candidates.iterrows():
            comp = row.get('composition', f"Comp_{_}")
            properties = row.to_dict()
            kg.add_composition(comp, properties)
    
    # Add synthesis methods
    synthesis_methods = session_data.get('synthesis_methods', {})
    for comp, method in synthesis_methods.items():
        kg.add_process(comp, method, {'method': method})
    
    # Add applications (default: tandem solar cell)
    for comp in candidates.get('composition', []):
        kg.add_application(comp, 'tandem_solar_cell', 
                          performance=candidates[candidates['composition'] == comp]['efficiency'].values[0]
                          if 'efficiency' in candidates.columns else None)
    
    # Add discovery path
    opt_history = session_data.get('optimization_history', [])
    if opt_history:
        kg.add_discovery_path(opt_history)
    
    # Add similarity links (based on composition similarity)
    # Simplified: link compositions with similar bandgaps
    if 'bandgap' in candidates.columns:
        comps = candidates['composition'].values
        bandgaps = candidates['bandgap'].values
        
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                bg_diff = abs(bandgaps[i] - bandgaps[j])
                if bg_diff < 0.2:  # Similar bandgap
                    similarity = 1.0 - bg_diff / 0.2
                    kg.add_similarity_link(comps[i], comps[j], similarity)
    
    return kg


def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph creation"""
    kg = KnowledgeGraph()
    
    # Add some sample data
    compositions = {
        'MAPbI3': {'bandgap': 1.55, 'stability': 0.65, 'efficiency': 20.1},
        'FAPbI3': {'bandgap': 1.48, 'stability': 0.72, 'efficiency': 21.5},
        'CsPbI3': {'bandgap': 1.73, 'stability': 0.85, 'efficiency': 18.3},
        'Cs0.1FA0.9PbI3': {'bandgap': 1.50, 'stability': 0.88, 'efficiency': 22.8}
    }
    
    for comp, props in compositions.items():
        kg.add_composition(comp, props)
        kg.add_process(comp, 'one_step_spin_coating', {'temperature': 100, 'time': 10})
        kg.add_application(comp, 'tandem_solar_cell', props['efficiency'])
    
    # Add similarity links
    kg.add_similarity_link('MAPbI3', 'FAPbI3', 0.85)
    kg.add_similarity_link('FAPbI3', 'Cs0.1FA0.9PbI3', 0.92)
    
    # Add discovery path
    path = [
        {'composition': 'MAPbI3', 'iteration': 0, 'score': 0.65},
        {'composition': 'FAPbI3', 'iteration': 1, 'score': 0.72},
        {'composition': 'Cs0.1FA0.9PbI3', 'iteration': 2, 'score': 0.88}
    ]
    kg.add_discovery_path(path)
    
    # Create visualization
    fig = kg.visualize_interactive()
    
    # Get statistics
    stats = kg.get_summary_statistics()
    
    return kg, fig, stats
