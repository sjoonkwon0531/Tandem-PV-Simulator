#!/usr/bin/env python3
"""
Decision Matrix & Multi-Criteria Analysis for AlphaMaterials V10
================================================================

TOPSIS, AHP, and weighted scoring for material selection:
- Side-by-side comparison
- Radar charts, comparison tables
- Weighted scoring with user-defined criteria
- "Which candidate should I synthesize first?" ranking
- Export decision rationale

Author: OpenClaw Agent
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Criterion:
    """Decision criterion"""
    name: str
    weight: float  # 0-1, sum to 1.0
    direction: str  # "maximize" or "minimize"
    ideal_value: Optional[float] = None


@dataclass
class Alternative:
    """Decision alternative (candidate material)"""
    id: str
    name: str
    properties: Dict[str, float]
    scores: Dict[str, float] = None
    final_score: float = 0.0
    rank: int = 0


class DecisionMatrix:
    """Multi-criteria decision analysis"""
    
    def __init__(self, criteria: List[Criterion], alternatives: List[Alternative]):
        """
        Initialize decision matrix
        
        Args:
            criteria: List of Criterion objects
            alternatives: List of Alternative objects
        """
        self.criteria = criteria
        self.alternatives = alternatives
        
        # Validate weights sum to 1.0
        total_weight = sum(c.weight for c in criteria)
        if not np.isclose(total_weight, 1.0):
            # Normalize weights
            for c in criteria:
                c.weight /= total_weight
        
        # Build matrix
        self.matrix = self._build_matrix()
    
    def _build_matrix(self) -> pd.DataFrame:
        """Build decision matrix DataFrame"""
        data = {}
        
        for criterion in self.criteria:
            data[criterion.name] = [
                alt.properties.get(criterion.name, 0.0) 
                for alt in self.alternatives
            ]
        
        df = pd.DataFrame(data, index=[alt.name for alt in self.alternatives])
        return df
    
    def compute_weighted_score(self) -> pd.DataFrame:
        """Simple weighted scoring"""
        # Normalize matrix
        normalized = self.matrix.copy()
        
        for criterion in self.criteria:
            col = criterion.name
            
            if criterion.direction == "maximize":
                # Higher is better: normalize to 0-1
                normalized[col] = (self.matrix[col] - self.matrix[col].min()) / \
                                 (self.matrix[col].max() - self.matrix[col].min() + 1e-10)
            else:
                # Lower is better: inverse normalize
                normalized[col] = (self.matrix[col].max() - self.matrix[col]) / \
                                 (self.matrix[col].max() - self.matrix[col].min() + 1e-10)
        
        # Apply weights
        scores = pd.Series(0.0, index=self.matrix.index)
        
        for criterion in self.criteria:
            scores += normalized[criterion.name] * criterion.weight
        
        # Update alternatives
        for i, alt in enumerate(self.alternatives):
            alt.final_score = scores.iloc[i]
            alt.scores = normalized.iloc[i].to_dict()
        
        # Rank alternatives
        ranked = scores.sort_values(ascending=False)
        for rank, (name, score) in enumerate(ranked.items(), 1):
            alt = next(a for a in self.alternatives if a.name == name)
            alt.rank = rank
        
        results = pd.DataFrame({
            'Alternative': self.matrix.index,
            'Score': scores.values,
            'Rank': [alt.rank for alt in self.alternatives]
        })
        
        return results.sort_values('Rank')
    
    def compute_topsis(self) -> pd.DataFrame:
        """
        TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
        
        Steps:
        1. Normalize decision matrix
        2. Calculate weighted normalized matrix
        3. Determine ideal and negative-ideal solutions
        4. Calculate separation measures
        5. Calculate relative closeness to ideal solution
        6. Rank alternatives
        """
        n_alternatives, n_criteria = self.matrix.shape
        
        # Step 1: Normalize matrix (vector normalization)
        normalized = self.matrix.copy()
        
        for col in self.matrix.columns:
            norm = np.sqrt((self.matrix[col] ** 2).sum())
            normalized[col] = self.matrix[col] / (norm + 1e-10)
        
        # Step 2: Apply weights
        weighted = normalized.copy()
        
        for criterion in self.criteria:
            weighted[criterion.name] *= criterion.weight
        
        # Step 3: Determine ideal and negative-ideal solutions
        ideal_solution = {}
        negative_ideal_solution = {}
        
        for criterion in self.criteria:
            col = criterion.name
            
            if criterion.direction == "maximize":
                ideal_solution[col] = weighted[col].max()
                negative_ideal_solution[col] = weighted[col].min()
            else:
                ideal_solution[col] = weighted[col].min()
                negative_ideal_solution[col] = weighted[col].max()
        
        # Step 4: Calculate separation measures
        ideal_separations = []
        negative_ideal_separations = []
        
        for i in range(n_alternatives):
            # Distance to ideal solution
            d_plus = np.sqrt(sum(
                (weighted.iloc[i][c.name] - ideal_solution[c.name]) ** 2 
                for c in self.criteria
            ))
            ideal_separations.append(d_plus)
            
            # Distance to negative-ideal solution
            d_minus = np.sqrt(sum(
                (weighted.iloc[i][c.name] - negative_ideal_solution[c.name]) ** 2 
                for c in self.criteria
            ))
            negative_ideal_separations.append(d_minus)
        
        # Step 5: Calculate relative closeness
        closeness = [
            d_minus / (d_plus + d_minus + 1e-10)
            for d_plus, d_minus in zip(ideal_separations, negative_ideal_separations)
        ]
        
        # Step 6: Rank alternatives
        results = pd.DataFrame({
            'Alternative': self.matrix.index,
            'TOPSIS_Score': closeness,
            'Ideal_Distance': ideal_separations,
            'Negative_Ideal_Distance': negative_ideal_separations
        })
        
        results = results.sort_values('TOPSIS_Score', ascending=False)
        results['Rank'] = range(1, len(results) + 1)
        
        # Update alternatives
        for i, alt in enumerate(self.alternatives):
            alt.final_score = closeness[i]
            alt.rank = results[results['Alternative'] == alt.name]['Rank'].values[0]
        
        return results
    
    def compute_ahp(self, pairwise_matrices: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        AHP (Analytic Hierarchy Process)
        
        Simplified version using weights only (full AHP requires pairwise comparisons)
        
        Args:
            pairwise_matrices: Optional dict of criterion -> pairwise comparison matrix
                              If None, uses predefined weights
        """
        # For simplicity, use weighted scoring with AHP-style presentation
        # Full AHP would require user to input pairwise comparison matrices
        
        return self.compute_weighted_score()
    
    def visualize_comparison(self, top_n: int = 5) -> go.Figure:
        """
        Create comparison visualization with radar chart and bar chart
        
        Args:
            top_n: Number of top alternatives to visualize
        
        Returns:
            Plotly figure
        """
        # Get top N alternatives
        top_alts = sorted(self.alternatives, key=lambda a: a.rank)[:top_n]
        
        # Create subplots: radar chart + bar chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Multi-Criteria Radar Chart', 'Overall Scores'),
            specs=[[{'type': 'scatterpolar'}, {'type': 'bar'}]]
        )
        
        # Radar chart
        categories = [c.name.replace('_', ' ').title() for c in self.criteria]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, alt in enumerate(top_alts):
            # Get normalized scores
            values = [alt.scores.get(c.name, 0.0) for c in self.criteria]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=alt.name,
                    marker=dict(color=colors[i % len(colors)]),
                    opacity=0.6
                ),
                row=1, col=1
            )
        
        # Bar chart
        names = [alt.name for alt in top_alts]
        scores = [alt.final_score for alt in top_alts]
        ranks = [alt.rank for alt in top_alts]
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=scores,
                text=[f"Rank {r}" for r in ranks],
                textposition='outside',
                marker=dict(
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score", x=1.15)
                )
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor='#0a0e1a',
            paper_bgcolor='#0a0e1a',
            font=dict(color='white'),
            polar=dict(
                bgcolor='#1e2130',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='#34495e'
                )
            )
        )
        
        fig.update_xaxes(title_text="Candidate", row=1, col=2)
        fig.update_yaxes(title_text="Overall Score", row=1, col=2)
        
        return fig
    
    def visualize_criteria_weights(self) -> go.Figure:
        """Visualize criteria weights as pie chart"""
        names = [c.name.replace('_', ' ').title() for c in self.criteria]
        weights = [c.weight for c in self.criteria]
        
        fig = go.Figure(data=[go.Pie(
            labels=names,
            values=weights,
            hole=0.3,
            marker=dict(colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        )])
        
        fig.update_layout(
            title="Criteria Weights",
            plot_bgcolor='#0a0e1a',
            paper_bgcolor='#0a0e1a',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def generate_decision_rationale(self, top_n: int = 3) -> str:
        """
        Generate text rationale for decision
        
        Args:
            top_n: Number of top candidates to include
        
        Returns:
            Markdown-formatted rationale
        """
        top_alts = sorted(self.alternatives, key=lambda a: a.rank)[:top_n]
        
        rationale = f"""# Decision Analysis Report

**Method:** TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)  
**Number of Alternatives:** {len(self.alternatives)}  
**Number of Criteria:** {len(self.criteria)}  
**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

---

## Criteria Weights

"""
        
        for criterion in sorted(self.criteria, key=lambda c: c.weight, reverse=True):
            direction = "↑ Maximize" if criterion.direction == "maximize" else "↓ Minimize"
            rationale += f"- **{criterion.name.replace('_', ' ').title()}:** {criterion.weight:.1%} ({direction})\n"
        
        rationale += "\n---\n\n## Recommendation\n\n"
        
        # Top candidate
        best = top_alts[0]
        rationale += f"### 🥇 Recommended: **{best.name}** (Rank #{best.rank})\n\n"
        rationale += f"**Overall Score:** {best.final_score:.3f}\n\n"
        rationale += "**Strengths:**\n"
        
        # Identify strengths (top 3 criteria scores)
        strengths = sorted(best.scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for criterion, score in strengths:
            rationale += f"- {criterion.replace('_', ' ').title()}: {score:.2f}/1.00 ✅\n"
        
        rationale += "\n**Performance Summary:**\n\n"
        rationale += "| Criterion | Raw Value | Normalized Score |\n"
        rationale += "|-----------|-----------|------------------|\n"
        
        for criterion in self.criteria:
            raw_value = best.properties.get(criterion.name, 0.0)
            norm_score = best.scores.get(criterion.name, 0.0)
            rationale += f"| {criterion.name.replace('_', ' ').title()} | {raw_value:.3f} | {norm_score:.3f} |\n"
        
        rationale += "\n---\n\n## Alternative Candidates\n\n"
        
        for i, alt in enumerate(top_alts[1:], 2):
            rationale += f"### {['🥈', '🥉'][i-2] if i <= 3 else '📍'} Rank #{alt.rank}: **{alt.name}**\n\n"
            rationale += f"**Overall Score:** {alt.final_score:.3f}\n\n"
            
            # Compare to best
            score_diff = best.final_score - alt.final_score
            rationale += f"*Score gap from best: {score_diff:.3f} ({score_diff/best.final_score*100:.1f}%)*\n\n"
            
            # Key differentiators
            rationale += "**Key Differences:**\n"
            diffs = []
            for criterion in self.criteria:
                best_score = best.scores.get(criterion.name, 0.0)
                alt_score = alt.scores.get(criterion.name, 0.0)
                diff = best_score - alt_score
                if abs(diff) > 0.1:
                    direction = "higher" if diff > 0 else "lower"
                    diffs.append((criterion.name, diff, direction))
            
            diffs_sorted = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)[:2]
            for criterion, diff, direction in diffs_sorted:
                rationale += f"- {criterion.replace('_', ' ').title()}: {abs(diff):.2f} {direction} than best\n"
            
            rationale += "\n"
        
        rationale += "---\n\n## Decision Rationale\n\n"
        rationale += f"""Based on the multi-criteria analysis using TOPSIS methodology, **{best.name}** is the 
recommended candidate for synthesis. This decision is based on:

1. **Highest overall score** ({best.final_score:.3f}) when considering all weighted criteria
2. **Balanced performance** across multiple objectives
3. **Closest to ideal solution** in the multi-dimensional criteria space

### Synthesis Priority Ranking:

"""
        
        for i, alt in enumerate(top_alts, 1):
            rationale += f"{i}. **{alt.name}** (Score: {alt.final_score:.3f})\n"
        
        rationale += "\n### Next Steps:\n\n"
        rationale += "1. ✅ **Synthesize top candidate** for experimental validation\n"
        rationale += "2. 🔬 **Characterize properties** (XRD, UV-Vis, device performance)\n"
        rationale += "3. 📊 **Compare experimental vs predicted** to refine model\n"
        rationale += f"4. 🔄 **If top candidate fails**, proceed to Rank #2: {top_alts[1].name}\n"
        
        rationale += "\n---\n\n*This decision analysis provides a systematic, data-driven recommendation. "
        rationale += "Always consider expert judgment and experimental constraints in final decision.*\n"
        
        return rationale
    
    def sensitivity_analysis(self, criterion_name: str, 
                           weight_range: Tuple[float, float] = (0.1, 0.5)) -> pd.DataFrame:
        """
        Perform sensitivity analysis on criterion weight
        
        Args:
            criterion_name: Name of criterion to vary
            weight_range: (min, max) weight range
        
        Returns:
            DataFrame with results for different weights
        """
        results = []
        
        # Save original weight
        original_weights = {c.name: c.weight for c in self.criteria}
        
        # Vary weight
        weights = np.linspace(weight_range[0], weight_range[1], 10)
        
        for w in weights:
            # Update weight
            for c in self.criteria:
                if c.name == criterion_name:
                    c.weight = w
                else:
                    # Redistribute remaining weight proportionally
                    remaining = 1.0 - w
                    original_sum = sum(original_weights[c2.name] for c2 in self.criteria if c2.name != criterion_name)
                    c.weight = original_weights[c.name] / original_sum * remaining
            
            # Recalculate scores
            scores = self.compute_weighted_score()
            
            # Store results
            for _, row in scores.iterrows():
                results.append({
                    'Weight': w,
                    'Alternative': row['Alternative'],
                    'Score': row['Score'],
                    'Rank': row['Rank']
                })
        
        # Restore original weights
        for c in self.criteria:
            c.weight = original_weights[c.name]
        
        return pd.DataFrame(results)
    
    def export_decision_matrix(self) -> pd.DataFrame:
        """Export complete decision matrix with scores and ranks"""
        df = self.matrix.copy()
        df['Overall_Score'] = [alt.final_score for alt in self.alternatives]
        df['Rank'] = [alt.rank for alt in self.alternatives]
        
        return df.sort_values('Rank')


def demonstrate_decision_analysis():
    """Demonstrate decision matrix analysis"""
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
            id='A1',
            name='MAPbI3',
            properties={'bandgap': 1.55, 'stability': 0.65, 'efficiency': 20.1, 'cost': 0.45}
        ),
        Alternative(
            id='A2',
            name='FAPbI3',
            properties={'bandgap': 1.48, 'stability': 0.72, 'efficiency': 21.5, 'cost': 0.48}
        ),
        Alternative(
            id='A3',
            name='CsPbI3',
            properties={'bandgap': 1.73, 'stability': 0.85, 'efficiency': 18.3, 'cost': 0.52}
        ),
        Alternative(
            id='A4',
            name='Cs0.1FA0.9PbI3',
            properties={'bandgap': 1.50, 'stability': 0.88, 'efficiency': 22.8, 'cost': 0.46}
        ),
        Alternative(
            id='A5',
            name='Cs0.1FA0.9PbI2.8Br0.2',
            properties={'bandgap': 1.35, 'stability': 0.85, 'efficiency': 22.3, 'cost': 0.47}
        )
    ]
    
    # Create decision matrix
    dm = DecisionMatrix(criteria, alternatives)
    
    # Compute TOPSIS
    results = dm.compute_topsis()
    
    # Generate rationale
    rationale = dm.generate_decision_rationale(top_n=3)
    
    # Create visualization
    fig = dm.visualize_comparison(top_n=5)
    
    return dm, results, rationale, fig


if __name__ == "__main__":
    dm, results, rationale, fig = demonstrate_decision_analysis()
    print(results)
    print("\n" + "="*80 + "\n")
    print(rationale)
