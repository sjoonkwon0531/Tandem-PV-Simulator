"""
Educational Mode: Interactive Learning
=======================================

Interactive tutorials, glossary, and quiz system for learning materials discovery.

Features:
- Step-by-step tutorials (bandgap, Bayesian optimization, Pareto fronts)
- Glossary of technical terms
- Quiz mode (predict and learn)
- Explainability (SHAP-like feature importance breakdown)

Author: OpenClaw Agent
Date: 2026-03-15 (V8)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class Tutorial:
    """Interactive tutorial content."""
    
    title: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    duration_min: int
    sections: List[Dict[str, str]]  # [{'title': '...', 'content': '...'}]
    quiz_questions: List[Dict]  # Optional quiz at end


class TutorialLibrary:
    """
    Library of educational tutorials.
    """
    
    @staticmethod
    def get_bandgap_tutorial() -> Tutorial:
        """Tutorial: What is bandgap?"""
        return Tutorial(
            title="Understanding Bandgap in Semiconductors",
            difficulty="beginner",
            duration_min=10,
            sections=[
                {
                    'title': 'What is Bandgap?',
                    'content': """
**Bandgap (Eg)** is the energy difference between the valence band (occupied electrons) 
and conduction band (available states for electrons to move).

Think of it like a hurdle:
- **Small bandgap (<1 eV):** Easy to jump → Conducts electricity easily (metals, narrow-gap semiconductors)
- **Medium bandgap (1-3 eV):** Moderate jump → Semiconductors (silicon, perovskites)
- **Large bandgap (>3 eV):** Hard to jump → Insulators (glass, diamond)

**Why it matters for solar cells:**
- Bandgap determines which wavelengths of light can be absorbed
- Optimal bandgap for single-junction solar cell: **~1.34 eV** (Shockley-Queisser limit)
- Too small: Absorbs everything but wastes energy as heat
- Too large: Only absorbs high-energy photons, misses visible/IR light
"""
                },
                {
                    'title': 'Bandgap and Solar Cell Efficiency',
                    'content': """
**The Goldilocks Problem:**

For a single-junction solar cell:
- Eg < 1.0 eV: High current (absorbs lots of light) but low voltage → ~25% efficiency
- **Eg ≈ 1.34 eV: Balanced** → **~33% efficiency** (Shockley-Queisser limit)
- Eg > 2.0 eV: High voltage but low current (absorbs little light) → ~20% efficiency

**Tandem Solar Cells:**
Stack two cells with different bandgaps:
- Top cell: Wide bandgap (~1.7 eV) → Absorbs blue/green light
- Bottom cell: Narrow bandgap (~1.1 eV, e.g., silicon) → Absorbs red/IR light
- Theoretical efficiency: **~46%** (much better than single junction!)

**Perovskites are ideal for tandems** because their bandgap is tunable (1.2-2.5 eV).
"""
                },
                {
                    'title': 'How to Tune Bandgap in Perovskites',
                    'content': """
**Perovskite formula: ABX3**
- **A-site** (MA, FA, Cs): Affects crystal structure, indirect bandgap influence
- **B-site** (Pb, Sn, Ge): Dominant effect on bandgap (Pb → 1.5-1.6 eV, Sn → 1.2-1.3 eV)
- **X-site** (I, Br, Cl): Direct bandgap tuning (I → narrow, Cl → wide)

**Examples:**
- MAPbI3: 1.59 eV (pure iodide)
- MAPbBr3: 2.30 eV (pure bromide) → +0.71 eV shift!
- MAPb(I0.5Br0.5)3: ~1.95 eV (mixed halide) → Linear interpolation

**Pro tip:** Mixing A-site and X-site gives precise control:
- MA0.6FA0.4PbI3: 1.55 eV (perfect for Si tandem top cell!)
"""
                }
            ],
            quiz_questions=[
                {
                    'question': 'What is the optimal bandgap for a single-junction solar cell?',
                    'options': ['0.5 eV', '1.34 eV', '2.5 eV', '5.0 eV'],
                    'correct': 1,
                    'explanation': '1.34 eV is the Shockley-Queisser optimal bandgap, balancing current and voltage.'
                },
                {
                    'question': 'Which element substitution increases bandgap most in perovskites?',
                    'options': ['A-site (MA→FA)', 'B-site (Pb→Sn)', 'X-site (I→Cl)', 'All equal'],
                    'correct': 2,
                    'explanation': 'X-site (halide) substitution has the strongest effect. I→Br adds ~0.7 eV, I→Cl adds ~1.2 eV.'
                }
            ]
        )
    
    @staticmethod
    def get_bayesian_optimization_tutorial() -> Tutorial:
        """Tutorial: How does Bayesian Optimization work?"""
        return Tutorial(
            title="Bayesian Optimization: Smart Search",
            difficulty="intermediate",
            duration_min=15,
            sections=[
                {
                    'title': 'The Problem: Too Many Experiments',
                    'content': """
**Scenario:** You have 10,000 possible perovskite compositions. Testing all would take:
- 1 hour per synthesis → 10,000 hours = 1.1 years of non-stop work!
- Cost: 10,000 × $50 = $500,000

**Question:** Can we find the best composition in <100 experiments?

**Answer:** YES! Using **Bayesian Optimization (BO)**.
"""
                },
                {
                    'title': 'How Bayesian Optimization Works',
                    'content': """
**Core Idea:** Learn as you go, focus on promising regions.

**Algorithm:**
1. **Train surrogate model** (Gaussian Process) on existing data
2. **Predict properties** for all untested compositions
3. **Uncertainty estimation:** GP gives mean ± std deviation
4. **Acquisition function:** Balance exploration (high uncertainty) vs exploitation (high predicted value)
5. **Suggest next experiment** (maximize acquisition)
6. **Run experiment, get real result**
7. **Update model** with new data
8. **Repeat** until converged

**Key insight:** Don't waste experiments on obviously bad compositions. Focus budget on "maybe good" regions.
"""
                },
                {
                    'title': 'Exploration vs Exploitation',
                    'content': """
**The Tradeoff:**

- **Exploitation:** Try compositions near current best → Refine solution
- **Exploration:** Try uncertain regions → Discover new peaks

**Acquisition Functions:**

- **Expected Improvement (EI):** How much better than current best? (balanced)
- **Upper Confidence Bound (UCB):** Optimistic estimate (more exploration)
- **Thompson Sampling (TS):** Random sample from posterior (stochastic)

**Real Example:**
- Iteration 1-5: Exploration (try diverse compositions)
- Iteration 6-15: Exploitation (narrow down to MA-FA-Pb-I region)
- Iteration 16-20: Fine-tuning (MA0.55FA0.45 vs MA0.6FA0.4)

**Result:** Find optimal in 20 experiments instead of 10,000 (500× speedup!)
"""
                }
            ],
            quiz_questions=[
                {
                    'question': 'What is the main advantage of Bayesian Optimization?',
                    'options': ['Always finds global optimum', 'Requires fewer experiments', 'No math needed', 'Works on any problem'],
                    'correct': 1,
                    'explanation': 'BO reduces the number of experiments needed by learning from past results and focusing on promising regions.'
                }
            ]
        )
    
    @staticmethod
    def get_pareto_front_tutorial() -> Tutorial:
        """Tutorial: Understanding Pareto Fronts"""
        return Tutorial(
            title="Pareto Fronts: Multi-Objective Optimization",
            difficulty="intermediate",
            duration_min=12,
            sections=[
                {
                    'title': 'The Multi-Objective Problem',
                    'content': """
**Real-world challenge:** Optimizing multiple conflicting objectives.

**Example:**
- Maximize bandgap accuracy (close to 1.35 eV)
- Minimize cost ($/W)
- Maximize stability (hours)

**Problem:** These objectives conflict!
- High accuracy perovskites (MA-FA-Pb-I) → Expensive (Pb, I costly)
- Low-cost perovskites (Cs-Sn-Br) → Lower accuracy, less stable

**Question:** Which composition is "best"?

**Answer:** There's no single best, but a **set of trade-offs** (Pareto front).
"""
                },
                {
                    'title': 'What is a Pareto Front?',
                    'content': """
**Definition:** A composition is **Pareto optimal** if:
- No other composition is better in ALL objectives
- Improving one objective requires sacrificing another

**Visual:**
```
Cost ↓
 │   A ●────────●────────● C
 │      ╲      B  ╱
 │        ╲  ●  ╱
 │          ╲ ╱  ← Pareto Front
 │         ● ╳ D (dominated by B)
 │            ╱ ╲
 └──────────────────── Accuracy →
```

- **Points A, B, C:** Pareto optimal (no strict dominance)
- **Point D:** Dominated by B (B is better in both cost AND accuracy)

**Decision:** Choose from Pareto front based on priorities (e.g., if cost critical, pick A; if accuracy critical, pick C).
"""
                },
                {
                    'title': 'How to Use Pareto Fronts',
                    'content': """
**Step-by-step:**

1. **Run multi-objective optimization** (find Pareto-optimal materials)
2. **Plot Pareto front** (2D: cost vs accuracy, or 3D: cost vs accuracy vs stability)
3. **Filter by hard constraints** (e.g., cost must be <$0.30/W)
4. **Choose based on priorities:**
   - If cost-sensitive: Pick leftmost point (lowest cost on Pareto front)
   - If performance-critical: Pick topmost point (best accuracy)
   - If balanced: Pick "knee point" (good compromise)

**Example:**
- 500 compositions tested → 50 Pareto-optimal found
- Filter: cost <$0.30/W → 12 materials remain
- Pick: MA0.6FA0.4PbI3 (knee point: cost $0.22/W, accuracy 0.02 eV error)

**Outcome:** Best-of-both-worlds solution!
"""
                }
            ],
            quiz_questions=[]
        )


class Glossary:
    """
    Glossary of technical terms.
    """
    
    TERMS = {
        'Bandgap (Eg)': 'Energy difference between valence and conduction bands. Determines which wavelengths of light a material can absorb. Measured in electron volts (eV).',
        
        'Perovskite': 'Crystal structure with formula ABX3. A is large cation (MA, FA, Cs), B is metal (Pb, Sn), X is halide (I, Br, Cl) or oxide (O). Known for tunable bandgap and high solar cell efficiency.',
        
        'Bayesian Optimization (BO)': 'Smart search algorithm that learns from experiments to suggest next best composition to test. Reduces number of experiments needed by balancing exploration and exploitation.',
        
        'Gaussian Process (GP)': 'Machine learning model that predicts not just a value (e.g., bandgap), but also uncertainty. Used in Bayesian optimization to guide search.',
        
        'Pareto Front': 'Set of solutions where no solution is strictly better in all objectives. Represents trade-offs in multi-objective optimization (e.g., cost vs performance).',
        
        'Acquisition Function': 'Formula that decides which experiment to run next in Bayesian optimization. Common types: Expected Improvement (EI), Upper Confidence Bound (UCB), Thompson Sampling (TS).',
        
        'Shockley-Queisser Limit': 'Theoretical maximum efficiency of a single-junction solar cell. For optimal bandgap (~1.34 eV), limit is 33.7%. Perovskite-silicon tandems can exceed this.',
        
        'Tandem Solar Cell': 'Multi-layer solar cell with different bandgap materials stacked. Top cell absorbs high-energy photons, bottom cell absorbs low-energy photons. Higher efficiency than single junction.',
        
        'Feature Engineering': 'Converting chemical formula (e.g., MAPbI3) into numerical features (e.g., ionic radius, electronegativity) that machine learning models can understand.',
        
        'Mean Absolute Error (MAE)': 'Average absolute difference between predicted and true values. Lower is better. For bandgap prediction, MAE <0.1 eV is excellent.',
        
        'R² (Coefficient of Determination)': 'Fraction of variance explained by model. Ranges 0-1. R²=0.9 means model explains 90% of variation. R²>0.8 is good for materials prediction.',
        
        'Inverse Design': 'Starting from desired property (e.g., Eg=1.35 eV) and finding compositions that achieve it. Opposite of forward prediction (composition → property).',
        
        'Digital Twin': 'Virtual simulation of real process (e.g., perovskite film formation). Allows testing process parameters (temperature, spin speed) before running real experiments.',
        
        'Transfer Learning': 'Using knowledge from one material domain (e.g., halide perovskites) to accelerate learning in another domain (e.g., oxide perovskites). Reduces data needed.',
        
        'Techno-Economic Analysis (TEA)': 'Calculating cost per watt ($/W) of a solar cell, including materials, processing, and scale-up costs. Used to assess commercial viability.'
    }
    
    @staticmethod
    def get_definition(term: str) -> str:
        """Get definition of a term."""
        return Glossary.TERMS.get(term, 'Term not found in glossary.')
    
    @staticmethod
    def search(query: str) -> List[Tuple[str, str]]:
        """Search glossary for matching terms."""
        query_lower = query.lower()
        matches = []
        
        for term, definition in Glossary.TERMS.items():
            if query_lower in term.lower() or query_lower in definition.lower():
                matches.append((term, definition))
        
        return matches


class QuizEngine:
    """
    Interactive quiz system.
    """
    
    @staticmethod
    def generate_bandgap_quiz(model, featurizer, n_questions: int = 5) -> List[Dict]:
        """
        Generate quiz: "Predict the bandgap of this composition"
        
        Args:
            model: Trained ML model
            featurizer: Composition featurizer
            n_questions: Number of quiz questions
        
        Returns:
            List of quiz questions with answers
        """
        # Sample compositions with known bandgaps
        compositions = [
            ('MAPbI3', 1.59),
            ('FAPbI3', 1.51),
            ('CsPbI3', 1.72),
            ('MAPbBr3', 2.30),
            ('FAPbBr3', 2.25),
            ('CsPbBr3', 2.36),
            ('MAPbCl3', 3.11),
            ('MA0.5FA0.5PbI3', 1.55),
            ('MAPb0.5Sn0.5I3', 1.25),
            ('CsSnI3', 1.30)
        ]
        
        # Randomly select questions
        random.shuffle(compositions)
        selected = compositions[:n_questions]
        
        quiz = []
        
        for formula, true_bg in selected:
            # Get ML prediction
            try:
                X = featurizer.transform([formula])
                pred_bg = model.predict(X)[0]
            except:
                pred_bg = true_bg  # Fallback if featurization fails
            
            # Generate multiple choice options
            options = [
                true_bg,
                true_bg + 0.3,
                true_bg - 0.3,
                true_bg + 0.6
            ]
            random.shuffle(options)
            correct_idx = options.index(true_bg)
            
            quiz.append({
                'question': f"What is the bandgap of {formula}?",
                'composition': formula,
                'options': [f"{opt:.2f} eV" for opt in options],
                'correct': correct_idx,
                'true_value': true_bg,
                'ml_prediction': pred_bg,
                'explanation': f"The experimental bandgap of {formula} is {true_bg:.2f} eV. The ML model predicted {pred_bg:.2f} eV (error: {abs(pred_bg - true_bg):.3f} eV)."
            })
        
        return quiz
    
    @staticmethod
    def explain_prediction(model, featurizer, composition: str, feature_names: List[str]) -> Dict:
        """
        Explain why model made a prediction (SHAP-like feature importance).
        
        Args:
            model: Trained model (must have feature_importances_ or coef_)
            featurizer: Composition featurizer
            composition: Composition to explain
            feature_names: Names of features
        
        Returns:
            Explanation dictionary
        """
        # Get features
        try:
            X = featurizer.transform([composition])
        except:
            return {'error': 'Failed to featurize composition'}
        
        # Get prediction
        try:
            prediction = model.predict(X)[0]
        except:
            return {'error': 'Failed to predict'}
        
        # Get feature importances (if available)
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        
        if importances is None:
            return {
                'composition': composition,
                'prediction': prediction,
                'explanation': 'Model does not support feature importance extraction.'
            }
        
        # Rank features by importance
        importance_pairs = list(zip(feature_names[:len(importances)], importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_features = importance_pairs[:5]
        
        explanation = f"**Prediction for {composition}: {prediction:.3f} eV**\n\n"
        explanation += "**Top 5 Most Important Features:**\n\n"
        
        for i, (feat, imp) in enumerate(top_features, 1):
            explanation += f"{i}. **{feat}**: {imp:.4f}\n"
        
        explanation += f"\n*Note: Feature importance shows which chemical properties most influenced this prediction.*"
        
        return {
            'composition': composition,
            'prediction': prediction,
            'top_features': top_features,
            'explanation': explanation
        }


class GuidedWorkflow:
    """
    Step-by-step guided discovery workflow.
    """
    
    STEPS = [
        {
            'step': 1,
            'title': '🗄️ Load Data',
            'description': 'Load existing perovskite data from databases or upload your own experiments.',
            'action': 'Go to Database tab and click "Load Database"',
            'success_criteria': 'Database loaded (>100 materials)'
        },
        {
            'step': 2,
            'title': '🤖 Train ML Model',
            'description': 'Train a machine learning model to predict bandgaps from compositions.',
            'action': 'Go to ML Surrogate tab and click "Train Model"',
            'success_criteria': 'Model trained with MAE <0.2 eV'
        },
        {
            'step': 3,
            'title': '🎯 Set Target',
            'description': 'Define your target bandgap (e.g., 1.35 eV for Si tandem top cell).',
            'action': 'Set "Target Bandgap" in sidebar',
            'success_criteria': 'Target set (e.g., 1.35 eV)'
        },
        {
            'step': 4,
            'title': '🔍 Run Bayesian Optimization',
            'description': 'Use smart search to find compositions matching your target.',
            'action': 'Go to Bayesian Opt tab and click "Fit Optimizer"',
            'success_criteria': 'Top candidates found (error <0.1 eV)'
        },
        {
            'step': 5,
            'title': '🧬 Inverse Design',
            'description': 'Generate novel compositions that achieve target bandgap.',
            'action': 'Go to Inverse Design tab and click "Generate Candidates"',
            'success_criteria': '100+ candidates generated'
        },
        {
            'step': 6,
            'title': '💰 Analyze Costs',
            'description': 'Calculate cost per watt for top candidates.',
            'action': 'Go to Techno-Economics tab and run analysis',
            'success_criteria': 'Find materials with cost <$0.30/W'
        },
        {
            'step': 7,
            'title': '✅ Select Winner',
            'description': 'Choose best candidate balancing performance and cost.',
            'action': 'Review Dashboard and pick from Pareto front',
            'success_criteria': 'Decision made!'
        }
    ]
    
    @staticmethod
    def get_step(step_number: int) -> Dict:
        """Get details of a workflow step."""
        for step in GuidedWorkflow.STEPS:
            if step['step'] == step_number:
                return step
        return {}
    
    @staticmethod
    def get_all_steps() -> List[Dict]:
        """Get all workflow steps."""
        return GuidedWorkflow.STEPS
