"""
AlphaMaterials V11: Smart Recommendations Engine
==================================================
Context-aware suggestions based on user activity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class Recommendation:
    """A single recommendation."""
    title: str
    description: str
    icon: str
    priority: int  # 1=high, 2=medium, 3=low
    action_tab: Optional[str] = None
    category: str = "general"


class SmartRecommendations:
    """Generate context-aware recommendations based on user activity."""

    def __init__(self):
        self.activity_log: List[Dict] = []

    def log_activity(self, action: str, details: Dict = None):
        self.activity_log.append({
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })

    def _has_done(self, action: str) -> bool:
        return any(a["action"] == action for a in self.activity_log)

    def _count(self, action: str) -> int:
        return sum(1 for a in self.activity_log if a["action"] == action)

    def generate(self, session_state: Dict = None) -> List[Recommendation]:
        """Generate recommendations based on current state."""
        recs = []
        state = session_state or {}

        # Check what user has/hasn't done
        has_db = state.get("db_loaded", False)
        has_upload = state.get("user_data_uploaded", False)
        has_model = state.get("model_trained", False)
        has_bo = state.get("bo_run", False)
        has_pareto = state.get("pareto_run", False)
        has_inverse = state.get("inverse_run", False)
        has_protocol = state.get("protocol_generated", False)
        has_report = state.get("report_generated", False)
        has_fl = state.get("fl_run", False)
        best_candidate = state.get("best_candidate", None)

        # Getting started
        if not has_db:
            recs.append(Recommendation(
                "Load Database First",
                "Start by loading materials data from public databases (Materials Project, AFLOW, JARVIS).",
                "📂", 1, "Database Explorer", "getting_started"
            ))

        if has_db and not has_model:
            recs.append(Recommendation(
                "Train Your Model",
                "You have data loaded — train a surrogate model to enable predictions and optimization.",
                "🧠", 1, "ML Surrogate", "getting_started"
            ))

        # Exploration suggestions
        if has_model and not has_bo:
            recs.append(Recommendation(
                "Try Bayesian Optimization",
                "Your model is ready! Use BO to efficiently find optimal compositions.",
                "🎯", 1, "Bayesian Optimization", "explore"
            ))

        if has_bo and not has_pareto:
            recs.append(Recommendation(
                "Explore Multi-Objective Trade-offs",
                "You've found good candidates — now balance bandgap, stability, and cost with Pareto optimization.",
                "🏆", 2, "Multi-Objective", "explore"
            ))

        if has_model and not has_inverse:
            recs.append(Recommendation(
                "Try Inverse Design",
                "Specify your TARGET properties and let AI generate matching compositions.",
                "🧬", 2, "Inverse Design", "explore"
            ))

        # Data enrichment
        if has_db and not has_upload:
            recs.append(Recommendation(
                "Upload Your Experimental Data",
                "Combine your lab data with the database for personalized predictions.",
                "📤", 2, "User Data Upload", "data"
            ))

        if has_upload and has_model:
            recs.append(Recommendation(
                "Fine-Tune with Your Data",
                "Your uploaded data can improve model accuracy for your specific materials.",
                "⚡", 2, "ML Surrogate", "data"
            ))

        # Analysis
        if best_candidate and not has_protocol:
            recs.append(Recommendation(
                "Generate Synthesis Protocol",
                f"You have a top candidate — generate a step-by-step lab protocol!",
                "🧪", 1, "Synthesis Protocol", "action"
            ))

        if has_bo and not has_report:
            recs.append(Recommendation(
                "Generate Research Report",
                "Document your discovery campaign with an auto-generated report.",
                "📄", 2, "Research Report", "action"
            ))

        # Advanced features
        if has_model and not has_fl:
            recs.append(Recommendation(
                "Explore Federated Learning",
                "See how multi-lab collaboration could improve your model without sharing data.",
                "🤝", 3, "Federated Learning", "advanced"
            ))

        if best_candidate and "Pb" in str(best_candidate):
            recs.append(Recommendation(
                "Run 'What-If Pb Ban' Scenario",
                "Your best candidate contains lead — check what happens if Pb is restricted.",
                "⚠️", 2, "What-If Scenarios", "risk"
            ))

        # Sort by priority
        recs.sort(key=lambda r: r.priority)
        return recs

    def get_tip(self, current_tab: str) -> Optional[str]:
        """Get a contextual tip for the current tab."""
        tips = {
            "Database Explorer": "💡 Tip: Use filters to narrow down to your target bandgap range before training.",
            "ML Surrogate": "💡 Tip: More training data = better predictions. Upload your experimental data too!",
            "Bayesian Optimization": "💡 Tip: Start with Expected Improvement (EI) — it balances exploration and exploitation.",
            "Multi-Objective": "💡 Tip: Drag the weight sliders to prioritize what matters most to your application.",
            "Inverse Design": "💡 Tip: Set realistic constraints — very narrow ranges may return no candidates.",
            "Federated Learning": "💡 Tip: Try ε=1.0 for a good privacy-accuracy balance.",
            "Decision Matrix": "💡 Tip: TOPSIS works best with 3-7 criteria. Too many dilutes the analysis.",
        }
        return tips.get(current_tab)
