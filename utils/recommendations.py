#!/usr/bin/env python3
"""
Smart Recommendations Engine
=============================

Context-aware action suggestions based on user history:
- "You've screened 500 materials but haven't tried inverse design yet"
- "Your best candidate has high Pb — consider running What-If Pb-ban scenario"
- Next-step recommendations
- Feature discovery prompts

Part of AlphaMaterials V11
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RecommendationType(Enum):
    """Types of recommendations"""
    NEXT_STEP = "next_step"
    FEATURE_DISCOVERY = "feature_discovery"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    WARNING = "warning"


@dataclass
class Recommendation:
    """Individual recommendation"""
    type: RecommendationType
    priority: int  # 1-5, 5 = highest
    title: str
    message: str
    action: Optional[str] = None  # Tab to navigate to
    action_label: Optional[str] = None
    icon: str = "💡"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """Sort by priority (descending)"""
        return self.priority > other.priority


@dataclass
class UserActivity:
    """Track user activity for recommendations"""
    db_loaded: bool = False
    db_size: int = 0
    model_trained: bool = False
    model_accuracy: Optional[float] = None
    
    bo_runs: int = 0
    bo_best_score: Optional[float] = None
    bo_iterations: int = 0
    
    mo_runs: int = 0
    
    inverse_design_runs: int = 0
    
    screening_runs: int = 0
    candidates_screened: int = 0
    
    protocols_generated: int = 0
    reports_generated: int = 0
    
    federated_trained: bool = False
    
    nl_queries: int = 0
    
    best_candidate_composition: Optional[str] = None
    best_candidate_has_lead: bool = False
    best_candidate_bandgap: Optional[float] = None
    best_candidate_stability: Optional[float] = None
    
    session_duration: int = 0  # minutes
    tabs_visited: List[str] = field(default_factory=list)
    
    last_activity: datetime = field(default_factory=datetime.now)


class RecommendationEngine:
    """
    Smart recommendation engine
    """
    
    def __init__(self):
        self.activity: UserActivity = UserActivity()
        self.recommendations: List[Recommendation] = []
        self.dismissed_recommendations: List[str] = []
    
    def update_activity(self, **kwargs):
        """Update user activity metrics"""
        for key, value in kwargs.items():
            if hasattr(self.activity, key):
                setattr(self.activity, key, value)
        
        self.activity.last_activity = datetime.now()
    
    def generate_recommendations(self) -> List[Recommendation]:
        """
        Generate context-aware recommendations
        
        Returns list sorted by priority
        """
        self.recommendations = []
        
        # 1. Getting Started Recommendations
        self._recommend_getting_started()
        
        # 2. Next Steps Recommendations
        self._recommend_next_steps()
        
        # 3. Optimization Suggestions
        self._recommend_optimizations()
        
        # 4. Validation Recommendations
        self._recommend_validation()
        
        # 5. Feature Discovery
        self._recommend_features()
        
        # 6. Warnings
        self._recommend_warnings()
        
        # Sort by priority
        self.recommendations.sort()
        
        # Filter dismissed
        self.recommendations = [
            r for r in self.recommendations 
            if r.title not in self.dismissed_recommendations
        ]
        
        return self.recommendations
    
    def _recommend_getting_started(self):
        """Recommendations for new users"""
        if not self.activity.db_loaded:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=5,
                title="Load Materials Database",
                message="Start by loading a materials database. Try the unified database client with multiple sources.",
                action="Database",
                action_label="Go to Database Tab",
                icon="🗄️"
            ))
        
        if self.activity.db_loaded and not self.activity.model_trained:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=5,
                title="Train ML Model",
                message=f"You've loaded {self.activity.db_size} materials. Train a machine learning model to enable predictions.",
                action="ML Model",
                action_label="Go to ML Model Tab",
                icon="🤖"
            ))
    
    def _recommend_next_steps(self):
        """Recommend next logical steps"""
        # Screened but not optimized
        if self.activity.candidates_screened > 50 and self.activity.bo_runs == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=4,
                title="Try Bayesian Optimization",
                message=f"You've screened {self.activity.candidates_screened} materials. Use Bayesian optimization to intelligently explore the design space.",
                action="Bayesian Opt",
                action_label="Start Optimization",
                icon="🎯"
            ))
        
        # Optimized but not inverse design
        if self.activity.bo_runs > 0 and self.activity.inverse_design_runs == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=3,
                title="Explore Inverse Design",
                message="You've found candidates via optimization. Try inverse design to generate novel compositions beyond the database.",
                action="Inverse Design",
                action_label="Launch Inverse Design",
                icon="🧬"
            ))
        
        # Has candidates but no protocol
        if self.activity.best_candidate_composition and self.activity.protocols_generated == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=4,
                title="Generate Synthesis Protocol",
                message=f"You have a top candidate: {self.activity.best_candidate_composition}. Generate a step-by-step synthesis protocol for the lab.",
                action="Synthesis Protocols",
                action_label="Create Protocol",
                icon="🧪"
            ))
        
        # Has results but no report
        if self.activity.bo_runs > 0 and self.activity.reports_generated == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.NEXT_STEP,
                priority=3,
                title="Generate Research Report",
                message="Document your discovery campaign. Auto-generate a research report (journal paper, internal report, or presentation).",
                action="Research Reports",
                action_label="Create Report",
                icon="📄"
            ))
    
    def _recommend_optimizations(self):
        """Optimization improvement suggestions"""
        # Low BO iterations
        if self.activity.bo_runs > 0 and self.activity.bo_iterations < 20:
            self.recommendations.append(Recommendation(
                type=RecommendationType.OPTIMIZATION,
                priority=3,
                title="Increase BO Iterations",
                message=f"Your last BO run used {self.activity.bo_iterations} iterations. Try 50+ iterations for better exploration.",
                action="Bayesian Opt",
                action_label="Adjust Settings",
                icon="⚙️"
            ))
        
        # Multi-objective not used
        if self.activity.bo_runs > 1 and self.activity.mo_runs == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.OPTIMIZATION,
                priority=2,
                title="Try Multi-Objective Optimization",
                message="Optimize multiple properties simultaneously (bandgap + stability + cost). Discover Pareto-optimal trade-offs.",
                action="Multi-Objective",
                action_label="Explore Multi-Objective",
                icon="🏆"
            ))
    
    def _recommend_validation(self):
        """Validation and verification recommendations"""
        # Model accuracy issues
        if self.activity.model_trained and self.activity.model_accuracy:
            if self.activity.model_accuracy < 0.7:
                self.recommendations.append(Recommendation(
                    type=RecommendationType.VALIDATION,
                    priority=4,
                    title="Improve Model Accuracy",
                    message=f"Your model has R²={self.activity.model_accuracy:.2f}. Consider: (1) More training data, (2) Feature engineering, (3) Transfer learning.",
                    action="Transfer Learning",
                    action_label="Try Transfer Learning",
                    icon="⚠️"
                ))
        
        # TEA not run
        if self.activity.best_candidate_composition and self.activity.session_duration > 10:
            if "Techno-Economics" not in self.activity.tabs_visited:
                self.recommendations.append(Recommendation(
                    type=RecommendationType.VALIDATION,
                    priority=2,
                    title="Run Techno-Economic Analysis",
                    message="Validate commercial viability. Calculate levelized cost of electricity (LCOE) and compare to silicon baseline.",
                    action="Techno-Economics",
                    action_label="Analyze Economics",
                    icon="💰"
                ))
    
    def _recommend_features(self):
        """Feature discovery prompts"""
        # Natural language not used
        if self.activity.session_duration > 5 and self.activity.nl_queries == 0:
            self.recommendations.append(Recommendation(
                type=RecommendationType.FEATURE_DISCOVERY,
                priority=2,
                title="Try Natural Language Queries",
                message='Ask questions in plain English: "Find me a cheap perovskite with bandgap 1.3 eV". No need to navigate tabs!',
                action="Natural Language",
                action_label="Try NL Interface",
                icon="🗣️"
            ))
        
        # Knowledge graph not used
        if self.activity.bo_runs > 0:
            if "Knowledge Graph" not in self.activity.tabs_visited:
                self.recommendations.append(Recommendation(
                    type=RecommendationType.FEATURE_DISCOVERY,
                    priority=1,
                    title="Visualize Knowledge Graph",
                    message="See how your discoveries connect. Build an interactive knowledge graph showing composition-property-process relationships.",
                    action="Knowledge Graph",
                    action_label="Build Graph",
                    icon="🕸️"
                ))
        
        # Federated learning not explored
        if self.activity.model_trained and not self.activity.federated_trained:
            if "Federated Learning" not in self.activity.tabs_visited:
                self.recommendations.append(Recommendation(
                    type=RecommendationType.FEATURE_DISCOVERY,
                    priority=1,
                    title="Explore Federated Learning",
                    message="Simulate multi-lab collaboration. Train models on distributed data without sharing raw datasets.",
                    action="Federated Learning",
                    action_label="Try Federated Mode",
                    icon="🤝"
                ))
    
    def _recommend_warnings(self):
        """Important warnings"""
        # Lead toxicity warning
        if self.activity.best_candidate_has_lead:
            self.recommendations.append(Recommendation(
                type=RecommendationType.WARNING,
                priority=5,
                title="Lead Toxicity Alert",
                message=f"Your top candidate ({self.activity.best_candidate_composition}) contains lead (Pb). Consider: (1) Lead-free alternatives, (2) Pb-ban scenario analysis, (3) Proper safety protocols.",
                action="Scenarios",
                action_label="Run Pb-Ban Scenario",
                icon="☠️"
            ))
        
        # Low stability warning
        if self.activity.best_candidate_stability and self.activity.best_candidate_stability < 0.6:
            self.recommendations.append(Recommendation(
                type=RecommendationType.WARNING,
                priority=4,
                title="Low Stability Detected",
                message=f"Best candidate stability: {self.activity.best_candidate_stability:.2f}. This may degrade quickly. Consider: (1) Cs/Rb addition, (2) Mixed cations, (3) 2D perovskites.",
                action="Multi-Objective",
                action_label="Optimize for Stability",
                icon="⚠️"
            ))
        
        # Bandgap mismatch warning
        if self.activity.best_candidate_bandgap:
            if abs(self.activity.best_candidate_bandgap - 1.35) > 0.15:
                self.recommendations.append(Recommendation(
                    type=RecommendationType.WARNING,
                    priority=3,
                    title="Bandgap Mismatch for Tandem Cell",
                    message=f"Best candidate bandgap: {self.activity.best_candidate_bandgap:.2f} eV. Ideal for tandem: 1.35 eV. Consider halide mixing (I/Br) for tuning.",
                    action="Inverse Design",
                    action_label="Tune Bandgap",
                    icon="🎯"
                ))
    
    def dismiss_recommendation(self, title: str):
        """Mark recommendation as dismissed"""
        self.dismissed_recommendations.append(title)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get recommendation summary statistics"""
        recs = self.generate_recommendations()
        
        by_type = {}
        for rec_type in RecommendationType:
            by_type[rec_type.value] = len([r for r in recs if r.type == rec_type])
        
        by_priority = {
            "high": len([r for r in recs if r.priority >= 4]),
            "medium": len([r for r in recs if r.priority == 3]),
            "low": len([r for r in recs if r.priority <= 2])
        }
        
        return {
            "total": len(recs),
            "by_type": by_type,
            "by_priority": by_priority,
            "top_recommendation": recs[0] if recs else None
        }


def demonstrate_recommendations():
    """Demonstrate recommendation engine"""
    print("💡 AlphaMaterials Recommendations Engine Demo\n")
    
    engine = RecommendationEngine()
    
    # Scenario 1: New user
    print("📋 Scenario 1: New User")
    recs = engine.generate_recommendations()
    for rec in recs[:3]:
        print(f"  {rec.icon} [{rec.priority}★] {rec.title}")
        print(f"     {rec.message}\n")
    
    # Scenario 2: User has screened materials
    print("\n📋 Scenario 2: Experienced User (screened 500 materials, no optimization)")
    engine.update_activity(
        db_loaded=True,
        db_size=1000,
        model_trained=True,
        model_accuracy=0.85,
        candidates_screened=500,
        screening_runs=3,
        session_duration=20,
        tabs_visited=["Database", "ML Model", "Dashboard"]
    )
    
    recs = engine.generate_recommendations()
    for rec in recs[:5]:
        print(f"  {rec.icon} [{rec.priority}★] {rec.title}")
        print(f"     {rec.message}\n")
    
    # Scenario 3: Lead warning
    print("\n📋 Scenario 3: Lead Toxicity Warning")
    engine.update_activity(
        best_candidate_composition="MAPbI3",
        best_candidate_has_lead=True,
        best_candidate_bandgap=1.55,
        best_candidate_stability=0.65,
        bo_runs=2,
        bo_iterations=50
    )
    
    recs = engine.generate_recommendations()
    warnings = [r for r in recs if r.type == RecommendationType.WARNING]
    for rec in warnings:
        print(f"  {rec.icon} [{rec.priority}★] {rec.title}")
        print(f"     {rec.message}\n")
    
    # Summary
    print("\n📊 Recommendation Summary:")
    summary = engine.get_summary()
    print(f"  Total: {summary['total']}")
    print(f"  High Priority: {summary['by_priority']['high']}")
    print(f"  Medium Priority: {summary['by_priority']['medium']}")
    print(f"  Low Priority: {summary['by_priority']['low']}")
    
    if summary['top_recommendation']:
        top = summary['top_recommendation']
        print(f"\n  🏆 Top Recommendation: {top.title}")


if __name__ == "__main__":
    demonstrate_recommendations()
