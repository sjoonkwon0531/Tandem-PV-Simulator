"""
AlphaMaterials V11: Unified Workflow Engine
============================================
One-click full pipeline orchestration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time
import json


@dataclass
class PipelineStep:
    """A single step in the discovery pipeline."""
    name: str
    description: str
    icon: str
    estimated_seconds: float
    required: bool = True
    completed: bool = False
    result: Any = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.completed:
            return "completed"
        if self.start_time and not self.end_time:
            return "running"
        return "pending"


class WorkflowEngine:
    """Orchestrates the full AlphaMaterials discovery pipeline."""

    DEFAULT_STEPS = [
        PipelineStep("Load Database", "Connect to Materials Project, AFLOW, JARVIS", "📂", 10.0),
        PipelineStep("Screen Materials", "Filter by bandgap range and stability", "🔍", 5.0),
        PipelineStep("Train Model", "Build surrogate model from database", "🧠", 15.0),
        PipelineStep("Bayesian Optimization", "Find optimal compositions with BO", "🎯", 20.0),
        PipelineStep("Multi-Objective Pareto", "Balance bandgap, stability, cost", "🏆", 10.0),
        PipelineStep("Inverse Design", "Generate candidates from target properties", "🧬", 15.0, required=False),
        PipelineStep("Techno-Economics", "Calculate $/Watt and cost drivers", "💰", 5.0, required=False),
        PipelineStep("Risk Assessment", "Evaluate toxicity, supply chain, TRL", "⚠️", 5.0, required=False),
        PipelineStep("Generate Protocol", "Create synthesis procedures", "🧪", 3.0),
        PipelineStep("Generate Report", "Compile research report", "📄", 5.0),
    ]

    def __init__(self):
        self.steps = [PipelineStep(s.name, s.description, s.icon, s.estimated_seconds, s.required)
                      for s in self.DEFAULT_STEPS]
        self.pipeline_start: Optional[float] = None
        self.pipeline_end: Optional[float] = None
        self.config: Dict[str, bool] = {s.name: s.required for s in self.steps}

    def configure(self, step_toggles: Dict[str, bool]):
        """Configure which steps to include."""
        self.config = step_toggles
        for step in self.steps:
            if step.name in step_toggles:
                step.required = step_toggles[step.name]

    def get_active_steps(self) -> List[PipelineStep]:
        return [s for s in self.steps if self.config.get(s.name, s.required)]

    def total_estimated_time(self) -> float:
        return sum(s.estimated_seconds for s in self.get_active_steps())

    def completed_count(self) -> int:
        return sum(1 for s in self.get_active_steps() if s.completed)

    def total_count(self) -> int:
        return len(self.get_active_steps())

    def progress_pct(self) -> float:
        total = self.total_count()
        if total == 0:
            return 0.0
        return self.completed_count() / total

    def mark_step_start(self, step_name: str):
        for s in self.steps:
            if s.name == step_name:
                s.start_time = time.time()
                break

    def mark_step_complete(self, step_name: str, result: Any = None):
        for s in self.steps:
            if s.name == step_name:
                s.end_time = time.time()
                s.completed = True
                s.result = result
                break

    def mark_step_error(self, step_name: str, error: str):
        for s in self.steps:
            if s.name == step_name:
                s.end_time = time.time()
                s.error = error
                break

    def get_summary(self) -> Dict:
        active = self.get_active_steps()
        return {
            "total_steps": len(active),
            "completed": sum(1 for s in active if s.completed),
            "errors": sum(1 for s in active if s.error),
            "progress": self.progress_pct(),
            "estimated_total_s": self.total_estimated_time(),
            "actual_total_s": sum(s.duration or 0 for s in active if s.duration),
            "steps": [{
                "name": s.name,
                "icon": s.icon,
                "status": s.status,
                "duration": s.duration,
            } for s in active]
        }

    def to_json(self) -> str:
        return json.dumps(self.get_summary(), indent=2, default=str)


def demonstrate_workflow():
    """Demo the workflow engine."""
    engine = WorkflowEngine()

    # Configure: skip optional steps
    engine.configure({
        "Load Database": True,
        "Screen Materials": True,
        "Train Model": True,
        "Bayesian Optimization": True,
        "Multi-Objective Pareto": True,
        "Inverse Design": False,
        "Techno-Economics": True,
        "Risk Assessment": False,
        "Generate Protocol": True,
        "Generate Report": True,
    })

    print(f"Active steps: {engine.total_count()}")
    print(f"Estimated time: {engine.total_estimated_time():.0f}s")

    # Simulate running
    for step in engine.get_active_steps():
        engine.mark_step_start(step.name)
        time.sleep(0.01)  # Simulate work
        engine.mark_step_complete(step.name, result={"materials_found": 42})

    print(f"Progress: {engine.progress_pct():.0%}")
    print(engine.to_json())


if __name__ == "__main__":
    demonstrate_workflow()
