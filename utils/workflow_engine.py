#!/usr/bin/env python3
"""
Workflow Engine: Unified Pipeline Orchestration
================================================

Manages end-to-end discovery workflows:
- DB Load → Screen → Optimize → Inverse Design → Protocol → Report
- Progress tracking with estimated time
- Configurable pipeline (skip/include steps)
- Results persistence across steps
- Error handling and recovery

Part of AlphaMaterials V11
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class StepStatus(Enum):
    """Workflow step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    name: str
    description: str
    function: Optional[Callable] = None
    estimated_time: int = 60  # seconds
    required: bool = True
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Actual duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: Optional[float]
    steps: List[WorkflowStep]
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Export to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "success": self.success,
            "error": self.error,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration": s.duration,
                    "error": s.error
                } for s in self.steps
            ]
        }


class WorkflowEngine:
    """
    Unified workflow engine for materials discovery pipeline
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowResult] = {}
        self.current_workflow: Optional[WorkflowResult] = None
        
    def create_default_pipeline(self) -> List[WorkflowStep]:
        """
        Create default discovery pipeline
        """
        return [
            WorkflowStep(
                name="db_load",
                description="Load materials database",
                estimated_time=30,
                required=True,
                dependencies=[]
            ),
            WorkflowStep(
                name="screen",
                description="Screen candidates by criteria",
                estimated_time=45,
                required=False,
                dependencies=["db_load"]
            ),
            WorkflowStep(
                name="ml_train",
                description="Train ML surrogate model",
                estimated_time=120,
                required=True,
                dependencies=["db_load"]
            ),
            WorkflowStep(
                name="optimize",
                description="Run Bayesian optimization",
                estimated_time=180,
                required=False,
                dependencies=["ml_train"]
            ),
            WorkflowStep(
                name="inverse_design",
                description="Generate novel compositions",
                estimated_time=90,
                required=False,
                dependencies=["ml_train"]
            ),
            WorkflowStep(
                name="rank",
                description="Multi-criteria decision analysis",
                estimated_time=30,
                required=False,
                dependencies=["optimize", "inverse_design"]
            ),
            WorkflowStep(
                name="protocol",
                description="Generate synthesis protocol",
                estimated_time=20,
                required=False,
                dependencies=["rank"]
            ),
            WorkflowStep(
                name="report",
                description="Generate research report",
                estimated_time=40,
                required=False,
                dependencies=["rank", "protocol"]
            )
        ]
    
    def estimate_total_time(self, steps: List[WorkflowStep]) -> int:
        """Estimate total execution time in seconds"""
        # Only count non-skipped steps
        active_steps = [s for s in steps if s.status != StepStatus.SKIPPED]
        return sum(s.estimated_time for s in active_steps)
    
    def check_dependencies(self, step: WorkflowStep, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied"""
        if not step.dependencies:
            return True
        
        # At least one dependency must be completed
        return any(dep in completed_steps for dep in step.dependencies)
    
    def execute_workflow(
        self,
        steps: List[WorkflowStep],
        step_functions: Dict[str, Callable],
        progress_callback: Optional[Callable] = None
    ) -> WorkflowResult:
        """
        Execute workflow pipeline
        
        Args:
            steps: List of workflow steps
            step_functions: Dictionary mapping step names to functions
            progress_callback: Optional callback for progress updates
                             Signature: callback(step_name, status, progress_pct)
        
        Returns:
            WorkflowResult with execution details
        """
        workflow_id = f"workflow_{int(time.time())}"
        start_time = datetime.now()
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            start_time=start_time,
            end_time=None,
            total_duration=None,
            steps=steps,
            success=False
        )
        
        self.current_workflow = result
        completed_steps = []
        
        try:
            total_steps = len([s for s in steps if s.status != StepStatus.SKIPPED])
            completed_count = 0
            
            for i, step in enumerate(steps):
                # Check if step should be skipped
                if step.status == StepStatus.SKIPPED:
                    if progress_callback:
                        progress_callback(step.name, "skipped", completed_count / total_steps * 100)
                    continue
                
                # Check dependencies
                if not self.check_dependencies(step, completed_steps):
                    step.status = StepStatus.SKIPPED
                    step.error = "Dependencies not met"
                    if progress_callback:
                        progress_callback(step.name, "skipped_deps", completed_count / total_steps * 100)
                    continue
                
                # Execute step
                step.status = StepStatus.RUNNING
                step.start_time = datetime.now()
                
                if progress_callback:
                    progress_callback(step.name, "running", completed_count / total_steps * 100)
                
                try:
                    # Get function for this step
                    if step.name in step_functions:
                        func = step_functions[step.name]
                        step.result = func()
                    elif step.function:
                        step.result = step.function()
                    else:
                        # No function provided - mark as completed anyway
                        step.result = {"status": "placeholder"}
                    
                    step.status = StepStatus.COMPLETED
                    step.end_time = datetime.now()
                    completed_steps.append(step.name)
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(step.name, "completed", completed_count / total_steps * 100)
                    
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    step.end_time = datetime.now()
                    
                    if step.required:
                        # Required step failed - abort workflow
                        result.success = False
                        result.error = f"Required step '{step.name}' failed: {e}"
                        break
                    else:
                        # Optional step failed - continue
                        if progress_callback:
                            progress_callback(step.name, "failed", completed_count / total_steps * 100)
            
            # Check if workflow completed successfully
            required_steps = [s for s in steps if s.required]
            result.success = all(s.status == StepStatus.COMPLETED for s in required_steps)
            
        except Exception as e:
            result.success = False
            result.error = f"Workflow execution error: {e}"
        
        finally:
            result.end_time = datetime.now()
            result.total_duration = (result.end_time - result.start_time).total_seconds()
            self.workflows[workflow_id] = result
        
        return result
    
    def get_progress_summary(self, workflow: WorkflowResult) -> Dict[str, Any]:
        """Get workflow progress summary"""
        total_steps = len(workflow.steps)
        completed = len([s for s in workflow.steps if s.status == StepStatus.COMPLETED])
        failed = len([s for s in workflow.steps if s.status == StepStatus.FAILED])
        skipped = len([s for s in workflow.steps if s.status == StepStatus.SKIPPED])
        running = len([s for s in workflow.steps if s.status == StepStatus.RUNNING])
        pending = len([s for s in workflow.steps if s.status == StepStatus.PENDING])
        
        progress_pct = (completed / total_steps * 100) if total_steps > 0 else 0
        
        return {
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "running": running,
            "pending": pending,
            "progress_pct": progress_pct,
            "estimated_time_remaining": self._estimate_remaining_time(workflow),
            "elapsed_time": workflow.total_duration if workflow.end_time else (
                datetime.now() - workflow.start_time
            ).total_seconds()
        }
    
    def _estimate_remaining_time(self, workflow: WorkflowResult) -> int:
        """Estimate remaining time in seconds"""
        pending_steps = [s for s in workflow.steps if s.status == StepStatus.PENDING]
        return sum(s.estimated_time for s in pending_steps)
    
    def configure_pipeline(
        self,
        base_steps: List[WorkflowStep],
        config: Dict[str, bool]
    ) -> List[WorkflowStep]:
        """
        Configure pipeline - enable/disable optional steps
        
        Args:
            base_steps: Base pipeline steps
            config: Dictionary {step_name: enabled}
        
        Returns:
            Configured steps with skipped steps marked
        """
        for step in base_steps:
            if step.name in config:
                if not config[step.name]:
                    step.status = StepStatus.SKIPPED
        
        return base_steps
    
    def save_workflow(self, workflow_id: str, filepath: str):
        """Save workflow result to JSON"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        with open(filepath, 'w') as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    def load_workflow(self, filepath: str) -> WorkflowResult:
        """Load workflow from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct workflow (simplified - doesn't restore functions)
        # This is for viewing past workflows, not re-execution
        return data


def demonstrate_workflow():
    """Demonstrate workflow engine"""
    print("🔧 AlphaMaterials Workflow Engine Demo\n")
    
    engine = WorkflowEngine()
    
    # Create default pipeline
    steps = engine.create_default_pipeline()
    
    # Configure - skip inverse design
    config = {
        "inverse_design": False,
        "screen": True
    }
    
    steps = engine.configure_pipeline(steps, config)
    
    print("📋 Configured Pipeline:")
    for step in steps:
        status_icon = "⏭️" if step.status == StepStatus.SKIPPED else "✅"
        print(f"  {status_icon} {step.name}: {step.description} (~{step.estimated_time}s)")
    
    print(f"\n⏱️ Estimated total time: {engine.estimate_total_time(steps)}s")
    
    # Define mock step functions
    def mock_step():
        """Mock step execution"""
        time.sleep(0.5)  # Simulate work
        return {"status": "success", "timestamp": datetime.now().isoformat()}
    
    step_functions = {s.name: mock_step for s in steps}
    
    # Progress callback
    def progress_callback(step_name, status, progress):
        print(f"  📊 {step_name}: {status} ({progress:.1f}% complete)")
    
    print("\n🚀 Executing Workflow...\n")
    
    result = engine.execute_workflow(steps, step_functions, progress_callback)
    
    print(f"\n✅ Workflow Complete!")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.total_duration:.1f}s")
    print(f"  Steps completed: {len([s for s in result.steps if s.status == StepStatus.COMPLETED])}/{len(result.steps)}")
    
    # Progress summary
    summary = engine.get_progress_summary(result)
    print(f"\n📊 Summary:")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Progress: {summary['progress_pct']:.1f}%")


if __name__ == "__main__":
    demonstrate_workflow()
