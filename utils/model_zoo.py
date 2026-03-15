"""
Model Zoo: Foundation Model Hub
================================

Central registry for all trained models with metadata, versioning, and lifecycle management.

Features:
- Model cards (metadata, training data, metrics)
- Model versioning with changelog
- Compare models side-by-side
- Import/export (joblib serialization)
- Model families (base, fine-tuned, domain-specific)

Author: OpenClaw Agent
Date: 2026-03-15 (V8)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator


@dataclass
class ModelCard:
    """
    Model metadata card (inspired by Hugging Face model cards).
    
    Contains all information needed to understand, reproduce, and use a model.
    """
    
    # Identity
    model_id: str
    name: str
    version: str
    family: str  # 'base', 'fine-tuned', 'domain-specific', 'user-trained'
    
    # Training info
    training_data_size: int
    training_data_source: str
    features_used: List[str]
    target_property: str
    
    # Performance metrics
    mae: float
    r2: float
    rmse: float
    inference_speed_ms: float  # Average prediction time
    
    # Domain info
    domain: str  # 'halide_perovskites', 'oxide_perovskites', 'chalcogenides', 'general'
    bandgap_range: Tuple[float, float]  # Min, max coverage
    coverage: int  # Number of unique compositions can predict
    
    # Metadata
    author: str
    created_at: str
    description: str
    changelog: List[Dict[str, str]]  # Version history
    
    # Model artifact
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelCard':
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Central registry of all models.
    
    Manages model lifecycle: register → train → version → compare → export.
    """
    
    def __init__(self, registry_dir: str = './models/registry'):
        """
        Initialize registry.
        
        Args:
            registry_dir: Directory to store model cards and artifacts
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelCard] = {}
        self.artifacts: Dict[str, BaseEstimator] = {}  # In-memory model cache
        
        # Load existing registry
        self._load_registry()
    
    def register_model(self,
                      model: BaseEstimator,
                      model_id: str,
                      name: str,
                      version: str,
                      family: str,
                      training_data: pd.DataFrame,
                      features_used: List[str],
                      target_property: str,
                      metrics: Dict[str, float],
                      domain: str = 'general',
                      author: str = 'User',
                      description: str = '') -> ModelCard:
        """
        Register a new model in the zoo.
        
        Args:
            model: Trained sklearn model
            model_id: Unique identifier (e.g., 'halide-rf-v1')
            name: Human-readable name (e.g., 'Halide Perovskite Predictor')
            version: Version string (e.g., '1.0.0')
            family: Model family ('base', 'fine-tuned', 'domain-specific', 'user-trained')
            training_data: Training dataset
            features_used: Feature names
            target_property: Target property name
            metrics: Performance metrics dict (must contain 'mae', 'r2', 'rmse', 'inference_speed_ms')
            domain: Material domain
            author: Model author
            description: Model description
        
        Returns:
            ModelCard with all metadata
        """
        
        # Calculate coverage
        bandgap_values = training_data[target_property] if target_property in training_data.columns else []
        bandgap_range = (float(np.min(bandgap_values)), float(np.max(bandgap_values))) if len(bandgap_values) > 0 else (0.0, 0.0)
        coverage = len(training_data)
        
        # Create model card
        card = ModelCard(
            model_id=model_id,
            name=name,
            version=version,
            family=family,
            training_data_size=len(training_data),
            training_data_source=f"{len(training_data)} materials",
            features_used=features_used,
            target_property=target_property,
            mae=metrics.get('mae', 0.0),
            r2=metrics.get('r2', 0.0),
            rmse=metrics.get('rmse', 0.0),
            inference_speed_ms=metrics.get('inference_speed_ms', 0.0),
            domain=domain,
            bandgap_range=bandgap_range,
            coverage=coverage,
            author=author,
            created_at=datetime.now().isoformat(),
            description=description,
            changelog=[{
                'version': version,
                'date': datetime.now().isoformat(),
                'changes': 'Initial release'
            }],
            model_path=str(self.registry_dir / f"{model_id}.joblib")
        )
        
        # Save model artifact
        joblib.dump(model, card.model_path)
        
        # Register in memory
        self.models[model_id] = card
        self.artifacts[model_id] = model
        
        # Persist registry
        self._save_registry()
        
        return card
    
    def get_model(self, model_id: str) -> Optional[Tuple[BaseEstimator, ModelCard]]:
        """
        Retrieve model and its card.
        
        Args:
            model_id: Model identifier
        
        Returns:
            (model, card) tuple or None if not found
        """
        if model_id not in self.models:
            return None
        
        card = self.models[model_id]
        
        # Load from cache or disk
        if model_id in self.artifacts:
            model = self.artifacts[model_id]
        else:
            if card.model_path and Path(card.model_path).exists():
                model = joblib.load(card.model_path)
                self.artifacts[model_id] = model
            else:
                return None
        
        return model, card
    
    def list_models(self, family: Optional[str] = None, domain: Optional[str] = None) -> List[ModelCard]:
        """
        List all models, optionally filtered by family or domain.
        
        Args:
            family: Filter by family (optional)
            domain: Filter by domain (optional)
        
        Returns:
            List of model cards
        """
        cards = list(self.models.values())
        
        if family:
            cards = [c for c in cards if c.family == family]
        
        if domain:
            cards = [c for c in cards if c.domain == domain]
        
        return cards
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            model_ids: List of model IDs to compare
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = []
        
        for model_id in model_ids:
            if model_id in self.models:
                card = self.models[model_id]
                comparison.append({
                    'Model ID': card.model_id,
                    'Name': card.name,
                    'Version': card.version,
                    'Family': card.family,
                    'Domain': card.domain,
                    'MAE': f"{card.mae:.4f}",
                    'R²': f"{card.r2:.4f}",
                    'RMSE': f"{card.rmse:.4f}",
                    'Speed (ms)': f"{card.inference_speed_ms:.2f}",
                    'Training Size': card.training_data_size,
                    'Coverage': card.coverage,
                    'Bandgap Range': f"{card.bandgap_range[0]:.2f}-{card.bandgap_range[1]:.2f} eV"
                })
        
        return pd.DataFrame(comparison)
    
    def update_version(self, model_id: str, new_version: str, changes: str):
        """
        Update model version with changelog entry.
        
        Args:
            model_id: Model to update
            new_version: New version string
            changes: Description of changes
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        card = self.models[model_id]
        card.version = new_version
        card.changelog.append({
            'version': new_version,
            'date': datetime.now().isoformat(),
            'changes': changes
        })
        
        self._save_registry()
    
    def export_model(self, model_id: str, export_path: str):
        """
        Export model and card to external location.
        
        Args:
            model_id: Model to export
            export_path: Destination directory
        """
        result = self.get_model(model_id)
        if not result:
            raise ValueError(f"Model {model_id} not found")
        
        model, card = result
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, export_dir / f"{model_id}.joblib")
        
        # Save card
        with open(export_dir / f"{model_id}_card.json", 'w') as f:
            json.dump(card.to_dict(), f, indent=2)
    
    def import_model(self, import_path: str) -> ModelCard:
        """
        Import model from external location.
        
        Args:
            import_path: Path to model directory (must contain .joblib and _card.json)
        
        Returns:
            Imported model card
        """
        import_dir = Path(import_path)
        
        # Find model files
        joblib_files = list(import_dir.glob("*.joblib"))
        if not joblib_files:
            raise ValueError(f"No .joblib file found in {import_path}")
        
        model_file = joblib_files[0]
        model_id = model_file.stem
        card_file = import_dir / f"{model_id}_card.json"
        
        if not card_file.exists():
            raise ValueError(f"Model card not found: {card_file}")
        
        # Load card
        with open(card_file, 'r') as f:
            card_data = json.load(f)
        
        card = ModelCard.from_dict(card_data)
        
        # Load model
        model = joblib.load(model_file)
        
        # Copy to registry
        new_path = self.registry_dir / f"{model_id}.joblib"
        joblib.dump(model, new_path)
        card.model_path = str(new_path)
        
        # Register
        self.models[model_id] = card
        self.artifacts[model_id] = model
        
        self._save_registry()
        
        return card
    
    def _save_registry(self):
        """Save registry index to disk."""
        index_path = self.registry_dir / 'registry_index.json'
        
        index = {
            model_id: card.to_dict()
            for model_id, card in self.models.items()
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _load_registry(self):
        """Load registry index from disk."""
        index_path = self.registry_dir / 'registry_index.json'
        
        if not index_path.exists():
            return
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        for model_id, card_data in index.items():
            self.models[model_id] = ModelCard.from_dict(card_data)


def create_sample_models(registry: ModelRegistry, base_model: BaseEstimator, training_data: pd.DataFrame) -> List[str]:
    """
    Create sample models for demo purposes.
    
    Args:
        registry: Model registry
        base_model: Base trained model
        training_data: Training dataset
    
    Returns:
        List of created model IDs
    """
    
    models_created = []
    
    # Base model
    model_id = 'halide-base-v1'
    try:
        card = registry.register_model(
            model=base_model,
            model_id=model_id,
            name='Halide Perovskite Base Model',
            version='1.0.0',
            family='base',
            training_data=training_data,
            features_used=['feature_1', 'feature_2', 'feature_3'],
            target_property='bandgap',
            metrics={'mae': 0.15, 'r2': 0.85, 'rmse': 0.20, 'inference_speed_ms': 5.2},
            domain='halide_perovskites',
            author='AlphaMaterials Team',
            description='Base model trained on 500 halide perovskite compositions'
        )
        models_created.append(model_id)
    except:
        pass
    
    return models_created
