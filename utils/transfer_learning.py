"""
Transfer Learning Across Materials Domains
===========================================

Learn from one material family, transfer knowledge to another.

Domains:
1. Halide perovskites (ABX3, X = I/Br/Cl)
2. Oxide perovskites (ABO3, B = Ti/Zr/Hf, O = oxygen)
3. Chalcogenides (A2BX4, X = S/Se/Te)

Features:
- Pre-trained models for each domain
- Shared feature representations
- Cross-domain insights ("patterns from oxides suggest X in halides")
- Domain adaptation via feature alignment

Author: OpenClaw Agent
Date: 2026-03-15 (V7)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_models import CompositionFeaturizer


class DomainKnowledgeBase:
    """
    Pre-trained models and statistics for different material domains.
    """
    
    DOMAINS = {
        'halide_perovskites': {
            'formula_pattern': r'(MA|FA|Cs|Rb)(Pb|Sn|Ge)(I|Br|Cl)3',
            'typical_bandgap_range': (1.2, 2.5),
            'key_features': ['ionic_radius_A', 'electronegativity_B', 'ionic_radius_X'],
            'sample_data': [
                {'formula': 'MAPbI3', 'bandgap': 1.59},
                {'formula': 'FAPbI3', 'bandgap': 1.51},
                {'formula': 'CsPbI3', 'bandgap': 1.72},
                {'formula': 'MAPbBr3', 'bandgap': 2.30},
                {'formula': 'FAPbBr3', 'bandgap': 2.25},
            ]
        },
        
        'oxide_perovskites': {
            'formula_pattern': r'(Sr|Ba|Ca)(Ti|Zr|Hf)O3',
            'typical_bandgap_range': (3.0, 5.5),
            'key_features': ['ionic_radius_A', 'ionic_radius_B', 'd_electrons_B'],
            'sample_data': [
                {'formula': 'SrTiO3', 'bandgap': 3.25},
                {'formula': 'BaTiO3', 'bandgap': 3.20},
                {'formula': 'CaTiO3', 'bandgap': 3.60},
                {'formula': 'SrZrO3', 'bandgap': 5.00},
                {'formula': 'BaZrO3', 'bandgap': 4.90},
            ]
        },
        
        'chalcogenides': {
            'formula_pattern': r'(Cu|Ag|Zn)2(Zn|Sn|Ge)(S|Se|Te)4',
            'typical_bandgap_range': (1.0, 2.5),
            'key_features': ['electronegativity_A', 'ionic_radius_B', 'electronegativity_X'],
            'sample_data': [
                {'formula': 'Cu2ZnSnS4', 'bandgap': 1.50},
                {'formula': 'Cu2ZnSnSe4', 'bandgap': 1.00},
                {'formula': 'Cu2ZnGeSe4', 'bandgap': 1.40},
                {'formula': 'Ag2ZnSnSe4', 'bandgap': 1.35},
                {'formula': 'Cu2ZnSnTe4', 'bandgap': 0.95},
            ]
        }
    }


class TransferLearningEngine:
    """
    Transfer learning across material domains.
    
    Strategy:
    1. Train base models on each domain
    2. Extract shared feature representations
    3. Adapt model from source domain to target domain
    4. Generate cross-domain insights
    """
    
    def __init__(self):
        """Initialize transfer learning engine."""
        self.featurizer = CompositionFeaturizer()
        self.domain_models = {}
        self.domain_scalers = {}
        self.knowledge_base = DomainKnowledgeBase()
        
        # Pre-train models on sample data
        self._pretrain_domain_models()
    
    def _pretrain_domain_models(self):
        """Pre-train models on built-in sample data for each domain."""
        
        for domain_name, domain_info in self.knowledge_base.DOMAINS.items():
            # Create sample DataFrame
            df = pd.DataFrame(domain_info['sample_data'])
            
            if len(df) < 3:
                continue  # Not enough data to train
            
            # Featurize
            formulas = df['formula'].tolist()
            bandgaps = df['bandgap'].values
            
            try:
                # Batch featurize (featurizer.featurize is single, so we loop)
                features = np.array([self.featurizer.featurize(f) for f in formulas])
                
                # Train small model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
                
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X_scaled, bandgaps)
                
                self.domain_models[domain_name] = model
                self.domain_scalers[domain_name] = scaler
                
            except Exception as e:
                print(f"Warning: Failed to pre-train {domain_name} model: {e}")
    
    def fine_tune_domain(self, 
                        domain: str,
                        training_data: pd.DataFrame,
                        formula_col: str = 'formula',
                        target_col: str = 'bandgap') -> Dict:
        """
        Fine-tune pre-trained model on user data for specific domain.
        
        Args:
            domain: Domain name ('halide_perovskites', 'oxide_perovskites', 'chalcogenides')
            training_data: User training data
            formula_col: Column name for formulas
            target_col: Column name for target property
        
        Returns:
            Training metrics dict
        """
        
        if domain not in self.knowledge_base.DOMAINS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.knowledge_base.DOMAINS.keys())}")
        
        # Featurize
        formulas = training_data[formula_col].tolist()
        targets = training_data[target_col].values
        
        # Batch featurize
        features = np.array([self.featurizer.featurize(f) for f in formulas])
        
        # Scale
        if domain in self.domain_scalers:
            scaler = self.domain_scalers[domain]
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(features)
        
        # Train new model (or fine-tune existing)
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        
        model.fit(X_scaled, targets)
        
        # Update stored models
        self.domain_models[domain] = model
        self.domain_scalers[domain] = scaler
        
        # Calculate metrics
        predictions = model.predict(X_scaled)
        mae = np.mean(np.abs(predictions - targets))
        r2 = model.score(X_scaled, targets)
        
        return {
            'domain': domain,
            'n_samples': len(training_data),
            'mae': float(mae),
            'r2': float(r2),
            'feature_importances': model.feature_importances_.tolist()
        }
    
    def transfer_knowledge(self,
                          source_domain: str,
                          target_domain: str,
                          target_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Transfer knowledge from source domain to target domain.
        
        Strategy:
        1. Use source model's feature importances
        2. Adapt to target domain via feature alignment
        3. Generate cross-domain insights
        
        Args:
            source_domain: Source domain (trained)
            target_domain: Target domain (to adapt)
            target_data: Optional target domain data for adaptation
        
        Returns:
            Transfer metrics dict
        """
        
        if source_domain not in self.domain_models:
            raise ValueError(f"Source domain {source_domain} not trained. Call fine_tune_domain first.")
        
        source_model = self.domain_models[source_domain]
        
        # Extract transferable knowledge
        feature_importances = source_model.feature_importances_
        
        # Key features from source domain
        top_features_idx = np.argsort(feature_importances)[-5:][::-1]
        
        # Map to feature names (simplified)
        feature_names = self.featurizer.feature_names if hasattr(self.featurizer, 'feature_names') else [f"feat_{i}" for i in range(len(feature_importances))]
        key_features = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_features_idx]
        
        # Cross-domain insights
        insights = self._generate_cross_domain_insights(
            source_domain, target_domain, key_features
        )
        
        # If target data provided, adapt model
        if target_data is not None and len(target_data) > 0:
            # Fine-tune on target domain
            target_metrics = self.fine_tune_domain(target_domain, target_data)
        else:
            target_metrics = {'status': 'No target data provided, using pre-trained model'}
        
        return {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'key_features': key_features,
            'feature_importances': feature_importances[top_features_idx].tolist(),
            'insights': insights,
            'target_metrics': target_metrics
        }
    
    def _generate_cross_domain_insights(self,
                                       source: str,
                                       target: str,
                                       key_features: List[str]) -> List[str]:
        """
        Generate human-readable cross-domain insights.
        
        Args:
            source: Source domain
            target: Target domain
            key_features: Important features from source
        
        Returns:
            List of insight strings
        """
        
        insights = []
        
        # Domain-specific insights
        if source == 'halide_perovskites' and target == 'oxide_perovskites':
            insights.append(
                f"Halide perovskites show strong dependence on {key_features[0]}. "
                f"In oxide perovskites, consider exploring similar A-site cation substitutions (Sr/Ba/Ca)."
            )
            insights.append(
                "Halides have lower bandgaps (1.2-2.5 eV) than oxides (3.0-5.5 eV). "
                "Use oxide insights for UV applications, halide insights for visible-NIR."
            )
        
        elif source == 'oxide_perovskites' and target == 'halide_perovskites':
            insights.append(
                f"Oxide perovskites are dominated by B-site chemistry. "
                f"Transfer this to halides: Pb/Sn substitution on B-site strongly affects bandgap."
            )
            insights.append(
                "Oxides are stable but insulating. Halides are less stable but have ideal bandgaps for PV. "
                "Consider hybrid structures: oxide scaffold + halide active layer."
            )
        
        elif source == 'halide_perovskites' and target == 'chalcogenides':
            insights.append(
                f"Halides show {key_features[0]} is critical. "
                f"In chalcogenides (Cu2ZnSnS4 family), explore similar compositional tuning on A-site (Cu/Ag)."
            )
            insights.append(
                "Both halides and chalcogenides are tunable in 1.0-2.5 eV range. "
                "Halides: easier synthesis, lower stability. Chalcogenides: harder synthesis, better stability."
            )
        
        elif source == 'chalcogenides' and target == 'halide_perovskites':
            insights.append(
                f"Chalcogenides demonstrate importance of {key_features[0]}. "
                f"Apply to halides: X-site engineering (I/Br/Cl mixing) for bandgap tuning."
            )
            insights.append(
                "Chalcogenides (CZTS) are Pb-free and earth-abundant. "
                "If halides face Pb regulation, CZTS insights can guide alternative chemistries."
            )
        
        else:
            # Generic insight
            insights.append(
                f"Key features from {source}: {', '.join(key_features[:3])}. "
                f"These may also govern {target} properties. Explore analogous compositions."
            )
        
        return insights
    
    def predict_cross_domain(self,
                            formulas: List[str],
                            source_domain: str) -> Tuple[np.ndarray, List[str]]:
        """
        Predict properties using source domain model (even for out-of-domain compositions).
        
        Useful for "what if we applied halide model to oxide compositions?"
        
        Args:
            formulas: List of compositions
            source_domain: Domain model to use
        
        Returns:
            (predictions, warnings)
        """
        
        if source_domain not in self.domain_models:
            raise ValueError(f"Domain {source_domain} not trained.")
        
        model = self.domain_models[source_domain]
        scaler = self.domain_scalers[source_domain]
        
        # Featurize
        features = np.array([self.featurizer.featurize(f) for f in formulas])
        X_scaled = scaler.transform(features)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Generate warnings for out-of-distribution
        warnings = []
        domain_info = self.knowledge_base.DOMAINS[source_domain]
        bg_min, bg_max = domain_info['typical_bandgap_range']
        
        for i, pred in enumerate(predictions):
            if pred < bg_min or pred > bg_max:
                warnings.append(
                    f"{formulas[i]}: Predicted {pred:.2f} eV outside typical {source_domain} range ({bg_min}-{bg_max} eV). "
                    f"Extrapolation uncertainty high."
                )
        
        return predictions, warnings
    
    def plot_domain_comparison(self, domains: List[str]) -> go.Figure:
        """
        Compare bandgap distributions across domains.
        
        Args:
            domains: List of domain names
        
        Returns:
            Plotly figure
        """
        
        fig = go.Figure()
        
        for domain in domains:
            if domain in self.knowledge_base.DOMAINS:
                domain_info = self.knowledge_base.DOMAINS[domain]
                sample_data = domain_info['sample_data']
                
                bandgaps = [d['bandgap'] for d in sample_data]
                formulas = [d['formula'] for d in sample_data]
                
                fig.add_trace(go.Box(
                    y=bandgaps,
                    name=domain.replace('_', ' ').title(),
                    text=formulas,
                    boxmean='sd'
                ))
        
        fig.update_layout(
            title='Bandgap Distribution Across Domains',
            yaxis_title='Bandgap (eV)',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def plot_feature_importance_transfer(self,
                                        source_domain: str,
                                        target_domain: str) -> go.Figure:
        """
        Visualize feature importance transfer between domains.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
        
        Returns:
            Plotly figure
        """
        
        if source_domain not in self.domain_models or target_domain not in self.domain_models:
            raise ValueError("Both domains must be trained first.")
        
        source_importances = self.domain_models[source_domain].feature_importances_
        target_importances = self.domain_models[target_domain].feature_importances_
        
        # Top 10 features
        n_features = min(10, len(source_importances))
        indices = np.argsort(source_importances)[-n_features:][::-1]
        
        feature_names = [f"Feature {i}" for i in indices]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_names,
            y=source_importances[indices],
            name=source_domain.replace('_', ' ').title(),
            marker_color='cyan'
        ))
        
        fig.add_trace(go.Bar(
            x=feature_names,
            y=target_importances[indices],
            name=target_domain.replace('_', ' ').title(),
            marker_color='orange'
        ))
        
        fig.update_layout(
            title=f'Feature Importance: {source_domain} → {target_domain}',
            xaxis_title='Feature',
            yaxis_title='Importance',
            barmode='group',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            height=500
        )
        
        return fig


class DomainSelector:
    """
    Helper class to select and switch between domains.
    """
    
    @staticmethod
    def detect_domain(formula: str) -> str:
        """
        Auto-detect domain from formula pattern.
        
        Args:
            formula: Chemical formula
        
        Returns:
            Domain name
        """
        
        import re
        
        for domain_name, domain_info in DomainKnowledgeBase.DOMAINS.items():
            pattern = domain_info['formula_pattern']
            if re.match(pattern, formula):
                return domain_name
        
        return 'unknown'
    
    @staticmethod
    def get_domain_info(domain: str) -> Dict:
        """Get information about a domain."""
        return DomainKnowledgeBase.DOMAINS.get(domain, {})
