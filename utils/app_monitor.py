#!/usr/bin/env python3
"""
Application Performance Monitor
================================

Track app performance and health metrics:
- Load time, prediction latency, memory usage
- Data quality indicators
- Model health (accuracy drift, retraining alerts)
- Usage analytics (feature popularity)

Part of AlphaMaterials V11
"""

import time
import psutil
import platform
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    threshold: Optional[float] = None  # Warning threshold
    
    @property
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range"""
        if self.threshold is None:
            return True
        return self.value <= self.threshold


@dataclass
class DataQuality:
    """Data quality indicators"""
    total_records: int
    completeness: float  # 0-1, fraction of non-null values
    freshness_days: int  # Days since last update
    coverage: float  # 0-1, fraction of feature space covered
    duplicates: int
    outliers: int
    
    @property
    def quality_score(self) -> float:
        """Overall quality score 0-100"""
        score = (
            self.completeness * 40 +  # 40% weight
            max(0, 1 - self.freshness_days / 365) * 20 +  # 20% weight
            self.coverage * 30 +  # 30% weight
            max(0, 1 - self.duplicates / max(1, self.total_records)) * 5 +  # 5% weight
            max(0, 1 - self.outliers / max(1, self.total_records)) * 5  # 5% weight
        )
        return min(100, max(0, score * 100))


@dataclass
class ModelHealth:
    """Model performance health"""
    model_name: str
    accuracy: float  # R² or similar
    baseline_accuracy: float  # Initial accuracy
    accuracy_drift: float  # Current - baseline
    predictions_count: int
    avg_prediction_time: float  # milliseconds
    last_training: datetime
    training_data_size: int
    
    @property
    def needs_retraining(self) -> bool:
        """Check if model needs retraining"""
        # Retrain if accuracy dropped >10% or no training in 30 days
        drift_threshold = -0.10
        time_threshold = timedelta(days=30)
        
        if self.accuracy_drift < drift_threshold:
            return True
        
        if datetime.now() - self.last_training > time_threshold:
            return True
        
        return False
    
    @property
    def health_status(self) -> str:
        """Health status string"""
        if self.needs_retraining:
            return "🔴 Needs Retraining"
        elif abs(self.accuracy_drift) > 0.05:
            return "🟡 Monitor"
        else:
            return "🟢 Healthy"


@dataclass
class FeatureUsage:
    """Feature/tab usage statistics"""
    feature_name: str
    visits: int = 0
    total_time: float = 0  # seconds
    last_visit: Optional[datetime] = None
    user_ratings: List[int] = field(default_factory=list)  # 1-5 stars
    
    @property
    def avg_time_per_visit(self) -> float:
        """Average time spent per visit (seconds)"""
        if self.visits == 0:
            return 0
        return self.total_time / self.visits
    
    @property
    def avg_rating(self) -> Optional[float]:
        """Average user rating"""
        if not self.user_ratings:
            return None
        return sum(self.user_ratings) / len(self.user_ratings)


class AppMonitor:
    """
    Application performance monitor
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Performance metrics history
        self.load_times: deque = deque(maxlen=history_size)
        self.prediction_times: deque = deque(maxlen=history_size)
        self.memory_usage: deque = deque(maxlen=history_size)
        
        # Data quality
        self.data_quality: Optional[DataQuality] = None
        
        # Model health
        self.models: Dict[str, ModelHealth] = {}
        
        # Feature usage
        self.features: Dict[str, FeatureUsage] = {}
        
        # System info
        self.system_info = self._get_system_info()
        
        # Session start
        self.session_start = datetime.now()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "hostname": platform.node()
        }
    
    def record_load_time(self, component: str, duration: float):
        """Record component load time"""
        metric = PerformanceMetric(
            name=f"load_time_{component}",
            value=duration,
            unit="seconds",
            threshold=5.0  # 5 second threshold
        )
        self.load_times.append(metric)
    
    def record_prediction_time(self, duration: float):
        """Record prediction latency"""
        metric = PerformanceMetric(
            name="prediction_latency",
            value=duration,
            unit="milliseconds",
            threshold=1000.0  # 1 second threshold
        )
        self.prediction_times.append(metric)
    
    def record_memory_usage(self):
        """Record current memory usage"""
        mem = psutil.virtual_memory()
        metric = PerformanceMetric(
            name="memory_usage",
            value=mem.percent,
            unit="percent",
            threshold=85.0  # 85% threshold
        )
        self.memory_usage.append(metric)
    
    def update_data_quality(
        self,
        total_records: int,
        completeness: float,
        freshness_days: int,
        coverage: float,
        duplicates: int = 0,
        outliers: int = 0
    ):
        """Update data quality metrics"""
        self.data_quality = DataQuality(
            total_records=total_records,
            completeness=completeness,
            freshness_days=freshness_days,
            coverage=coverage,
            duplicates=duplicates,
            outliers=outliers
        )
    
    def register_model(
        self,
        model_name: str,
        accuracy: float,
        training_data_size: int
    ):
        """Register a new model"""
        self.models[model_name] = ModelHealth(
            model_name=model_name,
            accuracy=accuracy,
            baseline_accuracy=accuracy,
            accuracy_drift=0.0,
            predictions_count=0,
            avg_prediction_time=0.0,
            last_training=datetime.now(),
            training_data_size=training_data_size
        )
    
    def update_model_accuracy(self, model_name: str, new_accuracy: float):
        """Update model accuracy (for drift detection)"""
        if model_name in self.models:
            model = self.models[model_name]
            model.accuracy = new_accuracy
            model.accuracy_drift = new_accuracy - model.baseline_accuracy
    
    def record_prediction(self, model_name: str, duration_ms: float):
        """Record a prediction event"""
        if model_name in self.models:
            model = self.models[model_name]
            model.predictions_count += 1
            
            # Update average prediction time (moving average)
            alpha = 0.1  # Smoothing factor
            model.avg_prediction_time = (
                alpha * duration_ms + (1 - alpha) * model.avg_prediction_time
            )
    
    def track_feature_visit(self, feature_name: str, duration: float = 0):
        """Track feature/tab visit"""
        if feature_name not in self.features:
            self.features[feature_name] = FeatureUsage(feature_name=feature_name)
        
        feature = self.features[feature_name]
        feature.visits += 1
        feature.total_time += duration
        feature.last_visit = datetime.now()
    
    def rate_feature(self, feature_name: str, rating: int):
        """Add user rating for feature (1-5 stars)"""
        if feature_name not in self.features:
            self.features[feature_name] = FeatureUsage(feature_name=feature_name)
        
        if 1 <= rating <= 5:
            self.features[feature_name].user_ratings.append(rating)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        summary = {
            "session_duration": (datetime.now() - self.session_start).total_seconds() / 60,  # minutes
            "system": self.system_info
        }
        
        # Load times
        if self.load_times:
            load_values = [m.value for m in self.load_times]
            summary["load_time"] = {
                "avg": sum(load_values) / len(load_values),
                "max": max(load_values),
                "min": min(load_values),
                "unit": "seconds",
                "healthy": all(m.is_healthy for m in self.load_times)
            }
        
        # Prediction times
        if self.prediction_times:
            pred_values = [m.value for m in self.prediction_times]
            summary["prediction_latency"] = {
                "avg": sum(pred_values) / len(pred_values),
                "max": max(pred_values),
                "min": min(pred_values),
                "unit": "milliseconds",
                "healthy": all(m.is_healthy for m in self.prediction_times)
            }
        
        # Memory usage
        if self.memory_usage:
            mem_values = [m.value for m in self.memory_usage]
            summary["memory"] = {
                "current": mem_values[-1],
                "avg": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "unit": "percent",
                "healthy": all(m.is_healthy for m in self.memory_usage)
            }
        else:
            # Get current memory if no history
            mem = psutil.virtual_memory()
            summary["memory"] = {
                "current": mem.percent,
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "unit": "percent"
            }
        
        return summary
    
    def get_data_quality_summary(self) -> Optional[Dict[str, Any]]:
        """Get data quality summary"""
        if not self.data_quality:
            return None
        
        dq = self.data_quality
        
        return {
            "records": dq.total_records,
            "completeness": f"{dq.completeness * 100:.1f}%",
            "freshness": f"{dq.freshness_days} days ago",
            "coverage": f"{dq.coverage * 100:.1f}%",
            "duplicates": dq.duplicates,
            "outliers": dq.outliers,
            "quality_score": f"{dq.quality_score:.1f}/100",
            "status": "🟢 Good" if dq.quality_score >= 80 else "🟡 Fair" if dq.quality_score >= 60 else "🔴 Poor"
        }
    
    def get_model_health_summary(self) -> Dict[str, Any]:
        """Get all models health summary"""
        if not self.models:
            return {"models": [], "alerts": []}
        
        models_summary = []
        alerts = []
        
        for name, model in self.models.items():
            models_summary.append({
                "name": name,
                "accuracy": f"{model.accuracy:.3f}",
                "drift": f"{model.accuracy_drift:+.3f}",
                "predictions": model.predictions_count,
                "avg_latency_ms": f"{model.avg_prediction_time:.1f}",
                "status": model.health_status,
                "needs_retraining": model.needs_retraining
            })
            
            if model.needs_retraining:
                alerts.append(f"⚠️ {name}: Retraining recommended (drift: {model.accuracy_drift:+.3f})")
        
        return {
            "models": models_summary,
            "alerts": alerts
        }
    
    def get_feature_analytics(self) -> Dict[str, Any]:
        """Get feature usage analytics"""
        if not self.features:
            return {"features": [], "insights": []}
        
        # Sort by visits
        sorted_features = sorted(
            self.features.values(),
            key=lambda f: f.visits,
            reverse=True
        )
        
        features_summary = []
        for feature in sorted_features:
            features_summary.append({
                "name": feature.feature_name,
                "visits": feature.visits,
                "avg_time": f"{feature.avg_time_per_visit:.1f}s",
                "rating": f"{feature.avg_rating:.1f}⭐" if feature.avg_rating else "N/A",
                "last_visit": feature.last_visit.strftime("%Y-%m-%d %H:%M") if feature.last_visit else "Never"
            })
        
        # Insights
        insights = []
        
        # Most popular
        if sorted_features:
            most_popular = sorted_features[0]
            insights.append(f"📊 Most visited: {most_popular.feature_name} ({most_popular.visits} visits)")
        
        # Least popular (excluding never visited)
        visited_features = [f for f in sorted_features if f.visits > 0]
        if len(visited_features) > 1:
            least_popular = visited_features[-1]
            insights.append(f"📉 Least visited: {least_popular.feature_name} ({least_popular.visits} visits)")
        
        # Best rated
        rated_features = [f for f in sorted_features if f.avg_rating is not None]
        if rated_features:
            best_rated = max(rated_features, key=lambda f: f.avg_rating)
            insights.append(f"⭐ Highest rated: {best_rated.feature_name} ({best_rated.avg_rating:.1f}/5)")
        
        return {
            "features": features_summary,
            "insights": insights
        }
    
    def export_report(self, filepath: str):
        """Export monitoring report to JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "performance": self.get_performance_summary(),
            "data_quality": self.get_data_quality_summary(),
            "model_health": self.get_model_health_summary(),
            "feature_analytics": self.get_feature_analytics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def demonstrate_monitor():
    """Demonstrate app monitor"""
    print("📊 AlphaMaterials App Monitor Demo\n")
    
    monitor = AppMonitor()
    
    # Simulate some activity
    print("🔄 Simulating app activity...\n")
    
    # Load times
    monitor.record_load_time("database", 2.3)
    monitor.record_load_time("ml_model", 4.1)
    monitor.record_load_time("dashboard", 1.5)
    
    # Predictions
    monitor.register_model("RandomForest", accuracy=0.89, training_data_size=1000)
    for i in range(10):
        monitor.record_prediction("RandomForest", duration_ms=45.0 + i * 2)
    
    # Simulate accuracy drift
    time.sleep(0.1)
    monitor.update_model_accuracy("RandomForest", 0.82)  # Dropped
    
    # Data quality
    monitor.update_data_quality(
        total_records=1500,
        completeness=0.95,
        freshness_days=7,
        coverage=0.75,
        duplicates=12,
        outliers=8
    )
    
    # Feature usage
    tabs = ["Database", "ML Model", "Bayesian Opt", "Inverse Design", "Natural Language"]
    visits = [15, 12, 8, 3, 2]
    times = [120, 200, 350, 180, 60]
    
    for tab, visit_count, total_time in zip(tabs, visits, times):
        for _ in range(visit_count):
            monitor.track_feature_visit(tab, duration=total_time / visit_count)
    
    # Ratings
    monitor.rate_feature("Bayesian Opt", 5)
    monitor.rate_feature("Bayesian Opt", 5)
    monitor.rate_feature("Natural Language", 4)
    
    # Memory
    monitor.record_memory_usage()
    
    # Print summaries
    print("⚡ Performance Summary:")
    perf = monitor.get_performance_summary()
    if "load_time" in perf:
        print(f"  Load Time: {perf['load_time']['avg']:.2f}s avg")
    if "prediction_latency" in perf:
        print(f"  Prediction: {perf['prediction_latency']['avg']:.1f}ms avg")
    if "memory" in perf:
        print(f"  Memory: {perf['memory']['current']:.1f}%")
    
    print(f"\n📊 Data Quality:")
    dq = monitor.get_data_quality_summary()
    if dq:
        print(f"  Records: {dq['records']}")
        print(f"  Completeness: {dq['completeness']}")
        print(f"  Quality Score: {dq['quality_score']}")
        print(f"  Status: {dq['status']}")
    
    print(f"\n🤖 Model Health:")
    models = monitor.get_model_health_summary()
    for model in models['models']:
        print(f"  {model['name']}: {model['status']}")
        print(f"    Accuracy: {model['accuracy']} (drift: {model['drift']})")
        print(f"    Predictions: {model['predictions']}, Latency: {model['avg_latency_ms']}ms")
    
    if models['alerts']:
        print(f"\n  Alerts:")
        for alert in models['alerts']:
            print(f"    {alert}")
    
    print(f"\n📈 Feature Analytics:")
    analytics = monitor.get_feature_analytics()
    print(f"  Top Features:")
    for feat in analytics['features'][:3]:
        print(f"    {feat['name']}: {feat['visits']} visits, {feat['avg_time']} avg, {feat['rating']} rating")
    
    print(f"\n  Insights:")
    for insight in analytics['insights']:
        print(f"    {insight}")


if __name__ == "__main__":
    demonstrate_monitor()
