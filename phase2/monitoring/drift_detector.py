#!/usr/bin/env python3
"""
Production Monitoring: Drift Detection & Retraining Triggers
=============================================================

Monitor deployed IDS model for concept drift and trigger retraining when needed.

Components:
1. DriftDetector - Detect accuracy degradation over time
2. ABTestingFramework - Compare old vs new models in production
3. RetrainingScheduler - Manage retraining triggers

Usage:
    # Monitor live predictions
    from phase2.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(baseline_f1=94.0)
    
    # Check for drift
    status = detector.update(y_true, y_pred)
    if status == 'RETRAIN_NEEDED':
        trigger_retraining()
"""

import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    alert_type: str  # 'warning', 'critical', 'retrain'
    metric: str
    current_value: float
    baseline_value: float
    threshold: float
    drop: float
    timestamp: str
    recommendation: str


class DriftDetector:
    """
    Detect concept drift by monitoring F1 score over time.
    
    Uses sliding window to compute recent performance and compares
    against baseline to detect degradation.
    
    Reference: Webb et al., "Characterizing Concept Drift", Data Mining and Knowledge Discovery, 2016
    """
    
    def __init__(
        self,
        baseline_f1: float = 94.0,
        warning_threshold: float = 1.5,
        critical_threshold: float = 3.0,
        retrain_threshold: float = 5.0,
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Args:
            baseline_f1: Expected F1 score from training/validation
            warning_threshold: F1 drop % that triggers warning
            critical_threshold: F1 drop % that triggers critical alert
            retrain_threshold: F1 drop % that triggers retraining
            window_size: Samples to keep in sliding window
            min_samples: Minimum samples before checking drift
        """
        self.baseline_f1 = baseline_f1
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.retrain_threshold = retrain_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Sliding window for recent predictions
        self.y_true_window: Deque[int] = deque(maxlen=window_size)
        self.y_pred_window: Deque[int] = deque(maxlen=window_size)
        
        # History for analysis
        self.f1_history: List[float] = []
        self.alerts: List[DriftAlert] = []
        self.last_check_time = time.time()
        self.total_samples = 0
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Update with new predictions and check for drift.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
        
        Returns:
            Status: 'OK', 'WARNING', 'CRITICAL', or 'RETRAIN_NEEDED'
        """
        # Add to window
        for yt, yp in zip(y_true.flatten(), y_pred.flatten()):
            self.y_true_window.append(int(yt))
            self.y_pred_window.append(int(yp))
            self.total_samples += 1
        
        # Check if enough samples
        if len(self.y_true_window) < self.min_samples:
            return 'OK'
        
        # Compute current F1
        current_f1 = float(f1_score(
            list(self.y_true_window),
            list(self.y_pred_window),
            average='macro',
            zero_division=0
        ) * 100)
        
        self.f1_history.append(current_f1)
        
        # Compute drop
        drop = float(self.baseline_f1 - current_f1)
        
        # Determine status
        if drop >= self.retrain_threshold:
            status = 'RETRAIN_NEEDED'
            self._add_alert('retrain', 'f1_macro', current_f1, drop)
        elif drop >= self.critical_threshold:
            status = 'CRITICAL'
            self._add_alert('critical', 'f1_macro', current_f1, drop)
        elif drop >= self.warning_threshold:
            status = 'WARNING'
            self._add_alert('warning', 'f1_macro', current_f1, drop)
        else:
            status = 'OK'
        
        return status
    
    def _add_alert(self, alert_type: str, metric: str, current_value: float, drop: float):
        """Add alert to history."""
        recommendations = {
            'warning': "Monitor closely. Consider collecting more labeled data.",
            'critical': "Investigate root cause. Prepare for potential retraining.",
            'retrain': "Initiate retraining pipeline immediately."
        }
        
        alert = DriftAlert(
            alert_type=alert_type,
            metric=metric,
            current_value=current_value,
            baseline_value=self.baseline_f1,
            threshold=getattr(self, f'{alert_type}_threshold'),
            drop=drop,
            timestamp=datetime.now().isoformat(),
            recommendation=recommendations.get(alert_type, "")
        )
        
        self.alerts.append(alert)
    
    def get_status(self) -> Dict:
        """Get current drift detection status."""
        if len(self.y_true_window) < self.min_samples:
            return {
                'status': 'INITIALIZING',
                'samples_collected': len(self.y_true_window),
                'min_samples_needed': self.min_samples
            }
        
        current_f1 = f1_score(
            list(self.y_true_window),
            list(self.y_pred_window),
            average='macro',
            zero_division=0
        ) * 100
        
        return {
            'status': self.update(np.array([]), np.array([])),  # Get status without adding
            'baseline_f1': self.baseline_f1,
            'current_f1': current_f1,
            'drop': self.baseline_f1 - current_f1,
            'total_samples': self.total_samples,
            'window_size': len(self.y_true_window),
            'n_alerts': len(self.alerts),
            'last_alert': asdict(self.alerts[-1]) if self.alerts else None
        }
    
    def get_f1_trend(self, n_points: int = 10) -> List[float]:
        """Get recent F1 trend for visualization."""
        if len(self.f1_history) <= n_points:
            return self.f1_history
        
        # Downsample to n_points
        step = len(self.f1_history) // n_points
        return self.f1_history[::step][-n_points:]
    
    def save_state(self, path: Path):
        """Save detector state to file."""
        state = {
            'baseline_f1': self.baseline_f1,
            'total_samples': self.total_samples,
            'f1_history': self.f1_history[-1000:],  # Keep last 1000
            'alerts': [asdict(a) for a in self.alerts[-100:]],  # Keep last 100
            'timestamp': datetime.now().isoformat()
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Path):
        """Load detector state from file."""
        with open(path) as f:
            state = json.load(f)
        
        self.total_samples = state.get('total_samples', 0)
        self.f1_history = state.get('f1_history', [])


class PerClassDriftDetector(DriftDetector):
    """
    Extended drift detector that monitors per-attack-class performance.
    
    Critical for detecting degradation in specific attack types
    (e.g., Bot, SSH-Patator) which may degrade faster than overall metrics.
    """
    
    def __init__(
        self,
        baseline_per_class: Dict[str, float],
        critical_attacks: List[str] = ['DDoS', 'PortScan', 'Bot', 'SSH-Patator'],
        **kwargs
    ):
        """
        Args:
            baseline_per_class: Dict mapping attack type -> baseline detection rate
            critical_attacks: Attack types with stricter monitoring
        """
        super().__init__(**kwargs)
        self.baseline_per_class = baseline_per_class
        self.critical_attacks = critical_attacks
        
        # Per-class windows
        self.per_class_true: Dict[str, Deque[int]] = {}
        self.per_class_pred: Dict[str, Deque[int]] = {}
        
        for attack in baseline_per_class.keys():
            self.per_class_true[attack] = deque(maxlen=self.window_size)
            self.per_class_pred[attack] = deque(maxlen=self.window_size)
    
    def update_with_attack_types(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        attack_types: np.ndarray
    ) -> Tuple[str, Dict[str, str]]:
        """
        Update with attack type information.
        
        Returns:
            Tuple of (overall_status, per_class_status_dict)
        """
        # Update overall
        overall_status = super().update(y_true, y_pred)
        
        # Update per-class
        per_class_status = {}
        
        for attack in np.unique(attack_types):
            if attack == 'BENIGN' or attack not in self.baseline_per_class:
                continue
            
            mask = attack_types == attack
            if mask.sum() == 0:
                continue
            
            # Add to window
            for yt, yp in zip(y_true[mask], y_pred[mask]):
                self.per_class_true[attack].append(int(yt))
                self.per_class_pred[attack].append(int(yp))
            
            # Check if enough samples
            if len(self.per_class_true[attack]) < 50:
                per_class_status[attack] = 'INITIALIZING'
                continue
            
            # Compute current detection rate
            tp = sum((t == 1 and p == 1) for t, p in 
                     zip(self.per_class_true[attack], self.per_class_pred[attack]))
            total_positive = sum(t == 1 for t in self.per_class_true[attack])
            
            current_dr = (tp / total_positive * 100) if total_positive > 0 else 0
            baseline_dr = self.baseline_per_class.get(attack, 0)
            drop = baseline_dr - current_dr
            
            # Stricter threshold for critical attacks
            threshold = self.warning_threshold if attack not in self.critical_attacks else self.warning_threshold * 0.5
            
            if drop >= threshold * 2:
                per_class_status[attack] = 'CRITICAL'
            elif drop >= threshold:
                per_class_status[attack] = 'WARNING'
            else:
                per_class_status[attack] = 'OK'
        
        return overall_status, per_class_status


class ABTestingFramework:
    """
    A/B testing framework for comparing models in production.
    
    Deploy both old and new models, split traffic, and compare performance
    before fully deploying the new compressed model.
    """
    
    def __init__(
        self,
        model_a,  # Current production model
        model_b,  # New compressed model
        traffic_split: float = 0.5,
        min_samples: int = 1000
    ):
        """
        Args:
            model_a: Current production model
            model_b: New compressed model to test
            traffic_split: Fraction of traffic routed to model B
            min_samples: Minimum samples before making decision
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.min_samples = min_samples
        
        # Results tracking
        self.results_a: List[Tuple[int, int]] = []  # (y_true, y_pred)
        self.results_b: List[Tuple[int, int]] = []
        self.latencies_a: List[float] = []
        self.latencies_b: List[float] = []
    
    def route_and_predict(self, x: np.ndarray, y_true: Optional[int] = None) -> Tuple[int, str]:
        """
        Route input to model A or B and record result.
        
        Args:
            x: Input features
            y_true: Ground truth (if available, for tracking)
        
        Returns:
            Tuple of (prediction, model_used)
        """
        import random
        
        if random.random() < self.traffic_split:
            model, results_list, latencies_list = self.model_b, self.results_b, self.latencies_b
            model_name = 'B'
        else:
            model, results_list, latencies_list = self.model_a, self.results_a, self.latencies_a
            model_name = 'A'
        
        # Inference with timing
        start = time.perf_counter()
        pred = model.predict(x)
        latency = (time.perf_counter() - start) * 1000  # ms
        
        latencies_list.append(latency)
        
        if y_true is not None:
            results_list.append((y_true, pred))
        
        return pred, model_name
    
    def analyze_results(self) -> Dict:
        """Analyze A/B test results and make recommendation."""
        if len(self.results_a) < self.min_samples or len(self.results_b) < self.min_samples:
            return {
                'status': 'INSUFFICIENT_DATA',
                'samples_a': len(self.results_a),
                'samples_b': len(self.results_b),
                'min_samples': self.min_samples
            }
        
        # Compute metrics
        y_true_a, y_pred_a = zip(*self.results_a)
        y_true_b, y_pred_b = zip(*self.results_b)
        
        f1_a = f1_score(y_true_a, y_pred_a, average='macro') * 100
        f1_b = f1_score(y_true_b, y_pred_b, average='macro') * 100
        
        acc_a = accuracy_score(y_true_a, y_pred_a) * 100
        acc_b = accuracy_score(y_true_b, y_pred_b) * 100
        
        latency_a = np.mean(self.latencies_a)
        latency_b = np.mean(self.latencies_b)
        
        # Decision logic
        # Deploy B if: F1_B >= F1_A - 1% AND Latency_B < Latency_A
        f1_acceptable = f1_b >= f1_a - 1.0
        latency_better = latency_b < latency_a
        
        if f1_acceptable and latency_better:
            recommendation = 'DEPLOY_MODEL_B'
            reason = f"Model B has acceptable F1 ({f1_b:.1f}% vs {f1_a:.1f}%) and better latency ({latency_b:.2f}ms vs {latency_a:.2f}ms)"
        elif f1_acceptable:
            recommendation = 'CONSIDER_MODEL_B'
            reason = f"Model B has acceptable F1 but higher latency"
        else:
            recommendation = 'KEEP_MODEL_A'
            reason = f"Model B has unacceptable F1 drop ({f1_a - f1_b:.1f}%)"
        
        return {
            'status': 'COMPLETE',
            'model_a': {
                'f1_macro': f1_a,
                'accuracy': acc_a,
                'latency_mean_ms': latency_a,
                'n_samples': len(self.results_a)
            },
            'model_b': {
                'f1_macro': f1_b,
                'accuracy': acc_b,
                'latency_mean_ms': latency_b,
                'n_samples': len(self.results_b)
            },
            'recommendation': recommendation,
            'reason': reason
        }


class RetrainingScheduler:
    """
    Manage retraining triggers and schedules.
    
    Supports:
    1. Scheduled retraining (e.g., quarterly)
    2. Triggered retraining (when drift detected)
    3. Incremental learning (continuous updates)
    """
    
    def __init__(
        self,
        scheduled_interval_days: int = 90,  # Quarterly
        drift_detector: Optional[DriftDetector] = None,
        output_dir: Path = Path("retraining_logs")
    ):
        self.scheduled_interval_days = scheduled_interval_days
        self.drift_detector = drift_detector
        self.output_dir = output_dir
        
        self.last_retraining: Optional[datetime] = None
        self.retraining_history: List[Dict] = []
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check scheduled retraining
        if self.last_retraining is None:
            return False, "No retraining history"
        
        days_since = (datetime.now() - self.last_retraining).days
        if days_since >= self.scheduled_interval_days:
            return True, f"Scheduled retraining due ({days_since} days since last)"
        
        # Check drift-triggered retraining
        if self.drift_detector:
            status = self.drift_detector.get_status()
            if status.get('status') == 'RETRAIN_NEEDED':
                return True, f"Drift detected: F1 drop of {status.get('drop', 0):.1f}%"
        
        return False, "No retraining needed"
    
    def record_retraining(self, metrics: Dict, model_path: str):
        """Record completed retraining."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': model_path,
            'trigger': 'scheduled' if self.drift_detector is None else 'drift'
        }
        
        self.retraining_history.append(record)
        self.last_retraining = datetime.now()
        
        # Save log
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "retraining_history.json", 'w') as f:
            json.dump(self.retraining_history, f, indent=2)


# ============ USAGE EXAMPLE ============
def example_usage():
    """Example of using drift detection in production."""
    
    # Initialize detector with Phase 1 baseline
    detector = DriftDetector(
        baseline_f1=94.0,  # Your Phase 1 production F1
        warning_threshold=1.5,
        critical_threshold=3.0,
        retrain_threshold=5.0
    )
    
    # Simulate production monitoring
    print("Simulating production monitoring...")
    
    for batch in range(10):
        # Simulate incoming predictions
        y_true = np.random.randint(0, 2, size=100)
        
        # Simulate model predictions (with some degradation over time)
        noise = 0.05 + batch * 0.02  # Increasing noise
        y_pred = np.where(np.random.rand(100) > noise, y_true, 1 - y_true)
        
        # Update detector
        status = detector.update(y_true, y_pred)
        
        print(f"Batch {batch+1}: Status = {status}")
        
        if status == 'RETRAIN_NEEDED':
            print("🚨 ALERT: Retraining triggered!")
            break
    
    # Get final status
    print("\nFinal Status:")
    print(json.dumps(detector.get_status(), indent=2))


if __name__ == '__main__':
    example_usage()
