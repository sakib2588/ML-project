# IDS Compression Project - Expert Recommendations

## Executive Summary

This document provides a comprehensive analysis of the current IDS compression project and outlines strategic improvements based on established machine learning research and best practices. Each recommendation is grounded in objective reasoning with expected outcomes and implementation priorities.

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Data & Preprocessing Improvements](#1-data--preprocessing-improvements)
3. [Model Architecture Enhancements](#2-model-architecture-enhancements)
4. [Training Strategy Improvements](#3-training-strategy-improvements)
5. [Evaluation Rigor](#4-evaluation-rigor)
6. [Model Compression Techniques](#5-model-compression-techniques)
7. [Code Quality & Reproducibility](#6-code-quality--reproducibility)
8. [Priority Action Plan](#priority-action-plan)
9. [Implementation Roadmap](#implementation-roadmap)
10. [References](#references)

---

## Current State Assessment

### Strengths ✓

| Aspect | Current Implementation | Assessment |
|--------|----------------------|------------|
| **Architecture** | Modular separation (data/models/training/utils) | Excellent - follows software engineering best practices |
| **Data Pipeline** | Chronological split, memmap support | Good - prevents temporal leakage |
| **Class Imbalance** | Weighted CrossEntropy, F1-macro monitoring | Good - addresses the 82%/18% imbalance |
| **Reproducibility** | Deterministic seeding, config-driven | Good - enables experiment reproduction |
| **Documentation** | Comprehensive DOCUMENTATION.md | Excellent - facilitates onboarding |

### Areas for Improvement

| Aspect | Current State | Gap Analysis |
|--------|---------------|--------------|
| **Loss Function** | Weighted CrossEntropy only | Missing focal loss for hard examples |
| **Metrics** | Accuracy, F1, Precision, Recall | Missing PR-AUC (critical for imbalanced data) |
| **Models** | 3 baselines (MLP, CNN, LSTM) | No ensemble, no attention mechanisms |
| **Compression** | Not yet implemented | Core project goal unfulfilled |
| **Hyperparameter Tuning** | Manual configuration | No automated search |

---

## 1. Data & Preprocessing Improvements

### 1.1 Gap Windows Between Splits

**Problem**: Even with chronological splitting, adjacent windows at split boundaries may share temporal context, causing subtle data leakage.

**Solution**: Insert buffer gaps between train/val/test splits.

```
Current:    [----TRAIN----][--VAL--][--TEST--]
                          ↑        ↑
                    Potential leakage points

Recommended: [----TRAIN----]___[--VAL--]___[--TEST--]
                           ↑            ↑
                      1000-flow gaps (discarded)
```

**Objective Reasoning**:
- Network attacks often span multiple consecutive flows
- A window at the end of training and start of validation might capture the same attack session
- Gap windows ensure complete temporal independence
- Cost: ~0.1% data loss for significantly improved evaluation validity

**Expected Impact**: More realistic (often slightly lower) validation metrics that better predict production performance.

---

### 1.2 Dynamic Class Weighting

**Problem**: Static class weights (e.g., `[1.0, 5.0]`) don't adapt to changing class distributions or hard-to-classify samples.

**Current Implementation**:
```python
# Static weights applied uniformly
weight = torch.tensor([1.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=weight)
```

**Recommended Enhancement**:
```python
# Option A: Inverse frequency weighting (computed from data)
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(weights)  # Normalize

# Option B: Effective number of samples (Cui et al., 2019)
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
weights = (1.0 - beta) / effective_num
weights = weights / weights.sum() * len(weights)
```

**Objective Reasoning**:
- Inverse frequency weighting is mathematically principled: rare classes receive proportionally higher gradients
- Effective number weighting accounts for diminishing returns of additional samples
- Both methods are data-driven rather than heuristic

**Expected Impact**: 2-5% F1-macro improvement, especially on minority attack class.

---

### 1.3 Focal Loss Implementation

**Problem**: CrossEntropy treats all misclassifications equally. In imbalanced datasets, easy-to-classify majority samples dominate the gradient.

**Mathematical Foundation**:

Standard CrossEntropy:
$$CE(p_t) = -\log(p_t)$$

Focal Loss (Lin et al., 2017):
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ = predicted probability for the correct class
- $\alpha_t$ = class balancing weight
- $\gamma$ = focusing parameter (typically 2.0)
- $(1 - p_t)^\gamma$ = modulating factor that down-weights easy examples

**Objective Reasoning**:
- When $p_t \to 1$ (confident correct prediction): $(1 - p_t)^\gamma \to 0$, loss approaches 0
- When $p_t \to 0$ (misclassification): $(1 - p_t)^\gamma \to 1$, loss remains high
- This automatically focuses training on hard examples without manual sample weighting
- Proven effective in object detection (RetinaNet) and medical imaging with severe imbalance

**Implementation**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
```

**Expected Impact**: 3-8% F1-macro improvement on hard-to-classify attack subtypes.

---

### 1.4 Feature Engineering Enhancements

**Current State**: 8 manually selected features based on domain knowledge.

**Recommended Additions**:

| Feature Type | Examples | Rationale |
|--------------|----------|-----------|
| **Statistical** | Rolling mean/std of packet sizes | Captures temporal patterns |
| **Ratio Features** | Fwd/Bwd packet ratio, bytes/packet | Normalizes across flow sizes |
| **Entropy** | Byte distribution entropy | Detects anomalous payload patterns |
| **Derived** | Inter-arrival time statistics | Network timing fingerprint |

**Automated Feature Selection**:
```python
from sklearn.feature_selection import mutual_info_classif

# Compute mutual information between features and labels
mi_scores = mutual_info_classif(X_train_flat, y_train)
top_features = np.argsort(mi_scores)[-k:]  # Select top k features
```

**Objective Reasoning**:
- Mutual information measures statistical dependency without assuming linearity
- Automated selection removes human bias in feature engineering
- Reduces dimensionality while preserving discriminative power

---

## 2. Model Architecture Enhancements

### 2.1 Transformer Encoder

**Problem**: LSTM processes sequences sequentially, limiting parallelization and struggling with very long-range dependencies.

**Solution**: Self-attention mechanism processes all positions simultaneously.

**Architecture**:
```
Input (batch, seq_len, features)
    ↓
Positional Encoding (learnable or sinusoidal)
    ↓
┌─────────────────────────────────┐
│ Transformer Encoder Block × N   │
│   Multi-Head Self-Attention     │
│   → Add & LayerNorm             │
│   → Feed-Forward Network        │
│   → Add & LayerNorm             │
└─────────────────────────────────┘
    ↓
Global Average Pooling or [CLS] token
    ↓
Classification Head
```

**Objective Reasoning**:
- Self-attention computes $O(n^2)$ pairwise relationships in one step vs LSTM's $O(n)$ sequential steps
- For window_length=15, attention is computationally feasible and captures all flow interactions
- Attention weights provide interpretability (which flows influence the prediction)

**Expected Impact**: +5-10% F1 on complex multi-flow attack patterns.

---

### 2.2 Temporal Convolutional Network (TCN)

**Problem**: Standard 1D CNNs have limited receptive fields determined by kernel size.

**Solution**: Dilated causal convolutions exponentially expand the receptive field.

```
Dilation Pattern:
Layer 1: dilation=1   [●][●][●]           Receptive field: 3
Layer 2: dilation=2   [●]  [●]  [●]       Receptive field: 7
Layer 3: dilation=4   [●]      [●]      [●]   Receptive field: 15
```

**Mathematical Formulation**:
For a 1D dilated convolution:
$$(F *_d x)(t) = \sum_{i=0}^{k-1} f(i) \cdot x(t - d \cdot i)$$

Where $d$ is the dilation factor, allowing receptive field growth of $O(2^L)$ with $L$ layers.

**Objective Reasoning**:
- TCN achieves same receptive field as LSTM with fewer parameters
- Parallel computation (unlike sequential LSTM)
- Causal padding ensures no future information leakage
- Proven to match or exceed LSTM on many sequence tasks (Bai et al., 2018)

---

### 2.3 Ensemble Methods

**Problem**: Individual models have different inductive biases and failure modes.

**Solution**: Combine predictions from multiple models.

**Ensemble Strategies**:

| Strategy | Formula | Use Case |
|----------|---------|----------|
| **Hard Voting** | $\hat{y} = \text{mode}(y_1, y_2, y_3)$ | When models have similar accuracy |
| **Soft Voting** | $\hat{y} = \arg\max \sum_i p_i(y)$ | When probability calibration is good |
| **Weighted Average** | $\hat{y} = \arg\max \sum_i w_i \cdot p_i(y)$ | When models have different strengths |
| **Stacking** | Train meta-learner on base model outputs | Maximum flexibility, risk of overfitting |

**Implementation**:
```python
class EnsemblePredictor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict_proba(self, x):
        probs = [model(x).softmax(dim=-1) for model in self.models]
        weighted_probs = sum(w * p for w, p in zip(self.weights, probs))
        return weighted_probs
```

**Objective Reasoning**:
- MLP captures global patterns, CNN captures local patterns, LSTM captures sequential patterns
- Errors from different models are often uncorrelated
- Ensemble variance is lower than individual model variance: $\text{Var}(\bar{f}) = \frac{1}{n}\text{Var}(f)$ for independent predictors

**Expected Impact**: +3-5% F1 with high confidence, reduced prediction variance.

---

## 3. Training Strategy Improvements

### 3.1 K-Fold Cross-Validation

**Problem**: Single train/val/test split may not represent the full data distribution.

**Solution**: K-Fold CV provides more robust performance estimates.

**Implementation for Time Series** (Walk-Forward Validation):
```
Fold 1: [TRAIN     ][ VAL ][ TEST ]
Fold 2: [TRAIN          ][ VAL ][ TEST ]
Fold 3: [TRAIN               ][ VAL ][ TEST ]
                                     ↑
                          Always predict future
```

**Objective Reasoning**:
- Standard K-Fold shuffles data, violating temporal ordering
- Walk-forward validation respects time: always train on past, validate on future
- Provides confidence intervals on metrics, not just point estimates

---

### 3.2 Learning Rate Scheduling

**Current**: ReduceLROnPlateau (reactive)

**Recommended**: Cosine Annealing with Warm Restarts (proactive)

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$$

**Objective Reasoning**:
- Cosine schedule smoothly decreases LR, avoiding sudden drops
- Warm restarts escape local minima by periodically resetting to high LR
- Empirically converges to better optima than step decay (Loshchilov & Hutter, 2017)

---

### 3.3 Automated Hyperparameter Optimization

**Problem**: Manual tuning is time-consuming and may miss optimal configurations.

**Solution**: Bayesian optimization with Optuna.

```python
import optuna

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    hidden_size = trial.suggest_categorical('hidden', [64, 128, 256])
    
    # Train model with these hyperparameters
    model = create_model(hidden_size, dropout)
    val_f1 = train_and_evaluate(model, lr)
    
    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Objective Reasoning**:
- Bayesian optimization uses surrogate model to predict promising regions
- More sample-efficient than grid search or random search
- Automatically handles continuous, discrete, and conditional parameters

---

## 4. Evaluation Rigor

### 4.1 Precision-Recall AUC (PR-AUC)

**Problem**: ROC-AUC can be misleadingly high with imbalanced data.

**Mathematical Explanation**:

ROC-AUC measures:
- True Positive Rate: $TPR = \frac{TP}{TP + FN}$
- False Positive Rate: $FPR = \frac{FP}{FP + TN}$

With 82% normal class, even a small FPR yields many false positives in absolute terms.

PR-AUC measures:
- Precision: $P = \frac{TP}{TP + FP}$
- Recall: $R = \frac{TP}{TP + FN}$

**Why PR-AUC is better for IDS**:
```
Example: 10,000 samples (8,200 normal, 1,800 attack)
Model predicts 2,000 as attack: 1,700 TP, 300 FP

ROC metrics:
- TPR = 1700/1800 = 94.4%
- FPR = 300/8200 = 3.7%  ← Looks great!

PR metrics:
- Precision = 1700/2000 = 85%
- Recall = 1700/1800 = 94.4%

But 300 false alarms per 10K samples = significant operational burden
```

**Objective Reasoning**:
- In production IDS, each false positive requires human investigation
- PR-AUC directly measures the precision-recall tradeoff
- Provides threshold-independent evaluation focused on the minority class

---

### 4.2 Per-Attack-Type Analysis

**Problem**: Aggregate F1 masks performance variation across attack types.

**Solution**: Compute metrics separately for DDoS, PortScan, Infiltration, etc.

```python
# Group test predictions by original attack type
attack_metrics = {}
for attack_type in ['DDoS', 'PortScan', 'WebAttack', 'Infiltration']:
    mask = original_labels == attack_type
    attack_metrics[attack_type] = {
        'precision': precision_score(y_true[mask], y_pred[mask]),
        'recall': recall_score(y_true[mask], y_pred[mask]),
        'f1': f1_score(y_true[mask], y_pred[mask]),
        'support': mask.sum()
    }
```

**Objective Reasoning**:
- Different attacks have different signatures (DDoS: high volume; PortScan: many connections)
- Model may excel at DDoS (easy) but fail at Infiltration (subtle)
- Per-attack analysis guides targeted improvements

---

### 4.3 Detection Latency Metric

**Problem**: F1 measures correctness but not speed of detection.

**Definition**: Number of flows from attack start until first correct detection.

$$\text{Detection Latency} = t_{first\_detection} - t_{attack\_start}$$

**Objective Reasoning**:
- In real networks, detecting DDoS after 1,000 flows vs 10 flows matters enormously
- Early detection enables faster mitigation
- Particularly important for streaming/online deployment

---

## 5. Model Compression Techniques

This is the **core objective** of the project. Below are the primary compression techniques in order of implementation priority.

### 5.1 Knowledge Distillation (Priority: HIGH)

**Concept**: Train a small "student" model to mimic a large "teacher" model's soft predictions.

**Mathematical Formulation**:

Standard training loss:
$$L_{hard} = CE(y_{student}, y_{true})$$

Distillation loss:
$$L_{soft} = KL(softmax(z_t/T), softmax(z_s/T))$$

Combined loss:
$$L = \alpha \cdot L_{hard} + (1-\alpha) \cdot T^2 \cdot L_{soft}$$

Where:
- $z_t, z_s$ = teacher and student logits
- $T$ = temperature (typically 3-20, higher = softer probabilities)
- $\alpha$ = balancing weight (typically 0.1-0.5)

**Why It Works**:
- Hard labels: "This is attack" (1 bit of information)
- Soft labels: "This is 95% attack, 5% normal" (more information)
- Soft labels encode inter-class relationships learned by the teacher
- Student learns from teacher's "dark knowledge" about similar classes

**Implementation**:
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.3):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        # Hard loss (standard CE with true labels)
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Soft loss (KL divergence with teacher's soft predictions)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_loss = self.kl_div(soft_student, soft_teacher)
        
        # Combined loss (T^2 scaling as per Hinton et al.)
        return self.alpha * hard_loss + (1 - self.alpha) * (self.T ** 2) * soft_loss
```

**Expected Compression**: 5-10x parameter reduction with <2% F1 drop.

---

### 5.2 Post-Training Quantization (Priority: HIGH)

**Concept**: Reduce numerical precision from FP32 to INT8.

**Mathematical Formulation**:

Quantization mapping:
$$x_q = \text{round}(\frac{x}{s}) + z$$

Where:
- $s$ = scale factor
- $z$ = zero point
- $x_q$ = quantized value (INT8: range [-128, 127])

**Types of Quantization**:

| Type | When Applied | Accuracy Impact | Speed Gain |
|------|--------------|-----------------|------------|
| **Dynamic** | Weights static, activations at runtime | Low | 2-3x |
| **Static** | Both pre-computed with calibration | Medium | 3-4x |
| **QAT** | During training | Lowest | 3-4x |

**Implementation** (PyTorch Dynamic Quantization):
```python
import torch.quantization as quant

# Quantize linear and LSTM layers to INT8
quantized_model = quant.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# Check size reduction
original_size = os.path.getsize('model.pth')
quantized_size = os.path.getsize('model_quantized.pth')
print(f"Compression: {original_size / quantized_size:.1f}x")
```

**Objective Reasoning**:
- INT8 uses 4x less memory than FP32
- Modern CPUs have optimized INT8 instructions (VNNI, AVX-512)
- Particularly effective for deployment on edge devices

**Expected Compression**: 4x size reduction, 2-4x inference speedup.

---

### 5.3 Structured Pruning (Priority: MEDIUM)

**Concept**: Remove entire neurons/filters rather than individual weights.

**Pruning Criteria**:

| Method | Formula | Rationale |
|--------|---------|-----------|
| **Magnitude** | Remove smallest $\|w\|$ | Small weights contribute least |
| **Gradient** | Remove smallest $\|w \cdot \nabla w\|$ | Low gradient = low impact |
| **Taylor** | Remove smallest $\|\delta L / \delta w\|$ | Direct loss impact estimation |

**Implementation**:
```python
import torch.nn.utils.prune as prune

# Prune 30% of channels with lowest L1 norm
prune.ln_structured(
    module=model.conv1,
    name='weight',
    amount=0.3,
    n=1,  # L1 norm
    dim=0  # Prune output channels
)

# Make pruning permanent
prune.remove(model.conv1, 'weight')
```

**Objective Reasoning**:
- Unstructured pruning creates sparse matrices (hard to accelerate without special hardware)
- Structured pruning removes entire filters → smaller dense matrices → direct speedup
- Can be combined with fine-tuning to recover accuracy

**Expected Compression**: 2-4x with iterative pruning + fine-tuning.

---

### 5.4 Neural Architecture Search (Priority: LOW)

**Concept**: Automatically search for optimal small architecture.

**Search Space Example**:
```python
search_space = {
    'num_layers': [1, 2, 3],
    'hidden_size': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'use_attention': [True, False],
}
# Search for architecture with <50K params and max F1
```

**Objective Reasoning**:
- Human-designed architectures may not be optimal for the specific task
- NAS can find unconventional but effective designs
- Constraint-aware NAS explicitly optimizes for size/speed/accuracy tradeoff

**Note**: NAS is computationally expensive. Recommend only after other techniques are exhausted.

---

## 6. Code Quality & Reproducibility

### 6.1 Experiment Tracking with MLflow/W&B

**Problem**: Manual tracking of experiments is error-prone and hard to compare.

**Solution**: Automated experiment logging.

```python
import mlflow

mlflow.set_experiment("IDS_Compression_Phase1")

with mlflow.start_run(run_name="LSTM_focal_loss"):
    # Log parameters
    mlflow.log_params({
        "model": "lstm",
        "hidden_size": 64,
        "loss": "focal",
        "gamma": 2.0
    })
    
    # Train model
    history = trainer.fit(epochs=20)
    
    # Log metrics
    mlflow.log_metrics({
        "val_f1_macro": best_f1,
        "val_pr_auc": pr_auc,
        "parameters": param_count
    })
    
    # Log model artifact
    mlflow.pytorch.log_model(model, "model")
```

**Benefits**:
- Automatic hyperparameter tracking
- Metric visualization and comparison
- Model versioning and artifact storage
- Collaboration features

---

### 6.2 Configuration Validation with Pydantic

**Problem**: YAML configs can have typos or invalid values silently.

**Solution**: Type-safe configuration with runtime validation.

```python
from pydantic import BaseModel, validator
from typing import List, Literal

class ModelConfig(BaseModel):
    name: Literal["mlp", "ds_cnn", "lstm"]
    hidden_sizes: List[int]
    dropout_rate: float
    
    @validator('dropout_rate')
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('dropout_rate must be between 0 and 1')
        return v

# Usage
config = ModelConfig(**yaml.safe_load(config_file))  # Validates automatically
```

---

### 6.3 Comprehensive Unit Testing

**Current State**: No automated tests.

**Recommended Test Coverage**:

| Module | Test Cases |
|--------|------------|
| `loaders.py` | Dataset loading, shape validation, memmap behavior |
| `windowing.py` | Window shapes, label strategies, edge cases |
| `models/*.py` | Forward pass shapes, gradient flow, save/load |
| `trainer.py` | Training loop, checkpoint saving, early stopping |

```python
# Example test
def test_ids_dataset_shapes():
    dataset = IDSDataset("data/processed/cic_ids_2017/train")
    x, y = dataset[0]
    assert x.shape == (15, 8), f"Expected (15, 8), got {x.shape}"
    assert y.dim() == 0, "Label should be scalar"
```

---

## Priority Action Plan

Based on impact-to-effort ratio, here is the recommended implementation order:

### Phase 1: Quick Wins (1-2 days each)

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| 1 | **Focal Loss** | +3-8% F1 on hard samples | Low |
| 2 | **PR-AUC Metric** | Better evaluation insight | Low |
| 3 | **Ensemble Predictor** | +3-5% F1, robust | Low |

### Phase 2: Core Compression (3-5 days each)

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| 4 | **Knowledge Distillation** | 5-10x compression | Medium |
| 5 | **INT8 Quantization** | 4x size, 2-4x speed | Low |
| 6 | **Structured Pruning** | 2-4x compression | Medium |

### Phase 3: Advanced Improvements (1-2 weeks each)

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| 7 | **Transformer Encoder** | +5-10% F1 on complex attacks | High |
| 8 | **Hyperparameter Optimization** | +2-5% F1 systematically | Medium |
| 9 | **MLflow Integration** | Better reproducibility | Medium |

---

## Implementation Roadmap

```
Week 1-2: Foundation Improvements
├── Implement Focal Loss in trainer.py
├── Add PR-AUC to evaluator.py
└── Create EnsemblePredictor class

Week 3-4: Compression Core
├── Implement DistillationTrainer
├── Add quantization utilities
└── Benchmark compressed models

Week 5-6: Advanced Features
├── Add Transformer model option
├── Integrate Optuna for hyperparameter search
└── Set up MLflow experiment tracking

Week 7-8: Refinement
├── Per-attack-type analysis
├── Comprehensive unit tests
└── Documentation updates
```

---

## References

1. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

2. **Knowledge Distillation**: Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Workshop.

3. **TCN**: Bai, S., et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv.

4. **Cosine Annealing**: Loshchilov, I., & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR.

5. **Class Imbalance**: Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR.

6. **Quantization**: Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR.

7. **Pruning**: Li, H., et al. (2017). "Pruning Filters for Efficient ConvNets." ICLR.

---

## Conclusion

This project has a solid foundation with modular architecture, proper data handling, and comprehensive documentation. The primary gaps are in:

1. **Loss function sophistication** (focal loss for hard examples)
2. **Evaluation depth** (PR-AUC, per-attack analysis)
3. **Model compression** (the core project goal)

Implementing the Phase 1 quick wins (focal loss, PR-AUC, ensemble) will provide immediate F1 improvements. The compression techniques in Phase 2 (distillation, quantization) will achieve the project's primary objective of efficient IDS models.

The recommendations in this document are grounded in peer-reviewed research and production ML best practices. Each suggestion includes objective reasoning and expected outcomes to enable informed prioritization.

---

*Document created: November 2025*
*Based on analysis of IDS Compression Project codebase*
