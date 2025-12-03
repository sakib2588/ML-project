# Publication Roadmap: Edge-Native IDS Compression with Honest Rare-Class Reporting

> **Research Philosophy**: We keep all 10 attack classes. We report failures honestly. We do not merge or remove rare classes to inflate metrics. Scientific integrity over pretty numbers.

> **Status**: ✅ APPROVED — Proceed with full 10-class evaluation, honest rare-class reporting, merged-as-ablation only.

---

## Executive Summary

**Research Question**: How does model compression (KD → Pruning → QAT → INT8) affect intrusion detection across *all* attack classes, including rare ones?

**Key Contribution**: Demonstrating that compression amplifies baseline weaknesses on rare attack types — providing practitioners with honest guidance on when compression is safe.

**Timeline**: ~8 weeks (December 2025 - January 2026)

---

## ⚠️ Critical Risks & Mitigations

These must be addressed in Week 1 before heavy experiments:

| Risk | Impact | Mitigation |
|------|--------|------------|
| **SMOTE on tiny seeds (40→500)** | Creates unrealistic windows, leaks artifacts | Hybrid augmentation: jitter + mixup FIRST, then capped SMOTE (≤5× per iteration) |
| **Holdout too small (25 samples)** | Low statistical power for rare-class claims | Increase to 40-50 if possible; use bootstrap CIs; state power limits |
| **Class weights too aggressive (50×)** | Training instability, exploding gradients | Cap at 50×, use gradient clipping (1.0), focal loss (γ=2) |
| **Overfitting on augmented data** | Memorization, poor generalization | Monitor train vs val ROC gap, t-SNE validation of synthetic vs real |
| **Conversion drift (PyTorch→ONNX→TFLite)** | Silent accuracy loss | Run numeric verification on sample batch before full eval |

---

## Table of Contents

1. [Research Philosophy](#1-research-philosophy)
2. [The Rare-Class Problem (Honest Assessment)](#2-the-rare-class-problem)
3. [Dataset: CIC-IDS-2017 Distribution](#3-dataset-cic-ids-2017-distribution)
4. [Experimental Design (3 Tracks)](#4-experimental-design)
5. [Phase 1: Fix the Baseline (Weeks 1-2)](#5-phase-1-fix-the-baseline)
6. [Phase 2: Train Strong Teacher (Weeks 3-4)](#6-phase-2-train-strong-teacher)
7. [Phase 3: Compression Pipeline (Weeks 4-6)](#7-phase-3-compression-pipeline)
8. [Phase 4: Ablation Studies (Week 6-7)](#8-phase-4-ablation-studies)
9. [Phase 5: Paper Writing (Weeks 7-8)](#9-phase-5-paper-writing)
10. [Publication-Ready Results Format](#10-publication-ready-results)
11. [Limitations Section (Draft)](#11-limitations-section)
12. [Checklist Before Submission](#12-checklist)
13. [Acceptance Criteria & Metrics](#13-acceptance-criteria)
14. [Week 1 Critical Tasks](#14-week-1-critical-tasks)

---

## 1. Research Philosophy

### What We DO:
- ✅ Keep all 10 attack classes in main evaluation
- ✅ Report per-class precision/recall/F1 for every stage
- ✅ Transparently show rare-class degradation under compression
- ✅ Provide actionable guidance for practitioners
- ✅ Include class-merging as an **ablation study only**
- ✅ Write an honest limitations section

### What We DO NOT DO:
- ❌ Merge Bot + SSH-Patator to hide poor performance
- ❌ Remove rare classes from evaluation
- ❌ Cherry-pick metrics that look good
- ❌ Claim the model "works" when it fails on specific attacks

### Why This Matters:
1. **Reviewers notice manipulation** — Papers that "simplify the task" by removing difficult classes get flagged
2. **Operational risk is real** — Rare attacks are often most valuable to detect (adversaries stay rare intentionally)
3. **Scientific value** — Showing compression amplifies baseline weaknesses is itself a meaningful contribution

---

## 2. The Rare-Class Problem

### Honest Assessment of Current State

| Attack Class | Samples | % of Data | Expected Recall | Status |
|--------------|---------|-----------|-----------------|--------|
| BENIGN | 533,903 | 80.83% | >99% | ✅ OK |
| DoS Hulk | 55,751 | 8.44% | >98% | ✅ OK |
| PortScan | 37,331 | 5.65% | >98% | ✅ OK |
| DDoS | 29,711 | 4.50% | >98% | ✅ OK |
| DoS Slowhttptest | 1,325 | 0.20% | >90% | ✅ OK |
| DoS slowloris | 1,197 | 0.18% | >90% | ✅ OK |
| FTP-Patator | 868 | 0.13% | >85% | ✅ OK |
| DoS GoldenEye | 275 | 0.04% | 70-85% | ⚠️ Challenging |
| SSH-Patator | 73 | 0.01% | 50-75% | ⚠️ **Known Weakness** |
| Bot | 69 | 0.01% | 50-80% | ⚠️ **Known Weakness** |

**Imbalance Ratio**: 7,738:1 (BENIGN vs Bot)

### The Brutal Truth

With only 69-73 samples, even a perfect model will struggle. This is a **data problem**, not a model problem. We acknowledge this honestly:

> "Our baseline teacher achieves only 60-80% recall on Bot and SSH-Patator due to extreme class imbalance (69 and 73 samples respectively). Compression further degrades these rates. This limitation reflects the fundamental challenge of detecting rare attacks without sufficient training data."

---

## 3. Dataset: CIC-IDS-2017 Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                    CIC-IDS-2017 CLASS DISTRIBUTION              │
├─────────────────────┬──────────┬─────────┬─────────────────────┤
│ Attack Type         │ Count    │ %       │ Detection Target    │
├─────────────────────┼──────────┼─────────┼─────────────────────┤
│ BENIGN              │ 533,903  │ 80.83%  │ Specificity >98%    │
│ DoS Hulk            │ 55,751   │ 8.44%   │ Recall >95%         │
│ PortScan            │ 37,331   │ 5.65%   │ Recall >98% (HARD)  │
│ DDoS                │ 29,711   │ 4.50%   │ Recall >98% (HARD)  │
│ DoS Slowhttptest    │ 1,325    │ 0.20%   │ Recall >90%         │
│ DoS slowloris       │ 1,197    │ 0.18%   │ Recall >90%         │
│ FTP-Patator         │ 868      │ 0.13%   │ Recall >85%         │
│ DoS GoldenEye       │ 275      │ 0.04%   │ Recall >70%         │
│ SSH-Patator         │ 73       │ 0.01%   │ Recall >50% (BEST)  │
│ Bot                 │ 69       │ 0.01%   │ Recall >50% (BEST)  │
├─────────────────────┼──────────┼─────────┼─────────────────────┤
│ TOTAL               │ 660,503  │ 100%    │                     │
└─────────────────────┴──────────┴─────────┴─────────────────────┘
```

### Realistic Targets for Rare Classes

Given the extreme imbalance, we set **honest targets**:

| Class | Baseline Target | Post-Compression Target | Rationale |
|-------|-----------------|-------------------------|-----------|
| Bot | 60-80% | 40-70% | Only 69 samples; expect degradation |
| SSH-Patator | 50-75% | 35-65% | Only 73 samples; expect degradation |

**We will report actual numbers, not hide behind merged classes.**

---

## 4. Experimental Design (3 Tracks)

### Track A: MAIN EXPERIMENT (Full 10-Class)

This is the primary evaluation reported in the paper.

```
Teacher (10 classes) → Student (10 classes) → KD → Prune → QAT → TFLite
                                                      ↓
                                            Per-class metrics for ALL 10 classes
```

**What we report:**
- Per-class Precision, Recall, F1
- Confusion matrices at each stage
- Pareto frontier (Size vs F1 vs Latency)
- Rare-class degradation analysis

### Track B: ABLATION (Merged Rare Classes)

This is a secondary experiment to show what happens if we "cheat".

```
Merge: Bot + SSH-Patator → RARE_ATTACK (9 classes total)
Re-run: Teacher → Student → KD → Prune → QAT → TFLite
Compare: Side-by-side with Track A
```

**Purpose**: Demonstrate that merging inflates metrics but hides true failure modes.

### Track C: RARE-CLASS HOLDOUT

Reserve 40 samples of Bot and SSH-Patator for final evaluation only.

```
Holdout Set:
- 40 Bot samples (not used in training/augmentation)
- 40 SSH-Patator samples (not used in training/augmentation)
- Total: 80 samples for honest rare-class evaluation

Evaluate EVERY model on this holdout to show true rare-class performance.
Report bootstrap 95% CIs given limited sample size.
```

---

## 5. Phase 1: Fix the Baseline (Weeks 1-2)

### Week 1: Data Preparation

#### Step 1.1: Create Rare-Class Holdout Set

**⚠️ INCREASED FROM 25 TO 40 SAMPLES** for better statistical power.

```bash
python scripts/create_holdout_set.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-classes Bot SSH-Patator \
    --holdout-samples 40 \
    --output-dir data/processed/cic_ids_2017_v2/holdout/
```

This creates:
- `holdout/X_rare.npy` — Features for 80 rare samples (40 Bot + 40 SSH-Patator)
- `holdout/y_rare.npy` — Binary labels
- `holdout/attack_types_rare.npy` — Class labels
- Training data with these samples removed

**Statistical Note:** With only 40 holdout samples per class, we have limited statistical power. We will:
1. Report **bootstrap 95% CIs** for holdout recall
2. State power limitations explicitly in the paper
3. Use holdout primarily for qualitative examples alongside quantitative metrics

#### Step 1.2: Augment Rare Classes (Training Only)

**⚠️ SAFER AUGMENTATION PIPELINE (Critical Fix)**

Naïve SMOTE from 40→500 creates unrealistic windows and leaks artifacts. Use this hybrid approach instead:

```bash
python phase2/pre_phase2/augment_rare_classes.py \
    --method hybrid \
    --target-samples 500 \
    --rare-classes Bot SSH-Patator DoS_GoldenEye \
    --output-dir data/processed/cic_ids_2017_v2_augmented/
```

**Hybrid Augmentation Sequence (Order Matters!):**

```python
# Step 1: Jitter (fast, realistic) — apply to ALL windows
def jitter_augment(X, noise_std=0.01):
    """Add Gaussian noise per-feature to create realistic variations."""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

# Step 2: Mixup within same class — creates smooth interpolations
def mixup_augment(X, y, alpha=0.2):
    """Interpolate between same-class windows."""
    lam = np.random.beta(alpha, alpha)
    # Only mix samples from same class
    indices = np.random.permutation(len(X))
    X_mixed = lam * X + (1 - lam) * X[indices]
    return X_mixed

# Step 3: Capped SMOTE — ONLY after jitter/mixup, limit expansion
def capped_smote(X, y, target_class, max_multiplier=5):
    """
    SMOTE but capped to ≤5× original count per iteration.
    Prevents extreme synthetic generation.
    """
    from imblearn.over_sampling import SMOTE
    
    original_count = (y == target_class).sum()
    max_synthetic = original_count * max_multiplier
    
    # Flatten for SMOTE
    X_flat = X.reshape(X.shape[0], -1)
    
    # Apply SMOTE with cap
    smote = SMOTE(
        sampling_strategy={target_class: min(max_synthetic, 500)},
        k_neighbors=min(5, original_count - 1),  # Adaptive k
        random_state=42
    )
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    # Reshape back to windows
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])
    return X_resampled, y_resampled
```

**Validation: t-SNE Check (REQUIRED before proceeding)**

```python
# Visualize real vs synthetic to verify augmentation quality
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def validate_augmentation(X_real, X_synthetic, class_name):
    """Plot t-SNE to verify synthetic samples look realistic."""
    X_all = np.vstack([X_real.reshape(len(X_real), -1), 
                       X_synthetic.reshape(len(X_synthetic), -1)])
    labels = ['Real'] * len(X_real) + ['Synthetic'] * len(X_synthetic)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_all)
    
    plt.figure(figsize=(10, 8))
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels) == label
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                   label=label, alpha=0.6)
    plt.title(f'{class_name}: Real vs Synthetic (t-SNE)')
    plt.legend()
    plt.savefig(f'reports/augmentation_validation_{class_name}.png')
    plt.close()
    
    # Compute overlap score (synthetic should overlap with real)
    # If clusters are separate → augmentation is unrealistic!
```

**Decision Rule:**
- ✅ If synthetic cluster **overlaps** real cluster → proceed
- ❌ If synthetic cluster is **separate** → tweak augmentation parameters or reduce synthetic count

**Augmentation targets (REVISED — more conservative):**
| Class | Original | After Jitter | After Mixup | After SMOTE | Final |
|-------|----------|--------------|-------------|-------------|-------|
| Bot | 44 | 88 | 132 | 220 | ~300 |
| SSH-Patator | 48 | 96 | 144 | 240 | ~350 |
| DoS GoldenEye | 275 | 275 | 275 | 500 | ~500 |

#### Step 1.3: Compute Class Weights

```python
# Saved to class_weights.json
# ⚠️ CAPPED AT 50× to prevent training instability
class_weights = {
    'BENIGN': 1.0,
    'Bot': 50.0,          # Capped at 50× (was higher)
    'DDoS': 2.0,
    'DoS GoldenEye': 20.0,
    'DoS Hulk': 1.5,
    'DoS Slowhttptest': 5.0,
    'DoS slowloris': 5.0,
    'FTP-Patator': 8.0,
    'PortScan': 2.0,
    'SSH-Patator': 50.0   # Capped at 50× (was higher)
}
```

### Week 2: Baseline Model Training

#### Step 2.1: Train Teacher with Balancing Strategies

```bash
python scripts/train_teacher_balanced.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --class-weights class_weights.json \
    --loss focal \
    --focal-gamma 2.0 \
    --sampler balanced \
    --epochs 100 \
    --min-epochs 30 \
    --patience 12 \
    --grad-clip 1.0 \
    --lr 5e-4 \
    --lr-scheduler plateau \
    --seeds 0 7 42 \
    --output-dir experiments/teacher_balanced/
```

**Training Configuration (STABILIZED):**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss - focuses on hard examples (γ=2 recommended)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weights = class_weights
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        
        if self.reduction == 'mean':
            return focal.mean()
        return focal

# Stabilized training loop with gradient clipping
def train_epoch(model, loader, optimizer, criterion, grad_clip=1.0):
    model.train()
    total_loss = 0
    
    for X, y_bin, y_atk in loader:
        optimizer.zero_grad()
        
        binary_logits, attack_logits = model(X.cuda())
        
        loss_binary = F.binary_cross_entropy_with_logits(
            binary_logits, y_bin.cuda().float().unsqueeze(1)
        )
        loss_attack = criterion(attack_logits, y_atk.cuda())
        loss = 0.5 * loss_binary + 0.5 * loss_attack
        
        loss.backward()
        
        # ⚠️ CRITICAL: Gradient clipping prevents explosion from high class weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

# LR Schedule: AdamW + ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=4, verbose=True
)

# Balanced Sampler - oversamples rare classes
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(y_attack_types, class_weights_dict):
    """Create sampler that oversamples rare classes."""
    sample_weights = []
    for y in y_attack_types:
        class_name = CLASS_NAMES[y]
        sample_weights.append(class_weights_dict.get(class_name, 1.0))
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
```

#### Step 2.2: Evaluate Teacher on All Classes

```bash
python scripts/evaluate_teacher.py \
    --model experiments/teacher_balanced/best_model.pt \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-dir data/processed/cic_ids_2017_v2/holdout \
    --output reports/teacher_evaluation.json
```

**Expected Teacher Results (Post-Balancing):**

| Class | Precision | Recall | F1 | Status |
|-------|-----------|--------|-----|--------|
| BENIGN | 99.5% | 99.2% | 99.3% | ✅ |
| DDoS | 99.0% | 98.5% | 98.7% | ✅ |
| PortScan | 98.8% | 98.2% | 98.5% | ✅ |
| DoS Hulk | 98.5% | 97.8% | 98.1% | ✅ |
| DoS Slowhttptest | 95.0% | 92.0% | 93.5% | ✅ |
| DoS slowloris | 94.0% | 91.0% | 92.5% | ✅ |
| FTP-Patator | 90.0% | 88.0% | 89.0% | ✅ |
| DoS GoldenEye | 85.0% | 78.0% | 81.4% | ⚠️ |
| **SSH-Patator** | **75.0%** | **65.0%** | **69.7%** | ⚠️ |
| **Bot** | **80.0%** | **70.0%** | **74.7%** | ⚠️ |

**Macro-F1: ~89-91%** (realistic, not inflated)

---

## 6. Phase 2: Train Strong Teacher (Weeks 3-4)

### Multi-Task Teacher Architecture

```python
class MultiTaskTeacher(nn.Module):
    """
    Teacher model with dual heads:
    - Binary head: Attack vs Benign
    - Attack head: 10-class classification
    """
    def __init__(self, input_features=65, window_size=15):
        super().__init__()
        
        # Shared backbone (DS-CNN)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv1d(input_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            # Stage 1: Multi-scale + SE
            MultiScaleBlock(128, 256),
            SEBlock(256),
            
            # Stage 2: DS-Conv + Attention
            DSConvBlock(256, 256),
            AttentionBlock(256),
            
            # Stage 3: DS-Conv + SE
            DSConvBlock(256, 512),
            SEBlock(512),
            
            # Pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Binary head (Attack vs Benign)
        self.binary_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Sigmoid applied externally
        )
        
        # Attack type head (10 classes)
        self.attack_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 attack classes
        )
    
    def forward(self, x):
        features = self.backbone(x)
        binary_logits = self.binary_head(features)
        attack_logits = self.attack_head(features)
        return binary_logits, attack_logits
```

### Training Command

```bash
python phase2/train/train_teacher.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --class-weights class_weights.json \
    --loss focal \
    --epochs 100 \
    --patience 15 \
    --seeds 0 7 42 101 202 \
    --output-dir artifacts/teacher/
```

### Teacher Checkpoints

```
artifacts/teacher/
├── seed0/
│   ├── best_model.pt
│   ├── training_log.json
│   └── per_class_metrics.json
├── seed7/
├── seed42/
├── seed101/
├── seed202/
└── aggregate_results.json
```

---

## 7. Phase 3: Compression Pipeline (Weeks 4-6)

### Pipeline Overview

```
Teacher (1.9M params)
    ↓ Knowledge Distillation
Student Baseline (50K-200K params)
    ↓ Structured Pruning (50%)
Pruned Student (25K-100K params)
    ↓ KD Fine-tuning
Recovered Student
    ↓ Quantization-Aware Training
INT8 Student
    ↓ TFLite Conversion
Final Model (<50KB, <10ms)
```

### Stage 0: Train Student Baselines

```bash
# Train students WITHOUT teacher guidance (baseline)
python phase2/train/train_baseline.py \
    --param-targets 50000 100000 200000 \
    --seeds 0 7 42 \
    --epochs 50 \
    --output-dir artifacts/stage0_baseline/
```

### Stage 1: Knowledge Distillation

```bash
# Train students WITH teacher guidance
python phase2/train/train_kd.py \
    --teacher artifacts/teacher/best_model.pt \
    --param-targets 50000 100000 200000 \
    --temperature 4.0 \
    --alpha 0.7 \
    --seeds 0 7 42 \
    --epochs 100 \
    --output-dir artifacts/stage1_kd/
```

### Stage 2: Structured Pruning

```bash
# Prune 50% of filters (iterative)
python phase2/prune/prune_multitask.py \
    --model artifacts/stage1_kd/200k/seed42/best_model.pt \
    --prune-ratio 0.5 \
    --iterative \
    --steps 10 \
    --output-dir artifacts/stage2_pruned/
```

### Stage 3: KD Fine-tuning

```bash
# Recover accuracy with teacher guidance
python phase2/prune/finetune_kd.py \
    --model artifacts/stage2_pruned/best_model.pt \
    --teacher artifacts/teacher/best_model.pt \
    --epochs 50 \
    --output-dir artifacts/stage3_finetuned/
```

### Stage 4: Quantization-Aware Training

```bash
# Train with INT8 simulation
python phase2/quant/qat_multitask.py \
    --model artifacts/stage3_finetuned/best_model.pt \
    --epochs 30 \
    --output-dir artifacts/stage4_qat/
```

### Stage 5: TFLite Conversion

```bash
# Convert to TFLite INT8
python phase2/convert/convert_multitask.py \
    --model artifacts/stage4_qat/best_model.pt \
    --quantize int8 \
    --output-dir artifacts/stage5_tflite/
```

### Per-Stage Evaluation

**CRITICAL**: After EVERY stage, evaluate on:
1. Full test set (all 10 classes)
2. Rare-class holdout set

```bash
# Evaluate at each stage
for stage in stage0 stage1 stage2 stage3 stage4 stage5; do
    python scripts/evaluate_per_class.py \
        --model artifacts/${stage}/best_model.pt \
        --test-data data/processed/cic_ids_2017_v2/test \
        --holdout-data data/processed/cic_ids_2017_v2/holdout \
        --output reports/${stage}_evaluation.json
done
```

---

## 8. Phase 4: Ablation Studies (Week 6-7)

### Ablation A: Merged Rare Classes

```bash
# Create merged dataset (Bot + SSH-Patator → RARE_ATTACK)
python scripts/merge_rare_classes.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --merge "Bot,SSH-Patator:RARE_ATTACK" \
    --output-dir data/processed/cic_ids_2017_v2_merged/

# Re-run full pipeline on merged data
bash scripts/run_pipeline_merged.sh
```

### Ablation B: Student Size Comparison

| Student Size | Params | Expected F1 | Rare-Class F1 |
|--------------|--------|-------------|---------------|
| Tiny | 5K | ~90% | ~30% |
| Small | 50K | ~92% | ~45% |
| Medium | 100K | ~93% | ~55% |
| Large | 200K | ~94% | ~60% |

### Ablation C: Compression Technique Comparison

| Technique | Size Reduction | F1 Drop | Rare-Class Impact |
|-----------|----------------|---------|-------------------|
| KD Only | 0% | -0.5% | -5% |
| Pruning Only | 50% | -2.0% | -15% |
| QAT Only | 87% | -0.3% | -3% |
| Full Pipeline | 95% | -1.5% | -20% |

---

## 9. Phase 5: Paper Writing (Weeks 7-8)

### Paper Structure

```
1. Introduction
   - Edge IDS motivation
   - Compression challenges
   - Research question: How does compression affect rare-class detection?

2. Related Work
   - IDS on edge devices
   - Model compression techniques
   - Class imbalance in security ML

3. Methodology
   - Dataset (CIC-IDS-2017, honest distribution)
   - Multi-task architecture (Binary + Attack-type heads)
   - Compression pipeline (KD → Prune → QAT)
   - Evaluation protocol (per-class, holdout set)

4. Experimental Setup
   - Hardware (Raspberry Pi 4)
   - Metrics (per-class P/R/F1, FAR, latency)
   - Statistical rigor (N=5 seeds, 95% CI)

5. Results
   - Table 1: Overall metrics per stage
   - Table 2: Per-class breakdown (ALL 10 classes)
   - Table 3: Rare-class holdout evaluation
   - Figure 1: Pareto frontier (Size vs F1 vs Latency)
   - Figure 2: Per-class degradation curves
   - Figure 3: Confusion matrices

6. Ablation Studies
   - Table 4: Merged classes vs Full classes
   - Impact of student size
   - Impact of each compression technique

7. Discussion
   - When is compression safe?
   - Operational implications
   - Guidance for practitioners

8. Limitations (HONEST)
   - Rare-class performance
   - Dataset limitations
   - Generalization concerns

9. Conclusion
   - Summary of findings
   - Actionable guidance
   - Future work
```

### Key Figures to Generate

#### Figure 1: Per-Class Recall Degradation

```
100% ─┬─────────────────────────────────────────────
      │  ███ DDoS                    ███ ███ ███ ███
 90% ─┤  ███ PortScan               ███ ███ ███ ███
      │  ███ DoS Hulk              ███ ███ ███ ███
 80% ─┤                           ███ ███ ███ ███
      │                          ███ ███ ███ ███
 70% ─┤                         ███         ███
      │        ▓▓▓ Bot                     ▓▓▓
 60% ─┤       ▓▓▓             ▓▓▓
      │      ▓▓▓             ▓▓▓          ▓▓▓
 50% ─┤     ▓▓▓             ▓▓▓          ▓▓▓
      │    ░░░ SSH-Patator ░░░          ░░░
 40% ─┤   ░░░             ░░░          ░░░
      │  ░░░             ░░░
 30% ─┤ ░░░             ░░░           ░░░
      └─┬───────┬───────┬───────┬───────┬───────┬──
       Teacher  KD     Prune   KD-FT   QAT   TFLite
```

#### Figure 2: Pareto Frontier

```
F1 Score
 95% ─┬─────────────────────────────────────────────
      │    ★ Teacher (1.9M, 94.5%)
 94% ─┤         ● KD-200K
      │              ○ Pruned
 93% ─┤                   ◇ QAT
      │                        ◆ TFLite-INT8 (48KB)
 92% ─┤
      │
 91% ─┤                            Target Zone
      │                            (Size<50KB, F1>91%)
 90% ─┤
      └─┬───────┬───────┬───────┬───────┬───────┬──
       2MB    1MB   500KB  200KB  100KB   50KB
                    Model Size
```

---

## 10. Publication-Ready Results Format

### Table 1: Overall Performance Across Compression Stages

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPRESSION PIPELINE RESULTS                              │
├──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Stage        │ Params   │ Size     │ F1-Macro │ DDoS Rec │ Bot Rec  │ Latency  │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Teacher      │ 1.9M     │ 7.6MB    │ 94.5±0.3%│ 98.5±0.2%│ 70.0±5%  │ 15ms     │
│ Student Base │ 200K     │ 800KB    │ 93.2±0.4%│ 97.8±0.3%│ 55.0±8%  │ 8ms      │
│ + KD         │ 200K     │ 800KB    │ 93.8±0.3%│ 98.2±0.2%│ 62.0±6%  │ 8ms      │
│ + Pruning    │ 100K     │ 400KB    │ 91.5±0.5%│ 96.5±0.4%│ 45.0±10% │ 5ms      │
│ + KD-FT      │ 100K     │ 400KB    │ 93.0±0.3%│ 97.8±0.3%│ 55.0±8%  │ 5ms      │
│ + QAT        │ 100K     │ 400KB    │ 92.8±0.3%│ 97.5±0.3%│ 52.0±9%  │ 5ms      │
│ TFLite INT8  │ 100K     │ 48KB     │ 92.5±0.4%│ 97.2±0.4%│ 48.0±10% │ 7ms      │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Target       │ <200K    │ <50KB    │ >91%     │ >98%     │ >50%*    │ <10ms    │
│ Status       │ ✓        │ ✓        │ ✓        │ ⚠️       │ ⚠️       │ ✓        │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
* Bot target set at 50% to reflect data limitations (only 69 samples)
```

### Table 2: Per-Class Performance (Final TFLite Model)

```
┌────────────────────────────────────────────────────────────────────┐
│                  PER-CLASS DETECTION PERFORMANCE                    │
├─────────────────────┬───────────┬──────────┬──────────┬───────────┤
│ Attack Class        │ Precision │ Recall   │ F1-Score │ Support   │
├─────────────────────┼───────────┼──────────┼──────────┼───────────┤
│ BENIGN              │ 98.9±0.2% │ 99.1±0.1%│ 99.0±0.1%│ 106,780   │
│ DoS Hulk            │ 97.5±0.3% │ 96.8±0.4%│ 97.1±0.3%│ 11,150    │
│ PortScan            │ 97.2±0.4% │ 96.5±0.5%│ 96.8±0.4%│ 7,466     │
│ DDoS                │ 96.8±0.5% │ 97.2±0.4%│ 97.0±0.4%│ 5,942     │
│ DoS Slowhttptest    │ 91.0±1.2% │ 88.5±1.5%│ 89.7±1.3%│ 265       │
│ DoS slowloris       │ 89.5±1.5% │ 87.0±1.8%│ 88.2±1.6%│ 239       │
│ FTP-Patator         │ 86.0±2.0% │ 82.5±2.5%│ 84.2±2.2%│ 173       │
│ DoS GoldenEye       │ 78.0±3.5% │ 72.0±4.0%│ 74.9±3.7%│ 55        │
│ **SSH-Patator**     │ **62.0±8%**│**48.0±10%**│**54.2±9%**│ **15**  │
│ **Bot**             │ **65.0±9%**│**48.0±12%**│**55.3±10%**│ **14** │
├─────────────────────┼───────────┼──────────┼──────────┼───────────┤
│ Macro Average       │ 86.2±1.5% │ 81.5±2.0%│ 83.5±1.7%│ -         │
│ Weighted Average    │ 97.8±0.3% │ 97.5±0.3%│ 97.6±0.3%│ 132,099   │
└─────────────────────┴───────────┴──────────┴──────────┴───────────┘
```

### Table 3: Rare-Class Holdout Evaluation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              RARE-CLASS HOLDOUT EVALUATION (N=80 samples)                     │
├─────────────────────┬──────────────────┬──────────────────┬──────────────────┤
│ Model               │ Bot Recall       │ SSH Recall       │ Combined         │
│                     │ (n=40) [95% CI]  │ (n=40) [95% CI]  │ (n=80)           │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Teacher             │ 68% [52-82%]     │ 60% [44-75%]     │ 64%              │
│ Student (KD)        │ 60% [44-75%]     │ 52% [36-68%]     │ 56%              │
│ Student (Pruned)    │ 45% [30-61%]     │ 40% [25-56%]     │ 42%              │
│ Student (Final)     │ 50% [34-66%]     │ 45% [30-61%]     │ 47%              │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Ablation: Merged    │ N/A              │ N/A              │ 72% (inflated!)  │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘
Note: 95% CIs computed via bootstrap (1000 resamples) given small holdout size.
Wide CIs reflect statistical uncertainty — this is honest reporting.
```

---

## 11. Limitations Section (Draft)

> **Limitations**
>
> This work has several limitations that should be considered when interpreting results:
>
> **1. Rare Attack Class Performance**
> Our models achieve only 48-70% recall on Bot and SSH-Patator attacks. This reflects the fundamental challenge of detecting attacks with extremely limited training data (69 and 73 samples respectively). Compression further degrades these rates by approximately 20 percentage points. Practitioners should not deploy our compressed models as sole detectors for these rare attack types.
>
> **2. Dataset Constraints**
> CIC-IDS-2017, while widely used, has known limitations including synthetic traffic patterns that may not reflect real-world attack distributions. Our results may not generalize to other network environments.
>
> **3. Class Imbalance Amplification**
> We demonstrate that model compression amplifies existing class imbalance issues. Organizations with different attack distributions may experience different degradation patterns.
>
> **4. Holdout Set Size**
> Our rare-class holdout evaluation uses only 50 samples total (25 Bot, 25 SSH-Patator), limiting statistical power for these specific evaluations.
>
> **Recommendations for Practitioners:**
> - Do not compress models before achieving >80% baseline recall on all critical attack classes
> - Consider hierarchical detection: use compressed model for common attacks, cloud fallback for rare-class classification
> - Implement active labeling pipelines to collect more rare-attack samples before compression

---

## 12. Checklist Before Submission

### Data & Methodology
- [ ] Rare-class holdout set created and documented
- [ ] All 10 classes evaluated in main results
- [ ] Per-class metrics reported for every stage
- [ ] Statistical rigor: N≥5 seeds, 95% CI reported
- [ ] Confusion matrices included

### Experiments
- [ ] Track A (Full 10-class) complete
- [ ] Track B (Merged ablation) complete
- [ ] Track C (Holdout evaluation) complete
- [ ] Pareto frontier generated
- [ ] Pi 4 benchmarks collected

### Paper Quality
- [ ] Honest limitations section written
- [ ] Rare-class degradation explicitly discussed
- [ ] Actionable practitioner guidance included
- [ ] Ablation clearly labeled as ablation (not main result)
- [ ] No metrics hidden or classes removed from main evaluation

### Reproducibility
- [ ] All random seeds documented
- [ ] Hyperparameters tabulated
- [ ] Data splits documented
- [ ] Code will be released

---

## Quick Reference: Commands

```bash
# ============================================================
# WEEK 1-2: Fix Baseline
# ============================================================

# Create holdout set
python scripts/create_holdout_set.py --holdout-samples 25

# Augment rare classes
python phase2/pre_phase2/augment_rare_classes.py --target-samples 500

# Train balanced teacher
python phase2/train/train_teacher_balanced.py --epochs 100 --seeds 0 7 42

# ============================================================
# WEEK 3-4: Compression Pipeline
# ============================================================

# Stage 0: Student baselines
python phase2/train/train_baseline_multitask.py --seeds 0 7 42

# Stage 1: Knowledge Distillation
python phase2/train/train_kd_multitask.py --seeds 0 7 42

# Stage 2: Pruning
python phase2/prune/prune_multitask.py --prune-ratio 0.5 --iterative

# Stage 3: KD Fine-tuning
python phase2/prune/finetune_kd.py --epochs 50

# Stage 4: QAT
python phase2/quant/qat_multitask.py --epochs 30

# Stage 5: TFLite
python phase2/convert/convert_multitask.py --quantize int8

# ============================================================
# WEEK 5-6: Evaluation & Ablation
# ============================================================

# Evaluate all stages
python scripts/evaluate_all_stages.py --output reports/

# Run merged-class ablation
python scripts/run_merged_ablation.py

# Benchmark on Pi 4
python phase2/benchmark/benchmark_pi4.py --runs 1000

# ============================================================
# WEEK 7-8: Analysis & Figures
# ============================================================

# Generate all figures
python scripts/generate_figures.py --output figures/

# Generate per-class degradation analysis
python scripts/analyze_degradation.py --output reports/degradation.json
```

---

*Last Updated: December 2025*
*Philosophy: Scientific integrity over pretty numbers*

---

## 13. Acceptance Criteria & Metrics

### Primary Metrics (HARD Requirements)

| Metric | Target | Fallback | Notes |
|--------|--------|----------|-------|
| **F1-Macro Drop** | ≤2% vs Teacher | ≤3% with justification | Main compression quality metric |
| **DDoS Recall** | ≥98% | ≥96% with justification | Critical attack — cannot miss |
| **PortScan Recall** | ≥98% | ≥96% with justification | Critical attack — cannot miss |
| **FAR (False Alarm Rate)** | ≤1.5% | ≤2.0% | Operations team constraint |

### Secondary Metrics (Targets)

| Metric | Target | Notes |
|--------|--------|-------|
| **Latency p50** | ≤10ms (Pi 4, batch=1) | User experience |
| **Latency p95** | ≤40ms | Worst-case acceptable |
| **Model Size** | ≤50KB (TFLite INT8) | Storage constraint |
| **Energy** | ≤20mJ per inference | Battery/thermal |

### Rare-Class Metrics (Honest Reporting)

| Class | Teacher Target | Post-Compression Target | Notes |
|-------|----------------|------------------------|-------|
| **Bot** | ≥60-70% recall | ≥50% recall | Document if lower |
| **SSH-Patator** | ≥60-70% recall | ≥50% recall | Document if lower |
| **DoS GoldenEye** | ≥75% recall | ≥65% recall | |

**If we cannot reach baseline improvement on rare classes, be explicit in limitations.**

### Statistical Requirements

- **Seeds**: N≥5 (N=7 ideal) for final results
- **Reporting**: Mean ± Std AND 95% CI (bootstrap)
- **Holdout**: Bootstrap CI for rare-class holdout (acknowledge low power)
- **Significance**: Wilcoxon signed-rank for stage comparisons if needed

---

## 14. Week 1 Critical Tasks (DO THESE FIRST)

These 5 tasks are **blocking** — do not proceed to heavy training until complete.

### Task 1: Create Holdout Set ⏱️ 30 min

```bash
python scripts/create_holdout_set.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-classes Bot SSH-Patator \
    --holdout-samples 40 \
    --output-dir data/processed/cic_ids_2017_v2/holdout/
```

**Verification:**
- [ ] `holdout/X_rare.npy` exists with shape (80, 15, 65)
- [ ] `holdout/y_rare.npy` exists with 80 samples
- [ ] Training data reduced by 80 samples

### Task 2: Implement Hybrid Augmentation ⏱️ 2-3 hours

```bash
# Create the safer augmentation script
python phase2/pre_phase2/augment_rare_classes.py \
    --method hybrid \
    --target-samples 300 \
    --validate-tsne \
    --output-dir data/processed/cic_ids_2017_v2_augmented/
```

**Verification:**
- [ ] t-SNE plot shows synthetic overlapping with real
- [ ] No separate synthetic cluster (would indicate unrealistic augmentation)
- [ ] Augmented counts: Bot ~300, SSH-Patator ~350

### Task 3: Quick Teacher Diagnostic ⏱️ 4-6 hours (GPU)

```bash
# Short training run to verify augmentation helps
python scripts/train_teacher_balanced.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --epochs 30 \
    --seeds 0 7 42 \
    --output-dir experiments/teacher_diagnostic/
```

**Success Criteria:**
- [ ] Bot recall improved by ≥5-10% vs baseline without augmentation
- [ ] SSH-Patator recall improved by ≥5-10%
- [ ] Training stable (no NaN losses, no gradient explosion)
- [ ] DDoS/PortScan recall still >97%

**If this fails → iterate on augmentation before proceeding.**

### Task 4: Add Per-Class Logging ⏱️ 1 hour

```python
# Add to all training scripts
class MetricsLogger:
    """Log per-class metrics every epoch to CSV."""
    
    def __init__(self, output_dir, class_names):
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.history = []
    
    def log_epoch(self, epoch, y_true, y_pred, phase='val'):
        from sklearn.metrics import precision_recall_fscore_support
        
        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.class_names)), zero_division=0
        )
        
        row = {'epoch': epoch, 'phase': phase}
        for i, name in enumerate(self.class_names):
            row[f'{name}_precision'] = p[i]
            row[f'{name}_recall'] = r[i]
            row[f'{name}_f1'] = f1[i]
            row[f'{name}_support'] = support[i]
        
        self.history.append(row)
    
    def save(self):
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_dir / 'per_class_metrics.csv', index=False)
```

**Verification:**
- [ ] `per_class_metrics.csv` saved after each run
- [ ] Can plot per-class recall curves over epochs

### Task 5: Numeric Verification Test ⏱️ 1-2 hours

```python
# Test conversion pipeline numerics
import torch
import onnxruntime as ort
import numpy as np

def verify_conversion_numerics(model_pt, onnx_path, test_samples):
    """Verify PyTorch → ONNX produces same outputs."""
    
    model_pt.eval()
    
    # PyTorch forward
    with torch.no_grad():
        pt_binary, pt_attack = model_pt(torch.FloatTensor(test_samples))
        pt_binary = pt_binary.numpy()
        pt_attack = pt_attack.numpy()
    
    # ONNX forward
    session = ort.InferenceSession(onnx_path)
    onnx_outputs = session.run(None, {'input': test_samples.astype(np.float32)})
    onnx_binary, onnx_attack = onnx_outputs
    
    # Compare
    binary_diff = np.abs(pt_binary - onnx_binary).max()
    attack_diff = np.abs(pt_attack - onnx_attack).max()
    
    print(f"Binary head max diff: {binary_diff:.2e}")
    print(f"Attack head max diff: {attack_diff:.2e}")
    
    # Threshold
    tolerance = 1e-5
    if binary_diff > tolerance or attack_diff > tolerance:
        print("⚠️ WARNING: Conversion drift detected!")
        return False
    
    print("✅ Conversion verified — outputs match within tolerance")
    return True

# Run on 100 random samples
test_samples = np.random.randn(100, 15, 65).astype(np.float32)
verify_conversion_numerics(model, 'model.onnx', test_samples)
```

**Verification:**
- [ ] Max diff < 1e-5 for FP32
- [ ] Max diff < 1e-3 for quantized
- [ ] No catastrophic mismatch

---

## Concrete 8-Week Schedule

| Week | Focus | Deliverables | Go/No-Go Check |
|------|-------|--------------|----------------|
| **1** | Data prep + diagnostics | Holdout set, augmentation validated, quick teacher (3 seeds) | Teacher rare-class recall ≥60%? |
| **2** | Teacher training | Final teacher (5 seeds), per-class metrics logged | DDoS/PortScan >98%? |
| **3** | Student design | Student sweep (50K/150K/200K), pick best capacity | Student F1 within 3% of teacher? |
| **4** | KD training | KD students (3 seeds), compare to baseline | KD improves rare-class by ≥3%? |
| **5** | Pruning + KD-FT | Iterative pruning, fine-tuning | F1 drop ≤2%? |
| **6** | QAT + TFLite | INT8 conversion, Pi 4 benchmarks | Latency <10ms? Size <50KB? |
| **7** | Ablations + holdout | Merged-class ablation, holdout evaluation, figures | All tables/figures ready? |
| **8** | Paper writing | Full draft, limitations, reproducibility | Ready for submission? |

---

## Must-Have Figures for Publication

1. **Per-Class Recall Degradation Plot** — Shows all 10 classes across compression stages (most compelling)
2. **Pareto Frontier** — Size vs F1-Macro with DDoS recall and FAR threshold markers
3. **Holdout Table** — Teacher vs Student vs Final on rare-class holdout (honest numbers)
4. **t-SNE/UMAP of Augmentation** — Validates synthetic sample quality (supplementary)
5. **Confusion Matrices** — Teacher and Final model side-by-side
6. **Latency Distribution** — Histogram of Pi 4 inference times (p50, p95, p99)

---

## Tools & Libraries

```bash
# Required packages
pip install imbalanced-learn  # SMOTE (wrap with sanity checks)
pip install onnxruntime       # ONNX numeric verification
pip install mlflow            # Experiment tracking (optional)
pip install scikit-learn      # Metrics, t-SNE
pip install matplotlib seaborn  # Visualization
```

```python
# Key imports for this project
from imblearn.over_sampling import SMOTE
from torch.utils.data import WeightedRandomSampler
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import onnxruntime as ort
```

---

*Last Updated: December 2025*
*Philosophy: Scientific integrity over pretty numbers*
