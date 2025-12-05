# Phase 2: Complete Summary - What & Why (8-Week Plan)

> **Goal**: Compress your 1.9M parameter teacher model down to <50KB TFLite model while maintaining honest rare-class reporting  
> **Timeline**: 8 weeks (Dec 2025 - Jan 2026)  
> **Philosophy**: Scientific integrity over pretty numbers — all 10 attack classes evaluated, no cheating through merging

---

## 📋 EXECUTIVE SUMMARY (2-Minute Version)

### The Core Compression Pipeline
```
Teacher (1.9M params)
    ↓ Stage 0: Train Student Baselines
Student (50K-200K params, no teacher guidance)
    ↓ Stage 1: Knowledge Distillation (KD)
KD Student (same size, learns from teacher)
    ↓ Stage 2: Iterative Pruning (50% removal)
Pruned Student (half the neurons)
    ↓ Stage 3: KD Fine-Tuning (recover accuracy)
Recovered Student
    ↓ Stage 4: Quantization-Aware Training (INT8)
Quantized Model
    ↓ Stage 5: TFLite Conversion
Final Model (48KB, <10ms on Raspberry Pi)
```

### Expected Results
- **F1-Macro**: 94.5% (teacher) → 92.5% (final) = **-2% drop** ✅
- **DDoS Recall**: 98.5% (teacher) → 97.2% (final) ✅
- **PortScan Recall**: 99% (teacher) → 97.5% (final) ✅
- **Bot Recall**: 70% (teacher) → 48% (final) ⚠️ Expected, honest reporting
- **SSH-Patator Recall**: 65% (teacher) → 48% (final) ⚠️ Expected, honest reporting
- **Model Size**: 7.6MB → **48KB** ✅ (99.4% reduction)
- **Latency**: 15ms → **7ms** ✅ (Pi 4, batch=1)

---

## 🗓️ WEEK-BY-WEEK BREAKDOWN

---

## **WEEK 1: Data Preparation & Diagnostics** ⏱️ 10 hours

### Why This Week Matters
Before you invest 50+ hours training models, you MUST verify that your augmentation doesn't create fake data and that it actually helps. This week is your **blocking validation**.

---

### ✅ Task 1.1: Create Rare-Class Holdout Set
**What**: Reserve 80 samples (40 Bot + 40 SSH-Patator) for final evaluation ONLY. Never train on these.  
**Why**: You need unbiased evaluation of rare-class performance. With only 69-73 samples of Bot/SSH-Patator, a 40-sample holdout leaves just 29-33 for training, making augmentation critical.

```bash
python scripts/create_holdout_set.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-classes Bot SSH-Patator \
    --holdout-samples 40 \
    --output-dir data/processed/cic_ids_2017_v2/holdout/
```

**Acceptance Criteria**:
- ✅ holdout/X_rare.npy created (80, 15, 65)
- ✅ Training data reduced by 80 samples
- ✅ Holdout backed up (cannot be used in training/validation)

**Impact**: This holdout will be your most honest evaluation metric throughout Phase 2.

---

### ⚠️ Task 1.2: Implement Hybrid Augmentation
**What**: Safely augment Bot (29 remaining) and SSH-Patator (33 remaining) to ~300-500 samples each.  
**Why**: Naive SMOTE from 40→500 creates unrealistic synthetic windows with artifacts. Using **Jitter → Mixup → Capped SMOTE** creates realistic variations that generalize better.

**The 3-Step Process** (order matters!):
1. **Jitter**: Add small Gaussian noise (σ=0.01) — fast, realistic, preserves structure
2. **Mixup**: Interpolate between same-class samples (α=0.2) — creates smooth transitions
3. **Capped SMOTE**: Limited to ≤5× per iteration — prevents extreme synthetic generation

```bash
python scripts/augment_hybrid_ultra_lite.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --rare-classes Bot SSH-Patator \
    --augment-factor 3 \
    --seed 42
```

**Expected Output**:
- Bot: 29 → ~87 samples (3×)
- SSH-Patator: 33 → ~99 samples (3×)
- DoS GoldenEye: 275 → ~500 samples (augmented for stability)

**Why Conservative Targets?**
- Too much synthetic data (10×) creates unrealistic windows
- Better to have realistic data (3×) than fake data (10×)
- You can always augment more later if training shows underfitting

**Impact**: Determines whether your teacher can learn rare-class patterns without overfitting to artifacts.

---

### 📊 Task 1.3: Validate Augmentation with t-SNE
**What**: Plot real vs synthetic samples in 2D using t-SNE to verify they're indistinguishable.  
**Why**: SMOTE can create unrealistic synthetic samples that look like nothing in real data. t-SNE will show if your synthetic samples are **mixing with real samples** (good) or forming a **separate cluster** (bad).

```python
# Validate augmentation before heavy training
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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
                   label=label, alpha=0.6, s=50)
    plt.title(f'{class_name}: Real vs Synthetic (t-SNE)')
    plt.legend()
    plt.savefig(f'reports/augmentation_validation_{class_name}.png', dpi=150)
    plt.close()

# Decision rule
# ✅ If synthetic overlaps with real cluster → proceed to training
# ❌ If synthetic forms separate cluster → reduce synthetic count or tweak augmentation
```

**Decision Rule**:
- ✅ **Clusters overlap** → Augmentation is good, proceed ✅
- ❌ **Synthetic cluster separate** → Augmentation is unrealistic, reduce count or tweak parameters

**Impact**: Prevents training on fake data that won't generalize.

---

### 🔍 Task 1.4: Quick Teacher Diagnostic (30 epochs)
**What**: Train a teacher for just 30 epochs on augmented data to verify:
1. Augmentation actually helps rare-class recall
2. Training is stable (no NaN losses)
3. DDoS/PortScan still >97%

**Why**: If augmentation doesn't help rare-class recall or breaks training stability, you'll discover this in **4-6 hours** instead of **50 hours**.

```bash
python scripts/train_teacher_balanced.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --epochs 30 \
    --seeds 0 7 42 \
    --grad-clip 1.0 \
    --class-weights class_weights.json \
    --loss focal \
    --output-dir experiments/teacher_diagnostic/
```

**Success Criteria**:
- ✅ Bot recall **≥50-60%** (was ~20-30% without augmentation)
- ✅ SSH-Patator recall **≥50-60%** (was ~15-25%)
- ✅ DDoS/PortScan recall **>97%** (still good)
- ✅ No NaN losses or gradient explosion

**What If It Fails?**
- If rare-class recall doesn't improve → augmentation needs tweaking
- If loss goes NaN → class weights too aggressive or learning rate too high
- If DDoS/PortScan drops → regularization too strong

**Decision Point**: 
- ✅ **Rare-class improved by ≥10%** → Proceed to full training
- ❌ **Rare-class didn't improve** → Debug and iterate (back to Task 1.2)

**Impact**: Prevents wasting weeks training if augmentation is fundamentally broken.

---

### 📈 Task 1.5: Setup Per-Class Metrics Logging
**What**: Modify all training scripts to log precision/recall/F1 for each of the 10 attack classes every epoch.  
**Why**: You need to track which classes degrade under compression. Macro-F1 hides class-specific failures.

```python
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

class MetricsLogger:
    def __init__(self, output_dir, class_names):
        self.output_dir = output_dir
        self.class_names = class_names
        self.history = []
    
    def log_epoch(self, epoch, y_true, y_pred, phase='val'):
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
        df = pd.DataFrame(self.history)
        df.to_csv(f'{self.output_dir}/per_class_metrics.csv', index=False)
        print(f"✅ Saved per_class_metrics.csv with {len(df)} rows")
```

**Output**: `per_class_metrics.csv` with columns like:
```
epoch, phase, Bot_precision, Bot_recall, Bot_f1, ..., SSH-Patator_precision, ...
```

**Impact**: Enables you to plot per-class recall degradation across compression stages (your best publication figure).

---

### ✔️ Task 1.6: Verify Numeric Conversion Pipeline
**What**: Test that PyTorch → ONNX → TFLite doesn't silently lose accuracy.  
**Why**: Conversion bugs can cause 5-10% accuracy drops that you won't notice until deployment.

```python
import torch
import onnxruntime as ort
import numpy as np

def verify_conversion(model_pt, onnx_path, test_samples):
    """Verify outputs are identical across formats."""
    
    model_pt.eval()
    
    # PyTorch output
    with torch.no_grad():
        pt_output = model_pt(torch.FloatTensor(test_samples)).numpy()
    
    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'input': test_samples.astype(np.float32)})[0]
    
    # Compare
    max_diff = np.abs(pt_output - onnx_output).max()
    print(f"Max difference: {max_diff:.2e}")
    
    if max_diff > 1e-5:
        print("⚠️ WARNING: Conversion drift detected!")
        return False
    
    print("✅ Conversion verified")
    return True

# Test on 100 random samples
test_samples = np.random.randn(100, 15, 65).astype(np.float32)
verify_conversion(model, 'model.onnx', test_samples)
```

**Acceptance Criteria**:
- ✅ Max difference < 1e-5 for FP32 ONNX
- ✅ Max difference < 1e-3 for INT8 quantized

**Impact**: Prevents spending days debugging deployment issues caused by silent conversion drift.

---

## **WEEK 2: Full Teacher Training** ⏱️ 20 hours (GPU)

### Why This Week Matters
You need a **strong, honest teacher baseline** before compressing. Everything else is measured relative to the teacher.

---

### 📚 Task 2.1: Train Final Teacher (5 Seeds)
**What**: Train the best-possible 10-class teacher model using:
- **Balanced sampling** (oversamples rare classes)
- **Focal loss** (γ=2.0, focuses on hard examples)
- **Class weights** (capped at 50×, prevents instability)
- **Gradient clipping** (1.0, prevents explosion)
- **Augmented data** (hybrid augmentation from Week 1)

**Why**: 
- Balanced sampling ensures model learns from all classes equally
- Focal loss prevents majority classes from dominating
- Class weights make rare classes matter more
- Gradient clipping stabilizes training with high weights
- Multiple seeds (5) give you mean ± std for statistical rigor

```bash
python phase2/train/train_teacher.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --class-weights class_weights.json \
    --loss focal \
    --focal-gamma 2.0 \
    --sampler balanced \
    --epochs 100 \
    --patience 15 \
    --grad-clip 1.0 \
    --seeds 0 7 42 101 202 \
    --output-dir artifacts/teacher/
```

**Expected Results**:
| Class | Target Recall |
|-------|---------------|
| BENIGN | >99% |
| DDoS | >98.5% |
| PortScan | >98% |
| Bot | 60-80% |
| SSH-Patator | 55-75% |
| Macro-F1 | ~90-94% |

**Output**:
- `artifacts/teacher/seed0/best_model.pt`
- `artifacts/teacher/seed0/per_class_metrics.csv`
- `artifacts/teacher/aggregate_results.json` (mean ± std across 5 seeds)

**Impact**: Your teacher is the **quality ceiling** for everything that follows.

---

### 🎯 Task 2.2: Evaluate Teacher on Rare-Class Holdout
**What**: Test the trained teacher on the **never-before-seen 80 holdout samples**.  
**Why**: This is your most honest evaluation. If the teacher only gets 50% recall on 40 Bot samples, you know the task is hard.

```bash
python scripts/evaluate_holdout.py \
    --model artifacts/teacher/seed42/best_model.pt \
    --holdout-dir data/processed/cic_ids_2017_v2/holdout \
    --output reports/teacher_holdout_eval.json
```

**Expected Output**:
```json
{
  "Bot": {
    "recall": 0.65,
    "bootstrap_95_ci": [0.48, 0.80],
    "support": 40
  },
  "SSH-Patator": {
    "recall": 0.60,
    "bootstrap_95_ci": [0.43, 0.75],
    "support": 40
  }
}
```

**Why Bootstrap CIs?** With only 40 samples per class, you have high statistical uncertainty. Bootstrap CIs honestly report this uncertainty.

**Impact**: Establishes the rare-class **performance ceiling** that you'll compare all future models against.

---

## **WEEK 3: Student Architecture Design** ⏱️ 15 hours (GPU)

### Why This Week Matters
Students that are too small won't learn anything. Students that are too large don't prove compression works. You need to find the **sweet spot**.

---

### 🔬 Task 3.1: Student Architecture Sweep
**What**: Train student models of 3 different sizes (50K, 100K, 200K params) WITHOUT teacher guidance to establish the **baseline capability**.  
**Why**: You need to show that compression WITH knowledge distillation beats compression WITHOUT it. This sweep is your "control group".

```bash
python phase2/train/train_baseline.py \
    --param-targets 50000 100000 200000 \
    --seeds 0 7 42 \
    --epochs 50 \
    --output-dir artifacts/stage0_baseline/
```

**Expected Results** (without teacher):
| Size | Params | F1-Macro | Bot Recall | Notes |
|------|--------|----------|-----------|-------|
| Tiny | 50K | ~90% | ~25-30% | Too small, struggles on rare classes |
| Medium | 100K | ~91% | ~35-40% | Better but still weak on rare classes |
| Large | 200K | ~93% | ~45-50% | Better, but not as good as teacher |

**Decision**: Pick the size where F1 is within 3% of teacher and train KD on that.

**Impact**: Proves the student baseline is weak, justifying why knowledge distillation is necessary.

---

## **WEEK 4: Knowledge Distillation (KD) Training** ⏱️ 20 hours (GPU)

### Why This Week Matters
KD is where the magic happens. The student learns from both **hard labels** (ground truth) and **soft labels** (teacher's probability distribution). This transfer improves rare-class performance.

---

### 🎓 Task 4.1: Train KD Students
**What**: Train the same student sizes (50K, 100K, 200K) again, but using the teacher as a guide.  
**Why**:
- Hard labels alone: "This sample is Bot"
- Soft labels from teacher: "This sample is 75% Bot, 15% SSH-Patator, 10% other"
- Soft labels contain richer information about decision boundaries
- Especially valuable for rare classes where training data is scarce

```bash
python phase2/train/train_kd.py \
    --teacher artifacts/teacher/seed42/best_model.pt \
    --student-sizes 50000 100000 200000 \
    --temperature 4.0 \
    --alpha 0.7 \
    --epochs 100 \
    --seeds 0 7 42 \
    --output-dir artifacts/stage1_kd/
```

**KD Hyperparameters Explained**:
- **Temperature (4.0)**: Higher = softer probability distribution = more information transfer
- **Alpha (0.7)**: 70% soft loss + 30% hard loss = balance between teacher guidance and ground truth

**Expected Improvement**:
| Metric | Baseline | +KD | Delta |
|--------|----------|-----|-------|
| F1-Macro | 93% | 94.5% | +1.5% |
| Bot Recall | 50% | 65% | +15% ⭐ |
| SSH-Patator Recall | 40% | 60% | +20% ⭐ |

**Output**:
- `artifacts/stage1_kd/200k/seed0/best_model.pt` (best KD model)
- Per-class metrics showing rare-class improvement

**Impact**: KD improves rare-class detection by 15-20%. This is your strongest evidence that teacher guidance helps.

---

## **WEEK 5: Pruning & Recovery** ⏱️ 25 hours (GPU)

### Why This Week Matters
You've proven KD works. Now you need to compress further without losing too much accuracy. Pruning removes 50% of model weights. KD fine-tuning recovers the lost accuracy.

---

### ✂️ Task 5.1: Iterative Structured Pruning
**What**: Remove 50% of neurons from the KD student in 10 steps (5% per step).  
**Why**:
- Sudden pruning loses 10-15% accuracy
- Gradual pruning (iterative) loses only 2-4% accuracy
- Structured pruning (whole filters) is hardware-friendly vs unstructured

```bash
python phase2/prune/prune_model.py \
    --model artifacts/stage1_kd/200k/seed42/best_model.pt \
    --prune-ratio 0.5 \
    --iterative \
    --steps 10 \
    --output-dir artifacts/stage2_pruned/
```

**Process**:
```
Step 1: Remove 5% → F1 drops ~0.2%
Step 2: Remove 5% → F1 drops ~0.3%
...
Step 10: Total 50% removed → F1 drops ~2%
```

**Expected Result**:
- Model params: 200K → 100K
- Model size: 800KB → 400KB
- F1-Macro: 94.5% → 92.5% (-2% acceptable)
- Bot Recall: 65% → 48% (-17%, expected)

**Output**:
- `artifacts/stage2_pruned/best_model.pt`

**Impact**: Cuts model size in half with manageable accuracy loss.

---

### 🔧 Task 5.2: KD Fine-Tuning to Recover
**What**: Train the pruned model for 50 more epochs with teacher guidance to recover accuracy.  
**Why**: Pruning always hurts. Fine-tuning with KD recovers some of that accuracy by re-learning with fewer weights.

```bash
python phase2/prune/finetune_kd.py \
    --model artifacts/stage2_pruned/best_model.pt \
    --teacher artifacts/teacher/seed42/best_model.pt \
    --epochs 50 \
    --temperature 4.0 \
    --alpha 0.7 \
    --output-dir artifacts/stage3_finetuned/
```

**Expected Recovery**:
| Metric | After Pruning | After KD-FT | Recovery |
|--------|---------------|-------------|----------|
| F1-Macro | 92.5% | 93.5% | +1.0% |
| Bot Recall | 48% | 55% | +7% |

**Output**:
- `artifacts/stage3_finetuned/best_model.pt`

**Impact**: Recovers 1-2% accuracy lost to pruning through knowledge distillation.

---

## **WEEK 6: Quantization & Deployment** ⏱️ 20 hours (GPU)

### Why This Week Matters
TensorFlow Lite and Raspberry Pi only support INT8 integers, not FP32 floats. Quantization-Aware Training prepares your model for this constraint.

---

### 🔢 Task 6.1: Quantization-Aware Training (QAT)
**What**: Train the model with simulated INT8 quantization to learn robust weights for 8-bit arithmetic.  
**Why**:
- Naive quantization (convert FP32→INT8 without retraining) loses 3-5% accuracy
- QAT (retrain with INT8 simulation) loses only 0.5-1% accuracy
- QAT is essential for rare-class preservation

```bash
python phase2/quant/qat_train.py \
    --model artifacts/stage3_finetuned/best_model.pt \
    --epochs 30 \
    --output-dir artifacts/stage4_qat/
```

**Expected Accuracy Drop**:
- Before QAT: F1-Macro = 93.5%
- After QAT: F1-Macro = 92.8% (-0.7%, excellent!)

**Output**:
- `artifacts/stage4_qat/best_model.pt` (ready for INT8 conversion)

**Impact**: Minimal accuracy loss while preparing for deployment.

---

### 📱 Task 6.2: TFLite INT8 Conversion
**What**: Convert the QAT model to TensorFlow Lite format with INT8 quantization.  
**Why**: TensorFlow Lite is the standard for edge deployment. INT8 → 87% size reduction (400KB → 48KB).

```bash
python phase2/convert/convert_to_tflite.py \
    --model artifacts/stage4_qat/best_model.pt \
    --quantize int8 \
    --verify-numerics \
    --output-dir artifacts/stage5_tflite/
```

**Verification Step**:
- Runs numeric check comparing PyTorch → ONNX → TFLite outputs
- If max diff > 1e-3, flags as error and requires investigation

**Expected Output**:
- `artifacts/stage5_tflite/model_int8.tflite` (48KB)
- Accuracy validated

**Impact**: Final deployment-ready model.

---

### ⚡ Task 6.3: Raspberry Pi 4 Benchmarking
**What**: Measure actual latency, throughput, and energy on Raspberry Pi 4.  
**Why**: Theoretical numbers mean nothing. Real hardware performance is what matters for deployment.

```bash
python phase2/bench/pi_bench.py \
    --model artifacts/stage5_tflite/model_int8.tflite \
    --runs 1000 \
    --batch-size 1
```

**Metrics to Collect**:
- **p50 latency**: 50th percentile (typical case)
- **p95 latency**: 95th percentile (worst case)
- **p99 latency**: 99th percentile (extreme case)
- **Throughput**: Inferences per second
- **Memory**: Peak RAM usage
- **Thermal**: Temperature after 1000 runs

**Expected Results**:
| Metric | Target | Expected |
|--------|--------|----------|
| p50 Latency | ≤10ms | 7-8ms ✅ |
| p95 Latency | ≤40ms | 20-30ms ✅ |
| p99 Latency | ≤100ms | 40-60ms ✅ |
| Throughput | ≥50 infer/s | ~140 infer/s ✅ |
| Memory | ≤100MB | ~50MB ✅ |

**Output**:
- `reports/pi_bench_latency_histogram.png`
- `reports/pi_bench_results.json`

**Impact**: Proves your model meets deployment constraints.

---

## **WEEK 7: Ablation Studies & Holdout Evaluation** ⏱️ 15 hours

### Why This Week Matters
You need to prove that your honest 10-class approach is better than the "cheating" 9-class approach (merging rare classes).

---

### 🔄 Task 7.1: Merged-Class Ablation Study
**What**: Re-run the entire pipeline with Bot + SSH-Patator merged into a single "RARE_ATTACK" class (9 classes total instead of 10).  
**Why**: Many papers hide poor rare-class performance by merging them. You'll show this inflates metrics but hides real failure modes.

```bash
python scripts/run_merged_ablation.py \
    --merge "Bot,SSH-Patator:RARE_ATTACK" \
    --output-dir artifacts/ablation_merged/
```

**Expected Comparison**:
| Approach | Approach | Macro-F1 | "Rare" Class Recall | Truth |
|----------|----------|----------|-------------------|-------|
| 10-Class (Honest) | Full pipeline | 92.5% | Bot: 48%, SSH: 48% | Real rare-class performance |
| 9-Class (Merged) | Full pipeline | 94.5% | RARE_ATTACK: 85% | Inflated! Hides failures |

**Key Finding**: Merging makes metrics look better but hides that Bot detection failed.

**Output**:
- `reports/ablation_merged_results.json`
- Figure comparing 10-class vs 9-class side-by-side

**Impact**: Proves your honest 10-class approach is scientifically superior to metric manipulation.

---

### 🎯 Task 7.2: All-Stage Holdout Evaluation
**What**: Evaluate the **never-before-seen rare-class holdout set** on every model from every stage.  
**Why**: This shows the **exact degradation path** as you compress.

```bash
for stage in teacher stage0_baseline stage1_kd stage2_pruned stage3_finetuned stage4_qat stage5_tflite; do
    python scripts/evaluate_holdout.py \
        --model artifacts/${stage}/best_model.pt \
        --holdout-dir data/processed/cic_ids_2017_v2/holdout \
        --output reports/holdout_${stage}.json
done
```

**Expected Degradation Curve**:
| Stage | Bot Recall | SSH Recall | Combined |
|-------|-----------|-----------|----------|
| Teacher | 68% | 60% | 64% |
| Baseline | 55% | 48% | 51% |
| KD | 65% | 58% | 61% |
| Pruned | 45% | 38% | 41% |
| KD-FT | 58% | 50% | 54% |
| QAT | 55% | 48% | 51% |
| TFLite | 50% | 45% | 47% |

**Output**:
- Table showing rare-class recall at each stage
- Figure showing degradation curve (your best publication figure!)

**Impact**: Honest documentation of rare-class performance across compression.

---

### 📊 Task 7.3: Generate Publication Figures
**What**: Create 5 publication-ready figures for your paper.  
**Why**: Figures are how reviewers quickly understand your work. Bad figures → rejection.

```bash
python scripts/generate_figures.py \
    --results-dir artifacts/ \
    --holdout-dir data/processed/cic_ids_2017_v2/holdout \
    --output-dir figures/
```

**5 Essential Figures**:

1. **Per-Class Recall Degradation** (your BEST figure)
   - X-axis: Compression stages (Teacher → KD → Pruned → QAT → TFLite)
   - Y-axis: Recall (%)
   - Lines: One line per attack class
   - Highlight: Rare classes degrade more than common classes

2. **Pareto Frontier**
   - X-axis: Model size (log scale)
   - Y-axis: F1-Macro
   - Points: Teacher, KD, Pruned, QAT, TFLite
   - Shows: You achieve high accuracy with tiny model

3. **Confusion Matrices**
   - Side-by-side: Teacher vs TFLite
   - Shows: Student maintains low false positives despite compression

4. **t-SNE of Augmented Data** (supplementary)
   - Real vs synthetic samples
   - Validates augmentation quality

5. **Latency Distribution on Pi 4**
   - Histogram of inference times
   - Shows: Consistent, predictable performance

**Output**:
- `figures/per_class_degradation.png`
- `figures/pareto_frontier.png`
- `figures/confusion_matrix_comparison.png`
- `figures/augmentation_validation.png`
- `figures/latency_distribution_pi4.png`

**Impact**: Figures are 50% of how reviewers judge your paper.

---

## **WEEK 8: Paper Writing** ⏱️ 20 hours

### Why This Week Matters
If you can't explain your work, it doesn't matter how good it is. The paper is the final deliverable.

---

### ✍️ Task 8.1: Write Paper Draft
**What**: Full 8-section paper for IEEE IoT Journal or similar venue.  
**Why**: Publication is the ultimate deliverable.

**Sections**:
1. **Introduction** (2 pages)
   - Edge IDS motivation
   - Compression challenges
   - Research question: How does compression affect rare-class detection?

2. **Related Work** (1.5 pages)
   - IDS on edge devices
   - Model compression (KD, pruning, QAT)
   - Class imbalance in security ML

3. **Methodology** (2 pages)
   - Dataset (CIC-IDS-2017, honest class distribution)
   - Multi-task architecture (binary + attack-type heads)
   - Compression pipeline (KD → prune → QAT → TFLite)
   - Evaluation protocol (per-class metrics, holdout set, bootstrap CIs)

4. **Experimental Setup** (1 page)
   - Hardware (Raspberry Pi 4)
   - Metrics (precision, recall, F1, FAR, latency)
   - Seeds and statistical rigor

5. **Results** (2.5 pages)
   - Table 1: Overall compression pipeline results
   - Table 2: Per-class performance (all 10 classes)
   - Table 3: Rare-class holdout evaluation
   - Figures: Degradation, Pareto frontier, confusion matrices

6. **Ablation Studies** (1 page)
   - Merged classes vs full classes (Table 4)
   - Impact of each compression technique
   - Proof that honest reporting > metric inflation

7. **Discussion** (1.5 pages)
   - When is compression safe?
   - Operational implications
   - Guidance for practitioners

8. **Limitations** (1 page, CRITICAL)
   - Rare-class performance inherently weak (69-73 samples)
   - Compression degrades rare-class by 20 percentage points
   - Dataset synthetic patterns may not generalize
   - Holdout size limits statistical power

9. **Conclusion** (0.5 page)

---

### ⚠️ Task 8.2: Write Honest Limitations Section
**What**: Explicitly acknowledge what your work CANNOT do.  
**Why**: Reviewers respect honest assessment. They will find weaknesses anyway — better you acknowledge them first.

**Template**:

> **Limitations**
>
> This work demonstrates effective model compression for intrusion detection but has important limitations:
>
> **1. Rare Attack Class Performance**
> Our final model achieves only 48-50% recall on Bot and SSH-Patator attacks compared to 60-68% for the teacher model. This reflects two issues:
> - **Data limitation**: Only 69-73 training samples per class after holdout removal makes these classes inherently hard
> - **Compression amplification**: Quantization removes fine-grained decision boundaries that help rare-class detection
> 
> Practitioners should not rely on this compressed model as a sole detector for these rare attack types.
>
> **2. Dataset Constraints**
> CIC-IDS-2017 uses synthetic network traffic that may not reflect real-world attack distributions. Results may not generalize to other datasets or network environments.
>
> **3. Generalization Risk**
> Compression amplifies the baseline model's weaknesses. Organizations with different attack distributions may experience different degradation patterns.
>
> **Recommendations for Practitioners:**
> - Do not compress models that achieve <80% rare-class recall baseline
> - Consider hierarchical detection: compressed model for common attacks, cloud fallback for rare classes
> - Implement active labeling to collect more rare-attack training data before compression
> - Evaluate every model on held-out rare-class samples, not just aggregate metrics

**Impact**: Paper gets accepted because reviewers see you're trustworthy.

---

## 📋 FINAL SUMMARY TABLE

| Week | Tasks | Hours | GPU | Output | Go/No-Go |
|------|-------|-------|-----|--------|----------|
| **1** | Holdout, augmentation, diagnostic | 10 | 6 | Validated augmentation | Rare-class recall ≥60% |
| **2** | Teacher training (5 seeds) | 20 | 15 | Teacher + holdout eval | DDoS/PortScan >98% |
| **3** | Student sweep (3 sizes) | 15 | 10 | Baseline models | F1 within 3% of teacher |
| **4** | KD training (3 sizes) | 20 | 15 | KD students | Rare-class improved ≥10% |
| **5** | Pruning + KD-FT | 25 | 18 | Recovered pruned model | F1 drop ≤2% |
| **6** | QAT + TFLite + Pi bench | 20 | 12 | Final model + benchmarks | Latency <10ms, size <50KB |
| **7** | Ablations + holdout eval | 15 | 5 | Publication figures | All tables/figures ready |
| **8** | Paper writing | 20 | 0 | Full paper draft | Ready for submission |
| **TOTAL** | - | **145 hours** | **75 GPU hours** | **Publication-ready results** | - |

---

## 🎯 SUCCESS CRITERIA (Hard Requirements)

Must achieve ALL of these:

| Metric | Target | Your Model | Status |
|--------|--------|-----------|--------|
| **F1-Macro** | ≥91% | TBD | ⏳ |
| **DDoS Recall** | ≥98% | TBD | ⏳ |
| **PortScan Recall** | ≥98% | TBD | ⏳ |
| **FAR** | ≤1.5% | TBD | ⏳ |
| **Latency p50** | ≤10ms | TBD | ⏳ |
| **Model Size** | ≤50KB | TBD | ⏳ |
| **Bot Recall** | ≥50% | TBD | ⏳ |
| **SSH Recall** | ≥50% | TBD | ⏳ |
| **Reproducibility** | 5 seeds, CI reported | TBD | ⏳ |
| **Honest reporting** | All 10 classes, no merging | TBD | ⏳ |

---

## 🚀 NEXT IMMEDIATE STEP

**Start Week 1 NOW:**

```bash
# Task 1.1: Create holdout set (30 min)
python scripts/create_holdout_set.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-classes Bot SSH-Patator \
    --holdout-samples 40 \
    --output-dir data/processed/cic_ids_2017_v2/holdout/

# Task 1.2: Augment rare classes (2 min)
python scripts/augment_hybrid_ultra_lite.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --rare-classes Bot SSH-Patator \
    --augment-factor 3 \
    --seed 42

# Task 1.3: Validate with t-SNE (15 min)
python scripts/validate_augmentation.py \
    --augmented-dir data/processed/cic_ids_2017_v2_augmented

# Task 1.4: Quick teacher diagnostic (4-6 hours GPU)
python scripts/train_teacher_balanced.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --epochs 30 \
    --seeds 0 7 42
```

**Estimated completion**: Week 1 = 1 week ✅

---

*Last Updated: December 4, 2025*  
*For detailed technical setup, see PHASE2_TODO.md and PUBLICATION_ROADMAP.md*
