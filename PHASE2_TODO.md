# Phase 2: Complete TODO List with Dynamic Progress Tracking

> **Objective**: Compress DS-CNN IDS model for Raspberry Pi 4 while maintaining honest rare-class reporting
> **Timeline**: 8 weeks (December 2025 - January 2026)
> **Philosophy**: Scientific integrity over pretty numbers — keep all 10 classes, report failures honestly

---

## 📊 Progress Dashboard

| Phase | Status | Progress | Key Metric Target | Current |
|-------|--------|----------|-------------------|---------|
| **Week 1: Data Prep** | 🔄 IN PROGRESS | 40% | Holdout created, augmentation validated | Holdout ✅ |
| **Week 2: Baseline** | ⏳ PENDING | 0% | Teacher rare-class recall ≥60-70% | - |
| **Week 3: Student Design** | ⏳ PENDING | 0% | Student F1 within 3% of teacher | - |
| **Week 4: KD Training** | ⏳ PENDING | 0% | KD improves rare-class by ≥3% | - |
| **Week 5: Pruning** | ⏳ PENDING | 0% | F1 drop ≤2% after iterative pruning | - |
| **Week 6: QAT + TFLite** | ⏳ PENDING | 0% | <50KB, <10ms latency | - |
| **Week 7: Ablations** | ⏳ PENDING | 0% | All figures ready | - |
| **Week 8: Paper** | ⏳ PENDING | 0% | Submission ready | - |

---

## 🎯 Acceptance Criteria (HARD Requirements)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| F1-Macro Drop vs Teacher | ≤2% | - | ⏳ |
| DDoS Recall | ≥98% | - | ⏳ |
| PortScan Recall | ≥98% | - | ⏳ |
| False Alarm Rate | ≤1.5% | - | ⏳ |
| Latency p50 (Pi 4) | ≤10ms | - | ⏳ |
| Model Size (TFLite INT8) | ≤50KB | - | ⏳ |
| Bot Recall (Teacher) | ≥60-70% | - | ⏳ |
| SSH-Patator Recall (Teacher) | ≥60-70% | - | ⏳ |

---

## Week 1: Data Preparation & Diagnostics

### ✅ Task 1.1: Create Holdout Set [COMPLETED]
**Objective**: Reserve 80 rare-class samples for unbiased final evaluation

```bash
# Command executed:
python scripts/create_holdout_set.py \
    --data-dir data/processed/cic_ids_2017_v2 \
    --holdout-classes Bot SSH-Patator \
    --holdout-samples 40 \
    --seed 42
```

**Results**:
- [x] Holdout created: `data/processed/cic_ids_2017_v2/holdout/`
- [x] Bot: 40 samples reserved (29 remaining in training)
- [x] SSH-Patator: 40 samples reserved (33 remaining in training)
- [x] Original data backed up: `train_original_backup/`
- [x] Training reduced: 660,503 → 660,423 samples

**Verification**:
```
holdout/X_rare.npy: (80, 15, 65) - ✅
holdout/y_rare.npy: (80,) - ✅
holdout/attack_types_rare.npy: (80,) - ✅
```

---

### 🔄 Task 1.2: Implement Hybrid Augmentation [IN PROGRESS]
**Objective**: Safer augmentation that doesn't create unrealistic synthetic samples

**Method sequence** (order matters!):
1. **Jitter**: Add Gaussian noise (σ=0.01) to create realistic variations
2. **Mixup**: Interpolate within same class (α=0.2)
3. **Capped SMOTE**: ≤5× per iteration to prevent artifacts

**Target counts** (conservative):
| Class | Original | After Augmentation |
|-------|----------|-------------------|
| Bot | 29 | ~300-500 |
| SSH-Patator | 33 | ~350-500 |
| DoS GoldenEye | 275 | ~500 |

**Validation**: t-SNE plot must show synthetic overlapping with real

---

### ⏳ Task 1.3: t-SNE Validation [PENDING]
**Objective**: Verify synthetic samples look realistic

**Decision Rule**:
- ✅ Synthetic cluster **overlaps** real cluster → proceed
- ❌ Synthetic cluster **separate** → tweak parameters, reduce count

**Output**: `reports/augmentation_validation_{class}.png`

---

### ⏳ Task 1.4: Quick Teacher Diagnostic [PENDING]
**Objective**: Verify augmentation actually helps rare-class recall

```bash
python scripts/train_teacher_balanced.py \
    --data-dir data/processed/cic_ids_2017_v2_augmented \
    --epochs 30 \
    --seeds 0 7 42 \
    --grad-clip 1.0 \
    --output-dir experiments/teacher_diagnostic/
```

**Success Criteria**:
- [ ] Bot recall improved by ≥10% vs no augmentation
- [ ] SSH-Patator recall improved by ≥10%
- [ ] No NaN losses or gradient explosion
- [ ] DDoS/PortScan still >97%

**GO/NO-GO**: If rare-class recall <60%, iterate on augmentation

---

### ⏳ Task 1.5: Per-Class MetricsLogger [PENDING]
**Objective**: Track all 10 classes every epoch

**Output**: `per_class_metrics.csv` with columns:
- epoch, phase (train/val)
- {class}_precision, {class}_recall, {class}_f1, {class}_support

---

### ⏳ Task 1.6: Numeric Verification Test [PENDING]
**Objective**: Ensure PyTorch→ONNX→TFLite doesn't lose accuracy

**Tolerance**:
- FP32: max diff < 1e-5
- INT8: max diff < 1e-3

---

## Week 2: Teacher Training

### ⏳ Task 2.1: Train Full Teacher [PENDING]
**Objective**: Best possible 10-class teacher model

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
| DDoS | >98% |
| PortScan | >98% |
| Bot | 60-80% |
| SSH-Patator | 60-75% |

---

### ⏳ Task 2.2: Evaluate on Holdout [PENDING]
**Objective**: Unbiased rare-class evaluation

```bash
python scripts/evaluate_holdout.py \
    --model artifacts/teacher/best_model.pt \
    --holdout-dir data/processed/cic_ids_2017_v2/holdout \
    --output reports/teacher_holdout_eval.json
```

**Report**: Bootstrap 95% CIs given only 40 samples per class

---

## Week 3: Student Architecture Design

### ⏳ Task 3.1: Student Sweep [PENDING]
**Objective**: Find optimal student size

| Size | Params | Expected F1 | Notes |
|------|--------|-------------|-------|
| Tiny | ~50K | ~91% | May miss rare classes |
| Medium | ~100K | ~93% | Good balance |
| Large | ~200K | ~94% | Best accuracy |

```bash
python phase2/train/train_baseline.py \
    --param-targets 50000 100000 200000 \
    --seeds 0 7 42 \
    --epochs 50 \
    --output-dir artifacts/stage0_baseline/
```

**Decision**: Pick size where F1 within 3% of teacher

---

## Week 4: Knowledge Distillation

### ⏳ Task 4.1: KD Training [PENDING]
**Objective**: Transfer teacher knowledge to student

```bash
python phase2/train/train_kd.py \
    --teacher artifacts/teacher/best_model.pt \
    --student-size 200000 \
    --temperature 4.0 \
    --alpha 0.7 \
    --epochs 100 \
    --seeds 0 7 42 101 202 \
    --output-dir artifacts/stage1_kd/
```

**Success Criteria**:
- [ ] KD improves rare-class recall by ≥3-5%
- [ ] F1-Macro ≥93%

---

## Week 5: Pruning & Recovery

### ⏳ Task 5.1: Iterative Pruning [PENDING]
**Objective**: 50% filter reduction with minimal accuracy drop

```bash
python phase2/prune/prune_model.py \
    --model artifacts/stage1_kd/best_model.pt \
    --schedule uniform_50 \
    --iterative \
    --steps 10 \
    --output-dir artifacts/stage2_pruned/
```

**Schedule**: 5% per step × 10 steps = 50% total

---

### ⏳ Task 5.2: KD Fine-tuning [PENDING]
**Objective**: Recover accuracy lost to pruning

```bash
python phase2/prune/fine_tune_kd.py \
    --model artifacts/stage2_pruned/best_model.pt \
    --teacher artifacts/teacher/best_model.pt \
    --epochs 50 \
    --output-dir artifacts/stage3_finetuned/
```

**Target**: F1 drop ≤2% vs pre-pruning

---

## Week 6: Quantization & Deployment

### ⏳ Task 6.1: QAT Training [PENDING]
**Objective**: Train with simulated INT8

```bash
python phase2/quant/qat_train.py \
    --model artifacts/stage3_finetuned/best_model.pt \
    --epochs 30 \
    --output-dir artifacts/stage4_qat/
```

---

### ⏳ Task 6.2: TFLite Conversion [PENDING]
**Objective**: Final deployment model

```bash
python phase2/convert/convert_to_tflite.py \
    --model artifacts/stage4_qat/best_model.pt \
    --quantize int8 \
    --verify-numerics \
    --output-dir artifacts/stage5_tflite/
```

**Verification**: ONNX numeric check must pass

---

### ⏳ Task 6.3: Pi 4 Benchmarking [PENDING]
**Objective**: Verify deployment constraints

```bash
python phase2/bench/pi_bench.py \
    --model artifacts/stage5_tflite/model_int8.tflite \
    --runs 1000
```

**Targets**:
- [ ] p50 latency ≤10ms
- [ ] p95 latency ≤40ms
- [ ] Model size ≤50KB

---

## Week 7: Ablation & Analysis

### ⏳ Task 7.1: Merged-Class Ablation [PENDING]
**Objective**: Show metric inflation from merging

```bash
python scripts/run_merged_ablation.py \
    --merge "Bot,SSH-Patator:RARE_ATTACK"
```

**Compare**: Track A (10-class) vs Track B (9-class merged)

---

### ⏳ Task 7.2: All-Stage Holdout Evaluation [PENDING]
**Objective**: Track rare-class degradation across pipeline

```bash
for stage in teacher stage0 stage1 stage2 stage3 stage4 stage5; do
    python scripts/evaluate_holdout.py \
        --model artifacts/${stage}/best_model.pt \
        --holdout-dir data/processed/cic_ids_2017_v2/holdout \
        --output reports/${stage}_holdout.json
done
```

---

### ⏳ Task 7.3: Generate Figures [PENDING]
**Objective**: Publication-ready visualizations

1. **Per-Class Recall Degradation** — Most compelling figure
2. **Pareto Frontier** — Size vs F1 with threshold markers
3. **Confusion Matrices** — Teacher vs Final model
4. **t-SNE Augmentation Validation** — Supplementary
5. **Latency Distribution** — Pi 4 histogram

---

## Week 8: Paper Writing

### ⏳ Task 8.1: Draft Paper [PENDING]
**Sections**:
1. Introduction (edge IDS motivation)
2. Related Work
3. Methodology (compression pipeline)
4. Experimental Setup
5. Results (per-class, honest reporting)
6. Ablation Studies
7. Discussion (when is compression safe?)
8. **Limitations** (critical — honest assessment)
9. Conclusion

---

### ⏳ Task 8.2: Limitations Section [PENDING]
**Must Include**:
- Rare-class performance (48-70% recall is realistic)
- Holdout size (40 samples limits statistical power)
- Dataset constraints (CIC-IDS-2017 synthetic patterns)
- Compression amplifies baseline weaknesses

---

## 🔧 Scripts to Create

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/create_holdout_set.py` | Extract rare-class holdout | ✅ DONE |
| `phase2/pre_phase2/augment_rare_classes.py` | Hybrid augmentation | 🔄 UPDATING |
| `src/utils/metrics_logger.py` | Per-class epoch logging | ⏳ TODO |
| `scripts/train_teacher_balanced.py` | Teacher with focal loss | ⏳ TODO |
| `scripts/evaluate_holdout.py` | Bootstrap CI evaluation | ⏳ TODO |
| `scripts/verify_conversion.py` | ONNX numeric check | ⏳ TODO |
| `scripts/run_merged_ablation.py` | Track B experiment | ⏳ TODO |
| `scripts/generate_figures.py` | Publication plots | ⏳ TODO |

---

## 📁 Directory Structure (Target)

```
ids-compression/
├── data/processed/cic_ids_2017_v2/
│   ├── train/                    # Training data (660,423 samples)
│   ├── train_original_backup/    # Pre-holdout backup
│   ├── holdout/                  # 80 rare-class samples ✅
│   │   ├── X_rare.npy
│   │   ├── y_rare.npy
│   │   └── attack_types_rare.npy
│   ├── val/
│   └── test/
├── data/processed/cic_ids_2017_v2_augmented/  # After augmentation
├── artifacts/
│   ├── teacher/                  # Teacher models (5 seeds)
│   ├── stage0_baseline/          # Student baselines
│   ├── stage1_kd/                # After KD
│   ├── stage2_pruned/            # After pruning
│   ├── stage3_finetuned/         # After KD-FT
│   ├── stage4_qat/               # After QAT
│   └── stage5_tflite/            # Final TFLite models
├── reports/
│   ├── augmentation_validation_*.png
│   ├── *_holdout_eval.json
│   └── degradation_analysis.json
├── figures/                      # Publication figures
└── experiments/
    └── teacher_diagnostic/       # Quick validation runs
```

---

## 🚨 Critical Reminders

1. **DO NOT merge rare classes** — Report honestly, failures included
2. **Validate augmentation with t-SNE** before heavy training
3. **Check numeric conversion** before trusting TFLite accuracy
4. **Use gradient clipping** (1.0) with high class weights
5. **Cap class weights at 50×** to prevent instability
6. **Run holdout evaluation at EVERY stage** — not just final

---

*Last Updated: December 3, 2025*
*Status: Week 1 - Task 1.2 In Progress*
