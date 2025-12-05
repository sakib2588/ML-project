# Phase 2 Binary Classification Plan: Honest IDS Compression

**Project**: Edge-Deployable IDS via Model Compression  
**Approach**: Binary Classification with Per-Attack Breakdown Analysis  
**Target Venue**: IEEE/ACM Security or ML Systems Conference  
**Confidence Level**: 85% if executed exactly as specified

---

## The Novel Contribution

### One-Liner for Paper

> "We systematically analyze how model compression (pruning, quantization, knowledge distillation) affects per-attack detection rates in a binary intrusion detection system, revealing that compression disproportionately impacts rare attack classes—providing practitioners with actionable guidance on compression safety margins."

### Why This is Publishable

1. **Novel angle**: Nobody has done per-attack breakdown analysis of compression effects
2. **Honest science**: We report failures (Bot: 45% → 12%) not just successes
3. **Practical value**: Tells security teams when compression is safe vs dangerous
4. **Reproducible**: CIC-IDS-2017 is public, code will be open-sourced

### What Reviewers Will Like

- Clear problem statement
- Honest methodology
- Actionable findings
- Edge deployment validated on real hardware (Pi4)

---

## Dataset Final Configuration

### After Label Fix

| Class | Train Samples | Holdout | Strategy |
|-------|---------------|---------|----------|
| BENIGN | 530,903 | 3,000 | Stratified sample |
| DoS Hulk | 55,251 | 500 | Stratified sample |
| PortScan | 37,031 | 300 | Stratified sample |
| DDoS | 29,411 | 300 | Stratified sample |
| DoS Slowhttptest | 1,275 | 50 | 5% holdout |
| DoS slowloris | 1,147 | 50 | 5% holdout |
| FTP-Patator | 838 | 30 | 5% holdout |
| DoS GoldenEye | 265 | 10 | Minimal for testing |
| SSH-Patator | 63 | 10 | Minimal for testing |
| Bot | 59 | 10 | Minimal for testing |

**Binary Distribution**: 
- BENIGN: 530,903 (80.4%)
- ATTACK: 129,247 (19.6%)

**Per-attack holdout total**: ~4,260 samples

---

## Architecture

### Teacher Model (Full Precision)
```
Input: (15, 65) - 15 timesteps, 65 features
├── Conv1D(64, k=3) + BatchNorm + ReLU
├── Conv1D(64, k=3) + BatchNorm + ReLU
├── MaxPool1D(2)
├── Conv1D(128, k=3) + BatchNorm + ReLU
├── Conv1D(128, k=3) + BatchNorm + ReLU
├── GlobalAvgPool1D
├── Dense(256) + Dropout(0.3)
├── Dense(128) + Dropout(0.3)
└── Dense(1, sigmoid)

Parameters: ~800K
Size: ~3.2 MB
```

### Student Model (Target for Edge)
```
Input: (15, 65)
├── DepthwiseSeparableConv1D(32, k=3) + ReLU
├── DepthwiseSeparableConv1D(64, k=3) + ReLU
├── GlobalAvgPool1D
├── Dense(64) + Dropout(0.2)
└── Dense(1, sigmoid)

Parameters: ~50K
Size: ~200 KB (FP32)
Target after INT8: ~50 KB
```

---

## Week-by-Week Execution Plan

### Week 0: Foundation (Day 1)
**Time**: 3 hours

1. **Fix labels** (30 min)
   - Regenerate y.npy from attack_types.npy
   - Verify 10-class distribution restored

2. **Create proper holdout** (1 hour)
   - Stratified sampling per attack type
   - Store attack_types for per-attack analysis
   - Never touch holdout during training

3. **Binary label conversion** (30 min)
   - y_binary = (y > 0).astype(int)
   - Keep attack_types for breakdown

4. **Verify GPU + environment** (1 hour)
   - CUDA available
   - TensorFlow 2.x with XLA
   - tflite-runtime on Pi4

### Week 1: Teacher Training (Days 2-4)
**Time**: 12 hours

1. **Train 5 seeds** (8 hours GPU)
   - Seeds: [0, 7, 42, 101, 202]
   - Binary cross-entropy + focal loss fallback
   - Balanced batch sampling
   - Early stopping patience=10

2. **Evaluate on validation** (2 hours)
   - Binary F1, precision, recall
   - Per-attack breakdown with bootstrap CIs
   - t-SNE of learned embeddings

3. **Select best seed** (2 hours)
   - Criteria: highest worst-case per-attack recall
   - Save model + training logs

**Exit Criteria**:
- Binary F1 ≥ 97%
- DDoS recall ≥ 98%
- DoS Hulk recall ≥ 98%
- Bot recall: document baseline (expected 30-60%)

### Week 2: Student Baseline (Days 5-7)
**Time**: 10 hours

1. **Train student without KD** (6 hours)
   - Same 5 seeds
   - Document accuracy gap from teacher

2. **Ablation: model size vs accuracy** (4 hours)
   - Student-XS (25K params)
   - Student-S (50K params)
   - Student-M (100K params)
   - Select best size/accuracy trade-off

**Exit Criteria**:
- Student binary F1 ≥ 93% (before KD)
- Size ≤ 400KB (FP32)

### Week 3: Knowledge Distillation (Days 8-10)
**Time**: 12 hours

1. **Implement KD loss** (2 hours)
   ```python
   loss = α * CE(y_true, y_student) + (1-α) * KL(teacher_logits, student_logits)
   ```

2. **Hyperparameter search** (8 hours)
   - Temperature T: [2, 4, 6, 10]
   - Alpha α: [0.3, 0.5, 0.7, 0.9]
   - Grid search with validation F1

3. **Per-attack analysis** (2 hours)
   - Compare student vs teacher per-attack
   - Document compression degradation

**Exit Criteria**:
- Student binary F1 ≥ 96% (with KD)
- Gap from teacher < 2% overall
- Document per-attack degradation %

### Week 4: Pruning + Quantization (Days 11-14)
**Time**: 15 hours

1. **Structured pruning** (5 hours)
   - 20%, 40%, 60% sparsity
   - Fine-tune after pruning
   - Measure per-attack impact

2. **INT8 quantization** (5 hours)
   - Post-training quantization (PTQ)
   - Quantization-aware training (QAT) if PTQ fails
   - TFLite conversion

3. **Combined pipeline** (5 hours)
   - Best pruning level + INT8
   - Full per-attack evaluation
   - Document compression cascade

**Exit Criteria**:
- Final model < 100KB
- Binary F1 ≥ 94%
- TFLite model runs on Pi4

### Week 5: Pi4 Deployment (Days 15-17)
**Time**: 10 hours

1. **Benchmark latency** (4 hours)
   - Single sample inference
   - Batch inference (8, 16, 32)
   - p50, p95, p99 latencies

2. **Power measurement** (3 hours)
   - Idle vs inference power draw
   - Samples/second/watt metric

3. **Stress testing** (3 hours)
   - Continuous inference for 1 hour
   - Thermal throttling analysis
   - Memory footprint

**Exit Criteria**:
- Latency p95 < 15ms (single sample)
- Throughput > 100 samples/second
- Stable operation 1 hour

### Week 6-7: Paper Writing (Days 18-28)
**Time**: 30 hours

1. **Figures** (8 hours)
   - Per-attack recall heatmap (Teacher vs Student vs Pruned vs Quantized)
   - Compression vs accuracy Pareto front
   - t-SNE embeddings comparison
   - Latency distribution on Pi4

2. **Tables** (4 hours)
   - Main results: binary metrics
   - Per-attack breakdown
   - Ablation studies
   - Comparison with baselines

3. **Writing** (18 hours)
   - Abstract (last)
   - Introduction + related work
   - Methodology
   - Experiments
   - Discussion (honest failures)
   - Conclusion

### Week 8: Submission Prep (Days 29-35)
**Time**: 15 hours

1. **Code cleanup** (5 hours)
   - Documentation
   - Requirements.txt
   - README with reproduction steps

2. **Supplementary materials** (5 hours)
   - Extended results
   - Hyperparameter sensitivity
   - Additional ablations

3. **Revisions + polish** (5 hours)
   - Internal review
   - Final polish
   - Arxiv submission

---

## Success Metrics

### Required (Paper Accepted)
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Binary Teacher F1 | ≥ 97% | - | ⏳ |
| Binary Student (KD) F1 | ≥ 96% | - | ⏳ |
| Binary Final (TFLite) F1 | ≥ 94% | - | ⏳ |
| Model size reduction | > 95% | - | ⏳ |
| Pi4 latency p95 | < 15ms | - | ⏳ |
| DDoS recall (final) | ≥ 95% | - | ⏳ |

### Expected Failures (Report Honestly)
| Metric | Expected | Notes |
|--------|----------|-------|
| Bot recall (teacher) | 30-60% | Only 55 training samples |
| Bot recall (final) | 10-30% | Compression amplifies |
| SSH-Patator (teacher) | 35-65% | Only 58 training samples |
| SSH-Patator (final) | 15-35% | Compression amplifies |

---

## Risk Mitigation

### Risk 1: Teacher doesn't reach 97% binary F1
**Probability**: 10%
**Mitigation**: 
- Check class imbalance handling
- Try focal loss with gamma=2
- Increase model capacity

### Risk 2: Knowledge distillation doesn't help
**Probability**: 20%
**Mitigation**:
- Try attention transfer instead of logit matching
- Use intermediate layer matching
- Fall back to direct student training

### Risk 3: Quantization kills accuracy
**Probability**: 25%
**Mitigation**:
- Use QAT instead of PTQ
- Try INT16 as fallback
- Accept slightly larger model

### Risk 4: Pi4 latency too high
**Probability**: 15%
**Mitigation**:
- Further pruning
- NEON optimization
- Batch processing

---

## File Structure

```
ids-compression/
├── scripts/
│   ├── 00_fix_labels.py
│   ├── 01_create_holdout.py
│   ├── 02_train_teacher.py
│   ├── 03_evaluate_teacher.py
│   ├── 04_train_student.py
│   ├── 05_knowledge_distillation.py
│   ├── 06_prune.py
│   ├── 07_quantize.py
│   ├── 08_convert_tflite.py
│   └── 09_benchmark_pi4.py
├── experiments/
│   └── phase2_binary/
│       ├── logs/
│       ├── models/
│       ├── plots/
│       └── results/
├── paper/
│   ├── figures/
│   ├── tables/
│   └── draft.tex
└── PHASE2_RUNBOOK.md  # Execution log
```

---

## Seed Protocol

All experiments use seeds: **[0, 7, 42, 101, 202]**

Report: mean ± std across seeds

---

## Let's Start

**Immediate next step**: Run `scripts/00_fix_labels.py` to restore 10-class labels

Then: Choose your path and confirm before proceeding.

---

*"Honest science beats impressive-looking nonsense. Always."*
