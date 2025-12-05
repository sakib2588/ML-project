# Phase 2: Quick Reference Guide (1-Page Cheat Sheet)

## 🎯 THE GOAL IN ONE SENTENCE
**Compress a 1.9M parameter teacher down to 48KB INT8 model while maintaining >50% recall on rare attacks (Bot, SSH-Patator)**

---

## 📊 THE PIPELINE (8 Stages)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        COMPRESSION JOURNEY                                    │
└──────────────────────────────────────────────────────────────────────────────┘

Teacher (1.9M params, 7.6MB)
    ↓ Train from scratch with balanced sampling + focal loss + class weights
Stage 0: Student Baseline (200K params, no teacher)
    ↓ Proves student can't match teacher alone
Stage 1: Knowledge Distillation (200K params, guided by teacher)
    ↓ Student learns soft labels from teacher
Stage 2: Pruning (100K params, 50% removal)
    ↓ Iteratively remove 5% × 10 steps
Stage 3: KD Fine-Tuning (100K params, recovered)
    ↓ Re-learn with teacher guidance on pruned model
Stage 4: Quantization-Aware Training (100K params, INT8 simulation)
    ↓ Learn robust weights for 8-bit arithmetic
Stage 5: TFLite Conversion (48KB, INT8)
    ↓ Deploy to Raspberry Pi
Final Model (48KB, <10ms latency)
```

---

## 📈 EXPECTED RESULTS AT EACH STAGE

| Stage | Params | Size | F1 | DDoS | Bot | Comment |
|-------|--------|------|----|----|-----|---------|
| Teacher | 1.9M | 7.6MB | 94.5% | 98.5% | 70% | 🎓 Reference |
| Baseline | 200K | 800KB | 93% | 97.8% | 50% | Weak |
| KD | 200K | 800KB | 94.5% | 98.2% | 65% | +15% rare! ⭐ |
| Pruned | 100K | 400KB | 91.5% | 96.5% | 45% | Ouch |
| KD-FT | 100K | 400KB | 93% | 97.8% | 55% | Recovered |
| QAT | 100K | 400KB | 92.8% | 97.5% | 52% | Stable |
| TFLite | 100K | 48KB | 92.5% | 97.2% | 48% | ✅ Deploy ready |

---

## ⏱️ THE 8-WEEK TIMELINE

### Week 1: Prepare Data (10 hrs)
- ✅ Create 80-sample rare-class holdout
- ✅ Augment Bot (29→87), SSH-Patator (33→99)
- ✅ Validate with t-SNE
- ✅ Quick teacher test (30 epochs)

### Week 2: Train Teacher (20 hrs)
- ✅ Teacher training (5 seeds, 100 epochs)
- ✅ Evaluate on rare-class holdout
- ⏳ Benchmark: Teacher gets 64% rare-class recall

### Week 3: Design Student (15 hrs)
- ✅ Student sweep (50K, 100K, 200K)
- ✅ Pick size where F1 within 3% of teacher

### Week 4: Knowledge Distillation (20 hrs)
- ✅ Train KD students (5 seeds)
- ⏳ Expect: +15% improvement on rare classes
- ✅ Validate KD worth the effort

### Week 5: Prune & Recover (25 hrs)
- ✅ Iterative pruning (50% removal)
- ✅ KD fine-tuning (recover accuracy)
- ⏳ Target: F1 drop ≤2%

### Week 6: Quantize & Deploy (20 hrs)
- ✅ QAT training (INT8 simulation)
- ✅ TFLite conversion (48KB)
- ✅ Pi 4 benchmarking (<10ms latency)

### Week 7: Evaluate & Ablate (15 hrs)
- ✅ All-stage holdout evaluation
- ✅ Merged-class ablation (9-class version)
- ✅ Generate 5 publication figures

### Week 8: Write Paper (20 hrs)
- ✅ Full paper draft
- ✅ **Honest limitations section** ⚠️
- ✅ Ready for submission

---

## 🔑 KEY CONCEPTS (Understand These!)

### Why Holdout Set?
**Problem**: You only have 69 Bot and 73 SSH-Patator samples total. If you train on all of them, you can't honestly evaluate.  
**Solution**: Reserve 40 of each for final testing only. Train on the remaining 29 Bot, 33 SSH-Patator.  
**Result**: Honest rare-class evaluation on never-before-seen samples.

### Why Augmentation?
**Problem**: 29 Bot samples is too few to learn patterns.  
**Solution**: Jitter + Mixup + Capped SMOTE → 87 Bot samples (realistic synthetic data).  
**Result**: More training data without degrading generalization.

### Why Knowledge Distillation?
**Problem**: Student baseline only gets 50% rare-class recall.  
**Solution**: Train student to mimic teacher's soft labels (probabilities) not just hard labels (classes).  
**Result**: Student learns smoother decision boundaries → 65% rare-class recall (+15%!).

### Why Pruning?
**Problem**: 200K parameter model is too big for Pi.  
**Solution**: Remove 50% of filters that contribute least to accuracy.  
**Result**: 200K→100K params, model still works.

### Why QAT?
**Problem**: Naive INT8 conversion loses 5-10% accuracy.  
**Solution**: Train with simulated INT8 arithmetic so weights learn to be robust.  
**Result**: INT8 conversion loses only 0.5-1% accuracy.

### Why Holdout Evaluation at Every Stage?
**Problem**: Macro-F1 hides rare-class failures. Bot recall could drop 30% and you wouldn't notice if macro-F1 drops 2%.  
**Solution**: Evaluate rare classes separately at every compression stage.  
**Result**: Honest documentation of rare-class degradation.

---

## ⚠️ THE HONEST TRUTH

| Metric | Teacher | Final Model | Degradation | Why |
|--------|---------|-------------|------------|-----|
| **Macro-F1** | 94.5% | 92.5% | -2% | Acceptable |
| **DDoS Recall** | 98.5% | 97.2% | -1.3% | Minimal |
| **PortScan Recall** | 99% | 97.5% | -1.5% | Minimal |
| **Bot Recall** | 70% | 48% | -22% | ⚠️ Real degradation |
| **SSH Recall** | 65% | 48% | -17% | ⚠️ Real degradation |

**The Hard Truth**: Compression hurts rare-class detection. Rare classes go from ~65% to ~48% recall. This is your most important finding — **compression is not safe for rare attacks.**

**What to Say in Your Paper**:
> "While the final model achieves 92.5% macro-F1 (only 2% drop from teacher), rare-class recall degrades by 15-20 percentage points. Organizations must decide if 48% Bot detection is acceptable, or if cloud fallback is needed for rare-class detection."

---

## 📋 WEEK-BY-WEEK COMMANDS

### WEEK 1
```bash
# Day 1: Create holdout + augment
python scripts/create_holdout_set.py --holdout-samples 40
python scripts/augment_hybrid_ultra_lite.py --augment-factor 3

# Day 2: Validate + Quick test (4-6 hours GPU)
python scripts/train_teacher_balanced.py --epochs 30 --seeds 0 7 42
```

### WEEK 2
```bash
# Full teacher training (15+ hours GPU)
python phase2/train/train_teacher.py \
    --epochs 100 --seeds 0 7 42 101 202 \
    --output-dir artifacts/teacher/

# Evaluate on holdout
python scripts/evaluate_holdout.py --model artifacts/teacher/seed42/best_model.pt
```

### WEEK 3
```bash
# Student baseline sweep (10 hours GPU)
python phase2/train/train_baseline.py \
    --param-targets 50000 100000 200000 \
    --seeds 0 7 42
```

### WEEK 4
```bash
# KD training (15 hours GPU)
python phase2/train/train_kd.py \
    --teacher artifacts/teacher/seed42/best_model.pt \
    --param-targets 50000 100000 200000 \
    --seeds 0 7 42
```

### WEEK 5
```bash
# Pruning (10 hours GPU)
python phase2/prune/prune_model.py \
    --model artifacts/stage1_kd/200k/seed42/best_model.pt \
    --prune-ratio 0.5 --iterative --steps 10

# KD Fine-tuning (5 hours GPU)
python phase2/prune/finetune_kd.py \
    --model artifacts/stage2_pruned/best_model.pt \
    --teacher artifacts/teacher/seed42/best_model.pt \
    --epochs 50
```

### WEEK 6
```bash
# QAT (5 hours GPU)
python phase2/quant/qat_train.py \
    --model artifacts/stage3_finetuned/best_model.pt --epochs 30

# Convert to TFLite
python phase2/convert/convert_to_tflite.py \
    --model artifacts/stage4_qat/best_model.pt \
    --quantize int8

# Pi benchmarking (run on Pi, 30 min)
python phase2/bench/pi_bench.py \
    --model model_int8.tflite --runs 1000
```

### WEEK 7
```bash
# Ablation: Merged classes
python scripts/run_merged_ablation.py \
    --merge "Bot,SSH-Patator:RARE_ATTACK"

# All-stage holdout evaluation
for stage in teacher stage1_kd stage2_pruned stage3_finetuned stage4_qat stage5_tflite; do
    python scripts/evaluate_holdout.py --model artifacts/${stage}/best_model.pt
done

# Generate figures
python scripts/generate_figures.py --results-dir artifacts/
```

### WEEK 8
```bash
# Write paper (offline, no code)
# Use PUBLICATION_ROADMAP.md as template
```

---

## 🎓 SUCCESS CHECKLIST

- [ ] **Rare-class holdout created** (80 samples, never trained)
- [ ] **Augmentation validated** (t-SNE shows synthetic overlaps real)
- [ ] **Teacher trains successfully** (rare-class recall ≥60%)
- [ ] **KD improves rare-class** (bot recall baseline 50% → KD 65%)
- [ ] **Pruning + KD-FT stable** (F1 drop ≤2%)
- [ ] **QAT doesn't hurt** (accuracy drop ≤1%)
- [ ] **TFLite works** (size <50KB, latency <10ms)
- [ ] **All-stage holdout evaluation complete** (degradation curve shown)
- [ ] **Ablation proves 10-class > 9-class** (honest reporting > metric inflation)
- [ ] **Paper includes honest limitations** (rare-class failures acknowledged)
- [ ] **5 publication figures ready**

---

## 🚀 START NOW

**Next 2 hours:**
```bash
# Create holdout
python scripts/create_holdout_set.py --holdout-samples 40

# Augment
python scripts/augment_hybrid_ultra_lite.py --augment-factor 3

# Validate t-SNE
python scripts/validate_augmentation.py

# Check all files created
ls data/processed/cic_ids_2017_v2/holdout/
ls data/processed/cic_ids_2017_v2_augmented/
```

**Then:**
```bash
# Week 1 task: Quick teacher (4-6 hours, GPU)
python scripts/train_teacher_balanced.py --epochs 30 --seeds 0 7 42
```

---

## 📞 KEY FILES

- `PHASE2_TODO.md` — Detailed checklist (use this for daily tracking)
- `PUBLICATION_ROADMAP.md` — Technical deep-dive (use for implementation)
- `PHASE2_SUMMARY.md` — This document (quick reference)
- `scripts/augment_hybrid_ultra_lite.py` — Fast augmentation script
- `src/utils/metrics_logger.py` — Per-class metrics tracking

---

*Philosophy: Scientific integrity over pretty numbers*  
*Timeline: 8 weeks, ~145 hours total work*  
*Deadline: January 31, 2026 for publication*

**You've got this. Now go build the best IDS edge deployment paper ever.** 🚀
