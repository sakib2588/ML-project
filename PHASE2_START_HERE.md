# Phase 2: START HERE 🚀

**Read this first. It tells you everything you need to know in 5 minutes.**

---

## 🎯 Your Mission (8 Weeks)

Compress your 1.9M parameter teacher model down to 48KB for Raspberry Pi deployment while being **brutally honest** about rare-attack detection performance.

```
Teacher (1.9M params) → KD → Prune → QAT → TFLite (48KB)
```

---

## 📊 Expected Results

| Metric | Teacher | Final Model | Status |
|--------|---------|-------------|--------|
| **Model Size** | 7.6MB | **48KB** | ✅ 99.4% reduction |
| **Latency (Pi4)** | 15ms | **7ms** | ✅ <10ms target |
| **Common Attacks (DDoS/PortScan)** | 98.5% | **97%** | ✅ Acceptable loss |
| **Rare Attacks (Bot/SSH)** | 65-68% | **48-50%** | ⚠️ **Expected degradation** |
| **Macro-F1** | 94.5% | **92.5%** | ✅ 2% drop acceptable |

**The Honest Truth**: Compression hurts rare-attack detection. Go from 65% → 50% recall. This is your most important finding.

---

## 📚 Documentation (Read in This Order)

1. **This file** (5 min) — Overview
2. `PHASE2_QUICKREF.md` (10 min) — One-page cheat sheet
3. `PHASE2_SUMMARY.md` (30 min) — Detailed what/why per task
4. `PHASE2_WEEKLY_VISUAL.md` (20 min) — Visual week-by-week breakdown
5. `PHASE2_TODO.md` (60 min) — Technical deep-dive, use for daily tracking
6. `PUBLICATION_ROADMAP.md` (60 min) — Research philosophy and standards

---

## ⏱️ The 8-Week Timeline

### Week 1: Prepare Data (10 hours, 6 GPU)
- ✅ Create 80-sample rare-class holdout (never train on this)
- ✅ Augment Bot (29→87), SSH-Patator (33→99)
- ✅ Validate with t-SNE (synthetic must overlap real)
- ✅ Quick teacher test (30 epochs to validate augmentation helps)

**Go/No-Go**: Rare-class recall improved by ≥10%?

### Week 2: Train Teacher (20 hours, 15 GPU)
- ✅ Full teacher training (100 epochs, 5 seeds)
- ✅ Evaluate on holdout (honest rare-class baseline)

**Go/No-Go**: Teacher gets ≥60% rare-class recall?

### Week 3: Design Student (15 hours, 10 GPU)
- ✅ Train student baselines (50K, 100K, 200K params)
- ✅ Pick size where F1 within 3% of teacher

**Decision**: Use 200K for KD stage

### Week 4: Knowledge Distillation (20 hours, 15 GPU)
- ✅ Train KD students (same sizes, guided by teacher)
- ✅ **Expect +15% rare-class improvement** ⭐

**Go/No-Go**: KD improves rare-class by ≥10%?

### Week 5: Pruning & Recovery (25 hours, 18 GPU)
- ✅ Iterative pruning (50% removal, 10 steps)
- ✅ KD fine-tuning (recover lost accuracy)

**Go/No-Go**: F1 drop ≤2%?

### Week 6: Quantize & Deploy (20 hours, 12 GPU)
- ✅ QAT training (INT8 simulation)
- ✅ TFLite conversion (48KB)
- ✅ Pi 4 benchmarking (<10ms latency)

**Go/No-Go**: Latency <10ms, size <50KB?

### Week 7: Evaluate & Analyze (15 hours, 5 GPU)
- ✅ All-stage holdout evaluation (shows degradation curve)
- ✅ Merged-class ablation (prove 10-class > 9-class)
- ✅ Generate 5 publication figures

**Go/No-Go**: All figures publication-ready?

### Week 8: Write Paper (20 hours, 0 GPU)
- ✅ Full paper draft
- ✅ **Honest limitations section** ⚠️
- ✅ Ready for submission

---

## 🚨 Critical Reminders

1. **DO NOT MERGE RARE CLASSES** in main results. Show merged as ablation only.
2. **EVALUATE ON HOLDOUT AT EVERY STAGE** to track degradation honestly.
3. **USE BOOTSTRAP CIs** for rare-class metrics (40 samples = low power).
4. **REPORT FAILURES** — compression fails for rare attacks. Say it clearly.
5. **CLASS WEIGHTS CAPPED AT 50×** — prevents training instability.
6. **GRADIENT CLIPPING 1.0** — essential with high class weights.

---

## 📋 Success Criteria (Hard Requirements)

```
✅ F1-Macro:              ≥91% (2% drop from teacher acceptable)
✅ DDoS Recall:           ≥98% (CRITICAL, cannot miss)
✅ PortScan Recall:       ≥98% (CRITICAL, cannot miss)
✅ False Alarm Rate:      ≤1.5% (operations constraint)
✅ Model Size:            ≤50KB (TFLite INT8)
✅ Latency p50:           ≤10ms (Pi 4, batch=1)
✅ Rare-Class Recall:     ≥50% (documentation honest)
✅ Reproducibility:       5 seeds, 95% CIs reported
✅ All 10 classes:        Evaluated in main results
✅ Limitations included:  Honest assessment of failures
```

---

## 🎓 Key Concepts

### Why Holdout Set?
Reserve 40 rare-class samples for **final evaluation only**. This is your only honest way to measure rare-class performance on unseen data.

### Why Augmentation?
Bot (29) + SSH-Patator (33) are too small to learn from. Jitter + Mixup + Capped SMOTE → realistic synthetic data that helps training without memorization.

### Why Knowledge Distillation?
Student baseline gets 50% rare-class. With KD guidance, student gets 65% (+15%!). This transfer learning is essential for rare classes.

### Why Pruning?
Compress 200K → 100K params for deployment. Do it iteratively (10 steps of 5%) to minimize accuracy loss.

### Why QAT?
Naive INT8 conversion loses 5-10% accuracy. QAT training learns INT8-robust weights. Loss drops to 0.5-1%.

### Why Honest Reporting?
Papers with metric inflation get caught by reviewers. Honest reporting of failures is actually a **strength** — shows you understand the problem space.

---

## 🔑 Quick Commands

### Week 1
```bash
# Create holdout
python scripts/create_holdout_set.py --holdout-samples 40

# Augment
python scripts/augment_hybrid_ultra_lite.py --augment-factor 3

# Quick teacher test (4-6 hours GPU)
python scripts/train_teacher_balanced.py --epochs 30 --seeds 0 7 42
```

### Weeks 2-6
See `PHASE2_WEEKLY_VISUAL.md` for detailed commands per week.

### Week 7
```bash
# All-stage holdout evaluation
for stage in teacher stage1_kd stage2_pruned stage3_finetuned stage4_qat stage5_tflite; do
    python scripts/evaluate_holdout.py --model artifacts/${stage}/best_model.pt
done

# Generate figures
python scripts/generate_figures.py --results-dir artifacts/
```

---

## 📊 Key Metrics to Track

### Rare-Class Metrics (Track These Obsessively)
- Bot recall at every stage (Teacher→Final)
- SSH-Patator recall at every stage
- Bootstrap 95% CI for both (acknowledge uncertainty)

### Common-Class Metrics (Must Not Drop Below)
- DDoS recall ≥98%
- PortScan recall ≥98%
- BENIGN specificity >99% (FP rate <1%)

### Efficiency Metrics (Prove Compression Works)
- Model size reduction (7.6MB → 48KB = 99%)
- Latency improvement (15ms → 7ms)
- Throughput increase (on Pi 4)

---

## 🎓 Paper Structure (Week 8)

1. **Introduction** (2 pg) — Edge IDS challenge + research question
2. **Related Work** (1.5 pg) — Compression + class imbalance
3. **Methodology** (2 pg) — Dataset, architecture, pipeline, evaluation
4. **Setup** (0.75 pg) — Hardware, metrics, seeds
5. **Results** (2.5 pg) — ⭐ Tables + figures + analysis
6. **Ablation** (1 pg) — Merged vs honest 10-class
7. **Discussion** (1.5 pg) — When is compression safe?
8. **Limitations** (1 pg) — ⚠️ **CRITICAL** Rare-class failures acknowledged
9. **Conclusion** (0.5 pg) — Summary + practitioner guidance

---

## ⚠️ Honest Limitations Section (Draft for Week 8)

```
Our models achieve only 48-50% recall on Bot and SSH-Patator after 
compression. This reflects two realities:

(1) DATA LIMITATION: Only 69-73 original samples per class makes these 
    inherently hard to learn. This is a data problem, not a model problem.

(2) COMPRESSION AMPLIFICATION: Quantization removes fine-grained decision 
    boundaries. Rare classes are most affected.

PRACTITIONERS SHOULD NOT:
- Deploy this compressed model as sole detector for rare attacks
- Trust 48% recall on Bot/SSH detection
- Ignore active labeling to collect more rare-attack data

PRACTITIONERS SHOULD:
- Implement hierarchical detection (compressed for common, cloud for rare)
- Evaluate on YOUR data before deployment
- Collect more rare-attack samples if available
```

---

## 📈 Expected Week-by-Week Progress

```
Week 1: ████░░░░░░ (Validate augmentation)
Week 2: ████████░░ (Train teacher)
Week 3: ██████████ (Design student)
Week 4: ████████░░ (KD improves rare-class +15%)
Week 5: ████████░░ (Prune + recover)
Week 6: ████████░░ (Deploy to Pi)
Week 7: ████████░░ (Analysis + figures)
Week 8: ██████████ (Paper ready)
```

---

## 🎯 Success Looks Like

✅ **Week 1 Success**: t-SNE shows synthetic overlapping real. Teacher diagnostic gets ≥60% rare-class recall.

✅ **Week 2 Success**: Full teacher trained, holdout evaluation shows 65-68% rare-class recall.

✅ **Week 4 Success**: KD students beat baselines. +15% improvement on rare classes.

✅ **Week 6 Success**: TFLite runs on Pi in <10ms. Size 48KB.

✅ **Week 7 Success**: Degradation curve plotted. 10-class > 9-class proven.

✅ **Week 8 Success**: Paper submission-ready. Honest limitations included.

---

## 📞 Key Files

| File | Purpose | When |
|------|---------|------|
| `PHASE2_QUICKREF.md` | 1-page cheat sheet | Daily reference |
| `PHASE2_TODO.md` | Full checklist | Weekly planning |
| `PHASE2_SUMMARY.md` | Detailed what/why | Task implementation |
| `PHASE2_WEEKLY_VISUAL.md` | Visual breakdown | Understanding pipeline |
| `PUBLICATION_ROADMAP.md` | Deep technical dive | Detailed reference |

---

## 🚀 START NOW

**Next 2 hours:**
```bash
# Create holdout
python scripts/create_holdout_set.py --holdout-samples 40

# Augment
python scripts/augment_hybrid_ultra_lite.py --augment-factor 3

# Validate
python scripts/validate_augmentation.py
```

**Then (4-6 hours, GPU):**
```bash
# Week 1 task: Quick teacher
python scripts/train_teacher_balanced.py --epochs 30 --seeds 0 7 42
```

**After that:**
Read `PHASE2_SUMMARY.md` for detailed Week 2 tasks.

---

## 🎓 Final Thought

You're not just building a compression pipeline. You're proving that **scientific integrity > metric inflation**. 

Showing that compression fails for rare attacks is actually your **strongest research contribution**. It tells practitioners when compression is safe (DDoS/PortScan ✅) and when it's dangerous (Bot/SSH ❌).

**This is the paper reviewers want to see.**

Now go build it. 8 weeks. 145 hours. One incredible research project.

Let's go. 🚀
