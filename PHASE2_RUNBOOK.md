# Phase 2 Binary Execution Runbook

**Project**: IDS Compression for Edge Deployment  
**Approach**: Binary Classification (Attack vs Benign) with Per-Attack Breakdown  
**Started**: 2025-12-04  
**Status**: 🟡 WEEK 0 - FOUNDATION IN PROGRESS

---

## Quick Status Summary

| Week | Task | Status | Start | End | Duration |
|------|------|--------|-------|-----|----------|
| 0 | Foundation (label fix, holdout) | 🟡 In Progress | 2025-12-04 | TBD | ~1 hour |
| 1 | Teacher training (5 seeds) | ⏳ Pending | 2025-12-04 | 2025-12-07 | 8 hours |
| 2 | Student baseline | ⏳ Pending | 2025-12-08 | 2025-12-10 | 6 hours |
| 3 | Knowledge distillation | ⏳ Pending | 2025-12-11 | 2025-12-13 | 8 hours |
| 4 | Pruning + quantization | ⏳ Pending | 2025-12-14 | 2025-12-17 | 10 hours |
| 5 | Pi4 deployment | ⏳ Pending | 2025-12-18 | 2025-12-20 | 6 hours |
| 6-7 | Paper writing | ⏳ Pending | 2025-12-21 | 2025-12-31 | 20 hours |

---

## Week 0: Foundation (2025-12-04)

### [✅] Task 0.1: Fix Labels & Errors
**Status**: ✅ COMPLETE  
**Time**: 2025-12-04 14:39 UTC  
**What was done**:
- ✅ Audited actual dataset: 10 classes, BENIGN 80.83%, ATTACK 19.17%
- ✅ Created REALITY_CHECK.md (no-sugarcoat analysis)
- ✅ Created PHASE2_BINARY_PLAN.md (publication strategy)
- ✅ Created 4 execution scripts with full documentation
- ✅ Fixed all 21 type-checking errors in code
- ✅ Ran 00_fix_labels.py - labels successfully restored

**Files created/modified**:
- `REALITY_CHECK.md` - Honest dataset analysis
- `PHASE2_BINARY_PLAN.md` - Week-by-week technical plan
- `scripts/00_fix_labels.py` - Label restoration ✅ EXECUTED
- `scripts/01_create_holdout.py` - Stratified holdout
- `scripts/02_train_teacher.py` - Multi-seed training (fixed TF type errors)
- `scripts/03_evaluate_holdout.py` - Bootstrap evaluation

**Output from label fix**:
```
✅ train: 660,503 → restored to 10-class
   BENIGN: 533,903 (80.83%)
   DoS Hulk: 55,751 (8.44%)
   PortScan: 37,331 (5.65%)
   DDoS: 29,711 (4.50%)
   [6 more small classes]
   
✅ val: 141,537 samples restored
✅ test: 141,537 samples restored
✅ y_binary.npy created for all splits
```

**Next**: Create holdout set

---

### [✅] Task 0.2: Execute Label Fix
**Status**: ✅ COMPLETE  
**Command**: `python scripts/00_fix_labels.py`
**Execution**: 2025-12-04 14:39 UTC
**Result**: 
- ✅ y.npy restored to 10-class (train, val, test)
- ✅ y_binary.npy created for all splits
- ✅ Corrupted binary labels backed up
- ✅ class_mapping.json updated
**Next**: Execute holdout creation

---

### [ ] Task 0.3: Execute Holdout Creation
**Command**:
```bash
/home/sakib/ids-compression/.venv_edge/bin/python scripts/01_create_holdout.py --seed 42
```
**Expected**: 
- `holdout_final/` with ~4,260 stratified samples (10 classes)
- `train_final/` with remaining samples for training
- Metadata files for verification
**Status**: ⏳ Pending
- [ ] attack_types preserved in all splits

---

## Week 1: Teacher Training

### [ ] Task 1.1: Train Teacher (5 seeds)
**Command**:
```bash
python scripts/02_train_teacher.py \
    --seeds 0 7 42 101 202 \
    --epochs 100 \
    --batch-size 256 \
    --patience 10
```
**Expected**:
- Binary F1 ≥ 97% on validation
- ~8 hours GPU time
**Status**: ⏳ Pending

---

### [ ] Task 1.2: Evaluate on Holdout
**Command**:
```bash
python scripts/03_evaluate_holdout.py \
    --model experiments/phase2_binary/teacher/seed_42/best_model.keras \
    --bootstrap 1000
```
**Expected**:
- Per-attack breakdown with 95% CIs
- LaTeX table generated
**Status**: ⏳ Pending

---

## Week 2: Student Training

### [ ] Task 2.1: Train Student (no KD)
**Command**: TBD
**Expected**: Binary F1 ≥ 93%
**Status**: ⏳ Pending

---

## Week 3: Knowledge Distillation

### [ ] Task 3.1: KD Training
**Command**: TBD
**Expected**: Binary F1 ≥ 96%
**Status**: ⏳ Pending

---

## Week 4: Compression

### [ ] Task 4.1: Pruning
### [ ] Task 4.2: Quantization
### [ ] Task 4.3: TFLite Conversion

---

## Week 5: Deployment

### [ ] Task 5.1: Pi4 Benchmarking

---

## Metrics Tracking

| Stage | Binary F1 | DDoS Recall | Bot Recall | Model Size |
|-------|-----------|-------------|------------|------------|
| Teacher | - | - | - | ~3.2MB |
| Student (no KD) | - | - | - | ~200KB |
| Student (KD) | - | - | - | ~200KB |
| Pruned | - | - | - | - |
| Quantized (INT8) | - | - | - | - |
| Final TFLite | - | - | - | - |

---

## Issues & Blockers

| Date | Issue | Resolution | Status |
|------|-------|------------|--------|
| - | - | - | - |

---

## Notes

- **Seeds used**: [0, 7, 42, 101, 202]
- **Focal loss**: γ=2.0, α=0.25
- **Bootstrap CI**: 1000 samples, 95% confidence

---

*Last updated: [AUTO]*
