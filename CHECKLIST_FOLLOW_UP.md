# Follow-Up Checklist - What to Do After Pre-training

## ✅ Session Completed (Dec 4 22:55)

Your transfer learning setup is now running stably. Pre-training will take ~3-4 hours.

---

## STEP 1: Monitor Pre-training (Every 30 minutes)

### Command to check status:
```bash
tail -30 /tmp/train_2018.log | tail -15
```

### What to look for:
- ✅ Epoch number increasing (currently Epoch 3)
- ✅ Loss decreasing steadily
- ✅ Validation AUC increasing
- ✅ No error messages
- ✅ Memory stable around 3GB

### If something looks wrong:
```bash
# View full error
tail -100 /tmp/train_2018.log

# Kill the process
pkill -f train_pretrain

# Re-run with smaller batch size
python scripts/05_train_pretrain_2018.py --batch-size 128 --seeds 0
```

---

## STEP 2: When Complete (~2-3 AM Dec 5)

### 1. Check results file exists:
```bash
ls -lh experiments/phase2_binary/teacher_2018/aggregate_results.json
```

### 2. View the results:
```bash
cat experiments/phase2_binary/teacher_2018/aggregate_results.json | python -m json.tool
```

### 3. Identify best seed (highest F1):
Example output:
```json
{
  "seed_0": {"f1": 0.943, "recall": 0.91, ...},
  "seed_7": {"f1": 0.947, "recall": 0.92, ...},  ← BEST
  "seed_42": {"f1": 0.938, "recall": 0.89, ...}
}
```

### 4. Note the best seed and F1 score
- Best seed: `_____`
- F1 score: `_____`

---

## STEP 3: Run Transfer Learning

Once you have the best seed, run:

```bash
/home/sakib/ids-compression/.venv_edge/bin/python scripts/06_transfer_2017.py \
  --model-2018 experiments/phase2_binary/teacher_2018/seed_7/best_model.keras \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.0001
```

**Replace `seed_7` with your actual best seed!**

What happens:
- Phase 1 (25 epochs): Fine-tune output layers only
- Phase 2 (25 epochs): Fine-tune all layers with 10x lower LR
- Evaluates on 2017 holdout
- Saves: `experiments/phase2_binary/teacher_transferred/best_model.keras`

Expected time: 30-40 minutes

Monitor with:
```bash
tail -f transfer_2017.log  # (if redirected to log file)
```

---

## STEP 4: Evaluate Transferred Model

```bash
/home/sakib/ids-compression/.venv_edge/bin/python scripts/03_evaluate_holdout.py \
  --model experiments/phase2_binary/teacher_transferred/best_model.keras
```

This will show:
- Binary metrics (Precision, Recall, F1, AUC)
- Per-attack breakdown (Bot, SSH-Patator, DoS, etc.)
- Bootstrap confidence intervals

### Success criteria:
- [ ] F1 ≥ 95% (vs 92.9% baseline)
- [ ] Bot recall ≥ 50% (vs 0% baseline)
- [ ] SSH-Patator recall ≥ 40%
- [ ] Common attacks recall ≥ 80%

---

## STEP 5: If Transfer Successful (F1 ≥ 95%)

Proceed to compression:

```bash
# Create compression script (to be done)
python scripts/07_knowledge_distillation.py \
  --teacher experiments/phase2_binary/teacher_transferred/best_model.keras \
  --student-params 50000 \
  --epochs 100

# Then pruning
python scripts/08_pruning_and_qat.py \
  --model experiments/phase2_binary/student/best_model.keras \
  --sparsity 0.5

# Then TFLite conversion
python scripts/09_tflite_export.py \
  --model experiments/phase2_binary/compressed/best_model.keras
```

---

## STEP 6: If Transfer Not Successful (F1 < 95%)

Options (in order of preference):

### Option A: Ensemble (Quick, +1-2% F1)
```bash
# Combine multiple 2018 seeds
python scripts/05_train_pretrain_2018.py --seeds 0 7 42 101 202
# Create voting ensemble
# Transfer ensemble to 2017
```

### Option B: Different architecture (Medium, +1-3% F1)
- Try CNN instead of MLP
- Try more layers (8-10 layers)
- Try wider layers (512, 256 units)

### Option C: More epochs (Slow, +0.5-1% F1)
- Increase epochs to 200
- Reduce learning rate more aggressively
- Enable gradient accumulation

---

## Important Files to Keep

```
data/
├── raw/cic_ids_2018/          ← Downloaded CSVs (1.4GB)
└── processed/cic_ids_2018/    ← Processed data (1.1GB) **KEEP**

experiments/phase2_binary/
├── teacher_2018/              ← Pre-trained models **KEEP**
├── teacher_transferred/       ← Transfer learning results **KEEP**
├── student/                   ← Knowledge distillation **KEEP**
└── compressed/                ← Final compressed model **KEEP**

scripts/
├── 04_preprocess_2018_streaming.py
├── 05_train_pretrain_2018.py
└── 06_transfer_2017.py
```

---

## Estimated Timeline

| Phase | Start | Duration | Status |
|-------|-------|----------|--------|
| Pre-training 2018 | Dec 4 22:48 | 3-4h | 🔄 RUNNING |
| Transfer to 2017 | Dec 5 02:00 | 30-40m | ⏳ PENDING |
| Evaluation | Dec 5 02:45 | 5m | ⏳ PENDING |
| Compression (if F1≥95%) | Dec 5 02:50 | 2-3h | ⏳ PENDING |
| **FINAL** | Dec 5 05:50 | - | ⏳ PENDING |

---

## Success Metrics

### For Session Success ✅
- [x] Pre-training runs without RAM issues
- [x] Epoch throughput: 170s/epoch (good)
- [x] Validation loss decreasing (good)
- [ ] Pre-training F1 ≥ 94%
- [ ] Transfer F1 ≥ 95%
- [ ] Compression F1 ≥ 93%

### For Production ✅
- [ ] TFLite model runs on Pi4 in <50ms
- [ ] Model size ≤ 500KB
- [ ] Per-attack coverage: Bot≥50%, SSH-Patator≥40%
- [ ] No crashes or OOM on Pi4

---

## Questions & Debugging

**Q: Pre-training slow?**
A: Expected! 3.3M samples × batch size 256 = 13,058 batches/epoch
Expected time: 170-200 sec/epoch

**Q: High validation loss?**
A: Focal loss can seem high. Look at AUC instead (should be >0.95)

**Q: Transfer learning OOM?**
A: Reduce batch size to 128 or epochs to 30

**Q: Can't find best_model.keras?**
A: Check: `ls experiments/phase2_binary/teacher_2018/seed_*/`

---

**Ready to go!** Check back when pre-training completes. ✅
