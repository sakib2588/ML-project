# Immediate Next Steps - Transfer Learning Pipeline

## 🔄 CURRENTLY RUNNING (Dec 4, 22:55)

**Process**: Pre-training CIC-IDS-2018 models  
**PID**: 15758  
**CPU**: 136%  
**RAM**: 3GB (stable - no more freeze!)  
**Status**: Seed 0, Epoch ~2-3 (with 3.3M samples)  
**Expected completion**: ~3-4 hours from start (22:48)  

**Log file**: `/tmp/train_2018.log`  
**Output**: `experiments/phase2_binary/teacher_2018/seed_{0,7,42}/best_model.keras`

---

## ✅ SOLUTION TO RAM FREEZE

**Problem**: Tried loading 1.4GB files directly → froze system

**Solution Implemented**:
```bash
# Old (BROKEN): pd.read_csv("big.csv")  # Entire file in RAM
# New (WORKING): pd.read_csv("big.csv", chunksize=50000)  # 50K rows at a time
```

- Preprocessing time: 15 minutes (vs system freeze)
- Memory peak: 3GB (vs 20GB+ swap death)
- Result: 3.3M samples successfully processed!

---

## 📋 WHAT TO DO AFTER PRE-TRAINING COMPLETES

### Step 1: Check Results (do this in ~4 hours)
```bash
# Check if training finished
tail -30 /tmp/train_2018.log

# If finished, view results
cat experiments/phase2_binary/teacher_2018/aggregate_results.json | python -m json.tool
```

Expected output format:
```json
{
  "seed_0": {"f1": 0.94, "recall": 0.92, ...},
  "seed_7": {"f1": 0.95, "recall": 0.93, ...},
  "seed_42": {"f1": 0.943, "recall": 0.91, ...}
}
```

### Step 2: Run Transfer Learning
Once best seed identified (say seed_7):

```bash
/home/sakib/ids-compression/.venv_edge/bin/python scripts/06_transfer_2017.py \
  --model-2018 experiments/phase2_binary/teacher_2018/seed_7/best_model.keras \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.0001
```

**What happens**:
- Phase 1 (25 epochs): Freeze early layers, fine-tune output only
- Phase 2 (25 epochs): Unfreeze all, fine-tune with 10x lower LR
- Evaluates on 2017 holdout
- Saves: `experiments/phase2_binary/teacher_transferred/best_model.keras`

**Expected**: 95-97% F1 (vs 92.9% baseline)

### Step 3: Evaluate Transferred Model
```bash
/home/sakib/ids-compression/.venv_edge/bin/python scripts/03_evaluate_holdout.py \
  --model experiments/phase2_binary/teacher_transferred/best_model.keras
```

Output: Per-attack F1 scores with bootstrap CIs

### Step 4: If F1 ≥ 95%: Proceed to Compression
- Knowledge Distillation (teacher → 50K param student)
- 50% Pruning
- INT8 Quantization
- TFLite conversion + Pi4 benchmarking

---

## 📊 WHY THIS WORKS

| Metric | 2017 Only | 2017+2018 Transfer |
|--------|-----------|-------------------|
| Training data | 659K | 3.3M |
| Bot samples | 69 | 286K |
| SSH-Patator | 73 | 188K |
| F1 on holdout | 92.9% | **95-97%** |
| Rare class coverage | Poor | Excellent |

The 2018 dataset has massive amounts of the rare attacks we suck at (Bot, SSH brute force).

---

## 🚨 If Something Goes Wrong

**Training crashes?**
```bash
# Check for errors
tail -100 /tmp/train_2018.log

# Re-run with smaller seeds
/home/sakib/ids-compression/.venv_edge/bin/python scripts/05_train_pretrain_2018.py \
  --epochs 50 \
  --batch-size 128 \
  --seeds 0
```

**Transfer learning OOM?**
```bash
# Use smaller batch size
--batch-size 128

# Or reduce to 30 epochs
--epochs 30
```

---

## 📁 Key Files to Monitor

- **Pre-training log**: `/tmp/train_2018.log`
- **Pre-training results**: `experiments/phase2_binary/teacher_2018/aggregate_results.json`
- **Transfer script**: `scripts/06_transfer_2017.py` (READY)
- **Compression plan**: `scripts/07-09_compression.py` (TO CREATE AFTER TRANSFER)

---

## Timeline

- **Dec 4 22:48**: Pre-training started (3 seeds)
- **Dec 5 ~02:00**: Pre-training should complete
- **Dec 5 02:30**: Transfer learning (50 epochs ~ 30min)
- **Dec 5 03:00**: Evaluation (5 min)
- **Dec 5 03:30**: Compression pipeline (2-3 hours)
- **Dec 5 06:30**: Final model ready for Pi4!

---

**BOTTOM LINE**: You fixed the RAM freeze problem! Training is running stably now. 
Check back in 4 hours for results. ✅
