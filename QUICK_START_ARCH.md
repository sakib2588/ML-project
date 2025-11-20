# Quick Reference: Running Phase 1 on Arch Linux

## Your System
- **RAM**: 11.8 GB (plenty for this project ✅)
- **CPU**: Intel i3-7100U (4 threads)
- **Location**: `/workspaces/ids-compression/` or clone to your preferred location

---

## Copy Data from Dev Container to Host

If your data is still in the dev container:

```bash
# From your Arch Linux host, copy the repository
# Option 1: Use git (recommended)
cd ~/path/to/your/projects
git clone <your-repo-url>
cd ids-compression

# Option 2: If you have Docker access, copy from container
docker cp <container-id>:/workspaces/ids-compression ./ids-compression

# Copy raw data if you have it in the container
docker cp <container-id>:/workspaces/ids-compression/data/raw ./ids-compression/data/
```

---

## Complete Setup (5 Commands)

```bash
# 1. Navigate to project
cd ~/ids-compression  # Adjust path as needed

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify data is present
ls -lh data/raw/cic_ids_2017/  # Should show 8 CSV files (~844MB total)

# 4. Run preprocessing (choose ONE):
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick   # Fast: ~10 min
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode medium  # Medium: ~20 min
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full    # Full: ~45 min

# 5. Validate data
python -m src.data.cli validate --dataset cic_ids_2017
```

---

## Run Experiments

### Quick Test (Recommended First Run)
```bash
python main_phase1.py --quick
# Time: ~30 minutes
# Trains 3 models for 10 epochs each
```

### Full Training (Production Results)
```bash
python main_phase1.py
# Time: ~3 hours
# Trains 3 models for 50 epochs each
```

---

## What Changed from Dev Container?

The code has been optimized for your 11.8GB RAM machine:

**Before (Dev Container - 2.7GB RAM):**
- Split sample size across files (~12,500 rows per file)
- Complex chunked reading with aggressive memory limits
- Frequently crashed with OOM

**Now (Your Arch Linux - 11.8GB RAM):**
- Load all files normally (pandas handles it efficiently)
- Sample once at the end (preserves chronological order)
- Much faster and more stable
- No memory issues expected

---

## Expected Results

After running `python main_phase1.py --quick`, you'll see:

```
experiments/Phase1_Baseline_CIC_IDS_2017_<timestamp>/
├── all_results.json          # Compare all 3 models
├── plots/
│   ├── model_comparison.png  # Accuracy comparison
│   ├── parameter_efficiency.png
│   └── ...
└── runs/
    ├── mlp/
    ├── ds_cnn/
    └── lstm/
```

**Expected Accuracy (on CIC-IDS2017):**
- MLP: ~94-96%
- DS-CNN: ~96-98% (best efficiency)
- LSTM: ~96-98%

---

## Monitoring During Training

Open a second terminal and run:

```bash
# Watch memory usage
watch -n 2 'free -h | head -3'

# Watch CPU usage
htop

# Watch log output
tail -f experiments/Phase1_*/logs/*.log
```

---

## Common Commands

```bash
# Clean preprocessed data and start fresh
rm -rf data/processed/cic_ids_2017/

# Clean all experiments
rm -rf experiments/Phase1_*

# Check data sizes
du -sh data/raw/cic_ids_2017/        # Raw: ~844MB
du -sh data/processed/cic_ids_2017/  # Processed: depends on mode

# List saved models
find experiments/ -name "*.pth"
```

---

## Troubleshooting One-Liners

```bash
# Python version check
python --version  # Need 3.8+

# Package verification
python -c "import torch, pandas, numpy; print('All good!')"

# Check if preprocessed data exists
ls -lh data/processed/cic_ids_2017/train/*.npy

# Quick test without training
python -c "from src.data.loaders import create_data_loaders; print('Loaders work!')"

# Memory available
free -h | grep Mem
```

---

## Timeline

| Task | Time | When to Run |
|------|------|-------------|
| Setup + Install | 5 min | Once |
| Preprocess (quick) | 10 min | Once per mode |
| Validate | <1 min | After preprocessing |
| Train (quick) | 30 min | For testing |
| Train (full) | 3 hours | For final results |

**Total for quick test:** ~45 minutes  
**Total for full experiment:** ~4 hours

---

## Success Indicators

✅ Preprocessing successful:
```
✓ train/X.npy: shape=(70000, 8, 12)
✓ All checks passed
```

✅ Training successful:
```
Epoch 10/10: 100%|████████| loss: 0.0234, acc: 0.9856
Saved best model: experiments/.../mlp/checkpoints/best_model_epoch_8.pth
```

✅ Experiment complete:
```
All models trained successfully!
Results saved to: experiments/Phase1_Baseline_CIC_IDS_2017_<timestamp>/
```

---

## Need Help?

1. Check `ARCH_SETUP.md` for detailed troubleshooting
2. Check `PHASE1_HOWTO.md` for full documentation
3. Look at logs: `cat experiments/Phase1_*/logs/*.log`
4. Validate data: `python -m src.data.cli validate --dataset cic_ids_2017`
