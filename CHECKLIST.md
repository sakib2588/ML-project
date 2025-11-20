# Pre-Flight Checklist: Running on Arch Linux

Complete this checklist before running experiments on your host machine.

## ☑️ System Requirements

- [ ] **RAM**: At least 8 GB (you have 11.8 GB ✅)
- [ ] **Storage**: At least 5 GB free for data + experiments
- [ ] **Python**: Version 3.8 or higher
- [ ] **Internet**: For pip installs (one-time only)

Check:
```bash
free -h          # Confirm 11.8 GB RAM
df -h .          # Check disk space
python --version # Confirm Python 3.8+
```

---

## ☑️ Repository Setup

- [ ] Repository cloned/copied to your Arch machine
- [ ] Can navigate to project directory

```bash
cd ~/path/to/ids-compression  # Adjust to your path
ls -la  # Should see main_phase1.py, requirements.txt, etc.
```

---

## ☑️ Dependencies Installed

- [ ] All Python packages installed
- [ ] No import errors

```bash
pip install -r requirements.txt

# Test imports
python -c "import torch; import pandas; import numpy; import sklearn; print('✅ All packages OK')"
```

Expected output: `✅ All packages OK`

---

## ☑️ Raw Data Present

- [ ] CIC-IDS2017 CSV files are in `data/raw/cic_ids_2017/`
- [ ] All 8 files present (~844 MB total)

```bash
ls -lh data/raw/cic_ids_2017/

# Should list:
# Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
# Friday-WorkingHours-Morning.pcap_ISCX.csv
# Monday-WorkingHours.pcap_ISCX.csv
# Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
# Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
# Tuesday-WorkingHours.pcap_ISCX.csv
# Wednesday-workingHours.pcap_ISCX.csv
```

If files are missing, download from:
- https://www.unb.ca/cic/datasets/ids-2017.html
- Or: `kaggle datasets download -d cicdataset/cicids2017`

---

## ☑️ Configuration Files

- [ ] `configs/data_config.yaml` exists
- [ ] `configs/preprocess_config.yaml` exists
- [ ] `configs/phase1_config.yaml` exists (if needed)

```bash
ls -lh configs/
```

Should show at least 2 YAML files.

---

## ☑️ Code Updated

- [ ] Pipeline optimized for host machine (already done ✅)
- [ ] No Pylance errors (already verified ✅)

---

## ☑️ Ready to Preprocess

Choose your mode:

### Option 1: Quick Mode (Recommended First)
```bash
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick --overwrite
```
- **Samples**: ~100K
- **Time**: ~10 minutes
- **Memory**: ~2-3 GB
- **Use case**: Initial testing

### Option 2: Medium Mode
```bash
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode medium --overwrite
```
- **Samples**: ~250K
- **Time**: ~20 minutes
- **Memory**: ~3-4 GB
- **Use case**: Validation runs

### Option 3: Full Mode
```bash
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full --overwrite
```
- **Samples**: ~2.8M (all data)
- **Time**: ~45 minutes
- **Memory**: ~4-6 GB
- **Use case**: Final production results

---

## ☑️ Validate Preprocessing

After preprocessing completes:

```bash
python -m src.data.cli validate --dataset cic_ids_2017
```

Expected output:
```
✓ train/X.npy: shape=(N, 8, 12), dtype=float32
✓ train/y.npy: shape=(N,), dtype=int64
✓ val/X.npy: shape=(M, 8, 12), dtype=float32
✓ val/y.npy: shape=(M,), dtype=int64
✓ test/X.npy: shape=(K, 8, 12), dtype=float32
✓ test/y.npy: shape=(K,), dtype=int64
✓ scaler.joblib: exists
✓ label_map.joblib: exists
✓ preprocessing_report.json: exists

✓ ALL CHECKS PASSED - Data is ready for training
```

If any files are missing, re-run preprocessing.

---

## ☑️ Ready to Train

Choose your experiment:

### Option 1: Quick Test (Recommended First)
```bash
python main_phase1.py --quick
```
- **Epochs**: 10 per model
- **Models**: MLP, DS-CNN, LSTM
- **Time**: ~30 minutes
- **Purpose**: Verify everything works

### Option 2: Full Training
```bash
python main_phase1.py
```
- **Epochs**: 50 per model
- **Models**: MLP, DS-CNN, LSTM
- **Time**: ~3 hours
- **Purpose**: Production results

### Option 3: Specific Models
```bash
python main_phase1.py --quick --models mlp ds_cnn
```
- Train only selected models
- Useful for debugging

---

## ☑️ Monitor Training (Optional)

Open a second terminal:

```bash
# Watch memory
watch -n 2 'free -h | head -3'

# Watch CPU
htop

# Watch logs
tail -f experiments/Phase1_*/logs/*.log
```

---

## ☑️ After Training Completes

- [ ] Check results directory exists
- [ ] Review `all_results.json`
- [ ] View plots in `plots/` directory

```bash
# Find results
ls -lh experiments/Phase1_Baseline*/

# View results
cat experiments/Phase1_Baseline*/all_results.json | python -m json.tool

# Open plots
xdg-open experiments/Phase1_Baseline*/plots/model_comparison.png
```

---

## Success Criteria

✅ **Preprocessing Successful:**
- All .npy files created
- Validation passes
- No errors in preprocessing_report.json

✅ **Training Successful:**
- All 3 models trained
- Checkpoints saved
- Accuracy > 90% for each model

✅ **Experiment Complete:**
- `all_results.json` contains metrics for all models
- Plots generated
- No Python errors

---

## If Something Goes Wrong

### Preprocessing Fails
1. Check available RAM: `free -h`
2. Close other applications
3. Try `quick` mode first
4. Check raw data files exist

### Training Fails
1. Check preprocessed data: `python -m src.data.cli validate --dataset cic_ids_2017`
2. Check Python imports: `python -c "import torch; import src.models.mlp"`
3. Look at error logs in `experiments/*/logs/`
4. Try with single model: `python main_phase1.py --quick --models mlp`

### Out of Memory During Training
1. Reduce batch size in config
2. Close other applications
3. Train one model at a time

---

## Estimated Timeline

| Phase | Time | Notes |
|-------|------|-------|
| Setup + Install | 5 min | One-time |
| Preprocessing (quick) | 10 min | Per mode |
| Validation | <1 min | Quick check |
| Training (quick) | 30 min | 3 models |
| Review results | 5 min | Check plots/metrics |
| **Total (quick)** | **~50 min** | First complete run |
| **Total (full)** | **~4.5 hours** | Production results |

---

## Ready to Start?

Run these commands in order:

```bash
# 1. Verify setup
python --version && python -c "import torch, pandas, numpy; print('✅ Ready')"

# 2. Preprocess (choose mode)
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick --overwrite

# 3. Validate
python -m src.data.cli validate --dataset cic_ids_2017

# 4. Train (choose mode)
python main_phase1.py --quick

# 5. Check results
ls -lh experiments/Phase1_Baseline*/
```

Good luck! 🚀

---

## Documentation Reference

- **Quick Start**: `QUICK_START_ARCH.md`
- **Detailed Setup**: `ARCH_SETUP.md`
- **Migration Info**: `MIGRATION_SUMMARY.md`
- **General Guide**: `PHASE1_HOWTO.md`
