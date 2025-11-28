# Setup Guide for Arch Linux Host Machine

This guide is optimized for your Arch Linux system (Intel i3-7100U, 11.8GB RAM).

## Prerequisites

Your system specs:
- **OS**: Arch Linux x86_64
- **CPU**: Intel i3-7100U (4 cores) @ 2.400GHz
- **RAM**: 11.8 GB (sufficient for full dataset processing)
- **GPU**: Intel HD Graphics 620 (integrated)

## Step 1: Install Python Dependencies

### Option A: Using pip (Recommended)

```bash
# Ensure you have Python 3.8+ installed
python --version

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import pandas; import numpy; print('All packages installed successfully')"
```

### Option B: Using pacman + pip

```bash
# Install system packages (if not already installed)
sudo pacman -S python python-pip python-numpy python-pandas python-matplotlib python-yaml

# Install PyTorch and other dependencies
pip install torch torchvision scikit-learn seaborn tqdm psutil joblib
```

## Step 2: Prepare the Data Directory

Ensure your raw CIC-IDS2017 CSV files are in the correct location:

```bash
# Check if data exists
ls -lh data/raw/cic_ids_2017/

# You should see 8 CSV files:
# - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
# - Friday-WorkingHours-Morning.pcap_ISCX.csv
# - Monday-WorkingHours.pcap_ISCX.csv
# - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
# - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
# - Tuesday-WorkingHours.pcap_ISCX.csv
# - Wednesday-workingHours.pcap_ISCX.csv
```

If data is not present, download it from:
- Official source: https://www.unb.ca/cic/datasets/ids-2017.html
- Or use Kaggle: `kaggle datasets download -d cicdataset/cicids2017`

## Step 3: Run Preprocessing (UPDATED FOR HOST MACHINE)

The preprocessing pipeline has been optimized for your system's 11.8GB RAM:

### Quick Mode (Recommended for initial testing)
```bash
# Clears old data and processes ~100K samples
rm -rf data/processed/cic_ids_2017
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick --overwrite

# Expected time: ~5-10 minutes on your i3-7100U
```

### Medium Mode (Balanced for validation)
```bash
# Processes ~250K samples
rm -rf data/processed/cic_ids_2017
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode medium --overwrite

# Expected time: ~15-20 minutes
```

### Full Mode (All data - ~2.8M samples)
```bash
# Processes entire dataset
rm -rf data/processed/cic_ids_2017
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full --overwrite

# Expected time: ~30-45 minutes
# Memory usage: Peak ~4-6 GB (well within your 11.8GB)
```

**What changed from dev container version:**
- Removed aggressive per-file sampling (was needed for 2.7GB container)
- Load all files normally, then sample once at the end
- Much faster and preserves chronological order better
- Your 11.8GB RAM can easily handle the full ~844MB raw data

## Step 4: Validate Preprocessing

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
```

## Step 5: Run Phase 1 Experiment

### Quick Test (10 epochs)
```bash
python main_phase1.py --quick

# Expected time: ~20-30 minutes for 3 models
# Results saved to: experiments/Phase1_Baseline_CIC_IDS_2017_<timestamp>/
```

### Full Training (50 epochs)
```bash
python main_phase1.py

# Expected time: ~2-3 hours for 3 models on CPU
# Use this for final results
```

### Train Specific Models
```bash
# Only train MLP and DS-CNN (skip LSTM)
python main_phase1.py --quick --models mlp ds_cnn
```

## Performance Expectations on Your Hardware

| Task | Expected Time | Memory Usage |
|------|---------------|--------------|
| Preprocessing (quick) | 5-10 min | ~2-3 GB |
| Preprocessing (full) | 30-45 min | ~4-6 GB |
| Training 1 model (quick) | 5-10 min | ~1-2 GB |
| Training 1 model (full) | 40-60 min | ~1-2 GB |
| Full experiment (3 models, quick) | 20-30 min | ~1-2 GB |
| Full experiment (3 models, full) | 2-3 hours | ~1-2 GB |

**CPU Notes:**
- Your i3-7100U is dual-core with hyperthreading (4 threads)
- Neural network training is CPU-bound without GPU
- Consider running overnight for full 50-epoch training

**GPU Notes:**
- Intel HD Graphics 620 is integrated graphics, not suitable for PyTorch training
- If you have access to a discrete NVIDIA GPU, you can enable GPU training
- Otherwise, CPU training is perfectly fine for this experiment

## Monitoring System Resources

While training, monitor resource usage:

```bash
# Watch memory usage in real-time
watch -n 2 free -h

# Monitor CPU usage
htop

# Check process memory
ps aux --sort=-%mem | head -10
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Still getting OOM during preprocessing
```bash
# Very unlikely with 11.8GB, but if it happens:
# 1. Close other applications (browser, etc.)
# 2. Use quick mode instead of full
# 3. Check available RAM: free -h
```

### Issue: Training is very slow
```bash
# Reduce batch size in the config
# Edit configs/phase1_config.yaml:
training:
  batch_size: 32  # Try 16 or 8 if needed
```

### Issue: Python using system Python instead of venv
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps After Successful Run

1. Check results in `experiments/Phase1_Baseline_*/all_results.json`
2. View plots in `experiments/Phase1_Baseline_*/plots/`
3. Compare models: DS-CNN should have best accuracy/efficiency tradeoff
4. Proceed to Phase 2 (Knowledge Distillation)

## Estimated Total Time for Complete Phase 1

**Quick Mode (Recommended for initial test):**
- Preprocessing: ~10 minutes
- Training (3 models, 10 epochs each): ~30 minutes
- **Total: ~40 minutes**

**Full Mode (Production results):**
- Preprocessing: ~45 minutes
- Training (3 models, 50 epochs each): ~3 hours
- **Total: ~4 hours**

You can run the full experiment overnight or while working on other tasks.
