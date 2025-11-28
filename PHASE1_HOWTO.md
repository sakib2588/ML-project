# Phase 1 Baseline Model Comparison

This guide walks you through testing three baseline models on CIC-IDS2017 to measure:
- **Classification metrics**: Accuracy, F1-score (macro), per-class recall
- **Efficiency metrics**: Parameter count, FLOPs
- **Performance metrics**: CPU inference time, training time

## Models Being Tested

1. **Small MLP** (~50K parameters)
   - 2-3 dense layers (128 → 64 → 32 neurons)
   - Batch normalization and dropout
   - Flattens windowed input

2. **DS-1D-CNN** (~80K parameters)
   - 3 depthwise-separable conv blocks (32, 64, 64 filters)
   - Parameter-efficient convolutions
   - Global average pooling + classifier

3. **Small LSTM** (~90-120K parameters)
   - 2 LSTM layers (64 hidden units)
   - Handles sequential/windowed data naturally
   - Dropout for regularization

## Quick Start (3 Steps)

> **Note for Host Machine Users:** If you're running on Arch Linux or another host system (not dev container), see `ARCH_SETUP.md` for optimized instructions specific to your hardware.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`

### Step 2: Preprocess the Data

Convert raw CIC-IDS2017 CSV files into windowed numpy arrays using the built-in preprocessing pipeline:

```bash
# Quick mode (uses sample of data for testing)
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick

# Full mode (processes all data - takes longer)
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full
```

**What this does:**
- Loads all CSV files from `data/raw/cic_ids_2017/` (configured in `configs/data_config.yaml`)
- Applies feature engineering pipeline (from `src/preprocessing/feature_engineering.py`)
- Cleans data (handles inf/nan per `configs/preprocess_config.yaml`)
- Encodes labels using trained label map
- Creates sliding windows (configured windowing strategy)
- Splits chronologically into train/val/test
- Scales features using fitted scaler
- Saves as `X.npy` and `y.npy` in `data/processed/cic_ids_2017/{train,val,test}/`
- Saves scaler, label_map, and preprocessing report for reproducibility

**Output structure:**
```
data/processed/cic_ids_2017/
├── train/
│   ├── X.npy  # shape: (N_train, 8, 12)
│   └── y.npy  # shape: (N_train,)
├── val/
│   ├── X.npy
│   └── y.npy
├── test/
│   ├── X.npy
│   └── y.npy
└── metadata.json
```

**Note:** The `quick` mode uses a sample configured in `configs/data_config.yaml` (under `sampling.quick_mode.samples_per_dataset`). The `full` mode processes all data. The `medium` mode provides a middle ground for validation runs.

### Step 3: Validate Data (Optional but Recommended)

Check that preprocessing worked correctly:

```bash
python -m src.data.cli validate --dataset cic_ids_2017
```

This will:
- Verify all data files exist (X.npy, y.npy for each split)
- Check array shapes and dtypes
- Verify scaler and label_map artifacts
- Display preprocessing report (sample counts, label distribution)
- Confirm data is ready for training

### Step 4: Run the Experiment

**Quick test (10 epochs, ~20-30 minutes on CPU):**
```bash
python main_phase1.py --quick
```

**Full run (50 epochs, ~2-3 hours on CPU):**
```bash
python main_phase1.py
```

**Train only specific models:**
```bash
python main_phase1.py --quick --models mlp ds_cnn
```

## What Happens During Training

The experiment script will:

1. **Load preprocessed data** using memory-mapped numpy arrays (efficient for large datasets)

2. **Train each model sequentially**:
   - Small MLP → DS-1D-CNN → Small LSTM
   - Each gets its own subdirectory in `experiments/Phase1_Baseline_CIC_IDS_2017_<timestamp>/runs/<model_name>/`

3. **For each model**:
   - Train for specified epochs with early stopping
   - Save best checkpoint based on validation loss
   - Log metrics to JSON and CSV files
   - Show progress bars with tqdm

4. **After all models finish**:
   - Generate comparison plots (accuracy, parameter efficiency, etc.)
   - Save aggregated results to `all_results.json`
   - Print summary table

## Understanding the Results

### Results Directory Structure

```
experiments/Phase1_Baseline_CIC_IDS_2017_<timestamp>/
├── config.json                    # Experiment configuration
├── all_results.json              # All metrics for all models
├── all_histories.json            # Training curves
├── logs/                         # Experiment-level logs
├── plots/                        # Comparison visualizations
│   ├── model_comparison.png
│   ├── parameter_efficiency.png
│   └── ...
└── runs/
    ├── mlp/
    │   ├── checkpoints/
    │   │   ├── best_model_*.pth
    │   │   └── final_model.pth
    │   ├── logs/
    │   │   └── metrics_*.json
    │   ├── results.json          # MLP evaluation results
    │   └── training_history.json
    ├── ds_cnn/
    │   └── ...
    └── lstm/
        └── ...
```

### Key Metrics in `results.json`

Each model's `results.json` contains:

```json
{
  "model_name": "ds_cnn",
  "accuracy": 0.9823,
  "f1_macro": 0.9756,
  "f1_weighted": 0.9834,
  "precision_macro": 0.9712,
  "recall_macro": 0.9801,
  "per_class_metrics": {
    "0": {"precision": 0.99, "recall": 0.98, "f1-score": 0.98},
    "1": {"precision": 0.95, "recall": 0.98, "f1-score": 0.97}
  },
  "confusion_matrix": [[1234, 23], [15, 892]],
  "parameters": 79824,
  "flops": 124567,
  "inference_time": {
    "mean_ms": 2.34,
    "std_ms": 0.12,
    "min_ms": 2.21,
    "max_ms": 3.45
  },
  "training_epochs": 35,
  "timestamp": "20231120_143022"
}
```

### Interpreting Results

**Classification Performance:**
- `accuracy`: Overall correctness (should be >95% for good IDS)
- `f1_macro`: Balanced metric accounting for both classes
- `per_class_metrics`: Check recall for attack class (class 1) - high recall means fewer false negatives

**Efficiency:**
- `parameters`: Model size (lower is better for edge deployment)
- `flops`: Computational cost (lower is better)

**Performance:**
- `inference_time.mean_ms`: Average time per sample (lower is better)
- `training_time_minutes`: How long training took

**What to look for:**
- DS-1D-CNN should have similar accuracy to LSTM but with ~30-40% fewer parameters
- MLP will likely be fastest but may have slightly lower accuracy
- LSTM may have best accuracy but slowest inference

## Troubleshooting

### Issue: "No CSV files found"
**Solution:** Ensure raw CSV files are in the directory specified in `configs/data_config.yaml` under `datasets.cic_ids_2017.raw_data_dir`

### Issue: "Missing data files"
**Solution:** Run preprocessing: `python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick`

### Issue: "RuntimeError: Train loader is empty"
**Solution:** Check that `X.npy` and `y.npy` files are not empty/corrupted using the validate command

### Issue: Out of memory during preprocessing
**Solution:** Use `quick` mode instead of `full`, or adjust `samples_per_dataset` in `configs/data_config.yaml`

### Issue: Training is very slow
**Solutions:**
- Use `--quick` mode (10 epochs instead of 50)
- Reduce `num_workers` in config (set to 2 or 0)
- Use smaller sample size during preprocessing
- If you have a GPU, set `use_amp: True` in the config

### Issue: "ImportError: No module named 'src.models.mlpp'"
**Solution:** The factory in `main_phase1.py` expects `src/models/mlp.py` to contain `SmallMLP` class. Check the import statement matches your actual class name.

## Advanced Usage

### Custom Configuration

Edit `run_phase1_experiment.py` to modify:
- Batch size
- Learning rate
- Number of epochs
- Model hyperparameters
- Callbacks (early stopping, checkpointing)

### Running Individual Models

To train only one model:

```python
python -c "
from run_phase1_experiment import build_config
from main_phase1 import run_full_experiment

config = build_config(quick_mode=True)
config['models'] = ['ds_cnn']  # Only train DS-CNN

run_full_experiment(config, seed=42)
"
```

### Using GPU

If you have a CUDA-capable GPU:

1. Install PyTorch with CUDA support
2. Edit `run_phase1_experiment.py`:
   ```python
   "training": {
       "use_amp": True,  # Enable automatic mixed precision
       ...
   }
   ```

The trainer will automatically use GPU if available.

## Next Steps

After completing Phase 1:

1. **Analyze results** in `experiments/.../all_results.json`
2. **Review plots** in `experiments/.../plots/`
3. **Select best architecture** (likely DS-1D-CNN based on efficiency + accuracy)
4. **Proceed to Phase 2**: Knowledge distillation using the selected student architecture

## Expected Timeline

- **Preprocessing**: 5-10 minutes (200K samples), 30-60 minutes (full dataset)
- **Quick experiment**: 20-30 minutes (10 epochs × 3 models)
- **Full experiment**: 2-3 hours (50 epochs × 3 models)

Times are approximate for CPU (i3-7100U). GPU would be 5-10x faster.

## Questions or Issues?

Check the logs in `experiments/.../logs/` for detailed error messages and training progress.
