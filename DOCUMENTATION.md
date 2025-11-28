# IDS Compression Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Directory Structure](#directory-structure)
4. [Module Documentation](#module-documentation)
   - [Data Module](#data-module-srcdata)
   - [Preprocessing Module](#preprocessing-module-srcpreprocessing)
   - [Models Module](#models-module-srcmodels)
   - [Training Module](#training-module-srctraining)
   - [Utils Module](#utils-module-srcutils)
5. [Data Flow](#data-flow)
6. [Function Reference](#function-reference)
7. [Configuration Files](#configuration-files)
8. [Usage Guide](#usage-guide)

---

## Project Overview

This project implements a **research-grade Intrusion Detection System (IDS)** using deep learning with a focus on model compression and efficiency. The pipeline processes network traffic data (CIC-IDS2017, TON-IoT), applies feature engineering, creates windowed sequences, and trains multiple baseline models (MLP, DS-CNN, LSTM) for binary classification (Normal vs Attack).

### Key Features
- **Modular Architecture**: Clean separation between data, preprocessing, models, and training
- **Research-Grade Code**: Type hints, logging, atomic file operations, reproducibility
- **Memory-Efficient**: Memory-mapped data loading, configurable batch sizes
- **Class Imbalance Handling**: Weighted loss functions, F1-macro monitoring
- **Comprehensive Evaluation**: Multiple metrics, efficiency profiling (FLOPs, inference time)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main_phase1.py                                  │
│                         (Experiment Orchestrator)                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   src/data/     │    │  src/models/    │    │ src/training/   │
│                 │    │                 │    │                 │
│ • loaders.py    │    │ • base_model.py │    │ • trainer.py    │
│ • validators.py │    │ • mlp.py        │    │ • evaluator.py  │
│ • cli.py        │    │ • ds_cnn.py     │    │ • callbacks.py  │
└────────┬────────┘    │ • lstm.py       │    └────────┬────────┘
         │             └────────┬────────┘             │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        src/preprocessing/                        │
│                                                                  │
│  • pipeline.py ──────┬──── feature_engineering.py                │
│                      ├──── scalers.py                            │
│                      └──── windowing.py                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                          src/utils/                              │
│                                                                  │
│  • config.py          • logging_utils.py    • metrics.py         │
│  • visualization.py   • system_utils.py                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
ids-compression/
├── main_phase1.py              # Main experiment runner
├── requirements.txt            # Python dependencies
├── configs/
│   ├── data_config.yaml        # Dataset paths, sampling config
│   ├── preprocess_config.yaml  # Feature mappings, windowing settings
│   └── phase1_config.yaml      # Model/training hyperparameters
├── data/
│   ├── raw/                    # Original CSV datasets
│   │   ├── cic_ids_2017/       # CIC-IDS2017 CSV files
│   │   └── ton_iot/            # TON-IoT CSV files
│   ├── processed/              # Preprocessed numpy arrays
│   │   └── cic_ids_2017/
│   │       ├── train/X.npy, y.npy
│   │       ├── val/X.npy, y.npy
│   │       ├── test/X.npy, y.npy
│   │       ├── scaler.joblib
│   │       └── label_map.joblib
│   └── samples/                # Sample data for testing
├── src/
│   ├── data/                   # Data loading and validation
│   ├── preprocessing/          # Feature engineering pipeline
│   ├── models/                 # Neural network architectures
│   ├── training/               # Training loop and evaluation
│   └── utils/                  # Utilities and helpers
├── experiments/                # Experiment outputs
│   └── Phase1_Baseline_CIC_IDS_2017/
│       ├── config.json
│       ├── all_results.json
│       ├── runs/{model_name}/
│       │   ├── checkpoints/
│       │   ├── logs/
│       │   └── results.json
│       └── plots/
└── logs/                       # Global logs
```

---

## Module Documentation

### Data Module (`src/data/`)

This module handles data loading, validation, and PyTorch Dataset/DataLoader creation.

#### `loaders.py`

**Purpose**: Creates PyTorch DataLoaders for training, validation, and testing.

##### Classes

| Class | Description |
|-------|-------------|
| `IDSDataset` | PyTorch Dataset that loads preprocessed numpy arrays (X.npy, y.npy) |

##### Key Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `create_data_loaders()` | `train_path`, `val_path`, `test_path`, `batch_size`, `num_workers`, `mode`, etc. | `Tuple[DataLoader, DataLoader, DataLoader]` | Factory function to create train/val/test DataLoaders |
| `_default_worker_init_fn()` | `worker_id`, `seed` | `None` | Worker initialization for multiprocessing; reopens memmaps in worker processes |
| `default_collate_variable_length()` | `batch` | `Dict[str, Tensor]` | Collate function that pads variable-length sequences |

##### `IDSDataset` Class

```python
class IDSDataset(Dataset):
    def __init__(
        self,
        data_path: Path,           # Directory containing X.npy, y.npy
        mode: str = "memmap",      # "memmap" or "memory"
        transform: Optional = None,
        target_transform: Optional = None,
        return_dict: bool = False  # Return {"x": x, "y": y} or (x, y)
    )
```

**How it works**:
1. Loads `X.npy` (features) and `y.npy` (labels) from `data_path`
2. If `mode="memmap"`, uses memory-mapped files (efficient for large datasets)
3. If `mode="memory"`, loads entire arrays into RAM (faster but more memory)
4. `__getitem__` returns individual samples, converting to PyTorch tensors

---

#### `validators.py`

**Purpose**: Validates dataset integrity before processing.

##### Classes

| Class | Description |
|-------|-------------|
| `DataValidator` | Checks DataFrames for completeness, quality, and consistency |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `validate_dataframe()` | `df`, `required_columns`, `label_column` | `bool` | Validates a DataFrame, checking for missing values, duplicates, class imbalance |
| `get_validation_report()` | None | `Dict` | Returns detailed validation report with issues |

**How it works**:
1. Checks for required columns
2. Validates label column exists
3. Reports missing value percentages
4. Detects class imbalance using configurable threshold
5. Raises errors in `strict_mode` or logs warnings in `flexible_mode`

---

#### `cli.py`

**Purpose**: Command-line interface for data preprocessing.

##### Commands

```bash
# Preprocess raw data
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick

# Validate preprocessed data
python -m src.data.cli validate --dataset cic_ids_2017
```

**How it works**:
1. Loads configuration files (`data_config.yaml`, `preprocess_config.yaml`)
2. Instantiates `PreprocessingPipeline`
3. Processes dataset with specified mode (quick/medium/full)
4. Reports memory usage and output statistics

---

### Preprocessing Module (`src/preprocessing/`)

This module transforms raw CSV data into windowed, scaled numpy arrays ready for training.

#### `pipeline.py`

**Purpose**: Main preprocessing orchestrator that coordinates all transformation steps.

##### Classes

| Class | Description |
|-------|-------------|
| `PreprocessingPipeline` | End-to-end preprocessing from raw CSV to train/val/test numpy arrays |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `process_dataset()` | `dataset_name`, `output_dir`, `mode`, `overwrite` | `Dict[str, Any]` | Full preprocessing pipeline |

**Pipeline Steps** (in order):
```
Raw CSVs → Load & Concat → Chronological Sort → Train/Val/Test Split
    → Feature Engineering (per split) → Label Encoding → Windowing
    → Scaling (fit on train) → Save numpy arrays + artifacts
```

**How `process_dataset()` works**:

```python
def process_dataset(self, dataset_name, output_dir, mode="quick", overwrite=False):
    # 1. Load raw CSV files
    df = self._load_raw_data(dataset_name, mode)
    
    # 2. Sort by time column (chronological)
    df = df.sort_values(time_column)
    
    # 3. Chronological hard split (NO shuffling)
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]
    
    # 4. Feature engineering per split
    fe_train, _ = self._apply_feature_engineer(df_train, dataset_name)
    fe_val, _ = self._apply_feature_engineer(df_val, dataset_name)
    fe_test, _ = self._apply_feature_engineer(df_test, dataset_name)
    
    # 5. Create label mapping from train data only
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # 6. Create windows
    X_train_w, y_train_w, _, _ = self.windower.create_windows(X_train_flow, y_train_mapped)
    
    # 7. Fit scaler on train, transform all splits
    self.scaler.fit(X_train_w)
    X_train_scaled = self.scaler.transform(X_train_w)
    
    # 8. Save arrays atomically
    np.save(output_dir / "train" / "X.npy", X_train_scaled)
    np.save(output_dir / "train" / "y.npy", y_train_w)
```

---

#### `feature_engineering.py`

**Purpose**: Transforms raw columns into canonical features using config-driven mappings.

##### Classes

| Class | Description |
|-------|-------------|
| `FeatureEngineer` | Config-driven feature extraction and transformation |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `transform()` | `df` | `Tuple[DataFrame, Dict]` | Transform raw DataFrame to canonical features |

**Configuration-Driven Approach**:

The `preprocess_config.yaml` defines feature mappings:

```yaml
feature_mappings:
  cic_ids_2017:
    Flow Duration:
      source_column: "Flow Duration"
      transform: "divide_by_1000000"  # Convert microseconds to seconds
      dtype: "float"
    
    Total Fwd Packets:
      source_column: "Total Fwd Packets"
      transform: "to_numeric"
    
    # Derived feature using formula
    Fwd_Packets_per_sec:
      formula: "Total_Fwd_Packets / Flow_Duration"
```

**Transform Registry**:

| Transform | Description |
|-----------|-------------|
| `identity` | No transformation |
| `log` | Natural log (with epsilon) |
| `log1p` | log(1+x) for values near zero |
| `sqrt` | Square root |
| `to_numeric` | Convert to numeric, coerce errors |
| `divide_by_N` | Divide by constant N |
| `binary_encode` | Encode as 0/1 for labels |

**How `transform()` works**:

```python
def transform(self, df):
    canonical_df = pd.DataFrame(index=df.index)
    
    for feature_name in canonical_features:
        mapping = self.mapping.get(feature_name, {})
        
        if "formula" in mapping:
            # Safe AST-based formula evaluation
            series = _safe_eval_series(mapping["formula"], df)
        elif "source_column" in mapping:
            series = df[mapping["source_column"]]
        
        # Apply transform
        if mapping.get("transform"):
            transform_fn = get_transform_callable(mapping["transform"])
            series = transform_fn(series)
        
        canonical_df[feature_name] = series
    
    return canonical_df, diagnostics
```

---

#### `windowing.py`

**Purpose**: Creates fixed-length windows from sequential flow data.

##### Classes

| Class | Description |
|-------|-------------|
| `FlowWindower` | Creates sliding or grouped windows from flow sequences |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_windows()` | `features`, `labels`, `return_indices` | `Tuple[np.ndarray, np.ndarray, Dict, Optional[np.ndarray]]` | Create windows from numpy arrays |
| `from_dataframe()` | `df`, `feature_columns`, `label_column`, `groupby` | Same as above | Create windows from DataFrame |

##### Configuration Options

| Option | Values | Description |
|--------|--------|-------------|
| `window_length` | int (default: 8) | Number of flows per window |
| `stride` | int (default: 1) | Step between consecutive windows |
| `mode` | "sliding" / "grouped" | Sliding = overlapping; Grouped = non-overlapping |
| `padding_strategy` | "zero" / "repeat" / "none" | How to handle incomplete windows |
| `label_strategy` | "any_malicious" / "majority" / "first" / "last" | How to assign window label |

**Label Strategies**:

| Strategy | Description |
|----------|-------------|
| `any_malicious` | Window is attack (1) if ANY flow in window is attack |
| `majority` | Window label = majority class in window |
| `first` | Window label = first flow's label |
| `last` | Window label = last flow's label |

**How `create_windows()` works**:

```python
def create_windows(self, features, labels):
    # features: (n_flows, n_features)
    # labels: (n_flows,)
    
    windows_list = []
    for start in range(0, n_flows - window_length + 1, stride):
        end = start + window_length
        window = features[start:end]  # Shape: (window_length, n_features)
        window_label = self._label_window(labels[start:end])
        windows_list.append(window)
    
    # Handle padding for remaining flows if needed
    if padding_strategy != "none":
        # Pad incomplete window
        
    return np.stack(windows_list), np.array(labels_list), diagnostics, indices
```

---

#### `scalers.py`

**Purpose**: Feature scaling/normalization with support for 3D windowed data.

##### Classes

| Class | Description |
|-------|-------------|
| `FeatureScaler` | Wrapper around sklearn scalers supporting 2D and 3D arrays |

##### Supported Methods

| Method | Description |
|--------|-------------|
| `standard` | StandardScaler (z-score normalization) |
| `minmax` | MinMaxScaler (scale to [0, 1]) |
| `robust` | RobustScaler (median-based, robust to outliers) |
| `power` | PowerTransformer (Yeo-Johnson) |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit()` | `X` | `self` | Fit scaler on training data |
| `transform()` | `X` | `np.ndarray` | Transform data using fitted scaler |
| `fit_transform()` | `X` | `np.ndarray` | Fit and transform in one step |
| `save()` | `path` | None | Save scaler to disk (joblib) |
| `load()` | `path` | `FeatureScaler` | Load scaler from disk |

**How it handles 3D windowed data**:

```python
def fit(self, X):
    # X shape: (n_windows, window_length, n_features)
    if X.ndim == 3 and self.fit_mode == "global":
        # Reshape to 2D for fitting
        n, L, f = X.shape
        X_flat = X.reshape(-1, f)  # (n*L, f)
        self._scaler.fit(X_flat)
    
def transform(self, X):
    # Flatten, transform, reshape back
    n, L, f = X.shape
    X_flat = X.reshape(-1, f)
    X_scaled_flat = self._scaler.transform(X_flat)
    return X_scaled_flat.reshape(n, L, f)
```

---

### Models Module (`src/models/`)

This module contains neural network architectures for IDS classification.

#### `base_model.py`

**Purpose**: Abstract base class defining common interface for all models.

##### Classes

| Class | Description |
|-------|-------------|
| `BaseIDSModel` | Abstract base class with common methods |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `forward()` | `x: Tensor` | `Tensor` | Abstract - forward pass (returns logits) |
| `count_parameters()` | None | `int` | Count trainable parameters |
| `get_flops()` | `input_tensor` | `int` | Calculate FLOPs using thop library |
| `get_model_info()` | `input_tensor` | `Dict` | Get model metadata dictionary |

---

#### `mlp.py`

**Purpose**: Multi-Layer Perceptron baseline model.

##### Classes

| Class | Description |
|-------|-------------|
| `SmallMLP` | Configurable MLP with optional BatchNorm, LayerNorm, dropout |

##### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `Tuple[int, int]` | Required | (window_length, n_features) |
| `num_classes` | `int` | 2 | Number of output classes |
| `hidden_sizes` | `Tuple[int, ...]` | (128, 64, 32) | Hidden layer sizes |
| `dropout_rate` | `float` | 0.3 | Dropout probability |
| `activation` | `str` | "relu" | Activation function |
| `use_batchnorm` | `bool` | False | Use BatchNorm layers |
| `flatten_input` | `bool` | True | Flatten 3D input to 2D |

**Architecture**:

```
Input (batch, window_len, n_features)
    ↓
Flatten → (batch, window_len * n_features)
    ↓
┌─────────────────────────────┐
│ For each hidden_size:       │
│   Linear → BatchNorm/LN?    │
│   → Activation → Dropout    │
└─────────────────────────────┘
    ↓
Linear → (batch, num_classes)  [logits]
```

---

#### `ds_cnn.py`

**Purpose**: Depthwise Separable 1D CNN (parameter-efficient).

##### Classes

| Class | Description |
|-------|-------------|
| `DS_1D_CNN` | Depthwise separable convolutions for sequence classification |
| `DepthwiseSeparableConv1d` | Single DS conv block |
| `_SEBlock` | Optional Squeeze-and-Excitation attention |

##### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `Tuple[int, int]` | Required | (window_length, n_features) |
| `conv_channels` | `Sequence[int]` | (32, 64, 64) | Output channels per conv block |
| `kernel_size` | `int` or `Sequence` | 3 | Convolution kernel size |
| `dropout_rate` | `float` | 0.2 | Dropout probability |
| `use_se` | `bool` | False | Use Squeeze-and-Excitation |
| `classifier_hidden` | `int` | 64 | Dense layer before output |

**Architecture**:

```
Input (batch, window_len, n_features)
    ↓
Transpose → (batch, n_features, window_len)  [channels first]
    ↓
┌─────────────────────────────────────┐
│ For each conv_channel:              │
│   Depthwise Conv1d (groups=in_ch)   │
│   → Pointwise Conv1d (1x1)          │
│   → BatchNorm → Activation          │
│   → Optional SE Block → Dropout     │
└─────────────────────────────────────┘
    ↓
Global Average Pooling → (batch, last_channels)
    ↓
Linear → ReLU → Dropout → Linear → (batch, num_classes)
```

**Why Depthwise Separable?**
- Standard conv: params = in_ch × out_ch × kernel_size
- DS conv: params = in_ch × kernel_size + in_ch × out_ch
- **Much fewer parameters** for similar representational power

---

#### `lstm.py`

**Purpose**: LSTM sequence model for capturing temporal dependencies.

##### Classes

| Class | Description |
|-------|-------------|
| `LSTMModel` | LSTM with optional bidirectional and classifier head |

##### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `Tuple[int, int]` | Required | (window_length, n_features) |
| `hidden_size` | `int` | 128 | LSTM hidden state dimension |
| `num_layers` | `int` | 2 | Number of stacked LSTM layers |
| `dropout` | `float` | 0.2 | Dropout between LSTM layers |
| `bidirectional` | `bool` | False | Use bidirectional LSTM |
| `fc_hidden` | `int` | 64 | Hidden size of classifier MLP |

**Architecture**:

```
Input (batch, window_len, n_features)
    ↓
LSTM (num_layers, hidden_size, bidirectional?)
    ↓
Take last timestep output → (batch, hidden_size * (2 if bidir else 1))
    ↓
Optional LayerNorm
    ↓
Linear → ReLU → Dropout → Linear → (batch, num_classes)
```

**Weight Initialization**:
- Input weights: Xavier uniform
- Hidden weights: Orthogonal (prevents gradient explosion in RNNs)
- Bias: Zero, except forget gate bias = 1.0 (helps long-term memory)

---

### Training Module (`src/training/`)

This module handles the training loop, evaluation, and callbacks.

#### `trainer.py`

**Purpose**: Main training loop with gradient accumulation, AMP, and callback support.

##### Classes

| Class | Description |
|-------|-------------|
| `Trainer` | Research-grade training loop |

##### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to train |
| `train_loader` | `DataLoader` | Training data loader |
| `val_loader` | `DataLoader` | Validation data loader |
| `experiment_dir` | `Path` | Directory for checkpoints/logs |
| `config` | `Dict` | Training configuration |
| `device` | `str` | "cpu" or "cuda" |
| `logger` | `ExperimentLogger` | Logging utility |
| `resume_from` | `Optional[Path]` | Checkpoint to resume from |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit()` | `epochs` | `Dict[str, list]` | Run training loop, return history |
| `train_epoch()` | `epoch` | `Dict[str, float]` | Run one training epoch |
| `validate()` | `epoch` | `Dict[str, float]` | Run validation, returns loss/acc/f1 |

**Training Loop**:

```python
def fit(self, epochs):
    for epoch in range(1, epochs + 1):
        # Training phase
        train_metrics = self.train_epoch(epoch)
        
        # Validation phase (includes F1-macro calculation)
        val_metrics = self.validate(epoch)
        
        # LR scheduler step
        if self.lr_scheduler:
            self.lr_scheduler(metrics={"val_loss": val_metrics["loss"]})
        
        # Log metrics
        self.metric_logger.log_metrics({
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            ...
        })
        
        # Model checkpoint (saves best by val_f1_macro)
        if self.checkpoint:
            self.checkpoint(model, metrics, epoch)
        
        # Early stopping check
        if self.early_stopping(metrics):
            break
    
    return self.history
```

**Supported Optimizers**:
- `adam`: Standard Adam
- `adamw`: AdamW with decoupled weight decay
- `sgd`: SGD with momentum

**Class Weights for Imbalanced Data**:

```python
# In config:
"loss": {
    "name": "crossentropy",
    "class_weights": [1.0, 5.0]  # [normal_weight, attack_weight]
}

# Applied as:
weight_tensor = torch.tensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

---

#### `evaluator.py`

**Purpose**: Model evaluation with comprehensive metrics.

##### Classes

| Class | Description |
|-------|-------------|
| `ModelEvaluator` | Computes accuracy, F1, precision, recall, confusion matrix |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `evaluate()` | `calculate_efficiency`, `input_shape_for_profiling` | `Dict[str, Any]` | Full evaluation with optional efficiency metrics |
| `evaluate_cross_dataset()` | `model`, `cross_test_loader` | `Dict[str, Any]` | Evaluate on a different dataset |

**Metrics Computed**:

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `f1_macro` | F1 score averaged across classes (important for imbalance) |
| `f1_weighted` | F1 score weighted by class support |
| `precision_macro` | Precision averaged across classes |
| `recall_macro` | Recall averaged across classes |
| `confusion_matrix` | 2D array of true vs predicted |
| `per_class_metrics` | Detailed metrics per class |

**Efficiency Metrics** (if `calculate_efficiency=True`):
- `parameters`: Number of trainable parameters
- `flops`: Floating point operations
- `inference_time_ms`: Mean inference time per batch

---

#### `callbacks.py`

**Purpose**: Training callbacks for early stopping, checkpointing, and LR scheduling.

##### Classes

| Class | Description |
|-------|-------------|
| `EarlyStopping` | Stop training when metric stops improving |
| `ModelCheckpoint` | Save model checkpoints atomically |
| `LearningRateScheduler` | Wrapper for PyTorch LR schedulers |

##### EarlyStopping

```python
early_stopping = EarlyStopping(
    patience=10,         # Epochs without improvement before stopping
    min_delta=0.001,     # Minimum change to count as improvement
    monitor="val_f1_macro",  # Metric to monitor
    mode="max",          # "max" for accuracy/F1, "min" for loss
    restore_best=True    # Restore best weights after stopping
)

# Usage in training loop:
stop = early_stopping(metrics={"val_f1_macro": 0.85}, epoch=5)
if stop:
    early_stopping.restore_best_checkpoint(model)
    break
```

##### ModelCheckpoint

```python
checkpoint = ModelCheckpoint(
    save_dir=Path("checkpoints"),
    monitor="val_f1_macro",
    mode="max",
    save_best_only=True,
    save_weights_only=True,  # or False to save optimizer state too
    filename="best_model_epoch{epoch}_f1{metric:.4f}.pth"
)

# Usage:
best_path, last_path = checkpoint(model, metrics, epoch, optimizer)
```

**Atomic Saves**: Uses temp file + rename to prevent corruption.

---

### Utils Module (`src/utils/`)

Utility functions for configuration, logging, metrics, visualization, and system monitoring.

#### `config.py`

**Purpose**: Load and validate YAML configuration files.

##### Classes

| Class | Description |
|-------|-------------|
| `ConfigLoader` | Loads, merges, and validates config files |

##### Key Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load_config()` | `config_name` | `Dict` | Load a single config file |
| `load_all_configs()` | None | `Dict` | Load data, preprocess, phase1 configs |
| `merge_configs()` | `base`, `override` | `Dict` | Recursively merge configs |
| `get_experiment_config()` | `mode` | `Dict` | Get config with mode overrides |

---

#### `logging_utils.py`

**Purpose**: Structured logging with tqdm-safe console output.

##### Classes

| Class | Description |
|-------|-------------|
| `ExperimentLogger` | Main logger with rotating file handler + console |
| `MetricLogger` | Tracks numerical metrics, exports to JSON/CSV |
| `TqdmLoggingHandler` | Console handler that doesn't break progress bars |

##### ExperimentLogger Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `info()`, `debug()`, `warning()`, `error()` | `message` | None | Standard logging methods |
| `section()` | `title` | None | Log a section header with separators |
| `progress_bar()` | `iterable`, `desc` | `tqdm` | Create tqdm progress bar |
| `log_metadata()` | `metadata`, `filename`, `format` | `Path` | Save config/metadata to JSON/YAML |

##### MetricLogger Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `log_metrics()` | `metrics`, `step`, `epoch` | None | Log a dictionary of metrics |
| `save_metrics()` | `format="both"` | `Dict[str, Path]` | Save to JSON and/or CSV |
| `reset()` | None | None | Clear all logged metrics |

---

#### `metrics.py`

**Purpose**: Model profiling utilities (FLOPs, inference time, memory).

##### Key Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `count_parameters()` | `model` | `int` | Count trainable parameters |
| `count_flops()` | `model`, `input_shape`, `device` | `Dict` | Calculate FLOPs using thop |
| `measure_inference_time()` | `model`, `input_shape`, `num_runs` | `Dict` | Measure forward pass latency |
| `profile_model()` | `model`, `input_shape`, `device` | `Dict` | Composite profiling (params + FLOPs + timing) |

---

#### `visualization.py`

**Purpose**: Plotting utilities for training curves, confusion matrices, model comparisons.

##### Key Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `plot_training_curves()` | `metrics_history`, `save_path` | `Figure` | Plot loss/accuracy curves |
| `plot_confusion_matrix()` | `cm`, `class_names`, `normalize` | `Figure` | Plot confusion matrix heatmap |
| `plot_model_comparison()` | `results`, `metrics_to_plot` | `Figure` | Bar chart comparing models |
| `plot_parameter_efficiency()` | `results` | `Figure` | Scatter plot: accuracy vs params |

---

#### `system_utils.py`

**Purpose**: System resource monitoring (CPU, memory, disk).

##### Classes

| Class | Description |
|-------|-------------|
| `SystemMonitor` | Monitors CPU/memory/disk for current process |
| `TimingContext` | Context manager for timing code blocks |

##### SystemMonitor Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_cpu_usage()` | `float` | CPU usage percentage |
| `get_memory_usage()` | `Dict` | RSS, VMS, available memory |
| `get_disk_usage()` | `Dict` | Disk total/used/free |
| `check_memory_available()` | `bool` | Check if enough memory available |

##### TimingContext Usage

```python
with TimingContext("Training", logger=exp_logger) as timer:
    trainer.fit()

print(f"Elapsed: {timer.elapsed:.2f}s")
```

---

## Data Flow

### End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA (CSV Files)                           │
│  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv, etc.                  │
│  Columns: Flow Duration, Total Fwd Packets, Label, ...                   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE (pipeline.py)                   │
│                                                                          │
│  1. Load CSVs & concatenate                                              │
│  2. Sort by timestamp (chronological)                                    │
│  3. Hard split (70/15/15): train/val/test                               │
│  4. Feature engineering (per split)                                      │
│  5. Label encoding (train-based mapping)                                 │
│  6. Windowing (create sequences)                                         │
│  7. Scaling (fit on train only)                                          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        PROCESSED DATA (numpy arrays)                      │
│                                                                          │
│  train/X.npy: (n_train, window_len, n_features) = (1.4M, 15, 8)         │
│  train/y.npy: (n_train,) = (1.4M,) binary labels                        │
│  val/X.npy, val/y.npy                                                    │
│  test/X.npy, test/y.npy                                                  │
│  scaler.joblib, label_map.joblib                                         │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADING (loaders.py)                         │
│                                                                          │
│  IDSDataset: Memory-mapped numpy arrays                                  │
│  DataLoader: Batching, shuffling, multi-worker loading                   │
│  Output: {"x": Tensor(batch, 15, 8), "y": Tensor(batch,)}               │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRAINING (trainer.py)                           │
│                                                                          │
│  For each epoch:                                                         │
│    - Forward pass → logits                                               │
│    - Loss (CrossEntropy with class weights)                              │
│    - Backward pass → gradients                                           │
│    - Optimizer step                                                      │
│    - Validation → F1-macro, accuracy                                     │
│    - Callbacks (checkpoint, early stopping, LR scheduler)                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION (evaluator.py)                         │
│                                                                          │
│  Metrics: accuracy, F1-macro, precision, recall, confusion matrix        │
│  Efficiency: parameters, FLOPs, inference time                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              OUTPUTS                                      │
│                                                                          │
│  experiments/Phase1_Baseline_CIC_IDS_2017/                               │
│  ├── all_results.json (aggregated metrics)                               │
│  ├── plots/model_comparison.png                                          │
│  └── runs/{mlp,ds_cnn,lstm}/                                            │
│      ├── checkpoints/best_model.pth                                      │
│      ├── logs/metrics.csv                                                │
│      └── results.json                                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Files

### `configs/data_config.yaml`

```yaml
datasets:
  cic_ids_2017:
    raw_data_dir: "data/raw/cic_ids_2017"
    label_column: "Label"
    time_column: null  # CIC-IDS2017 lacks proper timestamp

sampling:
  quick_mode:
    samples_per_dataset: 200000
  full_mode:
    samples_per_dataset: null  # All data

split_ratios:
  train: 0.70
  val: 0.15
  test: 0.15

loading:
  batch_size: 64
  num_workers: 4
```

### `configs/preprocess_config.yaml`

```yaml
canonical_features:
  critical:
    - "Flow Duration"
    - "Total Fwd Packets"
    - "Total Backward Packets"
  important:
    - "Total Length of Fwd Packets"
    - "Flow Bytes/s"
  optional:
    - "Label"

feature_mappings:
  cic_ids_2017:
    Flow Duration:
      source_column: "Flow Duration"
      transform: "divide_by_1000000"
    Label:
      source_column: "Label"
      transform: "binary_encode"

windowing:
  window_length: 15
  stride: 1
  mode: "sliding"
  padding_strategy: "zero"
  label_strategy: "any_malicious"

transformations:
  normalization:
    method: "standard"

missing_features:
  mode: "flexible"
  imputation_value: 0.0
```

### `configs/phase1_config.yaml`

```yaml
models:
  mlp:
    hidden_sizes: [128, 64, 32]
    dropout_rate: 0.5
  ds_cnn:
    conv_channels: [32, 64, 64]
    dropout_rate: 0.4
  lstm:
    hidden_size: 64
    num_layers: 2

training:
  epochs: 50
  optimizer:
    name: "adamw"
    learning_rate: 0.001
    weight_decay: 0.02

modes:
  quick:
    epochs: 20
    data_samples: 200000
  full:
    epochs: 50
    data_samples: null
```

---

## Usage Guide

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data (first time only)
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick

# 3. Validate preprocessed data
python -m src.data.cli validate --dataset cic_ids_2017

# 4. Run experiment (quick mode)
python main_phase1.py --quick

# 5. Run experiment (full mode - overnight)
python main_phase1.py
```

### Training Specific Models

```bash
# Train only MLP
python main_phase1.py --quick --models mlp

# Train MLP and LSTM
python main_phase1.py --quick --models mlp lstm

# Custom experiment directory
python main_phase1.py --quick --experiment-dir experiments/my_experiment
```

### Accessing Results

```python
import json

# Load experiment results
with open("experiments/Phase1_Baseline_CIC_IDS_2017/all_results.json") as f:
    results = json.load(f)

for model, metrics in results.items():
    print(f"{model}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1_macro']:.2%}")
```

### Loading a Trained Model

```python
import torch
from src.models.mlp import SmallMLP

# Create model with same config
model = SmallMLP(
    input_shape=(15, 8),
    num_classes=2,
    hidden_sizes=(128, 64, 32),
    dropout_rate=0.5
)

# Load weights
checkpoint = torch.load("experiments/.../runs/mlp/checkpoints/best_model.pth")
model.load_state_dict(checkpoint)
model.eval()

# Inference
with torch.no_grad():
    x = torch.randn(1, 15, 8)  # Single sample
    logits = model(x)
    pred = logits.argmax(dim=1)
```

---

## Function Interrelationships

### How Files Call Each Other

```
main_phase1.py
    ├── src.data.loaders.create_data_loaders()
    │       └── IDSDataset (loads X.npy, y.npy)
    │
    ├── src.models.mlp.SmallMLP()
    ├── src.models.ds_cnn.DS_1D_CNN()
    ├── src.models.lstm.LSTMModel()
    │       └── All inherit from BaseIDSModel
    │
    ├── src.training.trainer.Trainer()
    │       ├── src.training.callbacks.EarlyStopping()
    │       ├── src.training.callbacks.ModelCheckpoint()
    │       ├── src.training.callbacks.LearningRateScheduler()
    │       └── src.utils.logging_utils.ExperimentLogger()
    │
    ├── src.training.evaluator.ModelEvaluator()
    │       └── src.utils.metrics.profile_model()
    │
    └── src.utils.visualization.plot_model_comparison()


src.data.cli (preprocessing)
    └── src.preprocessing.pipeline.PreprocessingPipeline()
            ├── src.preprocessing.feature_engineering.FeatureEngineer()
            ├── src.preprocessing.windowing.FlowWindower()
            ├── src.preprocessing.scalers.FeatureScaler()
            └── src.data.validators.DataValidator()
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM (Out of Memory) | Reduce `batch_size`, `num_workers`, or use `mode="memmap"` |
| Poor F1 Score | Check class weights, increase attack class weight (e.g., 5.0) |
| Training Too Slow | Reduce model size, use GPU if available |
| LSTM Missing from Results | Check if training was interrupted (Ctrl+C) |
| Exit Code 134 (SIGABRT) | Memory issue - reduce batch_size to 64, close browsers |

### Recommended Settings for 12GB RAM

```python
config = {
    "data": {
        "batch_size": 64,
        "num_workers": 2,
        "mode": "memmap"
    },
    "model_defaults": {
        "lstm": {
            "hidden_size": 64,
            "bidirectional": False
        }
    }
}
```

---

## License

This project is for research purposes. See LICENSE file for details.

---

*Last updated: November 2025*
