# IDS Model Compression for Edge Deployment

A complete pipeline for training and compressing Intrusion Detection System (IDS) models for deployment on edge devices like Raspberry Pi 4.

## Project Overview

This project implements a **DS-1D-CNN (Depthwise Separable 1D CNN)** model for network intrusion detection, optimized through **Knowledge Distillation**, **Structured Pruning**, and **Quantization** for deployment on resource-constrained devices.

### Key Results

| Metric | Phase 1 Baseline | Phase 2 Target |
|--------|------------------|----------------|
| Accuracy | 95.4% | ≥94.4% |
| F1-Macro | 94.8% | ≥93.8% |
| DDoS Recall | 99.2% | >98% |
| PortScan Recall | 98.7% | >98% |
| Model Size | ~870 KB | ≤50 KB |
| Latency (Pi 4) | N/A | p50 ≤10ms |

## Project Structure

```
ids-compression/
├── train_dscnn_v4.py          # Phase 1: Train baseline DS-CNN v4
├── phase2/                     # Phase 2: Model compression pipeline
│   ├── config.py              # Configuration & targets
│   ├── utils.py               # Utilities & metrics
│   ├── models.py              # Student/Teacher architectures
│   ├── train/                 # Training scripts
│   │   ├── train_baseline.py  # Stage 0: Baseline training
│   │   ├── train_kd.py        # Stage 1: Knowledge Distillation
│   │   └── train_teacher.py   # Teacher model training
│   ├── prune/                 # Pruning scripts
│   │   ├── prune_model.py     # Stage 2: Structured pruning
│   │   └── fine_tune_kd.py    # Stage 3: KD fine-tuning
│   ├── quant/                 # Quantization
│   │   └── qat_train.py       # Stage 4: QAT training
│   ├── convert/               # Model conversion
│   │   └── convert_to_tflite.py # Stage 5: TFLite export
│   ├── bench/                 # Benchmarking
│   │   └── pi_bench.py        # Pi 4 latency/energy testing
│   └── analysis/              # Results analysis
│       └── analyze_results.py # Statistical analysis
├── data/                      # Dataset directory
│   ├── raw/cic_ids_2017/      # Raw CIC-IDS2017 CSVs
│   └── processed/             # Preprocessed numpy arrays
├── experiments/               # Experiment outputs
├── artifacts/                 # Phase 2 model artifacts
└── src/                       # Phase 1 preprocessing modules
```

## Quick Start

### Prerequisites

- Python 3.11+ (for Phase 2 edge deployment)
- CIC-IDS2017 dataset in `data/raw/cic_ids_2017/`

### Phase 1: Train Baseline Model

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full

# Train DS-CNN v4 (5-fold CV)
python train_dscnn_v4.py --mode cv
```

### Phase 2: Compress Model for Edge

```bash
# Create edge-specific environment (Python 3.11)
python3.11 -m venv .venv_edge
source .venv_edge/bin/activate
pip install -r phase2/requirements_edge.txt

# Run full compression pipeline
cd phase2
bash run_pipeline.sh
```

## Phase 1: Baseline Training

The Phase 1 DS-CNN v4 model features:

- **Multi-scale temporal convolutions** (kernel sizes 1, 3, 5, 7)
- **Squeeze-and-Excitation attention blocks**
- **Focal Loss** for class imbalance
- **5-fold cross-validation** for stable results

**Training modes:**
```bash
# Quick test (1 fold, 50 epochs)
python train_dscnn_v4.py --mode single

# Full CV (5 folds, 150 epochs each)
python train_dscnn_v4.py --mode cv

# Full pipeline with ensemble
python train_dscnn_v4.py --mode full
```

## Phase 2: Compression Pipeline

Six-stage pipeline for edge deployment:

| Stage | Description | Tool |
|-------|-------------|------|
| 0 | Baseline | `train_baseline.py` |
| 1 | Knowledge Distillation | `train_kd.py` |
| 2 | Structured Pruning | `prune_model.py` |
| 3 | KD Fine-tuning | `fine_tune_kd.py` |
| 4 | Quantization-Aware Training | `qat_train.py` |
| 5 | TFLite Conversion | `convert_to_tflite.py` |

### Compression Targets

| Target | Threshold |
|--------|-----------|
| Critical Recall (DDoS/PortScan) | >98% |
| F1-Macro Drop | ≤1% |
| False Alarm Rate | ≤1.5% |
| Latency p50 (Pi 4) | ≤10ms |
| Model Size (TFLite) | ≤50KB |

### Statistical Rigor

- **N=7 seeds**: [0, 7, 21, 42, 101, 202, 303]
- **Reporting**: Mean ± Std, 95% CI
- **Significance testing**: Wilcoxon signed-rank test

## Dataset

**CIC-IDS2017** (Canadian Institute for Cybersecurity)
- 8 CSV files (~844MB total)
- ~2.8M network flow records
- Binary classification: Benign vs Attack

Download from: https://www.unb.ca/cic/datasets/ids-2017.html

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| Storage | 5GB | 10GB |
| CPU | 4 cores | 8 cores |
| GPU | Not required | NVIDIA (optional) |

**Target Device:** Raspberry Pi 4 (4GB RAM)

## Documentation

- [Phase 1 How-To](PHASE1_HOWTO.md) - Baseline training guide
- [Phase 2 Guide](PHASE2_GUIDE.md) - Compression pipeline details
- [Architecture](DOCUMENTATION.md) - Code architecture reference

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite:
```
@misc{ids-compression,
  title={IDS Model Compression for Edge Deployment},
  author={Sakib},
  year={2025},
  publisher={GitHub}
}
```
