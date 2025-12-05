# Compressing DS_1D_CNN for Edge Network Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

> **Systematic Analysis of Compression-Attack Degradation for Multi-Dataset Network Intrusion Detection on Resource-Constrained Devices**

This repository contains the complete implementation, data pipelines, and reproducibility package for our research on compressing deep learning-based network intrusion detection systems (IDS) for deployment on Raspberry Pi 4.

📄 **Paper**: [Link to paper] (upon publication)  
🎯 **Contribution**: First comprehensive per-attack-family compression impact study with cross-dataset generalization analysis

---

## 🎯 Key Contributions

1. **Per-Attack-Family Compression Analysis**: Systematic study of how multi-stage compression (KD → Pruning → QAT) affects 20 different attack types
2. **Cross-Dataset Generalization Study**: Quantified impact of dataset fusion on model robustness under compression
3. **Reproducible Benchmark Suite**: Complete pipeline with provenance tracking for IDS compression research
4. **Production-Ready Models**: INT8 models (62KB/53KB) achieving 96.5%/94.8% accuracy with <8ms latency on Raspberry Pi 4

---

## 📊 Main Results

| Model | Size | Latency (Pi4) | Binary Acc. | Macro-F1 | DDoS Recall | Infiltration Recall |
|-------|------|---------------|-------------|----------|-------------|---------------------|
| Teacher | 2.4 MB | 45 ms | 98.5% | 96.2% | 99.1% | 89.1% |
| Student (KD) | 350 KB | 12 ms | 97.2% | 94.5% | 98.5% | 85.2% |
| Pruned (30%) | 245 KB | 10 ms | 96.8% | 93.8% | 98.0% | 80.3% |
| **Final (QAT)** | **62 KB** | **7.5 ms** | **96.5%** | **93.5%** | **97.8%** | **78.1%** |

**Key Finding**: High-volume attacks (DDoS, DoS, PortScan) maintain >95% recall even at 40% pruning, while sophisticated attacks (Infiltration, Web-Attack) degrade significantly (5-11%) under compression.

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
```bash
# Python 3.9+ required
python --version

# Clone repository
git clone https://github.com/yourusername/ids-compression-benchmark.git
cd ids-compression-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo with Sample Data
```bash
# Download sample dataset (1000 flows, ~5 MB)
python scripts/download_sample_data.py

# Run inference with pretrained model
python demo/run_inference.py \
  --model pretrained/final_compressed_30.tflite \
  --data data/samples/sample_1000.parquet \
  --output results/demo_predictions.csv

# Expected output: ~96% accuracy, <10ms average latency
```

### Evaluate Pretrained Models
```bash
# Evaluate on sample data
python scripts/evaluate_pretrained.py \
  --model pretrained/final_compressed_30.tflite \
  --test-data data/samples/test_sample.parquet \
  --output results/evaluation_report.html

# Generates: Confusion matrix, ROC curves, per-attack metrics
```

---

## 📦 Full Reproduction (Step-by-Step)

### Phase 1: Data Preparation & Baseline Models (Weeks 1-6)

#### Step 1: Download Datasets
```bash
# NF-UQ-NIDS-v2 + NF-CSE-CIC-IDS2018-v2
# Total size: ~50 GB compressed, ~200 GB uncompressed
python scripts/01_download_datasets.py \
  --datasets nf-uq-nids-v2 nf-cse-cic-ids2018-v2 \
  --output data/raw/

# Expected time: 2-4 hours (depends on connection)
# Validates checksums automatically
```

#### Step 2: Data Inventory & Quality Check
```bash
python scripts/01_data_inventory.py \
  --input data/raw/ \
  --output reports/phase1/data_inventory.json

# Outputs:
# - Per-class distribution statistics
# - Missing value analysis
# - Feature distribution plots
# - Class decision matrix
```

#### Step 3: Feature Alignment & Canonical Schema
```bash
python scripts/02_canonical_schema.py \
  --input data/raw/ \
  --output data/canonical/merged_canonical.parquet \
  --schema configs/canonical_schema.yaml

# Creates unified 43-feature NetFlow schema
# Adds provenance metadata (dataset_origin column)
```

#### Step 4: Data Cleaning & Splitting
```bash
python scripts/04_data_cleaning.py \
  --input data/canonical/merged_canonical.parquet \
  --output data/cleaned/merged_cleaned.parquet \
  --config configs/cleaning_config.yaml

python scripts/05_dataset_split.py \
  --input data/cleaned/merged_cleaned.parquet \
  --output data/splits/ \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --cross-dataset-holdout nf-cse-cic-ids2018-v2
```

#### Step 5: Train Baseline Models
```bash
# Train all three baselines (MLP, LSTM, DS-CNN)
python main_phase1.py \
  --mode full \
  --config configs/phase1_config.yaml \
  --output experiments/phase1_baselines/

# Expected time: 6-8 hours (GPU)
# Outputs: Trained models + evaluation metrics + plots
```

#### Step 6: Generate Phase 1 Decision Report
```bash
python scripts/06_generate_phase1_report.py \
  --experiments experiments/phase1_baselines/ \
  --output reports/phase1_decision_report.md
```

### Phase 2: Compression Pipeline (Weeks 7-14)

#### Step 7: Train Teacher Model
```bash
python training/train_teacher.py \
  --data data/splits/train.parquet \
  --config configs/teacher_config.yaml \
  --output experiments/phase2/teacher/ \
  --epochs 50

# Expected time: 20-30 hours (GPU)
```

#### Step 8: Knowledge Distillation
```bash
# Grid search over temperature and alpha
python training/kd_grid_search.py \
  --teacher experiments/phase2/teacher/best_model.h5 \
  --data data/splits/train.parquet \
  --temperatures 2.0 4.0 6.0 \
  --alphas 0.5 0.7 0.9 \
  --output experiments/phase2/kd_grid_search/

# Train final student with best config
python training/train_student_kd.py \
  --teacher experiments/phase2/teacher/best_model.h5 \
  --config experiments/phase2/kd_grid_search/best_config.yaml \
  --output experiments/phase2/student_kd/ \
  --epochs 50
```

#### Step 9: Structured Pruning
```bash
# Iterative pruning with per-stage evaluation
python compression/structured_pruning.py \
  --model experiments/phase2/student_kd/best_model.h5 \
  --pruning-schedule configs/pruning_schedule.yaml \
  --output experiments/phase2/pruning/ \
  --teacher experiments/phase2/teacher/best_model.h5  # For KD during fine-tuning

# Generates models at 10%, 20%, 30%, 40%, 50% pruning ratios
```

#### Step 10: Quantization-Aware Training
```bash
# QAT on best pruned models
python compression/qat.py \
  --model experiments/phase2/pruning/pruned_30_finetuned.h5 \
  --calibration-data data/splits/calibration_1000.npy \
  --output experiments/phase2/qat/pruned_30_qat/ \
  --epochs 20

python compression/qat.py \
  --model experiments/phase2/pruning/pruned_40_finetuned.h5 \
  --calibration-data data/splits/calibration_1000.npy \
  --output experiments/phase2/qat/pruned_40_qat/ \
  --epochs 20
```

#### Step 11: Convert to TFLite
```bash
# Convert to INT8 TFLite models
python deployment/convert_to_tflite.py \
  --model experiments/phase2/qat/pruned_30_qat/best_model.h5 \
  --output pretrained/final_compressed_30.tflite \
  --quantization int8 \
  --representative-dataset data/splits/calibration_1000.npy

python deployment/convert_to_tflite.py \
  --model experiments/phase2/qat/pruned_40_qat/best_model.h5 \
  --output pretrained/final_compressed_40.tflite \
  --quantization int8 \
  --representative-dataset data/splits/calibration_1000.npy
```

#### Step 12: Generate Phase 2 Analysis
```bash
# Complete compression pipeline evaluation
python scripts/10_full_pipeline_eval.py \
  --experiments experiments/phase2/ \
  --test-data data/splits/test_same.parquet \
  --cross-test-data data/splits/test_cross.parquet \
  --output reports/phase2_compression_analysis.md
```

### Phase 3: Deployment & Evaluation (Weeks 15-20)

#### Step 13: Raspberry Pi 4 Setup
```bash
# On Raspberry Pi 4
# Copy scripts and models
scp -r deployment/ pi@raspberrypi:/home/pi/ids/
scp pretrained/*.tflite pi@raspberrypi:/home/pi/ids/pretrained/

# SSH into Pi
ssh pi@raspberrypi

# Install dependencies on Pi
cd /home/pi/ids/
bash deployment/pi4_setup.sh
```

#### Step 14: Hardware Benchmarking
```bash
# On Raspberry Pi 4
python benchmarking/pi4_benchmark.py \
  --model pretrained/final_compressed_30.tflite \
  --scenarios single_flow_latency throughput_optimized sustained_load \
  --output results/pi4_benchmarks_30.json

python benchmarking/pi4_benchmark.py \
  --model pretrained/final_compressed_40.tflite \
  --scenarios single_flow_latency throughput_optimized sustained_load \
  --output results/pi4_benchmarks_40.json
```

#### Step 15: Ablation Studies
```bash
# On workstation
python experiments/ablation_studies.py \
  --ablations dataset_fusion compression_stages augmentation kd_config pruning_strategy \
  --output reports/phase3/ablation_studies.md
```

---

## 📁 Repository Structure

```
ids-compression-benchmark/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment (optional)
├── LICENSE                      # MIT License
│
├── configs/                     # Configuration files (YAML)
│   ├── canonical_schema.yaml   # 43-feature NetFlow schema
│   ├── cleaning_config.yaml    # Data cleaning parameters
│   ├── phase1_config.yaml      # Baseline training config
│   ├── teacher_config.yaml     # Teacher model config
│   ├── kd_config.yaml          # KD hyperparameters
│   ├── pruning_schedule.yaml   # Pruning ratios and fine-tuning
│   └── qat_config.yaml         # QAT parameters
│
├── data/                        # Data directory (gitignored, except samples/)
│   ├── raw/                    # Downloaded datasets
│   ├── canonical/              # Aligned canonical schema
│   ├── cleaned/                # Cleaned and preprocessed
│   ├── splits/                 # Train/val/test splits
│   └── samples/                # Small samples for testing (included)
│       ├── sample_1000.parquet
│       └── test_sample.parquet
│
├── src/                         # Source code
│   ├── data/
│   │   ├── downloaders.py      # Dataset download with progress
│   │   ├── loaders.py          # PyTorch/TF data loaders
│   │   └── validators.py      # Data quality checks
│   │
│   ├── preprocessing/
│   │   ├── feature_engineering.py  # Canonical schema implementation
│   │   ├── windowing.py        # Flow windowing
│   │   ├── scalers.py          # Normalization
│   │   └── pipeline.py         # End-to-end preprocessing
│   │
│   ├── models/
│   │   ├── base_model.py       # Base class for all models
│   │   ├── teacher.py          # High-capacity teacher model
│   │   ├── ds_cnn.py           # DS-1D-CNN (primary architecture)
│   │   ├── mlp.py              # MLP baseline
│   │   └── lstm.py             # LSTM baseline
│   │
│   ├── training/
│   │   ├── trainer.py          # Unified training loop
│   │   ├── kd_trainer.py       # Knowledge distillation trainer
│   │   ├── evaluator.py        # Evaluation metrics
│   │   └── callbacks.py        # Training callbacks
│   │
│   ├── compression/
│   │   ├── knowledge_distillation.py  # KD loss and training
│   │   ├── structured_pruning.py      # Filter pruning
│   │   └── qat.py              # Quantization-aware training
│   │
│   └── utils/
│       ├── config.py           # Config loading
│       ├── logging_utils.py    # Logging setup
│       ├── metrics.py          # Custom metrics (FLOPs, etc.)
│       ├── visualization.py    # Plotting functions
│       └── system_utils.py     # System monitoring
│
├── scripts/                     # Executable scripts
│   ├── 01_download_datasets.py
│   ├── 01_data_inventory.py
│   ├── 02_canonical_schema.py
│   ├── 03_feature_validation.py
│   ├── 04_data_cleaning.py
│   ├── 05_dataset_split.py
│   ├── 06_evaluate_baselines.py
│   ├── 06_generate_phase1_report.py
│   ├── 07_kd_impact_analysis.py
│   ├── 08_pruning_attack_tracking.py
│   ├── 09_qat_validation.py
│   ├── 10_full_pipeline_eval.py
│   └── download_sample_data.py
│
├── deployment/                  # Deployment scripts
│   ├── convert_to_tflite.py    # Model conversion
│   ├── pi4_setup.sh            # Pi4 environment setup
│   └── optimize_for_edge.py    # Edge optimization
│
├── benchmarking/                # Benchmarking scripts
│   ├── pi4_benchmark.py        # Pi4 hardware benchmarks
│   ├── traffic_simulation.py   # Realistic traffic simulation
│   └── evaluate_all.py         # Comprehensive evaluation
│
├── experiments/                 # Experiment results (gitignored)
│   ├── phase1_baselines/
│   ├── phase2_compression/
│   └── phase3_deployment/
│
├── reports/                     # Generated reports
│   ├── phase1_decision_report.md
│   ├── phase2_compression_analysis.md
│   └── phase3/
│       ├── ablation_studies.md
│       ├── failure_analysis.md
│       └── deployment_benchmarks.md
│
├── plots/                       # All figures (for paper)
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
│
├── pretrained/                  # Released models
│   ├── teacher_model/
│   ├── final_compressed_30.tflite
│   ├── final_compressed_40.tflite
│   └── model_cards/
│       ├── final_compressed_30.md
│       └── final_compressed_40.md
│
├── tests/                       # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_compression.py
│   └── test_deployment.py
│
├── demo/                        # Demo scripts
│   ├── run_inference.py        # Simple inference demo
│   └── interactive_demo.ipynb  # Jupyter notebook demo
│
├── paper/                       # Paper materials (LaTeX)
│   ├── main.tex
│   ├── figures/
│   ├── tables/
│   └── compile.sh
│
└── docs/                        # Documentation
    ├── INSTALL.md              # Installation guide
    ├── DATASETS.md             # Dataset documentation
    ├── MODELS.md               # Model architecture details
    ├── COMPRESSION.md          # Compression pipeline guide
    └── DEPLOYMENT.md           # Deployment guide
```

---

## 🧹 File Management Utilities

### Delete Multiple Files from Terminal

This repository includes a powerful file deletion utility for safely managing files from the command line:

```bash
# Interactive mode - select files to delete
python scripts/delete_files.py --interactive

# Delete files by pattern (with dry-run preview)
python scripts/delete_files.py --pattern "*.log" --dry-run
python scripts/delete_files.py --pattern "*.log"

# Delete specific files
python scripts/delete_files.py --files old_file1.txt old_file2.log

# Delete with exclusions
python scripts/delete_files.py --pattern "*.md" --exclude README.md
```

**Safety Features:**
- 🔒 Protected files (Python scripts, git files, requirements.txt)
- 👁️ Dry-run mode to preview before deletion
- ✅ Confirmation prompts
- 📊 Size display and deletion summary

See [File Deletion Utility Documentation](docs/FILE_DELETION_UTILITY.md) for complete guide.

---

## 🔧 Hardware Requirements

### Development/Training
- **GPU**: NVIDIA RTX 3080+ (12GB+ VRAM) or equivalent
- **RAM**: 32GB+ (for large dataset loading)
- **Storage**: 500GB+ SSD (datasets ~200GB + experiments ~100GB)
- **CPU**: 8+ cores (for data preprocessing)

### Deployment/Testing
- **Device**: Raspberry Pi 4 Model B (4GB or 8GB RAM)
- **Storage**: 32GB+ MicroSD (Class 10)
- **Power**: Official 5V/3A power supply
- **Cooling**: Heatsink or fan recommended for sustained load

### Cloud Alternatives
- **Google Colab Pro**: $10/month (30-50 GPU hours, sufficient for Phase 1)
- **AWS EC2 p3.2xlarge**: ~$3/hour (V100 GPU)
- **Kaggle Notebooks**: 30 hours/week free GPU (limited but usable)

---

## 📊 Datasets

### Primary Datasets (Used in Paper)

**1. NF-UQ-NIDS-v2**
- **Size**: ~76 million flows
- **Features**: 43 NetFlow v2 features
- **Attack Types**: 20 categories (DDoS, DoS, PortScan, etc.)
- **Source**: [University of Queensland](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **License**: Research use
- **Download**: ~30 GB compressed

**2. NF-CSE-CIC-IDS2018-v2**
- **Size**: ~19 million flows
- **Features**: 43 NetFlow v2 features (aligned with NF-UQ)
- **Attack Types**: 15 categories (overlaps with NF-UQ)
- **Source**: [Same repository](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **License**: Research use
- **Download**: ~10 GB compressed

### Feature Schema (43 NetFlow Features)
See `configs/canonical_schema.yaml` for complete list. Key features:
- Flow duration, packet counts, byte counts (fwd/bwd)
- Inter-arrival times (min/max/mean/std)
- Packet length statistics (fwd/bwd)
- Flow flags (FIN, SYN, RST, PSH, ACK, URG)
- Window sizes, header lengths
- And more...

### Attack Categories (20 Total)
**High-volume** (>1M samples): DDoS, DoS-Hulk, DoS-SlowHTTPTest, PortScan  
**Medium** (100K-1M): SSH-Patator, FTP-Patator, Bot, Brute-Force  
**Low** (10K-100K): Infiltration, Web-Attack-*, Heartbleed  
**Rare** (<10K): Some specialized attacks (documented as inconclusive)

---

## 🎓 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{yourname2025compressing,
  title={Systematic Analysis of Compression-Attack Degradation for Multi-Dataset Network Intrusion Detection on Resource-Constrained Devices},
  author={Your Name and Collaborators},
  booktitle={Proceedings of [Conference Name]},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Dataset Licenses**: NF-UQ-NIDS-v2 and NF-CSE-CIC-IDS2018-v2 are provided for research use. Please refer to the original dataset authors for licensing terms.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution**:
- Additional datasets (ToN-IoT, UNSW-NB15, etc.)
- New compression techniques (NAS, lottery ticket, etc.)
- Different architectures (Transformers, Graph NNs)
- Additional edge devices (Jetson Nano, Coral TPU)
- Deployment optimizations

---

## 🐛 Issues & Support

- **Bug reports**: [GitHub Issues](https://github.com/yourusername/ids-compression-benchmark/issues)
- **Questions**: [GitHub Discussions](https://github.com/yourusername/ids-compression-benchmark/discussions)
- **Email**: your.email@university.edu

---

## 📚 Related Work

This research builds upon:

**Model Compression**:
- Hinton et al. (2015): Knowledge Distillation
- Han et al. (2016): Deep Compression
- Howard et al. (2017): MobileNets

**Network Intrusion Detection**:
- Sharafaldin et al. (2018): CIC-IDS2018
- Sarhan et al. (2021): NetFlow Datasets for ML
- Recent 2024-2025 work on edge IDS

**Our Contribution**: First systematic per-attack-family compression analysis with cross-dataset generalization study.

---

## 🙏 Acknowledgments

- University of Queensland for providing NF-UQ-NIDS-v2 dataset
- Canadian Institute for Cybersecurity for CIC-IDS2018
- Reviewers and community feedback

---

## 📈 Project Status

- ✅ Phase 1: Data preparation & baselines complete
- ✅ Phase 2: Compression pipeline complete
- ✅ Phase 3: Deployment & evaluation complete
- 📝 Paper: Under review at [Conference Name]
- 🚀 Pretrained models: Released
- 📦 Full code: Released

**Last Updated**: December 2025

---

## 🔗 Links

- **Paper**: [arXiv link] (upon publication)
- **Demo**: [Live demo link] (if applicable)
- **Dataset**: [Download instructions](docs/DATASETS.md)
- **Documentation**: [Full docs](docs/)
- **Tutorial**: [Step-by-step guide](docs/TUTORIAL.md)

---

**Made with ❤️ for the security research community**