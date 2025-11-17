#!/usr/bin/env python3
"""
Phase 1 Setup Script
Creates the complete project structure and generates all necessary files.
Run this script first to initialize your Phase 1 research environment.

Usage:
    python setup_phase1.py --project-dir ~/ids_compression_research
"""

import os
import argparse
from pathlib import Path
import sys


def create_project_structure(base_dir: Path):
    """
    Creates the complete directory structure for Phase 1 research.
    
    The structure separates concerns clearly:
    - src: all source code organized by functionality
    - data: raw and processed datasets with clear separation
    - experiments: timestamped experiment runs with full reproducibility
    - configs: versioned configuration files
    - docs: documentation and research notes
    """
    
    directories = [
        # Source code organization
        "src/data",              # Data downloading and loading
        "src/preprocessing",     # Feature engineering and windowing
        "src/models",           # Model architectures
        "src/training",         # Training and evaluation logic
        "src/utils",            # Utilities and helpers
        
        # Data directories with clear separation
        "data/raw/cic_ids_2017",
        "data/raw/ton_iot",
        "data/processed/cic_ids_2017",
        "data/processed/ton_iot",
        "data/samples",         # For quick prototyping
        
        # Experiment tracking
        "experiments",          # Each run gets timestamped subdirectory
        
        # Configuration files
        "configs",
        
        # Documentation
        "docs",
        
        # Logs and outputs
        "logs",
        "outputs/plots",
        "outputs/reports",
        "outputs/models",
    ]
    
    print(f"Creating project structure at: {base_dir}")
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    # Create __init__.py files to make src a package
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/preprocessing/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/utils/__init__.py",
    ]
    
    for init_file in init_files:
        (base_dir / init_file).touch()
    
    print("\n✓ Project structure created successfully!")


def create_requirements_txt(base_dir: Path):
    """
    Generates requirements.txt with all necessary dependencies.
    Versions are pinned for reproducibility but not overly restrictive.
    """
    
    requirements = """# Core deep learning and data processing
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Data handling and performance
pyarrow>=12.0.0
tables>=3.8.0

# Metrics and evaluation
thop>=0.1.1  # For FLOPs counting
seaborn>=0.12.0
matplotlib>=3.7.0

# Configuration and logging
pyyaml>=6.0
tqdm>=4.65.0

# Dataset downloading
requests>=2.31.0
kaggle>=1.5.16  # For Kaggle datasets

# Utilities
psutil>=5.9.0  # For system monitoring
"""
    
    req_file = base_dir / "requirements.txt"
    req_file.write_text(requirements)
    print(f"\n✓ Created requirements.txt")
    print("  Install with: pip install -r requirements.txt")


def create_gitignore(base_dir: Path):
    """Creates a comprehensive .gitignore for the project."""
    
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Data files (too large for git)
data/raw/
data/processed/
data/samples/
*.csv
*.parquet
*.npy
*.npz

# Model checkpoints
outputs/models/*.pth
outputs/models/*.pt

# Experiments (track configs but not full runs)
experiments/*/models/
experiments/*/data/
experiments/*/*.pth

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Keep directory structure
!.gitkeep
"""
    
    gitignore_file = base_dir / ".gitignore"
    gitignore_file.write_text(gitignore)
    print(f"✓ Created .gitignore")


def create_readme(base_dir: Path):
    """Creates the main README with project overview and usage instructions."""
    
    readme = """# Lightweight IDS on Edge Devices - Phase 1

## Performance and Efficiency Trade-offs in Model Compression

### Phase 1: Baseline Architecture Selection

This repository contains the complete implementation for Phase 1 of the research project,
focusing on validating DS-1D-CNN as the optimal student architecture before investing
in compression techniques.

## Project Structure

```
.
├── src/                    # Source code
│   ├── data/              # Dataset downloading and loading
│   ├── preprocessing/     # Feature engineering and windowing
│   ├── models/            # Model architectures (MLP, DS-1D-CNN, LSTM)
│   ├── training/          # Training and evaluation logic
│   └── utils/             # Utilities and helpers
├── data/                  # Data storage (gitignored)
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Preprocessed features
│   └── samples/          # Sampled datasets for prototyping
├── experiments/          # Experiment runs with full reproducibility
├── configs/              # Configuration files
├── outputs/              # Generated plots, reports, models
└── docs/                 # Documentation
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Arch Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Preprocess Datasets

```bash
# Download CIC-IDS2017 and TON-IoT
python -m src.data.download_datasets --config configs/data_config.yaml

# Preprocess to canonical feature schema
python -m src.preprocessing.preprocess --config configs/preprocess_config.yaml --mode sample
```

### 3. Run Phase 1 Baseline Experiments

```bash
# Train all three baseline models and evaluate
python main_phase1.py --config configs/phase1_config.yaml --mode quick

# For full dataset runs (overnight)
python main_phase1.py --config configs/phase1_config.yaml --mode full
```

### 4. View Results

Results are saved in timestamped experiment directories under `experiments/`.
Each run includes:
- Training logs and metrics
- Model checkpoints
- Evaluation results on both datasets
- Cross-dataset validation results
- Decision report with architecture selection recommendation

## Configuration

All experiments are configured via YAML files in `configs/`:
- `data_config.yaml`: Dataset paths and download settings
- `preprocess_config.yaml`: Feature engineering and windowing parameters
- `phase1_config.yaml`: Model architectures and training hyperparameters

## Research Objectives

Phase 1 validates that DS-1D-CNN is the optimal student architecture by:

1. Training three models (MLP, DS-1D-CNN, LSTM) from scratch on CIC-IDS2017
2. Evaluating on held-out test set using:
   - Accuracy, F1-score (macro), per-class recall
   - Parameter count, FLOPs
   - CPU inference time (proxy for edge feasibility)
   - Training time
3. Cross-dataset validation: Train on CIC-IDS2017 → evaluate on TON-IoT
4. Optional: Train on TON-IoT → evaluate on CIC-IDS2017

### Decision Criteria

Select DS-1D-CNN if:
- ✓ Accuracy within 2% of best model
- ✓ Params < 150K (leaves room for compression)
- ✓ Inference time competitive (within 2× of fastest)

## Hardware Requirements

Minimum:
- CPU: Dual-core processor (tested on Intel i3-7100U)
- RAM: 8GB (11.8GB recommended)
- Storage: 50GB free space for datasets

Recommended:
- GPU: Any CUDA-compatible GPU (optional, speeds up training)
- RAM: 16GB for full dataset processing

## Citation

If you use this code in your research, please cite:

```
[Your citation will go here after publication]
```

## License

[Specify your license]
"""
    
    readme_file = base_dir / "README.md"
    readme_file.write_text(readme)
    print(f"✓ Created README.md")


def main():
    parser = argparse.ArgumentParser(description="Setup Phase 1 project structure")
    parser.add_argument(
        "--project-dir",
        type=str,
        default="./ids_phase1_research",
        help="Base directory for the project (default: ./ids_phase1_research)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.project_dir).expanduser().resolve()
    
    print("=" * 70)
    print("Phase 1: Baseline Architecture Selection - Project Setup")
    print("=" * 70)
    
    # Create project structure
    create_project_structure(base_dir)
    
    # Create configuration files
    create_requirements_txt(base_dir)
    create_gitignore(base_dir)
    create_readme(base_dir)
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print(f"\nProject created at: {base_dir}")
    print("\nNext steps:")
    print("  1. cd", base_dir)
    print("  2. python -m venv venv && source venv/bin/activate")
    print("  3. pip install -r requirements.txt")
    print("  4. Review configs/*.yaml and adjust as needed")
    print("  5. Run: python main_phase1.py --help")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()