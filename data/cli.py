#!/usr/bin/env python3
"""
CLI for data preprocessing using the existing PreprocessingPipeline.

Usage:
    python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick
    python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full
    python -m src.data.cli validate --dataset cic_ids_2017
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import psutil
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.data.validators import DataValidator


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_memory():
    """Get system memory info in MB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                meminfo[parts[0][:-1]] = int(parts[1])
        return {
            'total': meminfo.get('MemTotal', 0) / 1024,
            'available': meminfo.get('MemAvailable', 0) / 1024,
            'used': (meminfo.get('MemTotal', 0) - meminfo.get('MemAvailable', 0)) / 1024
        }
    except:
        return None


def load_config(config_paths: list[str]) -> dict:
    """Load and merge multiple YAML config files."""
    # Load data config
    with open('configs/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Load preprocess config
    with open('configs/preprocess_config.yaml', 'r') as f:
        preprocess_config = yaml.safe_load(f)
    
    # Structure config properly for PreprocessingPipeline
    merged = {
        "data": data_config,  # Wrap data_config under "data" key
        "preprocess": preprocess_config  # Wrap preprocess_config under "preprocess" key
    }
    
    return merged


def preprocess_command(args):
    """Run preprocessing pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configs
    config_files = [
        'configs/data_config.yaml',
        'configs/preprocess_config.yaml'
    ]
    
    logger.info("Loading configuration files...")
    config = load_config(config_files)
    
    # Show system memory before starting
    sys_mem = get_system_memory()
    if sys_mem:
        logger.info(f"System Memory: {sys_mem['total']:.0f} MB total, {sys_mem['available']:.0f} MB available")
    process_mem_start = get_memory_usage()
    logger.info(f"Process Memory (start): {process_mem_start:.1f} MB")
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        config=config,
        logger=logger,
        seed=args.seed
    )
    
    # Process dataset
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/processed/{args.dataset}")
    
    logger.info(f"Starting preprocessing for dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        report = pipeline.process_dataset(
            dataset_name=args.dataset,
            output_dir=output_dir,
            mode=args.mode,
            overwrite=args.overwrite
        )
        
        # Show memory after completion
        process_mem_end = get_memory_usage()
        sys_mem_end = get_system_memory()
        logger.info(f"Process Memory (end): {process_mem_end:.1f} MB (peak delta: {process_mem_end - process_mem_start:.1f} MB)")
        if sys_mem_end:
            logger.info(f"System Memory (end): {sys_mem_end['available']:.0f} MB available")
        
        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Train windows: {report['n_train_windows']}")
        logger.info(f"Val windows: {report['n_val_windows']}")
        logger.info(f"Test windows: {report['n_test_windows']}")
        logger.info(f"Label map: {report['label_map']}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_command(args):
    """Validate preprocessed data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    data_dir = Path(f"data/processed/{args.dataset}")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please run preprocessing first:")
        logger.error(f"  python -m src.data.cli preprocess --dataset {args.dataset} --mode quick")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        ('train/X.npy', 'train/y.npy'),
        ('val/X.npy', 'val/y.npy'),
        ('test/X.npy', 'test/y.npy'),
        ('scaler.joblib',),
        ('label_map.joblib',),
        ('preprocessing_report.json',)
    ]
    
    import numpy as np
    
    logger.info("=" * 70)
    logger.info(f"VALIDATING DATA: {args.dataset}")
    logger.info("=" * 70)
    
    all_good = True
    for files in required_files:
        for file in files:
            file_path = data_dir / file
            if file_path.exists():
                if file.endswith('.npy'):
                    arr = np.load(file_path, mmap_mode='r')
                    logger.info(f"✓ {file}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    logger.info(f"✓ {file}: exists")
            else:
                logger.error(f"✗ {file}: MISSING")
                all_good = False
    
    # Load and display preprocessing report
    report_path = data_dir / 'preprocessing_report.json'
    if report_path.exists():
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        logger.info("\nPreprocessing Report:")
        logger.info(f"  Dataset: {report.get('dataset')}")
        logger.info(f"  Mode: {report.get('mode')}")
        logger.info(f"  Train windows: {report.get('n_train_windows')}")
        logger.info(f"  Val windows: {report.get('n_val_windows')}")
        logger.info(f"  Test windows: {report.get('n_test_windows')}")
        logger.info(f"  Scaler: {report.get('scaler_method')}")
        logger.info(f"  Label map: {report.get('label_map')}")
    
    logger.info("=" * 70)
    if all_good:
        logger.info("✓ ALL CHECKS PASSED - Data is ready for training")
    else:
        logger.error("✗ VALIDATION FAILED - Some files are missing")
        sys.exit(1)
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Data preprocessing CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw data')
    preprocess_parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., cic_ids_2017, ton_iot)'
    )
    preprocess_parser.add_argument(
        '--mode',
        choices=['quick', 'medium', 'full'],
        default='quick',
        help='Processing mode (quick=200K samples, full=all data)'
    )
    preprocess_parser.add_argument(
        '--output-dir',
        help='Custom output directory (default: data/processed/<dataset>)'
    )
    preprocess_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    preprocess_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing preprocessed data'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate preprocessed data')
    validate_parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name to validate'
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess_command(args)
    elif args.command == 'validate':
        validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
