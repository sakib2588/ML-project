# main_phase1.py
"""
Phase 1 experiment runner (research-grade).

Produces:
 - Per-model run directories with logs, checkpoints, metrics
 - Aggregated plots and minimal decision report
 - JSON artifacts for reproducibility
"""

from __future__ import annotations

import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import sys
import numpy as np
import torch

# Local imports - adjust if your package root differs
from src.data.loaders import create_data_loaders
from src.utils.logging_utils import get_loggers, ExperimentLogger, MetricLogger
from src.training.trainer import Trainer
from src.utils.logging_utils import ExperimentLogger
from src.training.evaluator import ModelEvaluator
from src.utils.visualization import (
    plot_model_comparison,
    plot_parameter_efficiency,
    plot_cross_dataset_performance,
)
from pathlib import Path
from typing import Optional




# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_all(seed: int = 42) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (may slow some ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Save object as JSON; creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _model_factory(model_name: str, input_shape: Tuple[int, int], num_classes: int, **kwargs):
    """
    Simple factory for models. Adjust if your modules have different names.
    Returns an instance of the requested model.
    """
    mn = model_name.lower()
    if mn in ("mlp", "mlpp"):
        from src.models.mlpp import SmallMLP  # type: ignore
        return SmallMLP(input_shape=input_shape, num_classes=num_classes, **kwargs)
    if mn in ("ds_cnn", "dscnn", "ds-1d-cnn"):
        from src.models.ds_cnn import DS_1D_CNN  # type: ignore
        return DS_1D_CNN(input_shape=input_shape, num_classes=num_classes, **kwargs)
    if mn in ("lstm",):
        from src.models.lstm import LSTMModel  # type: ignore
        return LSTMModel(input_shape=input_shape, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {model_name}")


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------
def run_full_experiment(
    config: Dict[str, Any],
    models_to_run: Optional[List[str]] = None,
    experiment_root: Optional[Union[str, Path]] = None,
    seed: int = 42,
    resume: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Run a full Phase 1 experiment and return the experiment directory (Path).

    Args:
        config: dictionary containing data / training / callbacks settings.
        models_to_run: optional list of model identifiers to run (overrides config).
        experiment_root: base path for the experiment. If None, uses ./experiments/<name_timestamp>.
        seed: RNG seed for reproducibility.
        resume: optional checkpoint path to resume from (applies per-model if desired).
    """
    
    # ----------------------------
    # Normalize experiment_root to Path (robust & type-safe)
    # ----------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment_name", f"phase1_{ts}")

    # Accept None | str | Path for experiment_root; convert to Path immediately
    if experiment_root is None:
        experiment_root_path = Path("experiments") / exp_name
    else:
        # cast PathLike/str/Path to Path safely
        experiment_root_path = Path(str(experiment_root))

    # Narrow type for Pylance / static checkers
    assert isinstance(experiment_root_path, Path)
    # Ensure directory exists
    experiment_root_path.mkdir(parents=True, exist_ok=True)

    # Use the normalized Path for the rest of the function
    experiment_root = experiment_root_path

    # Save experiment config
    save_json(config, experiment_root / "config.json")

    # Setup global loggers
    exp_logger, metric_logger_global = get_loggers(experiment_root / "logs")
    # get_loggers returns (ExperimentLogger, MetricLogger)
    # Use experiment_logger for high-level messages, metric_logger_global for aggregated metrics
    experiment_logger: ExperimentLogger = exp_logger
    experiment_logger.section(f"Starting Phase1 experiment: {exp_name}")
    experiment_logger.info(f"Experiment directory: {experiment_root}")
    experiment_logger.info(f"Seed: {seed}")

    # Seed everything
    seed_all(seed)

    # --- Create data loaders ---
    data_cfg = config.get("data", {})
    required_paths = ("train_path", "val_path", "test_path")
    for rp in required_paths:
        if rp not in data_cfg:
            raise KeyError(f"data config missing required key: {rp}")

    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]
    test_path = data_cfg["test_path"]

    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    mode = data_cfg.get("mode", "memmap")
    return_dict = bool(data_cfg.get("return_dict", False))
    collate_fn = data_cfg.get("collate_fn", None)

    experiment_logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path, val_path, test_path,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=mode,
        transform=None,
        target_transform=None,
        return_dict=return_dict,
        collate_fn=collate_fn,
        distributed=False,
        seed=seed,
    )

    # Infer input_shape (window_len, n_features) from a single sample in train loader
    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty; ensure data is available.")

    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 1:
        sample_x = sample_batch[0]
    elif isinstance(sample_batch, dict) and "x" in sample_batch:
        sample_x = sample_batch["x"]
    else:
        raise RuntimeError("Unable to infer input tensor from train_loader batch; expected (x,y) or dict{'x','y'}.")

    if hasattr(sample_x, "shape") and len(sample_x.shape) >= 3:
        _, window_len, n_features = sample_x.shape
        input_shape = (int(window_len), int(n_features))
    else:
        raise RuntimeError("Sample input shape unexpected. Expected tensor shape (batch, window_len, n_features).")

    num_classes = int(config.get("num_classes", 2))

    models_to_run = models_to_run or config.get("models", ["mlp", "ds_cnn", "lstm"])
    if not isinstance(models_to_run, (list, tuple)) or len(models_to_run) == 0:
        raise ValueError("models_to_run must be a non-empty list of model identifiers.")

    all_results: Dict[str, Dict[str, Any]] = {}
    all_histories: Dict[str, Dict[str, list]] = {}

    # Per-model loop
    for model_name in models_to_run:
        experiment_logger.section(f"Training model: {model_name}")

        model_exp_dir = experiment_root / "runs" / model_name
        model_exp_dir.mkdir(parents=True, exist_ok=True)

        # Model instance
        model_kwargs = config.get("model_defaults", {}).get(model_name, {})
        model = _model_factory(model_name, input_shape=input_shape, num_classes=num_classes, **(model_kwargs or {}))

        # Create per-model loggers (ExperimentLogger, MetricLogger)
        exp_log, model_metric_logger = get_loggers(model_exp_dir / "logs")
        # Trainer expects a standard logging.Logger for its `logger` parameter (per your trainer signature)
        trainer_logger = exp_log.logger  # Extract underlying logging.Logger
        trainer_logger.info(f"Starting run for model {model_name} in {model_exp_dir}")

        # Build trainer config or pass main config (depending on your Trainer expectations)
        trainer_config = config  # the Trainer implementation you have accepts a config dict

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device="cuda" if torch.cuda.is_available() else "cpu",
            experiment_dir=model_exp_dir,
            logger=exp_log,
            metric_logger=model_metric_logger,
            resume_from=resume,
        )

        # Train
        epochs = int(config.get("training", {}).get("epochs", 20))
        history = trainer.fit(epochs=epochs)
        save_json(history, model_exp_dir / "training_history.json")
        all_histories[model_name] = history

        # Evaluate
        evaluator = ModelEvaluator(
            model=trainer.model,
            test_loader=test_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            logger=trainer_logger
        )

        profiling_input_shape = (1, input_shape[0], input_shape[1])  # batch=1 for profiling
        try:
            results = evaluator.evaluate(calculate_efficiency=True, input_shape_for_profiling=profiling_input_shape)
        except Exception as e:
            trainer_logger.warning(f"Evaluator failed with exception: {e}")
            # fallback: attempt to run without efficiency profiling
            results = evaluator.evaluate(calculate_efficiency=False)

        results["model_name"] = model_name
        results["training_epochs"] = len(history.get("train_loss", []))
        results["input_shape"] = profiling_input_shape
        results["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
        save_json(results, model_exp_dir / "results.json")
        all_results[model_name] = results

        # Log a lightweight metric to global MetricLogger
        try:
            metric_logger_global.log_metrics(
                {"accuracy": results.get("accuracy", np.nan), "f1_macro": results.get("f1_macro", np.nan)},
                step=None,
                epoch=results["training_epochs"]
            )
        except Exception:
            # Continue even if global logging fails
            trainer_logger.debug("Global metric logging failed (non-fatal).")

        trainer_logger.info(f"Completed model {model_name}. Results saved to {model_exp_dir / 'results.json'}")

    # --- After all models: generate comparison plots ---
    plots_dir = experiment_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    experiment_logger.section("Generating comparison plots")

    # Normalize results for plotting
    for name, res in all_results.items():
        # make shallow copies of nested pieces to avoid mutation surprises
        if "params" in res and "parameters" not in res:
            try:
                res["parameters"] = float(res["params"])
            except Exception:
                res["parameters"] = res["params"]
        if "inference_time" in res and isinstance(res["inference_time"], dict):
            res["inference_time_mean_ms"] = float(res["inference_time"].get("mean_ms", np.nan))

    # Plot comparisons (wrapped in try/except so plotting doesn't break experiment)
    try:
        plot_model_comparison(
            results=all_results,
            metrics_to_plot=config.get("report_metrics", ["accuracy", "f1_macro", "parameters", "inference_time_mean_ms"]),
            save_path=plots_dir / "model_comparison.png",
            title="Model Comparison"
        )
    except Exception as e:
        experiment_logger.warning(f"plot_model_comparison failed: {e}")

    try:
        plot_parameter_efficiency(
            results=all_results,
            save_path=plots_dir / "parameter_efficiency.png",
            title="Parameter Efficiency"
        )
    except Exception as e:
        experiment_logger.warning(f"plot_parameter_efficiency failed: {e}")

    if "cross_dataset_results" in config:
        try:
            plot_cross_dataset_performance(
                results=config["cross_dataset_results"],
                save_path=plots_dir / "cross_dataset_performance.png",
                title="Cross-Dataset Performance"
            )
        except Exception as e:
            experiment_logger.warning(f"plot_cross_dataset_performance failed: {e}")

    # Save aggregated results and histories
    save_json(all_results, experiment_root / "all_results.json")
    save_json(all_histories, experiment_root / "all_histories.json")

    # Minimal decision report fallback if no custom generator is installed
    try:
        from src.reports.decision import generate_decision_report  # type: ignore
        generate_decision_report(results=all_results, exp_dir=experiment_root, config=config, logger=experiment_logger)
    except Exception:
        report = {
            "experiment": str(experiment_root),
            "timestamp": ts,
            "models": list(all_results.keys()),
            "summary": {k: {"accuracy": float(all_results[k].get("accuracy", np.nan)), "f1_macro": float(all_results[k].get("f1_macro", np.nan))} for k in all_results}
        }
        save_json(report, experiment_root / "decision_report_minimal.json")
        experiment_logger.info(f"Saved minimal decision report to {experiment_root / 'decision_report_minimal.json'}")

    experiment_logger.info(f"Experiment complete. Artifacts saved to: {experiment_root}")
    return experiment_root


# ---------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline model comparison")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase1_config.yaml',
        help='Path to experiment config file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/cic_ids_2017',
        help='Path to preprocessed data directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick mode (10 epochs instead of 50)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['mlp', 'ds_cnn', 'lstm'],
        default=None,
        help='Specific models to train (default: all three)'
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=None,
        help='Custom experiment directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 1: Baseline Model Comparison on CIC-IDS2017")
    print("=" * 70)
    print()
    
    if args.quick:
        print("🚀 Running in QUICK mode (10 epochs)")
    else:
        print("🐢 Running in FULL mode (50 epochs)")
    print()
    
    # Build configuration
    data_base = Path(args.data_dir)
    
    # Check if data exists
    required_paths = [
        data_base / 'train',
        data_base / 'val',
        data_base / 'test'
    ]
    
    print("Checking for preprocessed data...")
    all_exist = True
    for path in required_paths:
        x_file = path / 'X.npy'
        y_file = path / 'y.npy'
        if x_file.exists() and y_file.exists():
            print(f"✓ Found data in {path}")
        else:
            print(f"❌ Missing data files in {path}")
            print(f"   Expected: {x_file} and {y_file}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ ERROR: Preprocessed data not found!")
        print("\nPlease run the preprocessing first:")
        print("    python -m src.data.cli preprocess --dataset cic_ids_2017 --mode quick")
        print()
        sys.exit(1)
    
    print("\n✓ All data files found. Starting experiment...\n")
    
    # Build config
    config = {
        "experiment_name": "Phase1_Baseline_CIC_IDS_2017",
        "data": {
            "train_path": str(data_base / "train"),
            "val_path": str(data_base / "val"),
            "test_path": str(data_base / "test"),
            "batch_size": 64,
            "num_workers": 4,
            "mode": "memmap",
            "return_dict": True,
        },
        "models": args.models or ["mlp", "ds_cnn", "lstm"],
        "model_defaults": {
            "mlp": {
                "hidden_sizes": (128, 64, 32),
                "dropout_rate": 0.3,
                "activation": "relu",
                "use_batchnorm": True,
                "flatten_input": True
            },
            "ds_cnn": {
                "conv_channels": (32, 64, 64),
                "kernel_size": 3,
                "dropout_rate": 0.2,
                "use_bn": True,
                "activation": "relu",
                "classifier_hidden": 64
            },
            "lstm": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False
            }
        },
        "num_classes": 2,
        "training": {
            "epochs": 10 if args.quick else 50,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "use_amp": False
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        },
        "loss": {
            "name": "crossentropy"
        },
        "callbacks": {
            "early_stopping": {
                "enabled": True,
                "patience": 5 if args.quick else 10,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "mode": "min"
            },
            "model_checkpoint": {
                "enabled": True,
                "monitor": "val_loss",
                "mode": "min",
                "save_best_only": True,
                "save_weights_only": True,
                "filename": "best_model_epoch{epoch}_loss{metric:.4f}.pth",
                "save_last": True
            },
            "lr_scheduler": {
                "enabled": True,
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 3,
                "monitor": "val_loss"
            }
        },
        "report_metrics": [
            "accuracy", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "parameters", "flops", "inference_time_mean_ms"
        ]
    }
    
    # Run experiment
    start_time = time.time()
    
    try:
        experiment_dir = run_full_experiment(
            config=config,
            models_to_run=config["models"],
            experiment_root=args.experiment_dir,
            seed=args.seed
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Experiment completed in {elapsed_time / 60:.1f} minutes")
        
        # Print summary
        results_file = experiment_dir / "all_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\n" + "=" * 70)
            print("EXPERIMENT SUMMARY")
            print("=" * 70)
            print(f"\nResults saved to: {experiment_dir}")
            print("\nModel Performance:")
            print("-" * 70)
            print(f"{'Model':<15} {'Accuracy':<12} {'F1-Macro':<12} {'Params':<12} {'Inference (ms)':<15}")
            print("-" * 70)
            
            for model_name, metrics in results.items():
                accuracy = metrics.get('accuracy', 0) * 100
                f1 = metrics.get('f1_macro', 0) * 100
                params = metrics.get('parameters', metrics.get('params', 0))
                
                inf_time = metrics.get('inference_time_mean_ms', 'N/A')
                if isinstance(inf_time, dict):
                    inf_time = inf_time.get('mean_ms', 'N/A')
                if isinstance(inf_time, (int, float)):
                    inf_time = f"{inf_time:.2f}"
                
                print(f"{model_name:<15} {accuracy:>10.2f}%  {f1:>10.2f}%  {params:>10}  {inf_time:>13}")
            
            print("-" * 70)
            print(f"\nDetailed results: {experiment_dir / 'all_results.json'}")
            print(f"Plots: {experiment_dir / 'plots'}")
            print(f"Per-model logs: {experiment_dir / 'runs'}")
            print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: Experiment failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
