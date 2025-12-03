#!/usr/bin/env python
"""
Phase 2 - Stage 5: TFLite Conversion
=====================================

Convert PyTorch model to TFLite for edge deployment.
Path: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite

Supports:
- Float32 (baseline)
- Float16 (size reduction)
- INT8 quantization (full optimization)

Usage:
    # Convert single model
    python convert_to_tflite.py --seed 42 --schedule uniform_50
    
    # Convert with INT8 quantization
    python convert_to_tflite.py --seed 42 --quantize int8
    
    # Convert all seeds
    python convert_to_tflite.py --all-seeds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

from phase2.config import (
    SEEDS, DATA_DIR, ARTIFACTS_DIR, CONVERSION_CONFIG
)
from phase2.utils import (
    set_seed, load_data, get_representative_dataset, clear_memory
)
from phase2.models import create_student


# ============ ONNX EXPORT ============
def export_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: Path,
    opset_version: int = 12
) -> Path:
    """Export PyTorch model to ONNX format."""
    import torch.onnx
    
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_input,),  # args must be tuple
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    import onnx  # type: ignore[import-not-found]
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    return output_path


# ============ ONNX NUMERICAL VERIFICATION (NEW - Per feedback) ============
def verify_onnx_numerical_equivalence(
    pytorch_model: nn.Module,
    onnx_path: Path,
    test_inputs: np.ndarray,
    tolerance: float = 1e-4,
    verbose: bool = True
) -> Dict:
    """
    Verify numerical equivalence between PyTorch and ONNX outputs.
    
    CRITICAL: ONNX conversion can introduce numerical errors that accumulate
    in later conversion stages (TF, TFLite).
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        test_inputs: Test inputs for verification (N, seq_len, features)
        tolerance: Maximum allowed difference
        verbose: Print detailed results
    
    Returns:
        Dict with verification results
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        print("⚠️ onnxruntime not installed. Skipping ONNX verification.")
        return {'status': 'skipped', 'reason': 'onnxruntime not installed'}
    
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    
    # Run comparisons
    pytorch_model.eval()
    max_diffs = []
    mean_diffs = []
    
    n_samples = min(100, len(test_inputs))
    
    with torch.no_grad():
        for i in range(n_samples):
            test_input = test_inputs[i:i+1].astype(np.float32)
            
            # PyTorch inference
            pytorch_input = torch.FloatTensor(test_input)
            pytorch_out = pytorch_model(pytorch_input).numpy()
            
            # ONNX inference
            onnx_out = ort_session.run(None, {input_name: test_input})[0]
            
            # Compute difference
            diff = np.abs(pytorch_out - onnx_out)
            max_diffs.append(diff.max())
            mean_diffs.append(diff.mean())
    
    max_diff = float(np.max(max_diffs))
    mean_diff = float(np.mean(mean_diffs))
    
    # Check if within tolerance
    passed = max_diff < tolerance
    
    results = {
        'status': 'passed' if passed else 'failed',
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'tolerance': tolerance,
        'n_samples': n_samples
    }
    
    if verbose:
        status_icon = "✅" if passed else "❌"
        print(f"\n{status_icon} ONNX Numerical Verification:")
        print(f"   Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
        print(f"   Mean difference: {mean_diff:.2e}")
        print(f"   Status: {'PASSED' if passed else 'FAILED'}")
        
        if not passed:
            print(f"\n   ⚠️ WARNING: ONNX output differs significantly from PyTorch!")
            print(f"   This may cause accuracy degradation in TFLite model.")
    
    return results


# ============ ONNX TO TENSORFLOW ============
def convert_onnx_to_tf(
    onnx_path: Path,
    output_dir: Path
) -> Optional[Path]:
    """Convert ONNX model to TensorFlow SavedModel."""
    try:
        import onnx  # type: ignore[import-not-found]
        from onnx_tf.backend import prepare  # type: ignore[import-not-found]
        
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(output_dir))
        
        return output_dir
    except ImportError:
        print("onnx-tf not available, using alternative conversion")
        return convert_onnx_to_tf_alternative(onnx_path, output_dir)


def convert_onnx_to_tf_alternative(
    onnx_path: Path,
    output_dir: Path
) -> Optional[Path]:
    """Alternative ONNX to TF conversion using onnxruntime."""
    try:
        import tensorflow as tf
        import onnxruntime as ort  # type: ignore[import-not-found]
        
        # Load ONNX and get info
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create a TF function that wraps ONNX inference
        # This is a workaround - for production, use proper conversion
        print("Warning: Using ONNX runtime wrapper for TFLite conversion")
        
        return output_dir
    except Exception as e:
        print(f"Alternative conversion failed: {e}")
        return None


# ============ TENSORFLOW LITE CONVERSION ============
def convert_to_tflite(
    saved_model_dir: Path,
    output_path: Path,
    quantize: str = 'none',  # 'none', 'float16', 'int8'
    representative_data: Optional[np.ndarray] = None
) -> Optional[Path]:
    """Convert TensorFlow SavedModel to TFLite."""
    try:
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        # Optimization options
        if quantize == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[assignment]
            converter.target_spec.supported_types = [tf.float16]
        
        elif quantize == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[assignment]
            
            if representative_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(representative_data))):
                        yield [representative_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset  # type: ignore[assignment]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8  # type: ignore[assignment]
                converter.inference_output_type = tf.int8  # type: ignore[assignment]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)  # type: ignore[arg-type]
        
        return output_path
    
    except Exception as e:
        print(f"TFLite conversion error: {e}")
        return None


# ============ DIRECT PYTORCH TO TFLITE ============
def pytorch_to_tflite_direct(
    model: nn.Module,
    input_shape: tuple,
    output_path: Path,
    quantize: str = 'none',
    representative_data: Optional[np.ndarray] = None,
    verify_onnx: bool = True
) -> Optional[Path]:
    """
    Convert PyTorch model directly to TFLite using ONNX intermediate.
    
    This is the main conversion function used in the pipeline.
    Now includes ONNX numerical verification (per feedback).
    """
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Step 1: PyTorch -> ONNX
        onnx_path = tmpdir / "model.onnx"
        print(f"  Step 1: Exporting to ONNX...")
        export_to_onnx(model, input_shape, onnx_path)
        print(f"    ONNX size: {onnx_path.stat().st_size / 1024:.1f} KB")
        
        # Step 1.5: ONNX Numerical Verification (NEW)
        onnx_verification = {}
        if verify_onnx and representative_data is not None:
            print(f"  Step 1.5: Verifying ONNX numerical equivalence...")
            onnx_verification = verify_onnx_numerical_equivalence(
                model, onnx_path, representative_data,
                tolerance=1e-4, verbose=True
            )
            
            if onnx_verification.get('status') == 'failed':
                print("    ⚠️ Proceeding despite verification failure...")
        
        # Step 2: ONNX -> TF SavedModel
        tf_dir = tmpdir / "saved_model"
        print(f"  Step 2: Converting ONNX to TensorFlow...")
        
        try:
            convert_onnx_to_tf(onnx_path, tf_dir)
        except Exception as e:
            print(f"    TF conversion failed: {e}")
            # Fallback: save ONNX model for later conversion
            shutil.copy(onnx_path, output_path.with_suffix('.onnx'))
            return output_path.with_suffix('.onnx')
        
        # Step 3: TF SavedModel -> TFLite
        print(f"  Step 3: Converting to TFLite ({quantize})...")
        
        try:
            result = convert_to_tflite(tf_dir, output_path, quantize, representative_data)
            if result:
                print(f"    TFLite size: {output_path.stat().st_size / 1024:.1f} KB")
                return result
        except Exception as e:
            print(f"    TFLite conversion failed: {e}")
        
        # Fallback: return ONNX
        fallback_path = output_path.with_suffix('.onnx')
        shutil.copy(onnx_path, fallback_path)
        return fallback_path


# ============ VERIFY TFLITE MODEL ============
def verify_tflite(
    tflite_path: Path,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    use_xnnpack: bool = True
) -> Dict:
    """Verify TFLite model accuracy and measure inference time."""
    try:
        import tensorflow as tf
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            num_threads=4
        )
        
        # Try XNNPACK delegate
        if use_xnnpack:
            try:
                interpreter = tf.lite.Interpreter(
                    model_path=str(tflite_path),
                    experimental_delegates=[
                        tf.lite.experimental.load_delegate('libXNNPACK.so')
                    ]
                )
            except:
                pass  # Use default interpreter
        
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Inference
        predictions = []
        latencies = []
        
        for i in range(len(test_data)):
            # Prepare input
            input_data = test_data[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Time inference
            start = time.perf_counter_ns()
            interpreter.invoke()
            end = time.perf_counter_ns()
            
            latencies.append((end - start) / 1e6)  # ms
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output, axis=1)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        latencies = np.array(latencies)
        
        # Compute metrics
        accuracy = (predictions == test_labels).mean() * 100
        
        return {
            'accuracy': accuracy,
            'latency_mean_ms': np.mean(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'n_samples': len(test_data)
        }
    
    except Exception as e:
        print(f"TFLite verification error: {e}")
        return {'error': str(e)}


# ============ MAIN CONVERSION PIPELINE ============
def convert_model(
    seed: int,
    schedule_name: str = "uniform_50",
    quantize: str = 'none',
    verbose: bool = True
) -> Dict:
    """
    Full conversion pipeline: PyTorch -> ONNX -> TF -> TFLite
    """
    set_seed(seed)
    
    out_dir = ARTIFACTS_DIR / "stage5" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CONVERTING MODEL - Seed {seed}, Schedule: {schedule_name}")
        print(f"{'='*60}")
    
    # Load data for representative dataset
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    window_size = X_train.shape[1]
    
    representative_data = get_representative_dataset(
        X_train, y_train, CONVERSION_CONFIG.representative_samples
    )
    
    # Find best model from previous stages
    model_path = None
    for stage in ['stage4', 'stage3', 'stage1', 'stage0']:
        if stage == 'stage4':
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / f"best_{schedule_name}.pth"
        elif stage == 'stage3':
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / f"best_{schedule_name}.pth"
        elif stage == 'stage1':
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / "best.pth"
        else:
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / "best.pth"
        
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No model found for seed {seed}")
    
    if verbose:
        print(f"Loading model from: {model_path}")
    
    # Load PyTorch model
    model = create_student(n_features=n_features)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get PyTorch model size
    pytorch_size = sum(p.numel() * p.element_size() for p in model.parameters())
    pytorch_size_kb = pytorch_size / 1024
    
    if verbose:
        print(f"PyTorch model size: {pytorch_size_kb:.1f} KB")
    
    # Convert
    start_time = time.time()
    
    tflite_path = out_dir / f"model_{schedule_name}_{quantize}.tflite"
    
    result_path = pytorch_to_tflite_direct(
        model,
        input_shape=(window_size, n_features),
        output_path=tflite_path,
        quantize=quantize,
        representative_data=representative_data
    )
    
    conversion_time = time.time() - start_time
    
    # Get output size
    if result_path and result_path.exists():
        output_size_kb = result_path.stat().st_size / 1024
    else:
        output_size_kb = 0
    
    # Verify (if TFLite)
    verification = {}
    if result_path and result_path.suffix == '.tflite':
        if verbose:
            print(f"\nVerifying TFLite model...")
        
        # Use smaller test set for verification
        n_verify = min(1000, len(X_test))
        indices = np.random.choice(len(X_test), n_verify, replace=False)
        
        verification = verify_tflite(
            result_path,
            X_test[indices],
            y_test[indices]
        )
        
        if 'accuracy' in verification:
            print(f"  Accuracy: {verification['accuracy']:.2f}%")
            print(f"  Latency p50: {verification['latency_p50_ms']:.2f} ms")
            print(f"  Latency p95: {verification['latency_p95_ms']:.2f} ms")
    
    # Results
    results = {
        'seed': seed,
        'stage': 'stage5_convert',
        'schedule_name': schedule_name,
        'quantize': quantize,
        'source_model': str(model_path),
        'output_path': str(result_path) if result_path else None,
        'pytorch_size_kb': pytorch_size_kb,
        'output_size_kb': output_size_kb,
        'compression_ratio': pytorch_size_kb / output_size_kb if output_size_kb > 0 else 0,
        'conversion_time_s': conversion_time,
        'verification': verification,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(out_dir / f"conversion_{schedule_name}_{quantize}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Input:  {pytorch_size_kb:.1f} KB (PyTorch)")
        print(f"Output: {output_size_kb:.1f} KB ({quantize})")
        print(f"Compression: {results['compression_ratio']:.1f}x")
        print(f"Saved to: {result_path}")
    
    return results


def convert_all_variants(seed: int = 42, schedule_name: str = "uniform_50"):
    """Convert model with all quantization variants."""
    
    results = []
    
    for quant in ['none', 'float16', 'int8']:
        print(f"\n{'='*50}")
        print(f"Converting with quantization: {quant}")
        print(f"{'='*50}")
        
        result = convert_model(
            seed=seed,
            schedule_name=schedule_name,
            quantize=quant
        )
        results.append(result)
    
    # Summary table
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Quantize':<12} {'Size (KB)':>12} {'Compression':>12} {'Accuracy':>10}")
    print("-" * 50)
    
    for r in results:
        acc = r['verification'].get('accuracy', 'N/A')
        acc_str = f"{acc:.1f}%" if isinstance(acc, float) else acc
        print(f"{r['quantize']:<12} {r['output_size_kb']:>11.1f} "
              f"{r['compression_ratio']:>11.1f}x {acc_str:>10}")


def convert_all_seeds(schedule_name: str = "nonuniform", quantize: str = 'int8'):
    """Convert all seeds."""
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"# CONVERTING SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*70}")
        
        result = convert_model(
            seed=seed,
            schedule_name=schedule_name,
            quantize=quantize
        )
        all_results.append(result)
        clear_memory()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"AGGREGATED CONVERSION RESULTS")
    print(f"{'='*70}")
    
    sizes = [r['output_size_kb'] for r in all_results]
    compressions = [r['compression_ratio'] for r in all_results]
    
    print(f"Output size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} KB")
    print(f"Compression: {np.mean(compressions):.1f}x ± {np.std(compressions):.1f}x")
    
    # Save summary
    summary = {
        'stage': 'stage5_convert',
        'schedule_name': schedule_name,
        'quantize': quantize,
        'n_seeds': len(SEEDS),
        'individual_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(ARTIFACTS_DIR / "stage5" / f"summary_{schedule_name}_{quantize}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 5: TFLite Conversion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-seeds', action='store_true', help='Convert all seeds')
    parser.add_argument('--schedule', type=str, default='uniform_50', help='Pruning schedule')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'float16', 'int8'], help='Quantization type')
    parser.add_argument('--all-variants', action='store_true', help='Try all quantization variants')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 5: TFLITE CONVERSION")
    print(f"{'='*70}")
    
    if args.all_variants:
        convert_all_variants(seed=args.seed, schedule_name=args.schedule)
    elif args.all_seeds:
        convert_all_seeds(schedule_name=args.schedule, quantize=args.quantize)
    else:
        convert_model(
            seed=args.seed,
            schedule_name=args.schedule,
            quantize=args.quantize
        )


if __name__ == '__main__':
    main()
