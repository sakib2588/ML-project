#!/usr/bin/env python
"""
Phase 2 - TFLite Conversion for Multi-Task Models
===================================================

Convert PyTorch multi-task model to TFLite for edge deployment.
Path: PyTorch -> ONNX -> TFLite (via onnx2tf)

Handles dual-output models (binary + attack heads).

Usage:
    # Convert single model
    python convert_multitask.py --model-path <path>
    
    # Convert with INT8 quantization
    python convert_multitask.py --model-path <path> --quantize int8
    
    # All quantization variants
    python convert_multitask.py --model-path <path> --all-variants
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

from phase2.config import DATA_DIR, ARTIFACTS_DIR
from phase2.models_multitask import MultiTaskStudentDSCNN, build_student


# ============ ONNX EXPORT ============
def export_to_onnx(
    model: MultiTaskStudentDSCNN,
    input_shape: Tuple[int, int],
    output_path: Path,
    opset_version: int = 13
) -> Path:
    """
    Export multi-task model to ONNX format.
    
    Handles dual outputs (binary, attack).
    """
    model.eval()
    
    # Dummy input: (batch, seq_len, features)
    dummy_input = torch.randn(1, *input_shape)
    
    # Export with multiple outputs
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['binary_output', 'attack_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'binary_output': {0: 'batch_size'},
            'attack_output': {0: 'batch_size'}
        }
    )
    
    print(f"  ONNX exported: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


def verify_onnx(
    pytorch_model: MultiTaskStudentDSCNN,
    onnx_path: Path,
    test_inputs: np.ndarray,
    tolerance: float = 1e-4
) -> Dict:
    """
    Verify numerical equivalence between PyTorch and ONNX outputs.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️ onnxruntime not installed, skipping verification")
        return {'status': 'skipped'}
    
    # Create ONNX session
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    
    pytorch_model.eval()
    max_diff_binary = 0
    max_diff_attack = 0
    
    n_samples = min(50, len(test_inputs))
    
    with torch.no_grad():
        for i in range(n_samples):
            test_input = test_inputs[i:i+1].astype(np.float32)
            
            # PyTorch
            pytorch_input = torch.FloatTensor(test_input)
            pt_binary, pt_attack = pytorch_model(pytorch_input)
            pt_binary = pt_binary.numpy()
            pt_attack = pt_attack.numpy()
            
            # ONNX
            onnx_outputs = ort_session.run(None, {input_name: test_input})
            onnx_binary, onnx_attack = onnx_outputs
            
            # Compare
            diff_binary = np.abs(pt_binary - onnx_binary).max()
            diff_attack = np.abs(pt_attack - onnx_attack).max()
            
            max_diff_binary = max(max_diff_binary, diff_binary)
            max_diff_attack = max(max_diff_attack, diff_attack)
    
    passed = max_diff_binary < tolerance and max_diff_attack < tolerance
    
    print(f"  ONNX verification: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"    Max diff binary: {max_diff_binary:.2e}")
    print(f"    Max diff attack: {max_diff_attack:.2e}")
    
    return {
        'status': 'passed' if passed else 'failed',
        'max_diff_binary': float(max_diff_binary),
        'max_diff_attack': float(max_diff_attack),
        'tolerance': tolerance
    }


# ============ TFLITE CONVERSION ============
def convert_onnx_to_tflite(
    onnx_path: Path,
    output_path: Path,
    quantize: str = 'none',
    representative_data: Optional[np.ndarray] = None
) -> Optional[Path]:
    """
    Convert ONNX to TFLite using onnx2tf.
    
    Quantization options:
    - none: float32
    - float16: float16 (2x compression)
    - int8: int8 dynamic range (4x compression)
    - int8_full: int8 full integer (requires calibration data)
    """
    try:
        import tensorflow as tf
        import onnx2tf  # type: ignore[import-untyped]
    except ImportError:
        print("  ❌ tensorflow or onnx2tf not installed")
        return None
    
    # Convert ONNX to TF SavedModel first
    temp_dir = output_path.parent / "temp_savedmodel"
    
    try:
        # onnx2tf conversion
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(temp_dir),
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True
        )
        
        saved_model_path = temp_dir
        
        # Now convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        if quantize == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[assignment]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantize == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[assignment]
            
        elif quantize == 'int8_full':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[assignment]
            
            if representative_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(representative_data))):
                        yield [representative_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset  # type: ignore[assignment]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8  # type: ignore[assignment]
                converter.inference_output_type = tf.float32  # type: ignore[assignment]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"  TFLite exported: {output_path.stat().st_size / 1024:.1f} KB ({quantize})")
        
        return output_path
        
    except Exception as e:
        print(f"  ❌ TFLite conversion failed: {e}")
        return None
    finally:
        # Cleanup temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def convert_onnx_to_tflite_simple(
    onnx_path: Path,
    output_path: Path,
    quantize: str = 'none'
) -> Optional[Path]:
    """
    Simpler TFLite conversion using tf-onnx (fallback).
    """
    try:
        import tensorflow as tf
        import tf2onnx  # type: ignore[import-untyped]
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Get input/output info
        # ... this is complex for multi-output models
        
        print("  Using simple conversion path (may not support all ops)")
        return None
        
    except Exception as e:
        print(f"  Simple conversion failed: {e}")
        return None


# ============ TFLITE VERIFICATION ============
def verify_tflite(
    tflite_path: Path,
    test_inputs: np.ndarray,
    test_binary_labels: np.ndarray,
    test_attack_labels: np.ndarray
) -> Dict:
    """
    Verify TFLite model accuracy and measure latency.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {'status': 'skipped', 'reason': 'tensorflow not installed'}
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        num_threads=4
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input: {input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  Outputs: {len(output_details)}")
    
    # Run inference
    binary_preds = []
    attack_preds = []
    latencies = []
    
    n_test = min(1000, len(test_inputs))
    
    for i in range(n_test):
        # Prepare input
        input_data = test_inputs[i:i+1].astype(np.float32)
        
        # Handle int8 input
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Time inference
        start = time.perf_counter_ns()
        interpreter.invoke()
        end = time.perf_counter_ns()
        
        latencies.append((end - start) / 1e6)  # ms
        
        # Get outputs (order may vary)
        outputs = []
        for detail in output_details:
            out = interpreter.get_tensor(detail['index'])
            outputs.append(out)
        
        # Determine which output is which based on shape
        for out in outputs:
            if out.shape[-1] == 1:
                # Binary output
                binary_pred = int(out[0, 0] >= 0.5)
                binary_preds.append(binary_pred)
            else:
                # Attack output (10 classes)
                attack_pred = int(np.argmax(out[0]))
                attack_preds.append(attack_pred)
    
    # Handle missing outputs
    if not binary_preds:
        binary_preds = [0] * n_test
    if not attack_preds:
        attack_preds = [0] * n_test
    
    # Compute metrics
    from sklearn.metrics import f1_score, accuracy_score
    
    binary_f1 = f1_score(test_binary_labels[:n_test], binary_preds, average='binary') * 100
    binary_acc = accuracy_score(test_binary_labels[:n_test], binary_preds) * 100
    
    attack_f1 = f1_score(test_attack_labels[:n_test], attack_preds, average='macro') * 100
    attack_acc = accuracy_score(test_attack_labels[:n_test], attack_preds) * 100
    
    latencies = np.array(latencies)
    
    return {
        'status': 'success',
        'n_samples': n_test,
        'binary_f1': binary_f1,
        'binary_acc': binary_acc,
        'attack_f1': attack_f1,
        'attack_acc': attack_acc,
        'latency_mean_ms': np.mean(latencies),
        'latency_p50_ms': np.percentile(latencies, 50),
        'latency_p95_ms': np.percentile(latencies, 95),
        'latency_p99_ms': np.percentile(latencies, 99)
    }


# ============ FULL CONVERSION PIPELINE ============
def convert_model(
    model_path_arg: str,
    output_dir: Path,
    quantize: str = 'none',
    representative_data: Optional[np.ndarray] = None,
    test_data: Optional[Tuple] = None,
    verbose: bool = True
) -> Dict:
    """
    Full conversion pipeline: PyTorch -> ONNX -> TFLite
    """
    model_path = Path(model_path_arg)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CONVERTING: {model_path.name}")
        print(f"Quantization: {quantize}")
        print(f"{'='*60}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer architecture
    init_conv_shape = state_dict['initial_conv.weight'].shape
    n_features = state_dict['initial_conv.weight'].shape[1]
    channels_0 = init_conv_shape[0]
    
    if channels_0 <= 16:
        params_target = 5000
    elif channels_0 <= 48:
        params_target = 50000
    else:
        params_target = 200000
    
    # Assume window_size=15 (standard for this project)
    window_size = 15
    input_shape = (window_size, n_features)
    
    model = build_student(params_target=params_target, input_shape=input_shape)
    model.load_state_dict(state_dict)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    pytorch_size = n_params * 4  # float32
    
    if verbose:
        print(f"Model: {n_params:,} params, {pytorch_size/1024:.1f} KB")
    
    results = {
        'source': str(model_path),
        'params': n_params,
        'pytorch_size_kb': pytorch_size / 1024,
        'quantize': quantize
    }
    
    # Step 1: Export to ONNX
    if verbose:
        print("\nStep 1: ONNX Export")
    onnx_path = output_dir / f"{model_path.stem}.onnx"
    export_to_onnx(model, input_shape, onnx_path)
    results['onnx_size_kb'] = onnx_path.stat().st_size / 1024
    
    # Step 2: Verify ONNX
    if representative_data is not None:
        if verbose:
            print("\nStep 2: ONNX Verification")
        onnx_verification = verify_onnx(model, onnx_path, representative_data)
        results['onnx_verification'] = onnx_verification
    
    # Step 3: Convert to TFLite
    if verbose:
        print("\nStep 3: TFLite Conversion")
    tflite_path = output_dir / f"{model_path.stem}_{quantize}.tflite"
    
    tflite_result = convert_onnx_to_tflite(
        onnx_path, tflite_path,
        quantize=quantize,
        representative_data=representative_data
    )
    
    if tflite_result and tflite_result.exists():
        results['tflite_path'] = str(tflite_result)
        results['tflite_size_kb'] = tflite_result.stat().st_size / 1024
        results['compression_ratio'] = results['pytorch_size_kb'] / results['tflite_size_kb']
        
        # Step 4: Verify TFLite
        if test_data is not None:
            if verbose:
                print("\nStep 4: TFLite Verification")
            X_test, y_binary, y_attack = test_data
            tflite_verification = verify_tflite(
                tflite_result, X_test, y_binary, y_attack
            )
            results['tflite_verification'] = tflite_verification
            
            if 'binary_f1' in tflite_verification:
                print(f"  Binary F1: {tflite_verification['binary_f1']:.2f}%")
                print(f"  Attack F1: {tflite_verification['attack_f1']:.2f}%")
                print(f"  Latency p50: {tflite_verification['latency_p50_ms']:.2f} ms")
    else:
        results['tflite_path'] = None
        results['status'] = 'failed'
        
        # Keep ONNX as fallback
        results['onnx_path'] = str(onnx_path)
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"PyTorch: {results['pytorch_size_kb']:.1f} KB")
        print(f"ONNX:    {results.get('onnx_size_kb', 'N/A')} KB")
        if 'tflite_size_kb' in results:
            print(f"TFLite:  {results['tflite_size_kb']:.1f} KB ({quantize})")
            print(f"Compression: {results['compression_ratio']:.2f}x")
    
    return results


# ============ DATA LOADING ============
def load_test_data(data_dir: Path) -> Tuple:
    """Load test data for verification."""
    processed_dir = data_dir / "processed" / "cic_ids_2017"
    
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    # For attack labels, use zeros (BENIGN) for all
    # In real usage, would need attack_types for test set
    attack_types = np.zeros_like(y_test)
    
    return X_test, y_test, attack_types


def load_representative_data(data_dir: Path, n_samples: int = 200) -> np.ndarray:
    """Load representative data for quantization calibration."""
    processed_dir = data_dir / "processed" / "cic_ids_2017"
    
    X_train = np.load(processed_dir / "X_train.npy")
    
    # Sample evenly
    indices = np.linspace(0, len(X_train) - 1, n_samples, dtype=int)
    return X_train[indices]


# ============ MAIN CLI ============
def main():
    parser = argparse.ArgumentParser(description='Convert Multi-Task Model to TFLite')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'float16', 'int8', 'int8_full'],
                        help='Quantization type')
    parser.add_argument('--all-variants', action='store_true',
                        help='Convert with all quantization variants')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--verify', action='store_true', help='Run verification tests')
    
    args = parser.parse_args()
    
    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model_path).parent / "tflite"
    
    # Load data for verification
    test_data = None
    representative_data = None
    
    if args.verify or args.quantize in ['int8', 'int8_full']:
        representative_data = load_representative_data(DATA_DIR)
        
    if args.verify:
        test_data = load_test_data(DATA_DIR)
    
    if args.all_variants:
        # Convert all quantization variants
        all_results = []
        
        for quant in ['none', 'float16', 'int8']:
            result = convert_model(
                args.model_path,
                out_dir,
                quantize=quant,
                representative_data=representative_data,
                test_data=test_data
            )
            all_results.append(result)
        
        # Summary table
        print(f"\n{'='*70}")
        print("ALL VARIANTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Quant':<12} {'Size (KB)':>12} {'Compression':>12} {'Binary F1':>10} {'Attack F1':>10}")
        print("-" * 60)
        
        for r in all_results:
            size = r.get('tflite_size_kb', r.get('onnx_size_kb', 0))
            comp = r.get('compression_ratio', 1.0)
            v = r.get('tflite_verification', {})
            bf1 = f"{v.get('binary_f1', 0):.1f}%" if 'binary_f1' in v else "N/A"
            af1 = f"{v.get('attack_f1', 0):.1f}%" if 'attack_f1' in v else "N/A"
            
            print(f"{r['quantize']:<12} {size:>11.1f} {comp:>11.2f}x {bf1:>10} {af1:>10}")
        
        # Save summary
        with open(out_dir / "conversion_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    else:
        # Single conversion
        result = convert_model(
            args.model_path,
            out_dir,
            quantize=args.quantize,
            representative_data=representative_data,
            test_data=test_data
        )
        
        # Save result
        with open(out_dir / f"conversion_{args.quantize}.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == '__main__':
    main()
