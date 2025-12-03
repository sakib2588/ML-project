#!/usr/bin/env python
"""
Phase 2 - Raspberry Pi 4 Benchmark Suite
=========================================

Benchmark inference performance for deployment on ARM CPU.
Measures latency, throughput, memory usage, and power estimation.

Usage:
    # Benchmark PyTorch model
    python benchmark_pi4.py --model-path <path> --format pytorch
    
    # Benchmark TFLite model
    python benchmark_pi4.py --model-path <path> --format tflite
    
    # Full benchmark suite
    python benchmark_pi4.py --model-path <path> --full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import gc
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings('ignore')


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    warmup_iterations: int = 50
    benchmark_iterations: int = 500
    batch_sizes: Optional[List[int]] = None
    num_threads: int = 4
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]


@dataclass
class BenchmarkResults:
    """Benchmark results container."""
    model_name: str
    model_format: str
    model_size_kb: float
    params: int
    
    # Latency metrics (ms)
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float
    
    # Throughput
    throughput_fps: float
    
    # Memory
    memory_mb: float
    
    # Per-batch results
    batch_results: Dict
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_format': self.model_format,
            'model_size_kb': self.model_size_kb,
            'params': self.params,
            'latency': {
                'mean_ms': self.latency_mean,
                'std_ms': self.latency_std,
                'p50_ms': self.latency_p50,
                'p95_ms': self.latency_p95,
                'p99_ms': self.latency_p99,
                'min_ms': self.latency_min,
                'max_ms': self.latency_max
            },
            'throughput_fps': self.throughput_fps,
            'memory_mb': self.memory_mb,
            'batch_results': self.batch_results
        }
    
    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {self.model_name}")
        print(f"{'='*60}")
        print(f"Format:      {self.model_format}")
        print(f"Size:        {self.model_size_kb:.1f} KB")
        print(f"Parameters:  {self.params:,}")
        print(f"\nLatency (batch=1):")
        print(f"  Mean:      {self.latency_mean:.2f} ms")
        print(f"  Std:       {self.latency_std:.2f} ms")
        print(f"  P50:       {self.latency_p50:.2f} ms")
        print(f"  P95:       {self.latency_p95:.2f} ms")
        print(f"  P99:       {self.latency_p99:.2f} ms")
        print(f"\nThroughput:  {self.throughput_fps:.1f} samples/sec")
        print(f"Memory:      {self.memory_mb:.1f} MB")


# ============ PYTORCH BENCHMARK ============
def benchmark_pytorch(
    model_path: Path,
    input_shape: Tuple[int, int],
    config: BenchmarkConfig
) -> BenchmarkResults:
    """Benchmark PyTorch model."""
    import torch
    from phase2.models_multitask import build_student
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer architecture
    init_conv_shape = state_dict['initial_conv.weight'].shape
    n_features = init_conv_shape[1]
    channels_0 = init_conv_shape[0]
    
    if channels_0 <= 16:
        params_target = 5000
    elif channels_0 <= 48:
        params_target = 50000
    else:
        params_target = 200000
    
    model = build_student(params_target=params_target, input_shape=(input_shape[0], n_features))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Set number of threads
    torch.set_num_threads(config.num_threads)
    
    # Get model info
    n_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"Model: {n_params:,} params, {model_size/1024:.1f} KB")
    print(f"Threads: {config.num_threads}")
    
    # Benchmark each batch size
    batch_results = {}
    latencies: np.ndarray = np.array([])  # Initialize to avoid unbound error
    batch_sizes = config.batch_sizes or [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(config.warmup_iterations):
                _ = model(dummy_input)
        
        # Benchmark
        latency_list = []
        with torch.no_grad():
            for _ in range(config.benchmark_iterations):
                start = time.perf_counter_ns()
                _ = model(dummy_input)
                end = time.perf_counter_ns()
                latency_list.append((end - start) / 1e6)  # ms
        
        latencies = np.array(latency_list)
        
        batch_results[batch_size] = {
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'throughput_fps': batch_size * 1000 / np.mean(latencies)
        }
        
        print(f"  Latency: {batch_results[batch_size]['latency_mean_ms']:.2f} ± "
              f"{batch_results[batch_size]['latency_std_ms']:.2f} ms")
        print(f"  Throughput: {batch_results[batch_size]['throughput_fps']:.1f} samples/sec")
    
    # Use batch_size=1 for main metrics
    bs1 = batch_results[1]
    
    # Estimate memory usage
    import tracemalloc
    tracemalloc.start()
    with torch.no_grad():
        _ = model(torch.randn(1, *input_shape))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return BenchmarkResults(
        model_name=model_path.stem,
        model_format='pytorch',
        model_size_kb=model_size / 1024,
        params=n_params,
        latency_mean=float(bs1['latency_mean_ms']),
        latency_std=float(bs1['latency_std_ms']),
        latency_p50=float(bs1['latency_p50_ms']),
        latency_p95=float(bs1['latency_p95_ms']),
        latency_p99=float(bs1['latency_p99_ms']),
        latency_min=float(min(latencies)),
        latency_max=float(max(latencies)),
        throughput_fps=float(bs1['throughput_fps']),
        memory_mb=peak / (1024 * 1024),
        batch_results=batch_results
    )


# ============ TFLITE BENCHMARK ============
def benchmark_tflite(
    model_path: Path,
    config: BenchmarkConfig
) -> Optional[BenchmarkResults]:
    """Benchmark TFLite model."""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed, skipping TFLite benchmark")
        return None
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=config.num_threads
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Input shape: {input_shape}, dtype: {input_dtype}")
    print(f"Threads: {config.num_threads}")
    
    # Get model size
    model_size = model_path.stat().st_size
    
    # Estimate params from size (rough estimate for int8)
    # float32: 4 bytes/param, int8: 1 byte/param
    if input_dtype == np.int8:
        n_params = model_size
    else:
        n_params = model_size // 4
    
    # Benchmark batch_size=1 only for TFLite (typical edge deployment)
    print("\nBenchmarking batch_size=1...")
    
    # Create dummy input
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        dummy_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    else:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(config.warmup_iterations):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    latencies = []
    for _ in range(config.benchmark_iterations):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        start = time.perf_counter_ns()
        interpreter.invoke()
        end = time.perf_counter_ns()
        
        latencies.append((end - start) / 1e6)
    
    latencies = np.array(latencies)
    
    batch_results = {
        1: {
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000 / np.mean(latencies)
        }
    }
    
    print(f"  Latency: {batch_results[1]['latency_mean_ms']:.2f} ± "
          f"{batch_results[1]['latency_std_ms']:.2f} ms")
    print(f"  Throughput: {batch_results[1]['throughput_fps']:.1f} samples/sec")
    
    # Memory estimate (TFLite model size + runtime overhead)
    memory_mb = model_size / (1024 * 1024) + 5  # 5MB overhead estimate
    
    return BenchmarkResults(
        model_name=model_path.stem,
        model_format='tflite',
        model_size_kb=model_size / 1024,
        params=n_params,
        latency_mean=float(batch_results[1]['latency_mean_ms']),
        latency_std=float(batch_results[1]['latency_std_ms']),
        latency_p50=float(batch_results[1]['latency_p50_ms']),
        latency_p95=float(batch_results[1]['latency_p95_ms']),
        latency_p99=float(batch_results[1]['latency_p99_ms']),
        latency_min=float(min(latencies)),
        latency_max=float(max(latencies)),
        throughput_fps=float(batch_results[1]['throughput_fps']),
        memory_mb=memory_mb,
        batch_results=batch_results
    )


# ============ ONNX RUNTIME BENCHMARK ============
def benchmark_onnx(
    model_path: Path,
    config: BenchmarkConfig
) -> Optional[BenchmarkResults]:
    """Benchmark ONNX model with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed, skipping")
        return None
    
    # Create session with optimizations
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = config.num_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(str(model_path), sess_options)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Replace dynamic dims
    input_shape = [1 if isinstance(d, str) else d for d in input_shape]
    
    print(f"Input: {input_name}, shape: {input_shape}")
    print(f"Threads: {config.num_threads}")
    
    model_size = model_path.stat().st_size
    n_params = model_size // 4  # Estimate
    
    print("\nBenchmarking batch_size=1...")
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(config.warmup_iterations):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    latencies = []
    for _ in range(config.benchmark_iterations):
        start = time.perf_counter_ns()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)
    
    latencies = np.array(latencies)
    
    batch_results = {
        1: {
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000 / np.mean(latencies)
        }
    }
    
    print(f"  Latency: {batch_results[1]['latency_mean_ms']:.2f} ± "
          f"{batch_results[1]['latency_std_ms']:.2f} ms")
    
    return BenchmarkResults(
        model_name=model_path.stem,
        model_format='onnx',
        model_size_kb=model_size / 1024,
        params=n_params,
        latency_mean=float(batch_results[1]['latency_mean_ms']),
        latency_std=float(batch_results[1]['latency_std_ms']),
        latency_p50=float(batch_results[1]['latency_p50_ms']),
        latency_p95=float(batch_results[1]['latency_p95_ms']),
        latency_p99=float(batch_results[1]['latency_p99_ms']),
        latency_min=float(min(latencies)),
        latency_max=float(max(latencies)),
        throughput_fps=float(batch_results[1]['throughput_fps']),
        memory_mb=model_size / (1024 * 1024) + 5,
        batch_results=batch_results
    )


# ============ POWER ESTIMATION ============
def estimate_power_consumption(
    model_size_kb: float,
    latency_ms: float,
    throughput_fps: float
) -> Dict:
    """
    Estimate power consumption on Raspberry Pi 4.
    
    Based on empirical measurements:
    - Pi 4 idle: ~2.5W
    - Pi 4 CPU stress: ~5.5W
    - Typical inference adds 0.5-2W depending on load
    """
    # Base power
    idle_power = 2.5  # W
    
    # Inference power (scales with throughput)
    # Higher throughput = more CPU usage = more power
    inference_power = min(2.0, throughput_fps / 500)  # Cap at 2W
    
    # Memory power (small models = less memory bandwidth)
    memory_power = min(0.5, model_size_kb / 1000)  # Cap at 0.5W
    
    total_power = idle_power + inference_power + memory_power
    
    # Energy per inference
    energy_per_inference_mj = total_power * latency_ms  # mJ
    
    # Battery life estimate (10000 mAh @ 5V = 50Wh)
    battery_capacity_wh = 50
    runtime_hours = battery_capacity_wh / total_power
    inferences_on_battery = runtime_hours * 3600 * throughput_fps
    
    return {
        'idle_power_w': idle_power,
        'inference_power_w': inference_power,
        'memory_power_w': memory_power,
        'total_power_w': total_power,
        'energy_per_inference_mj': energy_per_inference_mj,
        'battery_runtime_hours': runtime_hours,
        'inferences_on_battery': int(inferences_on_battery),
        'note': 'Estimates based on Raspberry Pi 4 typical power consumption'
    }


# ============ TARGET COMPLIANCE CHECK ============
def check_target_compliance(results: BenchmarkResults) -> Dict:
    """
    Check if results meet Pi 4 deployment targets.
    
    Targets:
    - Latency p50 ≤ 10ms
    - Model size ≤ 50KB (ideal), ≤ 200KB (acceptable)
    - Throughput ≥ 100 samples/sec
    """
    targets = {
        'latency_p50_ms': 10.0,
        'model_size_kb_ideal': 50.0,
        'model_size_kb_max': 200.0,
        'throughput_fps_min': 100.0
    }
    
    compliance = {
        'latency_ok': results.latency_p50 <= targets['latency_p50_ms'],
        'size_ideal': results.model_size_kb <= targets['model_size_kb_ideal'],
        'size_ok': results.model_size_kb <= targets['model_size_kb_max'],
        'throughput_ok': results.throughput_fps >= targets['throughput_fps_min']
    }
    
    compliance['overall'] = (
        compliance['latency_ok'] and 
        compliance['size_ok'] and 
        compliance['throughput_ok']
    )
    
    return {
        'targets': targets,
        'results': {
            'latency_p50_ms': results.latency_p50,
            'model_size_kb': results.model_size_kb,
            'throughput_fps': results.throughput_fps
        },
        'compliance': compliance
    }


# ============ MAIN CLI ============
def main():
    parser = argparse.ArgumentParser(description='Benchmark for Pi 4 Deployment')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'pytorch', 'tflite', 'onnx'],
                        help='Model format')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--iterations', type=int, default=500, help='Benchmark iterations')
    parser.add_argument('--full', action='store_true', help='Full benchmark suite')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Detect format
    if args.format == 'auto':
        suffix = model_path.suffix.lower()
        if suffix == '.pt' or suffix == '.pth':
            model_format = 'pytorch'
        elif suffix == '.tflite':
            model_format = 'tflite'
        elif suffix == '.onnx':
            model_format = 'onnx'
        else:
            print(f"Unknown model format: {suffix}")
            sys.exit(1)
    else:
        model_format = args.format
    
    config = BenchmarkConfig(
        num_threads=args.threads,
        benchmark_iterations=args.iterations
    )
    
    print(f"\n{'='*60}")
    print(f"RASPBERRY PI 4 BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Format: {model_format}")
    
    # Run benchmark
    if model_format == 'pytorch':
        # Default input shape for our models
        input_shape = (15, 65)  # window_size, n_features
        results = benchmark_pytorch(model_path, input_shape, config)
    elif model_format == 'tflite':
        results = benchmark_tflite(model_path, config)
    elif model_format == 'onnx':
        results = benchmark_onnx(model_path, config)
    else:
        print(f"Unsupported format: {model_format}")
        sys.exit(1)
    
    if results is None:
        print("Benchmark failed")
        sys.exit(1)
    
    # Print summary
    results.print_summary()
    
    # Power estimation
    power = estimate_power_consumption(
        results.model_size_kb,
        results.latency_mean,
        results.throughput_fps
    )
    
    print(f"\nPower Estimation:")
    print(f"  Total:     {power['total_power_w']:.2f} W")
    print(f"  Per infer: {power['energy_per_inference_mj']:.2f} mJ")
    print(f"  Battery:   {power['battery_runtime_hours']:.1f} hours (50Wh)")
    
    # Target compliance
    compliance = check_target_compliance(results)
    
    print(f"\n{'='*60}")
    print("TARGET COMPLIANCE")
    print(f"{'='*60}")
    for target, value in compliance['compliance'].items():
        status = '✅' if value else '❌'
        print(f"  {target}: {status}")
    
    overall = '✅ PASS' if compliance['compliance']['overall'] else '❌ FAIL'
    print(f"\nOverall: {overall}")
    
    # Save results
    if args.output or args.full:
        output_path = args.output or model_path.with_suffix('.benchmark.json')
        
        full_results = {
            'benchmark': results.to_dict(),
            'power_estimation': power,
            'target_compliance': compliance,
            'config': {
                'threads': config.num_threads,
                'iterations': config.benchmark_iterations,
                'warmup': config.warmup_iterations
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
