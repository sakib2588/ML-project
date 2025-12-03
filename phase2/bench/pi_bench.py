#!/usr/bin/env python
"""
Phase 2 - Pi Benchmarking Harness
==================================

Comprehensive benchmarking on Raspberry Pi 4:
- Latency (p50, p95, p99)
- Energy consumption (with INA219)
- Memory footprint
- Thermal monitoring

Usage (on Pi):
    # Basic latency benchmark
    python pi_bench.py --model model.tflite --runs 1000
    
    # With energy measurement (requires INA219)
    python pi_bench.py --model model.tflite --energy
    
    # Full benchmark suite
    python pi_bench.py --model model.tflite --full
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess

import numpy as np

# ============ SYSTEM UTILITIES ============
def is_raspberry_pi() -> bool:
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False

def get_cpu_temp() -> float:
    """Get CPU temperature in Celsius."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 0.0

def get_cpu_freq() -> float:
    """Get current CPU frequency in MHz."""
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 0.0

def get_cpu_governor() -> str:
    """Get CPU governor."""
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            return f.read().strip()
    except:
        return "unknown"

def set_cpu_governor(governor: str = 'performance'):
    """Set CPU governor (requires sudo)."""
    try:
        subprocess.run(
            ['sudo', 'cpufreq-set', '-g', governor],
            check=True, capture_output=True
        )
        return True
    except:
        return False

def get_memory_info() -> Dict[str, float]:
    """Get memory information."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_percent': mem.percent
        }
    except:
        return {'total_gb': 0, 'available_gb': 0, 'used_percent': 0}

def get_process_memory(pid: int) -> float:
    """Get process RSS memory in MB."""
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024  # KB to MB
    except:
        pass
    return 0.0


# ============ TFLITE INFERENCE ============
class TFLiteInference:
    """TFLite inference wrapper with XNNPACK support."""
    
    def __init__(self, model_path: str, use_xnnpack: bool = True, num_threads: int = 4):
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore[import-not-found]
        except ImportError:
            import tensorflow.lite as tflite  # type: ignore[import-untyped]
        
        self.model_path = model_path
        self.use_xnnpack = use_xnnpack
        
        # Create interpreter
        if use_xnnpack:
            try:
                # Try loading XNNPACK delegate
                xnnpack_delegate = tflite.load_delegate('libXNNPACK.so')  # type: ignore[attr-defined]
                self.interpreter = tflite.Interpreter(  # type: ignore[attr-defined]
                    model_path=model_path,
                    experimental_delegates=[xnnpack_delegate],
                    num_threads=num_threads
                )
                self.delegate = 'xnnpack'
            except:
                self.interpreter = tflite.Interpreter(  # type: ignore[attr-defined]
                    model_path=model_path,
                    num_threads=num_threads
                )
                self.delegate = 'cpu'
        else:
            self.interpreter = tflite.Interpreter(  # type: ignore[attr-defined]
                model_path=model_path,
                num_threads=num_threads
            )
            self.delegate = 'cpu'
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run single inference."""
        x = x.astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])
    
    def get_model_size(self) -> int:
        """Get model file size in bytes."""
        return os.path.getsize(self.model_path)


# ============ ENERGY MEASUREMENT ============
class EnergyMonitor:
    """Energy monitoring using INA219 sensor."""
    
    def __init__(self, i2c_address: int = 0x40, sample_rate_hz: int = 100):
        self.available = False
        self.sample_rate = sample_rate_hz
        self.samples: list = []
        self.ina = None  # type: ignore[var-annotated]
        
        try:
            from ina219 import INA219, DeviceRangeError  # type: ignore[import-not-found]
            self.ina = INA219(shunt_ohms=0.1, address=i2c_address)
            self.ina.configure()
            self.available = True
        except ImportError:
            print("INA219 library not available. Energy measurement disabled.")
        except Exception as e:
            print(f"INA219 initialization failed: {e}")
    
    def measure_power(self) -> float:
        """Get current power consumption in mW."""
        if not self.available or self.ina is None:
            return 0.0
        return self.ina.power()  # type: ignore[no-any-return]
    
    def measure_idle_power(self, duration_s: float = 60.0) -> Tuple[float, float]:
        """Measure idle power for baseline."""
        if not self.available:
            return 0.0, 0.0
        
        samples = []
        interval = 1.0 / self.sample_rate
        end_time = time.time() + duration_s
        
        while time.time() < end_time:
            samples.append(self.measure_power())
            time.sleep(interval)
        
        return float(np.mean(samples)), float(np.std(samples))
    
    def start_measurement(self):
        """Start continuous measurement."""
        self.samples = []
        self.start_time = time.time()
    
    def record_sample(self):
        """Record a power sample."""
        if self.available:
            self.samples.append((time.time(), self.measure_power()))
    
    def stop_measurement(self) -> Dict:
        """Stop measurement and compute statistics."""
        if not self.samples:
            return {'available': False}
        
        times, powers = zip(*self.samples)
        powers = np.array(powers)
        
        # Compute energy using trapezoidal integration
        dt = np.diff(times)
        energy_mj = np.sum((powers[:-1] + powers[1:]) / 2 * dt)  # mW * s = mJ
        
        return {
            'available': True,
            'mean_power_mw': np.mean(powers),
            'peak_power_mw': np.max(powers),
            'total_energy_mj': energy_mj,
            'duration_s': times[-1] - times[0],
            'n_samples': len(powers)
        }


# ============ BENCHMARK RUNNER ============
class BenchmarkRunner:
    """Comprehensive benchmark runner."""
    
    def __init__(
        self,
        model_path: str,
        warmup_runs: int = 200,
        benchmark_runs: int = 1000,
        batch_size: int = 1,
        use_xnnpack: bool = True
    ):
        self.model_path = model_path
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.batch_size = batch_size
        
        # Initialize inference engine
        self.engine = TFLiteInference(model_path, use_xnnpack)
        
        # Generate dummy data matching input shape
        self.input_shape = self.engine.input_shape[1:]  # Remove batch dim
        
        # Energy monitor
        self.energy_monitor = EnergyMonitor()
        
        # Results
        self.results = {}
    
    def generate_test_data(self, n_samples: int) -> np.ndarray:
        """Generate random test data."""
        shape = (n_samples,) + tuple(self.input_shape)
        return np.random.randn(*shape).astype(np.float32)
    
    def run_warmup(self):
        """Run warmup inferences."""
        print(f"Running {self.warmup_runs} warmup inferences...")
        data = self.generate_test_data(1)
        
        for _ in range(self.warmup_runs):
            _ = self.engine.predict(data)
    
    def run_latency_benchmark(self) -> Dict:
        """Benchmark inference latency."""
        print(f"Running {self.benchmark_runs} latency measurements...")
        
        data = self.generate_test_data(1)
        latencies = []
        
        # Run in batches of 100 for stability
        batch_count = self.benchmark_runs // 100
        
        for batch_idx in range(batch_count):
            batch_latencies = []
            
            for _ in range(100):
                start = time.perf_counter_ns()
                _ = self.engine.predict(data)
                end = time.perf_counter_ns()
                
                batch_latencies.append((end - start) / 1e6)  # ns to ms
            
            latencies.extend(batch_latencies)
            
            # Check temperature
            temp = get_cpu_temp()
            if temp > 75:
                print(f"  Warning: CPU temp {temp:.1f}°C, cooling down...")
                time.sleep(5)
        
        latencies = np.array(latencies)
        
        return {
            'n_runs': len(latencies),
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p75_ms': np.percentile(latencies, 75),
            'p90_ms': np.percentile(latencies, 90),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'raw_latencies': latencies.tolist()
        }
    
    def run_energy_benchmark(self, n_inferences: int = 100) -> Dict:
        """Benchmark energy consumption."""
        if not self.energy_monitor.available:
            return {'available': False, 'error': 'INA219 not available'}
        
        print(f"Measuring idle power (60s)...")
        idle_mean, idle_std = self.energy_monitor.measure_idle_power(60.0)
        
        print(f"Running {n_inferences} inferences with energy measurement...")
        data = self.generate_test_data(1)
        
        self.energy_monitor.start_measurement()
        
        for _ in range(n_inferences):
            _ = self.engine.predict(data)
            self.energy_monitor.record_sample()
        
        energy_stats = self.energy_monitor.stop_measurement()
        
        # Compute per-inference energy
        if energy_stats['available']:
            net_power = energy_stats['mean_power_mw'] - idle_mean
            energy_per_inference = net_power * (energy_stats['duration_s'] / n_inferences)
            
            return {
                'available': True,
                'idle_power_mw': idle_mean,
                'idle_power_std_mw': idle_std,
                'inference_power_mw': energy_stats['mean_power_mw'],
                'net_power_mw': net_power,
                'total_energy_mj': energy_stats['total_energy_mj'],
                'energy_per_inference_mj': energy_per_inference,
                'n_inferences': n_inferences
            }
        
        return energy_stats
    
    def run_memory_benchmark(self, n_inferences: int = 100) -> Dict:
        """Benchmark memory usage."""
        import os
        pid = os.getpid()
        
        print(f"Measuring memory usage...")
        
        # Initial memory
        initial_mem = get_process_memory(pid)
        
        # Run inferences and sample memory
        data = self.generate_test_data(1)
        mem_samples = []
        
        for i in range(n_inferences):
            _ = self.engine.predict(data)
            if i % 10 == 0:
                mem_samples.append(get_process_memory(pid))
        
        mem_samples = np.array(mem_samples)
        
        return {
            'initial_mb': initial_mem,
            'peak_mb': np.max(mem_samples),
            'mean_mb': np.mean(mem_samples),
            'model_size_kb': self.engine.get_model_size() / 1024
        }
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print(f"\n{'='*60}")
        print("FULL BENCHMARK SUITE")
        print(f"{'='*60}")
        
        # System info
        system_info = {
            'is_pi': is_raspberry_pi(),
            'cpu_governor': get_cpu_governor(),
            'cpu_freq_mhz': get_cpu_freq(),
            'cpu_temp_c': get_cpu_temp(),
            'memory': get_memory_info(),
            'model_path': self.model_path,
            'delegate': self.engine.delegate,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nSystem Info:")
        print(f"  Raspberry Pi: {system_info['is_pi']}")
        print(f"  CPU Governor: {system_info['cpu_governor']}")
        print(f"  CPU Freq: {system_info['cpu_freq_mhz']:.0f} MHz")
        print(f"  CPU Temp: {system_info['cpu_temp_c']:.1f}°C")
        print(f"  Delegate: {system_info['delegate']}")
        
        # Warmup
        self.run_warmup()
        
        # Latency benchmark
        print("\n--- Latency Benchmark ---")
        latency_results = self.run_latency_benchmark()
        
        print(f"  Mean: {latency_results['mean_ms']:.3f} ms")
        print(f"  P50:  {latency_results['p50_ms']:.3f} ms")
        print(f"  P95:  {latency_results['p95_ms']:.3f} ms")
        print(f"  P99:  {latency_results['p99_ms']:.3f} ms")
        
        # Memory benchmark
        print("\n--- Memory Benchmark ---")
        memory_results = self.run_memory_benchmark()
        
        print(f"  Peak RSS: {memory_results['peak_mb']:.1f} MB")
        print(f"  Model size: {memory_results['model_size_kb']:.1f} KB")
        
        # Energy benchmark (if available)
        print("\n--- Energy Benchmark ---")
        energy_results = self.run_energy_benchmark()
        
        if energy_results.get('available', False):
            print(f"  Idle power: {energy_results['idle_power_mw']:.1f} mW")
            print(f"  Inference power: {energy_results['inference_power_mw']:.1f} mW")
            print(f"  Energy/inference: {energy_results['energy_per_inference_mj']:.3f} mJ")
        else:
            print("  Energy measurement not available")
        
        # Final temperature
        final_temp = get_cpu_temp()
        print(f"\n  Final CPU temp: {final_temp:.1f}°C")
        
        # Compile results
        self.results = {
            'system': system_info,
            'latency': latency_results,
            'memory': memory_results,
            'energy': energy_results,
            'final_temp_c': final_temp
        }
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        # Remove raw latencies for compact output
        results_compact = self.results.copy()
        if 'latency' in results_compact and 'raw_latencies' in results_compact['latency']:
            del results_compact['latency']['raw_latencies']
        
        with open(output_path, 'w') as f:
            json.dump(results_compact, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def save_raw_latencies(self, output_path: str):
        """Save raw latencies to CSV."""
        if 'latency' in self.results and 'raw_latencies' in self.results['latency']:
            latencies = self.results['latency']['raw_latencies']
            with open(output_path, 'w') as f:
                f.write("run_id,latency_ms\n")
                for i, lat in enumerate(latencies):
                    f.write(f"{i},{lat}\n")
            print(f"Raw latencies saved to: {output_path}")


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Pi Benchmark Harness')
    parser.add_argument('--model', type=str, required=True, help='Path to TFLite model')
    parser.add_argument('--runs', type=int, default=1000, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=200, help='Number of warmup runs')
    parser.add_argument('--output', type=str, default='bench_results.json', help='Output file')
    parser.add_argument('--xnnpack', action='store_true', default=True, help='Use XNNPACK')
    parser.add_argument('--no-xnnpack', dest='xnnpack', action='store_false')
    parser.add_argument('--energy', action='store_true', help='Run energy benchmark')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite')
    parser.add_argument('--set-governor', type=str, help='Set CPU governor before benchmark')
    
    args = parser.parse_args()
    
    # Set CPU governor if requested
    if args.set_governor:
        print(f"Setting CPU governor to: {args.set_governor}")
        set_cpu_governor(args.set_governor)
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        model_path=args.model,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        use_xnnpack=args.xnnpack
    )
    
    # Run benchmarks
    if args.full:
        results = runner.run_full_benchmark()
    else:
        print(f"\n{'='*60}")
        print("LATENCY BENCHMARK")
        print(f"{'='*60}")
        
        runner.run_warmup()
        latency_results = runner.run_latency_benchmark()
        
        results = {
            'system': {
                'is_pi': is_raspberry_pi(),
                'cpu_temp_c': get_cpu_temp(),
                'delegate': runner.engine.delegate
            },
            'latency': latency_results
        }
        
        runner.results = results
        
        print(f"\nResults:")
        print(f"  Mean:  {latency_results['mean_ms']:.3f} ms")
        print(f"  P50:   {latency_results['p50_ms']:.3f} ms")
        print(f"  P95:   {latency_results['p95_ms']:.3f} ms")
        print(f"  P99:   {latency_results['p99_ms']:.3f} ms")
    
    # Save results
    runner.save_results(args.output)
    
    # Save raw latencies
    raw_output = args.output.replace('.json', '_raw.csv')
    runner.save_raw_latencies(raw_output)
    
    # Check targets
    print(f"\n{'='*60}")
    print("TARGET CHECKS")
    print(f"{'='*60}")
    
    p50 = results['latency']['p50_ms']
    p95 = results['latency']['p95_ms']
    
    checks = [
        (f"P50 ≤ 10 ms", p50 <= 10.0, p50),
        (f"P50 ≤ 7 ms (ideal)", p50 <= 7.0, p50),
        (f"P95 ≤ 40 ms", p95 <= 40.0, p95),
    ]
    
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status} ({value:.2f} ms)")


if __name__ == '__main__':
    main()
