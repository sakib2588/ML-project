"""
Utilities for calculating model metrics including FLOPs, parameters, inference time, and memory usage.

Research-grade goals / fixes:
- Robust handling of optional dependencies (thop, torch.profiler)
- Avoid unbound variables (orig_dev) and unsafe tuple unpacking
- Proper device handling (accepts str or torch.device)
- CUDA synchronization for correct timing on GPU
- JSON-serializable return values
- Clear logging and graceful fallbacks
"""
from __future__ import annotations

import time
import math
import logging
from typing import Tuple, Dict, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Optional dependency: thop for FLOPs counting
try:
    from thop import profile as thop_profile  # type: ignore
    from thop import clever_format as thop_clever_format  # type: ignore
    THOP_AVAILABLE = True
except Exception:
    thop_profile = None  # type: ignore
    thop_clever_format = None  # type: ignore
    THOP_AVAILABLE = False
    logger.debug("thop not available; FLOPs counting disabled.")


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Compute total size of model parameters + buffers in megabytes.
    """
    param_bytes = 0
    for p in model.parameters():
        param_bytes += p.nelement() * p.element_size()

    buffer_bytes = 0
    for b in model.buffers():
        buffer_bytes += b.nelement() * b.element_size()

    size_mb = float((param_bytes + buffer_bytes) / (1024 ** 2))
    return size_mb


def _ensure_device(device: Optional[torch.device | str]) -> torch.device:
    """Normalize device argument to torch.device."""
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def count_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device | str] = "cpu",
) -> Dict[str, Any]:
    """
    Count FLOPs and params using thop where available.

    Returns a dict with keys:
      - flops (int)
      - flops_str (readable string)
      - params (int)
      - params_str (readable string)

    If thop is unavailable or profiling fails, returns params and zeros/NA for flops.
    """
    dev = _ensure_device(device)
    orig_dev = None
    try:
        # move model to device safely, remembering original device if possible
        try:
            orig_dev = next(model.parameters()).device
        except StopIteration:
            orig_dev = torch.device("cpu")

        model = model.to(dev)
    except Exception:
        # If moving fails, continue and attempt profiling on current device
        dev = getattr(model, "device", torch.device("cpu"))

    params = count_parameters(model)

    if not THOP_AVAILABLE or thop_profile is None:
        return {
            "flops": 0,
            "flops_str": "N/A (thop not installed)",
            "params": params,
            "params_str": f"{params:,}",
        }

    # create dummy input
    dummy = torch.randn(input_shape, device=dev)
    try:
        # thop.profile may return (macs, params) or other shapes depending on version.
        res = thop_profile(model, inputs=(dummy,), verbose=False)  # type: ignore[arg-type]
        # be defensive: handle tuple-like and object-like returns
        macs = None
        thop_params = None
        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) >= 2:
                macs = int(res[0]) if res[0] is not None else 0
                thop_params = int(res[1]) if res[1] is not None else params
        elif hasattr(res, "flops") and hasattr(res, "params"):
            macs = int(getattr(res, "flops", 0))
            thop_params = int(getattr(res, "params", params))
        else:
            # best-effort: try indexing
            try:
                macs = int(res[0])
                thop_params = int(res[1])
            except Exception:
                macs = 0
                thop_params = params

        # convert MACs to FLOPs (approx): FLOPs ~ 2 * MACs
        flops = int(macs * 2) if macs is not None else 0

        # format strings if clever_format available
        if thop_clever_format is not None:
            try:
                flops_str, params_str = thop_clever_format([flops, thop_params], "%.3f")
            except Exception:
                flops_str = f"{flops:,}"
                params_str = f"{thop_params:,}"
        else:
            flops_str = f"{flops:,}"
            params_str = f"{thop_params:,}"

        return {
            "flops": flops,
            "flops_str": flops_str,
            "params": int(thop_params) if thop_params is not None else params,
            "params_str": params_str,
        }

    except Exception as e:
        logger.warning(f"thop profiling failed: {e}")
        return {
            "flops": 0,
            "flops_str": f"Error: {e}",
            "params": params,
            "params_str": f"{params:,}",
        }
    finally:
        # restore model device if possible
        try:
            if orig_dev is not None:
                model.to(orig_dev)
        except Exception:
            pass


def measure_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device | str] = "cpu",
) -> Dict[str, float]:
    """
    Measure forward pass latency (milliseconds) for a fixed input shape.

    - input_shape includes batch dimension, e.g., (1, window_len, n_features)
    - Uses torch.cuda.synchronize() when on CUDA for accurate timing.
    """
    dev = _ensure_device(device)
    orig_dev = None
    try:
        orig_dev = next(model.parameters()).device
    except Exception:
        orig_dev = torch.device("cpu")

    model = model.to(dev)
    model.eval()

    dummy = torch.randn(input_shape, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(int(warmup_runs)):
            out = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    times_ms = []
    with torch.no_grad():
        for _ in range(int(num_runs)):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms, dtype=float)
    stats = {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "num_runs": int(len(arr)),
    }

    # restore device if possible
    try:
        model.to(orig_dev)
    except Exception:
        pass

    return stats


def calculate_throughput(
    model: torch.nn.Module,
    single_input_shape: Tuple[int, ...],
    batch_size: int = 32,
    num_batches: int = 10,
    device: Optional[torch.device | str] = "cpu",
) -> Dict[str, Any]:
    """
    Measure throughput (samples/sec) for a given batch size.

    - single_input_shape is the shape of one sample (without batch dim), e.g., (window_len, n_features)
    """
    dev = _ensure_device(device)
    orig_dev = None
    try:
        orig_dev = next(model.parameters()).device
    except Exception:
        orig_dev = torch.device("cpu")

    model = model.to(dev)
    model.eval()

    batch_shape = (int(batch_size),) + tuple(int(x) for x in single_input_shape)
    dummy = torch.randn(batch_shape, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    total_samples = batch_size * int(num_batches)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(num_batches)):
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-12)
    throughput = float(total_samples / elapsed)

    # restore device
    try:
        model.to(orig_dev)
    except Exception:
        pass

    return {"throughput_samples_per_sec": throughput, "total_samples": total_samples, "elapsed_s": float(elapsed)}


def profile_memory_usage(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device | str] = "cpu",
    use_torch_profiler: bool = False,
) -> Dict[str, Any]:
    """
    Profile memory usage (best-effort).

    - On CUDA: uses torch.cuda.* peak memory stats.
    - Optionally can attempt torch.profiler usage (only when available and on CUDA).
    - On CPU: returns rough estimation based on model & input sizes.
    """
    dev = _ensure_device(device)
    orig_dev = None
    try:
        orig_dev = next(model.parameters()).device
    except Exception:
        orig_dev = torch.device("cpu")

    model = model.to(dev)
    model.eval()

    # For CUDA devices, use max_memory_allocated/reserved
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        dummy = torch.randn(input_shape, device=dev)
        with torch.no_grad():
            _ = model(dummy)
            torch.cuda.synchronize(dev)

        peak_alloc = torch.cuda.max_memory_allocated(dev)
        peak_reserved = torch.cuda.max_memory_reserved(dev)

        return {
            "peak_allocated_bytes": int(peak_alloc),
            "peak_allocated_mb": float(peak_alloc) / (1024 ** 2),
            "peak_reserved_bytes": int(peak_reserved),
            "peak_reserved_mb": float(peak_reserved) / (1024 ** 2),
        }

    # CPU estimation (rough)
    model_size_mb = get_model_size_mb(model)
    input_elems = int(np.prod(input_shape))
    input_size_mb = float(input_elems * 4) / (1024 ** 2)  # float32
    estimated_total_mb = model_size_mb + input_size_mb

    return {
        "estimated_total_mb": estimated_total_mb,
        "model_size_mb": model_size_mb,
        "input_size_mb": input_size_mb,
    }


def profile_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device | str] = "cpu",
) -> Dict[str, Any]:
    """
    Composite profiling that returns:
      - parameters, params_str
      - flops, flops_str (if available)
      - model_size_mb
      - inference_time stats (mean_ms, std_ms, ...)
      - throughput_samples_per_sec
    """
    dev = _ensure_device(device)
    results: Dict[str, Any] = {}

    # parameters & model size
    params = count_parameters(model)
    results["parameters"] = int(params)
    results["model_size_mb"] = float(get_model_size_mb(model))

    # FLOPs (thop)
    try:
        flops_info = count_flops(model, input_shape, device=dev)
        results.update(flops_info)
    except Exception as e:
        logger.warning(f"Failed to compute FLOPs: {e}")
        results.update({"flops": 0, "flops_str": "Error", "params": params, "params_str": f"{params:,}"})

    # Inference timing
    try:
        timing = measure_inference_time(model, input_shape, num_runs=50, warmup_runs=10, device=dev)
        results["inference_time_ms"] = timing
    except Exception as e:
        logger.warning(f"Inference timing failed: {e}")
        results["inference_time_ms"] = {}

    # Throughput: remove batch dim for single_input_shape
    try:
        if len(input_shape) >= 1:
            batch_for_timing = int(input_shape[0])
            single_input = tuple(int(x) for x in input_shape[1:]) if len(input_shape) > 1 else ()
            # if single_input is empty (scalar), still allow throughput measurement with shape ()
            tp = calculate_throughput(model, single_input, batch_size=batch_for_timing, num_batches=10, device=dev)
            results.update(tp)
    except Exception as e:
        logger.warning(f"Throughput measurement failed: {e}")

    # Return JSON-friendly types
    # Ensure all ints/floats are plain Python types
    for k, v in list(results.items()):
        if isinstance(v, np.generic):
            results[k] = v.item()
        elif isinstance(v, (np.ndarray,)):
            results[k] = v.tolist()

    return results


def run_full_profile(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device | str] = "cpu",
    profile_memory: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper that runs profile_model and optionally memory profiling.
    """
    results = profile_model(model, input_shape, device=device)
    if profile_memory:
        try:
            mem = profile_memory_usage(model, input_shape, device=device)
            results.update(mem)
        except Exception as e:
            logger.warning(f"Memory profiling failed: {e}")
    return results
