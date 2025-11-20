"""
System monitoring utilities for tracking resource usage during experiments.

Features:
- Process-specific CPU and memory monitoring
- Disk usage tracking
- System information snapshot
- Context manager for timing code blocks with automatic logging
- Optional integration with ExperimentLogger
- Optional integration with tqdm progress bars for timing context
- Resource sampling over time (with timestamps)
- Human-readable units for memory/disk/time
"""
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

def _bytes_to_readable(bytes_val: float) -> str:
    """Convert bytes to a human-readable string (e.g., '3.5 GB')."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"  # Fallback for extremely large values

def _seconds_to_readable(seconds: float) -> str:
    """Convert seconds to a human-readable string (e.g., '2h 30m 10s')."""
    if seconds < 0:
        return "N/A"
    intervals = [
        ('week', 604800),
        ('day', 86400),
        ('hour', 3600),
        ('minute', 60),
        ('second', 1)
    ]
    result = []
    for name, count in intervals:
        value = int(seconds // count)
        if value:
            s = 's' if value > 1 else ''
            result.append(f"{value}{name[0]}{s}")
            seconds -= value * count
    if not result:
        return "0s"
    return " ".join(result)

class SystemMonitor:
    """Monitors system resources (CPU, memory, disk) for the current process."""

    def __init__(self, logger=None):
        """
        Initialize system monitor for the current process.
        Captures baseline memory usage.

        Args:
            logger: Optional logger instance for logging metrics.
        """
        self.process = psutil.Process()
        self.logger = logger

        # Prime CPU percent to avoid first-call zero reading
        try:
            self.process.cpu_percent(None)
        except Exception as e:
            self._warn(f"Could not prime CPU percent: {e}")

        # Safe baseline memory capture
        bm = self.get_memory_usage()
        self.baseline_memory_info = bm if bm and 'rss_mb' in bm else {'rss_mb': 0.0}

    def _warn(self, msg: str):
        """Unified warning method using logger if available."""
        if self.logger:
            self.logger.warning(msg)
        else:
            print(f"Warning: {msg}")

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage for this process."""
        try:
            return self.process.cpu_percent(interval=0.1)
        except psutil.Error as e:
            self._warn(f"Could not get CPU usage: {e}")
            return 0.0

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for this process."""
        try:
            mem_info = self.process.memory_info()
            mem_percent = self.process.memory_percent()
            virtual_memory = psutil.virtual_memory()

            usage_dict = {
                'rss_bytes': mem_info.rss,
                'vms_bytes': mem_info.vms,
                'rss_mb': mem_info.rss / (1024 ** 2),
                'vms_mb': mem_info.vms / (1024 ** 2),
                'rss_gb': mem_info.rss / (1024 ** 3),
                'vms_gb': mem_info.vms / (1024 ** 3),
                'percent': mem_percent,
                'available_system_bytes': virtual_memory.available,
                'available_system_mb': virtual_memory.available / (1024 ** 2),
                'available_system_gb': virtual_memory.available / (1024 ** 3),
                'rss_readable': _bytes_to_readable(mem_info.rss),
                'vms_readable': _bytes_to_readable(mem_info.vms),
                'available_system_readable': _bytes_to_readable(virtual_memory.available),
            }
            return usage_dict
        except psutil.Error as e:
            self._warn(f"Could not get memory usage: {e}")
            return {}

    def get_memory_increase(self) -> float:
        """Return increase in RSS memory since initialization in MB."""
        try:
            curr = self.get_memory_usage()
            return curr.get('rss_mb', 0.0) - self.baseline_memory_info.get('rss_mb', 0.0)
        except psutil.Error as e:
            self._warn(f"Could not calculate memory increase: {e}")
            return 0.0

    def get_disk_usage(self, path: Union[str, Path] = '/') -> Dict[str, Any]:
        """Get disk usage for a specified path (supports str or Path)."""
        path_obj = Path(path)
        if not path_obj.exists():
            self._warn(f"Disk path does not exist: {path_obj}")
            return {}

        try:
            disk = psutil.disk_usage(str(path_obj))
            usage_dict = {
                'path': str(path_obj),
                'total_bytes': disk.total,
                'used_bytes': disk.used,
                'free_bytes': disk.free,
                'percent': disk.percent,
                'total_gb': disk.total / (1024 ** 3),
                'used_gb': disk.used / (1024 ** 3),
                'free_gb': disk.free / (1024 ** 3),
                'total_readable': _bytes_to_readable(disk.total),
                'used_readable': _bytes_to_readable(disk.used),
                'free_readable': _bytes_to_readable(disk.free),
            }
            return usage_dict
        except psutil.Error as e:
            self._warn(f"Could not get disk usage for {path_obj}: {e}")
            return {}

    def check_memory_available(self, required_mb: float) -> bool:
        """Check if the specified amount of memory is available."""
        try:
            mem_info = self.get_memory_usage()
            return mem_info.get('available_system_mb', 0) >= required_mb
        except psutil.Error as e:
            self._warn(f"Could not check available memory: {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get a snapshot of system info (CPU, memory, disk, boot time)."""
        try:
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            cpu_freq_info = psutil.cpu_freq()
            virtual_mem = psutil.virtual_memory()
            disk_root = psutil.disk_usage('/')
            boot_time_ts = psutil.boot_time()
        except psutil.Error as e:
            self._warn(f"Could not get system info: {e}")
            return {}

        info_dict = {
            'cpu_count_logical': cpu_count_logical,
            'cpu_count_physical': cpu_count_physical,
            'cpu_freq_current_mhz': cpu_freq_info.current if cpu_freq_info else None,
            'cpu_freq_max_mhz': cpu_freq_info.max if cpu_freq_info else None,
            'total_memory_bytes': virtual_mem.total,
            'total_memory_gb': virtual_mem.total / (1024 ** 3),
            'available_memory_bytes': virtual_mem.available,
            'available_memory_gb': virtual_mem.available / (1024 ** 3),
            'used_memory_bytes': virtual_mem.used,
            'used_memory_gb': virtual_mem.used / (1024 ** 3),
            'memory_percent': virtual_mem.percent,
            'disk_total_bytes': disk_root.total,
            'disk_total_gb': disk_root.total / (1024 ** 3),
            'disk_free_bytes': disk_root.free,
            'disk_free_gb': disk_root.free / (1024 ** 3),
            'disk_used_bytes': disk_root.used,
            'disk_used_gb': disk_root.used / (1024 ** 3),
            'disk_percent': disk_root.percent,
            'total_memory_readable': _bytes_to_readable(virtual_mem.total),
            'available_memory_readable': _bytes_to_readable(virtual_mem.available),
            'used_memory_readable': _bytes_to_readable(virtual_mem.used),
            'disk_total_readable': _bytes_to_readable(disk_root.total),
            'disk_free_readable': _bytes_to_readable(disk_root.free),
            'disk_used_readable': _bytes_to_readable(disk_root.used),
        }

        if boot_time_ts and boot_time_ts > 0:
            try:
                dt = datetime.fromtimestamp(boot_time_ts)
                info_dict['boot_time'] = dt.strftime("%Y-%m-%d %H:%M:%S")
                info_dict['boot_time_timestamp'] = boot_time_ts
            except Exception as e:
                self._warn(f"Could not format boot time: {e}")
                info_dict['boot_time'] = "N/A"
                info_dict['boot_time_timestamp'] = boot_time_ts
        else:
            info_dict['boot_time'] = "N/A"
            info_dict['boot_time_timestamp'] = boot_time_ts

        return info_dict

    def sample_resources(self, duration: float, interval: float = 1.0) -> Dict[str, Union[List[float], float]]:
        """
        Sample CPU and memory usage over time with timestamps.

        Args:
            duration: Total sampling duration in seconds.
            interval: Time between samples in seconds.

        Returns:
            Dict with list-valued keys 'timestamps', 'cpu_percent', 'memory_mb' and
            float-valued keys 'sample_interval_s', 'sample_duration_s'.
        """
        if duration <= 0 or interval <= 0:
            raise ValueError("duration and interval must be positive")

        cpu_samples, mem_samples, timestamps = [], [], []
        start = time.time()
        next_sample = start
        end = start + duration

        while next_sample <= end:
            timestamps.append(next_sample)
            cpu_samples.append(self.get_cpu_usage())
            mem_samples.append(self.get_memory_usage().get('rss_mb', 0.0))
            next_sample += interval
            time.sleep(max(0, next_sample - time.time()))

        return {
            'timestamps': timestamps,
            'cpu_percent': cpu_samples,
            'memory_mb': mem_samples,
            'sample_interval_s': interval,
            'sample_duration_s': duration
        }


class TimingContext:
    """Context manager for timing code blocks with optional logging or tqdm progress bar."""

    def __init__(self, name: str = "Operation", logger=None, pbar=None, verbose: bool = True):
        self.name = name
        self.logger = logger
        self.pbar = pbar
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        self.elapsed_readable: Optional[str] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.verbose and self.logger is None and self.pbar is None:
            print(f"Starting {self.name}...")
        elif self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed = self.end_time - self.start_time
            self.elapsed_readable = _seconds_to_readable(self.elapsed)

            msg = f"{self.name} completed in {self.elapsed:.2f} seconds ({self.elapsed_readable})"
            if self.logger:
                self.logger.info(msg)
            elif self.pbar:
                self.pbar.set_postfix_str(msg)
            elif self.verbose:
                print(msg)

        return False  # Do not suppress exceptions
