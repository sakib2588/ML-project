"""
Logging and Metrics utilities for experiment tracking and monitoring.

Features:
- Tqdm-safe console handler (prevents progress bar corruption)
- Rotating file handler with UTF-8 encoding
- Context manager support (__enter__/__exit__) and safe destructor
- Safe level handling (falls back to INFO on bad input)
- Metadata saving (JSON or YAML)
- add_handler/add_sink hooks for future sinks (TensorBoard, MLflow, etc.)
- get_log_file() helper to return Path to the active log file
- MetricLogger for tracking numerical metrics with JSON and CSV export
- Atomic saves for metrics to prevent corruption
- Deterministic CSV schema with 'epoch' and 'step' first
- Autosave and reset functionality for metrics
- Convenience function to get both loggers.

Caveats:
- RotatingFileHandler is NOT process-safe. For multi-process training
  consider per-worker logs or a process-safe handler such as those provided
  by third-party libraries (e.g., concurrent-log-handler) or central sinks
  (MLflow/ELK/TensorBoard).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Callable, Any, Dict, List, Union, Tuple
from tqdm import tqdm
import json
import csv
import tempfile
import os

# Optional YAML support
try:
    import yaml  # type: ignore
    # Annotate for type checkers that yaml is a dynamic module providing safe_dump, etc.
    yaml: Any = yaml  # type: ignore
    HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    HAS_YAML = False


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that routes logs through tqdm.write to avoid
    breaking progress bars.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Build the message robustly, with fallbacks to avoid swallowing errors silently.
        try:
            msg = self.format(record)
        except Exception:
            try:
                msg = record.getMessage()
            except Exception:
                msg = "<log formatting error>"

        try:
            # Use tqdm.write so the message doesn't interfere with active progress bars
            tqdm.write(msg)
        except Exception:
            # As a last-resort fallback, print to stderr
            try:
                print(msg, file=sys.stderr)
            except Exception:
                # If even printing fails, there's nothing more we can do safely
                pass


class ExperimentLogger:
    """
    ExperimentLogger manages structured logging for experiments.

    Usage:
        with ExperimentLogger("exp_name", Path("logs")) as logger:
            logger.info("Started")
            pbar = logger.progress_bar(range(10))
            for i in pbar:
                ...
            logger.log_metadata(config_dict)

    Note on multi-processing:
        RotatingFileHandler is not process-safe. If using multiprocessing workers,
        either create per-worker log files, use a process-safe handler, or centralize logs.
    """

    def __init__(
        self,
        name: str,
        log_dir: Path,
        level: str = "INFO",
        console_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        json_lines: bool = False,
    ) -> None:
        """
        Initialize ExperimentLogger.

        Args:
            name: Name for the logger (used in log filename and logger.name).
            log_dir: Directory path where log files and metadata will be stored.
            level: Logging level as string, e.g., "INFO", "DEBUG". Falls back to INFO.
            console_output: If True, enable tqdm-safe console logging.
            max_bytes: Max bytes before rotating the log file.
            backup_count: Number of rotated backups to keep.
            json_lines: If True, write file logs as JSON-lines when adding structured handlers later.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_lines = bool(json_lines)

        # Safe level handling (fallback to INFO)
        self.level_value = getattr(logging, level.upper(), logging.INFO)

        # Create or get logger and configure safely
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level_value)
        self.logger.propagate = False  # prevent double logging to root handlers

        # Remove and close any existing handlers attached to this logger
        for h in list(self.logger.handlers):
            try:
                self.logger.removeHandler(h)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass

        # Prepare file handler (RotatingFileHandler with UTF-8 encoding)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # record everything to file

        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
        self._console_handler: Optional[logging.Handler] = None

        # Console handler via TqdmLoggingHandler to keep tqdm intact
        if console_output:
            console_handler = TqdmLoggingHandler()
            console_handler.setLevel(self.level_value)
            console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                              datefmt="%H:%M:%S")
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            self._console_handler = console_handler

        # Expose the log_file as a Path
        self.log_file = Path(log_file)

    # -------------------------
    # Context manager & cleanup
    # -------------------------
    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # flush and close handlers
        self.close()

    def __del__(self) -> None:
        # Guard destructor to avoid errors during interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    # -------------------------
    # Basic logging wrappers
    # -------------------------
    def debug(self, message: str, *args, **kwargs) -> None:
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self.logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """
        Log an exception with traceback; use inside except blocks.
        """
        self.logger.exception(message, *args, **kwargs)

    # -------------------------
    # Utility methods
    # -------------------------
    def get_log_file(self) -> Path:
        """Return Path to the active log file."""
        return self.log_file

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Attach an additional logging handler (e.g., TensorBoard or custom sink).

        The caller is responsible for formatting/level on the handler.
        """
        self.logger.addHandler(handler)

    # alias for readability
    add_sink = add_handler

    def close(self) -> None:
        """
        Close and remove all handlers attached to this logger.
        Call at the end of runs or in tear-downs to free file descriptors.
        """
        for h in list(self.logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            try:
                self.logger.removeHandler(h)
            except Exception:
                pass

    # -------------------------
    # Features: metadata & sections
    # -------------------------
    def log_metadata(self,
                     metadata: dict,
                     filename: Optional[str] = None,
                     format: Literal["json", "yaml"] = "json") -> Path:
        """
        Save experiment metadata to a file for reproducibility.

        Args:
            metadata: Mapping to save (configs, hyperparams, notes).
            filename: Optional filename. If None, defaults to 'experiment_metadata.{ext}'.
            format: 'json' or 'yaml'. YAML requires PyYAML installed.

        Returns:
            Path to saved metadata file.
        """
        if filename is None:
            ext = "yaml" if format == "yaml" else "json"
            filename = f"experiment_metadata.{ext}"

        filepath = Path(self.log_file).parent / filename

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
        elif format == "yaml":
            if not HAS_YAML:
                raise ImportError("PyYAML is required to save metadata in YAML format.")
            with open(filepath, "w", encoding="utf-8") as f:
                # safe_dump + default_flow_style=False produces readable YAML
                yaml.safe_dump(metadata, f, default_flow_style=False)
        else:
            raise ValueError("format must be 'json' or 'yaml'")

        # log where metadata was saved
        self.info(f"Metadata saved to {filepath}")
        return filepath

    def section(self, title: str) -> None:
        """
        Log a section header. Includes separator lines for readability.
        """
        separator = "=" * 70
        self.info(separator)
        self.info(title)
        self.info(separator)

    def progress_bar(self, iterable, desc: str = "", total: Optional[int] = None):
        """
        Convenience wrapper for tqdm progress bars. Use `.update()` or iterate normally.

        Returns:
            tqdm iterator over the iterable
        """
        return tqdm(iterable, desc=desc, total=total, ncols=100)

    # -------------------------
    # Optional structured logging helper
    # -------------------------
    def info_structured(self, message: str, **fields) -> None:
        """
        Log a structured info message: the message text plus extra fields.

        Fields are appended to the message as key=value pairs and also kept
        JSON-serializable if needed for downstream ingestion.
        """
        if fields:
            try:
                fields_str = " | " + " ".join(f"{k}={v!r}" for k, v in fields.items())
            except Exception:
                fields_str = ""
            self.info(f"{message}{fields_str}")
        else:
            self.info(message)


# -------------------------
# Metric Logger (Research Grade Additions)
# -------------------------
class MetricLogger:
    """
    Tracks metrics per epoch/step and allows atomic JSON + CSV export.

    Features:
        - Atomic save to avoid corrupted files
        - Explicit 'epoch' and 'step' columns
        - Consistent CSV schema (all keys, 'epoch'/'step' first)
        - Autosave support
        - Reset functionality
    """
    def __init__(self, log_dir: Union[str, Path], filename_prefix: str = "metrics",
                 autosave_every: Optional[int] = None):
        """
        Args:
            log_dir: Directory to save logs
            filename_prefix: Prefix for filenames
            autosave_every: Save metrics automatically every N steps (optional)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file = self.log_dir / f"{filename_prefix}_{timestamp}.json"
        self.csv_file = self.log_dir / f"{filename_prefix}_{timestamp}.csv"

        self.records: List[Dict[str, Any]] = []
        self.all_keys: set = set()
        self.step_counter: int = 0
        self.autosave_every = autosave_every

    def reset(self):
        """Reset logger for a fresh run."""
        self.records.clear()
        self.all_keys.clear()
        self.step_counter = 0

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        """
        Log metrics for a given step/epoch.

        Args:
            metrics: dictionary of metric_name -> value
            step: optional batch step count
            epoch: optional epoch number
        """
        record = metrics.copy()
        record['epoch'] = epoch if epoch is not None else None

        self.step_counter += 1
        record['step'] = step if step is not None else self.step_counter

        self.records.append(record)
        self.all_keys.update(record.keys())

        if self.autosave_every and self.step_counter % self.autosave_every == 0:
            self.save_metrics()

    def save_metrics(self, filename: Optional[Union[str, Path]] = None,
                     format: str = 'both', overwrite: bool = True) -> Dict[str, Path]:
        """
        Save metrics to JSON and/or CSV.

        Args:
            filename: Optional override JSON path; CSV will use same prefix
            format: 'json', 'csv', or 'both'
            overwrite: overwrite existing files

        Returns:
            dict: {'json': Path, 'csv': Path}
        """
        json_path = Path(filename) if filename else self.json_file
        csv_path = Path(str(json_path).replace(".json", ".csv")) if filename else self.csv_file

        saved_paths = {}

        # --- JSON Save (Atomic) ---
        if format in ('json', 'both'):
            tmp_json = tempfile.NamedTemporaryFile('w', delete=False, dir=str(json_path.parent))
            try:
                json.dump(self.records, tmp_json, indent=2)
                tmp_json.close()
                if overwrite or not json_path.exists():
                    os.replace(tmp_json.name, json_path)
                saved_paths['json'] = json_path
            finally:
                if os.path.exists(tmp_json.name):
                    os.unlink(tmp_json.name)

        # --- CSV Save (Atomic, Deterministic Schema) ---
        if format in ('csv', 'both'):
            # Ensure 'epoch' and 'step' are first, then sort the rest
            all_keys_sorted = ['epoch', 'step'] + sorted(k for k in self.all_keys if k not in ('epoch', 'step'))
            tmp_csv = tempfile.NamedTemporaryFile('w', newline='', delete=False, dir=str(csv_path.parent))
            try:
                writer = csv.DictWriter(tmp_csv, fieldnames=all_keys_sorted)
                writer.writeheader()
                for record in self.records:
                    # Fill missing keys with empty string to maintain schema
                    row = {k: record.get(k, "") for k in all_keys_sorted}
                    writer.writerow(row)
                tmp_csv.close()
                if overwrite or not csv_path.exists():
                    os.replace(tmp_csv.name, csv_path)
                saved_paths['csv'] = csv_path
            finally:
                if os.path.exists(tmp_csv.name):
                    os.unlink(tmp_csv.name)

        return saved_paths


# -------------------------
# Convenience function
# -------------------------
def get_loggers(experiment_dir: Union[str, Path],
                autosave_every: Optional[int] = None) -> Tuple[ExperimentLogger, MetricLogger]:
    """
    Create ExperimentLogger and MetricLogger with proper subdirectories.

    Args:
        experiment_dir: base directory for logs
        autosave_every: optional batch frequency to autosave metrics
    """
    logs_dir = Path(experiment_dir) / "logs"
    exp_logger = ExperimentLogger(name="run_name_placeholder", log_dir=logs_dir)
    metric_logger = MetricLogger(log_dir=logs_dir, autosave_every=autosave_every)
    return exp_logger, metric_logger


# -------------------------
# Example basic usage (commented)
# -------------------------
# if __name__ == "__main__":
#     from pathlib import Path
#     cfg = {"lr": 0.001, "batch": 64}
#     exp_logger, metric_logger = get_loggers(Path("logs"), autosave_every=10)
#     with exp_logger as logger:
#         logger.section("Demo Run")
#         logger.info("Starting demo run")
#         logger.log_metadata(cfg, format="json")
#         for i in logger.progress_bar(range(5), desc="Loop"):
#             logger.debug(f"Step {i}")
#             metric_logger.log_metrics({"loss": 0.1 * i, "acc": 0.9 - 0.01 * i}, step=i, epoch=i)
#         logger.info("Finished demo")
#     metric_logger.save_metrics(format='both') # Save final metrics