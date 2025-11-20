"""
Configuration utilities for loading and validating YAML configurations.

This module handles loading your three config files (data_config.yaml,
preprocess_config.yaml, phase1_config.yaml) and merging them with any
command-line overrides. It also validates that required fields are present
to catch configuration errors early before expensive preprocessing or training.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """
    Loads and validates configuration files with hierarchical merging.

    The loader supports three config files that work together:
    - data_config: Dataset paths, sampling strategy, memory limits
    - preprocess_config: Feature engineering, windowing, scaling
    - phase1_config: Model architectures, training hyperparameters, evaluation

    Why separate configs? Each represents a different concern (data vs models vs training),
    and you might want to reuse the same data config with different model configs.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing configuration files.
                       If None, assumes configs/ relative to project root.
        """
        if config_dir is None:
            # Navigate up from src/utils/ to project root, then to configs/
            config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config_dir = Path(config_dir) # Ensure self.config_dir is always a Path object

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.config_dir}\n"
                f"Make sure you've created the configs/ directory with YAML files."
            )

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a single configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected files: data_config.yaml, preprocess_config.yaml, phase1_config.yaml"
            )

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ValueError(f"Config file {config_path} is empty")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all three configuration files.

        Returns:
            Dictionary with keys: 'data', 'preprocess', 'phase1'

        Why return a nested dict? This structure keeps configs organized by concern,
        making it clear which settings affect data vs preprocessing vs training.
        """
        configs = {}

        for config_name in ['data_config', 'preprocess_config', 'phase1_config']:
            # Calculate the key *before* the try block
            key = config_name.replace('_config', '')
            try:
                # Remove '_config' suffix for cleaner access: config['data'] not config['data_config']
                configs[key] = self.load_config(config_name)
            except FileNotFoundError as e:
                # Don't fail completely if one config is missing - warn and use empty dict
                # This allows partial functionality during development
                print(f"Warning: {e}")
                configs[key] = {} # key is now definitely defined in this scope

        return configs

    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Recursively merge override_config into base_config.

        This is useful for applying command-line overrides or mode-specific settings
        without modifying the base config files. For example, 'quick' mode might
        override the number of epochs from 50 to 20.

        Args:
            base_config: Base configuration dictionary
            override_config: Override values (takes precedence)

        Returns:
            Merged configuration (base_config is not modified)

        Why recursive? Config files are nested (e.g., training.optimizer.learning_rate),
        and we want to override at any depth without replacing entire sections.
        """
        merged = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Both are dicts, merge recursively
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # Override the value
                merged[key] = value

        return merged

    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration has required fields.

        This catches missing configuration early rather than failing deep in training.
        For example, if you forget to specify model architecture, you'd rather know
        immediately than after preprocessing your dataset.

        Args:
            config: Configuration dictionary to validate
            config_type: Type of config ('data', 'preprocess', 'phase1')

        Returns:
            True if valid

        Raises:
            ValueError: If required fields are missing
        """
        # Define what fields must be present in each config type
        required_fields = {
            'data': ['datasets', 'sampling', 'loading'],
            'preprocess': ['canonical_features', 'feature_mappings', 'windowing'],
            'phase1': ['models', 'training', 'evaluation'],
        }

        if config_type not in required_fields:
            raise ValueError(f"Unknown config type: {config_type}")

        missing_fields = []
        for field in required_fields[config_type]:
            if field not in config:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {config_type} config: {missing_fields}\n"
                f"Check your {config_type}_config.yaml file."
            )

        return True

    def get_experiment_config(self, mode: str = "quick") -> Dict[str, Any]:
        """
        Get complete configuration for an experiment with mode-specific overrides.

        Modes allow you to run the same code with different resource trade-offs:
        - quick: Small samples, fewer epochs - fast iteration (2 hours)
        - medium: Moderate samples, more epochs - validation (4-6 hours)
        - full: Complete datasets, full training - final results (overnight)

        Args:
            mode: Experiment mode ('quick', 'medium', 'full')

        Returns:
            Complete merged configuration ready for use

        Why modes? You want to iterate quickly during development (quick mode),
        then run more thorough experiments for validation, then full runs for
        final results. Same code, different resource profiles.
        """
        # Load all configs
        configs = self.load_all_configs()

        # Validate each config has required structure
        for config_type, config in configs.items():
            if config:  # Only validate non-empty configs
                try:
                    self.validate_config(config, config_type)
                except ValueError as e:
                    print(f"Warning: Validation failed for {config_type}: {e}")

        # Apply mode-specific overrides from phase1_config
        if 'modes' in configs.get('phase1', {}) and mode in configs['phase1']['modes']:
            mode_config = configs['phase1']['modes'][mode]

            # Override training epochs
            if 'epochs' in mode_config and 'training' in configs.get('phase1', {}):
                configs['phase1']['training']['epochs'] = mode_config['epochs']

            # Override data sampling
            if 'data_samples' in mode_config and 'sampling' in configs.get('data', {}):
                # Update the sampling config for the specified mode
                sample_key = f'{mode}_mode'
                if sample_key in configs['data']['sampling']:
                    configs['data']['sampling'][sample_key]['samples_per_dataset'] = mode_config['data_samples']

        return configs


# ============================================================================
# TEST SECTION - Run this to verify the config loader works with your files
# ============================================================================

if __name__ == "__main__":
    """
    Test the configuration loader with your actual config files.

    Run from project root:
        python -m src.utils.config

    This should load all three config files and print their contents.
    If any config file is missing or has syntax errors, you'll see
    a clear error message telling you what's wrong.
    """

    print("=" * 70)
    print("Testing Configuration Loader")
    print("=" * 70)

    try:
        # Initialize loader
        loader = ConfigLoader()
        print(f"✓ Config directory found: {loader.config_dir}")

        # Load individual configs
        print("\n" + "-" * 70)
        print("Loading individual configs...")
        print("-" * 70)

        data_config = loader.load_config('data_config')
        print(f"✓ data_config.yaml loaded: {len(data_config)} top-level keys")
        print(f"  Datasets: {list(data_config.get('datasets', {}).keys())}")

        preprocess_config = loader.load_config('preprocess_config')
        print(f"✓ preprocess_config.yaml loaded: {len(preprocess_config)} top-level keys")
        print(f"  Windowing: length={preprocess_config.get('windowing', {}).get('window_length', 'N/A')}")

        phase1_config = loader.load_config('phase1_config')
        print(f"✓ phase1_config.yaml loaded: {len(phase1_config)} top-level keys")
        print(f"  Models: {list(phase1_config.get('models', {}).keys())}")

        # Load all configs together
        print("\n" + "-" * 70)
        print("Loading all configs...")
        print("-" * 70)

        all_configs = loader.load_all_configs()
        print(f"✓ All configs loaded: {list(all_configs.keys())}")

        # Test mode-specific config
        print("\n" + "-" * 70)
        print("Testing mode-specific configuration (quick mode)...")
        print("-" * 70)

        exp_config = loader.get_experiment_config(mode='quick')

        if 'phase1' in exp_config and 'training' in exp_config['phase1']:
            epochs = exp_config['phase1']['training'].get('epochs', 'N/A')
            print(f"✓ Quick mode epochs: {epochs}")

        if 'data' in exp_config and 'sampling' in exp_config['data']:
            sample_size = exp_config['data']['sampling'].get('quick_mode', {}).get('samples_per_dataset', 'N/A')
            print(f"✓ Quick mode sample size: {sample_size}")

        print("\n" + "=" * 70)
        print("Configuration loader test PASSED!")
        print("=" * 70)
        print("\nYou can now use ConfigLoader in your code:")
        print("  from src.utils.config import ConfigLoader")
        print("  config = ConfigLoader().get_experiment_config(mode='quick')")

    except Exception as e:
        print("\n" + "=" * 70)
        print("Configuration loader test FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. All three YAML files exist in configs/")
        print("  2. YAML syntax is correct (no tabs, proper indentation)")
        print("  3. Required fields are present")
        import sys
        sys.exit(1)
