"""Configuration loader for Oculodoc."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from .schema import OculodocConfig
from ..errors import ConfigurationError


class ConfigLoader:
    """Configuration loader for Oculodoc."""

    @staticmethod
    def load_from_file(config_path: Path) -> OculodocConfig:
        """Load configuration from YAML or JSON file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            OculodocConfig instance.

        Raises:
            ConfigurationError: If file doesn't exist, can't be read, or is invalid.
            FileNotFoundError: If the configuration file doesn't exist.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}") from e
        except IOError as e:
            raise ConfigurationError(f"Failed to read configuration file: {e}") from e

        return ConfigLoader._create_config_from_dict(config_data)

    @staticmethod
    def load_from_env(prefix: str = "OCULODOC_") -> OculodocConfig:
        """Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables.

        Returns:
            OculodocConfig instance.
        """
        config_data = {}

        # Extract environment variables with the specified prefix
        env_vars = {
            key: value for key, value in os.environ.items() if key.startswith(prefix)
        }

        # Convert environment variables to nested dictionary
        for env_key, value in env_vars.items():
            # Remove prefix and convert to lowercase
            config_key = env_key[len(prefix) :].lower()

            # Handle nested keys (e.g., LAYOUT_MODEL_TYPE -> layout.model_type)
            parts = config_key.split("_")
            if len(parts) > 1:
                section = parts[0]
                key = "_".join(parts[1:])
                if section not in config_data:
                    config_data[section] = {}
                config_data[section][key] = ConfigLoader._parse_env_value(value)
            else:
                config_data[config_key] = ConfigLoader._parse_env_value(value)

        return ConfigLoader._create_config_from_dict(config_data)

    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> OculodocConfig:
        """Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            OculodocConfig instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        return ConfigLoader._create_config_from_dict(config_dict)

    @staticmethod
    def _create_config_from_dict(config_dict: Dict[str, Any]) -> OculodocConfig:
        """Create OculodocConfig from dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            OculodocConfig instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        try:
            # Create default config
            config = OculodocConfig()

            # Update with provided values
            if "layout" in config_dict:
                layout_data = config_dict["layout"]
                for key, value in layout_data.items():
                    if hasattr(config.layout, key):
                        setattr(config.layout, key, value)

            if "vlm" in config_dict:
                vlm_data = config_dict["vlm"]
                for key, value in vlm_data.items():
                    if hasattr(config.vlm, key):
                        setattr(config.vlm, key, value)

            if "processing" in config_dict:
                processing_data = config_dict["processing"]
                for key, value in processing_data.items():
                    if hasattr(config.processing, key):
                        setattr(config.processing, key, value)

            if "cache" in config_dict:
                cache_data = config_dict["cache"]
                for key, value in cache_data.items():
                    if hasattr(config.cache, key):
                        setattr(config.cache, key, value)

            if "batch" in config_dict:
                batch_data = config_dict["batch"]
                for key, value in batch_data.items():
                    if hasattr(config.batch, key):
                        setattr(config.batch, key, value)

            if "logging" in config_dict:
                logging_data = config_dict["logging"]
                for key, value in logging_data.items():
                    if hasattr(config.logging, key):
                        setattr(config.logging, key, value)

            if "metrics" in config_dict:
                metrics_data = config_dict["metrics"]
                for key, value in metrics_data.items():
                    if hasattr(config.metrics, key):
                        setattr(config.metrics, key, value)

            return config

        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {e}") from e

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment variable.

        Returns:
            Parsed value (int, float, bool, or str).
        """
        # Try to parse as boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # Try to parse as int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    @staticmethod
    def save_to_file(config: OculodocConfig, config_path: Path) -> None:
        """Save configuration to file.

        Args:
            config: OculodocConfig instance.
            config_path: Path to save the configuration.

        Raises:
            ConfigurationError: If saving fails.
        """
        try:
            config_data = config.to_dict()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_data, f, default_flow_style=False)
                elif config_path.suffix.lower() == ".json":
                    json.dump(config_data, f, indent=2)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}") from e
