"""Unit tests for configuration management."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oculodoc.config import (
    OculodocConfig,
    LayoutConfig,
    VLMConfig,
    ProcessingConfig,
    ConfigLoader,
)
from oculodoc.errors import ConfigurationError


class TestOculodocConfig:
    """Test cases for OculodocConfig."""

    def test_default_config_creation(self):
        """Test creating OculodocConfig with default values."""
        config = OculodocConfig()

        assert config.layout.model_type == "doclayout_yolo"
        assert config.vlm.model_type == "sglang_ocrflux"
        assert config.processing.use_hybrid is True
        assert config.cache.enabled is True
        assert config.logging.level == "INFO"

    def test_config_to_dict(self):
        """Test converting OculodocConfig to dictionary."""
        config = OculodocConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "layout" in config_dict
        assert "vlm" in config_dict
        assert "processing" in config_dict
        assert "cache" in config_dict
        assert "logging" in config_dict
        assert "metrics" in config_dict

        # Check specific values
        assert config_dict["layout"]["model_type"] == "doclayout_yolo"
        assert config_dict["vlm"]["model_type"] == "sglang_ocrflux"
        assert config_dict["processing"]["use_hybrid"] is True

    def test_custom_config_values(self):
        """Test OculodocConfig with custom values."""
        layout_config = LayoutConfig(
            model_type="generic_yolo",
            model_path="/path/to/model",
            confidence_threshold=0.8,
        )

        vlm_config = VLMConfig(
            model_type="transformers_qwen", host="custom-host", port=8080
        )

        config = OculodocConfig(layout=layout_config, vlm=vlm_config)

        assert config.layout.model_type == "generic_yolo"
        assert config.layout.model_path == "/path/to/model"
        assert config.layout.confidence_threshold == 0.8
        assert config.vlm.host == "custom-host"
        assert config.vlm.port == 8080


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def test_load_from_dict_basic(self):
        """Test loading configuration from basic dictionary."""
        config_dict = {
            "layout": {"model_type": "doclayout_yolo", "device": "cpu"},
            "vlm": {"model_type": "transformers_qwen", "port": 9090},
            "processing": {"max_concurrent_pages": 8},
        }

        config = ConfigLoader.load_from_dict(config_dict)

        assert config.layout.model_type == "doclayout_yolo"
        assert config.layout.device == "cpu"
        assert config.vlm.port == 9090
        assert config.processing.max_concurrent_pages == 8

    def test_load_from_dict_partial(self):
        """Test loading configuration with partial dictionary."""
        config_dict = {"layout": {"confidence_threshold": 0.9}}

        config = ConfigLoader.load_from_dict(config_dict)

        # Should have custom value
        assert config.layout.confidence_threshold == 0.9
        # Should have default values for other fields
        assert config.layout.model_type == "doclayout_yolo"
        assert config.vlm.model_type == "sglang_ocrflux"

    def test_load_from_dict_invalid_key(self):
        """Test loading configuration with invalid key."""
        config_dict = {"invalid_section": {"some_key": "some_value"}}

        config = ConfigLoader.load_from_dict(config_dict)

        # Should ignore invalid sections and use defaults
        assert config.layout.model_type == "doclayout_yolo"

    def test_parse_env_value_string(self):
        """Test parsing string environment variable values."""
        assert ConfigLoader._parse_env_value("cuda") == "cuda"
        assert ConfigLoader._parse_env_value("0.5") == 0.5  # Parsed as float
        assert ConfigLoader._parse_env_value("true") is True
        assert ConfigLoader._parse_env_value("false") is False

    def test_parse_env_value_int(self):
        """Test parsing integer environment variable values."""
        assert ConfigLoader._parse_env_value("8080") == 8080
        assert ConfigLoader._parse_env_value("4") == 4

    def test_parse_env_value_float(self):
        """Test parsing float environment variable values."""
        assert ConfigLoader._parse_env_value("0.85") == 0.85
        assert ConfigLoader._parse_env_value("1.0") == 1.0

    def test_parse_env_value_bool(self):
        """Test parsing boolean environment variable values."""
        assert ConfigLoader._parse_env_value("true") is True
        assert ConfigLoader._parse_env_value("True") is True
        assert ConfigLoader._parse_env_value("TRUE") is True
        assert ConfigLoader._parse_env_value("false") is False
        assert ConfigLoader._parse_env_value("False") is False
        assert ConfigLoader._parse_env_value("FALSE") is False

    @patch.dict(
        os.environ,
        {
            "OCULODOC_LAYOUT_MODEL_TYPE": "generic_yolo",
            "OCULODOC_LAYOUT_DEVICE": "cpu",
            "OCULODOC_VLM_PORT": "9090",
            "OCULODOC_PROCESSING_MAX_CONCURRENT_PAGES": "8",
        },
    )
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = ConfigLoader.load_from_env()

        assert config.layout.model_type == "generic_yolo"
        assert config.layout.device == "cpu"
        assert config.vlm.port == 9090
        assert config.processing.max_concurrent_pages == 8

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "layout": {"model_type": "doclayout_yolo", "device": "cuda"},
            "vlm": {"model_type": "sglang_ocrflux", "port": 30000},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = ConfigLoader.load_from_file(config_path)

            assert config.layout.model_type == "doclayout_yolo"
            assert config.layout.device == "cuda"
            assert config.vlm.model_type == "sglang_ocrflux"
            assert config.vlm.port == 30000
        finally:
            config_path.unlink()

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {"processing": {"use_hybrid": False, "max_concurrent_pages": 6}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = ConfigLoader.load_from_file(config_path)

            assert config.processing.use_hybrid is False
            assert config.processing.max_concurrent_pages == 6
        finally:
            config_path.unlink()

    def test_load_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        config_path = Path("/non/existent/config.yaml")

        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_from_file(config_path)

    def test_load_from_file_invalid_format(self):
        """Test loading configuration from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid format")
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigurationError, match="Unsupported configuration file format"
            ):
                ConfigLoader.load_from_file(config_path)
        finally:
            config_path.unlink()

    def test_load_from_file_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigurationError, match="Failed to parse configuration file"
            ):
                ConfigLoader.load_from_file(config_path)
        finally:
            config_path.unlink()

    def test_load_from_file_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")  # Invalid JSON
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigurationError, match="Failed to parse configuration file"
            ):
                ConfigLoader.load_from_file(config_path)
        finally:
            config_path.unlink()

    def test_save_to_file_yaml(self):
        """Test saving configuration to YAML file."""
        config = OculodocConfig()
        config.layout.model_type = "test_model"

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            ConfigLoader.save_to_file(config, config_path)

            # Load it back and verify
            loaded_config = ConfigLoader.load_from_file(config_path)
            assert loaded_config.layout.model_type == "test_model"
        finally:
            config_path.unlink()

    def test_save_to_file_json(self):
        """Test saving configuration to JSON file."""
        config = OculodocConfig()
        config.processing.use_hybrid = False

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config_path = Path(f.name)

        try:
            ConfigLoader.save_to_file(config, config_path)

            # Load it back and verify
            loaded_config = ConfigLoader.load_from_file(config_path)
            assert loaded_config.processing.use_hybrid is False
        finally:
            config_path.unlink()

    def test_save_to_file_invalid_format(self):
        """Test saving configuration to unsupported file format."""
        config = OculodocConfig()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigurationError, match="Unsupported configuration file format"
            ):
                ConfigLoader.save_to_file(config, config_path)
        finally:
            config_path.unlink()
