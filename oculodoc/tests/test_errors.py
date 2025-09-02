"""Unit tests for error handling."""

import pytest

from oculodoc.errors import (
    OculodocError,
    ModelLoadError,
    InferenceError,
    ConfigurationError,
    ProcessingError,
    ValidationError,
)


class TestOculodocError:
    """Test cases for base OculodocError."""

    def test_oculodoc_error_creation(self):
        """Test creating OculodocError with message."""
        error = OculodocError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_oculodoc_error_with_details(self):
        """Test creating OculodocError with details."""
        details = {"component": "layout_analyzer", "operation": "initialize"}
        error = OculodocError("Initialization failed", details)
        assert error.details == details

    def test_oculodoc_error_inheritance(self):
        """Test that OculodocError is a proper Exception subclass."""
        error = OculodocError("Test error")
        assert isinstance(error, Exception)


class TestModelLoadError:
    """Test cases for ModelLoadError."""

    def test_model_load_error_creation(self):
        """Test creating ModelLoadError with basic parameters."""
        error = ModelLoadError("Failed to load YOLO model")
        assert str(error) == "Failed to load YOLO model"
        assert error.model_type is None
        assert error.model_path is None
        assert error.details == {"model_type": None, "model_path": None}

    def test_model_load_error_with_model_info(self):
        """Test creating ModelLoadError with model information."""
        error = ModelLoadError(
            "Model file not found",
            model_type="doclayout_yolo",
            model_path="/path/to/model.pt",
        )
        assert error.model_type == "doclayout_yolo"
        assert error.model_path == "/path/to/model.pt"
        assert error.details["model_type"] == "doclayout_yolo"
        assert error.details["model_path"] == "/path/to/model.pt"

    def test_model_load_error_inheritance(self):
        """Test that ModelLoadError inherits from OculodocError."""
        error = ModelLoadError("Test error")
        assert isinstance(error, OculodocError)
        assert isinstance(error, Exception)


class TestInferenceError:
    """Test cases for InferenceError."""

    def test_inference_error_creation(self):
        """Test creating InferenceError with basic parameters."""
        error = InferenceError("Inference failed")
        assert str(error) == "Inference failed"
        assert error.model_type is None
        assert error.input_shape is None
        assert error.operation is None

    def test_inference_error_with_details(self):
        """Test creating InferenceError with detailed information."""
        error = InferenceError(
            "GPU memory exhausted",
            model_type="yolov10",
            input_shape=(1, 3, 640, 640),
            operation="forward_pass",
        )
        assert error.model_type == "yolov10"
        assert error.input_shape == (1, 3, 640, 640)
        assert error.operation == "forward_pass"
        assert error.details["model_type"] == "yolov10"
        assert error.details["input_shape"] == (1, 3, 640, 640)
        assert error.details["operation"] == "forward_pass"

    def test_inference_error_inheritance(self):
        """Test that InferenceError inherits from OculodocError."""
        error = InferenceError("Test error")
        assert isinstance(error, OculodocError)


class TestConfigurationError:
    """Test cases for ConfigurationError."""

    def test_configuration_error_creation(self):
        """Test creating ConfigurationError with basic parameters."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert error.config_key is None
        assert error.expected_type is None

    def test_configuration_error_with_details(self):
        """Test creating ConfigurationError with key and type information."""
        error = ConfigurationError(
            "Invalid device configuration", config_key="device", expected_type="str"
        )
        assert error.config_key == "device"
        assert error.expected_type == "str"
        assert error.details["config_key"] == "device"
        assert error.details["expected_type"] == "str"

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from OculodocError."""
        error = ConfigurationError("Test error")
        assert isinstance(error, OculodocError)


class TestProcessingError:
    """Test cases for ProcessingError."""

    def test_processing_error_creation(self):
        """Test creating ProcessingError with basic parameters."""
        error = ProcessingError("Document processing failed")
        assert str(error) == "Document processing failed"
        assert error.document_path is None
        assert error.page_number is None
        assert error.stage is None

    def test_processing_error_with_details(self):
        """Test creating ProcessingError with detailed information."""
        error = ProcessingError(
            "Failed to extract text from page",
            document_path="/docs/test.pdf",
            page_number=5,
            stage="text_extraction",
        )
        assert error.document_path == "/docs/test.pdf"
        assert error.page_number == 5
        assert error.stage == "text_extraction"
        assert error.details["document_path"] == "/docs/test.pdf"
        assert error.details["page_number"] == 5
        assert error.details["stage"] == "text_extraction"

    def test_processing_error_inheritance(self):
        """Test that ProcessingError inherits from OculodocError."""
        error = ProcessingError("Test error")
        assert isinstance(error, OculodocError)


class TestValidationError:
    """Test cases for ValidationError."""

    def test_validation_error_creation(self):
        """Test creating ValidationError with basic parameters."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.field is None
        assert error.value is None

    def test_validation_error_with_details(self):
        """Test creating ValidationError with field and value information."""
        error = ValidationError(
            "Invalid file format", field="file_extension", value=".exe"
        )
        assert error.field == "file_extension"
        assert error.value == ".exe"
        assert error.details["field"] == "file_extension"
        assert error.details["value"] == ".exe"

    def test_validation_error_inheritance(self):
        """Test that ValidationError inherits from OculodocError."""
        error = ValidationError("Test error")
        assert isinstance(error, OculodocError)


class TestErrorHierarchy:
    """Test cases for error hierarchy and relationships."""

    def test_all_errors_inherit_from_oculodoc_error(self):
        """Test that all specific errors inherit from OculodocError."""
        errors = [
            ModelLoadError("test"),
            InferenceError("test"),
            ConfigurationError("test"),
            ProcessingError("test"),
            ValidationError("test"),
        ]

        for error in errors:
            assert isinstance(error, OculodocError)
            assert isinstance(error, Exception)

    def test_error_details_preservation(self):
        """Test that error details are properly preserved in inheritance."""
        # Test with ModelLoadError
        model_error = ModelLoadError(
            "Load failed", model_type="test_model", model_path="/test/path"
        )
        assert model_error.details["model_type"] == "test_model"
        assert model_error.details["model_path"] == "/test/path"

        # Test with InferenceError
        inference_error = InferenceError(
            "Inference failed",
            model_type="test_model",
            input_shape=(1, 3, 224, 224),
            operation="predict",
        )
        assert inference_error.details["model_type"] == "test_model"
        assert inference_error.details["input_shape"] == (1, 3, 224, 224)
        assert inference_error.details["operation"] == "predict"

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        error = OculodocError("Custom error message")
        assert str(error) == "Custom error message"

        # Test that child classes preserve the message
        child_error = ModelLoadError("Child error message")
        assert str(child_error) == "Child error message"

    def test_error_details_are_optional(self):
        """Test that error details are optional and default to empty dict."""
        error = OculodocError("Test error")
        assert error.details == {}

        # Test with None details
        error_with_none = OculodocError("Test error", None)
        assert error_with_none.details == {}
