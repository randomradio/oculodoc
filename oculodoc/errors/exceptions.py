"""Exception classes for Oculodoc."""


class OculodocError(Exception):
    """Base exception for Oculodoc.

    All Oculodoc-specific exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: dict = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(OculodocError):
    """Exception raised when model loading/initialization fails."""

    def __init__(self, message: str, model_type: str = None, model_path: str = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            model_type: Type of model that failed to load.
            model_path: Path to the model that failed to load.
        """
        super().__init__(message, {"model_type": model_type, "model_path": model_path})
        self.model_type = model_type
        self.model_path = model_path


class InferenceError(OculodocError):
    """Exception raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_type: str = None,
        input_shape: tuple = None,
        operation: str = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            model_type: Type of model that failed during inference.
            input_shape: Shape of input that caused the failure.
            operation: The operation that failed.
        """
        super().__init__(
            message,
            {
                "model_type": model_type,
                "input_shape": input_shape,
                "operation": operation,
            },
        )
        self.model_type = model_type
        self.input_shape = input_shape
        self.operation = operation


class ConfigurationError(OculodocError):
    """Exception raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str = None, expected_type: str = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            config_key: The configuration key that is invalid.
            expected_type: The expected type for the configuration value.
        """
        super().__init__(
            message,
            {"config_key": config_key, "expected_type": expected_type},
        )
        self.config_key = config_key
        self.expected_type = expected_type


class ProcessingError(OculodocError):
    """Exception raised when document processing fails."""

    def __init__(
        self,
        message: str,
        document_path: str = None,
        page_number: int = None,
        stage: str = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            document_path: Path to the document being processed.
            page_number: Page number where the error occurred.
            stage: Processing stage where the error occurred.
        """
        super().__init__(
            message,
            {
                "document_path": document_path,
                "page_number": page_number,
                "stage": stage,
            },
        )
        self.document_path = document_path
        self.page_number = page_number
        self.stage = stage


class ValidationError(OculodocError):
    """Exception raised when input validation fails."""

    def __init__(self, message: str, field: str = None, value: str = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            field: The field that failed validation.
            value: The invalid value that was provided.
        """
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value
