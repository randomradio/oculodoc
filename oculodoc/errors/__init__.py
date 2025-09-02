"""Error handling framework for Oculodoc."""

from .exceptions import (
    OculodocError,
    ModelLoadError,
    InferenceError,
    ConfigurationError,
    ProcessingError,
    ValidationError,
)

__all__ = [
    "OculodocError",
    "ModelLoadError",
    "InferenceError",
    "ConfigurationError",
    "ProcessingError",
    "ValidationError",
]
