"""Oculodoc - Production-ready document processing system with Vision-Language Models."""

__version__ = "0.1.0"

from .interfaces import (
    ILayoutAnalyzer,
    LayoutDetection,
    IVLMAnalyzer,
    VLMAnalysisResult,
    IDocumentProcessor,
    ProcessingResult,
)
from .config import (
    OculodocConfig,
    ConfigLoader,
)
from .errors import (
    OculodocError,
    ModelLoadError,
    InferenceError,
    ConfigurationError,
    ProcessingError,
    ValidationError,
)

__all__ = [
    # Core interfaces
    "ILayoutAnalyzer",
    "LayoutDetection",
    "IVLMAnalyzer",
    "VLMAnalysisResult",
    "IDocumentProcessor",
    "ProcessingResult",
    # Configuration
    "OculodocConfig",
    "ConfigLoader",
    # Errors
    "OculodocError",
    "ModelLoadError",
    "InferenceError",
    "ConfigurationError",
    "ProcessingError",
    "ValidationError",
]
