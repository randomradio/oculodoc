"""Core interfaces for Oculodoc components."""

from .layout_analyzer import ILayoutAnalyzer, LayoutDetection
from .vlm_analyzer import IVLMAnalyzer, VLMAnalysisResult
from .document_processor import IDocumentProcessor, ProcessingResult

__all__ = [
    "ILayoutAnalyzer",
    "LayoutDetection",
    "IVLMAnalyzer",
    "VLMAnalysisResult",
    "IDocumentProcessor",
    "ProcessingResult",
]
