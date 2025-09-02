"""Abstract interface for Vision-Language Model analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class VLMAnalysisResult:
    """Standardized VLM analysis result.

    Attributes:
        content_type: The type of content ('text', 'table', 'image', 'formula').
        content: The extracted content as a string.
        bbox: Optional bounding box coordinates as [x1, y1, x2, y2].
        confidence: The confidence score for this analysis.
        metadata: Additional metadata associated with this analysis.
    """

    def __init__(
        self,
        content_type: str,
        content: str,
        bbox: Optional[List[int]] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a VLMAnalysisResult.

        Args:
            content_type: The type of content ('text', 'table', 'image', 'formula').
            content: The extracted content as a string.
            bbox: Optional bounding box coordinates as [x1, y1, x2, y2].
            confidence: The confidence score for this analysis.
            metadata: Additional metadata associated with this analysis.

        Raises:
            ValueError: If bbox is provided but doesn't contain exactly 4 coordinates.
        """
        if bbox is not None and len(bbox) != 4:
            raise ValueError("bbox must contain exactly 4 coordinates [x1, y1, x2, y2]")

        self.content_type = content_type
        self.content = content
        self.bbox = bbox
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            A dictionary representation of the analysis result.
        """
        return {
            "content_type": self.content_type,
            "content": self.content,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class IVLMAnalyzer(ABC):
    """Abstract interface for VLM-based document analysis."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize VLM analyzer.

        Args:
            config: Configuration dictionary for the analyzer.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    @abstractmethod
    async def analyze(
        self, image_data: str, prompt: str, **kwargs: Any
    ) -> List[VLMAnalysisResult]:
        """Analyze document image with VLM.

        Args:
            image_data: Base64 encoded image data or image path.
            prompt: The prompt for the VLM analysis.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            A list of VLMAnalysisResult objects.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement analyze method")

    @abstractmethod
    async def batch_analyze(
        self, image_data_list: List[str], prompts: List[str], **kwargs: Any
    ) -> List[List[VLMAnalysisResult]]:
        """Batch analyze multiple images.

        Args:
            image_data_list: List of base64 encoded image data or image paths.
            prompts: List of prompts for the VLM analysis.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            A list of lists of VLMAnalysisResult objects.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement batch_analyze method")
