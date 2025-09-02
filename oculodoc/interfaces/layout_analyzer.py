"""Abstract interface for layout analysis models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from PIL import Image


class LayoutDetection:
    """Standardized layout detection result.

    Attributes:
        category_id: The category ID for this detection.
        category_name: The human-readable name for this detection.
        bbox: The bounding box coordinates as [x1, y1, x2, y2].
        confidence: The confidence score for this detection.
        metadata: Additional metadata associated with this detection.
    """

    def __init__(
        self,
        category_id: int,
        category_name: str,
        bbox: List[int],
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a LayoutDetection.

        Args:
            category_id: The category ID for this detection.
            category_name: The human-readable name for this detection.
            bbox: The bounding box coordinates as [x1, y1, x2, y2].
            confidence: The confidence score for this detection.
            metadata: Additional metadata associated with this detection.

        Raises:
            ValueError: If bbox doesn't contain exactly 4 coordinates.
        """
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly 4 coordinates [x1, y1, x2, y2]")

        self.category_id = category_id
        self.category_name = category_name
        self.bbox = bbox
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the detection to a dictionary.

        Returns:
            A dictionary representation of the detection.
        """
        return {
            "category_id": self.category_id,
            "category_name": self.category_name,
            "bbox": self.bbox,
            "score": self.confidence,  # MinerU uses 'score' instead of 'confidence'
            "metadata": self.metadata,
        }

    def to_middle_format(self) -> Dict[str, Any]:
        """Convert to MinerU middle.json format.

        Returns:
            Dictionary in MinerU middle.json format.
        """
        return {
            "type": self.category_name,
            "bbox": self.bbox,
            "score": self.confidence,
            "category_id": self.category_id,
        }


class ILayoutAnalyzer(ABC):
    """Abstract interface for layout analysis models."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the layout analyzer with configuration.

        Args:
            config: Configuration dictionary for the analyzer.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    @abstractmethod
    async def analyze(self, image: Image.Image, **kwargs: Any) -> List[LayoutDetection]:
        """Analyze image and return layout detections.

        Args:
            image: The PIL Image to analyze.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            A list of LayoutDetection objects.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement analyze method")

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources (GPU memory, etc.).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement cleanup method")

    @property
    @abstractmethod
    def supported_categories(self) -> Dict[int, str]:
        """Return mapping of category IDs to names.

        Returns:
            A dictionary mapping category IDs to their human-readable names.

        Raises:
            NotImplementedError: This property must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement supported_categories property"
        )
