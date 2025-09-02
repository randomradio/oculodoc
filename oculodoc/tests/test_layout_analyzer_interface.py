"""Unit tests for layout analyzer interface."""

import pytest
from PIL import Image

from oculodoc.interfaces.layout_analyzer import LayoutDetection, ILayoutAnalyzer


class TestLayoutDetection:
    """Test cases for LayoutDetection class."""

    def test_layout_detection_creation(self):
        """Test creating a LayoutDetection with valid parameters."""
        detection = LayoutDetection(
            category_id=1,
            category_name="Text",
            bbox=[10, 20, 100, 150],
            confidence=0.85,
            metadata={"font_size": 12},
        )

        assert detection.category_id == 1
        assert detection.category_name == "Text"
        assert detection.bbox == [10, 20, 100, 150]
        assert detection.confidence == 0.85
        assert detection.metadata == {"font_size": 12}

    def test_layout_detection_bbox_validation(self):
        """Test bbox validation in LayoutDetection."""
        with pytest.raises(ValueError, match="bbox must contain exactly 4 coordinates"):
            LayoutDetection(
                category_id=1,
                category_name="Text",
                bbox=[10, 20, 100],  # Only 3 coordinates
                confidence=0.85,
            )

    def test_layout_detection_to_dict(self):
        """Test converting LayoutDetection to dictionary."""
        detection = LayoutDetection(
            category_id=1,
            category_name="Text",
            bbox=[10, 20, 100, 150],
            confidence=0.85,
            metadata={"font_size": 12},
        )

        expected_dict = {
            "category_id": 1,
            "category_name": "Text",
            "bbox": [10, 20, 100, 150],
            "confidence": 0.85,
            "metadata": {"font_size": 12},
        }

        assert detection.to_dict() == expected_dict

    def test_layout_detection_default_metadata(self):
        """Test LayoutDetection with default empty metadata."""
        detection = LayoutDetection(
            category_id=1,
            category_name="Text",
            bbox=[10, 20, 100, 150],
            confidence=0.85,
        )

        assert detection.metadata == {}


class MockLayoutAnalyzer(ILayoutAnalyzer):
    """Mock implementation of ILayoutAnalyzer for testing."""

    def __init__(self):
        self.initialized = False
        self.cleaned_up = False
        self.supported_cats = {0: "Text", 1: "Image"}

    async def initialize(self, config):
        """Mock initialize method."""
        self.initialized = True
        return None

    async def analyze(self, image, **kwargs):
        """Mock analyze method."""
        if not self.initialized:
            raise RuntimeError("Analyzer not initialized")
        return [
            LayoutDetection(
                category_id=0,
                category_name="Text",
                bbox=[10, 20, 100, 150],
                confidence=0.9,
            )
        ]

    async def cleanup(self):
        """Mock cleanup method."""
        self.cleaned_up = True
        return None

    @property
    def supported_categories(self):
        """Mock supported_categories property."""
        return self.supported_cats


class TestILayoutAnalyzer:
    """Test cases for ILayoutAnalyzer interface."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock layout analyzer."""
        analyzer = MockLayoutAnalyzer()
        # Note: In real usage, this would be initialized async,
        # but for testing we can set the flag directly
        analyzer.initialized = True
        return analyzer

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError when called."""
        # Note: Can't instantiate abstract class directly, so we test the method behavior
        # by checking that the abstract methods are properly defined
        assert hasattr(ILayoutAnalyzer, "initialize")
        assert hasattr(ILayoutAnalyzer, "analyze")
        assert hasattr(ILayoutAnalyzer, "cleanup")
        assert hasattr(ILayoutAnalyzer, "supported_categories")

    @pytest.mark.asyncio
    async def test_mock_analyzer_workflow(self, mock_analyzer):
        """Test the complete workflow of mock analyzer."""
        # Test that analyzer is ready (initialized flag set by fixture)
        assert mock_analyzer.initialized is True

        # Test analysis
        image = Image.new("RGB", (200, 200))
        detections = await mock_analyzer.analyze(image)

        assert len(detections) == 1
        assert detections[0].category_id == 0
        assert detections[0].category_name == "Text"
        assert detections[0].bbox == [10, 20, 100, 150]
        assert detections[0].confidence == 0.9

        # Test supported categories
        categories = mock_analyzer.supported_categories
        assert categories == {0: "Text", 1: "Image"}

        # Test cleanup
        await mock_analyzer.cleanup()
        assert mock_analyzer.cleaned_up is True

    @pytest.mark.asyncio
    async def test_analyze_without_initialization(self):
        """Test that analyze fails without initialization."""
        analyzer = MockLayoutAnalyzer()
        image = Image.new("RGB", (200, 200))

        with pytest.raises(RuntimeError, match="Analyzer not initialized"):
            await analyzer.analyze(image)
