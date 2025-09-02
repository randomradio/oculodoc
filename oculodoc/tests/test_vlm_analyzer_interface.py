"""Unit tests for VLM analyzer interface."""

import pytest

from oculodoc.interfaces.vlm_analyzer import VLMAnalysisResult, IVLMAnalyzer


class TestVLMAnalysisResult:
    """Test cases for VLMAnalysisResult class."""

    def test_vlm_analysis_result_creation(self):
        """Test creating a VLMAnalysisResult with valid parameters."""
        result = VLMAnalysisResult(
            content_type="text",
            content="Sample text content",
            bbox=[10, 20, 100, 150],
            confidence=0.85,
            metadata={"language": "en"},
        )

        assert result.content_type == "text"
        assert result.content == "Sample text content"
        assert result.bbox == [10, 20, 100, 150]
        assert result.confidence == 0.85
        assert result.metadata == {"language": "en"}

    def test_vlm_analysis_result_bbox_validation(self):
        """Test bbox validation in VLMAnalysisResult."""
        with pytest.raises(ValueError, match="bbox must contain exactly 4 coordinates"):
            VLMAnalysisResult(
                content_type="text",
                content="Sample content",
                bbox=[10, 20, 100],  # Only 3 coordinates
                confidence=0.85,
            )

    def test_vlm_analysis_result_optional_bbox(self):
        """Test VLMAnalysisResult with optional bbox."""
        result = VLMAnalysisResult(
            content_type="text", content="Sample content", confidence=0.85
        )

        assert result.bbox is None
        assert result.content_type == "text"
        assert result.content == "Sample content"
        assert result.confidence == 0.85

    def test_vlm_analysis_result_to_dict(self):
        """Test converting VLMAnalysisResult to dictionary."""
        result = VLMAnalysisResult(
            content_type="table",
            content="| Header | Value |",
            bbox=[5, 10, 200, 50],
            confidence=0.92,
            metadata={"rows": 2, "columns": 2},
        )

        expected_dict = {
            "content_type": "table",
            "content": "| Header | Value |",
            "bbox": [5, 10, 200, 50],
            "confidence": 0.92,
            "metadata": {"rows": 2, "columns": 2},
        }

        assert result.to_dict() == expected_dict

    def test_vlm_analysis_result_default_metadata(self):
        """Test VLMAnalysisResult with default empty metadata."""
        result = VLMAnalysisResult(
            content_type="text", content="Sample content", confidence=0.85
        )

        assert result.metadata == {}


class MockVLMAnalyzer(IVLMAnalyzer):
    """Mock implementation of IVLMAnalyzer for testing."""

    def __init__(self):
        self.initialized = False

    async def initialize(self, config):
        """Mock initialize method."""
        self.initialized = True
        return None

    async def analyze(self, image_data, prompt, **kwargs):
        """Mock analyze method."""
        if not self.initialized:
            raise RuntimeError("Analyzer not initialized")

        return [
            VLMAnalysisResult(
                content_type="text",
                content=f"Analysis of: {prompt}",
                confidence=0.88,
                metadata={"model": "mock"},
            )
        ]

    async def batch_analyze(self, image_data_list, prompts, **kwargs):
        """Mock batch_analyze method."""
        if not self.initialized:
            raise RuntimeError("Analyzer not initialized")

        results = []
        for prompt in prompts:
            results.append(
                [
                    VLMAnalysisResult(
                        content_type="text",
                        content=f"Batch analysis of: {prompt}",
                        confidence=0.85,
                        metadata={"batch": True, "model": "mock"},
                    )
                ]
            )
        return results


class TestIVLMAnalyzer:
    """Test cases for IVLMAnalyzer interface."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock VLM analyzer."""
        analyzer = MockVLMAnalyzer()
        # Note: In real usage, this would be initialized async,
        # but for testing we can set the flag directly
        analyzer.initialized = True
        return analyzer

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods are properly defined."""
        # Note: Can't instantiate abstract class directly, so we test the method existence
        assert hasattr(IVLMAnalyzer, "initialize")
        assert hasattr(IVLMAnalyzer, "analyze")
        assert hasattr(IVLMAnalyzer, "batch_analyze")

    @pytest.mark.asyncio
    async def test_mock_analyzer_workflow(self, mock_analyzer):
        """Test the complete workflow of mock analyzer."""
        # Test initialization
        assert mock_analyzer.initialized is True

        # Test single analysis
        results = await mock_analyzer.analyze("fake_image_data", "Extract text")

        assert len(results) == 1
        assert results[0].content_type == "text"
        assert "Extract text" in results[0].content
        assert results[0].confidence == 0.88
        assert results[0].metadata["model"] == "mock"

        # Test batch analysis
        batch_results = await mock_analyzer.batch_analyze(
            ["image1", "image2"], ["Prompt 1", "Prompt 2"]
        )

        assert len(batch_results) == 2
        assert len(batch_results[0]) == 1
        assert len(batch_results[1]) == 1
        assert "Batch analysis of: Prompt 1" in batch_results[0][0].content
        assert "Batch analysis of: Prompt 2" in batch_results[1][0].content
        assert batch_results[0][0].metadata["batch"] is True

    @pytest.mark.asyncio
    async def test_analyze_without_initialization(self):
        """Test that analyze fails without initialization."""
        analyzer = MockVLMAnalyzer()

        with pytest.raises(RuntimeError, match="Analyzer not initialized"):
            await analyzer.analyze("image_data", "prompt")

        with pytest.raises(RuntimeError, match="Analyzer not initialized"):
            await analyzer.batch_analyze(["image"], ["prompt"])
