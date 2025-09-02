"""Unit tests for document processor interface."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from oculodoc.interfaces.document_processor import ProcessingResult, IDocumentProcessor


class TestProcessingResult:
    """Test cases for ProcessingResult class."""

    def test_processing_result_creation(self):
        """Test creating a ProcessingResult with valid parameters."""
        pages = [
            {"page_number": 1, "content": "Page 1 content"},
            {"page_number": 2, "content": "Page 2 content"},
        ]
        metadata = {"title": "Test Document", "author": "Test Author"}
        errors = ["Warning: Low confidence detection"]

        result = ProcessingResult(
            total_pages=2,
            processing_time=1.5,
            pages=pages,
            metadata=metadata,
            errors=errors,
        )

        assert result.total_pages == 2
        assert result.processing_time == 1.5
        assert result.pages == pages
        assert result.metadata == metadata
        assert result.errors == errors

    def test_processing_result_to_dict(self):
        """Test converting ProcessingResult to dictionary."""
        pages = [{"page_number": 1, "content": "Test content"}]
        metadata = {"language": "en"}
        errors = []

        result = ProcessingResult(
            total_pages=1,
            processing_time=0.8,
            pages=pages,
            metadata=metadata,
            errors=errors,
        )

        expected_dict = {
            "total_pages": 1,
            "processing_time": 0.8,
            "pages": pages,
            "metadata": metadata,
            "errors": errors,
        }

        assert result.to_dict() == expected_dict

    def test_processing_result_is_successful(self):
        """Test is_successful property."""
        # Test successful result (no errors)
        success_result = ProcessingResult(
            total_pages=1,
            processing_time=1.0,
            pages=[{"content": "test"}],
            metadata={},
            errors=[],
        )
        assert success_result.is_successful is True

        # Test failed result (with errors)
        failed_result = ProcessingResult(
            total_pages=1,
            processing_time=1.0,
            pages=[{"content": "test"}],
            metadata={},
            errors=["Error occurred"],
        )
        assert failed_result.is_successful is False

    def test_processing_result_empty_pages(self):
        """Test ProcessingResult with empty pages."""
        result = ProcessingResult(
            total_pages=0, processing_time=0.0, pages=[], metadata={}, errors=[]
        )

        assert result.total_pages == 0
        assert result.pages == []
        assert result.is_successful is True


class MockDocumentProcessor(IDocumentProcessor):
    """Mock implementation of IDocumentProcessor for testing."""

    def __init__(self):
        self.processed_paths = []
        self.processed_bytes = []

    async def process_document(self, document_path, **kwargs):
        """Mock process_document method."""
        self.processed_paths.append(str(document_path))

        # Simulate processing based on file name
        if "error" in str(document_path).lower():
            return ProcessingResult(
                total_pages=1,
                processing_time=0.5,
                pages=[],
                metadata={},
                errors=["Mock processing error"],
            )
        else:
            return ProcessingResult(
                total_pages=2,
                processing_time=1.2,
                pages=[
                    {"page_number": 1, "content": f"Content from {document_path.name}"},
                    {"page_number": 2, "content": "Page 2 content"},
                ],
                metadata={"source": str(document_path)},
                errors=[],
            )

    async def process_document_bytes(self, document_bytes, **kwargs):
        """Mock process_document_bytes method."""
        self.processed_bytes.append(len(document_bytes))

        # Simulate processing based on bytes length
        if len(document_bytes) > 1000:
            return ProcessingResult(
                total_pages=1,
                processing_time=0.5,
                pages=[],
                metadata={},
                errors=["File too large"],
            )
        else:
            return ProcessingResult(
                total_pages=1,
                processing_time=0.8,
                pages=[
                    {
                        "page_number": 1,
                        "content": f"Processed {len(document_bytes)} bytes",
                    }
                ],
                metadata={"size": len(document_bytes)},
                errors=[],
            )


class TestIDocumentProcessor:
    """Test cases for IDocumentProcessor interface."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock document processor."""
        return MockDocumentProcessor()

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods are properly defined."""
        # Note: Can't instantiate abstract class directly, so we test the method existence
        assert hasattr(IDocumentProcessor, "process_document")
        assert hasattr(IDocumentProcessor, "process_document_bytes")

    @pytest.mark.asyncio
    async def test_mock_processor_document_path(self, mock_processor, tmp_path):
        """Test processing a document from file path."""
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("mock pdf content")

        result = await mock_processor.process_document(test_file)

        assert result.total_pages == 2
        assert result.processing_time == 1.2
        assert len(result.pages) == 2
        assert result.pages[0]["content"] == "Content from test.pdf"
        assert result.metadata["source"] == str(test_file)
        assert result.is_successful is True
        assert str(test_file) in mock_processor.processed_paths

    @pytest.mark.asyncio
    async def test_mock_processor_error_document(self, mock_processor, tmp_path):
        """Test processing a document that causes an error."""
        error_file = tmp_path / "error_document.pdf"
        error_file.write_text("error content")

        result = await mock_processor.process_document(error_file)

        assert result.total_pages == 1
        assert result.errors == ["Mock processing error"]
        assert result.is_successful is False

    @pytest.mark.asyncio
    async def test_mock_processor_bytes(self, mock_processor):
        """Test processing document from bytes."""
        test_bytes = b"This is test document content"
        result = await mock_processor.process_document_bytes(test_bytes)

        assert result.total_pages == 1
        assert result.processing_time == 0.8
        assert len(result.pages) == 1
        assert "Processed 29 bytes" in result.pages[0]["content"]
        assert result.metadata["size"] == 29
        assert result.is_successful is True
        assert 29 in mock_processor.processed_bytes

    @pytest.mark.asyncio
    async def test_mock_processor_large_bytes(self, mock_processor):
        """Test processing large document bytes."""
        large_bytes = b"x" * 1500  # Larger than 1000 bytes
        result = await mock_processor.process_document_bytes(large_bytes)

        assert result.errors == ["File too large"]
        assert result.is_successful is False
        assert 1500 in mock_processor.processed_bytes
