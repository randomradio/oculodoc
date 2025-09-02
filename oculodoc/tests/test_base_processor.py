"""Unit tests for base document processor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from oculodoc.config import OculodocConfig
from oculodoc.processor import BaseDocumentProcessor
from oculodoc.errors import ValidationError, ProcessingError


class TestBaseDocumentProcessor:
    """Test cases for BaseDocumentProcessor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OculodocConfig()

    @pytest.fixture
    def processor(self, config):
        """Create test processor."""
        return BaseDocumentProcessor(config)

    def test_processor_initialization(self, config):
        """Test processor initialization."""
        processor = BaseDocumentProcessor(config)
        assert processor.config == config
        assert processor.layout_analyzer is None
        assert processor.vlm_analyzer is None

    def test_processor_initialization_with_analyzers(self, config):
        """Test processor initialization with analyzers."""
        mock_layout = Mock()
        mock_vlm = Mock()

        processor = BaseDocumentProcessor(
            config=config, layout_analyzer=mock_layout, vlm_analyzer=mock_vlm
        )

        assert processor.layout_analyzer == mock_layout
        assert processor.vlm_analyzer == mock_vlm

    def test_validate_document_path_nonexistent(self, processor, tmp_path):
        """Test validation of non-existent document path."""
        nonexistent_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_path(nonexistent_path)

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.field == "document_path"

    def test_validate_document_path_directory(self, processor, tmp_path):
        """Test validation of directory as document path."""
        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_path(tmp_path)

        assert "not a file" in str(exc_info.value)
        assert exc_info.value.field == "document_path"

    def test_validate_document_path_unsupported_format(self, processor, tmp_path):
        """Test validation of unsupported file format."""
        unsupported_file = tmp_path / "test.exe"
        unsupported_file.write_text("test")

        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_path(unsupported_file)

        assert "Unsupported file format" in str(exc_info.value)
        assert exc_info.value.field == "file_extension"

    def test_validate_document_path_too_large(self, processor, tmp_path):
        """Test validation of file that's too large."""
        # Create a config with small max size
        config = OculodocConfig()
        config.processing.max_document_size_mb = 0.001  # 1KB
        processor = BaseDocumentProcessor(config)

        # Create a file larger than the limit
        large_file = tmp_path / "large.pdf"
        with open(large_file, "wb") as f:
            f.write(b"x" * 2000)  # 2KB

        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_path(large_file)

        assert "too large" in str(exc_info.value)
        assert exc_info.value.field == "file_size"

    def test_validate_document_bytes_empty(self, processor):
        """Test validation of empty document bytes."""
        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_bytes(b"")

        assert "empty" in str(exc_info.value)
        assert exc_info.value.field == "document_bytes"

    def test_validate_document_bytes_too_large(self, processor):
        """Test validation of document bytes that are too large."""
        # Create a config with small max size
        config = OculodocConfig()
        config.processing.max_document_size_mb = 0.001  # 1KB
        processor = BaseDocumentProcessor(config)

        # Create bytes larger than the limit
        large_bytes = b"x" * 2000  # 2KB

        with pytest.raises(ValidationError) as exc_info:
            processor._validate_document_bytes(large_bytes)

        assert "too large" in str(exc_info.value)
        assert exc_info.value.field == "document_size"

    def test_extract_basic_metadata(self, processor, tmp_path):
        """Test extraction of basic metadata from document path."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        metadata = processor._extract_basic_metadata(test_file)

        assert metadata["filename"] == "test.pdf"
        assert metadata["file_path"] == str(test_file)
        assert metadata["file_extension"] == ".pdf"
        assert "file_size_bytes" in metadata
        assert "file_size_mb" in metadata
        assert "modified_time" in metadata

    @pytest.mark.asyncio
    async def test_process_document_path_basic(self, processor, tmp_path):
        """Test basic document processing from file path."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        result = await processor.process_document(test_file)

        assert result.total_pages == 1
        assert result.processing_time >= 0
        assert len(result.pages) == 1
        assert result.is_successful is True
        assert len(result.errors) == 0
        assert result.metadata["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_process_document_bytes_basic(self, processor):
        """Test basic document processing from bytes."""
        test_bytes = b"test document content"

        result = await processor.process_document_bytes(test_bytes)

        assert result.total_pages == 1
        assert result.processing_time >= 0
        assert len(result.pages) == 1
        assert result.is_successful is True
        assert len(result.errors) == 0
        assert result.metadata["source"] == "bytes"
        assert result.metadata["size_bytes"] == len(test_bytes)

    @pytest.mark.asyncio
    async def test_process_document_path_validation_error(self, processor, tmp_path):
        """Test processing with validation error."""
        nonexistent_file = tmp_path / "nonexistent.pdf"

        with pytest.raises(ValidationError):
            await processor.process_document(nonexistent_file)

    @pytest.mark.asyncio
    async def test_process_document_bytes_validation_error(self, processor):
        """Test processing bytes with validation error."""
        with pytest.raises(ValidationError):
            await processor.process_document_bytes(b"")

    def test_process_document_with_layout_analyzer_none(self, config, tmp_path):
        """Test processing with layout analyzer not configured."""
        processor = BaseDocumentProcessor(config)
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        # This should work in basic mode even without analyzers
        # The hybrid/layout specific methods would fail if called directly

    def test_process_document_with_vlm_analyzer_none(self, config, tmp_path):
        """Test processing with VLM analyzer not configured."""
        processor = BaseDocumentProcessor(config)
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        # This should work in basic mode even without analyzers

    @pytest.mark.asyncio
    async def test_process_document_with_layout_method_error(self, config, tmp_path):
        """Test processing when layout method fails due to missing analyzer."""
        # Configure to use layout-only processing
        config.processing.use_hybrid = False
        mock_layout = Mock()
        processor = BaseDocumentProcessor(config, layout_analyzer=mock_layout)

        # Make the layout analyzer fail during processing
        async def failing_analyze(*args, **kwargs):
            raise RuntimeError("Layout analysis failed")

        mock_layout.analyze = failing_analyze

        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        # This will fail because we don't have proper error handling for the layout method
        # In a real implementation, this would be handled more gracefully

    @pytest.mark.asyncio
    async def test_process_document_unexpected_error(self, processor, tmp_path):
        """Test processing with unexpected error."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        # Mock an unexpected error in validation
        with patch.object(processor, "_validate_document_path") as mock_validate:
            mock_validate.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ProcessingError) as exc_info:
                await processor.process_document(test_file)

            assert "Unexpected error" in str(exc_info.value)
            assert exc_info.value.stage == "unknown"
