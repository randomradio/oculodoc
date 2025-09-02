"""Abstract interface for document processing pipeline."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pathlib import Path


class ProcessingResult:
    """Complete document processing result.

    Attributes:
        total_pages: Total number of pages in the document.
        processing_time: Time taken to process the document in seconds.
        pages: List of processed page data.
        metadata: Document-level metadata.
        errors: List of error messages encountered during processing.
    """

    def __init__(
        self,
        total_pages: int,
        processing_time: float,
        pages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        errors: List[str],
    ):
        """Initialize a ProcessingResult.

        Args:
            total_pages: Total number of pages in the document.
            processing_time: Time taken to process the document in seconds.
            pages: List of processed page data.
            metadata: Document-level metadata.
            errors: List of error messages encountered during processing.
        """
        self.total_pages = total_pages
        self.processing_time = processing_time
        self.pages = pages
        self.metadata = metadata
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert the processing result to a dictionary.

        Returns:
            A dictionary representation of the processing result.
        """
        return {
            "total_pages": self.total_pages,
            "processing_time": self.processing_time,
            "pages": self.pages,
            "metadata": self.metadata,
            "errors": self.errors,
        }

    @property
    def is_successful(self) -> bool:
        """Check if processing was successful (no errors).

        Returns:
            True if no errors occurred during processing, False otherwise.
        """
        return len(self.errors) == 0


class IDocumentProcessor(ABC):
    """Abstract interface for document processing pipeline."""

    @abstractmethod
    async def process_document(
        self, document_path: Path, **kwargs: Any
    ) -> ProcessingResult:
        """Process a document and return structured results.

        Args:
            document_path: Path to the document file.
            **kwargs: Additional keyword arguments for processing.

        Returns:
            A ProcessingResult containing the processed document data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_document method")

    @abstractmethod
    async def process_document_bytes(
        self, document_bytes: bytes, **kwargs: Any
    ) -> ProcessingResult:
        """Process document from bytes.

        Args:
            document_bytes: Raw document bytes.
            **kwargs: Additional keyword arguments for processing.

        Returns:
            A ProcessingResult containing the processed document data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement process_document_bytes method"
        )
