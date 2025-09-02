"""Base document processor implementation."""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from ..interfaces import (
    IDocumentProcessor,
    ProcessingResult,
    ILayoutAnalyzer,
    IVLMAnalyzer,
)
from ..config import OculodocConfig
from ..errors import ProcessingError, ValidationError


class BaseDocumentProcessor(IDocumentProcessor):
    """Base implementation of document processor.

    This provides common functionality for document processing pipelines
    without depending on specific model implementations.
    """

    def __init__(
        self,
        config: OculodocConfig,
        layout_analyzer: Optional[ILayoutAnalyzer] = None,
        vlm_analyzer: Optional[IVLMAnalyzer] = None,
    ):
        """Initialize the document processor.

        Args:
            config: Configuration for the processor.
            layout_analyzer: Optional layout analyzer implementation.
            vlm_analyzer: Optional VLM analyzer implementation.
        """
        self.config = config
        self.layout_analyzer = layout_analyzer
        self.vlm_analyzer = vlm_analyzer

    def _validate_document_path(self, document_path: Path) -> None:
        """Validate document path.

        Args:
            document_path: Path to the document.

        Raises:
            ValidationError: If the document path is invalid.
        """
        if not document_path.exists():
            raise ValidationError(
                f"Document does not exist: {document_path}",
                field="document_path",
                value=str(document_path),
            )

        if not document_path.is_file():
            raise ValidationError(
                f"Path is not a file: {document_path}",
                field="document_path",
                value=str(document_path),
            )

        # Check file size
        file_size_mb = document_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.processing.max_document_size_mb:
            raise ValidationError(
                f"Document too large: {file_size_mb:.2f}MB (max: {self.config.processing.max_document_size_mb}MB)",
                field="file_size",
                value=f"{file_size_mb:.2f}MB",
            )

        # Check file extension (basic validation)
        allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        if document_path.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"Unsupported file format: {document_path.suffix}",
                field="file_extension",
                value=document_path.suffix,
            )

    def _validate_document_bytes(self, document_bytes: bytes) -> None:
        """Validate document bytes.

        Args:
            document_bytes: Raw document bytes.

        Raises:
            ValidationError: If the document bytes are invalid.
        """
        if not document_bytes:
            raise ValidationError(
                "Document bytes are empty", field="document_bytes", value="empty"
            )

        # Check size
        size_mb = len(document_bytes) / (1024 * 1024)
        if size_mb > self.config.processing.max_document_size_mb:
            raise ValidationError(
                f"Document too large: {size_mb:.2f}MB (max: {self.config.processing.max_document_size_mb}MB)",
                field="document_size",
                value=f"{size_mb:.2f}MB",
            )

    def _extract_basic_metadata(self, document_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from document path.

        Args:
            document_path: Path to the document.

        Returns:
            Dictionary containing basic metadata.
        """
        stat = document_path.stat()
        return {
            "filename": document_path.name,
            "file_path": str(document_path),
            "file_size_bytes": stat.st_size,
            "file_size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "file_extension": document_path.suffix.lower(),
        }

    async def _process_document_with_layout(
        self, document_path: Path, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Process document with layout analysis.

        Args:
            document_path: Path to the document.
            **kwargs: Additional processing arguments.

        Returns:
            List of processed pages with layout information.
        """
        if self.layout_analyzer is None:
            raise ProcessingError(
                "Layout analyzer not configured",
                document_path=str(document_path),
                stage="layout_analysis",
            )

        # If the document is an image, run real layout analysis via analyzer
        image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        if document_path.suffix.lower() in image_exts:
            image = Image.open(document_path)

            detections = await self.layout_analyzer.analyze(image, **kwargs)

            # Map category names to simplified block types used by middle.json
            def _map_category_to_type(category_name: str) -> str:
                name = category_name.lower()
                if "title" in name:
                    return "title"
                if "table" in name:
                    return "table"
                if "image" in name or "figure" in name:
                    return "figure"
                return "text"

            width, height = image.size
            layout_elements: List[Dict[str, Any]] = []
            for det in detections:
                layout_elements.append(
                    {
                        "type": _map_category_to_type(det.category_name),
                        "bbox": det.bbox,
                        "content": "",  # content extraction handled by VLM stage
                        "confidence": det.confidence,
                        "category_id": det.category_id,
                    }
                )

            return [
                {
                    "page_number": 1,
                    "layout_elements": layout_elements,
                    "image_path": str(document_path),
                    "width": width,
                    "height": height,
                }
            ]

        # Fallback mock structure for non-image inputs (e.g., PDFs not yet rasterized)
        return [
            {
                "page_number": 1,
                "layout_elements": [
                    {
                        "type": "text",
                        "bbox": [10, 20, 200, 50],
                        "content": "Sample text content",
                        "confidence": 0.95,
                    }
                ],
                "image_path": str(document_path),
                "width": 800,
                "height": 600,
            }
        ]

    async def _process_document_with_vlm(
        self, document_path: Path, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Process document with VLM analysis.

        Args:
            document_path: Path to the document.
            **kwargs: Additional processing arguments.

        Returns:
            List of processed pages with VLM analysis.
        """
        if self.vlm_analyzer is None:
            raise ProcessingError(
                "VLM analyzer not configured",
                document_path=str(document_path),
                stage="vlm_analysis",
            )

        # For now, return a mock page structure
        # In a real implementation, this would:
        # 1. Convert PDF to images
        # 2. Run VLM analysis with prompts
        # 3. Extract content
        return [
            {
                "page_number": 1,
                "content": "Extracted text content from VLM analysis",
                "structured_data": {
                    "title": "Document Title",
                    "author": "Document Author",
                    "summary": "Document summary...",
                },
                "confidence": 0.88,
            }
        ]

    async def _process_document_hybrid(
        self, document_path: Path, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Process document with hybrid layout + VLM analysis.

        Args:
            document_path: Path to the document.
            **kwargs: Additional processing arguments.

        Returns:
            List of processed pages with combined analysis.
        """
        if self.layout_analyzer is None or self.vlm_analyzer is None:
            raise ProcessingError(
                "Both layout and VLM analyzers required for hybrid processing",
                document_path=str(document_path),
                stage="hybrid_analysis",
            )

        # For now, return a mock page structure
        # In a real implementation, this would combine both analyses
        return [
            {
                "page_number": 1,
                "layout_elements": [
                    {
                        "type": "text",
                        "bbox": [10, 20, 200, 50],
                        "content": "Enhanced text content from hybrid analysis",
                        "confidence": 0.92,
                    }
                ],
                "vlm_content": "Additional content extracted by VLM",
                "structured_data": {
                    "title": "Document Title",
                    "sections": ["Introduction", "Body", "Conclusion"],
                },
                "combined_confidence": 0.90,
            }
        ]

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
            ProcessingError: If processing fails.
            ValidationError: If document validation fails.
        """
        start_time = time.time()

        try:
            # Validate document
            self._validate_document_path(document_path)

            # Extract basic metadata
            metadata = self._extract_basic_metadata(document_path)

            # Choose processing method based on configuration
            if (
                self.config.processing.use_hybrid
                and self.layout_analyzer is not None
                and self.vlm_analyzer is not None
            ):
                pages = await self._process_document_hybrid(document_path, **kwargs)
            elif self.layout_analyzer is not None:
                pages = await self._process_document_with_layout(
                    document_path, **kwargs
                )
            elif self.vlm_analyzer is not None:
                pages = await self._process_document_with_vlm(document_path, **kwargs)
            else:
                # Fallback: basic processing without models
                pages = [
                    {
                        "page_number": 1,
                        "content": f"Basic processing of {document_path.name}",
                        "note": "No analyzers configured",
                    }
                ]

            processing_time = time.time() - start_time

            return ProcessingResult(
                total_pages=len(pages),
                processing_time=processing_time,
                pages=pages,
                metadata=metadata,
                errors=[],
            )

        except (ValidationError, ProcessingError):
            # Re-raise validation and processing errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            processing_time = time.time() - start_time
            raise ProcessingError(
                f"Unexpected error during processing: {e}",
                document_path=str(document_path),
                stage="unknown",
            ) from e

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
            ProcessingError: If processing fails.
            ValidationError: If document validation fails.
        """
        start_time = time.time()

        try:
            # Validate document bytes
            self._validate_document_bytes(document_bytes)

            # Basic metadata for bytes
            metadata = {
                "source": "bytes",
                "size_bytes": len(document_bytes),
                "size_mb": len(document_bytes) / (1024 * 1024),
            }

            # For bytes processing, we can only do basic analysis
            # In a real implementation, this would save bytes to temp file
            # and then process like a regular file
            pages = [
                {
                    "page_number": 1,
                    "content": f"Processed {len(document_bytes)} bytes",
                    "note": "Basic bytes processing - no model analysis available",
                }
            ]

            processing_time = time.time() - start_time

            return ProcessingResult(
                total_pages=len(pages),
                processing_time=processing_time,
                pages=pages,
                metadata=metadata,
                errors=[],
            )

        except (ValidationError, ProcessingError):
            # Re-raise validation and processing errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            processing_time = time.time() - start_time
            raise ProcessingError(
                f"Unexpected error during bytes processing: {e}", stage="unknown"
            ) from e

    def generate_middle_json(
        self, result: ProcessingResult, document_path: Path
    ) -> Dict[str, Any]:
        """Generate MinerU-style middle.json output.

        Args:
            result: ProcessingResult from document processing.
            document_path: Original document path.

        Returns:
            Dictionary in MinerU middle.json format.
        """
        # Build middle.json structure similar to MinerU
        middle_json = {
            "doc_layout_result": {"layout_dets": [], "page_info": []},
            "text_blocks": [],
            "title_blocks": [],
            "figure_blocks": [],
            "table_blocks": [],
            "footnote_blocks": [],
            "page_images": [],
            "meta_data": {
                "doc_name": document_path.name,
                "doc_path": str(document_path),
                "total_pages": result.total_pages,
                "processing_time": result.processing_time,
                "model_info": {
                    "layout_analyzer": (
                        self.layout_analyzer.__class__.__name__
                        if self.layout_analyzer
                        else None
                    ),
                    "vlm_analyzer": (
                        self.vlm_analyzer.__class__.__name__
                        if self.vlm_analyzer
                        else None
                    ),
                },
            },
            "version": "oculodoc-1.0",
            "timestamp": time.time(),
        }

        # Process each page
        for page_idx, page_data in enumerate(result.pages):
            page_info = {
                "page_no": page_idx + 1,
                "height": page_data.get("height", 0),
                "width": page_data.get("width", 0),
                "layout_dets": [],
            }

            # Add layout detections for this page if available
            if "layout_elements" in page_data:
                for element in page_data["layout_elements"]:
                    layout_det = {
                        "category_id": element.get("category_id", 0),
                        "type": element.get("type", "text"),
                        "bbox": element.get("bbox", [0, 0, 0, 0]),
                        "score": element.get("confidence", 0.0),
                    }
                    page_info["layout_dets"].append(layout_det)

                    # Also add to appropriate block lists
                    block_type = element.get("type", "text")
                    if block_type == "text":
                        middle_json["text_blocks"].append(
                            {
                                "page_no": page_idx + 1,
                                "bbox": element.get("bbox", [0, 0, 0, 0]),
                                "content": element.get("content", ""),
                                "score": element.get("confidence", 0.0),
                            }
                        )
                    elif block_type == "title":
                        middle_json["title_blocks"].append(
                            {
                                "page_no": page_idx + 1,
                                "bbox": element.get("bbox", [0, 0, 0, 0]),
                                "content": element.get("content", ""),
                                "score": element.get("confidence", 0.0),
                            }
                        )
                    elif block_type == "figure":
                        middle_json["figure_blocks"].append(
                            {
                                "page_no": page_idx + 1,
                                "bbox": element.get("bbox", [0, 0, 0, 0]),
                                "score": element.get("confidence", 0.0),
                            }
                        )
                    elif block_type == "table":
                        middle_json["table_blocks"].append(
                            {
                                "page_no": page_idx + 1,
                                "bbox": element.get("bbox", [0, 0, 0, 0]),
                                "score": element.get("confidence", 0.0),
                            }
                        )

            middle_json["doc_layout_result"]["page_info"].append(page_info)

        return middle_json
