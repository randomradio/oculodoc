"""Hybrid document processor: YOLO layout + VLM region recognition.

This processor runs layout detection to get regions and then uses the VLM
to recognize content per region (OCR for text/title, Markdown for tables,
short caption or OCR for figures). It returns page data with enriched
layout elements containing recognized content.
"""

import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64

from PIL import Image
import fitz  # PyMuPDF

from ..interfaces import (
    ProcessingResult,
    ILayoutAnalyzer,
    IVLMAnalyzer,
)
from ..config import OculodocConfig
from ..errors import ProcessingError, ValidationError
from .base_processor import BaseDocumentProcessor


class HybridDocumentProcessor(BaseDocumentProcessor):
    """Processor that combines layout detection with per-region VLM OCR."""

    def __init__(
        self,
        config: OculodocConfig,
        layout_analyzer: Optional[ILayoutAnalyzer],
        vlm_analyzer: Optional[IVLMAnalyzer],
    ) -> None:
        super().__init__(config, layout_analyzer, vlm_analyzer)

    def _encode_image(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _prompt_for_type(self, block_type: str) -> str:
        t = (block_type or "text").lower()
        if t == "table":
            return (
                "Extract the table as GitHub-flavored Markdown with correct rows and columns. "
                "Return only the Markdown table."
            )
        if t == "title":
            return "Transcribe the title text exactly. Return only the text."
        if t == "figure":
            return (
                "If there is visible text, transcribe it. Otherwise provide a concise caption. "
                "Return plain text."
            )
        return "Transcribe the text exactly. Return only the text."

    def _crop_bbox(self, page_image: Image.Image, bbox: List[float]) -> Image.Image:
        # bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        # Ensure integer pixel bounds within image
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(page_image.width, int(x2))
        y2 = min(page_image.height, int(y2))
        if x2 <= x1 or y2 <= y1:
            # Fallback to minimal 1x1 crop inside image
            x2 = min(page_image.width, x1 + 1)
            y2 = min(page_image.height, y1 + 1)
        return page_image.crop((x1, y1, x2, y2))

    async def _process_page_hybrid(
        self,
        page_image: Image.Image,
        page_number: int,
        layout_conf_threshold: float,
    ) -> Dict[str, Any]:
        assert self.layout_analyzer is not None
        assert self.vlm_analyzer is not None

        # Run layout on the page image
        detections = await self.layout_analyzer.analyze(page_image)

        width, height = page_image.size
        enriched_elements: List[Dict[str, Any]] = []

        for det in detections:
            if det.confidence < layout_conf_threshold:
                continue
            block_type = det.category_name
            region_img = self._crop_bbox(page_image, det.bbox)
            encoded = self._encode_image(region_img)
            prompt = self._prompt_for_type(block_type)

            try:
                results = await self.vlm_analyzer.analyze(
                    encoded,
                    prompt,
                    max_tokens=self.config.vlm.max_tokens,
                    temperature=self.config.vlm.temperature,
                    timeout=self.config.vlm.timeout,
                )
                # Prefer text content
                content_parts = [r.content for r in results if r.content]
                region_text = "\n".join(content_parts).strip()
                best_conf = max((r.confidence for r in results), default=0.0)
            except Exception as e:
                region_text = f"[VLM error: {e}]"
                best_conf = 0.0

            enriched_elements.append(
                {
                    "type": block_type.lower(),
                    "bbox": det.bbox,
                    "content": region_text,
                    "confidence": max(det.confidence, best_conf),
                    "category_id": det.category_id,
                }
            )

        return {
            "page_number": page_number,
            "layout_elements": enriched_elements,
            "image_path": None,
            "width": width,
            "height": height,
        }

    async def process_document(
        self, document_path: Path, **kwargs: Any
    ) -> ProcessingResult:
        """Override to run hybrid per page: layout + region VLM."""
        if self.layout_analyzer is None or self.vlm_analyzer is None:
            raise ProcessingError(
                "Both layout and VLM analyzers required for hybrid processing",
                document_path=str(document_path),
                stage="hybrid_analysis",
            )

        start_time = time.time()

        try:
            self._validate_document_path(document_path)
            metadata = self._extract_basic_metadata(document_path)

            pages: List[Dict[str, Any]] = []
            layout_conf_threshold = self.config.processing.layout_confidence_threshold

            if document_path.suffix.lower() == ".pdf":
                doc = fitz.open(str(document_path))
                try:
                    for idx in range(doc.page_count):
                        page = doc.load_page(idx)
                        pix = page.get_pixmap(dpi=150)
                        if pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_bytes = pix.tobytes("png")
                        image = Image.open(BytesIO(img_bytes))
                        page_result = await self._process_page_hybrid(
                            image, idx + 1, layout_conf_threshold
                        )
                        pages.append(page_result)
                finally:
                    doc.close()
            else:
                image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
                if document_path.suffix.lower() in image_exts:
                    image = Image.open(document_path)
                    page_result = await self._process_page_hybrid(
                        image, 1, layout_conf_threshold
                    )
                    pages.append(page_result)
                else:
                    raise ValidationError(
                        f"Unsupported file format for hybrid: {document_path.suffix}",
                        field="file_extension",
                        value=document_path.suffix,
                    )

            processing_time = time.time() - start_time

            return ProcessingResult(
                total_pages=len(pages),
                processing_time=processing_time,
                pages=pages,
                metadata=metadata,
                errors=[],
            )

        except (ValidationError, ProcessingError):
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            raise ProcessingError(
                f"Unexpected error during hybrid processing: {e}",
                document_path=str(document_path),
                stage="hybrid_analysis",
            ) from e
