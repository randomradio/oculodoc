"""DocLayout-YOLO layout analyzer implementation."""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import requests

from ..interfaces.layout_analyzer import ILayoutAnalyzer, LayoutDetection
from ..errors import ModelLoadError, InferenceError


class DocLayoutYOLOAnalyzer(ILayoutAnalyzer):
    """DocLayout-YOLO analyzer using ultralytics YOLO for document layout detection.

    This implementation is based on the DocLayout-YOLO model which can detect
    various document elements like titles, text blocks, images, tables, etc.
    """

    # DocLayout-YOLO category mapping
    CATEGORY_MAP = {
        0: "Title",
        1: "Text",
        2: "Abandon",
        3: "ImageBody",
        4: "ImageCaption",
        5: "TableBody",
        6: "TableCaption",
        7: "TableFootnote",
        8: "InterlineEquation_Layout",
        9: "InterlineEquationNumber_Layout",
        13: "InlineEquation",
        14: "InterlineEquation_YOLO",
        15: "OcrText",
        16: "LowScoreText",
    }

    # Default D cLayout-YOLO model from PDF-Extract-Kit
    DEFAULT_MODEL_URL = "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt"
    DEFAULT_MODEL_NAME = "doclayout_yolo_ft.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_batch_size: int = 4,
        model_cache_dir: Optional[str] = None,
    ):
        """Initialize the DocLayout-YOLO analyzer.

        Args:
            model_path: Path to the model file. If None, will download default model.
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps').
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            max_batch_size: Maximum batch size for inference.
            model_cache_dir: Directory to cache downloaded models.
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_batch_size = max_batch_size
        self.model_cache_dir = model_cache_dir or self._get_default_cache_dir()

        self.model: Optional[YOLO] = None
        self.logger = logging.getLogger(__name__)

    def _resolve_device(self, device: str) -> str:
        """Resolve device specification to actual device string."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _get_default_cache_dir(self) -> str:
        """Get default cache directory for models."""
        cache_dir = Path.home() / ".cache" / "oculodoc" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    def _download_model(self, url: str, save_path: Path) -> None:
        """Download model from URL to specified path atomically with verification."""
        self.logger.info(f"Downloading model from {url} to {save_path}")

        temp_path = save_path.with_suffix(save_path.suffix + ".part")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size_header = response.headers.get("content-length")
            total_size = int(total_size_header) if total_size_header else 0

            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded <= total_size:
                        progress = (downloaded / total_size) * 100
                        self.logger.debug(f"Download progress: {progress:.1f}%")

            # Verify size if server provided it
            if total_size > 0 and downloaded != total_size:
                raise ModelLoadError(
                    f"Downloaded size {downloaded} does not match expected {total_size}",
                    model_type="doclayout_yolo",
                )

            # Basic sanity check > 1MB
            if downloaded < 1 * 1024 * 1024:
                raise ModelLoadError(
                    f"Downloaded file too small ({downloaded} bytes)",
                    model_type="doclayout_yolo",
                )

            temp_path.replace(save_path)
            self.logger.info(f"Model downloaded successfully to {save_path}")

        except Exception as e:
            # Clean up temp file on error
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            raise ModelLoadError(
                f"Failed to download model from {url}: {e}", model_type="doclayout_yolo"
            ) from e

    def _ensure_model_available(self) -> str:
        """Ensure model file is available, downloading if necessary.

        Returns:
            Path to the model file as a string.
        """
        # If user provided explicit model path, require it to exist
        if self.model_path:
            user_path = Path(self.model_path)
            if not user_path.exists() or not user_path.is_file():
                raise ModelLoadError(
                    f"Provided model_path does not exist: {user_path}",
                    model_type="doclayout_yolo",
                    model_path=str(user_path),
                )
            return str(user_path)

        # Otherwise use cache location and download default if missing/invalid
        cache_dir = Path(self.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / self.DEFAULT_MODEL_NAME

        self.logger.info(f"Checking for model in {model_path}")

        need_download = False
        if not model_path.exists():
            need_download = True
        else:
            try:
                size = model_path.stat().st_size
                if size < 1 * 1024 * 1024:  # smaller than 1MB likely invalid
                    self.logger.warning(
                        f"Existing model file too small ({size} bytes), will re-download"
                    )
                    need_download = True
            except Exception:
                need_download = True

        if need_download:
            self.logger.info(
                f"DocLayout-YOLO model not found or invalid, downloading from {self.DEFAULT_MODEL_URL}"
            )
            # Remove any existing corrupted file
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception:
                pass
            self._download_model(self.DEFAULT_MODEL_URL, model_path)

        return str(model_path)

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the DocLayout-YOLO analyzer.

        Args:
            config: Configuration dictionary containing model settings.

        Raises:
            ModelLoadError: If model loading fails.
            ConfigurationError: If configuration is invalid.
        """
        self.logger.info(f"Initializing DocLayout-YOLO analyzer with config: {config}")
        try:
            # Update configuration from provided config
            self.conf_threshold = config.get(
                "confidence_threshold", self.conf_threshold
            )
            self.iou_threshold = config.get("iou_threshold", self.iou_threshold)
            self.max_batch_size = config.get("batch_size", self.max_batch_size)

            if "model_path" in config:
                self.model_path = config["model_path"]

            if "device" in config:
                self.device = self._resolve_device(config["device"])

            # Ensure model is available
            model_path = self._ensure_model_available()

            # Load model in a thread to avoid blocking
            def load_model():
                self.model = YOLO(model_path)
                return self.model

            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, load_model)

            self.logger.info(
                f"DocLayout-YOLO model loaded successfully on {self.device}"
            )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize DocLayout-YOLO model: {e}",
                model_type="doclayout_yolo",
                model_path=self.model_path,
            ) from e

    async def analyze(self, image: Image.Image, **kwargs: Any) -> List[LayoutDetection]:
        """Analyze image and return layout detections.

        Args:
            image: PIL Image to analyze.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            List of LayoutDetection objects.

        Raises:
            InferenceError: If inference fails.
        """
        if self.model is None:
            raise InferenceError(
                "Model not initialized. Call initialize() first.",
                model_type="doclayout_yolo",
            )

        try:
            # Convert PIL image to format expected by YOLO
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Prepare image for inference
            img_array = np.array(image)

            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def run_inference():
                results = self.model.predict(
                    img_array,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )
                return results[0] if results else None

            result = await loop.run_in_executor(None, run_inference)

            if result is None:
                return []

            # Convert results to LayoutDetection objects
            detections = []
            for box in result.boxes:
                if box.conf.item() >= self.conf_threshold:
                    # Get bounding box coordinates
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    # Get category information
                    category_id = int(box.cls.item())
                    category_name = self.CATEGORY_MAP.get(
                        category_id, f"Unknown_{category_id}"
                    )

                    detection = LayoutDetection(
                        category_id=category_id,
                        category_name=category_name,
                        bbox=bbox,
                        confidence=box.conf.item(),
                        metadata={
                            "model": "doclayout_yolo",
                            "device": self.device,
                        },
                    )
                    detections.append(detection)

            return detections

        except Exception as e:
            raise InferenceError(
                f"Inference failed: {e}",
                model_type="doclayout_yolo",
                operation="analyze",
            ) from e

    async def cleanup(self) -> None:
        """Cleanup resources (GPU memory, etc.)."""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            # Clear model reference
            self.model = None

        self.logger.info("DocLayout-YOLO analyzer cleaned up")

    @property
    def supported_categories(self) -> Dict[int, str]:
        """Return mapping of category IDs to names."""
        return self.CATEGORY_MAP.copy()

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_type": "doclayout_yolo",
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "supported_categories": self.supported_categories,
        }
