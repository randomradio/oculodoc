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
from torch.serialization import add_safe_globals

# Optional MinerU-style backend
try:
    from doclayout_yolo import YOLOv10 as MinerUDocLayoutYOLO
except Exception:
    MinerUDocLayoutYOLO = None

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

        self.model: Optional[object] = None
        self._loaded_model_path: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        self._backend: str = "ultralytics"

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
        print(f"Initializing DocLayout-YOLO analyzer with config: {config}")
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

            # Allowlist doclayout_yolo custom classes for safe torch.load (PyTorch >= 2.6)
            try:
                import doclayout_yolo.nn.tasks as dl_tasks  # type: ignore

                try:
                    add_safe_globals([dl_tasks.YOLOv10DetectionModel])  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    # Some versions expose a base YOLOv10Model; allowlist defensively
                    add_safe_globals([dl_tasks.YOLOv10Model])  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                # If the module is not available yet, YOLO may still handle load if weights are standard
                pass

            # Also allowlist common torch containers seen in pickled checkpoints
            try:
                import torch.nn.modules.container as torch_container  # type: ignore

                add_safe_globals([torch_container.Sequential])  # type: ignore[attr-defined]
            except Exception:
                pass

            # Ensure model is available
            model_path = self._ensure_model_available()

            # Load model in a thread to avoid blocking
            def load_model():
                # Validate file exists right before load
                mp = Path(model_path)
                if not mp.exists() or not mp.is_file():
                    raise ModelLoadError(
                        f"Model file not found at load time: {mp}",
                        model_type="doclayout_yolo",
                        model_path=str(mp),
                    )
                # Prefer MinerU backend if available
                if MinerUDocLayoutYOLO is not None:
                    try:
                        model = MinerUDocLayoutYOLO(str(mp))
                        # Move to desired device if supported
                        try:
                            model = model.to(self.device)
                        except Exception:
                            pass
                        self.model = model
                        self._backend = "mineru"
                        self._loaded_model_path = str(mp)
                        return self.model
                    except Exception as e:
                        # Fallback to Ultralytics if MinerU path fails
                        self.logger.warning(
                            f"MinerU backend failed to load, falling back to Ultralytics: {e}"
                        )

                # Ultralytics fallback
                self.model = YOLO(str(mp))
                self._backend = "ultralytics"
                self._loaded_model_path = str(mp)
                return self.model

            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, load_model)

            self.logger.info(
                f"DocLayout-YOLO model loaded successfully on {self.device} from {self._loaded_model_path}"
            )

        except ModelLoadError:
            # Propagate explicit model load errors without masking
            raise
        except Exception as e:
            # Wrap any other unexpected error
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
                # Some backends may already filter by conf; still gate here
                score_val = float(box.conf.item())
                if score_val >= self.conf_threshold:
                    bbox = [float(v) for v in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
                    category_id = int(box.cls.item())
                    category_name = self.CATEGORY_MAP.get(
                        category_id, f"Unknown_{category_id}"
                    )

                    detections.append(
                        LayoutDetection(
                            category_id=category_id,
                            category_name=category_name,
                            bbox=bbox,
                            confidence=score_val,
                            metadata={
                                "model": "doclayout_yolo",
                                "device": self.device,
                                "backend": self._backend,
                            },
                        )
                    )

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
            return {
                "status": "not_loaded",
                "configured_model_path": self.model_path,
                "loaded_model_path": None,
            }

        return {
            "status": "loaded",
            "model_type": "doclayout_yolo",
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "supported_categories": self.supported_categories,
            "configured_model_path": self.model_path,
            "loaded_model_path": self._loaded_model_path,
        }
