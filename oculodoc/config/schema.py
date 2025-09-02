"""Configuration schemas for Oculodoc components."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    """Configuration for logging.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format string.
        enable_structured: Whether to use structured logging.
        log_file: Optional path to log file.
    """

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_structured: bool = True
    log_file: Optional[str] = None


@dataclass
class MetricsConfig:
    """Configuration for metrics collection.

    Attributes:
        enabled: Whether metrics collection is enabled.
        port: Port for metrics server.
        host: Host for metrics server.
        prefix: Prefix for metric names.
    """

    enabled: bool = True
    port: int = 8001
    host: str = "localhost"
    prefix: str = "oculodoc"


@dataclass
class LayoutConfig:
    """Configuration for layout analysis models.

    Attributes:
        model_type: Type of layout model ("doclayout_yolo", "generic_yolo", "detr").
        model_path: Path to the model file.
        device: Device to run model on ("cuda", "cpu").
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        batch_size: Batch size for inference.
        max_batch_delay: Maximum delay for batching in seconds.
    """

    model_type: str = "doclayout_yolo"
    model_path: str = ""
    device: str = "cuda"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 4
    max_batch_delay: float = 0.1


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Models.

    Attributes:
        model_type: Type of VLM ("sglang_ocrflux", "transformers_qwen").
        model_path: Path to the model.
        host: Host for VLM service.
        port: Port for VLM service.
        max_tokens: Maximum tokens for generation.
        temperature: Temperature for generation.
        timeout: Request timeout in seconds.
    """

    model_type: str = "sglang_ocrflux"
    model_path: str = ""
    host: str = "localhost"
    port: int = 30000
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: float = 30.0


@dataclass
class ProcessingConfig:
    """Configuration for document processing.

    Attributes:
        use_hybrid: Whether to use hybrid processing (layout + VLM).
        max_concurrent_pages: Maximum concurrent pages to process.
        layout_confidence_threshold: Confidence threshold for layout analysis.
        vlm_confidence_threshold: Confidence threshold for VLM analysis.
        enable_visualization: Whether to generate visualization outputs.
        max_document_size_mb: Maximum document size in MB.
    """

    use_hybrid: bool = True
    max_concurrent_pages: int = 4
    layout_confidence_threshold: float = 0.5
    vlm_confidence_threshold: float = 0.7
    enable_visualization: bool = False
    max_document_size_mb: int = 100


@dataclass
class CacheConfig:
    """Configuration for model caching.

    Attributes:
        enabled: Whether caching is enabled.
        max_memory_gb: Maximum memory for cache in GB.
        ttl_seconds: Time-to-live for cached models in seconds.
        cleanup_interval: Interval for cleanup in seconds.
    """

    enabled: bool = True
    max_memory_gb: float = 8.0
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        max_batch_size: Maximum batch size.
        max_delay: Maximum delay for batching in seconds.
        adaptive_batching: Whether to use adaptive batching.
    """

    max_batch_size: int = 8
    max_delay: float = 0.1
    adaptive_batching: bool = True


@dataclass
class OculodocConfig:
    """Main configuration for Oculodoc.

    Attributes:
        layout: Configuration for layout analysis.
        vlm: Configuration for VLM analysis.
        processing: Configuration for document processing.
        cache: Configuration for model caching.
        batch: Configuration for batch processing.
        logging: Configuration for logging.
        metrics: Configuration for metrics.
    """

    layout: LayoutConfig = field(default_factory=LayoutConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "layout": {
                "model_type": self.layout.model_type,
                "model_path": self.layout.model_path,
                "device": self.layout.device,
                "confidence_threshold": self.layout.confidence_threshold,
                "iou_threshold": self.layout.iou_threshold,
                "batch_size": self.layout.batch_size,
                "max_batch_delay": self.layout.max_batch_delay,
            },
            "vlm": {
                "model_type": self.vlm.model_type,
                "model_path": self.vlm.model_path,
                "host": self.vlm.host,
                "port": self.vlm.port,
                "max_tokens": self.vlm.max_tokens,
                "temperature": self.vlm.temperature,
                "timeout": self.vlm.timeout,
            },
            "processing": {
                "use_hybrid": self.processing.use_hybrid,
                "max_concurrent_pages": self.processing.max_concurrent_pages,
                "layout_confidence_threshold": self.processing.layout_confidence_threshold,
                "vlm_confidence_threshold": self.processing.vlm_confidence_threshold,
                "enable_visualization": self.processing.enable_visualization,
                "max_document_size_mb": self.processing.max_document_size_mb,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "max_memory_gb": self.cache.max_memory_gb,
                "ttl_seconds": self.cache.ttl_seconds,
                "cleanup_interval": self.cache.cleanup_interval,
            },
            "batch": {
                "max_batch_size": self.batch.max_batch_size,
                "max_delay": self.batch.max_delay,
                "adaptive_batching": self.batch.adaptive_batching,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "enable_structured": self.logging.enable_structured,
                "log_file": self.logging.log_file,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
                "port": self.metrics.port,
                "host": self.metrics.host,
                "prefix": self.metrics.prefix,
            },
        }
