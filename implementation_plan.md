# Oculodoc Implementation Plan

## Executive Summary

This implementation plan outlines the architecture and development approach for Oculodoc, a production-ready document processing system combining layout analysis with Vision-Language Models (VLMs). The system emphasizes modularity, scalability, and flexibility, particularly supporting swappable layout analysis models.

## 1. System Architecture Overview

### 1.1 Core Design Principles

- **Modularity**: Each component implements clean interfaces and can be independently developed/tested
- **Swappability**: Layout analysis models follow a common interface for easy substitution
- **Async-First**: All I/O and compute operations are asynchronous
- **Error Resilience**: Graceful degradation when individual components fail
- **Observability**: Comprehensive logging, metrics, and tracing
- **Resource Efficiency**: Smart model caching and GPU memory management

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Oculodoc Service                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐   │
│  │  Document  │ │  Layout     │ │  VLM Analysis        │   │
│  │  Processor │ │  Engine     │ │  Engine              │   │
│  └─────────────┘ └─────────────┘ └───────────────────────┘   │
│           │            │                     │               │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐   │
│  │  PDF        │ │  Swappable  │ │  SGLang             │   │
│  │  Extractor  │ │  Layout     │ │  Inference          │   │
│  │             │ │  Model      │ │  Engine             │   │
│  └─────────────┘ └─────────────┘ └───────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐   │
│  │  Config     │ │  Metrics    │ │  Model Registry     │   │
│  │  Manager    │ │  & Tracing  │ │  & Cache            │   │
│  └─────────────┘ └─────────────┘ └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 2. Core Interfaces and Abstractions

### 2.1 Layout Analysis Interface

```python
# oculodoc/interfaces/layout_analyzer.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from PIL import Image

class LayoutDetection:
    """Standardized layout detection result"""
    category_id: int
    category_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    metadata: Dict[str, Any]

class ILayoutAnalyzer(ABC):
    """Abstract interface for layout analysis models"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the layout analyzer with configuration"""
        pass

    @abstractmethod
    async def analyze(self, image: Image.Image, **kwargs) -> List[LayoutDetection]:
        """Analyze image and return layout detections"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources (GPU memory, etc.)"""
        pass

    @property
    @abstractmethod
    def supported_categories(self) -> Dict[int, str]:
        """Return mapping of category IDs to names"""
        pass
```

### 2.2 VLM Analysis Interface

```python
# oculodoc/interfaces/vlm_analyzer.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class VLMAnalysisResult:
    """Standardized VLM analysis result"""
    content_type: str  # 'text', 'table', 'image', 'formula'
    content: str
    bbox: Optional[List[int]]
    confidence: float
    metadata: Dict[str, Any]

class IVLMAnalyzer(ABC):
    """Abstract interface for VLM-based document analysis"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize VLM analyzer"""
        pass

    @abstractmethod
    async def analyze(self, image_data: str, prompt: str, **kwargs) -> List[VLMAnalysisResult]:
        """Analyze document image with VLM"""
        pass

    @abstractmethod
    async def batch_analyze(self, image_data_list: List[str], prompts: List[str], **kwargs) -> List[List[VLMAnalysisResult]]:
        """Batch analyze multiple images"""
        pass
```

### 2.3 Document Processing Interface

```python
# oculodoc/interfaces/document_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path

class ProcessingResult:
    """Complete document processing result"""
    total_pages: int
    processing_time: float
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    errors: List[str]

class IDocumentProcessor(ABC):
    """Abstract interface for document processing pipeline"""

    @abstractmethod
    async def process_document(self, document_path: Path, **kwargs) -> ProcessingResult:
        """Process a document and return structured results"""
        pass

    @abstractmethod
    async def process_document_bytes(self, document_bytes: bytes, **kwargs) -> ProcessingResult:
        """Process document from bytes"""
        pass
```

## 3. Component Implementations

### 3.1 Layout Analyzers

#### 3.1.1 DocLayout-YOLO Implementation

```python
# oculodoc/layout/doclayout_yolo_analyzer.py
class DocLayoutYOLOAanalyzer(ILayoutAnalyzer):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLOv10] = None
        self.category_map = {
            0: "Title", 1: "Text", 2: "Abandon", 3: "ImageBody",
            4: "ImageCaption", 5: "TableBody", 6: "TableCaption",
            7: "TableFootnote", 8: "InterlineEquation_Layout",
            9: "InterlineEquationNumber_Layout", 13: "InlineEquation",
            14: "InterlineEquation_YOLO", 15: "OcrText", 16: "LowScoreText"
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Load and initialize YOLOv10 model"""
        # Implementation with proper error handling and GPU management
        pass

    async def analyze(self, image: Image.Image, **kwargs) -> List[LayoutDetection]:
        """Run inference with configurable parameters"""
        # Implementation with batch processing and memory optimization
        pass
```

#### 3.1.2 Generic YOLO Implementation

```python
# oculodoc/layout/generic_yolo_analyzer.py
class GenericYOLOAanalyzer(ILayoutAnalyzer):
    """Supports any YOLO model with standard interface"""
    # Implementation supporting ultralytics, yolov8, etc.
    pass
```

#### 3.1.3 DETR Layout Analyzer

```python
# oculodoc/layout/detr_analyzer.py
class DETRLayoutAnalyzer(ILayoutAnalyzer):
    """Microsoft DETR-based layout analysis"""
    # Implementation for transformer-based layout detection
    pass
```

### 3.2 VLM Analyzers

#### 3.2.1 SGLang OCRFlux Analyzer

```python
# oculodoc/vlm/sglang_ocrflux_analyzer.py
class SGLangOCRFluxAnalyzer(IVLMAnalyzer):
    def __init__(self, model_path: str, port: int = 30000):
        self.model_path = model_path
        self.port = port
        self.engine: Optional[Engine] = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize SGLang engine with OCRFlux model"""
        # Implementation with connection pooling and health checks
        pass

    async def analyze(self, image_data: str, prompt: str, **kwargs) -> List[VLMAnalysisResult]:
        """Run VLM inference via SGLang"""
        # Implementation with proper error handling and retries
        pass
```

#### 3.2.2 Transformers-based Analyzer

```python
# oculodoc/vlm/transformers_vlm_analyzer.py
class TransformersVLMAnalyzer(IVLMAnalyzer):
    """Direct transformers implementation for development/testing"""
    # Fallback implementation without SGLang dependency
    pass
```

### 3.3 Document Processors

#### 3.3.1 Hybrid Document Processor

```python
# oculodoc/processor/hybrid_processor.py
class HybridDocumentProcessor(IDocumentProcessor):
    def __init__(
        self,
        layout_analyzer: ILayoutAnalyzer,
        vlm_analyzer: IVLMAnalyzer,
        pdf_extractor: IPDFExtractor,
        config: Dict[str, Any]
    ):
        self.layout_analyzer = layout_analyzer
        self.vlm_analyzer = vlm_analyzer
        self.pdf_extractor = pdf_extractor
        self.config = config

    async def process_document(self, document_path: Path, **kwargs) -> ProcessingResult:
        """Main processing pipeline with layout + VLM analysis"""
        # Implementation of hybrid processing logic
        pass
```

## 4. Configuration Management

### 4.1 Configuration Schema

```python
# oculodoc/config/schema.py
@dataclass
class LayoutConfig:
    model_type: str  # "doclayout_yolo", "generic_yolo", "detr"
    model_path: str
    device: str = "cuda"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 4
    max_batch_delay: float = 0.1

@dataclass
class VLMConfig:
    model_type: str  # "sglang_ocrflux", "transformers_qwen"
    model_path: str
    host: str = "localhost"
    port: int = 30000
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: float = 30.0

@dataclass
class ProcessingConfig:
    use_hybrid: bool = True
    max_concurrent_pages: int = 4
    layout_confidence_threshold: float = 0.5
    vlm_confidence_threshold: float = 0.7
    enable_visualization: bool = False

@dataclass
class OculodocConfig:
    layout: LayoutConfig
    vlm: VLMConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    metrics: MetricsConfig
```

### 4.2 Configuration Loading

```python
# oculodoc/config/loader.py
class ConfigLoader:
    @staticmethod
    def load_from_file(path: Path) -> OculodocConfig:
        """Load configuration from YAML/JSON file"""
        pass

    @staticmethod
    def load_from_env() -> OculodocConfig:
        """Load configuration from environment variables"""
        pass
```

## 5. Error Handling and Resilience

### 5.1 Error Types

```python
# oculodoc/errors.py
class OculodocError(Exception):
    """Base exception for Oculodoc"""
    pass

class ModelLoadError(OculodocError):
    """Model loading/initialization errors"""
    pass

class InferenceError(OculodocError):
    """Model inference errors"""
    pass

class ConfigurationError(OculodocError):
    """Configuration-related errors"""
    pass
```

### 5.2 Circuit Breaker Pattern

```python
# oculodoc/resilience/circuit_breaker.py
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        pass

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        pass
```

### 5.3 Graceful Degradation

```python
# oculodoc/processor/resilient_processor.py
class ResilientDocumentProcessor:
    """Processor with automatic fallback strategies"""
    async def process_with_fallback(self, document_path: Path) -> ProcessingResult:
        """Process with automatic fallback to simpler methods on failure"""
        pass
```

## 6. Performance Optimization

### 6.1 Model Caching and Management

```python
# oculodoc/cache/model_cache.py
class ModelCache:
    """LRU cache for loaded models with automatic unloading"""
    def __init__(self, max_memory_gb: float = 8.0):
        pass

    async def get_model(self, model_key: str) -> Any:
        """Get model from cache or load if not present"""
        pass

    async def unload_unused_models(self) -> None:
        """Unload models that haven't been used recently"""
        pass
```

### 6.2 Batch Processing

```python
# oculodoc/batch/batch_processor.py
class BatchProcessor:
    """Intelligent batching for model inference"""
    def __init__(self, max_batch_size: int = 8, max_delay: float = 0.1):
        pass

    async def add_request(self, request: InferenceRequest) -> Future:
        """Add request to batch queue"""
        pass

    async def process_batch(self) -> None:
        """Process accumulated batch"""
        pass
```

### 6.3 GPU Memory Management

```python
# oculodoc/gpu/memory_manager.py
class GPUMemoryManager:
    """Monitor and manage GPU memory usage"""
    def __init__(self, device: str = "cuda", memory_threshold: float = 0.9):
        pass

    async def should_process_batch(self, batch_size: int) -> bool:
        """Check if there's enough GPU memory for batch"""
        pass

    async def cleanup_memory(self) -> None:
        """Force garbage collection and memory cleanup"""
        pass
```

## 7. Observability and Monitoring

### 7.1 Metrics

```python
# oculodoc/metrics/collector.py
class MetricsCollector:
    """Collect and export metrics"""
    def __init__(self):
        self.processing_time = Histogram("oculodoc_processing_time_seconds")
        self.model_inference_time = Histogram("oculodoc_inference_time_seconds")
        self.memory_usage = Gauge("oculodoc_memory_usage_bytes")
        self.error_count = Counter("oculodoc_errors_total")
```

### 7.2 Structured Logging

```python
# oculodoc/logging/logger.py
class StructuredLogger:
    """Structured logging with context propagation"""
    def __init__(self, service_name: str = "oculodoc"):
        pass

    def log_processing_start(self, document_id: str, page_count: int):
        pass

    def log_inference_complete(self, model_type: str, duration: float):
        pass
```

### 7.3 Health Checks

```python
# oculodoc/health/health_check.py
class HealthChecker:
    """Health checks for all components"""
    async def check_layout_analyzer(self) -> HealthStatus:
        pass

    async def check_vlm_analyzer(self) -> HealthStatus:
        pass

    async def overall_health(self) -> HealthStatus:
        pass
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/unit/test_layout_analyzer.py
class TestDocLayoutYOLOAanalyzer:
    @pytest.fixture
    def mock_model(self):
        # Mock YOLO model for testing
        pass

    def test_initialization(self):
        # Test model initialization
        pass

    def test_inference(self):
        # Test inference pipeline
        pass
```

### 8.2 Integration Tests

```python
# tests/integration/test_hybrid_processor.py
class TestHybridProcessor:
    @pytest.fixture
    async def processor(self):
        # Setup complete processor with test models
        pass

    async def test_full_pipeline(self):
        # Test complete document processing pipeline
        pass
```

### 8.3 Performance Tests

```python
# tests/performance/test_throughput.py
class TestThroughput:
    async def test_concurrent_processing(self):
        # Test concurrent document processing
        pass

    async def test_memory_usage(self):
        # Test memory usage under load
        pass
```

## 9. Deployment and Operations

### 9.1 Docker Configuration

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Multi-stage build for optimized image size
# Stage 1: Builder
FROM base AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM base AS runtime
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

EXPOSE 8000 30000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "oculodoc.main"]
```

### 9.2 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oculodoc
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: oculodoc
        image: oculodoc:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
```

### 9.3 Service Mesh Integration

- Istio integration for traffic management
- Service discovery and load balancing
- Circuit breaking and retry logic
- Distributed tracing with Jaeger

## 10. Security Considerations

### 10.1 Input Validation

```python
# oculodoc/security/validator.py
class InputValidator:
    """Validate document inputs for security"""
    def validate_pdf(self, pdf_bytes: bytes) -> bool:
        """Validate PDF file for malicious content"""
        pass

    def validate_image(self, image: Image.Image) -> bool:
        """Validate image dimensions and content"""
        pass
```

### 10.2 Model Sandboxing

- Run models in isolated containers
- Resource limits and quotas
- Model file integrity verification
- Secure model registry access

### 10.3 API Security

- Authentication and authorization
- Rate limiting
- Input size limits
- CORS configuration

## 11. Development Workflow

### 11.1 Code Organization

```
oculodoc/
├── interfaces/           # Abstract interfaces
├── layout/              # Layout analysis implementations
├── vlm/                 # VLM analysis implementations
├── processor/           # Document processing logic
├── config/              # Configuration management
├── cache/               # Caching mechanisms
├── batch/               # Batch processing
├── gpu/                 # GPU management
├── metrics/             # Metrics collection
├── logging/             # Logging utilities
├── errors/              # Error definitions
├── resilience/          # Circuit breakers, retries
├── security/            # Security utilities
├── health/              # Health checks
├── tests/               # Test suites
└── main.py             # Application entry point
```

### 11.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        pip install -r requirements-dev.txt
        pytest --cov=oculodoc --cov-report=xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Build Docker image
      run: docker build -t oculodoc:${{ github.sha }} .

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to staging
      # Deployment logic
```

## 12. Migration and Compatibility

### 12.1 Version Compatibility

- Semantic versioning for API changes
- Model version compatibility matrix
- Configuration migration utilities
- Backward compatibility testing

### 12.2 Migration Strategy

1. **Phase 1**: Implement core interfaces and base implementations
2. **Phase 2**: Add swappable layout analyzers
3. **Phase 3**: Integrate VLM analysis
4. **Phase 4**: Add resilience and monitoring
5. **Phase 5**: Performance optimization and scaling

## 13. Success Metrics

### 13.1 Performance Metrics

- **Latency**: P95 processing time < 30 seconds for 100-page document
- **Throughput**: Process 100 documents/minute
- **Accuracy**: Layout detection accuracy > 95%
- **Resource Usage**: GPU memory < 8GB per instance

### 13.2 Quality Metrics

- **Test Coverage**: > 90% code coverage
- **Error Rate**: < 0.1% processing failures
- **Uptime**: > 99.9% service availability
- **User Satisfaction**: > 95% accuracy rating

## 14. Risk Assessment and Mitigation

### 14.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model loading failures | High | Medium | Circuit breakers, fallback models |
| GPU memory exhaustion | High | Low | Memory monitoring, batch size limits |
| SGLang service unavailability | Medium | Low | Service mesh, health checks |
| Large document processing timeout | Medium | Medium | Chunking, progress tracking |

### 14.2 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model performance degradation | High | Low | Continuous monitoring, retraining pipeline |
| Security vulnerabilities | High | Low | Regular security audits, dependency updates |
| Configuration errors | Medium | Medium | Configuration validation, canary deployments |

## Conclusion

This implementation plan provides a comprehensive roadmap for building Oculodoc as a production-ready, scalable document processing system. The modular architecture with swappable components ensures flexibility and maintainability, while the focus on error handling, observability, and performance optimization ensures reliability and efficiency.

Key priorities for the initial implementation:
1. Core interfaces and abstractions
2. DocLayout-YOLO and SGLang OCRFlux implementations
3. Hybrid processing pipeline
4. Comprehensive testing and monitoring
5. Production deployment configuration

The plan emphasizes Google's engineering best practices: clean interfaces, comprehensive testing, observability, and operational excellence.
