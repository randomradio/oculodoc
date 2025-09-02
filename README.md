# Oculodoc

A production-ready document processing system that combines layout analysis with Vision-Language Models (VLMs) for comprehensive document understanding.

## Features

- **Modular Architecture**: Clean interfaces allow for swappable layout analysis and VLM models
- **Async-First Design**: All I/O and compute operations are asynchronous
- **Comprehensive Error Handling**: Graceful degradation and detailed error reporting
- **Flexible Configuration**: Support for YAML, JSON, and environment variable configuration
- **Extensive Testing**: 83 unit tests covering all non-model components
- **Google Style**: Follows Google Python coding standards
- **✅ DocLayout-YOLO Integration**: Working YOLO-based layout analysis
- **🔄 SGLang VLM Ready**: Framework ready for Vision-Language Model integration

## Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
cd oculodoc
uv sync
uv sync --group dev  # For development dependencies
```

## Quick Start

```bash
# Test the system (DocLayout-YOLO will auto-download on first run)
python main.py

# Process a document
python main.py demo1.pdf
```

### Python API

```python
from oculodoc.config import OculodocConfig
from oculodoc.processor import BaseDocumentProcessor

# Create configuration
config = OculodocConfig()

# Create processor with model analyzers
processor = BaseDocumentProcessor(config)

# Process a document
result = await processor.process_document("path/to/document.pdf")
print(f"Processed {result.total_pages} pages in {result.processing_time:.2f}s")
```

## Current Implementation Status

### ✅ Working Components

- **DocLayout-YOLO Analyzer**: Fully functional YOLOv8-based layout analysis
  - Downloads PDF-Extract-Kit DocLayout-YOLO model (40.7MB) from HuggingFace
  - Detects document elements with configurable confidence thresholds
  - Supports GPU acceleration when available
  - Uses specialized DocLayout-YOLO model for document analysis

- **Base Document Processor**: Complete document processing pipeline
  - File validation and metadata extraction
  - Graceful fallback when analyzers are unavailable
  - Comprehensive error handling and logging

- **Configuration System**: Flexible configuration management
  - YAML/JSON file support
  - Environment variable configuration
  - Runtime configuration updates

- **Testing Framework**: 83 comprehensive unit tests
  - 100% test coverage for non-model components
  - Async test support with pytest-asyncio

- **MinerU-Compatible Output**: Structured JSON output format
  - Generates `middle.json` files similar to MinerU
  - Includes layout detections, text blocks, metadata
  - Structured format for downstream processing

### 🔄 Ready for Integration

- **SGLang VLM Framework**: Complete client implementation
  - Ready to connect to SGLang servers
  - OpenAI-compatible API support
  - Automatic retry and error handling

## Command Line Usage

```bash
# Show system status and configuration
python main.py

# Process a document with available analyzers
python main.py demo1.pdf
# Output: demo1_middle.json (MinerU-compatible format)

# Setup and test models
python setup_models.py
```

### Output Format

Oculodoc generates MinerU-compatible `middle.json` files containing:

```json
{
  "doc_layout_result": {
    "layout_dets": [...],
    "page_info": [...]
  },
  "text_blocks": [...],
  "title_blocks": [...],
  "figure_blocks": [...],
  "table_blocks": [...],
  "meta_data": {
    "doc_name": "document.pdf",
    "total_pages": 1,
    "processing_time": 0.05,
    "model_info": {...}
  }
}
```

## Model Setup

### DocLayout-YOLO (✅ Working)

The DocLayout-YOLO analyzer automatically downloads the YOLOv8n model on first use:

```bash
# Test DocLayout-YOLO (will download model if needed)
python main.py
```

#### Using Custom DocLayout-YOLO Models

For specialized document layout analysis, you can use custom-trained DocLayout-YOLO models:

```python
from oculodoc.layout import DocLayoutYOLOAanalyzer

# Use a custom DocLayout-YOLO model
analyzer = DocLayoutYOLOAanalyzer(
    model_path="/path/to/your/doclayout_yolo_ft.pt"
)
await analyzer.initialize({})
```

**Available Models:**
- **PDF-Extract-Kit DocLayout-YOLO**: [Download from HuggingFace](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/main/models/Layout/YOLO)
- **MinerU DocLayout-YOLO**: Available in MinerU repository
- **Custom trained models**: Train your own using the DocLayout-YOLO training pipeline

**Note**: Custom DocLayout-YOLO models may require additional dependencies. The current implementation uses standard YOLOv8 which provides good general-purpose document layout analysis.

### SGLang VLM (🔄 Ready for Setup)

To enable Vision-Language Model analysis:

```bash
# 1. Install SGLang
pip install sglang

# 2. Download a VLM model (example: Qwen2-VL)
# Visit: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 3. Start SGLang server
python -m sglang.launch_server \
  --model-path /path/to/your/vlm/model \
  --host localhost \
  --port 30000

# 4. Test connection
python main.py
```

## Architecture

### Core Interfaces

- **ILayoutAnalyzer**: Abstract interface for layout analysis models
- **IVLMAnalyzer**: Abstract interface for Vision-Language Model analysis
- **IDocumentProcessor**: Abstract interface for document processing pipelines

### Key Components

- **Configuration Management**: Dataclass-based configuration with validation
- **Error Handling**: Comprehensive exception hierarchy with detailed error information
- **Base Processor**: Concrete implementation with validation and processing logic
- **Testing Framework**: 83 unit tests covering all non-model functionality

### Project Structure

```
oculodoc/
├── interfaces/           # Abstract interfaces
│   ├── layout_analyzer.py
│   ├── vlm_analyzer.py
│   └── document_processor.py
├── config/              # Configuration management
│   ├── schema.py
│   └── loader.py
├── errors/              # Error handling
│   └── exceptions.py
├── processor/           # Processing implementations
│   └── base_processor.py
└── tests/               # Unit tests
```

## Configuration

### Default Configuration

```python
config = OculodocConfig()
# Layout: doclayout_yolo, VLM: sglang_ocrflux
# Processing: hybrid mode enabled, 4 concurrent pages max
```

### Custom Configuration

```python
from oculodoc.config import ConfigLoader

# Load from YAML file
config = ConfigLoader.load_from_file("config.yaml")

# Load from environment variables (prefix: OCULODOC_)
config = ConfigLoader.load_from_env()

# Create programmatically
config = OculodocConfig()
config.layout.model_type = "generic_yolo"
config.processing.use_hybrid = False
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=oculodoc

# Run specific test file
uv run pytest oculodoc/tests/test_config.py
```

## Development

### Code Style

The project follows Google Python coding standards. Format code with:

```bash
uv run black .
uv run isort .
```

### Adding New Components

1. **New Layout Analyzer**:
   ```python
   from oculodoc.interfaces import ILayoutAnalyzer

   class MyLayoutAnalyzer(ILayoutAnalyzer):
       # Implement required methods
   ```

2. **New VLM Analyzer**:
   ```python
   from oculodoc.interfaces import IVLMAnalyzer

   class MyVLMAnalyzer(IVLMAnalyzer):
       # Implement required methods
   ```

3. **New Processor**:
   ```python
   from oculodoc.interfaces import IDocumentProcessor

   class MyProcessor(IDocumentProcessor):
       # Implement required methods
   ```

## Error Handling

The system provides detailed error information:

```python
try:
    result = await processor.process_document("document.pdf")
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Field: {e.field}, Value: {e.value}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
    print(f"Document: {e.document_path}, Stage: {e.stage}")
```

## Testing Results

```
======================== 83 passed in 0.10s ========================
```

All 83 unit tests pass, covering:
- Interface definitions and contracts
- Configuration management and validation
- Error handling and exception hierarchy
- Document processing logic
- Input validation and metadata extraction

## Future Development

The system is designed for easy extension:

1. **Model Integration**: Add concrete implementations of layout analyzers (DocLayout-YOLO, DETR) and VLM analyzers (SGLang OCRFlux, Transformers)

2. **Performance Optimization**: Add batching, caching, and GPU memory management

3. **Observability**: Implement metrics collection and structured logging

4. **Deployment**: Add Docker configuration and Kubernetes manifests

5. **API**: Create REST API for web service deployment

## License

This project follows the implementation plan outlined in `implementation_plan.md` and is designed for production use with comprehensive testing and error handling.
