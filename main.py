"""Main entry point for Oculodoc."""

import asyncio
import json
import sys
from pathlib import Path

from oculodoc.config import OculodocConfig
from oculodoc.processor import BaseDocumentProcessor
from oculodoc.layout import DocLayoutYOLOAnalyzer
from oculodoc.vlm import SGLangOCRFluxAnalyzer
from oculodoc.errors import OculodocError


async def create_processor_with_models(config: OculodocConfig):
    """Create processor with concrete model implementations."""
    # Initialize layout analyzer
    layout_analyzer = None
    try:
        layout_analyzer = DocLayoutYOLOAnalyzer()
        await layout_analyzer.initialize(config.layout.__dict__)
        print("✅ DocLayout-YOLO analyzer initialized")
    except Exception as e:
        print(f"⚠️  DocLayout-YOLO analyzer failed to initialize: {e}")

    # Initialize VLM analyzer
    vlm_analyzer = None
    try:
        vlm_analyzer = SGLangOCRFluxAnalyzer(host=config.vlm.host, port=config.vlm.port)
        await vlm_analyzer.initialize(config.vlm.__dict__)
        print("✅ SGLang OCRFlux analyzer initialized")
    except Exception as e:
        print(f"⚠️  SGLang OCRFlux analyzer failed to initialize: {e}")

    # Create processor with analyzers
    processor = BaseDocumentProcessor(
        config=config, layout_analyzer=layout_analyzer, vlm_analyzer=vlm_analyzer
    )

    return processor


async def main() -> None:
    """Main entry point."""
    try:
        # Load configuration
        config = OculodocConfig()

        print("🚀 Initializing Oculodoc...")
        processor = await create_processor_with_models(config)

        # Example usage
        if len(sys.argv) > 1:
            document_path = Path(sys.argv[1])
            if document_path.exists():
                print(f"\n📄 Processing document: {document_path}")
                result = await processor.process_document(document_path)

                print("✅ Processing complete!")
                print(f"   Total pages: {result.total_pages}")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Successful: {result.is_successful}")

                if result.pages:
                    content = result.pages[0].get("content", "N/A")
                    print(f"   Sample content: {content[:100]}...")

                if result.errors:
                    print("   Errors:")
                    for error in result.errors:
                        print(f"     - {error}")

                # Show metadata
                if result.metadata:
                    print("   Metadata:")
                    for key, value in result.metadata.items():
                        print(f"     {key}: {value}")

                # Generate and save middle.json (MinerU format)
                print("\n💾 Generating middle.json output...")
                middle_json = processor.generate_middle_json(result, document_path)

                # Save to middle.json file
                output_file = document_path.parent / f"{document_path.stem}_middle.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(middle_json, f, indent=2, ensure_ascii=False)

                print(f"   ✅ Saved middle.json to: {output_file}")
                print(
                    f"   📊 JSON contains {len(middle_json.get('text_blocks', []))} text blocks"
                )
                print(
                    f"   📊 JSON contains {len(middle_json.get('title_blocks', []))} title blocks"
                )
                print(
                    f"   📊 JSON contains {len(middle_json.get('figure_blocks', []))} figure blocks"
                )
                print(
                    f"   📊 JSON contains {len(middle_json.get('table_blocks', []))} table blocks"
                )

            else:
                print(f"❌ Document not found: {document_path}")
        else:
            print("\n📖 Usage: python main.py <document_path>")
            print("   Example: python main.py sample.pdf")

            # Show configuration
            print("\n⚙️  Current configuration:")
            config_dict = config.to_dict()
            print(f"   Layout model: {config_dict['layout']['model_type']}")
            print(f"   VLM model: {config_dict['vlm']['model_type']}")
            print(f"   Hybrid processing: {config_dict['processing']['use_hybrid']}")

            # Show available analyzers
            print("\n🔧 Available analyzers:")
            if hasattr(processor, "layout_analyzer") and processor.layout_analyzer:
                try:
                    info = await processor.layout_analyzer.get_model_info()
                    print(f"   ✅ Layout: {info.get('model_type', 'unknown')}")
                except Exception:
                    print("   ✅ Layout: DocLayout-YOLO")
            else:
                print("   ❌ Layout: Not available")

            if hasattr(processor, "vlm_analyzer") and processor.vlm_analyzer:
                try:
                    info = await processor.vlm_analyzer.get_server_info()
                    print(f"   ✅ VLM: {info.get('status', 'unknown')}")
                except Exception:
                    print("   ❌ VLM: Connection failed")
            else:
                print("   ❌ VLM: Not available")

    except OculodocError as e:
        print(f"❌ Oculodoc error: {e}")
        if hasattr(e, "details") and e.details:
            print(f"   Details: {e.details}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
