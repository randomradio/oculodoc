#!/usr/bin/env python3
"""Setup script for downloading models and starting SGLang server."""

import asyncio
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from oculodoc.layout import DocLayoutYOLOAanalyzer
from oculodoc.vlm import SGLangOCRFluxAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def download_doclayout_model(cache_dir: str = None):
    """Download DocLayout-YOLO model."""
    print("🔄 Setting up DocLayout-YOLO model...")

    try:
        analyzer = DocLayoutYOLOAanalyzer(model_cache_dir=cache_dir)

        # This will trigger model download if not present
        model_path = analyzer._ensure_model_available()
        print(f"✅ DocLayout-YOLO model ready at: {model_path}")

        # Test model loading
        print("🔄 Testing model loading...")
        await analyzer.initialize({})
        print("✅ DocLayout-YOLO model loaded successfully!")

        await analyzer.cleanup()

    except Exception as e:
        print(f"❌ Failed to setup DocLayout-YOLO model: {e}")
        return False

    return True


def check_sglang_server(host: str = "localhost", port: int = 30000):
    """Check if SGLang server is running."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


async def setup_sglang_server(
    host: str = "localhost", port: int = 30000, model_path: str = None
):
    """Setup and start SGLang server with OCRFlux model."""
    print(f"🔄 Setting up SGLang server on {host}:{port}...")

    # Check if server is already running
    if check_sglang_server(host, port):
        print(f"✅ SGLang server is already running on {host}:{port}")

        # Test connection
        try:
            analyzer = SGLangOCRFluxAnalyzer(host=host, port=port)
            await analyzer.initialize({})
            info = await analyzer.get_server_info()
            print(f"✅ SGLang server info: {info}")
            await analyzer.cleanup()
            return True
        except Exception as e:
            print(f"❌ SGLang server connection test failed: {e}")
            return False

    # Server is not running
    print(f"❌ SGLang server is not running on {host}:{port}")
    print("\n📝 To set up SGLang server manually:")
    print("   1. Install SGLang:")
    print("      pip install sglang")
    print("   2. Download a Vision-Language model, for example:")
    print("      - Qwen2-VL: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct")
    print("      - GPT-4V compatible models from HuggingFace")
    print("   3. Start the server:")
    print(
        f"      python -m sglang.launch_server --model-path <model_path> --host {host} --port {port}"
    )
    print("   4. Test the connection by running this script again")
    print("\n💡 Note: Oculodoc will work with basic processing even without VLM")
    print("   The DocLayout-YOLO analyzer is already working!")

    return False


async def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Oculodoc models and services")
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache directory for models"
    )
    parser.add_argument(
        "--sglang-host", type=str, default="localhost", help="SGLang server host"
    )
    parser.add_argument(
        "--sglang-port", type=int, default=30000, help="SGLang server port"
    )
    parser.add_argument(
        "--vlm-model-path",
        type=str,
        default=None,
        help="Path to VLM model for SGLang server",
    )
    parser.add_argument(
        "--skip-layout", action="store_true", help="Skip DocLayout-YOLO setup"
    )
    parser.add_argument("--skip-vlm", action="store_true", help="Skip SGLang VLM setup")

    args = parser.parse_args()
    setup_logging()

    print("🚀 Starting Oculodoc model setup...")
    print("=" * 50)

    success_count = 0
    total_count = 0

    # Setup DocLayout-YOLO
    if not args.skip_layout:
        total_count += 1
        if await download_doclayout_model(args.cache_dir):
            success_count += 1
        print()

    # Setup SGLang server
    if not args.skip_vlm:
        total_count += 1
        if await setup_sglang_server(
            args.sglang_host, args.sglang_port, args.vlm_model_path
        ):
            success_count += 1
        print()

    print("=" * 50)
    print(f"📊 Setup complete: {success_count}/{total_count} components successful")

    if success_count == total_count:
        print("🎉 All components are ready!")
        print("\nYou can now use Oculodoc with:")
        print("- DocLayout-YOLO for layout analysis")
        print("- SGLang OCRFlux for VLM analysis")
        return 0
    else:
        print("⚠️  Some components failed to setup")
        print("Please check the error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
