#!/usr/bin/env python3
"""
Debug script to visualize and inspect layout analysis results.

This script provides detailed debugging capabilities for the layout analysis pipeline:
1. Shows detected regions with bounding boxes and confidence scores
2. Creates annotated images with region labels
3. Extracts and saves individual region crops
4. Generates HTML report with region details
5. Shows statistics about detected elements

Usage:
    python scripts/debug_layout.py <pdf_path> [page_number]

Examples:
    python scripts/debug_layout.py demo1.pdf           # Process all pages
    python scripts/debug_layout.py demo1.pdf 5        # Process only page 5
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO
import json

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from oculodoc.config import OculodocConfig
from oculodoc.layout.doclayout_yolo_analyzer import DocLayoutYOLOAnalyzer


class LayoutDebugger:
    """Debug utilities for layout analysis."""

    def __init__(self, output_dir: Path = Path("output/debug")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color mapping for different element types
        self.colors = {
            "text": "#FF6B6B",  # Red
            "title": "#4ECDC4",  # Teal
            "table": "#45B7D1",  # Blue
            "tablebody": "#45B7D1",  # Blue
            "tablecaption": "#96CEB4",  # Light green
            "figure": "#FECA57",  # Yellow
            "imagebody": "#FECA57",  # Yellow
            "imagecaption": "#DDA0DD",  # Plum
            "equation": "#FF9FF3",  # Pink
            "interlineequation_layout": "#FF9FF3",  # Pink
            "interlineequationnumber_layout": "#FFB6C1",  # Light pink
            "abandon": "#808080",  # Gray
            "unknown": "#C0C0C0",  # Light gray
        }

    def _get_color_for_type(self, category_name: str) -> str:
        """Get color for element type."""
        name = (category_name or "unknown").lower()
        for key in self.colors:
            if key in name:
                return self.colors[key]
        return self.colors["unknown"]

    def _draw_bbox(
        self,
        draw: ImageDraw.Draw,
        bbox: List[float],
        color: str,
        label: str,
        confidence: float,
    ) -> None:
        """Draw bounding box with label."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        label_text = f"{label} ({confidence:.2f})"
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None

        # Get text size
        if font:
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        else:
            text_width = len(label_text) * 6
            text_height = 12

        # Draw label background
        label_bg = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
        draw.rectangle(label_bg, fill=color)

        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)

    async def debug_page(
        self, pdf_path: Path, page_number: int, analyzer: DocLayoutYOLOAnalyzer
    ) -> Dict[str, Any]:
        """Debug layout analysis for a single page."""
        print(f"\n🔍 Debugging page {page_number} of {pdf_path}")

        # Extract page image
        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_number - 1)  # Convert to 0-based
            pix = page.get_pixmap(dpi=150)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            image = Image.open(BytesIO(img_bytes))
        finally:
            doc.close()

        print(f"   📐 Page size: {image.size[0]}×{image.size[1]} pixels")

        # Run layout analysis
        print("   🤖 Running layout analysis...")
        detections = await analyzer.analyze(image)
        print(f"   📊 Found {len(detections)} regions")

        # Create annotated image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Statistics
        stats = {}
        regions_data = []

        # Process each detection
        for i, det in enumerate(detections):
            category = det.category_name or "unknown"
            color = self._get_color_for_type(category)

            # Draw bounding box
            self._draw_bbox(draw, det.bbox, color, category, det.confidence)

            # Update statistics
            if category not in stats:
                stats[category] = {"count": 0, "avg_conf": 0, "total_conf": 0}
            stats[category]["count"] += 1
            stats[category]["total_conf"] += det.confidence
            stats[category]["avg_conf"] = (
                stats[category]["total_conf"] / stats[category]["count"]
            )

            # Extract region
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            region_img = image.crop((x1, y1, x2, y2))

            # Save region
            region_filename = f"p{page_number:03d}_region_{i:03d}_{category}_conf{int(det.confidence * 100):02d}.jpg"
            region_path = (
                self.output_dir
                / "regions"
                / f"page_{page_number:03d}"
                / region_filename
            )
            region_path.parent.mkdir(parents=True, exist_ok=True)

            if region_img.mode != "RGB":
                region_img = region_img.convert("RGB")
            region_img.save(region_path, format="JPEG", quality=90)

            # Store region data
            regions_data.append(
                {
                    "index": i,
                    "category": category,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "area": (x2 - x1) * (y2 - y1),
                    "filename": region_filename,
                    "color": color,
                }
            )

        # Save annotated image
        annotated_path = self.output_dir / f"page_{page_number:03d}_annotated.jpg"
        if annotated.mode != "RGB":
            annotated = annotated.convert("RGB")
        annotated.save(annotated_path, format="JPEG", quality=95)

        # Print statistics
        print("   📈 Detection statistics:")
        for category, data in sorted(stats.items()):
            print(
                f"      {category}: {data['count']} regions (avg conf: {data['avg_conf']:.3f})"
            )

        return {
            "page_number": page_number,
            "image_size": image.size,
            "total_regions": len(detections),
            "stats": stats,
            "regions": regions_data,
            "annotated_path": str(annotated_path),
            "original_path": str(pdf_path),
        }

    def generate_html_report(
        self, debug_data: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Generate HTML debug report."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layout Analysis Debug Report</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .page-section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .page-header { border-bottom: 2px solid #eee; padding-bottom: 15px; margin-bottom: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
        .regions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .region-card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 15px; }
        .region-preview { width: 100%; max-height: 150px; object-fit: contain; border-radius: 4px; margin-bottom: 10px; }
        .bbox-info { font-family: monospace; font-size: 12px; color: #666; margin: 5px 0; }
        .confidence { font-weight: bold; }
        .confidence.high { color: #28a745; }
        .confidence.medium { color: #ffc107; }
        .confidence.low { color: #dc3545; }
        .annotated-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .color-legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }
        .color-item { display: flex; align-items: center; gap: 5px; }
        .color-box { width: 20px; height: 20px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Layout Analysis Debug Report</h1>
"""

        # Add color legend
        html += """
        <div class="page-section">
            <h2>Color Legend</h2>
            <div class="color-legend">
"""
        for category, color in self.colors.items():
            html += f'<div class="color-item"><div class="color-box" style="background-color: {color};"></div><span>{category}</span></div>'

        html += """
            </div>
        </div>
"""

        # Add page sections
        for page_data in debug_data:
            page_num = page_data["page_number"]
            html += f"""
        <div class="page-section">
            <div class="page-header">
                <h2>📄 Page {page_num}</h2>
                <p>Image size: {page_data["image_size"][0]}×{page_data["image_size"][1]} pixels | Total regions: {page_data["total_regions"]}</p>
            </div>
            
            <h3>📊 Detection Statistics</h3>
            <div class="stats-grid">
"""

            for category, stats in page_data["stats"].items():
                html += f"""
                <div class="stat-card">
                    <div style="color: {self._get_color_for_type(category)}; font-weight: bold;">{category}</div>
                    <div>Count: {stats["count"]}</div>
                    <div>Avg Confidence: {stats["avg_conf"]:.3f}</div>
                </div>
"""

            html += f"""
            </div>
            
            <h3>🖼️ Annotated Page</h3>
            <img src="{Path(page_data["annotated_path"]).name}" alt="Annotated Page {page_num}" class="annotated-image">
            
            <h3>🔍 Individual Regions</h3>
            <div class="regions-grid">
"""

            for region in page_data["regions"]:
                conf_class = (
                    "high"
                    if region["confidence"] > 0.8
                    else "medium"
                    if region["confidence"] > 0.5
                    else "low"
                )
                html += f"""
                <div class="region-card">
                    <img src="regions/page_{page_num:03d}/{region["filename"]}" alt="Region {region["index"]}" class="region-preview">
                    <div><strong>{region["category"]}</strong></div>
                    <div class="confidence {conf_class}">Confidence: {region["confidence"]:.3f}</div>
                    <div class="bbox-info">BBox: [{region["bbox"][0]:.0f}, {region["bbox"][1]:.0f}, {region["bbox"][2]:.0f}, {region["bbox"][3]:.0f}]</div>
                    <div class="bbox-info">Area: {region["area"]:.0f} px²</div>
                </div>
"""

            html += """
            </div>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        output_path.write_text(html, encoding="utf-8")
        print(f"   📄 HTML report saved to: {output_path}")


async def main():
    """Main debug function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_layout.py <pdf_path> [page_number]")
        print("Examples:")
        print(
            "  python scripts/debug_layout.py demo1.pdf           # Process all pages"
        )
        print(
            "  python scripts/debug_layout.py demo1.pdf 5        # Process only page 5"
        )
        return

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    # Determine pages to process
    if len(sys.argv) > 2:
        try:
            pages_to_process = [int(sys.argv[2])]
        except ValueError:
            print(f"❌ Invalid page number: {sys.argv[2]}")
            return
    else:
        # Process all pages
        doc = fitz.open(str(pdf_path))
        pages_to_process = list(range(1, doc.page_count + 1))
        doc.close()

    print(f"🚀 Starting layout analysis debug for {pdf_path}")
    print(f"   📄 Processing pages: {pages_to_process}")

    # Initialize analyzer
    config = OculodocConfig()
    config.layout.device = "auto"
    analyzer = DocLayoutYOLOAnalyzer()
    await analyzer.initialize(config.layout.__dict__)
    print("✅ Layout analyzer initialized")

    # Initialize debugger
    debugger = LayoutDebugger()

    # Process pages
    debug_data = []
    for page_num in pages_to_process:
        try:
            page_data = await debugger.debug_page(pdf_path, page_num, analyzer)
            debug_data.append(page_data)
        except Exception as e:
            print(f"❌ Error processing page {page_num}: {e}")

    if debug_data:
        # Generate HTML report
        report_path = debugger.output_dir / "debug_report.html"
        debugger.generate_html_report(debug_data, report_path)

        # Save JSON data
        json_path = debugger.output_dir / "debug_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 Debug data saved to: {json_path}")

        print(
            f"\n✅ Debug complete! Open {report_path} in your browser to view results."
        )
        print(f"   📁 Region images saved to: {debugger.output_dir}/regions/")
    else:
        print("❌ No pages were successfully processed")

    # Cleanup
    await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
