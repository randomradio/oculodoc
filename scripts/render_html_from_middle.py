import argparse
import json
import os
from html import escape
from typing import Any, Dict, List, Tuple


def read_middle_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_page_sizes(doc: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    page_sizes: Dict[int, Tuple[int, int]] = {}
    layout = doc.get("doc_layout_result", {})
    for page in layout.get("page_info", []) or []:
        page_no = int(page.get("page_no"))
        width = int(page.get("width", 0))
        height = int(page.get("height", 0))
        page_sizes[page_no] = (width, height)
    return page_sizes


def collect_blocks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    # Prefer text_blocks for reconstruction if present (already OCR'd content)
    if isinstance(doc.get("text_blocks"), list):
        for b in doc["text_blocks"]:
            blocks.append(
                {
                    "type": "text",
                    "page_no": int(b.get("page_no")),
                    "bbox": b.get("bbox", [0, 0, 0, 0]),
                    "content": b.get("content", ""),
                    "score": b.get("score"),
                }
            )
    # Future: merge image/table blocks if provided by other keys
    return blocks


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def style_block(left: float, top: float, width: float, height: float) -> str:
    return (
        f"left:{left:.2f}px;top:{top:.2f}px;width:{width:.2f}px;height:{height:.2f}px;"
        "position:absolute;border:1px solid rgba(0,0,0,0.2);"
        "background-color:rgba(255,255,0,0.08);overflow:hidden;"
    )


def style_page(width: int, height: int) -> str:
    return (
        f"width:{width}px;height:{height}px;position:relative;"
        "margin:24px auto;border:1px solid #c9c9c9;background:#fff;"
        "box-shadow:0 2px 6px rgba(0,0,0,0.06);"
    )


def style_badge() -> str:
    return (
        "position:absolute;left:2px;top:2px;background:rgba(0,0,0,0.6);"
        "color:#fff;padding:1px 4px;border-radius:3px;font-size:11px;"
        "line-height:1;"
    )


def render_page_html(
    page_no: int, width: int, height: int, blocks: List[Dict[str, Any]]
) -> str:
    # Assign a simple bbox-based order to label blocks (top then left)
    ordered = sorted(blocks, key=lambda b: (float(b["bbox"][1]), float(b["bbox"][0])))
    html_parts: List[str] = []
    html_parts.append(
        f'<section class="page" data-page="{page_no}" style="{style_page(width, height)}">'
    )
    html_parts.append(
        f'<div class="page-label" style="position:absolute;right:8px;top:8px;color:#666;font:12px system-ui;">Page {page_no} · {width}×{height}</div>'
    )

    for idx, b in enumerate(ordered, start=1):
        x0, y0, x1, y1 = [float(v) for v in b.get("bbox", [0, 0, 0, 0])]
        left, top = x0, y0
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        content = escape(str(b.get("content", "")))
        btype = escape(str(b.get("type", "unknown")))
        score = b.get("score")
        score_str = f"{float(score):.3f}" if isinstance(score, (int, float)) else ""

        html_parts.append(
            f'<div class="block block-{btype}" style="{style_block(left, top, w, h)}"'
            f' data-type="{btype}" data-idx="{idx}">'
            f'<div class="badge" style="{style_badge()}">{idx}</div>'
            f'<div class="meta" style="position:absolute;right:2px;top:2px;color:#444;font:10px system-ui;opacity:0.75;">{btype}{" · " + score_str if score_str else ""}</div>'
            f'<div class="content" style="position:absolute;left:6px;right:6px;top:18px;bottom:4px;font:12px/1.35 "Times New Roman", serif;white-space:pre-wrap;color:#111;">{content}</div>'
            f"</div>"
        )

    html_parts.append("</section>")
    return "\n".join(html_parts)


def build_document_html(
    page_sizes: Dict[int, Tuple[int, int]],
    blocks_by_page: Dict[int, List[Dict[str, Any]]],
) -> str:
    head = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Middle JSON Preview</title>\n"
        "<style>\n"
        "body{background:#f6f7f8;margin:0;padding:16px 8px;font:14px system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;}\n"
        ".doc{max-width:calc(1134px + 48px);margin:0 auto;}\n"
        ".legend{margin:8px auto 16px;color:#444;}\n"
        "</style>\n"
        '</head>\n<body>\n<div class="doc">\n'
        '<div class="legend">Absolutely positioned preview from middle.json (bbox-based order). Colors/boxes are for visualization only.</div>\n'
    )
    pages_html: List[str] = []
    for page_no in sorted(page_sizes.keys()):
        width, height = page_sizes[page_no]
        page_blocks = blocks_by_page.get(page_no, [])
        pages_html.append(render_page_html(page_no, width, height, page_blocks))
    tail = "\n</div>\n</body>\n</html>\n"
    return head + "\n".join(pages_html) + tail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render absolutely-positioned HTML from middle.json"
    )
    parser.add_argument("--input", required=True, help="Path to middle.json")
    parser.add_argument("--output", required=True, help="Output directory for HTML")
    args = parser.parse_args()

    ensure_dir(args.output)

    doc = read_middle_json(args.input)
    page_sizes = get_page_sizes(doc)
    blocks = collect_blocks(doc)

    # Group blocks by page number
    blocks_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        pno = int(b.get("page_no", 1))
        blocks_by_page.setdefault(pno, []).append(b)

    html_str = build_document_html(page_sizes, blocks_by_page)

    out_path = os.path.join(args.output, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
