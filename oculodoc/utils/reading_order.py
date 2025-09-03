"""Reading order utilities using XY-cut algorithm."""

import numpy as np
from typing import List, Dict, Any, Tuple


def _projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """Generate a 1D projection histogram from bounding boxes along a specified axis.

    Args:
        boxes: A (N, 4) array of bounding boxes defined by [x_min, y_min, x_max, y_max].
        axis: Axis for projection; 0 for horizontal (x-axis), 1 for vertical (y-axis).

    Returns:
        A 1D numpy array representing the projection histogram.
    """
    assert axis in [0, 1]
    max_length = int(np.max(boxes[:, axis::2]))
    projection = np.zeros(max_length, dtype=int)

    # Increment projection histogram over the interval defined by each bounding box
    for start, end in boxes[:, axis::2]:
        start, end = int(start), int(end)
        if start < end and start >= 0 and end <= max_length:
            projection[start:end] += 1

    return projection


def _split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """Split the projection profile into segments based on specified thresholds.

    Args:
        arr_values: 1D array representing the projection profile.
        min_value: Minimum value threshold to consider a profile segment significant.
        min_gap: Minimum gap width to consider a separation between segments.

    Returns:
        A tuple of start and end indices for each segment that meets the criteria.
    """
    # Identify indices where the projection exceeds the minimum value
    significant_indices = np.where(arr_values > min_value)[0]
    if not len(significant_indices):
        return None

    # Calculate gaps between significant indices
    index_diffs = significant_indices[1:] - significant_indices[:-1]
    gap_indices = np.where(index_diffs > min_gap)[0]

    # Determine start and end indices of segments
    segment_starts = np.insert(
        significant_indices[gap_indices + 1],
        0,
        significant_indices[0],
    )
    segment_ends = np.append(
        significant_indices[gap_indices],
        significant_indices[-1] + 1,
    )

    return segment_starts, segment_ends


def _recursive_xy_cut(
    boxes: np.ndarray,
    indices: List[int],
    res: List[int],
    segment_ids: List[int],
    current_id: int,
    min_gap: int = 1,
):
    """Recursively performs X-axis projection followed by Y-axis projection to segment bounding boxes.

    Args:
        boxes: A (N, 4) array representing bounding boxes with [x_min, y_min, x_max, y_max].
        indices: A list of indices representing the position of boxes in the original data.
        res: A list to store indices of bounding boxes that meet the criteria.
        segment_ids: A list to store segment IDs for each box.
        current_id: Current segment ID.
        min_gap: Minimum gap width to consider a separation between segments.
    """
    assert len(boxes) == len(indices)

    # Sort by x_min to prepare for X-axis projection
    x_sorted_indices = boxes[:, 0].argsort()
    x_sorted_boxes = boxes[x_sorted_indices]
    x_sorted_indices = np.array(indices)[x_sorted_indices]

    # Perform X-axis projection
    x_projection = _projection_by_bboxes(boxes=x_sorted_boxes, axis=0)
    x_intervals = _split_projection_profile(x_projection, 0, 1)

    if not x_intervals:
        for idx in indices:
            segment_ids.append(current_id)
        res.extend(indices)
        return

    # Process each segment defined by X-axis projection
    for x_start, x_end in zip(*x_intervals):
        # Select boxes within the current x interval
        x_interval_indices = (x_start <= x_sorted_boxes[:, 0]) & (
            x_sorted_boxes[:, 0] < x_end
        )
        x_boxes_chunk = x_sorted_boxes[x_interval_indices]
        x_indices_chunk = x_sorted_indices[x_interval_indices]

        # Sort selected boxes by y_min to prepare for Y-axis projection
        y_sorted_indices = x_boxes_chunk[:, 1].argsort()
        y_sorted_boxes_chunk = x_boxes_chunk[y_sorted_indices]
        y_sorted_indices_chunk = x_indices_chunk[y_sorted_indices]

        # Perform Y-axis projection
        y_projection = _projection_by_bboxes(boxes=y_sorted_boxes_chunk, axis=1)
        y_intervals = _split_projection_profile(y_projection, 0, min_gap)

        if not y_intervals:
            for idx in y_sorted_indices_chunk:
                segment_ids.append(current_id)
            current_id += 1
            res.extend(y_sorted_indices_chunk)
            continue

        # If Y-axis cannot be further segmented, add current indices to results
        if len(y_intervals[0]) == 1:
            for idx in y_sorted_indices_chunk:
                segment_ids.append(current_id)
            current_id += 1
            res.extend(y_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by Y-axis projection
        for y_start, y_end in zip(*y_intervals):
            y_interval_indices = (y_start <= y_sorted_boxes_chunk[:, 1]) & (
                y_sorted_boxes_chunk[:, 1] < y_end
            )
            _recursive_xy_cut(
                y_sorted_boxes_chunk[y_interval_indices],
                y_sorted_indices_chunk[y_interval_indices],
                res,
                segment_ids,
                current_id,
            )
            current_id += 1


def sort_by_xycut(
    block_bboxes: List[List[float]],
    direction: int = 0,
    min_gap: int = 1,
) -> Tuple[List[int], List[int]]:
    """Sort bounding boxes using recursive XY cut method.

    Args:
        block_bboxes: List of bounding boxes, each as [x_min, y_min, x_max, y_max].
        direction: Direction for the initial cut. Use 1 for Y-axis first and 0 for X-axis first.
        min_gap: Minimum gap width to consider a separation between segments.

    Returns:
        Tuple of (sorted_indices, segment_ids) where:
        - sorted_indices: List of indices representing the reading order
        - segment_ids: List of segment IDs for each box
    """
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []
    segment_ids = []
    current_id = 0

    if direction == 1:
        # Y-axis first (not implemented in the provided code, using X-axis first)
        pass
    else:
        _recursive_xy_cut(
            block_bboxes,
            np.arange(len(block_bboxes)).tolist(),
            res,
            segment_ids,
            current_id,
            min_gap,
        )

    return res, segment_ids


def order_layout_elements_by_xycut(
    layout_elements: List[Dict[str, Any]],
    min_gap: int = 1,
) -> List[Dict[str, Any]]:
    """Order layout elements using XY-cut algorithm.

    Args:
        layout_elements: List of layout elements with 'bbox', 'type', 'content' keys.
        min_gap: Minimum gap for XY-cut segmentation.

    Returns:
        List of layout elements ordered by reading sequence.
    """
    if not layout_elements:
        return []

    # Extract bboxes for XY-cut
    bboxes = []
    for element in layout_elements:
        bbox = element.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            bboxes.append(bbox)
        else:
            bboxes.append([0, 0, 0, 0])

    # Apply XY-cut sorting
    sorted_indices, segment_ids = sort_by_xycut(bboxes, direction=0, min_gap=min_gap)

    # Reorder elements according to XY-cut result
    ordered_elements = []
    for idx in sorted_indices:
        element = layout_elements[idx].copy()
        element["reading_order"] = len(ordered_elements) + 1
        element["segment_id"] = segment_ids[idx] if idx < len(segment_ids) else 0
        ordered_elements.append(element)

    return ordered_elements


def elements_to_markdown(
    layout_elements: List[Dict[str, Any]],
    use_xycut: bool = True,
    min_gap: int = 1,
) -> str:
    """Convert layout elements to Markdown with reading order preservation.

    Args:
        layout_elements: List of layout elements with 'bbox', 'type', 'content' keys.
        use_xycut: Whether to use XY-cut for ordering (True) or simple bbox sort (False).
        min_gap: Minimum gap for XY-cut segmentation.

    Returns:
        Markdown string with elements in reading order.
    """
    if not layout_elements:
        return ""

    # Filter out elements without content
    valid_elements = [e for e in layout_elements if e.get("content", "").strip()]

    if not valid_elements:
        return ""

    # Order elements
    if use_xycut:
        ordered_elements = order_layout_elements_by_xycut(valid_elements, min_gap)
    else:
        # Simple bbox sorting (top-left to bottom-right)
        ordered_elements = sorted(
            valid_elements,
            key=lambda e: (
                e.get("bbox", [0, 0, 0, 0])[1],
                e.get("bbox", [0, 0, 0, 0])[0],
            ),
        )

    # Convert to Markdown
    markdown_parts = []
    for element in ordered_elements:
        content = element.get("content", "").strip()
        if not content:
            continue

        element_type = element.get("type", "text").lower()

        if element_type == "title":
            # Convert titles to H1 headings
            markdown_parts.append(f"# {content}")
        elif element_type == "table" and content.startswith("<table>"):
            # Table content should already be in HTML format from VLM
            markdown_parts.append(content)
        elif element_type in ["figure", "image"]:
            # Handle figures/images - could be caption text or placeholder
            if content.lower().startswith("fig") or content.lower().startswith("image"):
                markdown_parts.append(f"*{content}*")
            else:
                markdown_parts.append(content)
        else:
            # Regular text content
            markdown_parts.append(content)

    return "\n\n".join(markdown_parts)


def pages_to_markdown(
    pages: List[Dict[str, Any]],
    use_xycut: bool = True,
    min_gap: int = 1,
) -> str:
    """Convert multiple pages to Markdown preserving reading order.

    Args:
        pages: List of page dictionaries from oculodoc processing.
        use_xycut: Whether to use XY-cut for ordering.
        min_gap: Minimum gap for XY-cut segmentation.

    Returns:
        Complete document Markdown string.
    """
    document_parts = []

    for page in pages:
        # Handle hybrid processor output (layout_elements)
        if "layout_elements" in page:
            page_md = elements_to_markdown(
                page["layout_elements"], use_xycut=use_xycut, min_gap=min_gap
            )
            if page_md.strip():
                document_parts.append(page_md)

        # Handle VLM-only output (content)
        elif "content" in page:
            content = page.get("content", "").strip()
            if content:
                document_parts.append(content)

        # Handle VLM content from hybrid processing
        elif "vlm_content" in page:
            content = page.get("vlm_content", "").strip()
            if content:
                document_parts.append(content)

    return "\n\n".join(document_parts)
