"""Line art tracing and skeleton extraction for ContourGenerator."""

from __future__ import annotations

import numpy as np

from plottter.generators._helpers import _px_to_mm
from plottter.models import Polyline

from ._smoothing import _chaikin_smooth


def _trace_line_art(
    gray: np.ndarray,
    threshold: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int,
    adaptive_threshold: bool = False,
    adaptive_c: float = 5.0,
) -> list[Polyline]:
    """Trace line art by direct binary thresholding + full-resolution contour tracing.

    Unlike the multi-level Contour Levels mode, this function uses a single
    threshold to separate ink from background and then traces the outlines of
    every ink region at full pixel resolution.  This is optimised for clean
    B&W line drawings (logos, icons, sketches) where the strokes should be
    reproduced as faithfully as possible rather than approximated by brightness
    isolines.

    Key differences from Contour Levels:
    - Single threshold (THRESH_BINARY_INV so dark pixels become the region)
    - cv2.RETR_CCOMP to capture both outer and hole contours (inner stroke edges)
    - cv2.CHAIN_APPROX_NONE for maximum contour resolution before RDP
    - Optional Chaikin smoothing to convert pixel-stepped boundaries into
      smooth plotter-friendly curves
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    from plottter.generators._helpers import _apply_threshold

    # Invert threshold: pixels *darker* than threshold are treated as ink
    binary = _apply_threshold(
        gray, threshold, adaptive_threshold, adaptive_c, cv2.THRESH_BINARY_INV
    )

    # RETR_CCOMP returns outer boundaries + inner hole boundaries
    # CHAIN_APPROX_NONE preserves every pixel on the contour boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    polylines: list[Polyline] = []
    for contour in contours:
        if len(contour) < min_length:
            continue

        # RDP simplification — gentler tolerance in line art mode
        if simplify_tol > 0:
            tol_px = max(0.5, simplify_tol * px_per_mm)
            contour = cv2.approxPolyDP(contour, tol_px, closed=True)

        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        # Convert pixel coords to mm
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]

        # Chaikin smoothing: converts pixel-stepped outlines into smooth curves
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=True)

        # Close the contour (always needed — Chaikin with an open input stays open)
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])

        polylines.append(poly)

    return polylines


def _trace_skeleton(
    gray: np.ndarray,
    threshold: int,
    cleanup_kernel: int,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    simplify_tol: float,
    min_length: int,
    smooth_iterations: int,
    merge_gap_mm: float = 0.0,
    adaptive_threshold: bool = False,
    adaptive_c: float = 5.0,
) -> list[Polyline]:
    """Trace centerlines of thick ink strokes via morphological skeletonization.

    Pipeline:
    1. Threshold (THRESH_BINARY_INV): dark ink → white foreground.
    2. Optional morphological cleanup (MORPH_CLOSE then MORPH_OPEN).
    3. Skeletonize: reduce ink regions to single-pixel-wide centerlines.
    4. Walk the skeleton pixel graph: extract one polyline per branch
       (segment between junction/endpoint pixels), producing true centerline
       paths rather than the boundary outline of each connected component.
    5. Optional RDP simplification + mm coordinate conversion.
    6. Optional fragment merging: connect nearby endpoint pairs within
       ``merge_gap_mm`` to reduce disconnected short segments at junctions.
    7. Optional Chaikin smoothing.

    This produces approximately **one polyline per original thick line**,
    unlike Line Art Trace which traces both the inner and outer edge of
    each stroke (two polylines per line).

    Parameters
    ----------
    gray:            Grayscale uint8 image (H×W).
    threshold:       Brightness threshold (0–255); pixels ≤ threshold are ink.
    cleanup_kernel:  Morphological kernel size (0 = disabled, 1–10 = enabled).
    img_w, img_h:    Source image dimensions in pixels.
    draw_x1..y2:     Canvas drawing area bounds in mm.
    simplify_tol:    RDP simplification tolerance in mm.
    min_length:      Minimum path length in pixels (shorter paths discarded).
    smooth_iterations: Number of Chaikin smoothing passes.
    merge_gap_mm:    Maximum endpoint distance (mm) for fragment merging
                     (0 = disabled).
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ContourGenerator.")

    from plottter.generators._helpers import (
        _apply_threshold,
        _px_to_mm,
        _skeletonize,
        _walk_skeleton_branches,
    )

    # Step 1: threshold — dark ink (≤ threshold) → white foreground
    binary = _apply_threshold(
        gray, threshold, adaptive_threshold, adaptive_c, cv2.THRESH_BINARY_INV
    )

    # Step 2: optional morphological cleanup
    if cleanup_kernel >= 1:
        k = int(cleanup_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # MORPH_CLOSE fills small gaps in the ink; MORPH_OPEN removes noise specks
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 3: skeletonize — reduce white ink regions to 1-pixel-wide centerlines
    skeleton = _skeletonize(binary)

    # Step 4: extract centerline polylines by walking the skeleton pixel graph.
    # Each branch between junction/endpoint pixels becomes one polyline, so
    # branching ink structures (e.g. letter strokes, face features) produce
    # multiple independent branches rather than a single boundary outline.
    pixel_paths = _walk_skeleton_branches(skeleton, min_length)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    # Step 5: RDP simplification and pixel→mm conversion
    raw_polylines: list[Polyline] = []
    for pixel_path in pixel_paths:
        if len(pixel_path) < 2:
            continue

        if simplify_tol > 0:
            tol_px = max(0.5, simplify_tol * px_per_mm)
            pts = np.array(pixel_path, dtype=np.float32).reshape(-1, 1, 2)
            pts = cv2.approxPolyDP(pts, tol_px, closed=False)
            pts = pts.reshape(-1, 2)
        else:
            pts = np.array(pixel_path, dtype=np.float32)

        if len(pts) < 2:
            continue

        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts
        ]
        if len(poly) >= 2:
            raw_polylines.append(poly)

    # Step 6: optional fragment merging — connect nearby endpoint pairs
    # before smoothing so that merged segments receive a single smooth pass.
    if merge_gap_mm > 0 and raw_polylines:
        from plottter.processing.merge import merge_fragments
        raw_polylines = merge_fragments(raw_polylines, merge_gap_mm)

    # Step 7: optional Chaikin smoothing applied to the (possibly merged) polylines
    polylines: list[Polyline] = []
    for poly in raw_polylines:
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=False)
        if len(poly) >= 2:
            polylines.append(poly)

    return polylines
