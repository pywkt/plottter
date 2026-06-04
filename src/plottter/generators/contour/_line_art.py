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
    supersample: int = 1,
) -> list[Polyline]:
    """Trace line art outlines using sub-pixel marching-squares contour extraction.

    Uses ``skimage.measure.find_contours`` (marching squares) instead of
    ``cv2.findContours``.  Marching squares linearly interpolates where the
    iso-value crosses *between* adjacent pixels, so diagonal edges become smooth
    diagonal lines rather than pixel-grid staircases.

    Key differences from the old ``cv2.findContours`` approach:

    - Standard threshold path: the grayscale is fed directly at ``float(threshold)``
      as the marching-squares iso-level (boundary between dark ``< threshold`` and
      light ``>= threshold`` pixels — identical to THRESH_BINARY_INV logic).
    - Open contours (those that touch the image border) are **not** force-closed;
      the old unconditional close produced a spurious chord for border-crossing
      strokes.
    - Chaikin smoothing and RDP use ``closed=is_closed`` so open strokes stay open.

    Adaptive threshold:
        When ``adaptive_threshold=True``, the adaptive binary mask (values 0/255)
        is computed first, then fed to ``extract_subpixel_contours`` at
        ``level=127``.  Marching squares on the 0/255 mask still yields cleaner
        half-pixel diagonals than ``findContours``, and the local thresholding
        behaviour is unchanged.

    Unlike the multi-level Contour Levels mode, this function uses a single
    threshold to separate ink from background and then traces the outlines of
    every ink region.  This is optimised for clean B&W line drawings (logos,
    icons, sketches) where the strokes should be reproduced as faithfully as
    possible rather than approximated by brightness isolines.
    """
    from ._subpixel import extract_subpixel_contours

    if adaptive_threshold:
        try:
            import cv2
        except ImportError:  # pragma: no cover
            raise RuntimeError("opencv-python is required for ContourGenerator.")
        from plottter.generators._helpers import _apply_threshold
        # Compute adaptive binary mask (dark ink → 255, light background → 0),
        # then feed to extract_subpixel_contours at level=127 (midpoint of 0/255).
        binary = _apply_threshold(
            gray, threshold, True, adaptive_c, cv2.THRESH_BINARY_INV
        )
        raw_contours = extract_subpixel_contours(binary, 127.0, min_length, supersample)
    else:
        # Marching squares at iso-level=threshold finds the boundary between
        # dark (< threshold) and light (>= threshold) pixels — equivalent to
        # THRESH_BINARY_INV followed by findContours, but with sub-pixel accuracy.
        raw_contours = extract_subpixel_contours(gray, float(threshold), min_length, supersample)

    px_per_mm = img_w / (draw_x2 - draw_x1) if (draw_x2 - draw_x1) > 0 else 1.0

    polylines: list[Polyline] = []
    for pts_xy, is_closed in raw_contours:
        # pts_xy is (N, 2) float array of (x, y) = (col, row) pixel coordinates

        # RDP simplification — gentler tolerance in line art mode
        if simplify_tol > 0:
            try:
                import cv2  # noqa: F811
            except ImportError:  # pragma: no cover
                raise RuntimeError("opencv-python is required for ContourGenerator.")
            tol_px = max(0.5, simplify_tol * px_per_mm)
            pts_arr = pts_xy.reshape(-1, 1, 2).astype(np.float32)
            pts_arr = cv2.approxPolyDP(pts_arr, tol_px, closed=is_closed)
            pts_xy = pts_arr.reshape(-1, 2)

        if len(pts_xy) < 2:
            continue

        # Convert pixel coords to mm
        poly: Polyline = [
            _px_to_mm(float(p[0]), float(p[1]), img_w, img_h,
                      draw_x1, draw_y1, draw_x2, draw_y2)
            for p in pts_xy
        ]

        # Chaikin smoothing with closed=is_closed: open strokes stay open
        if smooth_iterations > 0:
            poly = _chaikin_smooth(poly, smooth_iterations, closed=is_closed)

        # Close the contour only when it is a closed interior loop; open
        # border-crossing strokes must NOT be force-closed (doing so would add a
        # spurious chord from the last point back to the first).
        if is_closed and poly and poly[0] != poly[-1]:
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
