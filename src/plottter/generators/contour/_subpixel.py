"""Shared sub-pixel contour extraction helper.

Pure-numeric module — imports only numpy, skimage.measure, and cv2.
No Qt or plottter.gui imports allowed here.
"""

from __future__ import annotations

import numpy as np


def extract_subpixel_contours(
    gray: np.ndarray,
    level: float,
    min_points: int,
    supersample: int = 1,
) -> list[tuple[np.ndarray, bool]]:
    """Return sub-pixel iso-contours from a grayscale image.

    Uses ``skimage.measure.find_contours`` (marching squares) which linearly
    interpolates where the iso-value crosses between adjacent pixels, yielding
    smooth diagonal edges rather than axis-aligned staircases.

    Parameters
    ----------
    gray:
        ``uint8`` H×W grayscale image.
    level:
        Iso-value in ``[0, 255]``; boundary between ``< level`` and
        ``>= level`` pixels.
    min_points:
        Discard contours with fewer than this many vertices.
    supersample:
        1 = off.  2–4 upsamples *gray* with cubic interpolation before
        extraction to give marching squares more sub-pixel resolution on
        low-resolution sources.  The returned coordinates are always in
        original pixel space (divided back by the factor).

    Returns
    -------
    list of ``(pts_xy, is_closed)`` where

    *   ``pts_xy`` is an ``(N, 2)`` float array of ``(x, y) = (col, row)``
        pixel coordinates (skimage's ``(row, col)`` ordering is swapped).
    *   ``is_closed`` is ``True`` when the contour forms a closed loop
        (does not touch the image border); detected via
        ``np.allclose(pts[0], pts[-1])``.

    No RDP simplification, smoothing, or mm-conversion is applied here —
    callers own those steps so per-mode behaviour is preserved.

    Notes
    -----
    **Adaptive-threshold callers:** feed the binary mask (values 0 / 255)
    with ``level=127``.  Marching squares on a 0/255 mask still yields
    cleaner half-pixel diagonals compared with ``cv2.findContours``.
    """
    from skimage.measure import find_contours  # type: ignore[import]

    factor = max(1, int(supersample))

    if factor > 1:
        try:
            import cv2  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opencv-python is required for supersample > 1 in "
                "extract_subpixel_contours."
            ) from exc
        h, w = gray.shape
        new_w, new_h = w * factor, h * factor
        gray_up = cv2.resize(
            gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
        gray_f = gray_up.astype(np.float64)
    else:
        gray_f = gray.astype(np.float64)
        factor = 1

    raw = find_contours(gray_f, level)

    result: list[tuple[np.ndarray, bool]] = []
    for contour in raw:
        # contour is (N, 2) array of (row, col); swap to (x, y) = (col, row)
        pts = np.column_stack((contour[:, 1], contour[:, 0]))

        if factor > 1:
            pts = pts / factor

        if len(pts) < min_points:
            continue

        is_closed = bool(np.allclose(pts[0], pts[-1]))
        result.append((pts, is_closed))

    return result


def build_contour_hierarchy(
    closed_contours: list[np.ndarray],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Group rings into (outer, [holes]) pairs by containment nesting.

    Build a containment forest: ring A is a child of the smallest ring that
    fully contains a representative point of A.  A ring at even nesting depth
    (0, 2, …) is a filled region (outer); a ring at odd depth (1, 3, …) is a
    hole of its immediate parent.  Returns one ``(outer_ring, [hole_ring, …])``
    pair per even-depth ring, attaching only its *direct* odd-depth children as
    holes (mirrors and generalises ``cv2.RETR_CCOMP``'s two-level hierarchy to
    arbitrary nesting).

    Parameters
    ----------
    closed_contours:
        List of ``(N, 2)`` float arrays giving ``(x, y)`` pixel coordinates of
        closed rings.  Each ring must be closed (first ≈ last vertex), but the
        closing duplicate vertex need not be present — Shapely ``Polygon``
        handles both.

    Returns
    -------
    list of ``(outer_ring, hole_rings)`` where *outer_ring* is an ``(N, 2)``
    array and *hole_rings* is a (possibly empty) list of ``(M, 2)`` arrays.
    Rings are sorted so that larger outers come first.

    Notes
    -----
    Degenerate or self-intersecting rings are skipped defensively: a
    ``buffer(0)`` is applied to the Shapely ``Polygon``; if the result is not
    a simple ``Polygon``, or has area below ``1e-9``, the ring is dropped.
    """
    from shapely.geometry import MultiPolygon, Polygon  # already a dependency

    if not closed_contours:
        return []

    # Build Shapely polygons; fix / skip degenerate / self-intersecting rings
    valid_polys: list[Polygon] = []
    valid_rings: list[np.ndarray] = []

    for ring in closed_contours:
        try:
            poly: object = Polygon(ring)
            if not poly.is_valid:  # type: ignore[union-attr]
                poly = poly.buffer(0)  # type: ignore[union-attr]
            # buffer(0) may return MultiPolygon for badly self-intersecting rings
            if isinstance(poly, MultiPolygon) or poly.is_empty or poly.area < 1e-9:  # type: ignore[union-attr]
                continue
            valid_polys.append(poly)  # type: ignore[arg-type]
            valid_rings.append(ring)
        except Exception:  # pragma: no cover
            continue

    if not valid_polys:
        return []

    # Sort by area descending so a candidate parent is always encountered
    # before its children when building the forest
    order = sorted(range(len(valid_polys)), key=lambda i: valid_polys[i].area, reverse=True)
    sorted_polys = [valid_polys[i] for i in order]
    sorted_rings = [valid_rings[i] for i in order]

    n = len(sorted_polys)

    # Build parent[] array.
    # For ring i, scan from j = i-1 down to 0 (smallest → largest area).
    # The first j whose polygon contains a representative point of ring i is
    # the immediate parent (smallest containing ring).
    parent: list[int] = [-1] * n

    for i in range(1, n):
        rep = sorted_polys[i].representative_point()
        for j in range(i - 1, -1, -1):
            if sorted_polys[j].contains(rep):
                parent[i] = j
                break

    # Compute nesting depths via the parent[] array
    depth: list[int] = [0] * n
    for i in range(n):
        if parent[i] != -1:
            depth[i] = depth[parent[i]] + 1

    # Build children adjacency list, then collect (outer, holes) pairs
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] != -1:
            children[parent[i]].append(i)

    result: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for i in range(n):
        if depth[i] % 2 == 0:  # even depth → outer (filled region)
            hole_rings = [sorted_rings[j] for j in children[i] if depth[j] % 2 == 1]
            result.append((sorted_rings[i], hole_rings))

    return result
