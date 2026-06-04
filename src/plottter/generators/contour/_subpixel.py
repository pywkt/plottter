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
