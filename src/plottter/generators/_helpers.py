"""Shared helpers for image generators."""

from __future__ import annotations

import numpy as np


def _apply_threshold(
    gray: np.ndarray,
    threshold: int,
    adaptive: bool,
    adaptive_c: float,
    thresh_type: int,
) -> np.ndarray:
    """Apply global or adaptive (Gaussian-weighted) thresholding.

    Parameters
    ----------
    gray:        uint8 single-channel image.
    threshold:   Global threshold value (0–255), used when ``adaptive=False``.
    adaptive:    If True, uses ``cv2.ADAPTIVE_THRESH_GAUSSIAN_C`` with a
                 block size automatically derived from image dimensions.
    adaptive_c:  Constant subtracted from the local Gaussian-weighted mean.
                 Positive → stricter (fewer foreground pixels); negative →
                 more permissive.  Rounded to int before passing to OpenCV.
    thresh_type: ``cv2.THRESH_BINARY`` or ``cv2.THRESH_BINARY_INV``.

    Returns
    -------
    uint8 binary image with values 0 or 255.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "opencv-python is required for thresholding."
        ) from exc

    if adaptive:
        h, w = gray.shape[:2]
        # block_size must be odd and ≥ 3; scale proportionally with image size
        # so fine detail is captured in small images and larger neighbourhoods
        # are used for high-resolution scans.
        raw = int(min(h, w) * 0.02)
        block_size = max(3, raw | 1)  # bitwise OR with 1 ensures odd
        c_val = int(round(adaptive_c))
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            c_val,
        )
    else:
        _, binary = cv2.threshold(gray, threshold, 255, thresh_type)
        return binary


def _px_to_mm(
    px_x: float,
    px_y: float,
    img_w: int,
    img_h: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
) -> tuple[float, float]:
    """Convert image pixel coordinates to canvas mm coordinates."""
    x = draw_x1 + px_x * (draw_x2 - draw_x1) / img_w
    y = draw_y1 + px_y * (draw_y2 - draw_y1) / img_h
    return (x, y)


def compute_image_rect(
    fit_mode: str,
    image_w_px: int,
    image_h_px: int,
    draw_x1: float,
    draw_y1: float,
    draw_x2: float,
    draw_y2: float,
    custom_w_mm: float | None = None,
    custom_h_mm: float | None = None,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute the mm rectangle where the source image should be mapped.

    Parameters
    ----------
    fit_mode:
        ``"fill"`` — image fills the entire drawing area (default/current behavior).
        ``"fit"`` — scale image to fit within drawing area preserving aspect ratio,
        centered with offset applied.
        ``"custom"`` — use explicit ``custom_w_mm`` × ``custom_h_mm`` size, centered
        with offset applied.
    image_w_px, image_h_px:
        Source image dimensions in pixels.
    draw_x1, draw_y1, draw_x2, draw_y2:
        Drawing area rectangle in mm (from ``canvas.drawing_area()``).
    custom_w_mm, custom_h_mm:
        Explicit output size in mm; only used when ``fit_mode == "custom"``.
    offset_x_mm, offset_y_mm:
        Horizontal/vertical offset in mm from the centered position.  Only
        applied when ``fit_mode`` is ``"fit"`` or ``"custom"``.

    Returns
    -------
    ``(img_x1, img_y1, img_x2, img_y2)`` — the mm rectangle to map the image into.
    """
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1

    if fit_mode == "fill" or image_w_px <= 0 or image_h_px <= 0:
        return (draw_x1, draw_y1, draw_x2, draw_y2)

    cx = (draw_x1 + draw_x2) / 2.0 + offset_x_mm
    cy = (draw_y1 + draw_y2) / 2.0 + offset_y_mm

    if fit_mode == "fit":
        aspect = image_w_px / image_h_px
        canvas_aspect = draw_w / draw_h if draw_h > 0 else 1.0
        if aspect > canvas_aspect:
            w = draw_w
            h = draw_w / aspect
        else:
            h = draw_h
            w = draw_h * aspect
    elif fit_mode == "custom":
        w = custom_w_mm if custom_w_mm is not None else draw_w
        h = custom_h_mm if custom_h_mm is not None else draw_h
    else:
        return (draw_x1, draw_y1, draw_x2, draw_y2)

    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin foreground (255) pixels to single-pixel-wide centerlines.

    Tries ``cv2.ximgproc.thinning`` first (opencv-contrib-python), then
    ``skimage.morphology.thin``, then falls back to a pure NumPy Zhang-Suen
    implementation.

    Parameters
    ----------
    binary:
        uint8 image, foreground = 255, background = 0.

    Returns
    -------
    uint8 image with same shape; skeleton pixels = 255.
    """
    # Try OpenCV contrib (fastest — requires opencv-contrib-python)
    try:
        import cv2
        if hasattr(cv2, "ximgproc"):
            return cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
    except Exception:  # noqa: BLE001
        pass

    # Try scikit-image
    try:
        from skimage.morphology import thin as _skimage_thin  # type: ignore[import]
        skel = _skimage_thin(binary > 0)
        return skel.astype(np.uint8) * 255
    except ImportError:
        pass

    # Pure NumPy Zhang-Suen thinning fallback
    return _zhang_suen_thinning(binary)


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Vectorized Zhang-Suen binary thinning (pure NumPy).

    Reference: Zhang, T.Y. and Suen, C.Y. (1984), "A fast parallel
    algorithm for thinning digital patterns", Commun. ACM 27(3).

    Parameters
    ----------
    binary:
        uint8 image, foreground = 255, background = 0.

    Returns
    -------
    uint8 image with same shape; skeleton pixels = 255.
    """
    img = (binary > 0).astype(bool)

    while True:
        changed = False
        for step in range(2):
            P2 = img[:-2, 1:-1]   # North
            P3 = img[:-2, 2:]     # NE
            P4 = img[1:-1, 2:]    # East
            P5 = img[2:, 2:]      # SE
            P6 = img[2:, 1:-1]    # South
            P7 = img[2:, :-2]     # SW
            P8 = img[1:-1, :-2]   # West
            P9 = img[:-2, :-2]    # NW

            N = (
                P2.astype(np.int32) + P3.astype(np.int32)
                + P4.astype(np.int32) + P5.astype(np.int32)
                + P6.astype(np.int32) + P7.astype(np.int32)
                + P8.astype(np.int32) + P9.astype(np.int32)
            )

            # Count 0→1 transitions in clockwise order: P2,P3,P4,P5,P6,P7,P8,P9,P2
            A = (
                (~P2 & P3).astype(np.int32) + (~P3 & P4).astype(np.int32)
                + (~P4 & P5).astype(np.int32) + (~P5 & P6).astype(np.int32)
                + (~P6 & P7).astype(np.int32) + (~P7 & P8).astype(np.int32)
                + (~P8 & P9).astype(np.int32) + (~P9 & P2).astype(np.int32)
            )

            cond1 = (N >= 2) & (N <= 6)
            cond2 = A == 1

            if step == 0:
                cond3 = ~(P2 & P4 & P6)
                cond4 = ~(P4 & P6 & P8)
            else:
                cond3 = ~(P2 & P4 & P8)
                cond4 = ~(P2 & P6 & P8)

            center = img[1:-1, 1:-1]
            mask = center & cond1 & cond2 & cond3 & cond4

            if mask.any():
                changed = True
                img[1:-1, 1:-1] = center & ~mask

        if not changed:
            break

    return img.astype(np.uint8) * 255
