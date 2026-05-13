"""Shared helpers for image generators."""

from __future__ import annotations

import numpy as np


def _load_source_image(value: object) -> np.ndarray | None:
    """Load a source image from a file path or return an array as-is.

    Parameters
    ----------
    value:
        - ``np.ndarray``: returned unchanged.
        - Non-empty ``str``: loaded via :func:`plottter.io.image_import.load_image`.
        - ``None`` or empty string: returns ``None``.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str) and value:
        from plottter.io.image_import import load_image
        return load_image(value)
    return None


def _compute_etf(
    gray_f: np.ndarray,
    sigma_m: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Edge Tangent Flow (ETF) field.

    The ETF is a dense vector field where each pixel holds the unit tangent
    direction of the nearest edge.  It is initialised from the normalised
    Sobel gradient rotated 90° (tangent = perpendicular to gradient) and
    then iteratively smoothed with a magnitude-weighted Gaussian kernel.
    During each pass, tangent vectors at pixels with stronger gradients
    exert more influence, pulling nearby vectors into alignment and
    producing a smooth, coherent flow along edges.

    Parameters
    ----------
    gray_f:     Float32 grayscale image, values in [0, 1].
    sigma_m:    Spatial scale for ETF smoothing (pixels).  Controls how far
                edge tangents influence their neighbourhood.
    iterations: Number of smoothing passes (3–5 typical).

    Returns
    -------
    (tx, ty): Pair of float32 arrays of shape (H, W) holding the x and y
              components of the unit tangent at each pixel.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for FDoG/Coherent Line generation."
        ) from exc

    # Compute image gradient with a 5×5 Sobel kernel for stability
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=5)

    mag = np.sqrt(gx * gx + gy * gy)
    mag_max = float(mag.max())
    mag_norm = mag / (mag_max + 1e-8)

    # Initial tangent: rotate gradient 90° — edges run perpendicular to ∇f
    # gradient (gx, gy) → tangent (-gy, gx)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(mag > 1e-8, -gy / (mag + 1e-8), 0.0).astype(np.float32)
        ty = np.where(mag > 1e-8,  gx / (mag + 1e-8), 0.0).astype(np.float32)

    # Clamp sigma_m to avoid degenerate kernel sizes
    sigma_m = max(float(sigma_m), 0.5)
    r = max(2, int(round(2.5 * sigma_m)))
    ksize = 2 * r + 1

    for _ in range(max(1, iterations)):
        # Magnitude-weighted smoothing: pixels with larger gradients pull
        # nearby tangents into alignment.
        smooth_tx = cv2.GaussianBlur(tx * mag_norm, (ksize, ksize), sigma_m)
        smooth_ty = cv2.GaussianBlur(ty * mag_norm, (ksize, ksize), sigma_m)

        # Re-normalise to keep unit vectors
        new_mag = np.sqrt(smooth_tx * smooth_tx + smooth_ty * smooth_ty)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                new_mag > 1e-8, smooth_tx / (new_mag + 1e-8), 0.0
            ).astype(np.float32)
            ty = np.where(
                new_mag > 1e-8, smooth_ty / (new_mag + 1e-8), 0.0
            ).astype(np.float32)

        # Update the magnitude estimate used for the next pass
        smooth_mag = cv2.GaussianBlur(mag_norm, (ksize, ksize), sigma_m)
        smooth_max = float(smooth_mag.max())
        mag_norm = smooth_mag / (smooth_max + 1e-8)

    return tx, ty


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


def _walk_skeleton_branches(
    skeleton: np.ndarray,
    min_pixels: int,
) -> list[list[tuple[int, int]]]:
    """Walk a skeleton image and extract ordered polyline pixel paths.

    Extracts centerlines by pixel-graph walking: the skeleton is treated as
    a graph where each foreground pixel is a node connected to its 8-connected
    neighbours.  Branches between junction/endpoint pixels are each walked into
    one polyline, producing true centerline paths rather than boundary outlines.

    Parameters
    ----------
    skeleton:
        uint8 image, skeleton pixels = 255, background = 0.
    min_pixels:
        Minimum path length (in pixels) to keep; shorter paths are discarded.

    Returns
    -------
    List of pixel coordinate lists ``[(x, y), ...]`` (x first, image space).
    """
    from scipy.ndimage import convolve as _nd_convolve  # type: ignore[import]

    skel = skeleton > 0
    if not skel.any():
        return []

    # Count 8-connected skeleton neighbours per pixel with a 3x3 convolution.
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    ncount = _nd_convolve(skel.astype(np.int32), kernel, mode="constant", cval=0)
    ncount = ncount * skel  # zero out non-skeleton pixels

    # Classify pixels by connectivity.
    endpoint_mask = skel & (ncount == 1)
    junction_mask = skel & (ncount >= 3)

    ys, xs = np.where(skel)
    pixel_set: set[tuple[int, int]] = set(zip(ys.tolist(), xs.tolist()))

    endpoints: set[tuple[int, int]] = (
        set(zip(*np.where(endpoint_mask))) if endpoint_mask.any() else set()
    )
    junctions: set[tuple[int, int]] = (
        set(zip(*np.where(junction_mask))) if junction_mask.any() else set()
    )
    stop_pixels = endpoints | junctions

    # ------------------------------------------------------------------
    # Neighbour lookup (8-connected, staying in pixel_set).
    # ------------------------------------------------------------------
    def nbrs(y: int, x: int) -> list[tuple[int, int]]:
        result = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in pixel_set:
                    result.append(nb)
        return result

    # ------------------------------------------------------------------
    # Branch walking: follow the chain from a stop pixel to the next
    # stop pixel (or dead end).
    # ------------------------------------------------------------------
    def walk_branch(
        start: tuple[int, int], first: tuple[int, int]
    ) -> list[tuple[int, int]]:
        path = [start, first]
        prev, curr = start, first
        while curr not in stop_pixels:
            candidates = [n for n in nbrs(*curr) if n != prev]
            if not candidates:
                break
            prev, curr = curr, candidates[0]
            path.append(curr)
        return path

    walked_edges: set[tuple[tuple, tuple]] = set()
    visited: set[tuple[int, int]] = set()
    polylines: list[list[tuple[int, int]]] = []

    # Walk all branches from stop pixels (endpoints and junctions).
    for sp in stop_pixels:
        for n in nbrs(*sp):
            edge = (min(sp, n), max(sp, n))
            if edge in walked_edges:
                continue
            walked_edges.add(edge)
            path = walk_branch(sp, n)
            # Mark all consecutive edges in this path as walked.
            for i in range(len(path) - 1):
                walked_edges.add(
                    (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                )
            visited.update(path)
            if len(path) >= min_pixels:
                polylines.append([(x, y) for y, x in path])

    # ------------------------------------------------------------------
    # Cyclic components: closed loops with no endpoints or junctions.
    # Walk each remaining connected component as a single closed polygon.
    # ------------------------------------------------------------------
    remaining = pixel_set - visited
    while remaining:
        seed = next(iter(remaining))
        # BFS to collect the full connected component inside remaining.
        component: list[tuple[int, int]] = []
        queue = [seed]
        seen: set[tuple[int, int]] = {seed}
        while queue:
            p = queue.pop()
            component.append(p)
            for n in nbrs(*p):
                if n in remaining and n not in seen:
                    seen.add(n)
                    queue.append(n)
        remaining -= seen
        visited.update(component)

        if not component:
            continue

        # Walk the cycle: greedy DFS, close the loop at the end.
        comp_set = set(component)
        start_c = component[0]
        path_c: list[tuple[int, int]] = [start_c]
        path_set_c: set[tuple[int, int]] = {start_c}
        prev_c: tuple[int, int] | None = None
        curr_c = start_c

        while True:
            candidates = [
                n
                for n in nbrs(*curr_c)
                if n != prev_c and n in comp_set and n not in path_set_c
            ]
            if not candidates:
                # Close the cycle if start is reachable from current position.
                if start_c in nbrs(*curr_c) and len(path_c) > 2:
                    path_c.append(start_c)
                break
            prev_c, curr_c = curr_c, candidates[0]
            path_c.append(curr_c)
            path_set_c.add(curr_c)

        if len(path_c) >= min_pixels:
            polylines.append([(x, y) for y, x in path_c])

    return polylines
