"""LIC (Line Integral Convolution) generator — streamline-based flow visualisation.

Provides:
- ``_seed_grid()``         — generates a jittered regular grid of seed points
                             covering the canvas.
- ``_brightness_filter()`` — removes seeds in bright image regions using
                              bilinear brightness sampling.
- ``_trace_streamline()``  — traces a single streamline from a seed point in
                              both directions along a vector field.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Seed grid
# ---------------------------------------------------------------------------


def _seed_grid(
    canvas_w: float,
    canvas_h: float,
    seed_spacing_mm: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Generate a jittered regular grid of seed points over the canvas.

    The canvas is divided into cells of side ``seed_spacing_mm``.  One seed
    point is placed inside each cell at a uniformly random position within a
    jitter radius of ``seed_spacing_mm * 0.3`` around the cell centre.

    Parameters
    ----------
    canvas_w, canvas_h:
        Canvas dimensions in millimetres.
    seed_spacing_mm:
        Nominal distance between adjacent seed points (mm).  Controls grid
        resolution.  Smaller values produce denser streamlines.
    rng:
        NumPy random generator instance for reproducible results.

    Returns
    -------
    List of ``(x_mm, y_mm)`` tuples, one per grid cell, with positions
    clamped to ``[0, canvas_w] × [0, canvas_h]``.
    """
    seed_spacing_mm = max(1e-3, float(seed_spacing_mm))
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    nx = max(1, int(math.ceil(canvas_w / seed_spacing_mm)))
    ny = max(1, int(math.ceil(canvas_h / seed_spacing_mm)))

    jitter_radius = seed_spacing_mm * 0.3

    seeds: list[tuple[float, float]] = []
    for iy in range(ny):
        for ix in range(nx):
            # Cell centre in mm
            cx = (ix + 0.5) * canvas_w / nx
            cy = (iy + 0.5) * canvas_h / ny

            # Uniform jitter within a square of side 2*jitter_radius
            dx = float(rng.uniform(-jitter_radius, jitter_radius))
            dy = float(rng.uniform(-jitter_radius, jitter_radius))

            x = cx + dx
            y = cy + dy

            # Clamp to canvas bounds
            x = max(0.0, min(canvas_w, x))
            y = max(0.0, min(canvas_h, y))

            seeds.append((x, y))

    return seeds


# ---------------------------------------------------------------------------
# Brightness filter
# ---------------------------------------------------------------------------


def _brightness_filter(
    seeds: list[tuple[float, float]],
    canvas_w: float,
    canvas_h: float,
    brightness_img: np.ndarray,
    brightness_threshold: float,
) -> list[tuple[float, float]]:
    """Remove seed points that fall in bright image regions.

    Bilinearly samples ``brightness_img`` at each seed position (mapped from
    mm coordinates to pixel coordinates) and keeps only those seeds whose
    sampled brightness is *below* ``brightness_threshold``.

    Parameters
    ----------
    seeds:
        List of ``(x_mm, y_mm)`` points produced by :func:`_seed_grid`.
    canvas_w, canvas_h:
        Canvas dimensions in millimetres (must match the coordinate space of
        the seed positions).
    brightness_img:
        2-D grayscale array of shape ``(H, W)`` with float values in
        ``[0, 1]`` (0 = black, 1 = white).
    brightness_threshold:
        Seeds with sampled brightness ≥ this value are discarded.  A value of
        ``0.0`` removes all seeds (everything is at least as bright as 0),
        and ``1.0`` removes no seeds (nothing exceeds 1 in normalised images;
        passing ``> 1.0`` guarantees no seeds are removed even with slight
        floating-point drift, but the public interface uses ``255`` for 8-bit
        images — see notes below).

    Notes
    -----
    The caller is responsible for normalising ``brightness_img`` to ``[0, 1]``
    before calling this function.  To emulate a threshold expressed on a
    ``[0, 255]`` scale simply divide: ``brightness_threshold / 255``.

    Returns
    -------
    Filtered list of ``(x_mm, y_mm)`` tuples.
    """
    if len(seeds) == 0:
        return []

    img = np.asarray(brightness_img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(
            f"brightness_img must be 2-D (H, W), got shape {img.shape}"
        )

    h, w = img.shape
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    kept: list[tuple[float, float]] = []
    for x_mm, y_mm in seeds:
        brightness = _bilinear_sample(img, x_mm, y_mm, canvas_w, canvas_h, w, h)
        if brightness < brightness_threshold:
            kept.append((x_mm, y_mm))

    return kept


# ---------------------------------------------------------------------------
# Streamline tracer
# ---------------------------------------------------------------------------


def _trace_streamline(
    seed: tuple[float, float],
    vector_field: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    kernel_length_mm: float,
    step_size_mm: float,
) -> list[tuple[float, float]]:
    """Trace a single streamline from *seed* along *vector_field*.

    Integrates in both forward (+v) and backward (−v) directions using
    first-order Euler steps of length *step_size_mm*.  Each direction is
    capped at half of *kernel_length_mm* of arc length.

    Tracing stops early when any of the following conditions is met:

    * the current position exits ``[0, canvas_w] × [0, canvas_h]``
    * the bilinearly sampled vector magnitude falls below 0.01

    Parameters
    ----------
    seed:
        Starting point ``(x_mm, y_mm)`` in canvas millimetre coordinates.
    vector_field:
        NumPy array of shape ``(H, W, 2)`` containing unit direction vectors.
        ``field[row, col, 0]`` is the *x* component and
        ``field[row, col, 1]`` is the *y* component.  Matches the format
        returned by ``_compute_etf()`` in ``_helpers.py``.
    canvas_w, canvas_h:
        Canvas dimensions in millimetres.
    kernel_length_mm:
        Maximum total arc length of the streamline.  Each direction is
        traced for at most ``kernel_length_mm / 2`` mm.
    step_size_mm:
        Euler integration step size in millimetres.  Smaller values yield
        smoother curves but are more expensive to compute.

    Returns
    -------
    A single ``Polyline`` — a ``list[tuple[float, float]]`` of
    ``(x_mm, y_mm)`` points.  The seed point is always included (index 0 of
    the returned list after the backward segment is prepended).  If the seed
    is outside canvas bounds or the vector field is zero at the seed, a
    single-point polyline ``[seed]`` is returned.
    """
    step_size_mm = max(1e-4, float(step_size_mm))
    half_len = float(kernel_length_mm) / 2.0
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    vf = np.asarray(vector_field, dtype=np.float32)
    if vf.ndim != 3 or vf.shape[2] != 2:
        raise ValueError(
            f"vector_field must have shape (H, W, 2), got {vf.shape}"
        )
    vf_h, vf_w = vf.shape[:2]

    def _in_bounds(x: float, y: float) -> bool:
        return 0.0 <= x <= canvas_w and 0.0 <= y <= canvas_h

    def _sample_vf(x: float, y: float) -> tuple[float, float]:
        """Bilinearly sample *vector_field* at mm position (x, y)."""
        # Map mm → fractional pixel coordinates
        px = (x / canvas_w) * vf_w - 0.5
        py = (y / canvas_h) * vf_h - 0.5

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1

        x0c = max(0, min(vf_w - 1, x0))
        x1c = max(0, min(vf_w - 1, x1))
        y0c = max(0, min(vf_h - 1, y0))
        y1c = max(0, min(vf_h - 1, y1))

        fx = px - x0
        fy = py - y0

        vx = (
            float(vf[y0c, x0c, 0]) * (1 - fx) * (1 - fy)
            + float(vf[y0c, x1c, 0]) * fx * (1 - fy)
            + float(vf[y1c, x0c, 0]) * (1 - fx) * fy
            + float(vf[y1c, x1c, 0]) * fx * fy
        )
        vy = (
            float(vf[y0c, x0c, 1]) * (1 - fx) * (1 - fy)
            + float(vf[y0c, x1c, 1]) * fx * (1 - fy)
            + float(vf[y1c, x0c, 1]) * (1 - fx) * fy
            + float(vf[y1c, x1c, 1]) * fx * fy
        )
        return vx, vy

    def _trace_direction(
        start_x: float, start_y: float, sign: float
    ) -> list[tuple[float, float]]:
        """Trace one direction; *sign* is +1 (forward) or −1 (backward)."""
        pts: list[tuple[float, float]] = []
        x, y = start_x, start_y
        arc = 0.0

        while arc < half_len:
            vx, vy = _sample_vf(x, y)
            mag = math.hypot(vx, vy)
            if mag < 0.01:
                break

            # Normalise to unit step
            nx_v = vx / mag
            ny_v = vy / mag

            x += sign * nx_v * step_size_mm
            y += sign * ny_v * step_size_mm
            arc += step_size_mm

            if not _in_bounds(x, y):
                break

            pts.append((x, y))

        return pts

    sx, sy = float(seed[0]), float(seed[1])

    if not _in_bounds(sx, sy):
        return [(sx, sy)]

    forward = _trace_direction(sx, sy, +1.0)
    backward = _trace_direction(sx, sy, -1.0)

    # Combine: reversed-backward + seed + forward
    return list(reversed(backward)) + [(sx, sy)] + forward


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bilinear_sample(
    img: np.ndarray,
    x_mm: float,
    y_mm: float,
    canvas_w: float,
    canvas_h: float,
    img_w: int,
    img_h: int,
) -> float:
    """Bilinearly interpolate *img* at a canvas-mm coordinate.

    Maps ``(x_mm, y_mm)`` into pixel coordinates using the canvas dimensions
    and performs standard bilinear interpolation, clamping at image borders.

    Parameters
    ----------
    img:          Float32 array of shape ``(img_h, img_w)``.
    x_mm, y_mm:  Position in mm within ``[0, canvas_w] × [0, canvas_h]``.
    canvas_w/h:  Canvas size in mm.
    img_w/h:     Image dimensions in pixels.

    Returns
    -------
    Interpolated brightness in ``[0, 1]``.
    """
    # Map mm → fractional pixel coordinates (pixel centre = 0.5, 1.5, …)
    px = (x_mm / canvas_w) * img_w - 0.5
    py = (y_mm / canvas_h) * img_h - 0.5

    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp indices to valid range
    x0c = max(0, min(img_w - 1, x0))
    x1c = max(0, min(img_w - 1, x1))
    y0c = max(0, min(img_h - 1, y0))
    y1c = max(0, min(img_h - 1, y1))

    # Fractional parts for blending
    fx = px - x0
    fy = py - y0

    v00 = float(img[y0c, x0c])
    v10 = float(img[y0c, x1c])
    v01 = float(img[y1c, x0c])
    v11 = float(img[y1c, x1c])

    return (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )
