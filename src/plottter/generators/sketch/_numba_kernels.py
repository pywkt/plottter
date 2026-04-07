"""Numba-accelerated kernels for the Sketch generator.

Contains the JIT fallback decorator and the three numba-compiled functions
used during generation.
"""

from __future__ import annotations

import numpy as np

# Try to import Numba for JIT acceleration
try:
    from numba import njit as _numba_njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _numba_njit(func=None, **kwargs):  # type: ignore
        if func is not None:
            return func
        return lambda f: f
    _NUMBA_AVAILABLE = False


@_numba_njit(cache=True)
def _score_all_candidates(
    cur_x,
    cur_y,
    ends_x,
    ends_y,
    residual_dark,
    edge_normalized,
    coverage,
    max_pixel_coverage,
    h,
    w,
):
    """Score all candidate line endpoints using explicit for-loops (numba-compatible).

    Parameters mirror the arrays used in ``_find_darkest_line``.  The function
    is intentionally written with explicit ``for`` loops so that numba can
    compile it to fast native code.  Without numba the function runs as pure
    Python (slow), so ``_find_darkest_line`` falls back to the vectorized numpy
    path when ``_NUMBA_AVAILABLE`` is ``False``.

    Returns
    -------
    ``(scores, avg_brightnesses)`` arrays of length ``n_candidates``.
    ``avg_brightnesses[c]`` is the mean lightness (0–255) along the sampled
    points derived from ``residual_dark``: ``(1 - dark_mean) * 255``.
    """
    N_SAMPLES = 8
    n_candidates = ends_x.shape[0]
    denom_cov = 1.0 / (max_pixel_coverage if max_pixel_coverage > 0 else 1)

    scores = np.empty(n_candidates, dtype=np.float64)
    avg_brightnesses = np.empty(n_candidates, dtype=np.float64)

    for c in range(n_candidates):
        dark_sum = 0.0
        dark_peak = 0.0
        edge_sum = 0.0
        cov_sum = 0.0

        ex = int(ends_x[c])
        ey = int(ends_y[c])

        for s in range(N_SAMPLES):
            t = s / (N_SAMPLES - 1)
            px = int(round(cur_x + (ex - cur_x) * t))
            py = int(round(cur_y + (ey - cur_y) * t))
            if px < 0:
                px = 0
            elif px >= w:
                px = w - 1
            if py < 0:
                py = 0
            elif py >= h:
                py = h - 1

            dark_val = float(residual_dark[py, px])
            dark_sum += dark_val
            if dark_val > dark_peak:
                dark_peak = dark_val
            edge_sum += float(edge_normalized[py, px])
            cov_sum += float(coverage[py, px]) * denom_cov

        dark_mean = dark_sum / N_SAMPLES
        edge_mean = edge_sum / N_SAMPLES
        cov_mean = cov_sum / N_SAMPLES

        scores[c] = dark_mean * 1.45 + dark_peak * 0.45 + edge_mean * 0.24 - cov_mean * 1.08
        avg_brightnesses[c] = (1.0 - dark_mean) * 255.0

    return scores, avg_brightnesses


@_numba_njit(cache=True)
def _compute_weights(
    ds_dark: np.ndarray,
    ds_edge: np.ndarray,
    ds_cov: np.ndarray,
    dark_power: float,
    edge_bias: float,
    max_pixel_coverage: int,
    min_darkness: float,
) -> np.ndarray:
    """Compute per-pixel sampling weights from downsampled maps (numba-compatible).

    Written with explicit loops so numba can compile to native code.  Falls
    back to pure-Python when numba is unavailable (slow but correct).

    Parameters
    ----------
    ds_dark:
        Downsampled residual darkness map, float32 [0, 1].
    ds_edge:
        Downsampled edge magnitude map, float32 [0, 1].
    ds_cov:
        Downsampled coverage map, float32.
    dark_power:
        Exponent applied to the darkness term.
    edge_bias:
        Weight for the edge term in [0, 1].
    max_pixel_coverage:
        Maximum coverage count for ink penalty computation.
    min_darkness:
        Minimum darkness threshold; pixels below are effectively excluded.

    Returns
    -------
    weights: float64 array of shape ``(ds_h, ds_w)``.
    """
    ds_h = ds_dark.shape[0]
    ds_w = ds_dark.shape[1]
    denom_cov = 1.0 / (max_pixel_coverage if max_pixel_coverage > 0 else 1)
    weights = np.empty((ds_h, ds_w), dtype=np.float64)

    for i in range(ds_h):
        for j in range(ds_w):
            d = ds_dark[i, j] - min_darkness
            if d < 0.0:
                d = 0.0
            elif d > 1.0:
                d = 1.0
            edge_term = ds_edge[i, j] ** 1.2
            ink_val = 1.0 - ds_cov[i, j] * denom_cov
            if ink_val < 0.05:
                ink_val = 0.05
            elif ink_val > 1.0:
                ink_val = 1.0
            weights[i, j] = (d ** dark_power) * (0.85 + edge_bias * edge_term) * ink_val + 1e-9

    return weights


@_numba_njit(cache=True)
def _rasterize_path_numba(
    points_x,
    points_y,
    width,
    height,
    coverage_radius,
):
    """Rasterize a polyline to unique pixel coordinates, expanded by *coverage_radius*.

    Uses Bresenham-style interpolation (integer steps along the longer axis) for each
    consecutive pair of points, collects unique pixel coordinates, then expands by
    ``coverage_radius`` using nested offset loops over the square offset grid.

    Written with explicit ``for`` loops so numba can compile to fast native code.
    Falls back to pure-Python when numba is unavailable (slow but correct).

    Parameters
    ----------
    points_x, points_y:
        Float64 arrays of path point coordinates in pixel space.
    width, height:
        Image dimensions in pixels.
    coverage_radius:
        Pixel radius to expand around rasterized path pixels.

    Returns
    -------
    ``(xs, ys)`` arrays of int32 pixel coordinates within image bounds.
    """
    n_pts = points_x.shape[0]
    if n_pts < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # --- Step 1: Collect path pixels via Bresenham-style interpolation ---
    # Pre-estimate buffer size: sum of max(|dx|, |dy|) + 1 per segment
    path_est = 0
    for i in range(n_pts - 1):
        adx = abs(int(round(points_x[i + 1])) - int(round(points_x[i])))
        ady = abs(int(round(points_y[i + 1])) - int(round(points_y[i])))
        steps = adx if adx > ady else ady
        path_est += steps + 1

    path_buf = np.empty(path_est + n_pts, dtype=np.int32)
    path_sz = 0

    for i in range(n_pts - 1):
        x0 = int(round(points_x[i]))
        y0 = int(round(points_y[i]))
        x1 = int(round(points_x[i + 1]))
        y1 = int(round(points_y[i + 1]))

        adx = abs(x1 - x0)
        ady = abs(y1 - y0)
        steps = adx if adx > ady else ady

        for s in range(steps + 1):
            if steps > 0:
                t = float(s) / float(steps)
            else:
                t = 0.0
            px = int(round(float(x0) + float(x1 - x0) * t))
            py = int(round(float(y0) + float(y1 - y0) * t))
            if 0 <= px < width and 0 <= py < height:
                if path_sz < path_buf.shape[0]:
                    path_buf[path_sz] = py * width + px
                    path_sz += 1

    if path_sz == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Deduplicate path pixels via sort + unique iteration
    path_flat = path_buf[:path_sz].copy()
    path_flat.sort()

    # --- Step 2: Expand by coverage_radius ---
    if coverage_radius <= 0:
        # No expansion: extract unique path pixels directly
        unique_cnt = 1
        for i in range(1, path_sz):
            if path_flat[i] != path_flat[i - 1]:
                unique_cnt += 1
        xs_out = np.empty(unique_cnt, dtype=np.int32)
        ys_out = np.empty(unique_cnt, dtype=np.int32)
        xs_out[0] = path_flat[0] % width
        ys_out[0] = path_flat[0] // width
        out_idx = 0
        for i in range(1, path_sz):
            if path_flat[i] != path_flat[i - 1]:
                out_idx += 1
                xs_out[out_idx] = path_flat[i] % width
                ys_out[out_idx] = path_flat[i] // width
        return xs_out, ys_out

    # Count unique path pixels to size the expansion buffer
    unique_path_cnt = 1
    for i in range(1, path_sz):
        if path_flat[i] != path_flat[i - 1]:
            unique_path_cnt += 1

    er = 2 * coverage_radius + 1
    exp_est = unique_path_cnt * er * er + 10
    exp_buf = np.empty(exp_est, dtype=np.int32)
    exp_sz = 0

    # Expand each unique path pixel over the offset grid
    prev_fv = path_flat[0] - 1  # sentinel: guaranteed != path_flat[0]
    for i in range(path_sz):
        fv = path_flat[i]
        if fv == prev_fv:
            continue
        prev_fv = fv
        py_c = fv // width
        px_c = fv % width
        for dy_off in range(-coverage_radius, coverage_radius + 1):
            for dx_off in range(-coverage_radius, coverage_radius + 1):
                nx = px_c + dx_off
                ny = py_c + dy_off
                if 0 <= nx < width and 0 <= ny < height:
                    if exp_sz < exp_est:
                        exp_buf[exp_sz] = ny * width + nx
                        exp_sz += 1

    if exp_sz == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Deduplicate expanded pixels
    exp_flat = exp_buf[:exp_sz].copy()
    exp_flat.sort()

    unique_cnt = 1
    for i in range(1, exp_sz):
        if exp_flat[i] != exp_flat[i - 1]:
            unique_cnt += 1

    xs_out = np.empty(unique_cnt, dtype=np.int32)
    ys_out = np.empty(unique_cnt, dtype=np.int32)
    xs_out[0] = exp_flat[0] % width
    ys_out[0] = exp_flat[0] // width
    out_idx = 0
    for i in range(1, exp_sz):
        if exp_flat[i] != exp_flat[i - 1]:
            out_idx += 1
            xs_out[out_idx] = exp_flat[i] % width
            ys_out[out_idx] = exp_flat[i] // width

    return xs_out, ys_out
