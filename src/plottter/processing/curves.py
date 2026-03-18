"""Bezier curve fitting for post-processing plotter polylines.

Replaces jagged pixel-boundary contours with smooth cubic Bezier curves,
significantly reducing point count while staying within a user-specified
tolerance of the original polylines.

Algorithm (Schneider-inspired):
    1. Detect corner points using the tangent-angle change.
    2. Fit cubic Bezier curves between consecutive corners using least squares
       (chord-length parameterisation + Schneider linear system).
    3. Resample the fitted curves at the specified chord tolerance.

If the fitting error exceeds the tolerance the segment is recursively split
at the point of maximum error and re-fitted (Schneider's divide-and-conquer
strategy).

Note on pypotrace
-----------------
The ``pypotrace`` library (C Potrace wrapper) operates on binary bitmaps
and produces Bezier paths directly from raster data.  It cannot be used as
a post-processing step on polylines (which live in mm-space and have no
corresponding bitmap at call-time).  This module therefore always uses the
pure-Python algorithm.  If bitmap-level tracing is desired in the future,
it should be wired into the generator pipeline *before* contour extraction,
not as a polyline post-processor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plottter.models import Polyline

# Maximum recursion depth for the fit-and-split loop
_MAX_DEPTH = 8
# Angle change (degrees) that marks a corner; below this the curve is smooth
_CORNER_DEG = 60.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_curves(
    polylines: "list[Polyline]",
    tolerance_mm: float = 0.5,
) -> "list[Polyline]":
    """Fit smooth cubic Bezier curves to raw plotter polylines.

    Replaces jagged pixel-boundary points (from contour tracing) with smooth
    curves that significantly reduce point count while keeping the output
    within *tolerance_mm* of the original.

    Parameters
    ----------
    polylines:
        Input polylines.  Closed polylines (first point == last point) are
        handled correctly — the output polyline is also closed.
    tolerance_mm:
        Maximum allowed deviation from original points, in mm.
        Lower values → more faithful to the input, more output points.
        Higher values → smoother output, fewer points, may deviate more.

    Returns
    -------
    List of smoothed polylines.  Each output polyline has at least 2 points
    and the same open/closed topology as the corresponding input.
    """
    if tolerance_mm <= 0.0 or not polylines:
        return polylines

    out: list[list[tuple[float, float]]] = []
    for poly in polylines:
        fitted = _fit_polyline(poly, tolerance_mm)
        if fitted and len(fitted) >= 2:
            out.append(fitted)
    return out


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------


def _fit_polyline(
    poly: "Polyline",
    tol: float,
) -> "list[tuple[float, float]]":
    """Fit Bezier curves to a single polyline and return resampled points."""
    n = len(poly)
    if n < 3:
        return list(poly)

    pts = np.array(poly, dtype=np.float64)  # (N, 2)

    # Detect closed polyline (first ≈ last point)
    closed = bool(np.allclose(pts[0], pts[-1], atol=1e-9))
    # For closed curves, work without the duplicate closing point
    work_pts = pts[:-1] if closed else pts
    m = len(work_pts)
    if m < 2:
        return list(poly)

    # Find corners (high-curvature points) in the working sequence
    corners = _find_corners(work_pts, _CORNER_DEG, wrap=closed)

    # Segment boundaries: always include 0 and m-1, plus detected corners
    boundaries = sorted({0, m - 1} | corners)

    result_pts: list[tuple[float, float]] = []

    for seg_idx in range(len(boundaries) - 1):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        seg = work_pts[start : end + 1]  # inclusive slice

        if len(seg) < 2:
            continue

        # Estimate unit tangents at segment endpoints
        d_start = _tangent_at(work_pts, start, forward=True, wrap=closed)
        d_end = _tangent_at(work_pts, end, forward=False, wrap=closed)

        sampled = _fit_recursive(seg, d_start, d_end, tol, depth=0)
        if not sampled:
            sampled = [_pt(seg[0]), _pt(seg[-1])]

        if not result_pts:
            result_pts.extend(sampled)
        else:
            # Avoid duplicating the shared boundary point
            result_pts.extend(sampled[1:])

    if not result_pts:
        return list(poly)

    # Re-close if the input was closed
    if closed and len(result_pts) >= 2 and result_pts[0] != result_pts[-1]:
        result_pts.append(result_pts[0])

    return result_pts


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------


def _find_corners(
    pts: np.ndarray,
    thresh_deg: float,
    wrap: bool,
) -> set[int]:
    """Return indices of interior corner points (high curvature)."""
    cos_thresh = math.cos(math.radians(thresh_deg))
    n = len(pts)
    corners: set[int] = set()

    for i in range(n):
        prev_i = (i - 1) % n if wrap else (i - 1)
        next_i = (i + 1) % n if wrap else (i + 1)

        # Skip endpoints of open curves
        if not wrap and (prev_i < 0 or next_i >= n):
            continue

        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        len1 = _norm(v1)
        len2 = _norm(v2)

        if len1 < 1e-10 or len2 < 1e-10:
            # Zero-length segment → treat as corner
            corners.add(i)
            continue

        cos_a = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
        if cos_a < cos_thresh:
            corners.add(i)

    return corners


# ---------------------------------------------------------------------------
# Tangent estimation
# ---------------------------------------------------------------------------


def _tangent_at(
    pts: np.ndarray,
    idx: int,
    forward: bool,
    wrap: bool,
) -> np.ndarray:
    """Estimate unit tangent at pts[idx] in the forward or backward direction."""
    n = len(pts)

    if forward:
        j = (idx + 1) % n if wrap else min(idx + 1, n - 1)
        v = pts[j] - pts[idx]
    else:
        j = (idx - 1) % n if wrap else max(idx - 1, 0)
        v = pts[idx] - pts[j]

    length = _norm(v)
    # If the immediate neighbor is too close, try further neighbors
    if length < 1e-10:
        for delta in range(2, min(6, n)):
            if forward:
                j = (idx + delta) % n if wrap else min(idx + delta, n - 1)
                v = pts[j] - pts[idx]
            else:
                j = (idx - delta) % n if wrap else max(idx - delta, 0)
                v = pts[idx] - pts[j]
            length = _norm(v)
            if length >= 1e-10:
                break

    if length < 1e-10:
        return np.array([1.0, 0.0])
    return v / length


# ---------------------------------------------------------------------------
# Recursive Bezier fitting
# ---------------------------------------------------------------------------


def _fit_recursive(
    seg: np.ndarray,
    d_start: np.ndarray,
    d_end: np.ndarray,
    tol: float,
    depth: int,
) -> list[tuple[float, float]]:
    """Fit a cubic Bezier to seg; if error > tol, split and recurse."""
    n = len(seg)

    if n <= 2 or depth >= _MAX_DEPTH:
        return [_pt(seg[0]), _pt(seg[-1])]

    ctrl = _fit_cubic(seg, d_start, d_end)
    if ctrl is None:
        # Degenerate — fall back to keeping all original points
        return [_pt(p) for p in seg]

    # Compute max fitting error
    t_vals = _chord_params(seg)
    max_err = 0.0
    split_i = len(seg) // 2

    for i in range(n):
        err = float(np.linalg.norm(_bezier_eval(*ctrl, float(t_vals[i])) - seg[i]))
        if err > max_err:
            max_err = err
            split_i = i

    if max_err <= tol:
        # Fit is good enough — resample at tolerance spacing
        return _sample_bezier(*ctrl, tol)

    # Split at the worst-fitting point and recurse
    split_i = max(1, min(split_i, n - 2))
    d_mid = _tangent_mid(seg, split_i)

    left = _fit_recursive(seg[: split_i + 1], d_start, d_mid, tol, depth + 1)
    right = _fit_recursive(seg[split_i:], d_mid, d_end, tol, depth + 1)

    # Merge: skip the duplicated split point at the boundary
    return left + right[1:]


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------


def _chord_params(pts: np.ndarray) -> np.ndarray:
    """Chord-length parameterization normalized to [0, 1]."""
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    total = t[-1]
    if total > 0.0:
        t /= total
    return t


# ---------------------------------------------------------------------------
# Cubic Bezier fitting (Schneider least-squares)
# ---------------------------------------------------------------------------


def _fit_cubic(
    pts: np.ndarray,
    d_start: np.ndarray,
    d_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit cubic Bezier P(t) = Σ Bᵢ(t)·Pᵢ with endpoint tangent constraints.

    Sets P1 = P0 + α1·d_start and P2 = P3 − α2·d_end, then solves for
    α1, α2 via the Schneider least-squares linear system.

    Returns (P0, P1, P2, P3) or None for degenerate input.
    """
    if len(pts) < 2:
        return None

    P0 = pts[0].copy()
    P3 = pts[-1].copy()
    chord = float(np.linalg.norm(P3 - P0))

    if chord < 1e-10:
        return None

    t = _chord_params(pts)
    n = len(pts)

    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i in range(n):
        ti = float(t[i])
        b0 = (1.0 - ti) ** 3
        b1 = 3.0 * ti * (1.0 - ti) ** 2
        b2 = 3.0 * ti ** 2 * (1.0 - ti)
        b3 = ti ** 3

        # A[i][0] = B1(t) * d_start,  A[i][1] = -B2(t) * d_end
        A0 = b1 * d_start
        A1 = -b2 * d_end

        C[0][0] += float(np.dot(A0, A0))
        C[0][1] += float(np.dot(A0, A1))
        C[1][1] += float(np.dot(A1, A1))

        # rhs[i] = pts[i] - (B0+B1)*P0 - (B2+B3)*P3
        rhs = pts[i] - (b0 + b1) * P0 - (b2 + b3) * P3
        X[0] += float(np.dot(rhs, A0))
        X[1] += float(np.dot(rhs, A1))

    C[1][0] = C[0][1]
    det = C[0][0] * C[1][1] - C[0][1] * C[1][0]

    fallback = chord / 3.0

    if abs(det) < 1e-12:
        # Degenerate matrix: use a chord-length heuristic
        P1 = P0 + fallback * d_start
        P2 = P3 - fallback * d_end
        return (P0, P1, P2, P3)

    alpha1 = (X[0] * C[1][1] - C[0][1] * X[1]) / det
    alpha2 = (C[0][0] * X[1] - C[1][0] * X[0]) / det

    # Schneider's sanity check: alphas must be positive (control points
    # must lie "ahead" of the endpoints along the tangent direction)
    if alpha1 <= 0.0:
        alpha1 = fallback
    if alpha2 <= 0.0:
        alpha2 = fallback

    P1 = P0 + alpha1 * d_start
    P2 = P3 - alpha2 * d_end
    return (P0, P1, P2, P3)


# ---------------------------------------------------------------------------
# Bezier evaluation and sampling
# ---------------------------------------------------------------------------


def _bezier_eval(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a cubic Bezier curve at parameter t ∈ [0, 1]."""
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * t * (1.0 - t) ** 2
    b2 = 3.0 * t ** 2 * (1.0 - t)
    b3 = t ** 3
    return b0 * P0 + b1 * P1 + b2 * P2 + b3 * P3


def _sample_bezier(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    tolerance: float,
) -> list[tuple[float, float]]:
    """Sample a cubic Bezier curve at approximately *tolerance* spacing.

    Uses a dense uniform sample to approximate arc-length parameterisation,
    then walks along the dense samples keeping only points spaced by at least
    ``tolerance * 0.9`` (the 90% factor avoids a very tiny last segment).
    """
    # Approximate curve length via the control polygon and direct chord
    L_ctrl = (
        float(np.linalg.norm(P1 - P0))
        + float(np.linalg.norm(P2 - P1))
        + float(np.linalg.norm(P3 - P2))
    )
    L_chord = float(np.linalg.norm(P3 - P0))
    approx_len = (L_ctrl + L_chord) / 2.0

    # Dense sample count — at least 4, proportional to length / tolerance
    n_dense = max(4, int(approx_len / (tolerance * 0.5)) + 2)
    ts = np.linspace(0.0, 1.0, n_dense)
    dense = [_bezier_eval(P0, P1, P2, P3, float(t)) for t in ts]

    # Walk along dense points, keeping those at chord intervals ≥ tolerance
    out = [_pt(dense[0])]
    accum = 0.0
    thresh = tolerance * 0.9  # slight slack avoids tiny trailing segment

    for i in range(1, len(dense)):
        d = float(np.linalg.norm(dense[i] - dense[i - 1]))
        accum += d
        if accum >= thresh:
            out.append(_pt(dense[i]))
            accum = 0.0

    # Always include the endpoint
    last = _pt(dense[-1])
    if out[-1] != last:
        out.append(last)

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tangent_mid(seg: np.ndarray, idx: int) -> np.ndarray:
    """Estimate unit tangent at an interior split point (finite difference)."""
    n = len(seg)
    prev_i = max(0, idx - 1)
    next_i = min(n - 1, idx + 1)
    v = seg[next_i] - seg[prev_i]
    length = _norm(v)
    if length < 1e-10:
        return np.array([1.0, 0.0])
    return v / length


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _pt(arr: np.ndarray) -> tuple[float, float]:
    return (float(arr[0]), float(arr[1]))
