"""TAM (Tone-level Aware Marking) generator utilities.

Provides `_build_tone_levels()` — a grid-jittered stroke-placement engine
that constructs N hierarchically nested stroke sets suitable for multi-tone
pen-plotter rendering.

The *nesting property*: every stroke that appears at tone level K also
appears at every darker level K+1, K+2, … N-1.  This means the darkest
level is a superset of all lighter levels, so the artwork looks correct at
any intermediate tone.

Orientation modes (resolved by the caller before calling `_build_tone_levels`):

- ``"fixed"``    — pass a scalar float (radians) as ``orientation_field``.
- ``"gradient"`` — pass a 2-D ndarray of angles (radians) derived from
                   the Sobel gradient of the source image.
- ``"etf"``      — pass the angle of the ETF tangent field, computed via
                   :func:`plottter.generators._helpers._compute_etf`.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

OrientationField = Union[float, np.ndarray]


def _build_tone_levels(
    canvas_w: float,
    canvas_h: float,
    num_levels: int,
    stroke_density: float,
    orientation_field: OrientationField,
    rng: np.random.Generator,
) -> list[list[tuple[float, float, float]]]:
    """Build *N* nested tone-level stroke sets using grid-jittered seeding.

    The canvas is divided into a regular grid whose cell count matches the
    desired stroke density for the *darkest* level.  One candidate stroke is
    placed per cell at a jittered position within that cell.  The full
    collection of candidates is randomly shuffled, then distributed across
    levels as a nested prefix sequence:

    * level 0  (lightest)  — smallest prefix of the shuffled list
    * level K              — ``round(total * (K+1) / N)`` strokes
    * level N-1 (darkest)  — all candidates

    Because each level is a prefix of the same list, the nesting property
    holds by construction.

    Parameters
    ----------
    canvas_w, canvas_h:
        Canvas dimensions in millimetres.
    num_levels:
        Number of tone levels to generate (≥ 1).
    stroke_density:
        Target strokes per mm² for the darkest level.  Controls grid
        resolution.  Typical range: 0.01–2.0.
    orientation_field:
        Either a scalar ``float`` (constant angle in radians for every
        stroke — "fixed" mode) or a 2-D :class:`numpy.ndarray` of shape
        ``(H, W)`` containing pre-computed angle values in radians that
        are bilinearly interpolated at each stroke position.
    rng:
        NumPy random generator instance, used for jitter and shuffling so
        results are reproducible when seeded.

    Returns
    -------
    List of length ``num_levels``.  Each element is a list of
    ``(x_mm, y_mm, angle_rad)`` tuples.  Level 0 has the fewest strokes
    (lightest tone), level ``num_levels - 1`` has the most (darkest tone).
    All positions satisfy ``0 ≤ x_mm ≤ canvas_w`` and
    ``0 ≤ y_mm ≤ canvas_h``.
    """
    num_levels = max(1, int(num_levels))
    stroke_density = max(1e-9, float(stroke_density))
    canvas_w = float(canvas_w)
    canvas_h = float(canvas_h)

    # ------------------------------------------------------------------
    # 1. Determine grid dimensions
    # ------------------------------------------------------------------
    # Total candidate count for the darkest (finest) level.
    # Ensure at least one stroke per level so that every level is non-empty.
    total_strokes = max(num_levels, int(math.ceil(canvas_w * canvas_h * stroke_density)))

    # Square cell side length derived from target density
    cell_size = math.sqrt(canvas_w * canvas_h / total_strokes)
    cell_size = max(cell_size, 1e-3)

    nx = max(1, int(math.ceil(canvas_w / cell_size)))
    ny = max(1, int(math.ceil(canvas_h / cell_size)))

    # ------------------------------------------------------------------
    # 2. Grid-jittered candidate placement
    # ------------------------------------------------------------------
    candidates: list[tuple[float, float, float]] = []

    for iy in range(ny):
        for ix in range(nx):
            # Cell boundaries in mm
            x0 = ix * canvas_w / nx
            x1 = (ix + 1) * canvas_w / nx
            y0 = iy * canvas_h / ny
            y1 = (iy + 1) * canvas_h / ny

            # Jittered position uniformly distributed within the cell
            x = float(rng.uniform(x0, x1))
            y = float(rng.uniform(y0, y1))

            # Clamp to canvas bounds (in case of floating-point drift)
            x = max(0.0, min(canvas_w, x))
            y = max(0.0, min(canvas_h, y))

            angle = _sample_orientation(x, y, canvas_w, canvas_h, orientation_field)
            candidates.append((x, y, angle))

    # ------------------------------------------------------------------
    # 3. Shuffle candidates for random level assignment
    # ------------------------------------------------------------------
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    n_total = len(candidates)  # ≥ num_levels (enforced above)

    # ------------------------------------------------------------------
    # 4. Build nested levels as prefix slices
    # ------------------------------------------------------------------
    # Count for level K = round(n_total * (K+1) / num_levels), clamped so
    # that counts are strictly increasing and the last level gets everything.
    levels: list[list[tuple[float, float, float]]] = []
    prev_count = 0
    for k in range(num_levels):
        if k == num_levels - 1:
            count = n_total
        else:
            count = max(prev_count + 1, round(n_total * (k + 1) / num_levels))
            count = min(count, n_total - (num_levels - 1 - k))
        prev_count = count
        levels.append(list(candidates[:count]))

    return levels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_orientation(
    x_mm: float,
    y_mm: float,
    canvas_w: float,
    canvas_h: float,
    orientation_field: OrientationField,
) -> float:
    """Return the stroke orientation angle (radians) at a canvas position.

    Parameters
    ----------
    x_mm, y_mm:
        Position in canvas mm coordinates.
    canvas_w, canvas_h:
        Canvas dimensions in mm (used to normalise array lookups).
    orientation_field:
        Scalar float for a uniform angle, or a 2-D ndarray of angles.

    Returns
    -------
    Angle in radians.
    """
    if isinstance(orientation_field, (int, float, np.floating)):
        return float(orientation_field)

    arr = np.asarray(orientation_field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"orientation_field array must be 2-D, got shape {arr.shape}"
        )

    h, w = arr.shape

    # Map mm coordinates to fractional pixel coordinates
    px = (x_mm / canvas_w) * (w - 1) if canvas_w > 0 else 0.0
    py = (y_mm / canvas_h) * (h - 1) if canvas_h > 0 else 0.0

    # Clamp to valid array range
    px = max(0.0, min(w - 1.0, px))
    py = max(0.0, min(h - 1.0, py))

    ix = int(px)
    iy = int(py)
    ix1 = min(ix + 1, w - 1)
    iy1 = min(iy + 1, h - 1)
    fx = px - ix
    fy = py - iy

    # Bilinear interpolation
    # Note: angles are assumed to be in a smooth field without wrapping
    # discontinuities in the vicinity of any given stroke (valid for ETF
    # tangent fields and Sobel gradient fields away from singularities).
    v00 = float(arr[iy,  ix ])
    v10 = float(arr[iy,  ix1])
    v01 = float(arr[iy1, ix ])
    v11 = float(arr[iy1, ix1])

    return (v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy)
