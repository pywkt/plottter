"""TAM (Tone-level Aware Marking) generator utilities.

Provides `_build_tone_levels()` — a grid-jittered stroke-placement engine
that constructs N hierarchically nested stroke sets suitable for multi-tone
pen-plotter rendering.

Provides `_render_strokes()` — converts stroke tuples into polylines
suitable for the plotter path model (straight or curved streamlines).

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


def _render_strokes(
    strokes: list[tuple[float, float, float]],
    stroke_length_mm: float,
    orientation_field: OrientationField,
    canvas_w: float,
    canvas_h: float,
    curvature: float = 0.0,
    n_samples: int = 7,
) -> list[list[tuple[float, float]]]:
    """Convert stroke tuples into polylines.

    Each stroke ``(x_mm, y_mm, angle_rad)`` becomes a polyline of length
    approximately ``stroke_length_mm``, centered on ``(x_mm, y_mm)`` and
    initially oriented at ``angle_rad``.

    Parameters
    ----------
    strokes:
        List of ``(x_mm, y_mm, angle_rad)`` tuples, e.g. from
        :func:`_build_tone_levels`.
    stroke_length_mm:
        Target length of each stroke in millimetres.
    orientation_field:
        Scalar float or 2-D angle array used for field sampling when
        ``curvature > 0``.  Same semantics as :func:`_build_tone_levels`.
    canvas_w, canvas_h:
        Canvas dimensions in mm, required for array field lookups.
    curvature:
        Blend factor in ``[0.0, 1.0]``.

        * ``0.0`` — straight 2-point segment; the field is never sampled.
        * ``1.0`` — fully curved streamline; every step follows the local
          orientation field.
        * Intermediate values linearly blend the initial stroke angle with
          the sampled field angle at each integration step.
    n_samples:
        Number of points in curved polylines (``curvature > 0``).
        Clamped to ``≥ 2``.  Ignored when ``curvature == 0``.

    Returns
    -------
    List of polylines, one per input stroke.  Each polyline is a
    ``list[tuple[float, float]]`` with at least 2 points.  The arc length
    of each polyline equals ``stroke_length_mm`` exactly (straight) or
    within floating-point precision (curved, since step sizes are fixed).
    """
    curvature = max(0.0, min(1.0, float(curvature)))
    stroke_length_mm = max(0.0, float(stroke_length_mm))
    n_samples = max(2, int(n_samples))

    polylines: list[list[tuple[float, float]]] = []

    for x, y, angle in strokes:
        half = stroke_length_mm / 2.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        if curvature <= 0.0:
            # Straight 2-point segment centered at (x, y).
            start = (x - cos_a * half, y - sin_a * half)
            end = (x + cos_a * half, y + sin_a * half)
            polylines.append([start, end])
        else:
            # Curved streamline traced by Euler integration.
            # Start at one end of the nominal straight stroke and walk
            # forward, sampling the orientation field at each step.
            step = stroke_length_mm / (n_samples - 1)
            cx = x - cos_a * half
            cy = y - sin_a * half
            points: list[tuple[float, float]] = [(cx, cy)]

            for _ in range(n_samples - 1):
                field_angle = _sample_orientation(
                    cx, cy, canvas_w, canvas_h, orientation_field
                )
                # Linear blend: curvature=0 → initial angle, curvature=1 → field
                eff_angle = angle * (1.0 - curvature) + field_angle * curvature
                cx = cx + math.cos(eff_angle) * step
                cy = cy + math.sin(eff_angle) * step
                points.append((cx, cy))

            polylines.append(points)

    return polylines


def _select_strokes_for_image(
    tone_levels: list[list[tuple[float, float, float]]],
    image: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    density_curve: str = "linear",
) -> list[tuple[float, float, float]]:
    """Select which strokes to draw based on local image brightness.

    For each stroke position, the local image brightness (0.0 = black,
    1.0 = white) is mapped to a *tone index* in ``[0, N-1]`` where N is the
    number of tone levels.  A stroke that first appears at tone level K is
    included in the output if the mapped tone index ≥ K.

    Because tone levels are nested (level K is a subset of level K+1), the
    full darkest level is used as the master list and each stroke's *birth
    level* is determined as the smallest K for which it appears.

    Parameters
    ----------
    tone_levels:
        Nested stroke sets from :func:`_build_tone_levels`.  Level 0 is
        lightest (fewest strokes); level ``N-1`` is darkest (all strokes).
    image:
        Grayscale image as a 2-D float array with values in ``[0.0, 1.0]``.
        0.0 = black (maximum hatching), 1.0 = white (no hatching).
        May also be a 3-channel (H, W, 3) or 4-channel (H, W, 4) uint8
        array; in that case the luminance is computed automatically.
    canvas_w, canvas_h:
        Canvas dimensions in mm, used to map stroke positions to pixel
        coordinates.
    density_curve:
        Controls the non-linear mapping from brightness to tone index.

        * ``"linear"``      — tone index = darkness * N  (darkness = 1 - brightness)
        * ``"quadratic"``   — tone index = sqrt(darkness) * N; concave curve
          that produces more strokes than linear at mid-gray brightness.
        * ``"logarithmic"`` — tone index = log1p(darkness*(e-1)) * N;
          similar concave shape, slightly gentler than quadratic.

    Returns
    -------
    List of ``(x_mm, y_mm, angle_rad)`` tuples — the strokes that should be
    rendered for the given image.  Pure-white regions produce no strokes;
    pure-black regions produce all strokes from the darkest tone level.
    """
    if len(tone_levels) == 0:
        return []

    num_levels = len(tone_levels)

    # ------------------------------------------------------------------
    # 1. Normalise image to float32 luminance in [0, 1]
    # ------------------------------------------------------------------
    img = np.asarray(image)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float64) / 255.0

    if img.ndim == 3:
        # Take luminance: weighted RGB or simple mean
        if img.shape[2] >= 3:
            img = (
                0.2126 * img[:, :, 0]
                + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]
            )
        else:
            img = img[:, :, 0]

    img = np.asarray(img, dtype=np.float64)
    img = np.clip(img, 0.0, 1.0)

    h_px, w_px = img.shape

    # ------------------------------------------------------------------
    # 2. Determine the *birth level* of every stroke
    # ------------------------------------------------------------------
    # The darkest level contains all strokes; lighter levels are subsets.
    # Birth level K = smallest level index at which the stroke appears.
    # We build a mapping stroke_id → birth_level.
    all_strokes = tone_levels[num_levels - 1]  # all strokes (darkest level)
    n_strokes = len(all_strokes)

    # Build set membership: which strokes are in each level
    # Use tuple identity via position in the all_strokes list.
    # Because tone_levels are built as prefix slices, level K contains the
    # first len(tone_levels[K]) strokes from all_strokes.
    birth_level = np.zeros(n_strokes, dtype=np.int32)
    level_sizes = [len(lvl) for lvl in tone_levels]

    # Birth level for stroke i = smallest K such that i < level_sizes[K]
    # Since level_sizes is strictly increasing, we can vectorise this.
    sizes = np.array(level_sizes, dtype=np.int32)  # shape (N,)
    for i in range(n_strokes):
        # First level whose size exceeds i
        bl = np.searchsorted(sizes, i + 1, side="left")
        birth_level[i] = int(bl)

    # ------------------------------------------------------------------
    # 3. Vectorised brightness sampling
    # ------------------------------------------------------------------
    xs = np.array([s[0] for s in all_strokes], dtype=np.float64)
    ys = np.array([s[1] for s in all_strokes], dtype=np.float64)

    if w_px > 0 and canvas_w > 0:
        px_coords = np.clip((xs / canvas_w) * (w_px - 1), 0.0, w_px - 1.0)
    else:
        px_coords = np.zeros(n_strokes)
    if h_px > 0 and canvas_h > 0:
        py_coords = np.clip((ys / canvas_h) * (h_px - 1), 0.0, h_px - 1.0)
    else:
        py_coords = np.zeros(n_strokes)

    ix = np.round(px_coords).astype(np.int32)
    iy = np.round(py_coords).astype(np.int32)
    brightness = img[iy, ix]  # shape (n_strokes,)

    # ------------------------------------------------------------------
    # 4. Apply density curve: darkness → tone_index in [0, N]
    #
    # Mapping uses the open interval (0, N] so that:
    #   brightness = 1.0  →  darkness = 0.0  →  tone_index = 0.0  → no strokes
    #   brightness = 0.0  →  darkness = 1.0  →  tone_index = N    → all strokes
    #
    # Stroke born at level K is included when tone_index > K  (strict).
    #
    # Curve definitions (darkness ∈ [0, 1]):
    #   "linear"      — tone_index = darkness * N
    #   "quadratic"   — tone_index = sqrt(darkness) * N  (concave, > linear for mid)
    #                   Emphasises mid-tones: a small amount of darkness
    #                   quickly produces many strokes.
    #   "logarithmic" — tone_index = log1p(darkness*(e-1))/1 * N
    #                   Similar concave shape, slightly gentler than sqrt.
    # ------------------------------------------------------------------
    darkness = 1.0 - brightness  # 0.0 = white, 1.0 = black
    n_full = float(num_levels)

    curve = density_curve.lower().strip()
    if curve == "quadratic":
        # sqrt is a "quadratic" root — concave up, more strokes than linear
        tone_index = np.sqrt(darkness) * n_full
    elif curve == "logarithmic":
        import math as _math
        scale = _math.log1p(_math.e - 1.0)  # = 1.0 exactly (log1p(e-1) = 1)
        tone_index = np.log1p(darkness * (_math.e - 1.0)) / scale * n_full
    else:
        # "linear" (default)
        tone_index = darkness * n_full

    # ------------------------------------------------------------------
    # 5. Include stroke i if tone_index[i] > birth_level[i]  (strict)
    #
    # Strict inequality ensures brightness=1.0 (tone_index=0) excludes
    # all strokes, including those born at the lightest level 0.
    # ------------------------------------------------------------------
    mask = tone_index > birth_level.astype(np.float64)
    return [all_strokes[i] for i in range(n_strokes) if mask[i]]


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
