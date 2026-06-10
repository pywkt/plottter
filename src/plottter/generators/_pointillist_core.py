"""Pure-numeric helpers for the Pointillist generator.

No Qt imports.  No plottter.gui imports.  All functions are deterministic
for fixed inputs and operate entirely in numpy / pure Python.

Type aliases mirror the project-wide conventions:
    Point    = tuple[float, float]
    Polyline = list[Point]
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases (for documentation only — no runtime enforcement)
# ---------------------------------------------------------------------------
Point = tuple[float, float]
Polyline = list[Point]


# ---------------------------------------------------------------------------
# §5  Mitchell's best-candidate sampler
# ---------------------------------------------------------------------------

def mitchell_sample(
    mask: np.ndarray,
    n: int,
    seed: int,
    candidates: int = 10,
) -> np.ndarray:
    """Return up to *n* image-space (row, col) coordinates inside *mask == 255*.

    Uses Mitchell's best-candidate algorithm: for each dot to place, draw
    *candidates* random candidate pixels from the mask's white region and keep
    the one that is farthest from all already-placed dots.

    Parameters
    ----------
    mask:
        2-D ``uint8`` array; only pixels equal to 255 are candidates.
    n:
        Number of dots to place.  Fewer dots are returned if the mask has
        fewer than *n* white pixels.
    seed:
        Integer seed for the NumPy RNG — guarantees identical output for
        identical ``(mask, n, seed)`` inputs.
    candidates:
        Number of candidate pixels to evaluate per step (default 10).

    Returns
    -------
    np.ndarray
        Shape ``(k, 2)`` int64 array of ``(row, col)`` coordinates where
        ``k = min(n, number of white pixels in mask)``.  Returns a zero-row
        array with shape ``(0, 2)`` when the mask is entirely black.
    """
    # Collect all eligible pixels.
    rows, cols = np.where(mask == 255)
    m = len(rows)
    if m == 0 or n == 0:
        return np.empty((0, 2), dtype=np.int64)

    k = min(n, m)
    rng = np.random.default_rng(seed)

    # Pre-draw all candidate indices in one shot for speed.
    # Shape: (k, candidates)
    all_idxs = rng.integers(0, m, size=(k, candidates))

    result = np.empty((k, 2), dtype=np.int64)

    # First dot: just take the first candidate of the first step.
    result[0, 0] = rows[all_idxs[0, 0]]
    result[0, 1] = cols[all_idxs[0, 0]]

    for i in range(1, k):
        cand_rows = rows[all_idxs[i]]   # shape (candidates,)
        cand_cols = cols[all_idxs[i]]   # shape (candidates,)

        # Compute squared distance from each candidate to every placed dot.
        # placed: result[:i] shape (i, 2)
        placed_r = result[:i, 0]  # shape (i,)
        placed_c = result[:i, 1]  # shape (i,)

        # Broadcast: (candidates, i) - squared Euclidean distance
        dr = cand_rows[:, np.newaxis] - placed_r[np.newaxis, :]  # (cands, i)
        dc = cand_cols[:, np.newaxis] - placed_c[np.newaxis, :]  # (cands, i)
        sq_dist = dr * dr + dc * dc  # (cands, i)

        # Min distance to any placed dot for each candidate.
        min_sq = sq_dist.min(axis=1)  # (cands,)

        best_idx = int(np.argmax(min_sq))
        result[i, 0] = cand_rows[best_idx]
        result[i, 1] = cand_cols[best_idx]

    return result


# ---------------------------------------------------------------------------
# §5  Image → canvas mm coordinate mapping
# ---------------------------------------------------------------------------

def image_to_canvas_mm(
    rc: np.ndarray,
    image_shape: tuple[int, int],
    drawing_area: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert image-space (row, col) integer coordinates to canvas mm floats.

    Parameters
    ----------
    rc:
        ``(n, 2)`` int array of ``(row, col)`` image-space coordinates as
        returned by :func:`mitchell_sample`.
    image_shape:
        ``(H, W)`` of the source image.
    drawing_area:
        ``(left, top, right, bottom)`` in mm — as returned by
        ``Canvas.drawing_area()``.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` float64 array of ``(x_mm, y_mm)`` canvas coordinates.
    """
    if rc.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    H, W = image_shape
    left, top, right, bottom = drawing_area
    width = right - left
    height = bottom - top

    rows = rc[:, 0].astype(np.float64)
    cols = rc[:, 1].astype(np.float64)

    x_mm = left + (cols / W) * width
    y_mm = top + (rows / H) * height

    out = np.empty((len(rc), 2), dtype=np.float64)
    out[:, 0] = x_mm
    out[:, 1] = y_mm
    return out


# ---------------------------------------------------------------------------
# §6  Dot rendering
# ---------------------------------------------------------------------------

# Number of circle vertices (12 per spec, plus closing duplicate = 13 pts).
_CIRCLE_SIDES = 12

# Angular resolution of the disc-fill spiral (points per full turn).
_SPIRAL_PTS_PER_TURN = 24
# Default radial gap between successive spiral arms, in mm — roughly a
# fineliner's stroke width, so the disc reads as solid.
_DEFAULT_FILL_SPACING_MM = 0.35


def render_dots(
    coords_mm: np.ndarray,
    style: str,
    size_mm: float,
    fill_spacing_mm: float = _DEFAULT_FILL_SPACING_MM,
) -> list[Polyline]:
    """Turn an array of dot centres into plotter-ready polylines.

    Parameters
    ----------
    coords_mm:
        ``(n, 2)`` float array of ``(x_mm, y_mm)`` dot centres.
    style:
        One of ``"point"``, ``"cross"``, ``"circle"``, ``"disc"``.
    size_mm:
        Dot size in mm.  Used by ``"cross"`` (arm half-length = size_mm*0.5),
        ``"circle"`` and ``"disc"`` (radius = size_mm*0.5).  Ignored by
        ``"point"``.
    fill_spacing_mm:
        Radial gap between spiral arms for the ``"disc"`` style (smaller =
        denser fill).  Ignored by the other styles.

    Returns
    -------
    list[Polyline]
        All polylines for the entire dot set.  Every polyline satisfies
        ``len(polyline) >= 2`` (Generator ABC contract).

        - ``point``  → 1 polyline per dot, length 2.
        - ``cross``  → 2 polylines per dot, length 2 each.
        - ``circle`` → 1 closed polyline per dot, 12 vertices.
        - ``disc``   → 1 continuous polyline per dot: an Archimedean spiral
          from centre to the rim (single pen stroke, no lifts) finished with a
          full perimeter circle for a crisp edge.
    """
    if style == "point":
        return _render_points(coords_mm)
    elif style == "cross":
        return _render_crosses(coords_mm, size_mm)
    elif style == "circle":
        return _render_circles(coords_mm, size_mm)
    elif style == "disc":
        return _render_discs(coords_mm, size_mm, fill_spacing_mm)
    else:
        raise ValueError(
            f"Unknown dot style: {style!r}. "
            "Expected 'point', 'cross', 'circle', or 'disc'."
        )


# -- private helpers ---------------------------------------------------------

def _render_points(coords_mm: np.ndarray) -> list[Polyline]:
    """1 polyline (length 2) per dot — pen-down / immediate pen-up mark."""
    result: list[Polyline] = []
    for row in coords_mm:
        x, y = float(row[0]), float(row[1])
        result.append([(x, y), (x + 0.01, y)])
    return result


def _render_crosses(coords_mm: np.ndarray, size_mm: float) -> list[Polyline]:
    """2 polylines per dot — horizontal + vertical arms."""
    r = size_mm * 0.5
    result: list[Polyline] = []
    for row in coords_mm:
        x, y = float(row[0]), float(row[1])
        result.append([(x - r, y), (x + r, y)])
        result.append([(x, y - r), (x, y + r)])
    return result


def _render_circles(coords_mm: np.ndarray, size_mm: float) -> list[Polyline]:
    """1 closed polyline per dot: 12 circle vertices + duplicate of vertex 0."""
    r = size_mm * 0.5
    result: list[Polyline] = []
    angles = [2.0 * math.pi * k / _CIRCLE_SIDES for k in range(_CIRCLE_SIDES)]
    cos_a = [math.cos(a) for a in angles]
    sin_a = [math.sin(a) for a in angles]
    for row in coords_mm:
        x, y = float(row[0]), float(row[1])
        pts: Polyline = [(x + r * cos_a[k], y + r * sin_a[k]) for k in range(_CIRCLE_SIDES)]
        pts.append(pts[0])  # close the loop so the pen returns to start
        result.append(pts)
    return result


def _disc_template(r: float, spacing_mm: float) -> list[tuple[float, float]]:
    """Build the centre-relative point list for one spiral-filled disc.

    An Archimedean spiral ``rad = a*theta`` grows from the centre out to radius
    *r*, with one full turn per *spacing_mm* of radius so adjacent arms sit a
    pen-width apart.  A full perimeter circle is appended (starting at the
    spiral's end angle, so it joins seamlessly) to give a crisp outer edge.
    The same template is reused for every dot of this size, just translated.
    """
    spacing = max(1e-3, spacing_mm)
    turns = max(1.0, r / spacing)
    theta_max = 2.0 * math.pi * turns
    steps = max(2, int(round(turns * _SPIRAL_PTS_PER_TURN)))
    a = r / theta_max  # rad(theta_max) == r

    template: list[tuple[float, float]] = []
    for i in range(steps + 1):
        theta = theta_max * i / steps
        rad = a * theta
        template.append((rad * math.cos(theta), rad * math.sin(theta)))

    # Crisp rim: a full circle beginning at the spiral's final angle so the
    # first rim point coincides with the spiral's last point (one continuous
    # stroke, no pen lift).
    for k in range(_CIRCLE_SIDES + 1):
        ang = theta_max + 2.0 * math.pi * k / _CIRCLE_SIDES
        template.append((r * math.cos(ang), r * math.sin(ang)))
    return template


def _render_discs(
    coords_mm: np.ndarray, size_mm: float, spacing_mm: float
) -> list[Polyline]:
    """1 continuous polyline per dot: spiral fill from centre to rim + edge."""
    r = size_mm * 0.5
    if r <= 0.0:
        # Degenerate radius — fall back to a minimal mark per dot.
        return _render_points(coords_mm)
    template = _disc_template(r, spacing_mm)
    result: list[Polyline] = []
    for row in coords_mm:
        x, y = float(row[0]), float(row[1])
        result.append([(x + dx, y + dy) for dx, dy in template])
    return result
