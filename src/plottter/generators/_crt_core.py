"""Pure-numeric helpers for the CRT TV generator.

No Qt imports; no plottter.gui imports. All functions are stateless and
operate purely on numeric data.
"""

import math
import numpy as np


def subpixel_layout(
    mask_type: str,
    pen_index: int,
    n_pens: int,
    row: int = 0,
) -> tuple[float, float]:
    """Return (x_frac, y_frac) offset within a unit cell for pen *pen_index*.

    Coordinates are in [0, 1]² where (0, 0) is the cell top-left.

    Parameters
    ----------
    mask_type:
        One of ``"shadow_mask"``, ``"aperture_grille"``, ``"slot_mask"``.
    pen_index:
        Zero-based index of the pen within the palette.
    n_pens:
        Total number of pens in the palette.
    row:
        Source-pixel row index.  Only used by ``slot_mask`` to determine
        y-offset parity (even rows → 0.35, odd rows → 0.65).  Ignored by
        the other two mask types.

    Returns
    -------
    (x_frac, y_frac) both in [0, 1].

    Raises
    ------
    ValueError
        If *pen_index* is negative or >= *n_pens*.
    """
    if pen_index < 0 or pen_index >= n_pens:
        raise ValueError(
            f"pen_index {pen_index!r} out of range for n_pens={n_pens}"
        )

    if mask_type == "shadow_mask":
        return _shadow_mask_layout(pen_index, n_pens)
    elif mask_type == "aperture_grille":
        return _aperture_grille_layout(pen_index, n_pens)
    elif mask_type == "slot_mask":
        return _slot_mask_layout(pen_index, n_pens, row)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type!r}")


# ---------------------------------------------------------------------------
# Internal layout helpers
# ---------------------------------------------------------------------------

def _shadow_mask_layout(pen_index: int, n_pens: int) -> tuple[float, float]:
    """Shadow-mask (phosphor-dot triad) layout."""
    if n_pens == 3:
        positions = [
            (0.50, 0.25),  # Pen 0 — top centre
            (0.25, 0.75),  # Pen 1 — bottom-left
            (0.75, 0.75),  # Pen 2 — bottom-right
        ]
        return positions[pen_index]
    elif n_pens == 4:
        positions = [
            (0.25, 0.25),  # Pen 0
            (0.75, 0.25),  # Pen 1
            (0.25, 0.75),  # Pen 2
            (0.75, 0.75),  # Pen 3
        ]
        return positions[pen_index]
    else:
        # n_pens >= 5: regular polygon on a circle of radius 0.35 centred at
        # (0.5, 0.5), starting at -π/2 (first pen at top).
        radius = 0.35
        angle = -math.pi / 2 + pen_index * (2 * math.pi / n_pens)
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        return (x, y)


def _aperture_grille_layout(pen_index: int, n_pens: int) -> tuple[float, float]:
    """Aperture-grille (vertical stripe) layout — Sony Trinitron style."""
    x = (pen_index + 0.5) / n_pens
    y = 0.5
    return (x, y)


def _slot_mask_layout(pen_index: int, n_pens: int, row: int) -> tuple[float, float]:
    """Slot-mask layout — same x as aperture_grille, y depends on row parity."""
    x = (pen_index + 0.5) / n_pens
    y = 0.35 if (row % 2 == 0) else 0.65
    return (x, y)


# ---------------------------------------------------------------------------
# Scanline mask
# ---------------------------------------------------------------------------

def scanline_mask(
    rows: int,
    cols: int,
    intensity: float,
    period: int,
) -> np.ndarray:
    """Return a (rows, cols) float32 array of per-pixel keep multipliers.

    For every row ``r`` where ``r % period == 0`` the multiplier is
    ``(1 - intensity)``; all other rows have multiplier 1.0.

    Parameters
    ----------
    rows, cols:
        Output array dimensions.
    intensity:
        Scanline darkening strength in [0, 1].  0 → no effect, 1 → fully
        remove targeted rows.
    period:
        Apply darkening every *period*-th row.  ``period == 2`` → classic
        every-other-row scanline.

    Returns
    -------
    float32 array of shape (rows, cols).
    """
    mask = np.ones((rows, cols), dtype=np.float32)
    multiplier = 1.0 - float(intensity)
    # Mark every period-th row
    mask[::period, :] = multiplier
    return mask


# ---------------------------------------------------------------------------
# Vignette mask
# ---------------------------------------------------------------------------

def vignette_mask(
    rows: int,
    cols: int,
    strength: float,
) -> np.ndarray:
    """Return a (rows, cols) float32 array with smooth radial fall-off.

    Value is 1.0 at the centre and ``(1 - strength)`` at the corners.

    Parameters
    ----------
    rows, cols:
        Output array dimensions.
    strength:
        Vignette strength in [0, 1].  0 → no effect (all ones), 1 → full
        darkening at corners.

    Returns
    -------
    float32 array of shape (rows, cols).
    """
    # Pixel-centred normalisation so that:
    #   - The centre pixel (rows//2, cols//2) for odd dims has norm == 0.
    #   - The array is flip-symmetric: m[r,c] == m[r, cols-1-c] etc.
    # Uses (size-1)/2.0 as the centre, so corners map to ±1 exactly.
    half_r = (rows - 1) / 2.0 if rows > 1 else 1.0
    half_c = (cols - 1) / 2.0 if cols > 1 else 1.0

    r_idx = np.arange(rows, dtype=np.float64)
    c_idx = np.arange(cols, dtype=np.float64)

    norm_r = (r_idx - half_r) / half_r   # shape (rows,)
    norm_c = (c_idx - half_c) / half_c   # shape (cols,)

    # Squared distance from centre; max is 2 at corners ((±1)²+(±1)²).
    d2 = norm_r[:, np.newaxis] ** 2 + norm_c[np.newaxis, :] ** 2

    v = 1.0 - float(strength) * np.minimum(1.0, d2 / 2.0)
    return v.astype(np.float32)


# ---------------------------------------------------------------------------
# Subpixel rectangle rendering (for aperture_grille / slot_mask masks)
# ---------------------------------------------------------------------------

def render_subpixel_rects(
    coords_mm: np.ndarray,
    width_mm: float,
    height_mm: float,
) -> list:
    """Render each centre point as a filled vertical rectangle.

    Produces two polylines per centre: a 5-point closed outline rect plus a
    centred vertical fill stroke.  The two together visibly fill the bar at
    typical pen widths (the outline alone leaves a hollow centre at larger
    widths; the fill alone leaves a thin line).

    Parameters
    ----------
    coords_mm:
        (N, 2) float array of (x, y) bar-centre coordinates in mm.
    width_mm:
        Bar width in mm (horizontal extent).  Controls bar thickness.
    height_mm:
        Bar height in mm (vertical extent).  Typically ~85% of the
        per-cell vertical pitch so adjacent bars leave a visible gap.

    Returns
    -------
    list of polylines: ``[outline, fill, outline, fill, …]`` — two per centre.
    """
    half_w = width_mm / 2.0
    half_h = height_mm / 2.0
    out: list = []
    for row in coords_mm:
        cx, cy = float(row[0]), float(row[1])
        outline = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
            (cx - half_w, cy - half_h),
        ]
        fill = [(cx, cy - half_h), (cx, cy + half_h)]
        out.append(outline)
        out.append(fill)
    return out


# ---------------------------------------------------------------------------
# Barrel warp
# ---------------------------------------------------------------------------

def barrel_warp(
    coords_mm: np.ndarray,
    centre_mm: tuple[float, float],
    strength: float,
    max_radius_mm: float,
) -> np.ndarray:
    """Apply barrel distortion to an (N, 2) array of mm coordinates.

    Points are pushed *outward* from *centre_mm* by an amount proportional
    to their normalised squared radius.

    Parameters
    ----------
    coords_mm:
        (N, 2) float array of (x, y) coordinates in millimetres.
    centre_mm:
        (centre_x, centre_y) in millimetres — the distortion origin.
    strength:
        Barrel strength.  Silently clamped to [0, 0.15].  0 → no-op.
    max_radius_mm:
        Normalisation radius (typically half the canvas diagonal).

    Returns
    -------
    (N, 2) float array with the same dtype as *coords_mm* (or float64 if
    input dtype does not support it).
    """
    # Silent clamp — no exception.
    strength = float(np.clip(strength, 0.0, 0.15))

    coords = np.asarray(coords_mm, dtype=np.float64)
    cx, cy = float(centre_mm[0]), float(centre_mm[1])

    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy

    r = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r / float(max_radius_mm)

    factor = 1.0 + strength * r_norm ** 2

    out = np.empty_like(coords)
    out[:, 0] = cx + dx * factor
    out[:, 1] = cy + dy * factor

    return out
