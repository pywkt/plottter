"""Web Mercator projection and fit-to-canvas geometry for the Map generator.

See spec §6 in specs/map-generator.md for the canonical design.
"""

import math

_LAT_CLAMP = 85.05112877980659  # degrees — avoids log(0) singularity at ±90°


def mercator(lat: float, lon: float) -> tuple[float, float]:
    """Project (lat, lon) to Web Mercator (x, y) with unit sphere radius.

    Latitude is clamped to ±85.05° to avoid the pole singularity.
    The caller rescales the output in fit-to-canvas (§6.2).

    Args:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).

    Returns:
        (x, y) in radians — x increases eastward, y increases northward.
    """
    lat = max(-_LAT_CLAMP, min(_LAT_CLAMP, lat))
    x = math.radians(lon)
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


from dataclasses import dataclass


@dataclass
class FitTransform:
    """Affine transform mapping Mercator (x, y) to canvas mm coordinates.

    Apply as::

        px, py = mercator(lat, lon)
        canvas_x = transform.x_origin + px * transform.scale
        canvas_y = transform.y_origin - py * transform.scale

    The y-axis is flipped so that north (larger Mercator y) maps to a
    smaller canvas y (toward the top of the page).
    """

    scale: float
    x_origin: float  # canvas_x = x_origin + px * scale
    y_origin: float  # canvas_y = y_origin - py * scale  (y-flipped)


def fit_transform(features: list, canvas) -> FitTransform:
    """Derive an affine transform fitting all features inside canvas.drawing_area().

    Projects every coordinate of every feature via Web Mercator, computes the
    projected bounding box, then derives a uniform scale and centering offsets
    so the full map fits inside the printable area with the aspect ratio
    preserved (spec §6.2).  The y-axis is flipped so north is up.

    Args:
        features: Iterable of MapFeature objects (each has .coords as a list
                  of (lat, lon) pairs in decimal degrees).
        canvas:   Canvas dataclass whose drawing_area() returns
                  (left, top, right, bottom) in mm.

    Returns:
        A FitTransform instance.

    Raises:
        ValueError: If no coordinates are found across all features.
    """
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            xs.append(px)
            ys.append(py)

    if not xs:
        raise ValueError(
            "No coordinates found in features; cannot compute fit transform."
        )

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    left, top, right, bottom = canvas.drawing_area()
    pw = right - left
    ph = bottom - top

    span_x = maxx - minx
    span_y = maxy - miny

    if span_x == 0.0 and span_y == 0.0:
        scale = 1.0
    elif span_x == 0.0:
        scale = ph / span_y
    elif span_y == 0.0:
        scale = pw / span_x
    else:
        scale = min(pw / span_x, ph / span_y)

    # Centering margins so the drawing is centred inside the printable area.
    cx = (pw - span_x * scale) / 2.0
    cy = (ph - span_y * scale) / 2.0

    # Derive origins so the application formula is simply:
    #   canvas_x = x_origin + px * scale
    #   canvas_y = y_origin - py * scale
    x_origin = left + cx - minx * scale
    y_origin = top + cy + maxy * scale

    return FitTransform(scale=scale, x_origin=x_origin, y_origin=y_origin)


def project_feature(feature, transform: FitTransform) -> list[tuple[float, float]]:
    """Project a MapFeature's coordinates to canvas mm using the given transform.

    Args:
        feature:   MapFeature with .coords as [(lat, lon), ...].
        transform: FitTransform returned by fit_transform().

    Returns:
        List of (x_mm, y_mm) canvas coordinates in mm.
    """
    result: list[tuple[float, float]] = []
    for lat, lon in feature.coords:
        px, py = mercator(lat, lon)
        canvas_x = transform.x_origin + px * transform.scale
        canvas_y = transform.y_origin - py * transform.scale
        result.append((canvas_x, canvas_y))
    return result
