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
