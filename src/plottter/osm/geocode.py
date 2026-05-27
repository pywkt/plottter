"""Nominatim geocoding client for the OSM map generator.

Provides a thin wrapper around the Nominatim search endpoint using only
``urllib`` (no third-party HTTP library needed).  The function enforces
the Nominatim usage policy: a descriptive ``User-Agent`` header and a
minimum 1 request per second rate limit.

Usage::

    from plottter.osm.geocode import geocode, GeocodeResult, GeocodeError

    result = geocode("Kyoto, Japan", user_agent="Plottter/1.0 (pen-plotter map art)")
    if result:
        print(result.lat, result.lon)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class GeocodeResult:
    """Parsed result from a Nominatim geocode query.

    Attributes:
        display_name: Human-readable location name from Nominatim.
        lat:  Latitude  of the result centre (WGS84 degrees).
        lon:  Longitude of the result centre (WGS84 degrees).
        bbox: Bounding box as ``(south, north, west, east)`` in WGS84
              degrees, taken directly from Nominatim's ``boundingbox``
              field (``[minlat, maxlat, minlon, maxlon]`` string array).
    """

    display_name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float]


class GeocodeError(Exception):
    """Raised on HTTP or network errors from the Nominatim endpoint."""


# ---------------------------------------------------------------------------
# Module-level throttle state (Nominatim policy: max 1 req/s)
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
_MIN_INTERVAL: float = 1.0  # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ENDPOINT = "https://nominatim.openstreetmap.org/search"


def geocode(
    query: str,
    *,
    user_agent: str,
    timeout: int = 15,
) -> GeocodeResult | None:
    """Geocode *query* via Nominatim and return the top result.

    Args:
        query:      Free-text location string (e.g. ``"Kyoto, Japan"``).
        user_agent: Value for the ``User-Agent`` request header.  Nominatim
                    rejects requests without a descriptive UA string.
        timeout:    Socket timeout in seconds (default 15).

    Returns:
        A :class:`GeocodeResult` for the top hit, or ``None`` if Nominatim
        returns an empty result list.

    Raises:
        GeocodeError: On ``urllib.error.HTTPError`` or
                      ``urllib.error.URLError`` (network failure, DNS
                      error, timeout, etc.).
    """
    global _last_call_time

    # Enforce ≥1 s between successive calls (Nominatim usage policy).
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.monotonic()

    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": "1",
            "addressdetails": "0",
        }
    )
    url = f"{_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data: list[dict] = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise GeocodeError(
            f"Nominatim returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GeocodeError(
            f"Nominatim network error: {exc.reason}"
        ) from exc

    if not data:
        return None

    hit = data[0]
    # Nominatim boundingbox: [minlat, maxlat, minlon, maxlon]
    # i.e. [south, north, west, east]
    bb = hit["boundingbox"]
    return GeocodeResult(
        display_name=hit["display_name"],
        lat=float(hit["lat"]),
        lon=float(hit["lon"]),
        bbox=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
    )
