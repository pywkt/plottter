"""Overpass QL query builder and HTTP client for the Map generator.

``build_query`` assembles a pure-string Overpass QL query from a bbox and a
list of selector strings (as produced by ``categories.selectors_for_categories``).
``fetch_overpass`` sends the query to the Overpass API and returns the parsed
JSON response dict.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Sequence


class OverpassError(Exception):
    """Raised when the Overpass API returns an unrecoverable error."""


# ---------------------------------------------------------------------------
# Query builder — pure string, no network
# ---------------------------------------------------------------------------


def build_query(
    bbox: tuple[float, float, float, float],
    selectors: Sequence[str],
    timeout: int = 90,
) -> str:
    """Return an Overpass QL query string.

    Parameters
    ----------
    bbox:
        Geographic bounding box as ``(south, west, north, east)`` in WGS84
        degrees.
    selectors:
        Overpass tag-filter clauses *without* a bbox suffix, e.g.
        ``'way["highway"~"^(motorway|trunk)$"]'``.  Each clause has
        ``(S,W,N,E);`` appended and becomes one line inside the union block.
    timeout:
        Overpass ``[timeout:…]`` value in seconds.

    Returns
    -------
    str
        A complete Overpass QL query ready to POST to the interpreter
        endpoint.  Starts with ``[out:json]``, ends with ``out geom;``.
    """
    south, west, north, east = bbox
    bbox_str = f"({south},{west},{north},{east})"

    clause_lines = "".join(
        f"  {selector}{bbox_str};\n" for selector in selectors
    )

    return (
        f"[out:json][timeout:{timeout}];\n"
        f"(\n"
        f"{clause_lines}"
        f");\n"
        f"out geom;"
    )


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"

# Retry delays in seconds for 429 / 504 responses.
_RETRY_DELAYS = [2, 8]

# Tag keys whose presence (any value) marks a closed way as an area.
_AREA_TAG_KEYS = frozenset({"building", "leisure", "landuse", "amenity"})

# Values for the "natural" key that mark a closed way as an area.
_AREA_NATURAL_VALUES = frozenset({"water", "wood", "scrub", "grassland", "wetland"})


def _is_area_way(tags: dict, coords: list) -> bool:
    """Return True if a way with *tags* and *coords* represents an area polygon."""
    if len(coords) < 2 or coords[0] != coords[-1]:
        return False  # not a closed ring
    for key in _AREA_TAG_KEYS:
        if key in tags:
            return True
    if tags.get("natural") in _AREA_NATURAL_VALUES:
        return True
    return False


def _parse_elements(elements: list) -> list:
    """Parse a list of Overpass JSON elements into ``MapFeature`` objects.

    Parameters
    ----------
    elements:
        The ``"elements"`` list from a parsed Overpass JSON response.

    Returns
    -------
    list[MapFeature]
        One ``MapFeature`` per way with geometry and per outer relation member.
        Elements without a ``geometry`` array are silently skipped.
    """
    from .types import MapFeature

    features: list = []

    for elem in elements:
        elem_type = elem.get("type")

        if elem_type == "way":
            geometry = elem.get("geometry") or []
            if not geometry:
                continue
            coords = [(g["lat"], g["lon"]) for g in geometry]
            tags = elem.get("tags") or {}
            features.append(MapFeature(
                tags=tags,
                coords=coords,
                is_area=_is_area_way(tags, coords),
            ))

        elif elem_type == "relation":
            tags = elem.get("tags") or {}
            members = elem.get("members") or []

            outer_ways: list = []
            inner_ways: list = []

            for member in members:
                if member.get("type") != "way":
                    continue
                geometry = member.get("geometry") or []
                if not geometry:
                    continue
                coords = [(g["lat"], g["lon"]) for g in geometry]
                role = member.get("role", "outer")
                if role == "inner":
                    inner_ways.append(coords)
                else:
                    outer_ways.append(coords)

            # Stitch fragmented member ways into closed rings (spec §6.4). A big
            # river's outer boundary is split across many open bank segments;
            # joining them head-to-tail yields one valid ring instead of bogus
            # per-segment slivers.
            from .geometry import assemble_rings, point_in_ring

            outer_rings = assemble_rings(outer_ways)
            inner_rings = assemble_rings(inner_ways)

            for outer in outer_rings:
                if len(outer_rings) == 1:
                    holes = inner_rings
                else:
                    # Assign each inner ring to the outer ring that contains it.
                    holes = [
                        inr
                        for inr in inner_rings
                        if inr and point_in_ring(inr[0], outer)
                    ]
                features.append(MapFeature(
                    tags=tags,
                    coords=outer,
                    is_area=True,
                    inner_coords=holes,
                ))

    return features


def fetch_overpass(
    bbox: tuple[float, float, float, float],
    selectors: Sequence[str],
    *,
    endpoint: str = _DEFAULT_ENDPOINT,
    user_agent: str,
    timeout: int = 90,
) -> list:
    """POST an Overpass QL query, parse elements, and return MapFeature list.

    Parameters
    ----------
    bbox:
        ``(south, west, north, east)`` in WGS84 degrees.
    selectors:
        Tag-filter clauses (see ``build_query``).
    endpoint:
        Overpass API interpreter URL.
    user_agent:
        ``User-Agent`` header value.  Overpass asks for a descriptive UA.
    timeout:
        Query timeout in seconds (also used as the HTTP socket timeout).

    Returns
    -------
    list[MapFeature]
        Parsed features from the Overpass response elements.

    Raises
    ------
    OverpassError
        On HTTP 429 / 504 after retries, or any other non-200 response.
    """
    import json as _json

    query = build_query(bbox, selectors, timeout)
    body = urllib.parse.urlencode({"data": query}).encode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": user_agent,
    }

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
                data = _json.loads(resp.read().decode())
            return _parse_elements(data.get("elements", []))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 504):
                last_exc = exc
                continue
            raise OverpassError(
                f"Overpass API returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OverpassError(f"Network error querying Overpass: {exc.reason}") from exc

    assert last_exc is not None
    raise OverpassError(
        f"Overpass API overloaded (HTTP {last_exc.code}) after {len(_RETRY_DELAYS) + 1} "  # type: ignore[attr-defined]
        "attempts. Try a smaller radius or an alternate endpoint."
    ) from last_exc
