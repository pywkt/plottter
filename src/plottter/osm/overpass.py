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


def fetch_overpass(
    bbox: tuple[float, float, float, float],
    selectors: Sequence[str],
    *,
    endpoint: str = _DEFAULT_ENDPOINT,
    user_agent: str,
    timeout: int = 90,
) -> dict:
    """POST an Overpass QL query and return the parsed JSON response.

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
    dict
        Parsed JSON response from the Overpass API.

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
                return _json.loads(resp.read().decode())
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
