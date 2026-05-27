"""OSM data-access subpackage.

Re-exports the public API surface used by MapGenerator and _MapMixin.
Individual sub-modules (geocode, overpass, categories, geometry, cache)
are imported lazily in the functions that need them so that the subpackage
can be imported in a headless/test context without network calls.
"""

from plottter.osm.types import MapData, MapFeature
from plottter.osm.categories import FEATURE_CATEGORIES

# NOTE: 'geocode' is intentionally NOT imported at module level.
# Doing so would shadow the `plottter.osm.geocode` *submodule* with the
# function of the same name, breaking `import plottter.osm.geocode as mod`
# in existing code.  Callers should import the function directly:
#   from plottter.osm.geocode import geocode
# or use the module-level __getattr__ accessor defined below so that
#   from plottter.osm import geocode
# returns the function without touching the submodule attribute.

__all__ = [
    "MapData",
    "MapFeature",
    "FEATURE_CATEGORIES",
    "geocode",
    "fetch_map_data",
]


def __getattr__(name: str):
    """Lazy re-export of ``geocode`` without shadowing the submodule."""
    if name == "geocode":
        from plottter.osm.geocode import geocode as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def fetch_map_data(
    location: str,
    *,
    radius_km: float,
    extent_mode: str = "center_radius",
    enabled_categories: list,
    road_detail: str = "standard",
    endpoint: str = "https://overpass-api.de/api/interpreter",
    user_agent: str,
    progress_callback=None,
) -> "MapData":
    """Fetch OSM data for *location* and return a :class:`MapData` payload.

    Parameters
    ----------
    location:
        Free-text place query (e.g. ``"Kyoto, Japan"``).
    radius_km:
        Half-width of the square bounding box when ``extent_mode="center_radius"``.
    extent_mode:
        ``"center_radius"`` (default) — build bbox from geocoded centre +
        *radius_km*.  ``"place_bbox"`` — use Nominatim's own bounding box for
        the place.
    enabled_categories:
        Iterable of category ids (keys of :data:`FEATURE_CATEGORIES`) to fetch.
        Categories not in ``FEATURE_CATEGORIES`` are silently ignored.
    road_detail:
        Road-tier filter: ``"major_only"``, ``"standard"`` (default), or
        ``"all_streets"``.  Passed through to
        :func:`~plottter.osm.categories.selectors_for_categories`.
    endpoint:
        Overpass API interpreter URL.
    user_agent:
        ``User-Agent`` header for all outbound requests.
    progress_callback:
        Optional callable ``(int) -> None`` called with integer percentages.
        Geocode completion is reported at ~10 %; the remainder reaches 100 %
        after all category fetches complete.

    Returns
    -------
    MapData
        Populated payload with ``center``, ``bbox``, and ``features`` keyed by
        enabled category id.  Categories whose Overpass query returns no
        features are omitted from ``features``.

    Raises
    ------
    ValueError
        If the location string cannot be geocoded.
    ~plottter.osm.geocode.GeocodeError
        On geocoding network failures.
    ~plottter.osm.overpass.OverpassError
        On Overpass network failures after retries.
    """
    from math import cos, radians

    from .categories import FEATURE_CATEGORIES as _CATS
    from .categories import selectors_for_categories
    from .geocode import geocode as _geocode
    from .overpass import fetch_overpass

    def _progress(pct: int) -> None:
        if progress_callback is not None:
            progress_callback(pct)

    # ------------------------------------------------------------------
    # Step 1: Geocode the location string → centre (lat, lon)
    # ------------------------------------------------------------------
    result = _geocode(location, user_agent=user_agent)
    if result is None:
        raise ValueError(f"Could not geocode location: {location!r}")
    _progress(10)

    # ------------------------------------------------------------------
    # Step 2: Compute geographic bbox (south, west, north, east)
    # ------------------------------------------------------------------
    if extent_mode == "place_bbox":
        # Use Nominatim's own bounding box for the place.
        # GeocodeResult.bbox = (south, north, west, east)  ← Nominatim order
        # Overpass ordering:   (south, west, north, east)
        s, n, w, e = result.bbox
        bbox = (s, w, n, e)
    else:
        # Default: square centred on geocoded point, sized by radius_km.
        lat, lon = result.lat, result.lon
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * cos(radians(lat)))
        bbox = (lat - dlat, lon - dlon, lat + dlat, lon + dlon)

    # ------------------------------------------------------------------
    # Step 3 & 4: One Overpass request per enabled category.
    #
    # Calling fetch_overpass per category gives unambiguous categorisation
    # without requiring tag re-matching of a flat result list.
    # ------------------------------------------------------------------
    features: dict = {}
    active_cats = [c for c in enabled_categories if c in _CATS]
    n_cats = len(active_cats)

    for i, cat_id in enumerate(active_cats):
        cat_selectors = selectors_for_categories([cat_id], road_detail)
        if cat_selectors:
            cat_features = fetch_overpass(
                bbox,
                cat_selectors,
                endpoint=endpoint,
                user_agent=user_agent,
            )
            if cat_features:
                features[cat_id] = cat_features

        # Intermediate progress (11 % … 99 %) only when more than one category.
        if n_cats > 1:
            _progress(10 + int((i + 1) / n_cats * 89))

    # Always end at exactly 100 %.
    _progress(100)

    return MapData(
        location=location,
        center=(result.lat, result.lon),
        bbox=bbox,
        features=features,
    )

