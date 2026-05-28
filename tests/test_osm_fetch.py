"""Tests for osm/__init__.py — fetch_map_data orchestration (no live network)."""

from __future__ import annotations

from math import cos, radians
from unittest.mock import patch

import pytest

from plottter.osm import MapData, MapFeature, fetch_map_data
from plottter.osm.geocode import GeocodeResult

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_TEST_UA = "TestAgent/1.0 (pytest)"
_TEST_ENDPOINT = "http://test.example/overpass"

# Kyoto-ish coordinates
_LAT = 35.0116
_LON = 135.7681

# Nominatim bbox order: (south, north, west, east)
_NOMINATIM_BBOX = (34.90, 35.12, 135.60, 135.90)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geocode_result() -> GeocodeResult:
    return GeocodeResult(
        display_name="Kyoto, Japan",
        lat=_LAT,
        lon=_LON,
        bbox=_NOMINATIM_BBOX,
    )


def _road_feature() -> MapFeature:
    return MapFeature(
        tags={"highway": "primary"},
        coords=[(_LAT, _LON), (_LAT + 0.001, _LON + 0.001)],
        is_area=False,
    )


def _park_feature() -> MapFeature:
    return MapFeature(
        tags={"leisure": "park"},
        coords=[(_LAT, _LON), (_LAT + 0.002, _LON), (_LAT + 0.002, _LON + 0.002),
                (_LAT, _LON + 0.002), (_LAT, _LON)],
        is_area=True,
    )


def _call_fetch(
    enabled_categories: list[str] | None = None,
    extent_mode: str = "center_radius",
    road_detail: str = "standard",
    overpass_side_effect=None,
    overpass_return_value=None,
    geocode_return_value=None,
    progress_callback=None,
) -> MapData:
    """Helper: call fetch_map_data with mocked network."""
    if enabled_categories is None:
        enabled_categories = ["roads_major"]
    if geocode_return_value is None:
        geocode_return_value = _geocode_result()

    overpass_kwargs: dict = {}
    if overpass_side_effect is not None:
        overpass_kwargs["side_effect"] = overpass_side_effect
    else:
        overpass_kwargs["return_value"] = (
            [_road_feature()] if overpass_return_value is None else overpass_return_value
        )

    with patch("plottter.osm.geocode.geocode", return_value=geocode_return_value), \
         patch("plottter.osm.overpass.fetch_overpass", **overpass_kwargs):
        return fetch_map_data(
            "Kyoto",
            radius_km=1.5,
            extent_mode=extent_mode,
            enabled_categories=enabled_categories,
            road_detail=road_detail,
            endpoint=_TEST_ENDPOINT,
            user_agent=_TEST_UA,
            progress_callback=progress_callback,
        )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_map_data():
    """fetch_map_data must return a MapData instance."""
    result = _call_fetch()
    assert isinstance(result, MapData)


# ---------------------------------------------------------------------------
# Center and location
# ---------------------------------------------------------------------------


def test_center_is_geocoded_lat_lon():
    """center must be the (lat, lon) from the geocode result."""
    result = _call_fetch()
    assert result.center == (_LAT, _LON)


def test_location_stored_verbatim():
    """location field must echo the original query string."""
    with patch("plottter.osm.geocode.geocode", return_value=_geocode_result()), \
         patch("plottter.osm.overpass.fetch_overpass", return_value=[_road_feature()]):
        result = fetch_map_data(
            "Kyoto, Japan",
            radius_km=1.5,
            enabled_categories=["roads_major"],
            endpoint=_TEST_ENDPOINT,
            user_agent=_TEST_UA,
        )
    assert result.location == "Kyoto, Japan"


# ---------------------------------------------------------------------------
# Bbox — center+radius mode
# ---------------------------------------------------------------------------


def test_bbox_south_west_north_east_ordering():
    """bbox must be in (south, west, north, east) order."""
    result = _call_fetch(extent_mode="center_radius")
    s, w, n, e = result.bbox
    # south < north, west < east
    assert s < n
    assert w < e


def test_bbox_center_radius_values():
    """Bbox corners must match the §4.2 formula."""
    radius_km = 1.5
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * cos(radians(_LAT)))
    result = _call_fetch(extent_mode="center_radius")
    s, w, n, e = result.bbox
    assert abs(s - (_LAT - dlat)) < 1e-9
    assert abs(n - (_LAT + dlat)) < 1e-9
    assert abs(w - (_LON - dlon)) < 1e-9
    assert abs(e - (_LON + dlon)) < 1e-9


# ---------------------------------------------------------------------------
# Bbox — place_bbox mode
# ---------------------------------------------------------------------------


def test_place_bbox_uses_nominatim_bbox():
    """place_bbox mode must convert GeocodeResult.bbox to Overpass ordering."""
    result = _call_fetch(extent_mode="place_bbox")
    # GeocodeResult.bbox = (south, north, west, east)
    # Overpass ordering  = (south, west, north, east)
    nom_s, nom_n, nom_w, nom_e = _NOMINATIM_BBOX
    assert result.bbox == (nom_s, nom_w, nom_n, nom_e)


def test_place_bbox_center_still_geocoded():
    """center must still use the geocoded lat/lon in place_bbox mode."""
    result = _call_fetch(extent_mode="place_bbox")
    assert result.center == (_LAT, _LON)


# ---------------------------------------------------------------------------
# Features dict — enabled / disabled categories
# ---------------------------------------------------------------------------


def test_enabled_category_present_in_features():
    """An enabled category with returned features must appear in features."""
    result = _call_fetch(enabled_categories=["roads_major"])
    assert "roads_major" in result.features


def test_disabled_category_absent():
    """Categories not in enabled_categories must be absent from features."""
    result = _call_fetch(enabled_categories=["roads_major"])
    for cat in ("rail", "water", "waterways", "parks", "buildings", "coastline"):
        assert cat not in result.features, f"{cat} should not be in features"


def test_multiple_enabled_categories():
    """Multiple enabled categories each with data appear in features."""
    result = _call_fetch(
        enabled_categories=["roads_major", "parks"],
        overpass_side_effect=[[_road_feature()], [_park_feature()]],
    )
    assert "roads_major" in result.features
    assert "parks" in result.features


def test_category_with_empty_overpass_response_absent():
    """A category whose Overpass query returns [] must be absent from features."""
    result = _call_fetch(
        enabled_categories=["roads_major", "rail"],
        overpass_side_effect=[[_road_feature()], []],
    )
    assert "roads_major" in result.features
    assert "rail" not in result.features


def test_unknown_category_silently_ignored():
    """Unknown category ids not in FEATURE_CATEGORIES must be silently ignored."""
    result = _call_fetch(enabled_categories=["roads_major", "nonexistent_category"])
    assert "nonexistent_category" not in result.features
    assert "roads_major" in result.features


def test_features_values_are_map_feature_lists():
    """features[cat] must be a non-empty list of MapFeature objects."""
    result = _call_fetch(enabled_categories=["roads_major"])
    feats = result.features["roads_major"]
    assert isinstance(feats, list)
    assert len(feats) > 0
    assert all(isinstance(f, MapFeature) for f in feats)


# ---------------------------------------------------------------------------
# roads_minor + road_detail="major_only" → absent
# ---------------------------------------------------------------------------


def test_roads_minor_absent_when_major_only():
    """roads_minor must be absent when road_detail='major_only' (selector empty)."""
    result = _call_fetch(
        enabled_categories=["roads_major", "roads_minor"],
        road_detail="major_only",
        overpass_return_value=[_road_feature()],
    )
    # roads_minor has no selectors in major_only mode, so it must be absent.
    assert "roads_minor" not in result.features


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_progress_reaches_100():
    """progress_callback must be called with 100 before the function returns."""
    values: list[int] = []
    _call_fetch(progress_callback=values.append)
    assert 100 in values
    assert values[-1] == 100


def test_progress_starts_at_10_after_geocode():
    """First progress call must be 10 (geocode completed)."""
    values: list[int] = []
    _call_fetch(progress_callback=values.append)
    assert values[0] == 10


def test_progress_monotone():
    """Progress values must be non-decreasing."""
    values: list[int] = []
    _call_fetch(
        enabled_categories=["roads_major", "rail", "parks"],
        overpass_side_effect=[[_road_feature()], [], [_park_feature()]],
        progress_callback=values.append,
    )
    assert values == sorted(values), f"progress not monotone: {values}"


def test_progress_no_callback_does_not_raise():
    """Omitting progress_callback must not raise."""
    _call_fetch(progress_callback=None)  # no exception


def test_progress_zero_categories():
    """With no active categories, progress must still reach 100."""
    values: list[int] = []
    with patch("plottter.osm.geocode.geocode", return_value=_geocode_result()), \
         patch("plottter.osm.overpass.fetch_overpass") as mock_op:
        fetch_map_data(
            "Kyoto",
            radius_km=1.5,
            enabled_categories=[],
            endpoint=_TEST_ENDPOINT,
            user_agent=_TEST_UA,
            progress_callback=values.append,
        )
        mock_op.assert_not_called()
    assert values[-1] == 100


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


def test_geocode_none_raises_value_error():
    """None geocode result must raise ValueError mentioning the location."""
    with patch("plottter.osm.geocode.geocode", return_value=None):
        with pytest.raises(ValueError, match="geocode"):
            fetch_map_data(
                "unknown place xyzzy",
                radius_km=1.5,
                enabled_categories=["roads_major"],
                endpoint=_TEST_ENDPOINT,
                user_agent=_TEST_UA,
            )


def test_attribution_default():
    """MapData must carry the OSM attribution string."""
    result = _call_fetch()
    assert "OpenStreetMap" in result.attribution


# ---------------------------------------------------------------------------
# Places category — node elements
# ---------------------------------------------------------------------------

_PLACES_FIXTURE = (
    __import__("pathlib").Path(__file__).parent / "fixtures" / "osm" / "places_small.json"
)


def _load_places_features():
    """Parse the places fixture into MapFeature objects via _parse_elements."""
    import json

    from plottter.osm.overpass import _parse_elements

    data = json.loads(_PLACES_FIXTURE.read_text())
    return _parse_elements(data["elements"])


def test_places_category_populated_in_features():
    """fetch_map_data with 'places' enabled must populate features['places']."""
    place_features = _load_places_features()

    with patch("plottter.osm.geocode.geocode", return_value=_geocode_result()), \
         patch("plottter.osm.overpass.fetch_overpass", return_value=place_features):
        result = fetch_map_data(
            "Kyoto",
            radius_km=1.5,
            enabled_categories=["places"],
            endpoint=_TEST_ENDPOINT,
            user_agent=_TEST_UA,
        )

    assert "places" in result.features
    assert len(result.features["places"]) == 2


def test_places_features_are_point_nodes():
    """Place node features from the fixture must have a single coord and is_area=False."""
    place_features = _load_places_features()

    for feat in place_features:
        assert len(feat.coords) == 1, "place node must have exactly one coordinate"
        assert feat.is_area is False, "place node must not be flagged as an area"


def test_places_feature_tags_include_place_values():
    """Fixture place features must carry the expected place tag values."""
    place_features = _load_places_features()

    place_vals = {f.tags.get("place") for f in place_features}
    assert "island" in place_vals
    assert "neighbourhood" in place_vals


def test_places_absent_when_not_enabled():
    """'places' must be absent from features when it is not in enabled_categories."""
    place_features = _load_places_features()

    with patch("plottter.osm.geocode.geocode", return_value=_geocode_result()), \
         patch("plottter.osm.overpass.fetch_overpass", return_value=place_features):
        result = fetch_map_data(
            "Kyoto",
            radius_km=1.5,
            enabled_categories=["roads_major"],
            endpoint=_TEST_ENDPOINT,
            user_agent=_TEST_UA,
        )

    assert "places" not in result.features
