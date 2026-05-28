"""Tests for osm/overpass.py — build_query (pure string, no network)."""

import pytest

from plottter.osm.overpass import build_query


# Shared bbox for all tests: (south, west, north, east)
_BBOX = (34.9, 135.6, 35.1, 135.8)

# Representative selectors similar to those produced by categories.py
_SELECTORS = [
    'way["highway"~"^(motorway|trunk|primary|secondary)$"]',
    'way["railway"~"^(rail|subway|tram)$"]',
    'way["natural"="water"]',
]


def test_build_query_starts_with_out_json():
    """Output must start with [out:json]."""
    q = build_query(_BBOX, _SELECTORS)
    assert q.startswith("[out:json]")


def test_build_query_ends_with_out_geom():
    """Output must end with 'out geom;'."""
    q = build_query(_BBOX, _SELECTORS)
    assert q.rstrip().endswith("out geom;")


def test_build_query_contains_each_selector_with_bbox():
    """Each selector must appear in the output with the bbox substituted."""
    south, west, north, east = _BBOX
    q = build_query(_BBOX, _SELECTORS)
    for selector in _SELECTORS:
        assert selector in q, f"Selector not found in query: {selector!r}"
        # The bbox must be adjacent to the selector
        assert f"{selector}({south},{west},{north},{east});" in q


def test_build_query_one_clause_per_selector():
    """There must be exactly one clause line per selector."""
    q = build_query(_BBOX, _SELECTORS)
    for selector in _SELECTORS:
        assert q.count(selector) == 1, (
            f"Expected exactly one occurrence of selector {selector!r}"
        )


def test_build_query_timeout_default():
    """Default timeout of 90 must appear in the header."""
    q = build_query(_BBOX, _SELECTORS)
    assert "[timeout:90]" in q


def test_build_query_timeout_custom():
    """Custom timeout must be reflected in the header."""
    q = build_query(_BBOX, _SELECTORS, timeout=30)
    assert "[timeout:30]" in q
    assert "[timeout:90]" not in q


def test_build_query_empty_selectors():
    """An empty selector list must still produce a valid skeleton."""
    q = build_query(_BBOX, [])
    assert q.startswith("[out:json]")
    assert q.rstrip().endswith("out geom;")
    # The union body is empty but the parentheses must be present
    assert "(\n);" in q


def test_build_query_single_selector():
    """A single selector should yield one clause with the bbox."""
    south, west, north, east = _BBOX
    sel = 'way["building"]'
    q = build_query(_BBOX, [sel])
    assert q.startswith("[out:json]")
    assert q.rstrip().endswith("out geom;")
    assert f'{sel}({south},{west},{north},{east});' in q


# ---------------------------------------------------------------------------
# fetch_overpass — patching urlopen and time.sleep
# ---------------------------------------------------------------------------

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from plottter.osm.overpass import OverpassError, fetch_overpass

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "osm" / "overpass_small.json"
_FIXTURE_DATA = json.loads(_FIXTURE_PATH.read_bytes())

_UA = "TestAgent/1.0"
_BBOX = (34.9, 135.6, 35.1, 135.8)


def _make_urlopen_ok(data: dict):
    """Return a context-manager mock that yields a response reading *data*."""
    raw = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    cm = MagicMock()
    cm.return_value = resp
    return cm


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example.com",
        code=code,
        msg=str(code),
        hdrs=None,  # type: ignore[arg-type]
        fp=None,    # type: ignore[arg-type]
    )


class TestParseFixture:
    """fetch_overpass parses the small fixture into the expected MapFeatures."""

    def test_total_feature_count(self):
        """Fixture has 3 ways-with-geometry + 1 relation outer-ring = 4 features."""
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        assert len(features) == 4

    def test_is_area_streets_false(self):
        """Open highway ways must have is_area=False."""
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        streets = [f for f in features if "highway" in f.tags]
        assert len(streets) == 2
        assert all(not f.is_area for f in streets)

    def test_is_area_park_true(self):
        """Closed way with leisure=park must have is_area=True."""
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        parks = [f for f in features if f.tags.get("leisure") == "park"]
        assert len(parks) == 1
        assert parks[0].is_area is True

    def test_is_area_relation_true(self):
        """Relation member (water body) must have is_area=True."""
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        water = [f for f in features if f.tags.get("natural") == "water"]
        assert len(water) == 1
        assert water[0].is_area is True

    def test_relation_inner_coords_populated(self):
        """The water relation's inner ring must be stored in inner_coords."""
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        water = [f for f in features if f.tags.get("natural") == "water"]
        assert len(water[0].inner_coords) == 1
        assert len(water[0].inner_coords[0]) == 5  # 5-point closed ring

    def test_way_without_geometry_skipped(self):
        """Ways with an empty geometry array must be silently skipped."""
        # way 501 in the fixture has geometry=[], so it should be absent
        urlopen_mock = _make_urlopen_ok(_FIXTURE_DATA)
        with patch("plottter.osm.overpass.urllib.request.urlopen", urlopen_mock):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)
        ids_from_fixture = {101, 102, 201}  # ways with geometry
        # No feature should have highway=path (way 501, which had empty geometry)
        highway_paths = [f for f in features if f.tags.get("highway") == "path"]
        assert highway_paths == []


class TestMultipolygonStitching:
    """Relation outer boundaries split across open member ways are stitched into
    one closed ring (regression for water drawn over land / unfilled rivers)."""

    @staticmethod
    def _fragmented_river_relation():
        """A square water relation whose outer ring is split across 3 open ways
        (one reversed), plus one closed inner ring (an island)."""
        return [
            {
                "type": "relation",
                "tags": {"natural": "water", "name": "River"},
                "members": [
                    {"type": "way", "role": "outer", "geometry": [
                        {"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 1.0}]},
                    {"type": "way", "role": "outer", "geometry": [
                        {"lat": 0.0, "lon": 1.0}, {"lat": 1.0, "lon": 1.0},
                        {"lat": 1.0, "lon": 0.0}]},
                    # Reversed direction; shares endpoints with the others.
                    {"type": "way", "role": "outer", "geometry": [
                        {"lat": 0.0, "lon": 0.0}, {"lat": 1.0, "lon": 0.0}]},
                    {"type": "way", "role": "inner", "geometry": [
                        {"lat": 0.3, "lon": 0.3}, {"lat": 0.3, "lon": 0.6},
                        {"lat": 0.6, "lon": 0.6}, {"lat": 0.6, "lon": 0.3},
                        {"lat": 0.3, "lon": 0.3}]},
                ],
            }
        ]

    def test_fragmented_outer_becomes_single_ring(self):
        """3 open outer member ways stitch into exactly one closed outer feature
        (previously: 3 separate force-closed slivers)."""
        from plottter.osm.overpass import _parse_elements

        feats = _parse_elements(self._fragmented_river_relation())
        assert len(feats) == 1
        outer = feats[0]
        assert outer.is_area is True
        assert outer.coords[0] == outer.coords[-1]  # closed
        # The stitched ring covers all four corners of the square.
        corners = {(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)}
        assert corners.issubset(set(outer.coords))

    def test_inner_ring_assigned_as_hole(self):
        """The closed inner member is carried as a hole on the stitched outer."""
        from plottter.osm.overpass import _parse_elements
        from plottter.osm.geometry import point_in_ring

        feats = _parse_elements(self._fragmented_river_relation())
        outer = feats[0]
        assert len(outer.inner_coords) == 1
        # The hole lies inside the outer ring.
        assert point_in_ring(outer.inner_coords[0][0], outer.coords)


class TestNodeParsing:
    """_parse_elements handles node elements correctly."""

    def test_node_with_lat_lon_and_tags(self):
        """Node with lat/lon and name tag → MapFeature with coords length 1."""
        from plottter.osm.overpass import _parse_elements

        elements = [
            {
                "type": "node",
                "lat": 35.0,
                "lon": 135.7,
                "tags": {"name": "Test Node", "amenity": "cafe"},
            }
        ]
        feats = _parse_elements(elements)
        assert len(feats) == 1
        feat = feats[0]
        assert len(feat.coords) == 1
        assert feat.coords[0] == (35.0, 135.7)
        assert feat.is_area is False
        assert feat.tags["name"] == "Test Node"
        assert feat.tags["amenity"] == "cafe"

    def test_node_without_lat_lon_skipped(self):
        """Node without lat/lon is silently skipped."""
        from plottter.osm.overpass import _parse_elements

        elements = [
            {"type": "node", "tags": {"name": "Incomplete Node"}},
        ]
        feats = _parse_elements(elements)
        assert feats == []

    def test_node_missing_lat_only_skipped(self):
        """Node with only lon (no lat) is silently skipped."""
        from plottter.osm.overpass import _parse_elements

        elements = [
            {"type": "node", "lon": 135.7, "tags": {"name": "No Lat"}},
        ]
        feats = _parse_elements(elements)
        assert feats == []

    def test_node_missing_lon_only_skipped(self):
        """Node with only lat (no lon) is silently skipped."""
        from plottter.osm.overpass import _parse_elements

        elements = [
            {"type": "node", "lat": 35.0, "tags": {"name": "No Lon"}},
        ]
        feats = _parse_elements(elements)
        assert feats == []

    def test_node_without_tags_produces_feature_with_empty_tags(self):
        """Node without tags still produces a MapFeature with empty tags dict."""
        from plottter.osm.overpass import _parse_elements

        elements = [
            {"type": "node", "lat": 35.0, "lon": 135.7},
        ]
        feats = _parse_elements(elements)
        assert len(feats) == 1
        feat = feats[0]
        assert feat.tags == {}
        assert feat.coords == [(35.0, 135.7)]
        assert feat.is_area is False


class TestRetryBehavior:
    """fetch_overpass retries on 429/504 and raises OverpassError on persistent failure."""

    def test_429_then_200_retries_once(self):
        """A single 429 followed by 200 must succeed after one sleep(2)."""
        http_err = _make_http_error(429)
        ok_resp = MagicMock()
        ok_resp.read.return_value = json.dumps(_FIXTURE_DATA).encode()
        ok_resp.__enter__ = lambda s: s
        ok_resp.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise http_err
            return ok_resp

        with (
            patch(
                "plottter.osm.overpass.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
            patch("plottter.osm.overpass.time.sleep") as sleep_mock,
        ):
            features = fetch_overpass(_BBOX, [], user_agent=_UA)

        assert len(features) == 4
        sleep_mock.assert_called_once_with(2)

    def test_persistent_504_raises_overpass_error(self):
        """Three consecutive 504s must raise OverpassError."""
        http_err = _make_http_error(504)

        with (
            patch(
                "plottter.osm.overpass.urllib.request.urlopen",
                side_effect=http_err,
            ),
            patch("plottter.osm.overpass.time.sleep"),
        ):
            with pytest.raises(OverpassError, match="overloaded"):
                fetch_overpass(_BBOX, [], user_agent=_UA)

    def test_non_retryable_http_error_raises_immediately(self):
        """A 403 must be raised immediately without retrying."""
        http_err = _make_http_error(403)

        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            raise http_err

        with (
            patch(
                "plottter.osm.overpass.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
            patch("plottter.osm.overpass.time.sleep"),
        ):
            with pytest.raises(OverpassError, match="403"):
                fetch_overpass(_BBOX, [], user_agent=_UA)

        assert call_count == 1
