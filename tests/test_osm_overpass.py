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
