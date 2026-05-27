"""Tests for osm/geocode.py — Nominatim client (no live network calls)."""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import plottter.osm.geocode as geocode_mod
from plottter.osm.geocode import GeocodeError, GeocodeResult, geocode

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "osm" / "nominatim_kyoto.json"
)

_TEST_UA = "TestAgent/1.0 (pytest)"


def _make_response(payload: bytes) -> MagicMock:
    """Return a mock context-manager response that yields *payload*."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@pytest.fixture(autouse=True)
def reset_throttle():
    """Reset the module-level throttle before every test."""
    original = geocode_mod._last_call_time
    geocode_mod._last_call_time = 0.0
    yield
    geocode_mod._last_call_time = original


@pytest.fixture()
def kyoto_payload() -> bytes:
    return _FIXTURE_PATH.read_bytes()


# ---------------------------------------------------------------------------
# centre and bbox parsing
# ---------------------------------------------------------------------------


def test_geocode_returns_geocode_result(kyoto_payload):
    """geocode() must return a GeocodeResult for a non-empty response."""
    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep"):
            result = geocode("Kyoto, Japan", user_agent=_TEST_UA)

    assert isinstance(result, GeocodeResult)


def test_geocode_centre_lat_lon(kyoto_payload):
    """lat and lon must match the fixture values."""
    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep"):
            result = geocode("Kyoto, Japan", user_agent=_TEST_UA)

    assert result is not None
    assert abs(result.lat - 35.0116363) < 1e-6
    assert abs(result.lon - 135.7680294) < 1e-6


def test_geocode_display_name(kyoto_payload):
    """display_name must match the fixture value."""
    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep"):
            result = geocode("Kyoto, Japan", user_agent=_TEST_UA)

    assert result is not None
    assert result.display_name == "Kyoto, Kyoto Prefecture, Japan"


def test_geocode_bbox_tuple(kyoto_payload):
    """bbox must be a 4-tuple of floats in (south, north, west, east) order."""
    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep"):
            result = geocode("Kyoto, Japan", user_agent=_TEST_UA)

    assert result is not None
    south, north, west, east = result.bbox
    assert abs(south - 34.8891437) < 1e-6
    assert abs(north - 35.1328983) < 1e-6
    assert abs(west - 135.6137846) < 1e-6
    assert abs(east - 135.9195690) < 1e-6


# ---------------------------------------------------------------------------
# Empty result → None
# ---------------------------------------------------------------------------


def test_geocode_empty_array_returns_none():
    """An empty JSON array from Nominatim must produce None, not an error."""
    payload = b"[]"
    with patch("urllib.request.urlopen", return_value=_make_response(payload)):
        with patch("time.sleep"):
            result = geocode("xyzzy nowhere", user_agent=_TEST_UA)

    assert result is None


# ---------------------------------------------------------------------------
# HTTP error → GeocodeError
# ---------------------------------------------------------------------------


def test_geocode_http_error_raises_geocode_error():
    """An HTTPError from urlopen must be re-raised as GeocodeError."""
    http_err = urllib.error.HTTPError(
        url="https://nominatim.openstreetmap.org/search",
        code=403,
        msg="Forbidden",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with patch("time.sleep"):
            with pytest.raises(GeocodeError, match="403"):
                geocode("Kyoto, Japan", user_agent=_TEST_UA)


def test_geocode_url_error_raises_geocode_error():
    """A URLError (network failure) must also be re-raised as GeocodeError."""
    url_err = urllib.error.URLError(reason="Name or service not known")
    with patch("urllib.request.urlopen", side_effect=url_err):
        with patch("time.sleep"):
            with pytest.raises(GeocodeError):
                geocode("Kyoto, Japan", user_agent=_TEST_UA)


# ---------------------------------------------------------------------------
# Throttle: two rapid calls must trigger time.sleep
# ---------------------------------------------------------------------------


def test_throttle_second_rapid_call_sleeps(kyoto_payload):
    """A second call within <1 s of the first must invoke time.sleep."""
    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    # Set _last_call_time to "now" so the first geocode() call sees elapsed ≈ 0
    # and already triggers the throttle.  This guarantees sleep is called on
    # the very first call in this test (simulating a rapid second call in a
    # real session).
    geocode_mod._last_call_time = time.monotonic()

    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep", side_effect=fake_sleep):
            geocode("Kyoto, Japan", user_agent=_TEST_UA)

    assert len(sleep_calls) >= 1, "Expected time.sleep to be called for throttle"
    assert sleep_calls[0] > 0, "Sleep duration must be positive"


def test_throttle_sleep_duration_at_most_one_second(kyoto_payload):
    """The throttle sleep duration must never exceed _MIN_INTERVAL."""
    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    geocode_mod._last_call_time = time.monotonic()

    with patch("urllib.request.urlopen", return_value=_make_response(kyoto_payload)):
        with patch("time.sleep", side_effect=fake_sleep):
            geocode("Kyoto, Japan", user_agent=_TEST_UA)

    if sleep_calls:
        assert sleep_calls[0] <= geocode_mod._MIN_INTERVAL
