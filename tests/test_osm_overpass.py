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
