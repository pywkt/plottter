"""Tests for osm/categories.py — feature taxonomy and selector derivation."""

import re

import pytest

from plottter.osm.categories import (
    FEATURE_CATEGORIES,
    selectors_for_categories,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_CATS = list(FEATURE_CATEGORIES.keys())
ALL_ENABLED = set(ALL_CATS)


def _is_bracketed_tag_filter(selector: str) -> bool:
    """Return True if *selector* contains at least one bracketed tag filter."""
    return bool(re.search(r"\[.+\]", selector))


# ---------------------------------------------------------------------------
# FEATURE_CATEGORIES structure
# ---------------------------------------------------------------------------

def test_all_eight_categories_present():
    expected = {
        "roads_major", "roads_minor", "rail", "water",
        "waterways", "parks", "buildings", "coastline",
    }
    assert set(ALL_CATS) == expected


def test_category_keys_ordered():
    """Declaration order matches the spec table (roads first, coastline last)."""
    assert ALL_CATS[0] == "roads_major"
    assert ALL_CATS[-1] == "coastline"


def test_each_category_has_required_fields():
    for cat_id, cat in FEATURE_CATEGORIES.items():
        assert "selectors" in cat, f"{cat_id}: missing 'selectors'"
        assert "color" in cat, f"{cat_id}: missing 'color'"
        assert "kind" in cat, f"{cat_id}: missing 'kind'"


def test_each_selector_is_non_empty_bracketed_tag_filter():
    for cat_id, cat in FEATURE_CATEGORIES.items():
        assert cat["selectors"], f"{cat_id}: selectors list is empty"
        for sel in cat["selectors"]:
            assert sel, f"{cat_id}: empty selector string"
            assert _is_bracketed_tag_filter(sel), (
                f"{cat_id}: selector {sel!r} has no bracketed tag filter"
            )


def test_colors_are_hex():
    hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for cat_id, cat in FEATURE_CATEGORIES.items():
        assert hex_re.match(cat["color"]), (
            f"{cat_id}: color {cat['color']!r} is not a 6-digit hex colour"
        )


def test_kind_values():
    valid = {"line", "area"}
    for cat_id, cat in FEATURE_CATEGORIES.items():
        assert cat["kind"] in valid, (
            f"{cat_id}: kind {cat['kind']!r} is not 'line' or 'area'"
        )


def test_area_categories():
    areas = {k for k, v in FEATURE_CATEGORIES.items() if v["kind"] == "area"}
    assert areas == {"water", "parks", "buildings"}


def test_line_categories():
    lines = {k for k, v in FEATURE_CATEGORIES.items() if v["kind"] == "line"}
    assert lines == {"roads_major", "roads_minor", "rail", "waterways", "coastline"}


# ---------------------------------------------------------------------------
# selectors_for_categories — road_detail ordering
# ---------------------------------------------------------------------------

def test_all_streets_more_clauses_than_standard_than_major_only():
    all_s = selectors_for_categories(ALL_ENABLED, "all_streets")
    standard = selectors_for_categories(ALL_ENABLED, "standard")
    major = selectors_for_categories(ALL_ENABLED, "major_only")

    assert len(all_s) > len(standard), (
        "all_streets should yield more selectors than standard"
    )
    assert len(standard) > len(major), (
        "standard should yield more selectors than major_only"
    )


def test_major_only_excludes_roads_minor():
    selectors = selectors_for_categories(ALL_ENABLED, "major_only")
    # roads_minor tiers must not appear
    minor_types = ["tertiary", "residential", "living_street", "unclassified",
                   "service", "track", "footway", "path", "pedestrian", "cycleway"]
    combined = " ".join(selectors)
    for t in minor_types:
        assert t not in combined, (
            f"major_only should not include minor road type {t!r}"
        )


def test_standard_includes_minor_standard_tiers():
    selectors = selectors_for_categories(ALL_ENABLED, "standard")
    combined = " ".join(selectors)
    for t in ["tertiary", "residential", "living_street", "unclassified"]:
        assert t in combined, (
            f"standard should include roads_minor type {t!r}"
        )


def test_standard_excludes_extra_street_types():
    selectors = selectors_for_categories(ALL_ENABLED, "standard")
    combined = " ".join(selectors)
    for t in ["service", "track", "footway", "path", "pedestrian", "cycleway"]:
        assert t not in combined, (
            f"standard should not include extra street type {t!r}"
        )


def test_all_streets_includes_extra_types():
    selectors = selectors_for_categories(ALL_ENABLED, "all_streets")
    combined = " ".join(selectors)
    for t in ["service", "track", "footway", "path", "pedestrian", "cycleway"]:
        assert t in combined, (
            f"all_streets should include extra type {t!r}"
        )


def test_all_streets_also_includes_standard_minor_tiers():
    selectors = selectors_for_categories(ALL_ENABLED, "all_streets")
    combined = " ".join(selectors)
    for t in ["tertiary", "residential", "living_street", "unclassified"]:
        assert t in combined, (
            f"all_streets should still include standard minor type {t!r}"
        )


# ---------------------------------------------------------------------------
# selectors_for_categories — disabled categories contribute nothing
# ---------------------------------------------------------------------------

def test_disabled_categories_contribute_no_selectors():
    # Disable everything → empty list
    selectors = selectors_for_categories([], "standard")
    assert selectors == []


def test_disabled_single_category_absent():
    # Disable buildings specifically
    enabled = ALL_ENABLED - {"buildings"}
    all_s = selectors_for_categories(ALL_ENABLED, "standard")
    without_buildings = selectors_for_categories(enabled, "standard")

    # Every buildings selector must be absent when buildings is disabled
    buildings_selectors = set(FEATURE_CATEGORIES["buildings"]["selectors"])
    for sel in without_buildings:
        assert sel not in buildings_selectors, (
            f"disabled category 'buildings' contributed selector {sel!r}"
        )
    assert len(without_buildings) < len(all_s)


def test_only_rail_enabled():
    selectors = selectors_for_categories({"rail"}, "standard")
    assert selectors == FEATURE_CATEGORIES["rail"]["selectors"]


def test_only_water_enabled():
    selectors = selectors_for_categories({"water"}, "standard")
    assert selectors == FEATURE_CATEGORIES["water"]["selectors"]


def test_roads_minor_disabled_roads_major_still_included():
    # roads_major and roads_minor are separate category ids
    selectors = selectors_for_categories({"roads_major"}, "all_streets")
    assert len(selectors) == 1
    combined = selectors[0]
    assert "motorway" in combined
    # Minor types must not appear
    assert "tertiary" not in combined
    assert "service" not in combined


# ---------------------------------------------------------------------------
# selector format sanity
# ---------------------------------------------------------------------------

def test_all_returned_selectors_are_bracketed_tag_filters():
    for detail in ("major_only", "standard", "all_streets"):
        selectors = selectors_for_categories(ALL_ENABLED, detail)
        for sel in selectors:
            assert _is_bracketed_tag_filter(sel), (
                f"selector {sel!r} returned for road_detail={detail!r} "
                "has no bracketed tag filter"
            )


def test_selectors_start_with_way_or_relation():
    for detail in ("major_only", "standard", "all_streets"):
        selectors = selectors_for_categories(ALL_ENABLED, detail)
        for sel in selectors:
            assert sel.startswith("way[") or sel.startswith("relation["), (
                f"selector {sel!r} does not start with 'way[' or 'relation['"
            )
