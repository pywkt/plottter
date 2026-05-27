"""Feature taxonomy for the Map generator.

``FEATURE_CATEGORIES`` is the single source of truth mapping a category id to
its OSM selectors (Overpass tag-filter strings), default plot colour, and
geometry kind.  The generator's per-category enable flags and the Overpass
query are both derived from this table.

Selector strings are Overpass QL element-filter lines *without* the bbox
clause.  ``build_query`` in ``overpass.py`` appends ``(S,W,N,E);`` to each
one before assembling the full query body.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Road tier constants — kept here so phases can tune them without touching
# query-building logic.
# ---------------------------------------------------------------------------

#: Always included for ``roads_major`` regardless of road_detail.
ROAD_MAJOR_TYPES: list[str] = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
]

#: Included for ``roads_minor`` in ``standard`` and ``all_streets`` modes.
ROAD_MINOR_STANDARD_TYPES: list[str] = [
    "tertiary",
    "residential",
    "living_street",
    "unclassified",
]

#: Additional types added to ``roads_minor`` only when ``road_detail="all_streets"``.
ROAD_MINOR_EXTRA_TYPES: list[str] = [
    "service",
    "track",
    "footway",
    "path",
    "pedestrian",
    "cycleway",
]

# ---------------------------------------------------------------------------
# FEATURE_CATEGORIES
# ---------------------------------------------------------------------------

def _highway_filter(types: list[str]) -> str:
    return 'way["highway"~"^(' + "|".join(types) + ')$"]'


FEATURE_CATEGORIES: dict[str, dict] = {
    "roads_major": {
        "selectors": [_highway_filter(ROAD_MAJOR_TYPES)],
        "color": "#000000",
        "kind": "line",
    },
    "roads_minor": {
        # Standard tier; selectors_for_categories replaces these when
        # road_detail != "standard".
        "selectors": [_highway_filter(ROAD_MINOR_STANDARD_TYPES)],
        "color": "#444444",
        "kind": "line",
    },
    "rail": {
        "selectors": [
            'way["railway"~"^(rail|light_rail|subway|tram|monorail|narrow_gauge)$"]',
        ],
        "color": "#7A4A2B",
        "kind": "line",
    },
    "water": {
        "selectors": [
            'way["natural"="water"]',
            'relation["natural"="water"]',
        ],
        "color": "#1E6FD0",
        "kind": "area",
    },
    "waterways": {
        "selectors": [
            'way["waterway"~"^(river|stream|canal)$"]',
        ],
        "color": "#1E6FD0",
        "kind": "line",
    },
    "parks": {
        "selectors": [
            'way["leisure"~"^(park|garden|nature_reserve|recreation_ground)$"]',
            'way["landuse"~"^(forest|grass|meadow|cemetery|orchard)$"]',
            'way["natural"~"^(wood|scrub|grassland)$"]',
        ],
        "color": "#2E8B3D",
        "kind": "area",
    },
    "buildings": {
        "selectors": [
            'way["building"]',
            'relation["building"]',
        ],
        "color": "#8A6D3B",
        "kind": "area",
    },
    "coastline": {
        "selectors": [
            'way["natural"="coastline"]',
        ],
        "color": "#1E6FD0",
        "kind": "line",
    },
}

# Ensure the declaration order above is preserved as insertion order (Python
# 3.7+ dicts are ordered; this assertion makes the invariant explicit).
_EXPECTED_ORDER = [
    "roads_major",
    "roads_minor",
    "rail",
    "water",
    "waterways",
    "parks",
    "buildings",
    "coastline",
]
assert list(FEATURE_CATEGORIES.keys()) == _EXPECTED_ORDER, (
    "FEATURE_CATEGORIES key order changed; update _EXPECTED_ORDER."
)


# ---------------------------------------------------------------------------
# Selector derivation
# ---------------------------------------------------------------------------

def selectors_for_categories(
    enabled: Iterable[str],
    road_detail: str,
) -> list[str]:
    """Return Overpass tag-filter clauses for the given enabled categories.

    Parameters
    ----------
    enabled:
        Iterable of category ids (keys of ``FEATURE_CATEGORIES``) that are
        switched on.  Unknown ids are silently ignored.
    road_detail:
        One of ``"major_only"``, ``"standard"``, or ``"all_streets"``.

        * ``"major_only"``  — ``roads_minor`` is excluded entirely; only major
          road tiers are queried even if ``"roads_minor"`` is in *enabled*.
        * ``"standard"``    — standard minor-road tiers (tertiary, residential,
          living_street, unclassified).
        * ``"all_streets"`` — standard tiers **plus** service, track, footway,
          path, pedestrian, cycleway.

    Returns
    -------
    list[str]
        Selector strings in FEATURE_CATEGORIES declaration order, ready to be
        assembled into an Overpass QL query by ``overpass.build_query``.
    """
    enabled_set = set(enabled)
    result: list[str] = []

    for cat_id, cat in FEATURE_CATEGORIES.items():
        if cat_id not in enabled_set:
            continue

        if cat_id == "roads_minor":
            if road_detail == "major_only":
                # Minor roads are excluded at this detail level.
                continue
            elif road_detail == "all_streets":
                # Emit the standard tiers and the extra tiers as two separate
                # selectors.  Two selectors > one selector (standard mode) >
                # zero selectors (major_only), satisfying the count ordering
                # required by the spec and enforced by tests.
                result.append(_highway_filter(ROAD_MINOR_STANDARD_TYPES))
                result.append(_highway_filter(ROAD_MINOR_EXTRA_TYPES))
            else:
                # "standard" or any unrecognised value → use stored selectors.
                result.extend(cat["selectors"])
        else:
            result.extend(cat["selectors"])

    return result
