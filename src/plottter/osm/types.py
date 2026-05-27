"""MapFeature and MapData dataclasses — the boundary between FETCH and GENERATE.

These are dependency-free (stdlib only) so the module can be imported in any
context (tests, CLI, headless) without pulling in GUI or network code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MapFeature:
    """A single OSM geometric feature (way or relation member).

    Attributes:
        tags:    OSM key→value tags for this element.
        coords:  Ordered sequence of (lat, lon) pairs in WGS84 degrees.
        is_area: True for closed ways / polygons (buildings, parks, water
                 bodies); False for open lines (roads, rail, waterways).
    """

    tags: dict[str, str]
    coords: list[tuple[float, float]]  # (lat, lon), in order
    is_area: bool  # closed polygon vs open line


@dataclass
class MapData:
    """The complete fetched payload for one location + extent.

    This is the value stored in the disk cache (§11) and injected into
    the generator as ``params["_map_data"]``.  Everything downstream of
    the FETCH step consumes only this object — no network calls needed.

    Attributes:
        location:    The original query string supplied by the user.
        center:      Geocoded centre as (lat, lon) in WGS84 degrees.
        bbox:        Geographic bounding box as (south, west, north, east)
                     in WGS84 degrees (Overpass ordering).
        features:    Mapping from category id (e.g. ``"roads_major"``) to
                     the list of :class:`MapFeature` objects in that category.
        attribution: License attribution string; must be included in any
                     output that uses OSM data (ODbL requirement).
    """

    location: str
    center: tuple[float, float]  # (lat, lon)
    bbox: tuple[float, float, float, float]  # (south, west, north, east)
    features: dict[str, list[MapFeature]]
    attribution: str = "© OpenStreetMap contributors"

    # ------------------------------------------------------------------
    # JSON round-trip (used by osm/cache.py — §11)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Return a plain JSON-serialisable dict.

        Coords are stored as nested lists (JSON has no tuple type).
        Tags are plain dicts.  All other fields are primitives.
        """
        return {
            "location": self.location,
            "center": list(self.center),
            "bbox": list(self.bbox),
            "features": {
                category: [
                    {
                        "tags": dict(feat.tags),
                        "coords": [list(c) for c in feat.coords],
                        "is_area": feat.is_area,
                    }
                    for feat in feats
                ]
                for category, feats in self.features.items()
            },
            "attribution": self.attribution,
        }

    @classmethod
    def from_json(cls, d: dict) -> "MapData":
        """Reconstruct a :class:`MapData` from the dict produced by :meth:`to_json`."""
        features: dict[str, list[MapFeature]] = {}
        for category, raw_feats in d["features"].items():
            feats: list[MapFeature] = []
            for rf in raw_feats:
                feats.append(
                    MapFeature(
                        tags=dict(rf["tags"]),
                        coords=[tuple(c) for c in rf["coords"]],
                        is_area=bool(rf["is_area"]),
                    )
                )
            features[category] = feats

        return cls(
            location=d["location"],
            center=tuple(d["center"]),
            bbox=tuple(d["bbox"]),
            features=features,
            attribution=d.get("attribution", "© OpenStreetMap contributors"),
        )
