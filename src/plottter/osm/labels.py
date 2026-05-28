"""Label collection for OSM map features.

Provides the :class:`Label` dataclass and :func:`collect_water_labels` for
extracting place-name labels from a fetched :class:`~plottter.osm.types.MapData`
payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.osm.geometry import FitTransform
    from plottter.osm.types import MapData


@dataclass
class Label:
    """A text label to be drawn on the map.

    Attributes:
        text:             Display text resolved from OSM name tags.
        position:         Placement point as ``(x_mm, y_mm)`` canvas
                          coordinates (mm).
        priority:         Draw priority; higher values are drawn on top.
                          Water labels use priority 100 per spec §5.2.
        category:         Feature category id (e.g. ``"water"``).
        feature_size_mm:  Approximate linear size of the source feature
                          (√area in mm), suitable for font-size scaling.
    """

    text: str
    position: tuple[float, float]
    priority: int
    category: str
    feature_size_mm: float


def _resolve_name(tags: dict[str, str], language: str) -> str:
    """Resolve the display name from OSM tags.

    Lookup order: ``name:<language>`` → ``name``.  Returns an empty string
    when neither tag is present; the caller should skip features with an
    empty result.

    Args:
        tags:     OSM key→value tags for the feature.
        language: BCP-47-style language code (e.g. ``"en"``, ``"ja"``).

    Returns:
        Resolved name string, or ``""`` if no name is available.
    """
    lang_key = f"name:{language}"
    if lang_key in tags and tags[lang_key].strip():
        return tags[lang_key].strip()
    return tags.get("name", "").strip()


def collect_water_labels(
    map_data: "MapData",
    transform: "FitTransform",
    *,
    language: str,
    min_size_mm: float,
    clip_box_mm: tuple[float, float, float, float],
) -> list[Label]:
    """Collect labels for water-body features in *map_data*.

    For each feature in ``map_data.features["water"]``:

    1. Project ``(lat, lon)`` coordinates to canvas mm using *transform*.
    2. Clip the projected polygon to *clip_box_mm*; drop empty results.
    3. Drop features whose ``√area < min_size_mm``.
    4. Resolve the feature name via :func:`_resolve_name`; skip unnamed
       features.
    5. Place the label at the clipped polygon's ``representative_point()``
       (always lies strictly inside the polygon).

    Non-area features (``is_area=False``) in the water category are silently
    skipped — labels are only meaningful on polygon bodies.

    Args:
        map_data:     Fetched OSM payload from
                      :func:`plottter.osm.fetch_map_data`.
        transform:    Canvas projection transform from
                      :func:`plottter.osm.geometry.fit_transform`.
        language:     Preferred display language (BCP-47 code).
        min_size_mm:  Minimum feature size (``√area`` in mm) to label.
                      Features whose clipped area satisfies
                      ``√area < min_size_mm`` are skipped.
        clip_box_mm:  Clipping rectangle as ``(left, top, right, bottom)``
                      in mm.  Features entirely outside this rectangle are
                      dropped.

    Returns:
        List of :class:`Label` objects, one per qualifying named water
        feature, in the order they appear in *map_data*.
    """
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
    from shapely.geometry import box as shapely_box

    from plottter.osm.geometry import mercator

    left, top, right, bottom = clip_box_mm
    clip_box = shapely_box(left, top, right, bottom)

    water_features = map_data.features.get("water", [])
    labels: list[Label] = []

    for feature in water_features:
        if not feature.is_area or len(feature.coords) < 3:
            continue

        # --- project outer ring (lat, lon) → canvas mm -----------------
        outer: list[tuple[float, float]] = []
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            outer.append(
                (
                    transform.x_origin + px * transform.scale,
                    transform.y_origin - py * transform.scale,
                )
            )

        # --- project inner rings (holes) --------------------------------
        holes: list[list[tuple[float, float]]] = []
        for hole_coords in feature.inner_coords:
            if len(hole_coords) < 3:
                continue
            hole: list[tuple[float, float]] = []
            for lat, lon in hole_coords:
                px, py = mercator(lat, lon)
                hole.append(
                    (
                        transform.x_origin + px * transform.scale,
                        transform.y_origin - py * transform.scale,
                    )
                )
            holes.append(hole)

        # --- build shapely polygon and clip -----------------------------
        polygon = Polygon(outer, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(clip_box)
        if clipped.is_empty:
            continue

        # When the intersection is multi-part, keep the largest sub-polygon
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            candidates = [
                g
                for g in clipped.geoms
                if isinstance(g, Polygon) and not g.is_empty
            ]
            if not candidates:
                continue
            clipped = max(candidates, key=lambda g: g.area)

        if not isinstance(clipped, Polygon):
            continue

        # --- size filter ------------------------------------------------
        size_mm = math.sqrt(clipped.area)
        if size_mm < min_size_mm:
            continue

        # --- name resolution --------------------------------------------
        name = _resolve_name(feature.tags, language)
        if not name:
            continue

        # --- representative point (guaranteed inside) -------------------
        rp = clipped.representative_point()
        labels.append(
            Label(
                text=name,
                position=(rp.x, rp.y),
                priority=100,
                category="water",
                feature_size_mm=size_mm,
            )
        )

    return labels


def collect_park_labels(
    map_data: "MapData",
    transform: "FitTransform",
    *,
    language: str,
    min_size_mm: float,
    clip_box_mm: tuple[float, float, float, float],
) -> list[Label]:
    """Collect labels for park/green-space features in *map_data*.

    For each feature in ``map_data.features["parks"]``::

    1. Project ``(lat, lon)`` coordinates to canvas mm using *transform*.
    2. Clip the projected polygon to *clip_box_mm*; drop empty results.
    3. Drop features whose ``√area < min_size_mm``.
    4. Resolve the feature name via :func:`_resolve_name`; skip unnamed
       features.
    5. Place the label at the clipped polygon's ``representative_point()``
       (always lies strictly inside the polygon).

    Non-area features (``is_area=False``) in the parks category are silently
    skipped — labels are only meaningful on polygon bodies.

    Args:
        map_data:     Fetched OSM payload from
                      :func:`plottter.osm.fetch_map_data`.
        transform:    Canvas projection transform from
                      :func:`plottter.osm.geometry.fit_transform`.
        language:     Preferred display language (BCP-47 code).
        min_size_mm:  Minimum feature size (``√area`` in mm) to label.
                      Features whose clipped area satisfies
                      ``√area < min_size_mm`` are skipped.
        clip_box_mm:  Clipping rectangle as ``(left, top, right, bottom)``
                      in mm.  Features entirely outside this rectangle are
                      dropped.

    Returns:
        List of :class:`Label` objects, one per qualifying named park
        feature, in the order they appear in *map_data*.
    """
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
    from shapely.geometry import box as shapely_box

    from plottter.osm.geometry import mercator

    left, top, right, bottom = clip_box_mm
    clip_box = shapely_box(left, top, right, bottom)

    park_features = map_data.features.get("parks", [])
    labels: list[Label] = []

    for feature in park_features:
        if not feature.is_area or len(feature.coords) < 3:
            continue

        # --- project outer ring (lat, lon) → canvas mm -----------------
        outer: list[tuple[float, float]] = []
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            outer.append(
                (
                    transform.x_origin + px * transform.scale,
                    transform.y_origin - py * transform.scale,
                )
            )

        # --- project inner rings (holes) --------------------------------
        holes: list[list[tuple[float, float]]] = []
        for hole_coords in feature.inner_coords:
            if len(hole_coords) < 3:
                continue
            hole: list[tuple[float, float]] = []
            for lat, lon in hole_coords:
                px, py = mercator(lat, lon)
                hole.append(
                    (
                        transform.x_origin + px * transform.scale,
                        transform.y_origin - py * transform.scale,
                    )
                )
            holes.append(hole)

        # --- build shapely polygon and clip -----------------------------
        polygon = Polygon(outer, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(clip_box)
        if clipped.is_empty:
            continue

        # When the intersection is multi-part, keep the largest sub-polygon
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            candidates = [
                g
                for g in clipped.geoms
                if isinstance(g, Polygon) and not g.is_empty
            ]
            if not candidates:
                continue
            clipped = max(candidates, key=lambda g: g.area)

        if not isinstance(clipped, Polygon):
            continue

        # --- size filter ------------------------------------------------
        size_mm = math.sqrt(clipped.area)
        if size_mm < min_size_mm:
            continue

        # --- name resolution --------------------------------------------
        name = _resolve_name(feature.tags, language)
        if not name:
            continue

        # --- representative point (guaranteed inside) -------------------
        rp = clipped.representative_point()
        labels.append(
            Label(
                text=name,
                position=(rp.x, rp.y),
                priority=70,
                category="parks",
                feature_size_mm=size_mm,
            )
        )

    return labels
