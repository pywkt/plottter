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


def collect_place_labels(
    map_data: "MapData",
    transform: "FitTransform",
    *,
    language: str,
    min_size_mm: float,
    clip_box_mm: tuple[float, float, float, float],
) -> list[Label]:
    """Collect labels for place features (islands, islets, neighbourhoods, suburbs).

    Handles two geometry kinds produced by the ``places`` Overpass category:

    * **Point features** (``len(coords) == 1``): a single-coord node queried
      for ``place=island|islet|neighbourhood|suburb``.  The node's location is
      projected directly; no size filter is applied (``feature_size_mm=0``).
    * **Area features** (``is_area=True``): ways or relation members with a
      polygon outline.  The representative point of the clipped polygon is
      used, same as :func:`collect_water_labels`.

    Priority by place tag value (spec §5.2):

    * ``island`` / ``islet`` → 90
    * ``neighbourhood`` / ``suburb`` → 80

    Args:
        map_data:     Fetched OSM payload from
                      :func:`plottter.osm.fetch_map_data`.
        transform:    Canvas projection transform from
                      :func:`plottter.osm.geometry.fit_transform`.
        language:     Preferred display language (BCP-47 code).
        min_size_mm:  Minimum feature size (``√area`` in mm) to label.
                      Applied only to area features; point features are
                      always included if inside the clip box.
        clip_box_mm:  Clipping rectangle as ``(left, top, right, bottom)``
                      in mm.  Features entirely outside this rectangle are
                      dropped.

    Returns:
        List of :class:`Label` objects, one per qualifying named place
        feature, in the order they appear in *map_data*.
    """
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
    from shapely.geometry import box as shapely_box

    from plottter.osm.geometry import mercator

    _PRIORITY: dict[str, int] = {
        "island": 90,
        "islet": 90,
        "neighbourhood": 80,
        "suburb": 80,
    }

    left, top, right, bottom = clip_box_mm
    clip_box = shapely_box(left, top, right, bottom)

    place_features = map_data.features.get("places", [])
    labels: list[Label] = []

    for feature in place_features:
        name = _resolve_name(feature.tags, language)
        if not name:
            continue

        place_type = feature.tags.get("place", "")
        priority = _PRIORITY.get(place_type, 80)

        # --- point feature (node): single coord -------------------------
        if len(feature.coords) == 1:
            lat, lon = feature.coords[0]
            px, py = mercator(lat, lon)
            x_mm = transform.x_origin + px * transform.scale
            y_mm = transform.y_origin - py * transform.scale

            if not clip_box.contains(Point(x_mm, y_mm)):
                continue

            labels.append(
                Label(
                    text=name,
                    position=(x_mm, y_mm),
                    priority=priority,
                    category="place",
                    feature_size_mm=0.0,
                )
            )
            continue

        # --- area feature (way/relation): polygon centroid --------------
        if not feature.is_area or len(feature.coords) < 3:
            continue

        # project outer ring (lat, lon) → canvas mm
        outer: list[tuple[float, float]] = []
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            outer.append(
                (
                    transform.x_origin + px * transform.scale,
                    transform.y_origin - py * transform.scale,
                )
            )

        # project inner rings (holes)
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

        # build shapely polygon and clip
        polygon = Polygon(outer, holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(clip_box)
        if clipped.is_empty:
            continue

        # keep the largest sub-polygon when the intersection is multi-part
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

        # size filter
        size_mm = math.sqrt(clipped.area)
        if size_mm < min_size_mm:
            continue

        # representative point (guaranteed inside)
        rp = clipped.representative_point()
        labels.append(
            Label(
                text=name,
                position=(rp.x, rp.y),
                priority=priority,
                category="place",
                feature_size_mm=size_mm,
            )
        )

    return labels


def collect_waterway_labels(
    map_data: "MapData",
    transform: "FitTransform",
    *,
    language: str,
    clip_box_mm: tuple[float, float, float, float],
) -> list[Label]:
    """Collect labels for linear waterway features in *map_data*.

    For each feature in ``map_data.features["waterways"]``:

    1. Project ``(lat, lon)`` coordinates to canvas mm using *transform*.
    2. Clip the projected line to *clip_box_mm*; drop empty results.
    3. When the clipped result is a multi-part geometry, find the longest
       contiguous segment.
    4. Resolve the feature name via :func:`_resolve_name`; skip unnamed
       features.
    5. Place the label at ``longest_segment.interpolate(length * 0.5)``
       (midpoint of the longest contiguous clipped segment).

    Area features (``is_area=True``) are silently skipped -- waterways are
    linear in OSM.

    Args:
        map_data:    Fetched OSM payload from
                     :func:`plottter.osm.fetch_map_data`.
        transform:   Canvas projection transform from
                     :func:`plottter.osm.geometry.fit_transform`.
        language:    Preferred display language (BCP-47 code).
        clip_box_mm: Clipping rectangle as ``(left, top, right, bottom)``
                     in mm.  Features entirely outside this rectangle are
                     dropped.

    Returns:
        List of :class:`Label` objects, one per qualifying named waterway
        feature, in the order they appear in *map_data*.
    """
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import box as shapely_box

    from plottter.osm.geometry import mercator

    left, top, right, bottom = clip_box_mm
    clip_box = shapely_box(left, top, right, bottom)

    waterway_features = map_data.features.get("waterways", [])
    labels: list[Label] = []

    for feature in waterway_features:
        if feature.is_area or len(feature.coords) < 2:
            continue

        # --- resolve name; skip unnamed ---------------------------------
        name = _resolve_name(feature.tags, language)
        if not name:
            continue

        # --- project (lat, lon) -> canvas mm ----------------------------
        projected: list[tuple[float, float]] = []
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            projected.append(
                (
                    transform.x_origin + px * transform.scale,
                    transform.y_origin - py * transform.scale,
                )
            )

        line = LineString(projected)
        clipped = line.intersection(clip_box)

        if clipped.is_empty:
            continue

        # --- find the longest contiguous segment -----------------------
        if isinstance(clipped, MultiLineString):
            longest = max(clipped.geoms, key=lambda g: g.length)
        else:
            longest = clipped

        if longest.length == 0:
            continue

        # --- midpoint at 50% along the longest segment -----------------
        midpoint = longest.interpolate(longest.length * 0.5)
        labels.append(
            Label(
                text=name,
                position=(midpoint.x, midpoint.y),
                priority=60,
                category="waterways",
                feature_size_mm=longest.length,
            )
        )

    return labels


def collect_road_labels(
    map_data: "MapData",
    transform: "FitTransform",
    *,
    language: str,
    clip_box_mm: tuple[float, float, float, float],
) -> list[Label]:
    """Collect labels for major road features in *map_data*.

    Groups ``map_data.features["roads_major"]`` by ``tags["name"]``, stitches
    all segments of the same road name into connected runs via
    ``shapely.ops.linemerge``, picks the longest run, and places a single
    label at the midpoint of that run.

    Algorithm per road name:

    1. Project each segment's ``(lat, lon)`` coordinates to canvas mm.
    2. Clip each projected segment to *clip_box_mm*; discard empty results.
    3. Stitch the clipped segments with :func:`shapely.ops.linemerge` --
       segments sharing endpoints are joined into a single LineString.
    4. When multiple disconnected runs remain, pick the longest.
    5. Place the label at ``longest_run.interpolate(run.length * 0.5)``.

    Unnamed roads (no ``name`` tag) are silently skipped.  Exactly one
    :class:`Label` is emitted per distinct road name.

    Args:
        map_data:    Fetched OSM payload from
                     :func:`plottter.osm.fetch_map_data`.
        transform:   Canvas projection transform from
                     :func:`plottter.osm.geometry.fit_transform`.
        language:    Preferred display language (BCP-47 code).
        clip_box_mm: Clipping rectangle as ``(left, top, right, bottom)``
                     in mm.  Segments entirely outside this rectangle are
                     dropped.

    Returns:
        List of :class:`Label` objects, one per distinct named road, ordered
        by first occurrence of the road name in *map_data*.
    """
    from collections import defaultdict

    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import box as shapely_box
    from shapely.ops import linemerge

    from plottter.osm.geometry import mercator

    left, top, right, bottom = clip_box_mm
    clip_box = shapely_box(left, top, right, bottom)

    road_features = map_data.features.get("roads_major", [])

    # --- group segments by base road name (insertion-order preserved) ---
    name_order: list[str] = []
    name_to_features: dict[str, list] = defaultdict(list)
    for feature in road_features:
        road_name = feature.tags.get("name", "").strip()
        if not road_name:
            continue
        if road_name not in name_to_features:
            name_order.append(road_name)
        name_to_features[road_name].append(feature)

    labels: list[Label] = []

    for road_name in name_order:
        features = name_to_features[road_name]

        # --- project + clip each segment --------------------------------
        clipped_segments = []
        for feature in features:
            if len(feature.coords) < 2:
                continue

            projected: list[tuple[float, float]] = []
            for lat, lon in feature.coords:
                px, py = mercator(lat, lon)
                projected.append(
                    (
                        transform.x_origin + px * transform.scale,
                        transform.y_origin - py * transform.scale,
                    )
                )

            seg = LineString(projected)
            clipped = seg.intersection(clip_box)

            if clipped.is_empty:
                continue

            # Flatten multi-part intersection results
            if isinstance(clipped, MultiLineString):
                clipped_segments.extend(clipped.geoms)
            else:
                clipped_segments.append(clipped)

        if not clipped_segments:
            continue

        # --- stitch connected segments into runs ------------------------
        if len(clipped_segments) == 1:
            merged = clipped_segments[0]
        else:
            merged = linemerge(clipped_segments)

        # --- pick the longest run ---------------------------------------
        if isinstance(merged, MultiLineString):
            longest = max(merged.geoms, key=lambda g: g.length)
        else:
            longest = merged

        if longest.length == 0:
            continue

        # --- resolve display name (language preference) -----------------
        display_name = _resolve_name(features[0].tags, language) or road_name

        # --- midpoint at 50% along the longest run ----------------------
        midpoint = longest.interpolate(longest.length * 0.5)
        labels.append(
            Label(
                text=display_name,
                position=(midpoint.x, midpoint.y),
                priority=50,
                category="roads",
                feature_size_mm=longest.length,
            )
        )

    return labels


#: Default Hershey font for OSM map labels.  Chosen for legibility at the
#: small cap heights used on plotted maps (typically 2.5–4 mm).
DEFAULT_LABEL_FONT: str = "EMSReadability"


def _hershey_text_width(
    text: str,
    font_size_mm: float,
    font: str = DEFAULT_LABEL_FONT,
) -> float:
    """Return the rendered width of *text* in mm at *font_size_mm* cap height.

    Uses the named Hershey font's advance widths, scaled so the cap height
    equals *font_size_mm*.  No letter-spacing is added.
    """
    from plottter.fonts.hershey import CAP_HEIGHT, glyph_strokes

    if not text:
        return 0.0
    scale = font_size_mm / CAP_HEIGHT
    width = 0.0
    for ch in text:
        left, right, _ = glyph_strokes(ch, font)
        width += (right - left) * scale
    return width


def place_with_collision(
    labels: list[Label],
    font_size_mm: float,
    clip_box_mm: tuple[float, float, float, float],
    font: str = DEFAULT_LABEL_FONT,
) -> list[Label]:
    """Place labels with axis-aligned bbox collision detection.

    Implements the greedy placement algorithm from spec §6.2:

    1. Compute each label's axis-aligned bounding box (bbox) per §6.1
       using Hershey-measured text width, cap-height, and 0.25·font_size_mm
       padding on all four sides.
    2. Drop labels whose bbox falls (even partially) outside *clip_box_mm*.
    3. Sort by ``(priority desc, feature_size_mm desc, text)`` so that
       high-priority and large features win tie-breaks deterministically.
    4. Greedy accept: keep a label if its bbox does not intersect any
       already-accepted label's bbox.
    5. Return accepted labels sorted by ``(category, text)`` for a stable,
       deterministic plotting order.

    Args:
        labels:       Candidate labels from one or more ``collect_*`` calls.
        font_size_mm: Rendered font cap height in mm.  Governs both the bbox
                      height and the padding amount.
        clip_box_mm:  Printable area as ``(left, top, right, bottom)`` in mm.
                      Labels with any bbox corner outside this rectangle are
                      discarded before collision testing.

    Returns:
        Accepted labels in ``(category, text)`` order.
    """
    pad = font_size_mm * 0.25
    cl, ct, cr, cb = clip_box_mm

    def _bbox(label: Label) -> tuple[float, float, float, float]:
        x, y = label.position
        w = _hershey_text_width(label.text, font_size_mm, font)
        h = font_size_mm
        return (
            x - w / 2 - pad,
            y - h / 2 - pad,
            x + w / 2 + pad,
            y + h / 2 + pad,
        )

    def _overlaps(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        """Return True if the two axis-aligned bboxes intersect."""
        al, at, ar, ab = a
        bl, bt, br, bb = b
        return al < br and ar > bl and at < bb and ab > bt

    # 1. Compute bboxes and drop labels whose bbox falls outside clip_box_mm
    candidates: list[tuple[Label, tuple[float, float, float, float]]] = []
    for label in labels:
        bbox = _bbox(label)
        bl, bt, br, bb = bbox
        if bl < cl or bt < ct or br > cr or bb > cb:
            continue
        candidates.append((label, bbox))

    # 2. Sort: priority desc, feature_size_mm desc, text asc (deterministic)
    candidates.sort(
        key=lambda item: (-item[0].priority, -item[0].feature_size_mm, item[0].text)
    )

    # 3. Greedy placement: accept if no overlap with already-accepted bboxes
    accepted_bboxes: list[tuple[float, float, float, float]] = []
    accepted: list[Label] = []
    for label, bbox in candidates:
        if any(_overlaps(bbox, other) for other in accepted_bboxes):
            continue
        accepted_bboxes.append(bbox)
        accepted.append(label)

    # 4. Return in stable (category, text) order for deterministic plotting
    accepted.sort(key=lambda lb: (lb.category, lb.text))
    return accepted


def render_labels(
    labels: list[Label],
    font_size_mm: float,
    font: str = DEFAULT_LABEL_FONT,
) -> list:
    """Generate Hershey strokes for each label centred at ``label.position``.

    For each :class:`Label` in *labels*:

    * Measures the total rendered width using :func:`_hershey_text_width`.
    * Places the left edge of the baseline at
      ``(label.position[0] - width/2, label.position[1] + font_size_mm/2)``.
    * Emits strokes from the named single-stroke font, scaled so cap height
      equals *font_size_mm*, with the font's y-up coordinate flipped to
      canvas y-down.

    Missing glyphs fall back to ``"?"`` (handled by :func:`glyph_strokes`).

    Args:
        labels:       Placed labels (typically the output of
                      :func:`place_with_collision`).
        font_size_mm: Cap height of the rendered text in mm.
        font:         Canonical Hershey-font name from
                      :mod:`plottter.fonts.hershey`.  Defaults to
                      :data:`DEFAULT_LABEL_FONT` (``"EMSReadability"``),
                      chosen for legibility at small label sizes.

    Returns:
        A flat ``list[Polyline]`` in canvas mm coordinates.
    """
    from plottter.fonts.hershey import CAP_HEIGHT, glyph_strokes

    scale = font_size_mm / CAP_HEIGHT
    paths: list = []

    for label in labels:
        text = label.text
        width = _hershey_text_width(text, font_size_mm, font)
        pen_x = label.position[0] - width / 2.0
        baseline_y = label.position[1] + font_size_mm / 2.0

        for ch in text:
            lft, rgt, strokes = glyph_strokes(ch, font)
            for stroke in strokes:
                if len(stroke) < 2:
                    continue
                polyline = []
                for hx, hy in stroke:
                    x_mm = pen_x + (hx - lft) * scale
                    y_mm = baseline_y - hy * scale  # Hershey y-up → canvas y-down
                    polyline.append((x_mm, y_mm))
                paths.append(polyline)
            pen_x += (rgt - lft) * scale

    return paths
