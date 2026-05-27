"""Map generator — converts a real-world location into pen-plotter line art.

See specs/map-generator.md for the full specification.
"""

from __future__ import annotations

from typing import Any

from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    LayerSpec,
    Parameter,
    Preset,
)
from plottter.generators import register_generator
from plottter.models import Canvas

# Type alias
Polyline = list[tuple[float, float]]



def _hatch_polygon(
    polygon: "Any",
    angle_deg: float,
    spacing_mm: float,
) -> "list[Polyline]":
    """Generate parallel hatch lines clipped to a Shapely polygon.

    Produces lines at *angle_deg* degrees (measured from the x-axis) with
    *spacing_mm* between adjacent lines.  The polygon may contain holes;
    ``shapely.intersection`` handles the exclusion of interior holes
    automatically — no separate clipping step is needed.

    Args:
        polygon:    A ``shapely.geometry.Polygon`` (possibly with interior
                    rings / holes).
        angle_deg:  Hatch angle in degrees from the positive x-axis.
        spacing_mm: Distance between adjacent hatch lines (mm).

    Returns:
        List of polylines, each a ``list[tuple[float, float]]`` in mm.
    """
    import math
    from shapely.geometry import LineString, MultiLineString, GeometryCollection

    if spacing_mm <= 0 or polygon.is_empty or polygon.area == 0:
        return []

    minx, miny, maxx, maxy = polygon.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Half-diagonal — lines extended this far from their centre point are
    # guaranteed to span the entire bounding box.
    half_diag = math.hypot(maxx - minx, maxy - miny) / 2.0 + spacing_mm

    # Project bounding-box corners onto the perpendicular axis
    # (perpendicular unit vector is (-sin_a, cos_a)).
    corners = [(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)]
    perp_vals = [-(x - cx) * sin_a + (y - cy) * cos_a for x, y in corners]
    perp_min = min(perp_vals)
    perp_max = max(perp_vals)

    polylines: list[Polyline] = []
    perp_pos = perp_min
    while perp_pos <= perp_max + spacing_mm * 0.5:
        # Centre of this scan line in canvas mm
        sx = cx - perp_pos * sin_a
        sy = cy + perp_pos * cos_a

        # Full scan line in the hatch direction (± half_diag from centre)
        line = LineString(
            [
                (sx - half_diag * cos_a, sy - half_diag * sin_a),
                (sx + half_diag * cos_a, sy + half_diag * sin_a),
            ]
        )

        clipped = polygon.intersection(line)

        if clipped.is_empty:
            perp_pos += spacing_mm
            continue

        if isinstance(clipped, LineString):
            coords = list(clipped.coords)
            if len(coords) >= 2:
                polylines.append([(float(x), float(y)) for x, y in coords])
        elif isinstance(clipped, MultiLineString):
            for seg in clipped.geoms:
                coords = list(seg.coords)
                if len(coords) >= 2:
                    polylines.append([(float(x), float(y)) for x, y in coords])
        elif isinstance(clipped, GeometryCollection):
            for geom in clipped.geoms:
                if isinstance(geom, LineString) and not geom.is_empty:
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        polylines.append([(float(x), float(y)) for x, y in coords])
                elif isinstance(geom, MultiLineString):
                    for seg in geom.geoms:
                        coords = list(seg.coords)
                        if len(coords) >= 2:
                            polylines.append(
                                [(float(x), float(y)) for x, y in coords]
                            )

        perp_pos += spacing_mm

    return polylines


def _render_attribution(canvas: "Canvas") -> "list[Polyline]":
    """Render '© OpenStreetMap contributors' in the bottom margin.

    Places a single line of Hershey Simplex text (~3 mm cap height) centred
    horizontally in the canvas and vertically within the bottom margin strip
    (between the drawing-area bottom edge and the canvas edge).

    Returns a list of polylines in canvas mm coordinates.
    """
    from plottter.generators._hershey import CAP_HEIGHT, glyph_strokes

    text = "\xa9 OpenStreetMap contributors"  # © = U+00A9 (falls back to ?)
    font_size_mm = 3.0
    scale = font_size_mm / CAP_HEIGHT

    # Measure total text width in mm.
    text_width = 0.0
    for ch in text:
        lft, rgt, _ = glyph_strokes(ch, "Simplex")
        text_width += (rgt - lft) * scale

    # Position: horizontally centred, baseline in the middle of the bottom margin.
    left_da, _top_da, right_da, bottom_da = canvas.drawing_area()
    cx = (left_da + right_da) / 2.0

    # Bottom margin centre Y (canvas y is downward)
    margin_cy = (bottom_da + canvas.height_mm) / 2.0
    # Baseline sits below the top of the cap: shift down by half cap height.
    baseline_y = margin_cy + font_size_mm / 2.0

    pen_x = cx - text_width / 2.0
    paths: list[Polyline] = []
    for ch in text:
        lft, rgt, strokes = glyph_strokes(ch, "Simplex")
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            polyline: Polyline = []
            for hx, hy in stroke:
                x_mm = pen_x + (hx - lft) * scale
                y_mm = baseline_y - hy * scale  # Hershey y is up; canvas y is down
                polyline.append((x_mm, y_mm))
            paths.append(polyline)
        pen_x += (rgt - lft) * scale

    return paths

@register_generator
class MapGenerator(Generator):
    """Multi-layer generator that renders OpenStreetMap data as pen-plotter art."""

    name: str = "Map"
    category: str = "map"
    emits_multiple_layers: bool = True

    def get_parameters(self) -> list[Parameter]:
        return [
            FloatParam(
                name="radius_km",
                label="Radius (km)",
                min=0.2,
                max=10.0,
                step=0.1,
                default=1.5,
                description=(
                    "Radius around the geocoded centre used to frame the map "
                    "(when extent_mode=radius)."
                ),
            ),
            ChoiceParam(
                name="extent_mode",
                label="Extent Mode",
                choices=["radius", "place_bbox"],
                default="radius",
                description=(
                    "radius: use radius_km square around the centre. "
                    "place_bbox: use the bounding box returned by Nominatim."
                ),
            ),
            ChoiceParam(
                name="road_detail",
                label="Road Detail",
                choices=["major_only", "standard", "all_streets"],
                default="standard",
                description=(
                    "major_only: motorway/trunk/primary/secondary only. "
                    "standard: adds tertiary, residential, unclassified. "
                    "all_streets: also includes service, track, paths, cycleways."
                ),
            ),
            BoolParam(
                name="include_roads",
                label="Include Roads",
                default=True,
                description="Draw road network layers.",
            ),
            BoolParam(
                name="include_rail",
                label="Include Rail",
                default=True,
                description="Draw rail/tram/subway ways.",
            ),
            BoolParam(
                name="include_water",
                label="Include Water (areas)",
                default=True,
                description="Draw lakes, ponds, and sea polygons.",
            ),
            BoolParam(
                name="include_waterways",
                label="Include Waterways (lines)",
                default=True,
                description="Draw rivers, streams, and canals.",
            ),
            BoolParam(
                name="include_parks",
                label="Include Parks / Green Space",
                default=True,
                description="Draw parks, gardens, forests, and other green areas.",
            ),
            BoolParam(
                name="include_buildings",
                label="Include Buildings",
                default=False,
                description=(
                    "Draw building footprints. "
                    "Off by default — dense urban areas can be very slow."
                ),
            ),
            BoolParam(
                name="include_coastline",
                label="Include Coastline",
                default=True,
                description="Draw coastline ways (useful for coastal locations).",
            ),
            ChoiceParam(
                name="area_fill",
                label="Area Fill",
                choices=["none", "hatch", "cross_hatch"],
                default="none",
                description=(
                    "none: outline only. "
                    "hatch: parallel fill lines. "
                    "cross_hatch: two perpendicular sets of fill lines."
                ),
            ),
            FloatParam(
                name="fill_spacing_mm",
                label="Fill Spacing (mm)",
                min=0.3,
                max=10.0,
                step=0.1,
                default=2.0,
                visible_when={"area_fill": ["hatch", "cross_hatch"]},
                description="Spacing between hatch lines in mm.",
            ),
            FloatParam(
                name="fill_angle_deg",
                label="Fill Angle (deg)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=45.0,
                visible_when={"area_fill": ["hatch", "cross_hatch"]},
                description="Angle of hatch lines in degrees.",
            ),
            IntParam(
                name="major_road_strokes",
                label="Major Road Strokes",
                min=1,
                max=4,
                step=1,
                default=1,
                description=(
                    "Number of parallel strokes for major roads. "
                    ">1 draws offset copies for visual emphasis."
                ),
            ),
            FloatParam(
                name="simplify_mm",
                label="Simplify Tolerance (mm)",
                min=0.0,
                max=2.0,
                step=0.01,
                default=0.15,
                description="Douglas–Peucker simplification tolerance in mm (0 = off).",
            ),
            FloatParam(
                name="min_feature_mm",
                label="Min Feature Length (mm)",
                min=0.0,
                max=10.0,
                step=0.1,
                default=0.8,
                description="Drop polyline fragments shorter than this length in mm.",
            ),
            BoolParam(
                name="include_attribution",
                label="Include Attribution",
                default=True,
                description=(
                    'Emit "© OpenStreetMap contributors" credit as required '
                    "by the ODbL licence."
                ),
            ),
        ]

    def get_presets(self) -> list[Preset]:
        """Return city-style map presets varying enabled categories and fill.

        Preset params only contain keys that appear in get_parameters() so
        they can be applied safely via the standard preset machinery.
        """
        return [
            Preset(
                name="Minimal Streets",
                description=(
                    "Major and minor road network only — clean, fast, and "
                    "well-suited for large-format plotting."
                ),
                params={
                    "road_detail": "standard",
                    "include_roads": True,
                    "include_rail": False,
                    "include_water": False,
                    "include_waterways": False,
                    "include_parks": False,
                    "include_buildings": False,
                    "include_coastline": False,
                    "area_fill": "none",
                },
            ),
            Preset(
                name="Roads + Water",
                description=(
                    "Road network plus water bodies and waterways. "
                    "Water areas are hatched for visual contrast."
                ),
                params={
                    "road_detail": "standard",
                    "include_roads": True,
                    "include_rail": False,
                    "include_water": True,
                    "include_waterways": True,
                    "include_parks": False,
                    "include_buildings": False,
                    "include_coastline": True,
                    "area_fill": "hatch",
                    "fill_spacing_mm": 2.0,
                    "fill_angle_deg": 45.0,
                },
            ),
            Preset(
                name="Full City",
                description=(
                    "All feature categories including buildings. "
                    "Best for dense urban areas; may be slow for large radii."
                ),
                params={
                    "road_detail": "standard",
                    "include_roads": True,
                    "include_rail": True,
                    "include_water": True,
                    "include_waterways": True,
                    "include_parks": True,
                    "include_buildings": True,
                    "include_coastline": True,
                    "area_fill": "none",
                },
            ),
            Preset(
                name="Transit Map",
                description=(
                    "Major roads and rail only, with major roads drawn as "
                    "double strokes for emphasis — a schematic transit-poster look."
                ),
                params={
                    "road_detail": "major_only",
                    "include_roads": True,
                    "include_rail": True,
                    "include_water": False,
                    "include_waterways": False,
                    "include_parks": False,
                    "include_buildings": False,
                    "include_coastline": False,
                    "area_fill": "none",
                    "major_road_strokes": 2,
                },
            ),
        ]

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Generate one LayerSpec per enabled, non-empty OSM category.

        Area categories (water, parks, buildings) are emitted first in draw
        order as closed-polyline outlines (area_fill="none", spec §7), then
        line categories (waterways, roads_minor, roads_major, rail, coastline)
        per spec §9.

        For area_fill="none" (default) each area feature becomes a closed
        polyline (first ≈ last).  Hatch fill is handled in a later phase.

        Returns an empty list when no map data has been fetched yet.

        Progress is reported in two phases:
          0–40%  — projection / fit-transform setup
          40–100% — per-category build, distributed evenly
        Cancellation is checked before each enabled category.
        """
        from plottter.osm.categories import FEATURE_CATEGORIES
        from plottter.osm.geometry import (
            assemble,
            clip_lines,
            clip_polygons,
            fit_transform,
            mercator,
            project_feature,
        )
        from plottter.processing.simplify import simplify_paths

        map_data = params.get("_map_data")
        if map_data is None:
            # CLI convenience (spec §13): when no pre-fetched data is present
            # but a `location` param is provided, fetch inline so the CLI path
            # works without a GUI cache-priming step.
            location: str = str(params.get("location", "")).strip()
            if not location:
                return []

            from plottter.osm import fetch_map_data as _fetch_map_data

            _radius_km = float(params.get("radius_km", 1.5))
            _extent_mode = str(params.get("extent_mode", "radius"))
            _road_detail = str(params.get("road_detail", "standard"))
            _endpoint = str(
                params.get(
                    "_overpass_endpoint",
                    "https://overpass-api.de/api/interpreter",
                )
            )

            # Derive enabled categories from include_* flags (mirrors
            # _MapMixin._get_enabled_map_categories in the GUI panel).
            _enabled_cats: list[str] = []
            if params.get("include_roads", True):
                _enabled_cats.append("roads_major")
                if _road_detail != "major_only":
                    _enabled_cats.append("roads_minor")
            if params.get("include_rail", True):
                _enabled_cats.append("rail")
            if params.get("include_water", True):
                _enabled_cats.append("water")
            if params.get("include_waterways", True):
                _enabled_cats.append("waterways")
            if params.get("include_parks", True):
                _enabled_cats.append("parks")
            if params.get("include_buildings", False):
                _enabled_cats.append("buildings")
            if params.get("include_coastline", True):
                _enabled_cats.append("coastline")

            map_data = _fetch_map_data(
                location,
                radius_km=_radius_km,
                extent_mode=_extent_mode,
                enabled_categories=_enabled_cats,
                road_detail=_road_detail,
                endpoint=_endpoint,
                user_agent="Plottter/cli (pen-plotter map art)",
                progress_callback=progress_callback,
            )

        simplify_mm: float = params.get("simplify_mm", 0.15)
        min_feature_mm: float = params.get("min_feature_mm", 0.8)
        road_detail: str = params.get("road_detail", "standard")
        area_fill: str = params.get("area_fill", "none")
        fill_spacing_mm: float = params.get("fill_spacing_mm", 2.0)
        fill_angle_deg: float = params.get("fill_angle_deg", 45.0)
        major_road_strokes: int = max(1, int(params.get("major_road_strokes", 1)))

        # Spec §9 draw order: areas first, then lines.
        _AREA_ORDER = [
            ("water", "include_water", "Water"),
            ("parks", "include_parks", "Parks"),
            ("buildings", "include_buildings", "Buildings"),
        ]
        _LINE_ORDER = [
            ("waterways", "include_waterways", "Waterways"),
            ("roads_minor", "include_roads", "Roads (minor)"),
            ("roads_major", "include_roads", "Roads (major)"),
            ("rail", "include_rail", "Rail"),
            ("coastline", "include_coastline", "Coastline"),
        ]

        # Build area config.
        area_config: list[tuple[str, bool, str]] = []
        for cat_id, include_param, display_name in _AREA_ORDER:
            enabled: bool = params.get(include_param, True)
            area_config.append((cat_id, enabled, display_name))

        # Build line config, honouring road_detail.
        line_config: list[tuple[str, bool, str]] = []
        for cat_id, include_param, display_name in _LINE_ORDER:
            if cat_id == "roads_minor" and road_detail == "major_only":
                continue
            enabled = params.get(include_param, True)
            line_config.append((cat_id, enabled, display_name))

        # Collect all enabled features (areas + lines) to compute a shared
        # fit transform that registers all layers perfectly (spec §6.2).
        all_features = []
        for cat_id, enabled, _ in (*area_config, *line_config):
            if enabled:
                all_features.extend(map_data.features.get(cat_id, []))

        if not all_features:
            return []

        transform = fit_transform(all_features, canvas)
        left, top, right, bottom = canvas.drawing_area()
        bbox_rect = (left, top, right, bottom)

        # Projection phase complete — report 40%.
        if progress_callback is not None:
            progress_callback(40)

        # Pre-count enabled categories that have actual feature data for
        # distributing the 40–100% range evenly across them.
        enabled_with_data = sum(
            1
            for cat_id, enabled, _ in (*area_config, *line_config)
            if enabled and map_data.features.get(cat_id)
        )
        n_cats = max(1, enabled_with_data)
        processed = 0

        specs: list[LayerSpec] = []

        # ------------------------------------------------------------------ #
        # Area categories (water, parks, buildings).
        # area_fill="none"        → outline only (closed polyline, spec §7).
        # area_fill="hatch"       → outline + parallel hatch lines.
        # area_fill="cross_hatch" → outline + two perpendicular hatch sets.
        # ------------------------------------------------------------------ #
        for cat_id, enabled, display_name in area_config:
            if not enabled:
                continue

            # Check for cancellation between categories (spec §9).
            if cancelled_callback is not None and cancelled_callback():
                return specs

            features = map_data.features.get(cat_id, [])
            if not features:
                continue

            # Project each area feature ring (and its holes) to canvas mm.
            # raw_rings     — outer rings fed to clip_polygons for outlines.
            # proj_features — (outer_ring_mm, [hole_rings_mm]) for hatch fill.
            raw_rings: list[list[tuple[float, float]]] = []
            proj_features: list[
                tuple[
                    list[tuple[float, float]],
                    list[list[tuple[float, float]]],
                ]
            ] = []
            for feature in features:
                rings, holes_latlon = assemble(feature)
                for ring in rings:
                    # Project outer ring (lat, lon) → canvas mm.
                    proj: list[tuple[float, float]] = []
                    for lat, lon in ring:
                        px, py = mercator(lat, lon)
                        cx = transform.x_origin + px * transform.scale
                        cy = transform.y_origin - py * transform.scale
                        proj.append((cx, cy))
                    # Ensure closure (first == last).
                    if len(proj) >= 2 and proj[0] != proj[-1]:
                        proj = proj + [proj[0]]
                    if len(proj) < 3:
                        continue
                    raw_rings.append(proj)
                    # Project inner-ring holes so hatch fill correctly omits
                    # them (shapely.intersection subtracts holes automatically).
                    proj_holes: list[list[tuple[float, float]]] = []
                    for hole in holes_latlon:
                        ph: list[tuple[float, float]] = []
                        for lat_h, lon_h in hole:
                            px_h, py_h = mercator(lat_h, lon_h)
                            cx_h = transform.x_origin + px_h * transform.scale
                            cy_h = transform.y_origin - py_h * transform.scale
                            ph.append((cx_h, cy_h))
                        if len(ph) >= 3:
                            proj_holes.append(ph)
                    proj_features.append((proj, proj_holes))

            # Clip outlines to the printable area; drop rings below threshold.
            clipped = clip_polygons(raw_rings, bbox_rect, min_feature_mm)

            # Generate hatch fill when requested (spec §7).
            if area_fill in ("hatch", "cross_hatch") and proj_features:
                from shapely.geometry import (
                    GeometryCollection as _GC,
                    MultiPolygon as _MP,
                    Polygon as _SP,
                    box as _shapely_box,
                )

                clip_box = _shapely_box(left, top, right, bottom)
                for outer_ring, holes in proj_features:
                    if len(outer_ring) < 3:
                        continue
                    try:
                        shp = _SP(outer_ring, holes)
                        if not shp.is_valid:
                            shp = shp.buffer(0)
                        clipped_shp = shp.intersection(clip_box)
                    except Exception:
                        continue
                    if clipped_shp.is_empty:
                        continue
                    if isinstance(clipped_shp, _SP):
                        polys = [clipped_shp]
                    elif isinstance(clipped_shp, _MP):
                        polys = list(clipped_shp.geoms)
                    elif isinstance(clipped_shp, _GC):
                        polys = [
                            g
                            for g in clipped_shp.geoms
                            if isinstance(g, _SP) and not g.is_empty
                        ]
                    else:
                        polys = []
                    for poly in polys:
                        if poly.is_empty or poly.area == 0:
                            continue
                        clipped.extend(
                            _hatch_polygon(poly, fill_angle_deg, fill_spacing_mm)
                        )
                        if area_fill == "cross_hatch":
                            clipped.extend(
                                _hatch_polygon(
                                    poly,
                                    (fill_angle_deg + 90.0) % 180.0,
                                    fill_spacing_mm,
                                )
                            )

            # Simplify with Douglas–Peucker (skip when tolerance is zero).
            if simplify_mm > 0.0 and clipped:
                clipped = simplify_paths(clipped, simplify_mm)

            if not clipped:
                continue

            color = FEATURE_CATEGORIES[cat_id]["color"]
            specs.append(LayerSpec(name=display_name, color=color, paths=clipped))

            processed += 1
            if progress_callback is not None:
                progress_callback(40 + int(60 * processed / n_cats))

        # ------------------------------------------------------------------ #
        # Line categories (waterways, roads, rail, coastline).
        # ------------------------------------------------------------------ #
        for cat_id, enabled, display_name in line_config:
            if not enabled:
                continue

            # Check for cancellation between categories (spec §9).
            if cancelled_callback is not None and cancelled_callback():
                return specs

            features = map_data.features.get(cat_id, [])
            if not features:
                continue

            # Project each feature's coords to canvas mm.
            raw_polylines = [project_feature(f, transform) for f in features]

            # Clip to the printable area; drop fragments shorter than threshold.
            clipped = clip_lines(raw_polylines, bbox_rect, min_feature_mm)

            # Simplify with Douglas–Peucker (skip when tolerance is zero).
            if simplify_mm > 0.0 and clipped:
                clipped = simplify_paths(clipped, simplify_mm)

            # Major-road multi-stroke emphasis (spec §8).
            # Thicken roads_major by adding symmetric offset copies on both
            # sides of the centreline.  offset_paths(sides="both", count=k)
            # produces original + k left + k right = 1+2k paths per input.
            # Use max(1, (n-1)//2) so any n>1 always generates extra strokes:
            #   n=2 → k=1 → 3×   n=3 → k=1 → 3×   n=4 → k=1 → 3×
            if cat_id == "roads_major" and major_road_strokes > 1 and clipped:
                from plottter.processing.offset import offset_paths

                clipped = offset_paths(
                    clipped,
                    distance_mm=0.5,
                    sides="both",
                    count=max(1, (major_road_strokes - 1) // 2),
                    join_style="round",
                    include_original=True,
                )

            if not clipped:
                continue

            color = FEATURE_CATEGORIES[cat_id]["color"]
            specs.append(LayerSpec(name=display_name, color=color, paths=clipped))

            processed += 1
            if progress_callback is not None:
                progress_callback(40 + int(60 * processed / n_cats))

        # Full run complete — ensure 100 is reported.
        if progress_callback is not None:
            progress_callback(100)

        # ------------------------------------------------------------------ #
        # Attribution layer (spec §12, required by ODbL).
        # ------------------------------------------------------------------ #
        include_attribution: bool = params.get("include_attribution", True)
        if include_attribution:
            attr_paths = _render_attribution(canvas)
            if attr_paths:
                specs.append(
                    LayerSpec(name="Attribution", color="#000000", paths=attr_paths)
                )

        return specs

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Single-layer fallback: flatten all enabled categories into one list.

        Calls generate_layers() and concatenates all LayerSpec paths into a
        single flat list of polylines, as required for CLI/legacy paths (§9).
        """
        specs = self.generate_layers(params, canvas, progress_callback, cancelled_callback)
        result: list[Polyline] = []
        for spec in specs:
            result.extend(spec.paths)
        return result
