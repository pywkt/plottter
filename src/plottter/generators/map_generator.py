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
        # Presets are added in a later phase (148.1).
        return []

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Generate one LayerSpec per enabled, non-empty OSM line category.

        Line categories (roads_major, roads_minor, rail, waterways, coastline)
        are emitted in spec §9 draw order (areas are handled in a later phase).
        Returns an empty list when no map data has been fetched yet.
        """
        from plottter.osm.categories import FEATURE_CATEGORIES
        from plottter.osm.geometry import clip_lines, fit_transform, project_feature
        from plottter.processing.simplify import simplify_paths

        map_data = params.get("_map_data")
        if map_data is None:
            return []

        simplify_mm: float = params.get("simplify_mm", 0.15)
        min_feature_mm: float = params.get("min_feature_mm", 0.8)
        road_detail: str = params.get("road_detail", "standard")

        # Spec §9 draw order for line categories (areas come first in a later phase).
        # Each entry: (category_id, include_param_name, display_name)
        _LINE_ORDER = [
            ("waterways", "include_waterways", "Waterways"),
            ("roads_minor", "include_roads", "Roads (minor)"),
            ("roads_major", "include_roads", "Roads (major)"),
            ("rail", "include_rail", "Rail"),
            ("coastline", "include_coastline", "Coastline"),
        ]

        # Build the active list, honouring road_detail.
        category_config: list[tuple[str, bool, str]] = []
        for cat_id, include_param, display_name in _LINE_ORDER:
            if cat_id == "roads_minor" and road_detail == "major_only":
                continue
            enabled: bool = params.get(include_param, True)
            category_config.append((cat_id, enabled, display_name))

        # Collect all enabled features to compute a shared fit transform.
        all_features = []
        for cat_id, enabled, _ in category_config:
            if enabled:
                all_features.extend(map_data.features.get(cat_id, []))

        if not all_features:
            return []

        transform = fit_transform(all_features, canvas)
        left, top, right, bottom = canvas.drawing_area()
        bbox_rect = (left, top, right, bottom)

        specs: list[LayerSpec] = []
        for cat_id, enabled, display_name in category_config:
            if not enabled:
                continue

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

            if not clipped:
                continue

            color = FEATURE_CATEGORIES[cat_id]["color"]
            specs.append(LayerSpec(name=display_name, color=color, paths=clipped))

        if progress_callback is not None:
            progress_callback(100)

        return specs

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Single-layer fallback: flatten all enabled categories into one list.

        Full implementation follows in a later phase.
        """
        return []
