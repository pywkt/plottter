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
        """Generate one LayerSpec per enabled, non-empty OSM category.

        Returns an empty list when no map data has been fetched yet
        (params["_map_data"] is None or absent).  The full projection,
        clipping, and fill logic is implemented in a subsequent phase.
        """
        map_data = params.get("_map_data")
        if map_data is None:
            return []

        # Full implementation is added in a later phase.
        # This skeleton returns an empty list for any non-None map_data
        # until the geometry phase is wired in.
        return []

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
