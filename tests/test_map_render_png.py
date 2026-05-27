"""Phase 148.3 — Offline map render to docs/images/map_example.png.

Loads ``tests/fixtures/osm/overpass_small.json`` (committed fixture, no
network) into a ``MapData`` object, runs ``MapGenerator.generate_layers()``
and renders all LayerSpecs to a coloured PNG saved at
``docs/images/map_example.png``.

Invariants verified:
- ``docs/images/map_example.png`` is created and non-empty (> 0 bytes).
- The file is a valid PNG (magic bytes ``\\x89PNG``).
- At least one layer is generated from the fixture data.
- The rendered image has non-trivial content (not a blank white rectangle).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "osm" / "overpass_small.json"
_OUT_DIR = _REPO_ROOT / "docs" / "images"
_OUT_PNG = _OUT_DIR / "map_example.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_map_data():
    """Parse the Overpass fixture and return a categorised ``MapData``."""
    from plottter.osm.categories import (
        ROAD_MAJOR_TYPES,
        ROAD_MINOR_STANDARD_TYPES,
    )
    from plottter.osm.overpass import _parse_elements
    from plottter.osm.types import MapData

    raw = json.loads(_FIXTURE.read_text())
    all_features = _parse_elements(raw["elements"])

    # Manually bucket each feature into the matching OSM category based on tags,
    # mirroring what fetch_map_data() does (one Overpass call per category).
    features: dict[str, list] = {}
    for feat in all_features:
        t = feat.tags
        hw = t.get("highway", "")
        nat = t.get("natural", "")
        ww = t.get("waterway", "")
        lei = t.get("leisure", "")
        lu = t.get("landuse", "")
        ry = t.get("railway", "")

        if hw in ROAD_MAJOR_TYPES:
            cat = "roads_major"
        elif hw in ROAD_MINOR_STANDARD_TYPES:
            cat = "roads_minor"
        elif ry in ("rail", "light_rail", "subway", "tram", "monorail", "narrow_gauge"):
            cat = "rail"
        elif nat == "water":
            cat = "water"
        elif ww in ("river", "stream", "canal"):
            cat = "waterways"
        elif (
            lei in ("park", "garden", "nature_reserve", "recreation_ground")
            or lu in ("forest", "grass", "meadow", "cemetery", "orchard")
            or nat in ("wood", "scrub", "grassland")
        ):
            cat = "parks"
        elif t.get("building"):
            cat = "buildings"
        elif nat == "coastline":
            cat = "coastline"
        else:
            continue  # unrecognised tag combination — skip

        features.setdefault(cat, []).append(feat)

    # Derive a tight bbox from all parsed coordinates.
    all_lats = [c[0] for f in all_features for c in f.coords]
    all_lons = [c[1] for f in all_features for c in f.coords]
    # Also collect member geometry for relations that _parse_elements emits.
    south = min(all_lats)
    north = max(all_lats)
    west = min(all_lons)
    east = max(all_lons)
    center = ((south + north) / 2.0, (west + east) / 2.0)

    return MapData(
        location="overpass_small fixture",
        center=center,
        bbox=(south, west, north, east),
        features=features,
    )


def _render_layers_to_png(specs, canvas, out_path: Path) -> None:
    """Render a list of ``LayerSpec`` objects to a coloured PNG.

    Uses PIL to draw polylines for each spec in the spec's colour on a
    white background.  The image covers the canvas drawing area at 96 DPI.
    """
    from PIL import Image, ImageDraw

    draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
    draw_w_mm = draw_x2 - draw_x1
    draw_h_mm = draw_y2 - draw_y1

    dpi = 96
    px_per_mm = dpi / 25.4

    width_px = max(1, int(round(draw_w_mm * px_per_mm)))
    height_px = max(1, int(round(draw_h_mm * px_per_mm)))

    img = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    stroke_px = max(1, int(round(0.4 * px_per_mm)))

    for spec in specs:
        # Parse hex colour (e.g. "#1E6FD0") → (r, g, b)
        hex_color = spec.color.lstrip("#")
        color_rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        for polyline in spec.paths:
            if len(polyline) < 2:
                continue
            pixel_coords = [
                (
                    (pt[0] - draw_x1) * px_per_mm,
                    (pt[1] - draw_y1) * px_per_mm,
                )
                for pt in polyline
            ]
            draw.line(pixel_coords, fill=color_rgb, width=stroke_px)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestMapRenderPng:
    """Render fixture data → docs/images/map_example.png with no network."""

    def test_fixture_exists(self):
        assert _FIXTURE.exists(), f"Fixture not found: {_FIXTURE}"

    def test_render_produces_png(self):
        from plottter.generators.map_generator import MapGenerator
        from plottter.models import Canvas

        canvas = Canvas.from_preset("A4", margin=10.0)
        map_data = _load_map_data()

        gen = MapGenerator()
        params = {
            "_map_data": map_data,
            "include_water": True,
            "include_parks": True,
            "include_buildings": False,
            "include_waterways": True,
            "include_roads": True,
            "include_rail": True,
            "include_coastline": False,
            "include_attribution": False,
            "road_detail": "standard",
            "area_fill": "none",
            "simplify_mm": 0.0,
            "min_feature_mm": 0.0,
            "major_road_strokes": 1,
        }

        specs = gen.generate_layers(params, canvas)
        assert specs, "generate_layers() must return at least one LayerSpec for the fixture"

        _render_layers_to_png(specs, canvas, _OUT_PNG)

        assert _OUT_PNG.exists(), f"Output PNG not created: {_OUT_PNG}"
        assert _OUT_PNG.stat().st_size > 0, "Output PNG is empty (zero bytes)"

        # Verify PNG magic bytes.
        magic = _OUT_PNG.read_bytes()[:4]
        assert magic == b"\x89PNG", f"Not a valid PNG (magic={magic!r})"

    def test_png_has_non_blank_content(self):
        """The image must contain at least some non-white pixels."""
        import numpy as np
        from PIL import Image

        # Depends on test_render_produces_png having run first (same session).
        if not _OUT_PNG.exists():
            pytest.skip("PNG not yet generated — run test_render_produces_png first")

        img = Image.open(str(_OUT_PNG)).convert("RGB")
        arr = __import__("numpy").array(img)
        # Any pixel not pure-white (255,255,255) counts as content.
        non_white = (arr != 255).any(axis=2).sum()
        assert non_white > 0, "Rendered PNG appears to be a blank white image"
