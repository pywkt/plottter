"""Tests for MapGenerator skeleton (phase 142.1).

Verifies:
- "Map" is in GENERATORS with category="map" and emits_multiple_layers=True.
- All §8 parameter names, types, and defaults match the spec.
- generate_layers({}, canvas) returns [] (no map data → no output).
- generate_layers({"_map_data": None}, canvas) returns [].
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas


def make_canvas() -> Canvas:
    """Return a standard A4 canvas with a 10 mm margin."""
    return Canvas.from_preset("A4", margin=10.0)


class TestRegistration:
    def test_map_in_generators_registry(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        assert "Map" in GENERATORS

    def test_map_generator_class_attributes(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        cls = GENERATORS["Map"]
        assert cls.name == "Map"
        assert cls.category == "map"
        assert cls.emits_multiple_layers is True


class TestParameters:
    def setup_method(self):
        from plottter.generators.map_generator import MapGenerator

        self.gen = MapGenerator()
        self.params = self.gen.get_parameters()
        self.param_by_name = {p.name: p for p in self.params}

    # ------------------------------------------------------------------ names
    def test_all_required_param_names_present(self):
        required = {
            "radius_km",
            "extent_mode",
            "road_detail",
            "include_roads",
            "include_rail",
            "include_water",
            "include_waterways",
            "include_parks",
            "include_buildings",
            "include_coastline",
            "area_fill",
            "fill_spacing_mm",
            "fill_angle_deg",
            "major_road_strokes",
            "simplify_mm",
            "min_feature_mm",
            "include_attribution",
        }
        assert required == set(self.param_by_name)

    # ----------------------------------------------------------------- types
    def test_param_types(self):
        from plottter.generators.base import (
            BoolParam,
            ChoiceParam,
            FloatParam,
            IntParam,
        )

        assert isinstance(self.param_by_name["radius_km"], FloatParam)
        assert isinstance(self.param_by_name["extent_mode"], ChoiceParam)
        assert isinstance(self.param_by_name["road_detail"], ChoiceParam)
        assert isinstance(self.param_by_name["include_roads"], BoolParam)
        assert isinstance(self.param_by_name["include_rail"], BoolParam)
        assert isinstance(self.param_by_name["include_water"], BoolParam)
        assert isinstance(self.param_by_name["include_waterways"], BoolParam)
        assert isinstance(self.param_by_name["include_parks"], BoolParam)
        assert isinstance(self.param_by_name["include_buildings"], BoolParam)
        assert isinstance(self.param_by_name["include_coastline"], BoolParam)
        assert isinstance(self.param_by_name["area_fill"], ChoiceParam)
        assert isinstance(self.param_by_name["fill_spacing_mm"], FloatParam)
        assert isinstance(self.param_by_name["fill_angle_deg"], FloatParam)
        assert isinstance(self.param_by_name["major_road_strokes"], IntParam)
        assert isinstance(self.param_by_name["simplify_mm"], FloatParam)
        assert isinstance(self.param_by_name["min_feature_mm"], FloatParam)
        assert isinstance(self.param_by_name["include_attribution"], BoolParam)

    # --------------------------------------------------------------- defaults
    def test_defaults(self):
        p = self.param_by_name
        assert p["radius_km"].default == pytest.approx(1.5)
        assert p["extent_mode"].default == "radius"
        assert p["road_detail"].default == "standard"
        assert p["include_roads"].default is True
        assert p["include_rail"].default is True
        assert p["include_water"].default is True
        assert p["include_waterways"].default is True
        assert p["include_parks"].default is True
        assert p["include_buildings"].default is False
        assert p["include_coastline"].default is True
        assert p["area_fill"].default == "none"
        assert p["fill_spacing_mm"].default == pytest.approx(2.0)
        assert p["fill_angle_deg"].default == pytest.approx(45.0)
        assert p["major_road_strokes"].default == 1
        assert p["simplify_mm"].default == pytest.approx(0.15)
        assert p["min_feature_mm"].default == pytest.approx(0.8)
        assert p["include_attribution"].default is True

    # ----------------------------------------------------------------- ranges
    def test_float_ranges(self):
        p = self.param_by_name
        assert p["radius_km"].min == pytest.approx(0.2)
        assert p["radius_km"].max == pytest.approx(10.0)
        assert p["fill_spacing_mm"].min == pytest.approx(0.3)
        assert p["fill_spacing_mm"].max == pytest.approx(10.0)
        assert p["fill_angle_deg"].min == pytest.approx(0.0)
        assert p["fill_angle_deg"].max == pytest.approx(180.0)
        assert p["simplify_mm"].min == pytest.approx(0.0)
        assert p["simplify_mm"].max == pytest.approx(2.0)
        assert p["min_feature_mm"].min == pytest.approx(0.0)
        assert p["min_feature_mm"].max == pytest.approx(10.0)

    def test_int_ranges(self):
        p = self.param_by_name
        assert p["major_road_strokes"].min == 1
        assert p["major_road_strokes"].max == 4

    # ---------------------------------------------------------------- choices
    def test_choice_options(self):
        p = self.param_by_name
        assert set(p["extent_mode"].choices) == {"radius", "place_bbox"}
        assert set(p["road_detail"].choices) == {"major_only", "standard", "all_streets"}
        assert set(p["area_fill"].choices) == {"none", "hatch", "cross_hatch"}

    # --------------------------------------------------------- visible_when
    def test_fill_params_visible_when_area_fill_not_none(self):
        """fill_spacing_mm and fill_angle_deg are only visible when area_fill
        is hatch or cross_hatch (i.e. not 'none')."""
        p = self.param_by_name
        vw_spacing = p["fill_spacing_mm"].visible_when
        vw_angle = p["fill_angle_deg"].visible_when
        assert vw_spacing is not None
        assert vw_angle is not None
        assert "area_fill" in vw_spacing
        assert set(vw_spacing["area_fill"]) == {"hatch", "cross_hatch"}
        assert "area_fill" in vw_angle
        assert set(vw_angle["area_fill"]) == {"hatch", "cross_hatch"}


class TestGenerateLayers:
    def setup_method(self):
        from plottter.generators.map_generator import MapGenerator

        self.gen = MapGenerator()
        self.canvas = make_canvas()

    def test_no_map_data_key_returns_empty(self):
        """generate_layers with an empty params dict returns []."""
        result = self.gen.generate_layers({}, self.canvas)
        assert result == []

    def test_map_data_none_returns_empty(self):
        """generate_layers with _map_data=None returns []."""
        result = self.gen.generate_layers({"_map_data": None}, self.canvas)
        assert result == []

    def test_returns_list(self):
        """Return value is always a list (never None)."""
        result = self.gen.generate_layers({}, self.canvas)
        assert isinstance(result, list)

    def test_generate_returns_list(self):
        """generate() fallback also returns a list."""
        result = self.gen.generate({}, self.canvas)
        assert isinstance(result, list)

    def test_get_presets_returns_list(self):
        presets = self.gen.get_presets()
        assert isinstance(presets, list)


def _make_line_map_data():
    """Fixture MapData with roads_major, roads_minor, and rail features."""
    from plottter.osm.types import MapData, MapFeature

    # Three points spread across a ~0.02° grid near Paris so the fit-transform
    # produces a non-degenerate scale and all polylines have measurable length.
    roads_major = [
        MapFeature(
            tags={"highway": "primary"},
            coords=[(48.850, 2.340), (48.855, 2.350), (48.860, 2.360)],
            is_area=False,
        )
    ]
    roads_minor = [
        MapFeature(
            tags={"highway": "residential"},
            coords=[(48.851, 2.341), (48.853, 2.345), (48.855, 2.348)],
            is_area=False,
        )
    ]
    rail = [
        MapFeature(
            tags={"railway": "rail"},
            coords=[(48.848, 2.340), (48.850, 2.352), (48.852, 2.360)],
            is_area=False,
        )
    ]
    return MapData(
        location="Paris, France",
        center=(48.855, 2.350),
        bbox=(48.845, 2.335, 48.865, 2.365),
        features={
            "roads_major": roads_major,
            "roads_minor": roads_minor,
            "rail": rail,
        },
    )


def test_line_layers():
    """Phase 143.1 — generate_layers for line categories.

    Verifies:
    - Enabling roads+rail with a fixture MapData yields exactly those layers.
    - Disabled categories produce no LayerSpec.
    - All output coordinates fall within canvas.drawing_area().
    - Layer colors match FEATURE_CATEGORIES defaults.
    """
    from plottter.generators.map_generator import MapGenerator
    from plottter.osm.categories import FEATURE_CATEGORIES

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_line_map_data()

    base_params = {
        "_map_data": map_data,
        "road_detail": "standard",
        "simplify_mm": 0.0,      # no simplification — keep exact projected points
        "min_feature_mm": 0.0,   # keep all fragments regardless of length
        "include_water": False,
        "include_parks": False,
        "include_buildings": False,
        "include_attribution": False,  # attribution lives in margin; checked in test_attribution
    }

    # ------------------------------------------------------------------ #
    # 1. Enabling roads + rail yields Roads (major), Roads (minor), Rail.
    # ------------------------------------------------------------------ #
    params_roads_rail = {
        **base_params,
        "include_roads": True,
        "include_rail": True,
        "include_waterways": False,
        "include_coastline": False,
    }
    specs = gen.generate_layers(params_roads_rail, canvas)
    names = {s.name for s in specs}
    assert "Roads (major)" in names, f"Expected 'Roads (major)' in {names}"
    assert "Roads (minor)" in names, f"Expected 'Roads (minor)' in {names}"
    assert "Rail" in names, f"Expected 'Rail' in {names}"
    assert "Waterways" not in names, f"'Waterways' should be absent; got {names}"
    assert "Coastline" not in names, f"'Coastline' should be absent; got {names}"

    # ------------------------------------------------------------------ #
    # 2. Disabled category → no LayerSpec for that category.
    # ------------------------------------------------------------------ #
    params_no_rail = {
        **base_params,
        "include_roads": True,
        "include_rail": False,
        "include_waterways": False,
        "include_coastline": False,
    }
    specs_no_rail = gen.generate_layers(params_no_rail, canvas)
    names_no_rail = {s.name for s in specs_no_rail}
    assert "Rail" not in names_no_rail, f"'Rail' should be absent when disabled; got {names_no_rail}"
    assert "Roads (major)" in names_no_rail

    # ------------------------------------------------------------------ #
    # 3. All output coords are within canvas.drawing_area() (±0.01 mm).
    # ------------------------------------------------------------------ #
    left, top, right, bottom = canvas.drawing_area()
    tol = 0.01
    for spec in specs:
        for path in spec.paths:
            for x, y in path:
                assert left - tol <= x <= right + tol, (
                    f"x={x:.3f} outside [{left:.3f}, {right:.3f}]"
                )
                assert top - tol <= y <= bottom + tol, (
                    f"y={y:.3f} outside [{top:.3f}, {bottom:.3f}]"
                )

    # ------------------------------------------------------------------ #
    # 4. Colors match FEATURE_CATEGORIES defaults.
    # ------------------------------------------------------------------ #
    spec_by_name = {s.name: s for s in specs}
    assert spec_by_name["Roads (major)"].color == FEATURE_CATEGORIES["roads_major"]["color"]
    assert spec_by_name["Roads (minor)"].color == FEATURE_CATEGORIES["roads_minor"]["color"]
    assert spec_by_name["Rail"].color == FEATURE_CATEGORIES["rail"]["color"]

    # ------------------------------------------------------------------ #
    # 5. major_only road_detail suppresses Roads (minor).
    # ------------------------------------------------------------------ #
    params_major_only = {
        **base_params,
        "include_roads": True,
        "include_rail": False,
        "include_waterways": False,
        "include_coastline": False,
        "road_detail": "major_only",
    }
    specs_major = gen.generate_layers(params_major_only, canvas)
    names_major = {s.name for s in specs_major}
    assert "Roads (minor)" not in names_major, (
        f"'Roads (minor)' should be absent for major_only; got {names_major}"
    )
    assert "Roads (major)" in names_major


def test_progress_cancel():
    """Phase 143.2 — progress callback and cancellation for generate_layers.

    (A) A recording progress_callback receives a final value of 100 for a
        complete run.
    (B) A cancelled_callback that returns True after the first category is
        processed yields fewer layers than the full run, without raising.
    """
    from plottter.generators.map_generator import MapGenerator

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_line_map_data()

    # Fixture has roads_major, roads_minor, rail — 3 enabled line categories.
    params = {
        "_map_data": map_data,
        "road_detail": "standard",
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_roads": True,
        "include_rail": True,
        "include_water": False,
        "include_waterways": False,
        "include_coastline": False,
        "include_parks": False,
        "include_buildings": False,
    }

    # ------------------------------------------------------------------ #
    # (A) Full run: recording progress callback must end at 100.
    # ------------------------------------------------------------------ #
    progress_values: list[int] = []
    full_specs = gen.generate_layers(
        params, canvas, progress_callback=lambda v: progress_values.append(v)
    )
    assert progress_values, "progress_callback was never called"
    assert progress_values[-1] == 100, (
        f"Last progress value should be 100, got {progress_values[-1]}"
    )
    n_full = len(full_specs)
    assert n_full > 0, "Expected at least one layer for a full run"

    # ------------------------------------------------------------------ #
    # (B) Cancel after first category: fewer layers, no exception raised.
    # ------------------------------------------------------------------ #
    call_count: list[int] = [0]

    def cancel_after_one() -> bool:
        call_count[0] += 1
        return call_count[0] > 1

    cancelled_specs = gen.generate_layers(
        params, canvas, cancelled_callback=cancel_after_one
    )
    assert len(cancelled_specs) < n_full, (
        f"Cancelled run should produce fewer layers than full run ({n_full}); "
        f"got {len(cancelled_specs)}"
    )


def _make_area_map_data():
    """Fixture MapData with a park polygon and a building footprint."""
    from plottter.osm.types import MapData, MapFeature

    # A rectangular park near Paris (explicitly closed: first == last).
    park_coords = [
        (48.855, 2.340),
        (48.860, 2.340),
        (48.860, 2.350),
        (48.855, 2.350),
        (48.855, 2.340),
    ]
    parks = [
        MapFeature(
            tags={"leisure": "park"},
            coords=park_coords,
            is_area=True,
        )
    ]
    # A small building footprint (open — assemble() must close it).
    building_coords = [
        (48.856, 2.341),
        (48.858, 2.341),
        (48.858, 2.343),
        (48.856, 2.343),
    ]
    buildings = [
        MapFeature(
            tags={"building": "yes"},
            coords=building_coords,
            is_area=True,
        )
    ]
    return MapData(
        location="Paris, France",
        center=(48.857, 2.345),
        bbox=(48.850, 2.335, 48.865, 2.360),
        features={
            "parks": parks,
            "buildings": buildings,
        },
    )


def test_area_outlines():
    """Phase 144.1 — area outlines as closed polylines.

    Verifies:
    - A park polygon yields a LayerSpec whose polylines are closed
      (first ≈ last within float precision).
    - All park polyline coordinates lie inside canvas.drawing_area().
    - include_buildings=False → no Buildings LayerSpec.
    - Parks layer color matches FEATURE_CATEGORIES default.
    """
    from plottter.generators.map_generator import MapGenerator
    from plottter.osm.categories import FEATURE_CATEGORIES

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_area_map_data()

    base_params = {
        "_map_data": map_data,
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_roads": False,
        "include_rail": False,
        "include_water": False,
        "include_waterways": False,
        "include_coastline": False,
    }

    # ------------------------------------------------------------------ #
    # 1. Park polygon → closed polylines inside printable area.
    # ------------------------------------------------------------------ #
    params = {
        **base_params,
        "include_parks": True,
        "include_buildings": False,
    }
    specs = gen.generate_layers(params, canvas)
    names = {s.name for s in specs}
    assert "Parks" in names, f"Expected 'Parks' in {names}"
    assert "Buildings" not in names, f"'Buildings' should be absent; got {names}"

    parks_spec = next(s for s in specs if s.name == "Parks")
    assert parks_spec.paths, "Parks layer must have at least one path"

    # Closed polyline: first ≈ last within float precision.
    tol = 1e-6
    for path in parks_spec.paths:
        assert len(path) >= 3, "Park ring must have at least 3 points"
        dx = abs(path[0][0] - path[-1][0])
        dy = abs(path[0][1] - path[-1][1])
        assert dx <= tol and dy <= tol, (
            f"Park polyline not closed: first={path[0]}, last={path[-1]}"
        )

    # All coordinates inside printable area (±0.01 mm tolerance).
    left, top, right, bottom = canvas.drawing_area()
    coord_tol = 0.01
    for path in parks_spec.paths:
        for x, y in path:
            assert left - coord_tol <= x <= right + coord_tol, (
                f"x={x:.3f} outside [{left:.3f}, {right:.3f}]"
            )
            assert top - coord_tol <= y <= bottom + coord_tol, (
                f"y={y:.3f} outside [{top:.3f}, {bottom:.3f}]"
            )

    # ------------------------------------------------------------------ #
    # 2. include_buildings=False → no Buildings layer;
    #    include_buildings=True → Buildings layer present.
    # ------------------------------------------------------------------ #
    params_with_bld = {
        **base_params,
        "include_parks": False,
        "include_buildings": True,
    }
    specs_with_bld = gen.generate_layers(params_with_bld, canvas)
    assert "Buildings" in {s.name for s in specs_with_bld}, (
        "'Buildings' must be present when include_buildings=True and data exists"
    )
    assert "Parks" not in {s.name for s in specs_with_bld}, (
        "'Parks' must be absent when include_parks=False"
    )

    # ------------------------------------------------------------------ #
    # 3. Parks color matches FEATURE_CATEGORIES default.
    # ------------------------------------------------------------------ #
    spec_by_name = {s.name: s for s in specs}
    assert spec_by_name["Parks"].color == FEATURE_CATEGORIES["parks"]["color"]


def test_hatch_fill():
    """Phase 144.2 — hatch fill for area polygons.

    Verifies:
    - With area_fill="hatch", the Parks layer contains more than just outline
      paths (i.e. hatch lines were appended).
    - Every interior fill point lies inside the source polygon
      (shapely containment, tolerance 0.1 mm).
    - A polygon with a hole: no fill point lies inside the hole.
    - With area_fill="cross_hatch", two sets of hatch lines are generated.
    """
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    from plottter.generators.map_generator import MapGenerator
    from plottter.osm.types import MapData, MapFeature

    gen = MapGenerator()
    canvas = make_canvas()
    left, top, right, bottom = canvas.drawing_area()

    # ------------------------------------------------------------------ #
    # Build a fixture with a park that has an inner hole (simulating a
    # multipolygon relation inner ring so we can test hole exclusion).
    # ------------------------------------------------------------------ #
    # Outer park polygon — a 0.01° × 0.01° square near Paris.
    park_outer = [
        (48.855, 2.340),
        (48.865, 2.340),
        (48.865, 2.350),
        (48.855, 2.350),
        (48.855, 2.340),
    ]
    # Inner hole — a smaller square centred inside the park.
    hole_ring = [
        (48.858, 2.343),
        (48.862, 2.343),
        (48.862, 2.347),
        (48.858, 2.347),
        (48.858, 2.343),
    ]
    park_feature = MapFeature(
        tags={"leisure": "park"},
        coords=park_outer,
        is_area=True,
        inner_coords=[hole_ring],
    )

    map_data = MapData(
        location="Paris, France",
        center=(48.860, 2.345),
        bbox=(48.850, 2.335, 48.870, 2.360),
        features={"parks": [park_feature]},
    )

    base_params = {
        "_map_data": map_data,
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_roads": False,
        "include_rail": False,
        "include_water": False,
        "include_waterways": False,
        "include_coastline": False,
        "include_buildings": False,
        "include_parks": True,
        "fill_spacing_mm": 2.0,
        "fill_angle_deg": 45.0,
    }

    # ------------------------------------------------------------------ #
    # 1. area_fill="hatch" → Parks layer present with paths.
    # ------------------------------------------------------------------ #
    params_hatch = {**base_params, "area_fill": "hatch"}
    specs = gen.generate_layers(params_hatch, canvas)
    assert specs, "generate_layers must return at least one spec"
    parks_spec = next((s for s in specs if s.name == "Parks"), None)
    assert parks_spec is not None, "Parks layer must be present"
    assert parks_spec.paths, "Parks layer must have paths"

    # ------------------------------------------------------------------ #
    # 2. Every fill-line point lies inside the source polygon.
    #    We reconstruct the source polygon in canvas mm to check.
    # ------------------------------------------------------------------ #
    from plottter.osm.geometry import fit_transform, mercator

    from plottter.osm.types import MapFeature as MF

    # Derive the same fit-transform the generator used.
    all_feats = [park_feature]
    transform = fit_transform(all_feats, canvas)

    def project_pt(lat, lon):
        px, py = mercator(lat, lon)
        return (
            transform.x_origin + px * transform.scale,
            transform.y_origin - py * transform.scale,
        )

    outer_mm = [project_pt(lat, lon) for lat, lon in park_outer]
    hole_mm = [project_pt(lat, lon) for lat, lon in hole_ring]

    # Clip the source polygon to the canvas bbox (same as the generator).
    from shapely.geometry import box as shapely_box

    bbox_shp = shapely_box(left, top, right, bottom)
    source_poly = ShapelyPolygon(outer_mm, [hole_mm])
    if not source_poly.is_valid:
        source_poly = source_poly.buffer(0)
    source_poly = source_poly.intersection(bbox_shp)

    tol_mm = 0.1  # spec tolerance

    # Collect all non-outline paths from the hatch run.  The outline is the
    # closed path (first ≈ last); hatch lines are open segments.
    hatch_paths = [
        path
        for path in parks_spec.paths
        if not (len(path) >= 2 and abs(path[0][0] - path[-1][0]) < 1e-6
                and abs(path[0][1] - path[-1][1]) < 1e-6)
    ]
    assert hatch_paths, "At least one hatch segment must be present"

    for path in hatch_paths:
        for x, y in path:
            pt = Point(x, y)
            assert source_poly.distance(pt) <= tol_mm, (
                f"Hatch point ({x:.3f}, {y:.3f}) is {source_poly.distance(pt):.4f} mm "
                f"outside the source polygon (tol={tol_mm} mm)"
            )

    # ------------------------------------------------------------------ #
    # 3. No fill point lies inside the hole.
    # ------------------------------------------------------------------ #
    hole_poly = ShapelyPolygon(hole_mm)
    for path in hatch_paths:
        for x, y in path:
            pt = Point(x, y)
            # Allow a tiny tolerance for floating-point edge cases.
            assert not hole_poly.contains(pt), (
                f"Hatch point ({x:.3f}, {y:.3f}) lies inside the hole polygon"
            )

    # ------------------------------------------------------------------ #
    # 4. area_fill="cross_hatch" → more paths than "hatch" (two directions).
    # ------------------------------------------------------------------ #
    params_cross = {**base_params, "area_fill": "cross_hatch"}
    specs_cross = gen.generate_layers(params_cross, canvas)
    parks_cross = next((s for s in specs_cross if s.name == "Parks"), None)
    assert parks_cross is not None
    assert len(parks_cross.paths) > len(parks_spec.paths), (
        "cross_hatch must produce more paths than hatch"
    )


def test_cross_hatch():
    """Phase 144.3 — cross-hatch generates lines at fill_angle_deg and its perpendicular.

    Verifies:
    - cross_hatch yields strictly more fill segments than hatch on the same polygon.
    - Fill segments in cross_hatch span both fill_angle_deg and (fill_angle_deg + 90°).
    """
    import math

    from plottter.generators.map_generator import MapGenerator
    from plottter.osm.types import MapData, MapFeature

    gen = MapGenerator()
    canvas = make_canvas()

    # Large rectangular park to guarantee multiple hatch lines per direction.
    park_outer = [
        (48.850, 2.330),
        (48.870, 2.330),
        (48.870, 2.360),
        (48.850, 2.360),
        (48.850, 2.330),
    ]
    park_feature = MapFeature(
        tags={"leisure": "park"},
        coords=park_outer,
        is_area=True,
    )
    map_data = MapData(
        location="Paris, France",
        center=(48.860, 2.345),
        bbox=(48.845, 2.325, 48.875, 2.365),
        features={"parks": [park_feature]},
    )

    fill_angle_deg = 30.0
    fill_spacing_mm = 4.0

    base_params = {
        "_map_data": map_data,
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_roads": False,
        "include_rail": False,
        "include_water": False,
        "include_waterways": False,
        "include_coastline": False,
        "include_buildings": False,
        "include_parks": True,
        "fill_spacing_mm": fill_spacing_mm,
        "fill_angle_deg": fill_angle_deg,
    }

    def is_closed(path):
        return (
            len(path) >= 2
            and abs(path[0][0] - path[-1][0]) < 1e-6
            and abs(path[0][1] - path[-1][1]) < 1e-6
        )

    def segment_angle_deg(path):
        """Return the orientation angle (0–180°) of a segment's first→last vector."""
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        return math.degrees(math.atan2(dy, dx)) % 180.0

    def seg_len(path):
        return math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])

    def near_angle(seg_angle, target, tol=5.0):
        diff = abs(seg_angle - target) % 180.0
        return min(diff, 180.0 - diff) <= tol

    # ------------------------------------------------------------------ #
    # 1. cross_hatch → more fill segments than hatch.
    # ------------------------------------------------------------------ #
    specs_hatch = gen.generate_layers({**base_params, "area_fill": "hatch"}, canvas)
    specs_cross = gen.generate_layers({**base_params, "area_fill": "cross_hatch"}, canvas)

    parks_hatch = next((s for s in specs_hatch if s.name == "Parks"), None)
    parks_cross = next((s for s in specs_cross if s.name == "Parks"), None)
    assert parks_hatch is not None, "Parks layer missing for hatch"
    assert parks_cross is not None, "Parks layer missing for cross_hatch"

    hatch_segs = [p for p in parks_hatch.paths if not is_closed(p)]
    cross_segs = [p for p in parks_cross.paths if not is_closed(p)]

    assert len(hatch_segs) > 0, "hatch must produce at least one fill segment"
    assert len(cross_segs) > len(hatch_segs), (
        f"cross_hatch ({len(cross_segs)} segs) must exceed hatch ({len(hatch_segs)} segs)"
    )

    # ------------------------------------------------------------------ #
    # 2. Cross-hatch segments span both fill_angle_deg and +90°.
    # ------------------------------------------------------------------ #
    angle1 = fill_angle_deg % 180.0
    angle2 = (fill_angle_deg + 90.0) % 180.0

    # Ignore very short segments that may be numerical noise at polygon edges.
    meaningful = [p for p in cross_segs if seg_len(p) >= 0.1]
    assert meaningful, "cross_hatch must produce meaningful (non-degenerate) fill segments"

    angles_seen = [segment_angle_deg(p) for p in meaningful]

    has_angle1 = any(near_angle(a, angle1) for a in angles_seen)
    has_angle2 = any(near_angle(a, angle2) for a in angles_seen)

    assert has_angle1, (
        f"cross_hatch must include lines at fill_angle_deg={angle1:.1f}°; "
        f"seen angles (first 10): {sorted(angles_seen)[:10]}"
    )
    assert has_angle2, (
        f"cross_hatch must include lines at perpendicular {angle2:.1f}°; "
        f"seen angles (first 10): {sorted(angles_seen)[:10]}"
    )


def test_generate_fallback():
    """Phase 145.1 — generate() returns a non-empty flat polyline list.

    Verifies:
    - generate() returns a non-empty list of polylines when map data is present.
    - All coordinates lie inside the canvas printable area.
    - The list is flat (each element is a list of (x, y) tuples, not a LayerSpec).
    """
    from plottter.generators.map_generator import MapGenerator

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_line_map_data()
    left, top, right, bottom = canvas.drawing_area()

    params = {
        "_map_data": map_data,
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_roads": True,
        "include_rail": True,
        "include_water": False,
        "include_waterways": False,
        "include_parks": False,
        "include_buildings": False,
        "include_coastline": False,
        "include_attribution": False,  # attribution lives in margin; checked in test_attribution
    }

    result = gen.generate(params, canvas)

    assert result, "generate() must return a non-empty list when map data is present"
    assert isinstance(result, list), "generate() must return a list"

    # Each element must be a polyline (list of (x, y) tuples), not a LayerSpec.
    for polyline in result:
        assert isinstance(polyline, list), f"Each element must be a list, got {type(polyline)}"
        assert polyline, "Polylines must be non-empty"
        for point in polyline:
            x, y = point
            assert isinstance(x, (float, int)), f"x must be numeric, got {type(x)}"
            assert isinstance(y, (float, int)), f"y must be numeric, got {type(y)}"

    # All coordinates must lie inside the printable area (±0.01 mm tolerance).
    coord_tol = 0.01
    for polyline in result:
        for x, y in polyline:
            assert left - coord_tol <= x <= right + coord_tol, (
                f"x={x:.3f} outside printable area [{left:.3f}, {right:.3f}]"
            )
            assert top - coord_tol <= y <= bottom + coord_tol, (
                f"y={y:.3f} outside printable area [{top:.3f}, {bottom:.3f}]"
            )


def test_major_strokes():
    """Phase 145.2 — major_road_strokes=3 yields ~3× the polyline count vs =1.

    Verifies that when major_road_strokes=3, the Roads (major) layer contains
    approximately 3× as many polylines as when major_road_strokes=1.
    The test uses a fixture MapData with a single roads_major feature so the
    baseline count is predictable.
    """
    from plottter.generators.map_generator import MapGenerator

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_line_map_data()

    base_params = {
        "_map_data": map_data,
        "road_detail": "standard",
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_water": False,
        "include_parks": False,
        "include_buildings": False,
        "include_waterways": False,
        "include_coastline": False,
        "include_rail": False,
        "include_roads": True,
    }

    # Generate with 1 stroke (baseline).
    params_1 = {**base_params, "major_road_strokes": 1}
    specs_1 = gen.generate_layers(params_1, canvas)
    major_1 = next((s for s in specs_1 if s.name == "Roads (major)"), None)
    assert major_1 is not None, "Roads (major) layer must be present with strokes=1"
    count_1 = len(major_1.paths)
    assert count_1 >= 1, "Baseline must have at least one polyline"

    # Generate with 3 strokes.
    params_3 = {**base_params, "major_road_strokes": 3}
    specs_3 = gen.generate_layers(params_3, canvas)
    major_3 = next((s for s in specs_3 if s.name == "Roads (major)"), None)
    assert major_3 is not None, "Roads (major) layer must be present with strokes=3"
    count_3 = len(major_3.paths)

    # Should be approximately 3× the baseline (within 50% tolerance to allow for
    # clipping effects — offset strokes that extend slightly outside the bbox may
    # be trimmed, but the count must still increase significantly).
    ratio = count_3 / count_1
    assert ratio >= 2.5, (
        f"Expected ~3× polylines with strokes=3 vs strokes=1, "
        f"got {count_3} vs {count_1} (ratio={ratio:.2f})"
    )
    assert ratio <= 3.5, (
        f"Expected ~3× polylines with strokes=3 vs strokes=1, "
        f"got {count_3} vs {count_1} (ratio={ratio:.2f})"
    )

    # Non-major layers must not be affected by major_road_strokes.
    minor_1 = next((s for s in specs_1 if s.name == "Roads (minor)"), None)
    minor_3 = next((s for s in specs_3 if s.name == "Roads (minor)"), None)
    if minor_1 is not None and minor_3 is not None:
        assert len(minor_1.paths) == len(minor_3.paths), (
            "Roads (minor) count must be unaffected by major_road_strokes"
        )


def test_attribution():
    """Phase 145.3 — Attribution layer present/absent based on include_attribution.

    Verifies:
    - When include_attribution=True (default), generate_layers emits a LayerSpec
      named "Attribution" with non-empty paths whose Y coordinates fall in the
      canvas bottom margin (below the drawing-area bottom edge).
    - When include_attribution=False, no "Attribution" layer is emitted.
    """
    from plottter.generators.map_generator import MapGenerator

    gen = MapGenerator()
    canvas = make_canvas()
    map_data = _make_line_map_data()

    base_params = {
        "_map_data": map_data,
        "road_detail": "standard",
        "simplify_mm": 0.0,
        "min_feature_mm": 0.0,
        "include_water": False,
        "include_parks": False,
        "include_buildings": False,
        "include_waterways": False,
        "include_coastline": False,
        "include_rail": False,
        "include_roads": True,
    }

    # --- attribution enabled (default) ---
    specs_on = gen.generate_layers({**base_params, "include_attribution": True}, canvas)
    attr_layer = next((s for s in specs_on if s.name == "Attribution"), None)
    assert attr_layer is not None, "Attribution layer must be present when include_attribution=True"
    assert len(attr_layer.paths) > 0, "Attribution layer must have non-empty paths"

    # All path points should be in the bottom margin (Y > drawing area bottom).
    _left, _top, _right, bottom_da = canvas.drawing_area()
    for path in attr_layer.paths:
        for _x, y in path:
            assert y > bottom_da, (
                f"Attribution point y={y:.2f} must be below drawing area "
                f"bottom={bottom_da:.2f} (in the margin)"
            )

    # --- attribution disabled ---
    specs_off = gen.generate_layers({**base_params, "include_attribution": False}, canvas)
    attr_layer_off = next((s for s in specs_off if s.name == "Attribution"), None)
    assert attr_layer_off is None, "Attribution layer must be absent when include_attribution=False"
