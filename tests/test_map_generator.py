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
