"""Tests for PointillistGenerator (task 160.3).

Covers:
- Generator registered in GENERATORS under "Pointillist".
- get_parameters() returns every param from spec §7 with correct
  types/ranges/defaults; dot_style restricted to ["point"]; seed
  is the only randomizable param.
- generate_layers() on a small synthetic image with Basic 6 palette:
  - Returns ≤6 LayerSpecs.
  - Every path has len ≥ 2.
  - Every coordinate lies inside the canvas drawing area.
- skip_paper_white=True drops the #FFFFFF layer.
- Empty mask (palette color absent from image) drops that layer.
- Same seed → identical LayerSpec sequence across two runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas() -> Canvas:
    """A4 canvas with 10 mm margin — drawing area (10, 10, 200, 287)."""
    return Canvas.from_preset("A4", margin=10.0)


def _make_basic6_image(width: int = 16, height: int = 16) -> np.ndarray:
    """Small synthetic RGB image using only black and red from Basic 6.

    Top half: black (#000000 = [0, 0, 0])
    Bottom half: red  (#E63946 = [230, 57, 70])
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mid = height // 2
    img[:mid, :] = [0, 0, 0]       # Black
    img[mid:, :] = [230, 57, 70]   # Red (#E63946)
    return img


def _base_params(image: np.ndarray, **overrides) -> dict:
    return {
        "_source_image": image,
        "palette": "Basic 6",
        "density_per_cm2": 200.0,
        "dither": "none",
        "dot_style": "point",
        "dot_size_mm": 0.5,
        "seed": 42,
        "skip_paper_white": True,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registered_in_generators(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        assert "Pointillist" in GENERATORS

    def test_class_attributes(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        cls = GENERATORS["Pointillist"]
        assert cls.name == "Pointillist"
        assert cls.category == "image"
        assert cls.uses_source_image is True
        assert cls.emits_multiple_layers is True


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.params = self.gen.get_parameters()
        self.by_name = {p.name: p for p in self.params}

    def test_all_spec_params_present(self):
        expected = {
            "palette", "density_per_cm2", "dither",
            "dot_style", "dot_size_mm", "seed", "skip_paper_white",
        }
        assert expected.issubset(self.by_name.keys())

    def test_palette_choice_param(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["palette"]
        assert isinstance(p, ChoiceParam)
        assert p.default == "Basic 6"
        assert "Basic 6" in p.choices
        assert p.randomizable is False

    def test_density_per_cm2_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["density_per_cm2"]
        assert isinstance(p, FloatParam)
        assert p.default == 200.0
        assert p.min == 10.0
        assert p.max == 2000.0
        assert p.randomizable is False

    def test_dither_choice_param(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["dither"]
        assert isinstance(p, ChoiceParam)
        assert p.default == "floyd-steinberg"
        assert set(p.choices) == {"none", "floyd-steinberg", "ordered", "atkinson"}
        assert p.randomizable is False

    def test_dot_style_choices(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["dot_style"]
        assert isinstance(p, ChoiceParam)
        assert p.choices == ["point", "cross", "circle"]
        assert p.default == "point"
        assert p.randomizable is False

    def test_dot_size_mm_visible_when(self):
        p = self.by_name["dot_size_mm"]
        assert p.visible_when == {"dot_style": ["cross", "circle"]}

    def test_dot_size_mm_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["dot_size_mm"]
        assert isinstance(p, FloatParam)
        assert p.default == 0.5
        assert p.min == 0.1
        assert p.max == 3.0
        assert p.randomizable is False

    def test_seed_int_param_randomizable(self):
        from plottter.generators.base import IntParam

        p = self.by_name["seed"]
        assert isinstance(p, IntParam)
        assert p.default == 0
        assert p.min == 0
        assert p.max == 99999
        assert p.randomizable is True

    def test_skip_paper_white_bool_param(self):
        from plottter.generators.base import BoolParam

        p = self.by_name["skip_paper_white"]
        assert isinstance(p, BoolParam)
        assert p.default is True
        assert p.randomizable is False


# ---------------------------------------------------------------------------
# generate_layers — output shape and coordinate validity
# ---------------------------------------------------------------------------

class TestGenerateLayers:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def _run(self, **overrides):
        params = _base_params(self.image, **overrides)
        return self.gen.generate_layers(params, self.canvas)

    def test_returns_at_most_6_layers(self):
        specs = self._run()
        assert len(specs) <= 6

    def test_every_path_has_at_least_2_points(self):
        specs = self._run()
        assert len(specs) > 0, "Expected at least one layer"
        for spec in specs:
            for path in spec.paths:
                assert len(path) >= 2, f"Path too short: {path}"

    def test_all_coords_inside_drawing_area(self):
        specs = self._run()
        left, top, right, bottom = self.canvas.drawing_area()
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert left <= x <= right, f"x={x} outside [{left}, {right}]"
                    assert top <= y <= bottom, f"y={y} outside [{top}, {bottom}]"

    def test_layer_colors_match_palette(self):
        from plottter.color import get_preset

        palette = get_preset("Basic 6")
        specs = self._run()
        palette_hexes = {c.upper() for c in palette.colors}
        for spec in specs:
            assert spec.color.upper() in palette_hexes

    def test_returns_empty_when_no_source_image(self):
        params = _base_params(self.image)
        del params["_source_image"]
        specs = self.gen.generate_layers(params, self.canvas)
        assert specs == []


# ---------------------------------------------------------------------------
# dot styles
# ---------------------------------------------------------------------------

class TestDotStyles:
    """cross doubles polyline count; circle gives 13-point polylines; size scales."""

    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def _total_paths(self, **overrides) -> list:
        params = _base_params(self.image, **overrides)
        specs = self.gen.generate_layers(params, self.canvas)
        paths = []
        for spec in specs:
            paths.extend(spec.paths)
        return paths

    def test_cross_doubles_polyline_count_vs_point(self):
        point_paths = self._total_paths(dot_style="point")
        cross_paths = self._total_paths(dot_style="cross")
        assert len(point_paths) > 0, "Expected at least one layer with point style"
        assert len(cross_paths) == len(point_paths) * 2, (
            f"cross should produce 2x polylines: point={len(point_paths)}, "
            f"cross={len(cross_paths)}"
        )

    def test_circle_polylines_have_12_points(self):
        circle_paths = self._total_paths(dot_style="circle")
        assert len(circle_paths) > 0, "Expected at least one layer with circle style"
        for path in circle_paths:
            assert len(path) == 12, (
                f"circle polyline should have 12 vertices, got {len(path)}"
            )

    def test_dot_size_scales_cross_spatial_extent(self):
        small_paths = self._total_paths(dot_style="cross", dot_size_mm=0.2)
        large_paths = self._total_paths(dot_style="cross", dot_size_mm=2.0)
        assert len(small_paths) > 0

        def arm_length(path):
            x0, y0 = path[0]
            x1, y1 = path[1]
            return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

        small_len = arm_length(small_paths[0])
        large_len = arm_length(large_paths[0])
        assert large_len > small_len * 5, (
            f"larger dot_size_mm should produce proportionally longer cross arms: "
            f"small={small_len:.4f}, large={large_len:.4f}"
        )

    def test_dot_size_scales_circle_radius(self):
        import math

        small_paths = self._total_paths(dot_style="circle", dot_size_mm=0.4)
        large_paths = self._total_paths(dot_style="circle", dot_size_mm=2.0)
        assert len(small_paths) > 0

        def circle_radius(path):
            # Estimate radius from first vertex relative to centroid
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            return math.hypot(path[0][0] - cx, path[0][1] - cy)

        small_r = circle_radius(small_paths[0])
        large_r = circle_radius(large_paths[0])
        assert large_r > small_r * 3, (
            f"larger dot_size_mm should produce proportionally larger circle radius: "
            f"small_r={small_r:.4f}, large_r={large_r:.4f}"
        )


# ---------------------------------------------------------------------------
# skip_paper_white
# ---------------------------------------------------------------------------

class TestSkipPaperWhite:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()

    def _white_image(self) -> np.ndarray:
        """Solid white image — all pixels land on #FFFFFF in Basic 6."""
        return np.full((16, 16, 3), 255, dtype=np.uint8)

    def test_skip_paper_white_true_drops_white_layer(self):
        img = self._white_image()
        params = _base_params(img, skip_paper_white=True)
        specs = self.gen.generate_layers(params, self.canvas)
        hexes = [s.color.upper() for s in specs]
        assert "#FFFFFF" not in hexes

    def test_skip_paper_white_false_includes_white_layer(self):
        img = self._white_image()
        params = _base_params(img, skip_paper_white=False)
        specs = self.gen.generate_layers(params, self.canvas)
        hexes = [s.color.upper() for s in specs]
        assert "#FFFFFF" in hexes


# ---------------------------------------------------------------------------
# Empty mask dropped
# ---------------------------------------------------------------------------

class TestEmptyMaskDropped:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()

    def test_empty_mask_layers_not_emitted(self):
        """Image contains only black — 4 of 6 Basic 6 colors get empty masks."""
        img = np.zeros((16, 16, 3), dtype=np.uint8)  # pure black
        params = _base_params(img, skip_paper_white=True)
        specs = self.gen.generate_layers(params, self.canvas)
        # Only the black layer should appear.
        assert len(specs) == 1
        assert specs[0].color.upper() == "#000000"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def test_same_seed_same_output(self):
        params = _base_params(self.image, seed=7)
        specs1 = self.gen.generate_layers(params, self.canvas)
        specs2 = self.gen.generate_layers(params, self.canvas)

        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2):
            assert s1.color == s2.color
            assert s1.name == s2.name
            assert len(s1.paths) == len(s2.paths)
            for p1, p2 in zip(s1.paths, s2.paths):
                assert p1 == p2


# ---------------------------------------------------------------------------
# generate() single-layer fallback
# ---------------------------------------------------------------------------

class TestGenerateFallback:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def test_generate_returns_flat_polyline_list(self):
        params = _base_params(self.image)
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        for path in result:
            assert len(path) >= 2


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def setup_method(self):
        from plottter.generators.pointillist import PointillistGenerator

        self.gen = PointillistGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def _run_preset(self, name: str):
        presets = {p.name: p for p in self.gen.get_presets()}
        preset = presets[name]
        params = {**preset.params, "_source_image": self.image}
        return self.gen.generate_layers(params, self.canvas)

    # --- presence ---

    def test_pointillist_classic_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Pointillist Classic" in names

    def test_halftone_dots_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Halftone Dots" in names

    def test_big_cross_stipple_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Big Cross Stipple" in names

    def test_sketchy_mono_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Sketchy Mono" in names

    # --- param values ---

    def test_pointillist_classic_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        classic = presets["Pointillist Classic"]
        assert classic.params["palette"] == "Basic 6"
        assert classic.params["density_per_cm2"] == 250.0
        assert classic.params["dither"] == "floyd-steinberg"
        assert classic.params["dot_style"] == "point"
        assert classic.params["skip_paper_white"] is True

    def test_halftone_dots_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Halftone Dots"]
        assert p.params["palette"] == "Copic 12"
        assert p.params["density_per_cm2"] == 600.0
        assert p.params["dither"] == "ordered"
        assert p.params["dot_style"] == "point"
        assert p.params["dot_size_mm"] == 0.3

    def test_big_cross_stipple_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Big Cross Stipple"]
        assert p.params["palette"] == "Basic 6"
        assert p.params["density_per_cm2"] == 80.0
        assert p.params["dither"] == "floyd-steinberg"
        assert p.params["dot_style"] == "cross"
        assert p.params["dot_size_mm"] == 1.2

    def test_sketchy_mono_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Sketchy Mono"]
        assert p.params["palette"] == "Grayscale 5"
        assert p.params["density_per_cm2"] == 300.0
        assert p.params["dither"] == "none"
        assert p.params["dot_style"] == "point"
        assert p.params["dot_size_mm"] == 0.5

    # --- uniqueness ---

    def test_preset_names_are_unique(self):
        names = [p.name for p in self.gen.get_presets()]
        assert len(names) == len(set(names))

    # --- runs without error and emits >= 1 layer ---

    def test_halftone_dots_runs(self):
        specs = self._run_preset("Halftone Dots")
        assert len(specs) >= 1

    def test_big_cross_stipple_runs(self):
        specs = self._run_preset("Big Cross Stipple")
        assert len(specs) >= 1

    def test_sketchy_mono_runs(self):
        specs = self._run_preset("Sketchy Mono")
        assert len(specs) >= 1
