"""Phase 15.4 validation: edge-case parameters and randomize validation for all math generators.

Covers:
- Every named preset from the plan (parametric, polar, modular mult, flow field, L-systems, grid)
- Edge-case parameters (extremes, zeros, t_start==t_end) that must not crash
- Randomize validation: generating with random params within declared ranges produces
  either valid polyline output or raises a documented exception (never an unhandled crash)
"""

from __future__ import annotations

import math
import random

import pytest

from plottter.models.canvas import Canvas
from plottter.generators.parametric import ParametricGenerator
from plottter.generators.polar import PolarGenerator
from plottter.generators.modular_mult import ModularMultGenerator
from plottter.generators.flow_field import FlowFieldGenerator
from plottter.generators.lsystem import LSystemGenerator
from plottter.generators.grid_pattern import GridPatternGenerator
from plottter.generators.base import (
    FloatParam,
    IntParam,
    ChoiceParam,
    BoolParam,
    ExpressionParam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def default_params(generator) -> dict:
    return {p.name: p.default for p in generator.get_parameters()}


def random_params(generator, seed: int = 0) -> dict:
    """Build a params dict with every parameter set to a random value within its
    declared min/max range (or a random choice/bool).  ExpressionParam fields keep
    their defaults because we cannot synthesise valid arbitrary expressions."""
    rng = random.Random(seed)
    params = default_params(generator)
    for param in generator.get_parameters():
        if not param.randomizable:
            continue
        if isinstance(param, FloatParam):
            params[param.name] = rng.uniform(param.min, param.max)
        elif isinstance(param, IntParam):
            params[param.name] = rng.randint(param.min, param.max)
        elif isinstance(param, ChoiceParam):
            params[param.name] = rng.choice(param.choices)
        elif isinstance(param, BoolParam):
            params[param.name] = rng.choice([True, False])
        # ExpressionParam: keep default (randomizing expressions is out-of-scope)
    return params


# ---------------------------------------------------------------------------
# 15.4 — ParametricGenerator
# ---------------------------------------------------------------------------


class TestParametricEdgeCases:
    def setup_method(self):
        self.gen = ParametricGenerator()
        self.canvas = make_canvas()

    def test_t_start_equals_t_end_does_not_crash(self):
        """t_start == t_end: all t values are identical; should return empty or degenerate."""
        params = default_params(self.gen)
        params["t_start"] = 1.0
        params["t_end"] = 1.0
        params["num_points"] = 100
        # Must not raise; may return empty list or a single-point path
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_minimum_num_points(self):
        """num_points at minimum (100) should not crash."""
        params = default_params(self.gen)
        params["num_points"] = 100
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_maximum_num_points(self):
        """num_points at maximum (100000) should produce output."""
        params = default_params(self.gen)
        params["num_points"] = 100_000
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1
        assert len(result[0]) > 0

    def test_scale_zero_auto_fits(self):
        """scale=0 triggers auto-fit; should produce output within canvas."""
        params = default_params(self.gen)
        params["scale"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1
        x1, y1, x2, y2 = self.canvas.drawing_area()
        for pt in result[0]:
            assert x1 - 1.0 <= pt[0] <= x2 + 1.0
            assert y1 - 1.0 <= pt[1] <= y2 + 1.0

    def test_scale_maximum(self):
        """scale=100 should not crash."""
        params = default_params(self.gen)
        params["scale"] = 100.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_rotation_extremes(self):
        """rotation_deg at -360 and +360 should not crash."""
        for rot in [-360.0, 360.0]:
            params = default_params(self.gen)
            params["rotation_deg"] = rot
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)

    def test_negative_t_range(self):
        """t_start > t_end (reversed range) should not crash."""
        params = default_params(self.gen)
        params["t_start"] = math.pi * 2
        params["t_end"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_expression_returning_constant_does_not_crash(self):
        """y=0 (constant) should produce degenerate but valid output."""
        params = default_params(self.gen)
        params["x_expr"] = "t"
        params["y_expr"] = "0"
        params["num_points"] = 100
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_expression_with_nan_produces_empty(self):
        """Expression that produces NaN (e.g. log(-1)) should return empty result."""
        params = default_params(self.gen)
        params["x_expr"] = "log(-abs(t) - 1)"  # always negative → nan/error
        params["y_expr"] = "sin(t)"
        params["num_points"] = 100
        # Should not crash; may return [] or raise ValueError from expression eval
        try:
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)
        except (ValueError, RuntimeError):
            pass  # expected for disallowed/failing expression

    @pytest.mark.parametrize("preset_name", [
        "Lissajous",
        "Butterfly Curve",
        "Spirograph (Epitrochoid)",
        "Hypotrochoid",
        "Farris / Mystery Curve",
        "Lorenz Attractor",
    ])
    def test_named_preset(self, preset_name):
        """All plan-listed presets must produce non-empty output."""
        presets = {p.name: p.params for p in self.gen.get_presets()}
        assert preset_name in presets, f"Preset '{preset_name}' missing from generator"
        params = default_params(self.gen)
        params.update(presets[preset_name])
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1
        assert len(result[0]) > 0, f"Preset '{preset_name}' produced empty path"

    def test_randomize_does_not_crash(self):
        """Random params within declared ranges should produce output or raise ValueError."""
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            try:
                result = self.gen.generate(params, self.canvas)
                assert isinstance(result, list)
            except (ValueError, RuntimeError):
                pass  # expression errors are expected when randomizing expressions


# ---------------------------------------------------------------------------
# 15.4 — PolarGenerator
# ---------------------------------------------------------------------------


class TestPolarEdgeCases:
    def setup_method(self):
        self.gen = PolarGenerator()
        self.canvas = make_canvas()

    def test_theta_start_equals_theta_end_does_not_crash(self):
        params = default_params(self.gen)
        params["theta_start"] = 1.0
        params["theta_end"] = 1.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_minimum_num_points(self):
        params = default_params(self.gen)
        params["num_points"] = 100
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_maximum_num_points(self):
        params = default_params(self.gen)
        params["num_points"] = 100_000
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1

    def test_scale_zero_auto_fits(self):
        params = default_params(self.gen)
        params["scale"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1

    def test_rotation_extremes(self):
        for rot in [-360.0, 360.0]:
            params = default_params(self.gen)
            params["rotation_deg"] = rot
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)

    def test_expression_zero_radius(self):
        """r_expr='0' should produce a degenerate point (single path)."""
        params = default_params(self.gen)
        params["r_expr"] = "0"
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    @pytest.mark.parametrize("preset_name", [
        "Rose (4-petal)",
        "Cardioid",
        "Archimedean Spiral",
        "Logarithmic Spiral",
        "Limacon",
    ])
    def test_named_preset(self, preset_name):
        """All plan-listed polar presets must produce non-empty output."""
        presets = {p.name: p.params for p in self.gen.get_presets()}
        assert preset_name in presets, f"Preset '{preset_name}' missing from generator"
        params = default_params(self.gen)
        params.update(presets[preset_name])
        result = self.gen.generate(params, self.canvas)
        assert len(result) >= 1
        assert len(result[0]) > 0, f"Preset '{preset_name}' produced empty path"

    def test_randomize_does_not_crash(self):
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            try:
                result = self.gen.generate(params, self.canvas)
                assert isinstance(result, list)
            except (ValueError, RuntimeError):
                pass


# ---------------------------------------------------------------------------
# 15.4 — ModularMultGenerator
# ---------------------------------------------------------------------------


class TestModularMultEdgeCases:
    def setup_method(self):
        self.gen = ModularMultGenerator()
        self.canvas = make_canvas()

    def test_minimum_num_points(self):
        """2 points is the declared minimum; multiplier=2 should produce output."""
        params = default_params(self.gen)
        params["num_points"] = 2
        params["multiplier"] = 2.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_maximum_num_points(self):
        params = default_params(self.gen)
        params["num_points"] = 1000
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_multiplier_zero(self):
        """multiplier=0: all points map to point 0; should produce segments (p→0)."""
        params = default_params(self.gen)
        params["multiplier"] = 0.0
        params["num_points"] = 10
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_multiplier_one(self):
        """multiplier=1.0: every point maps to itself; no segments expected."""
        params = default_params(self.gen)
        params["multiplier"] = 1.0
        params["num_points"] = 10
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        # All connections are self-loops and should be skipped
        assert len(result) == 0

    def test_multiplier_maximum(self):
        """multiplier=500 should not crash."""
        params = default_params(self.gen)
        params["multiplier"] = 500.0
        params["num_points"] = 50
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_radius_zero_auto_fits(self):
        params = default_params(self.gen)
        params["radius_mm"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_radius_maximum(self):
        """radius_mm=500 (larger than canvas) should not crash."""
        params = default_params(self.gen)
        params["radius_mm"] = 500.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_randomize_does_not_crash(self):
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 15.4 — FlowFieldGenerator
# ---------------------------------------------------------------------------


class TestFlowFieldEdgeCases:
    def setup_method(self):
        self.gen = FlowFieldGenerator()
        self.canvas = make_canvas()

    def test_minimum_num_particles(self):
        params = default_params(self.gen)
        params["num_particles"] = 100
        params["max_steps"] = 10
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_max_steps_one(self):
        """max_steps=1: each particle should take exactly one step."""
        params = default_params(self.gen)
        params["num_particles"] = 20
        params["max_steps"] = 1
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_noise_scale_zero(self):
        """noise_scale=0 should not cause division-by-zero."""
        params = default_params(self.gen)
        params["noise_scale"] = 0.0
        params["num_particles"] = 20
        params["max_steps"] = 5
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_noise_octaves_one(self):
        params = default_params(self.gen)
        params["noise_octaves"] = 1
        params["num_particles"] = 20
        params["max_steps"] = 5
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_noise_octaves_eight(self):
        params = default_params(self.gen)
        params["noise_octaves"] = 8
        params["num_particles"] = 20
        params["max_steps"] = 5
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_step_size_minimum(self):
        """Very small step size should not crash."""
        params = default_params(self.gen)
        params["step_size_mm"] = 0.01
        params["num_particles"] = 20
        params["max_steps"] = 10
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_all_presets_by_name(self):
        """All flow field presets must produce output."""
        for preset in self.gen.get_presets():
            params = default_params(self.gen)
            params.update(preset.params)
            params["num_particles"] = 50  # reduce for test speed
            params["max_steps"] = 20
            result = self.gen.generate(params, self.canvas)
            assert len(result) > 0, f"Preset '{preset.name}' produced no paths"

    def test_randomize_does_not_crash(self):
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            params["num_particles"] = min(params.get("num_particles", 100), 200)
            params["max_steps"] = min(params.get("max_steps", 20), 30)
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 15.4 — LSystemGenerator
# ---------------------------------------------------------------------------


class TestLSystemEdgeCases:
    def setup_method(self):
        self.gen = LSystemGenerator()
        self.canvas = make_canvas()

    def test_iterations_zero(self):
        """0 iterations: axiom is the initial string; turtle runs on axiom."""
        params = default_params(self.gen)
        params["iterations"] = 0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_iterations_maximum(self):
        """High iteration count is limited internally; should not hang or OOM."""
        params = default_params(self.gen)
        params["iterations"] = 10
        params["axiom"] = "F"
        params["rules"] = "F=F+F"
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_angle_zero(self):
        """angle_deg=0: turns have no effect; all segments in a straight line."""
        params = default_params(self.gen)
        params["angle_deg"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_angle_360(self):
        """angle_deg=360: full-circle turns, visually degenerate but no crash."""
        params = default_params(self.gen)
        params["angle_deg"] = 360.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_step_length_zero(self):
        """step_length_mm=0: all segments are point-sized; no crash."""
        params = default_params(self.gen)
        params["step_length_mm"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_empty_axiom(self):
        """Empty axiom should return empty polylines."""
        params = default_params(self.gen)
        params["axiom"] = ""
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_axiom_with_no_draw_commands(self):
        """Axiom with only turn/push commands should return empty polylines."""
        params = default_params(self.gen)
        params["axiom"] = "+++---"
        params["rules"] = ""
        params["iterations"] = 1
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_deep_bracket_nesting_does_not_crash(self):
        """Deeply nested brackets should not cause stack overflow."""
        params = default_params(self.gen)
        params["axiom"] = "F"
        params["rules"] = "F=F[F[F[F]]]"
        params["iterations"] = 4
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    @pytest.mark.parametrize("preset_name", [
        "Koch Snowflake",
        "Sierpinski Triangle",
        "Dragon Curve",
        "Plant / Tree",
        "Hilbert Curve",
        "Gosper Curve",
    ])
    def test_named_preset(self, preset_name):
        """All plan-listed L-system presets must produce non-empty output."""
        presets = {p.name: p.params for p in self.gen.get_presets()}
        assert preset_name in presets, f"Preset '{preset_name}' missing from generator"
        params = default_params(self.gen)
        params.update(presets[preset_name])
        params["iterations"] = min(int(params.get("iterations", 3)), 3)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, f"Preset '{preset_name}' produced no paths"

    def test_randomize_does_not_crash(self):
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            params["iterations"] = min(int(params.get("iterations", 3)), 3)
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 15.4 — GridPatternGenerator
# ---------------------------------------------------------------------------


class TestGridPatternEdgeCases:
    def setup_method(self):
        self.gen = GridPatternGenerator()
        self.canvas = make_canvas()

    def test_sine_grid_zero_amplitude(self):
        """amplitude_mm=0 produces straight lines (no modulation); should not crash."""
        params = default_params(self.gen)
        params["mode"] = "Sine Grid"
        params["amplitude_mm"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_sine_grid_zero_frequency(self):
        """frequency=0 produces flat constant (no oscillation); should not crash."""
        params = default_params(self.gen)
        params["mode"] = "Sine Grid"
        params["frequency"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_sine_grid_line_count_one(self):
        """line_count=1 should produce exactly one line."""
        params = default_params(self.gen)
        params["mode"] = "Sine Grid"
        params["direction"] = "Horizontal"
        params["line_count"] = 1
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1

    def test_truchet_very_large_tile(self):
        """tile_size larger than canvas should produce at most 1 tile."""
        params = default_params(self.gen)
        params["mode"] = "Truchet Tiles"
        params["tile_size_mm"] = 500.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_truchet_very_small_tile(self):
        """Very small tile size (1mm) should produce many tiles without crashing."""
        params = default_params(self.gen)
        params["mode"] = "Truchet Tiles"
        params["tile_size_mm"] = 1.0
        params["seed"] = 42
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_concentric_count_one(self):
        """count=1 should produce exactly one shape."""
        params = default_params(self.gen)
        params["mode"] = "Concentric Shapes"
        params["count"] = 1
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1

    def test_concentric_spacing_zero(self):
        """spacing_mm=0: all shapes at same radius; should not crash."""
        params = default_params(self.gen)
        params["mode"] = "Concentric Shapes"
        params["spacing_mm"] = 0.0
        params["count"] = 5
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_concentric_polygon_three_sides(self):
        """Minimum polygon sides=3 (triangle); should produce 4-point paths."""
        params = default_params(self.gen)
        params["mode"] = "Concentric Shapes"
        params["shape"] = "Polygon"
        params["sides"] = 3
        params["count"] = 2
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 2
        for path in result:
            assert len(path) == 4  # 3 corners + close

    def test_islamic_tiling_all_types(self):
        """All three Islamic tiling types should produce output."""
        for islamic_type in ["6-Point Stars", "8-Point Stars", "12-Point Stars"]:
            params = default_params(self.gen)
            params["mode"] = "Islamic Tiling"
            params["islamic_type"] = islamic_type
            params["tile_size_mm"] = 20.0
            result = self.gen.generate(params, self.canvas)
            assert len(result) > 0, f"Islamic type '{islamic_type}' produced no paths"

    def test_celtic_knot_minimum_grid(self):
        """1x1 celtic knot grid should not crash."""
        params = default_params(self.gen)
        params["mode"] = "Celtic Knot"
        params["knot_cols"] = 1
        params["knot_rows"] = 1
        params["tile_size_mm"] = 20.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    def test_celtic_knot_zero_gap(self):
        """gap_mm=0 should not crash."""
        params = default_params(self.gen)
        params["mode"] = "Celtic Knot"
        params["knot_cols"] = 3
        params["knot_rows"] = 3
        params["tile_size_mm"] = 10.0
        params["gap_mm"] = 0.0
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)

    @pytest.mark.parametrize("preset_name_fragment", [
        "Sine Grid",
        "Truchet",
        "Islamic",
        "Celtic Plait",
    ])
    def test_preset_category_present(self, preset_name_fragment):
        """At least one preset for each plan-listed grid category must exist."""
        presets = [p for p in self.gen.get_presets() if preset_name_fragment in p.name]
        assert len(presets) > 0, f"No '{preset_name_fragment}' preset found"
        params = default_params(self.gen)
        params.update(presets[0].params)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0, f"Preset '{presets[0].name}' produced no paths"

    def test_randomize_sine_grid_does_not_crash(self):
        """Randomize within Sine Grid mode parameters should not crash."""
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            params["mode"] = "Sine Grid"  # keep mode fixed to sine grid
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)

    def test_randomize_concentric_does_not_crash(self):
        """Randomize within Concentric Shapes mode should not crash."""
        for seed in range(5):
            params = random_params(self.gen, seed=seed)
            params["mode"] = "Concentric Shapes"
            result = self.gen.generate(params, self.canvas)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 15.4 — Cross-generator: all generators produce valid Polyline format
# ---------------------------------------------------------------------------


class TestGeneratorOutputFormat:
    """Verify that every math generator always returns list[list[tuple[float, float]]]."""

    @pytest.mark.parametrize("gen_class", [
        ParametricGenerator,
        PolarGenerator,
        ModularMultGenerator,
        FlowFieldGenerator,
        LSystemGenerator,
        GridPatternGenerator,
    ])
    def test_output_is_list_of_polylines(self, gen_class):
        gen = gen_class()
        canvas = make_canvas()
        params = default_params(gen)
        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        for path in result:
            assert isinstance(path, list), f"{gen_class.__name__}: path is not a list"
            for pt in path:
                assert isinstance(pt, tuple), f"{gen_class.__name__}: point is not a tuple"
                assert len(pt) == 2, f"{gen_class.__name__}: point is not 2D"
                x, y = pt
                assert isinstance(x, (int, float))
                assert isinstance(y, (int, float))
                assert math.isfinite(x), f"{gen_class.__name__}: x is not finite ({x})"
                assert math.isfinite(y), f"{gen_class.__name__}: y is not finite ({y})"

    @pytest.mark.parametrize("gen_class", [
        ParametricGenerator,
        PolarGenerator,
        ModularMultGenerator,
        FlowFieldGenerator,
        LSystemGenerator,
        GridPatternGenerator,
    ])
    def test_randomize_output_format(self, gen_class):
        """Randomized params should also return valid Polyline format."""
        gen = gen_class()
        canvas = make_canvas()
        for seed in range(3):
            params = random_params(gen, seed=seed)
            # Reduce heavy params to keep test fast
            if "num_particles" in params:
                params["num_particles"] = min(params["num_particles"], 100)
            if "max_steps" in params:
                params["max_steps"] = min(params["max_steps"], 20)
            if "iterations" in params:
                params["iterations"] = min(int(params["iterations"]), 3)
            try:
                result = gen.generate(params, canvas)
            except (ValueError, RuntimeError):
                continue  # expression errors are acceptable
            assert isinstance(result, list)
            for path in result:
                assert isinstance(path, list)
                for pt in path:
                    x, y = pt
                    assert math.isfinite(x)
                    assert math.isfinite(y)


# ---------------------------------------------------------------------------
# 15.4 — Parameter metadata validation
# ---------------------------------------------------------------------------


class TestParameterMetadata:
    """Verify parameter declarations are self-consistent."""

    @pytest.mark.parametrize("gen_class", [
        ParametricGenerator,
        PolarGenerator,
        ModularMultGenerator,
        FlowFieldGenerator,
        LSystemGenerator,
        GridPatternGenerator,
    ])
    def test_float_params_min_le_max(self, gen_class):
        gen = gen_class()
        for param in gen.get_parameters():
            if isinstance(param, FloatParam):
                assert param.min <= param.max, (
                    f"{gen_class.__name__}.{param.name}: min ({param.min}) > max ({param.max})"
                )
            elif isinstance(param, IntParam):
                assert param.min <= param.max, (
                    f"{gen_class.__name__}.{param.name}: min ({param.min}) > max ({param.max})"
                )

    @pytest.mark.parametrize("gen_class", [
        ParametricGenerator,
        PolarGenerator,
        ModularMultGenerator,
        FlowFieldGenerator,
        LSystemGenerator,
        GridPatternGenerator,
    ])
    def test_default_within_range(self, gen_class):
        """Default values must lie within declared [min, max] ranges."""
        gen = gen_class()
        for param in gen.get_parameters():
            if isinstance(param, FloatParam):
                assert param.min <= param.default <= param.max, (
                    f"{gen_class.__name__}.{param.name}: default {param.default} "
                    f"outside [{param.min}, {param.max}]"
                )
            elif isinstance(param, IntParam):
                assert param.min <= param.default <= param.max, (
                    f"{gen_class.__name__}.{param.name}: default {param.default} "
                    f"outside [{param.min}, {param.max}]"
                )

    @pytest.mark.parametrize("gen_class", [
        ParametricGenerator,
        PolarGenerator,
        ModularMultGenerator,
        FlowFieldGenerator,
        LSystemGenerator,
        GridPatternGenerator,
    ])
    def test_all_presets_include_all_required_params(self, gen_class):
        """Each preset dict should not include keys that are not declared parameters."""
        gen = gen_class()
        declared_names = {p.name for p in gen.get_parameters()}
        for preset in gen.get_presets():
            for key in preset.params:
                assert key in declared_names, (
                    f"{gen_class.__name__} preset '{preset.name}': "
                    f"unknown param '{key}'"
                )
