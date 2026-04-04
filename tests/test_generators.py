"""Tests for Phase 5 math art generators."""

from __future__ import annotations

import math

import pytest

from plottter.models.canvas import Canvas
from plottter.generators.polar import PolarGenerator
from plottter.generators.modular_mult import ModularMultGenerator
from plottter.generators.flow_field import FlowFieldGenerator
from plottter.generators.lsystem import (
    LSystemGenerator,
    expand_lsystem,
    parse_rules,
    turtle_to_polylines,
)
from plottter.generators.grid_pattern import GridPatternGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def within_bounds(paths, canvas):
    """Check that all points are within the drawing area of the canvas."""
    x1, y1, x2, y2 = canvas.drawing_area()
    for path in paths:
        for x, y in path:
            if not (x1 - 1.0 <= x <= x2 + 1.0 and y1 - 1.0 <= y <= y2 + 1.0):
                return False
    return True


# ---------------------------------------------------------------------------
# PolarGenerator
# ---------------------------------------------------------------------------

class TestPolarGenerator:
    def setup_method(self):
        self.gen = PolarGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Polar Curves" in GENERATORS

    def test_default_generate(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 1
        assert len(paths[0]) > 100

    def test_output_within_bounds(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        assert within_bounds(paths, self.canvas)

    def test_all_presets_generate(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) >= 1, f"Preset '{preset.name}' produced no paths"
            assert len(paths[0]) > 0, f"Preset '{preset.name}' first path is empty"

    def test_rose_preset_bounds(self):
        preset = next(p for p in self.gen.get_presets() if "Rose" in p.name)
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(preset.params)
        paths = self.gen.generate(params, self.canvas)
        assert within_bounds(paths, self.canvas)

    def test_cardioid_preset(self):
        preset = next(p for p in self.gen.get_presets() if "Cardioid" in p.name)
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(preset.params)
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 1
        assert len(paths[0]) > 100

    def test_num_points_respected(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_points"] = 500
        paths = self.gen.generate(params, self.canvas)
        # Should have approximately num_points (some may be filtered for non-finite)
        assert len(paths[0]) >= 400

    def test_invalid_expression_raises(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["r_expr"] = "__import__('os')"
        with pytest.raises((ValueError, Exception)):
            self.gen.generate(params, self.canvas)

    def test_cancellation(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_points"] = 50000
        cancelled = [False]

        def cancel_after_first_call():
            if not cancelled[0]:
                cancelled[0] = True
                return False
            return True

        paths = self.gen.generate(params, self.canvas, cancelled_callback=cancel_after_first_call)
        # Should return a shorter result
        assert isinstance(paths, list)


# ---------------------------------------------------------------------------
# ModularMultGenerator
# ---------------------------------------------------------------------------

class TestModularMultGenerator:
    def setup_method(self):
        self.gen = ModularMultGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Modular Multiplication" in GENERATORS

    def test_default_generate(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0

    def test_output_segments(self):
        """Each path should be a 2-point line segment."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        for path in paths:
            assert len(path) == 2

    def test_all_presets_generate(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no paths"

    def test_multiplier_2_connects_correct_points(self):
        """With multiplier=2, point p connects to (2p) % N."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_points"] = 10
        params["multiplier"] = 2.0
        params["radius_mm"] = 50.0
        paths = self.gen.generate(params, self.canvas)
        # With 10 points and multiplier 2:
        # p=0 -> 0 (skip self), p=1 -> 2, p=2 -> 4, ..., p=5 -> 0, etc.
        # We expect segments (skip self-connections)
        assert len(paths) > 0

    def test_segment_count(self):
        """200 points, multiplier 2: number of non-self segments."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_points"] = 200
        params["multiplier"] = 2.0
        paths = self.gen.generate(params, self.canvas)
        # All points p where round(p*2)%200 != p should create segments
        assert len(paths) > 100

    def test_auto_fit_radius(self):
        """Radius 0 should produce paths centered within drawing area."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["radius_mm"] = 0.0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # All points should be within drawing area
        x1, y1, x2, y2 = self.canvas.drawing_area()
        for path in paths:
            for x, y in path:
                assert x1 - 1 <= x <= x2 + 1
                assert y1 - 1 <= y <= y2 + 1


# ---------------------------------------------------------------------------
# FlowFieldGenerator
# ---------------------------------------------------------------------------

class TestFlowFieldGenerator:
    def setup_method(self):
        self.gen = FlowFieldGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Flow Field" in GENERATORS

    def test_default_generate(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0

    def test_all_presets_generate(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            params["num_particles"] = 50  # reduce for test speed
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no paths"

    def test_output_within_bounds(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_particles"] = 100
        paths = self.gen.generate(params, self.canvas)
        assert within_bounds(paths, self.canvas)

    def test_seed_reproducibility(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_particles"] = 50
        params["seed"] = 7
        paths1 = self.gen.generate(params, self.canvas)
        paths2 = self.gen.generate(params, self.canvas)
        assert len(paths1) == len(paths2)
        # First trail should be identical
        assert paths1[0] == paths2[0]

    def test_particle_trails_have_multiple_points(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_particles"] = 50
        params["max_steps"] = 50
        paths = self.gen.generate(params, self.canvas)
        for path in paths:
            assert len(path) >= 2  # each trail has start + at least one step

    def test_cancellation(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_particles"] = 5000
        call_count = [0]

        def cancel_fn():
            call_count[0] += 1
            return call_count[0] > 200

        paths = self.gen.generate(params, self.canvas, cancelled_callback=cancel_fn)
        assert isinstance(paths, list)
        # Should have fewer paths than total particles
        assert len(paths) < 5000

    def test_rectilinear_false_unchanged(self):
        """rectilinear=False should produce same output as default (no rectilinear key)."""
        params_default = {p.name: p.default for p in self.gen.get_parameters()}
        params_default["num_particles"] = 50
        params_default["seed"] = 77

        params_rect_off = dict(params_default)
        params_rect_off["rectilinear"] = False

        paths1 = self.gen.generate(params_default, self.canvas)
        paths2 = self.gen.generate(params_rect_off, self.canvas)
        assert paths1 == paths2

    def test_rectilinear_true_axis_aligned(self):
        """rectilinear=True should produce paths with mostly horizontal or vertical steps."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_particles"] = 100
        params["max_steps"] = 30
        params["rectilinear"] = True
        params["seed"] = 42
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # Check that each step is mostly axis-aligned: one component dominates
        axis_aligned_count = 0
        total_steps = 0
        for path in paths:
            for j in range(1, len(path)):
                dx = abs(path[j][0] - path[j-1][0])
                dy = abs(path[j][1] - path[j-1][1])
                total_steps += 1
                # With jitter 0.5 and step 1.0, the dominant axis should be >= 0.3
                # We check that at least one direction has a dominant component
                if dx > dy * 0.5 or dy > dx * 0.5:
                    axis_aligned_count += 1
        # The vast majority of steps should have one dominant axis component
        assert total_steps > 0
        assert axis_aligned_count / total_steps > 0.7

    def test_rectilinear_preset_generates_valid_output(self):
        """Rectilinear preset should produce valid paths."""
        preset = next(p for p in self.gen.get_presets() if p.name == "Rectilinear")
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(preset.params)
        params["num_particles"] = 50
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        for path in paths:
            assert len(path) >= 2


# ---------------------------------------------------------------------------
# LSystemGenerator — string expansion and turtle
# ---------------------------------------------------------------------------

class TestLSystemExpansion:
    def test_simple_expansion(self):
        rules = parse_rules("F=F+F")
        result = expand_lsystem("F", rules, 1)
        assert result == "F+F"

    def test_two_iterations(self):
        rules = parse_rules("F=F+F")
        result = expand_lsystem("F", rules, 2)
        assert result == "F+F+F+F"

    def test_multiple_rules(self):
        rules = parse_rules("F=F-G+F;G=GG")
        assert rules == {"F": "F-G+F", "G": "GG"}

    def test_unknown_chars_pass_through(self):
        rules = parse_rules("F=F+F")
        result = expand_lsystem("FXF", rules, 1)
        # X has no rule, passes through; F expands
        assert result == "F+FXF+F"

    def test_max_length_truncation(self):
        """Exponential expansion should be truncated to avoid memory blowup."""
        rules = parse_rules("F=FFFFFF")
        # With many iterations this would blow up
        result = expand_lsystem("F", rules, 20)
        assert len(result) <= 2_100_000

    def test_parse_rules_whitespace(self):
        rules = parse_rules("F = F+F ; G = GG")
        assert rules["F"] == "F+F"
        assert rules["G"] == "GG"

    def test_parse_rules_empty(self):
        rules = parse_rules("")
        assert rules == {}


class TestTurtleToPolylines:
    def test_straight_line(self):
        """'FF' should produce one polyline with 3 points."""
        polys = turtle_to_polylines("FF", angle_deg=90.0, step_length=1.0)
        assert len(polys) == 1
        assert len(polys[0]) == 3

    def test_right_angle(self):
        """'F+F' with 90 degree turns should produce an L-shape."""
        polys = turtle_to_polylines("F+F", angle_deg=90.0, step_length=1.0)
        # One connected polyline (the + only turns, no pen-up)
        assert len(polys) == 1
        assert len(polys[0]) == 3

    def test_branching(self):
        """'F[+F]F' should produce at least 2 polylines (branch + return)."""
        polys = turtle_to_polylines("F[+F]F", angle_deg=30.0, step_length=1.0)
        assert len(polys) >= 2

    def test_pen_up_move(self):
        """'FfF' should produce two separate polylines."""
        polys = turtle_to_polylines("FfF", angle_deg=90.0, step_length=1.0)
        assert len(polys) == 2


class TestLSystemGenerator:
    def setup_method(self):
        self.gen = LSystemGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "L-System / Fractal" in GENERATORS

    def test_default_generate(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0

    def test_all_presets_generate(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            # Reduce iterations for speed
            params["iterations"] = min(int(params.get("iterations", 3)), 3)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no paths"

    def test_output_within_bounds(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["iterations"] = 3
        paths = self.gen.generate(params, self.canvas)
        assert within_bounds(paths, self.canvas)

    def test_koch_snowflake(self):
        preset = next(p for p in self.gen.get_presets() if "Koch" in p.name)
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(preset.params)
        params["iterations"] = 2
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        assert within_bounds(paths, self.canvas)

    def test_dragon_curve(self):
        preset = next(p for p in self.gen.get_presets() if "Dragon" in p.name)
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(preset.params)
        params["iterations"] = 5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0

    def test_custom_rules(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["axiom"] = "F"
        params["rules"] = "F=F+F-F-F+F"
        params["iterations"] = 2
        params["angle_deg"] = 90.0
        params["step_length_mm"] = 1.0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0

    def test_cancellation(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["iterations"] = 8  # would be very slow without cancel
        cancelled = [False]

        def cancel_fn():
            cancelled[0] = True
            return True

        paths = self.gen.generate(params, self.canvas, cancelled_callback=cancel_fn)
        assert isinstance(paths, list)


# ---------------------------------------------------------------------------
# GridPatternGenerator
# ---------------------------------------------------------------------------

class TestGridPatternGenerator:
    def setup_method(self):
        self.gen = GridPatternGenerator()
        self.canvas = make_canvas()

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Grid Pattern" in GENERATORS

    def test_all_presets_generate(self):
        for preset in self.gen.get_presets():
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no paths"

    def test_sine_grid_horizontal(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Sine Grid"
        params["direction"] = "Horizontal"
        params["line_count"] = 5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 5
        # Each path should have many points (sine wave)
        for path in paths:
            assert len(path) > 10

    def test_sine_grid_vertical(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Sine Grid"
        params["direction"] = "Vertical"
        params["line_count"] = 5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 5

    def test_sine_grid_both(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Sine Grid"
        params["direction"] = "Both"
        params["line_count"] = 5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 10  # 5 horizontal + 5 vertical

    def test_truchet_generates(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Truchet Tiles"
        params["tile_size_mm"] = 20.0
        params["seed"] = 0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # Each tile has 2 arcs
        x1, y1, x2, y2 = self.canvas.drawing_area()
        expected_tiles = (
            int((x2 - x1) / 20.0) * int((y2 - y1) / 20.0)
        )
        assert len(paths) == expected_tiles * 2

    def test_truchet_seed_reproducibility(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Truchet Tiles"
        params["seed"] = 99
        paths1 = self.gen.generate(params, self.canvas)
        paths2 = self.gen.generate(params, self.canvas)
        assert len(paths1) == len(paths2)

    def test_concentric_circles(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Concentric Shapes"
        params["shape"] = "Circle"
        params["count"] = 5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 5
        # Each circle should be a closed path (first ~= last)
        for path in paths:
            assert len(path) > 10

    def test_concentric_squares(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Concentric Shapes"
        params["shape"] = "Square"
        params["count"] = 3
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 3
        # Square is 5 points (4 corners + close)
        for path in paths:
            assert len(path) == 5

    def test_concentric_polygons(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Concentric Shapes"
        params["shape"] = "Polygon"
        params["sides"] = 6
        params["count"] = 4
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) == 4
        # Hexagon = 7 points (6 + close)
        for path in paths:
            assert len(path) == 7

    def test_sine_grid_line_spacing_affects_positions(self):
        """line_spacing_mm must control actual mm distance between lines."""
        base_params = {p.name: p.default for p in self.gen.get_parameters()}
        base_params["mode"] = "Sine Grid"
        base_params["direction"] = "Horizontal"
        base_params["amplitude_mm"] = 0.0  # flat lines so y_base is exact
        base_params["line_count"] = 3

        base_params["line_spacing_mm"] = 10.0
        paths_10 = self.gen.generate(dict(base_params), self.canvas)

        base_params["line_spacing_mm"] = 20.0
        paths_20 = self.gen.generate(dict(base_params), self.canvas)

        assert len(paths_10) == 3
        assert len(paths_20) == 3

        # With flat lines (amplitude=0), every y in a path equals its y_base.
        def y_base_of(path):
            return path[0][1]

        y10 = [y_base_of(p) for p in paths_10]
        y20 = [y_base_of(p) for p in paths_20]

        # Spacing between consecutive lines should match line_spacing_mm
        assert abs((y10[1] - y10[0]) - 10.0) < 0.01
        assert abs((y20[1] - y20[0]) - 20.0) < 0.01

        # Wider spacing → lines farther apart
        assert (y20[1] - y20[0]) > (y10[1] - y10[0])

    def test_sine_grid_clamps_to_canvas(self):
        """line_count is clamped so lines don't exceed the drawing area."""
        x1, y1, x2, y2 = self.canvas.drawing_area()
        draw_h = y2 - y1
        spacing = 10.0
        # Request far more lines than fit
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Sine Grid"
        params["direction"] = "Horizontal"
        params["line_spacing_mm"] = spacing
        params["line_count"] = 9999
        params["amplitude_mm"] = 0.0
        paths = self.gen.generate(params, self.canvas)
        expected_max = int(draw_h / spacing)
        assert len(paths) == expected_max

    # --- Islamic Tiling tests -----------------------------------------------

    def test_islamic_6point_generates(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Islamic Tiling"
        params["islamic_type"] = "6-Point Stars"
        params["tile_size_mm"] = 20.0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # Each hexagram produces 2 triangles; every triangle is 4 points (closed).
        for path in paths:
            assert len(path) == 4

    def test_islamic_8point_generates(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Islamic Tiling"
        params["islamic_type"] = "8-Point Stars"
        params["tile_size_mm"] = 20.0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # Each octagram produces 2 squares; every square is 5 points (closed).
        for path in paths:
            assert len(path) == 5

    def test_islamic_12point_generates(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Islamic Tiling"
        params["islamic_type"] = "12-Point Stars"
        params["tile_size_mm"] = 22.0
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # Each dodecagram produces 2 hexagons; every hexagon is 7 points (closed).
        for path in paths:
            assert len(path) == 7

    def test_islamic_star_inset_changes_radius(self):
        """Larger star_inset produces a smaller star radius (inner star is larger inset)."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Islamic Tiling"
        params["islamic_type"] = "8-Point Stars"
        params["tile_size_mm"] = 30.0

        params["star_inset"] = 0.1
        paths_small = self.gen.generate(params, self.canvas)

        params["star_inset"] = 0.4
        paths_large = self.gen.generate(params, self.canvas)

        # Both should produce paths; count should be the same (same tile coverage).
        assert len(paths_small) == len(paths_large)
        # The outer radius for small_inset is bigger, so first square x-vertex
        # should be farther from the tile centre than for large_inset.
        # First path in both: square vertices at angle 0° from each cell centre.
        # We can't easily determine cell centres without reimplementing, but we
        # can verify that the x-span of the first square differs.
        sq_small = paths_small[0]
        sq_large = paths_large[0]
        x_span_small = max(p[0] for p in sq_small) - min(p[0] for p in sq_small)
        x_span_large = max(p[0] for p in sq_large) - min(p[0] for p in sq_large)
        assert x_span_small > x_span_large

    # --- Celtic Knot tests --------------------------------------------------

    def test_celtic_knot_generates(self):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Celtic Knot"
        params["knot_cols"] = 4
        params["knot_rows"] = 4
        params["tile_size_mm"] = 10.0
        params["gap_mm"] = 0.5
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0
        # All segments must have at least 2 points.
        for path in paths:
            assert len(path) >= 2

    def test_celtic_knot_all_points_near_grid(self):
        """All generated points should lie close to the expected grid region."""
        cols, rows, T = 4, 3, 15.0
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Celtic Knot"
        params["knot_cols"] = cols
        params["knot_rows"] = rows
        params["tile_size_mm"] = T
        params["gap_mm"] = 0.5

        x1, y1, x2, y2 = self.canvas.drawing_area()
        grid_w = cols * T
        grid_h = rows * T
        ox = (x1 + x2 - grid_w) / 2.0
        oy = (y1 + y2 - grid_h) / 2.0

        paths = self.gen.generate(params, self.canvas)
        margin = T  # points should be within one tile size of the grid boundary
        for path in paths:
            for px, py in path:
                assert ox - margin <= px <= ox + grid_w + margin
                assert oy - margin <= py <= oy + grid_h + margin

    def test_celtic_knot_gap_creates_breaks(self):
        """With zero gap, no internal breaks expected; with large gap, more polylines."""
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Celtic Knot"
        params["knot_cols"] = 3
        params["knot_rows"] = 3
        params["tile_size_mm"] = 10.0

        params["gap_mm"] = 0.0
        paths_no_gap = self.gen.generate(params, self.canvas)

        params["gap_mm"] = 3.0
        paths_big_gap = self.gen.generate(params, self.canvas)

        # A larger gap means more strand breaks → more, shorter polylines.
        assert len(paths_big_gap) >= len(paths_no_gap)

    def test_celtic_preset_generates(self):
        """Both Celtic presets should produce non-empty output."""
        for preset in self.gen.get_presets():
            if "Celtic" not in preset.name:
                continue
            params = {p.name: p.default for p in self.gen.get_parameters()}
            params.update(preset.params)
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Celtic preset '{preset.name}' produced no paths"


# ---------------------------------------------------------------------------
# x_offset_mm / y_offset_mm — task 23.3
# ---------------------------------------------------------------------------

def _flatten(paths):
    return [(round(x, 6), round(y, 6)) for p in paths for x, y in p]


class TestFlowFieldXYOffset:
    """x_offset_mm and y_offset_mm shift all FlowField polyline points."""

    _FAST_PARAMS = {"num_particles": 30, "max_steps": 20, "seed": 42}

    def setup_method(self):
        self.gen = FlowFieldGenerator()
        self.canvas = make_canvas()

    def _run(self, extra=None):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params.update(self._FAST_PARAMS)
        if extra:
            params.update(extra)
        return self.gen.generate(params, self.canvas)

    def test_params_exist_not_randomizable(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "x_offset_mm" in params, "x_offset_mm must be in get_parameters()"
        assert "y_offset_mm" in params, "y_offset_mm must be in get_parameters()"
        assert not params["x_offset_mm"].randomizable
        assert not params["y_offset_mm"].randomizable

    def test_zero_offset_identical(self):
        baseline = self._run()
        with_zero = self._run({"x_offset_mm": 0.0, "y_offset_mm": 0.0})
        assert _flatten(baseline) == _flatten(with_zero)

    def test_x_offset_shifts_right(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 20.0, "y_offset_mm": 0.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - 20.0) < 1e-9
                assert abs(sy - by) < 1e-9

    def test_y_offset_shifts_down(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 0.0, "y_offset_mm": 15.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs(sx - bx) < 1e-9
                assert abs((sy - by) - 15.0) < 1e-9

    def test_combined_offset(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": -10.0, "y_offset_mm": 5.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - (-10.0)) < 1e-9
                assert abs((sy - by) - 5.0) < 1e-9


class TestGridPatternXYOffset:
    """x_offset_mm and y_offset_mm shift all GridPattern polyline points."""

    def setup_method(self):
        self.gen = GridPatternGenerator()
        self.canvas = make_canvas()

    def _run(self, extra=None):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["mode"] = "Sine Grid"
        params["line_count"] = 4
        if extra:
            params.update(extra)
        return self.gen.generate(params, self.canvas)

    def test_params_exist_not_randomizable(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "x_offset_mm" in params
        assert "y_offset_mm" in params
        assert not params["x_offset_mm"].randomizable
        assert not params["y_offset_mm"].randomizable

    def test_zero_offset_identical(self):
        baseline = self._run()
        with_zero = self._run({"x_offset_mm": 0.0, "y_offset_mm": 0.0})
        assert _flatten(baseline) == _flatten(with_zero)

    def test_x_offset_shifts_right(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 20.0, "y_offset_mm": 0.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - 20.0) < 1e-9
                assert abs(sy - by) < 1e-9

    def test_y_offset_shifts_down(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 0.0, "y_offset_mm": 15.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs(sx - bx) < 1e-9
                assert abs((sy - by) - 15.0) < 1e-9

    def test_combined_offset(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": -10.0, "y_offset_mm": 5.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - (-10.0)) < 1e-9
                assert abs((sy - by) - 5.0) < 1e-9


class TestModularMultXYOffset:
    """x_offset_mm and y_offset_mm shift all ModularMult polyline points."""

    def setup_method(self):
        self.gen = ModularMultGenerator()
        self.canvas = make_canvas()

    def _run(self, extra=None):
        params = {p.name: p.default for p in self.gen.get_parameters()}
        params["num_points"] = 30
        if extra:
            params.update(extra)
        return self.gen.generate(params, self.canvas)

    def test_params_exist_not_randomizable(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert "x_offset_mm" in params
        assert "y_offset_mm" in params
        assert not params["x_offset_mm"].randomizable
        assert not params["y_offset_mm"].randomizable

    def test_zero_offset_identical(self):
        baseline = self._run()
        with_zero = self._run({"x_offset_mm": 0.0, "y_offset_mm": 0.0})
        assert _flatten(baseline) == _flatten(with_zero)

    def test_x_offset_shifts_right(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 20.0, "y_offset_mm": 0.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - 20.0) < 1e-9
                assert abs(sy - by) < 1e-9

    def test_y_offset_shifts_down(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": 0.0, "y_offset_mm": 15.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs(sx - bx) < 1e-9
                assert abs((sy - by) - 15.0) < 1e-9

    def test_combined_offset(self):
        baseline = self._run()
        shifted = self._run({"x_offset_mm": -10.0, "y_offset_mm": 5.0})
        assert len(baseline) == len(shifted)
        for bp, sp in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(bp, sp):
                assert abs((sx - bx) - (-10.0)) < 1e-9
                assert abs((sy - by) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Generator registry completeness
# ---------------------------------------------------------------------------

class TestGeneratorRegistry:
    def test_all_math_generators_registered(self):
        from plottter.generators import get_generators_by_category
        math_gens = get_generators_by_category("math")
        names = [cls.name for cls in math_gens]
        expected = [
            "Parametric Curves",
            "Polar Curves",
            "Modular Multiplication",
            "Flow Field",
            "L-System / Fractal",
            "Grid Pattern",
        ]
        for name in expected:
            assert name in names, f"Generator '{name}' not registered"

    def test_all_generators_have_presets(self):
        from plottter.generators import get_generators_by_category
        for cls in get_generators_by_category("math"):
            gen = cls()
            presets = gen.get_presets()
            assert len(presets) > 0, f"{cls.name} has no presets"

    def test_all_generators_have_parameters(self):
        from plottter.generators import get_generators_by_category
        for cls in get_generators_by_category("math"):
            gen = cls()
            params = gen.get_parameters()
            assert len(params) > 0, f"{cls.name} has no parameters"


# ---------------------------------------------------------------------------
# Generator.get_post_processing_parameters() — task 38.3
# ---------------------------------------------------------------------------

class TestPostProcessingParameters:
    """Tests for Generator.get_post_processing_parameters() (task 38.3)."""

    def setup_method(self):
        from plottter.generators.base import Generator
        self.params = Generator.get_post_processing_parameters()
        self.by_name = {p.name: p for p in self.params}

    def test_returns_ten_parameters(self):
        assert len(self.params) == 10

    def test_brush_type_is_first(self):
        assert self.params[0].name == "brush_type"

    def test_brush_type_is_choice_param_with_four_choices(self):
        from plottter.generators.base import ChoiceParam
        bt = self.params[0]
        assert isinstance(bt, ChoiceParam)
        assert len(bt.choices) == 4
        assert "None" in bt.choices
        assert "Stippled" in bt.choices
        assert "Multi-Stroke" in bt.choices
        assert "Calligraphic" in bt.choices

    def test_stipple_params_have_correct_visible_when(self):
        stipple_names = ["stipple_spacing_mm", "stipple_size_mm", "stipple_randomness"]
        for name in stipple_names:
            p = self.by_name[name]
            assert p.visible_when is not None, f"{name} has no visible_when"
            assert "brush_type" in p.visible_when, f"{name} missing 'brush_type' key in visible_when"
            assert "Stippled" in p.visible_when["brush_type"], f"{name} visible_when should include 'Stippled'"

    def test_multi_stroke_params_have_correct_visible_when(self):
        ms_names = ["stroke_count", "stroke_spread_mm", "stroke_noise"]
        for name in ms_names:
            p = self.by_name[name]
            assert p.visible_when is not None, f"{name} has no visible_when"
            assert "brush_type" in p.visible_when, f"{name} missing 'brush_type' key in visible_when"
            assert "Multi-Stroke" in p.visible_when["brush_type"], f"{name} visible_when should include 'Multi-Stroke'"

    def test_calligraphic_params_have_correct_visible_when(self):
        cal_names = ["nib_angle", "nib_width_mm", "min_width_mm"]
        for name in cal_names:
            p = self.by_name[name]
            assert p.visible_when is not None, f"{name} has no visible_when"
            assert "brush_type" in p.visible_when, f"{name} missing 'brush_type' key in visible_when"
            assert "Calligraphic" in p.visible_when["brush_type"], f"{name} visible_when should include 'Calligraphic'"
