"""Tests for the ConcentricRingsGenerator (Phase 36.1)."""

from __future__ import annotations

import math

import pytest

from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


class TestConcentricRingsGenerator:
    def setup_method(self):
        from plottter.generators.concentric_rings import ConcentricRingsGenerator
        self.gen = ConcentricRingsGenerator()
        self.canvas = make_canvas()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Concentric Rings" in GENERATORS

    def test_category(self):
        assert self.gen.category == "math"

    # ------------------------------------------------------------------
    # Ring count
    # ------------------------------------------------------------------

    def test_circle_ring_count(self):
        paths = self.gen.generate(
            {"ring_count": 5, "ring_spacing_mm": 3.0, "ring_shape": "Circle"},
            self.canvas,
        )
        assert len(paths) == 5

    def test_ring_count_ten(self):
        paths = self.gen.generate(
            {"ring_count": 10, "ring_spacing_mm": 2.0, "ring_shape": "Circle"},
            self.canvas,
        )
        assert len(paths) == 10

    # ------------------------------------------------------------------
    # Ring spacing
    # ------------------------------------------------------------------

    def test_ring_spacing(self):
        """Ring radii should be exact multiples of ring_spacing_mm."""
        spacing = 3.0
        ring_count = 6
        paths = self.gen.generate(
            {"ring_count": ring_count, "ring_spacing_mm": spacing,
             "ring_shape": "Circle", "center_x_mm": 0.0, "center_y_mm": 0.0,
             "x_offset_mm": 0.0, "y_offset_mm": 0.0, "points_per_ring": 64,
             "noise_amplitude_mm": 0.0},
            self.canvas,
        )
        assert len(paths) == ring_count

        # Determine canvas centre
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        for i, path in enumerate(paths, start=1):
            # Compute radius of the first point
            x0, y0 = path[0]
            r = math.hypot(x0 - cx, y0 - cy)
            expected = i * spacing
            assert abs(r - expected) < 1e-6, (
                f"Ring {i}: expected radius {expected:.4f}, got {r:.4f}"
            )

    # ------------------------------------------------------------------
    # Polygon shapes
    # ------------------------------------------------------------------

    def test_polygon_shapes_vertex_counts(self):
        """Each polygon ring should have sides+1 points (closed)."""
        shape_sides = {
            "Square": 4,
            "Triangle": 3,
            "Pentagon": 5,
            "Hexagon": 6,
            "Octagon": 8,
        }
        for shape, sides in shape_sides.items():
            paths = self.gen.generate(
                {"ring_count": 3, "ring_spacing_mm": 2.0, "ring_shape": shape},
                self.canvas,
            )
            assert len(paths) == 3, f"{shape}: expected 3 rings"
            for path in paths:
                assert len(path) == sides + 1, (
                    f"{shape}: expected {sides + 1} vertices, got {len(path)}"
                )

    def test_polygon_shapes_are_closed(self):
        """Each polygon ring polyline should start and end at the same point."""
        for shape in ("Square", "Triangle", "Pentagon", "Hexagon", "Octagon"):
            paths = self.gen.generate(
                {"ring_count": 2, "ring_spacing_mm": 3.0, "ring_shape": shape},
                self.canvas,
            )
            for path in paths:
                assert path[0] == path[-1], f"{shape} ring is not closed"

    def test_circle_rings_are_closed(self):
        """Circle ring polylines should start and end at the same point."""
        paths = self.gen.generate(
            {"ring_count": 3, "ring_spacing_mm": 2.0, "ring_shape": "Circle",
             "points_per_ring": 32},
            self.canvas,
        )
        for path in paths:
            x0, y0 = path[0]
            xn, yn = path[-1]
            assert abs(x0 - xn) < 1e-9 and abs(y0 - yn) < 1e-9, \
                "Circle ring polyline should be closed"

    # ------------------------------------------------------------------
    # Centre offset
    # ------------------------------------------------------------------

    def test_center_offset_shifts_ring_centers(self):
        """center_x_mm / center_y_mm should shift all ring centres by that amount."""
        offset_x, offset_y = 10.0, 5.0
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        canvas_cx = (draw_x1 + draw_x2) / 2.0
        canvas_cy = (draw_y1 + draw_y2) / 2.0

        paths_no_offset = self.gen.generate(
            {"ring_count": 3, "ring_spacing_mm": 3.0, "ring_shape": "Circle",
             "center_x_mm": 0.0, "center_y_mm": 0.0,
             "x_offset_mm": 0.0, "y_offset_mm": 0.0, "points_per_ring": 64},
            self.canvas,
        )
        paths_offset = self.gen.generate(
            {"ring_count": 3, "ring_spacing_mm": 3.0, "ring_shape": "Circle",
             "center_x_mm": offset_x, "center_y_mm": offset_y,
             "x_offset_mm": 0.0, "y_offset_mm": 0.0, "points_per_ring": 64},
            self.canvas,
        )

        # For each ring, the centroid of points should shift by (offset_x, offset_y)
        for path_no, path_off in zip(paths_no_offset, paths_offset):
            cx_no = sum(x for x, _ in path_no) / len(path_no)
            cy_no = sum(y for _, y in path_no) / len(path_no)
            cx_off = sum(x for x, _ in path_off) / len(path_off)
            cy_off = sum(y for _, y in path_off) / len(path_off)
            assert abs((cx_off - cx_no) - offset_x) < 1e-6, \
                f"Center X shift: expected {offset_x}, got {cx_off - cx_no:.6f}"
            assert abs((cy_off - cy_no) - offset_y) < 1e-6, \
                f"Center Y shift: expected {offset_y}, got {cy_off - cy_no:.6f}"

    # ------------------------------------------------------------------
    # x_offset_mm / y_offset_mm
    # ------------------------------------------------------------------

    def test_x_y_offset_shifts_all_points(self):
        """x_offset_mm / y_offset_mm should translate every point uniformly."""
        x_off, y_off = 15.0, -8.0
        params_base = {
            "ring_count": 3, "ring_spacing_mm": 2.0, "ring_shape": "Circle",
            "center_x_mm": 0.0, "center_y_mm": 0.0, "points_per_ring": 32,
            "x_offset_mm": 0.0, "y_offset_mm": 0.0,
        }
        paths_no_offset = self.gen.generate(params_base, self.canvas)
        paths_offset = self.gen.generate(
            {**params_base, "x_offset_mm": x_off, "y_offset_mm": y_off},
            self.canvas,
        )

        for path_no, path_off in zip(paths_no_offset, paths_offset):
            for (xn, yn), (xo, yo) in zip(path_no, path_off):
                assert abs((xo - xn) - x_off) < 1e-9, \
                    f"x_offset: expected +{x_off}, got {xo - xn:.9f}"
                assert abs((yo - yn) - y_off) < 1e-9, \
                    f"y_offset: expected +{y_off}, got {yo - yn:.9f}"

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def test_presets_generate_paths(self):
        """Every preset should generate at least one path."""
        presets = self.gen.get_presets()
        assert len(presets) >= 1, "Expected at least one preset"
        for preset in presets:
            # Merge preset params with required defaults
            params = {
                "ring_count": 10,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                **preset.params,
            }
            paths = self.gen.generate(params, self.canvas)
            assert len(paths) > 0, f"Preset '{preset.name}' produced no paths"

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def test_cancellation_stops_early(self):
        """cancelled_callback returning True should stop generation before all rings."""
        call_count = [0]

        def cancelled():
            call_count[0] += 1
            # Cancel after first ring check
            return call_count[0] > 1

        paths = self.gen.generate(
            {"ring_count": 50, "ring_spacing_mm": 2.0, "ring_shape": "Circle"},
            self.canvas,
            cancelled_callback=cancelled,
        )
        # Should have stopped well before 50 rings
        assert len(paths) < 50, \
            f"Cancellation should stop early; got {len(paths)} rings"

    def test_no_cancellation_produces_all_rings(self):
        """With no cancellation callback, all rings should be generated."""
        ring_count = 20
        paths = self.gen.generate(
            {"ring_count": ring_count, "ring_spacing_mm": 2.0, "ring_shape": "Hexagon"},
            self.canvas,
        )
        assert len(paths) == ring_count


class TestConcentricRingsNoiseFeatures:
    """Tests for 36.2 noise distortion and radial variation features."""

    def setup_method(self):
        from plottter.generators.concentric_rings import ConcentricRingsGenerator, _NOISE_AVAILABLE
        self.gen = ConcentricRingsGenerator()
        self.canvas = make_canvas()
        self.noise_available = _NOISE_AVAILABLE

    def _canvas_centre(self):
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        return (draw_x1 + draw_x2) / 2.0, (draw_y1 + draw_y2) / 2.0

    # ------------------------------------------------------------------
    # (a) noise_amplitude_mm > 0 produces wavy rings
    # ------------------------------------------------------------------

    def test_noise_amplitude_nonzero_produces_wavy_rings(self):
        """noise_amplitude_mm > 0 produces rings whose points deviate from base radius."""
        if not self.noise_available:
            pytest.skip("noise library not installed")

        spacing = 3.0
        paths = self.gen.generate(
            {
                "ring_count": 5,
                "ring_spacing_mm": spacing,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 5.0,
                "amplitude_growth": "Constant",  # uniform amplitude on all rings
                "noise_seed": 42,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
                "points_per_ring": 64,
            },
            self.canvas,
        )
        cx, cy = self._canvas_centre()
        any_wavy = False
        for i, path in enumerate(paths, start=1):
            expected_r = i * spacing
            max_deviation = max(
                abs(math.hypot(x - cx, y - cy) - expected_r)
                for x, y in path[:-1]  # exclude repeated closing point
            )
            if max_deviation > 0.01:
                any_wavy = True
                break
        assert any_wavy, "Expected at least one wavy ring when noise_amplitude_mm > 0"

    # ------------------------------------------------------------------
    # (b) noise_amplitude_mm = 0 produces perfect geometric rings
    # ------------------------------------------------------------------

    def test_noise_amplitude_zero_produces_perfect_rings(self):
        """noise_amplitude_mm=0 yields all ring points at exactly i * ring_spacing from centre."""
        spacing = 3.0
        ring_count = 6
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": spacing,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 0.0,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
                "points_per_ring": 64,
            },
            self.canvas,
        )
        cx, cy = self._canvas_centre()
        for i, path in enumerate(paths, start=1):
            expected_r = i * spacing
            for x, y in path:
                r = math.hypot(x - cx, y - cy)
                assert abs(r - expected_r) < 1e-6, (
                    f"Ring {i}: expected radius {expected_r:.4f}, got {r:.6f}"
                )

    # ------------------------------------------------------------------
    # (c) amplitude_growth="Linear" makes outer rings more distorted
    # ------------------------------------------------------------------

    def test_amplitude_growth_linear_increases_with_ring_index(self):
        """Linear amplitude_growth: outermost ring has greater radial spread than innermost."""
        if not self.noise_available:
            pytest.skip("noise library not installed")

        spacing = 3.0
        ring_count = 20
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": spacing,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 5.0,
                "amplitude_growth": "Linear",
                "noise_seed": 42,
                "noise_evolution": 0.0,  # same noise pattern on all rings; only amplitude differs
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
                "points_per_ring": 64,
            },
            self.canvas,
        )
        cx, cy = self._canvas_centre()

        def _ring_radii_stddev(ring_idx: int) -> float:
            path = paths[ring_idx]
            expected_r = (ring_idx + 1) * spacing
            deviations = [math.hypot(x - cx, y - cy) - expected_r for x, y in path[:-1]]
            mean = sum(deviations) / len(deviations)
            variance = sum((d - mean) ** 2 for d in deviations) / len(deviations)
            return variance ** 0.5

        stddev_inner = _ring_radii_stddev(0)   # ring 1 — smallest amplitude
        stddev_outer = _ring_radii_stddev(-1)  # ring N — largest amplitude
        assert stddev_outer > stddev_inner, (
            f"Linear growth: outer ring std-dev ({stddev_outer:.4f}) should exceed "
            f"inner ring std-dev ({stddev_inner:.4f})"
        )

    # ------------------------------------------------------------------
    # (d) thickness_noise > 0 produces non-uniform ring spacing
    # ------------------------------------------------------------------

    def test_thickness_noise_produces_nonuniform_spacing(self):
        """thickness_noise > 0 causes at least some rings to sit away from i * ring_spacing."""
        if not self.noise_available:
            pytest.skip("noise library not installed")

        spacing = 3.0
        ring_count = 20
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": spacing,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 0.0,  # no radial noise so deviations come only from thickness
                "thickness_noise": 0.5,
                "noise_seed": 42,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
                "points_per_ring": 64,
            },
            self.canvas,
        )
        cx, cy = self._canvas_centre()
        any_deviated = False
        for i, path in enumerate(paths, start=1):
            expected_r = i * spacing
            x0, y0 = path[0]
            actual_r = math.hypot(x0 - cx, y0 - cy)
            if abs(actual_r - expected_r) > 0.01:
                any_deviated = True
                break
        assert any_deviated, (
            "Expected at least one ring to deviate from i * ring_spacing when thickness_noise > 0"
        )


class TestConcentricRingsFillPatterns:
    """Tests for 36.3 fill patterns and presets."""

    def setup_method(self):
        from plottter.generators.concentric_rings import ConcentricRingsGenerator, _NOISE_AVAILABLE
        self.gen = ConcentricRingsGenerator()
        self.canvas = make_canvas()
        self.noise_available = _NOISE_AVAILABLE

    # ------------------------------------------------------------------
    # (a) ring_gap_chance creates broken rings
    # ------------------------------------------------------------------

    def test_ring_gap_chance_zero_produces_complete_rings(self):
        """ring_gap_chance=0 should produce exactly ring_count full rings."""
        ring_count = 10
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "ring_gap_chance": 0.0,
                "noise_amplitude_mm": 0.0,
            },
            self.canvas,
        )
        assert len(paths) == ring_count

    def test_ring_gap_chance_nonzero_produces_more_paths(self):
        """ring_gap_chance > 0 should produce multiple arc segments (more paths than rings)."""
        if not self.noise_available:
            pytest.skip("noise library not installed")

        ring_count = 20
        # With high gap chance, rings get split into arcs → more polylines than ring_count
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "ring_gap_chance": 0.6,
                "noise_amplitude_mm": 0.0,
                "noise_seed": 42,
                "points_per_ring": 128,
            },
            self.canvas,
        )
        # Gaps create multiple arcs per ring OR some rings may be fully removed
        # Either way, no path should be the original closed ring length (129 pts)
        # At minimum, some rings must have been split or fully gaped out
        assert len(paths) < ring_count or all(len(p) < 129 for p in paths), (
            "ring_gap_chance=0.6 should produce arcs shorter than a full closed ring"
        )

    def test_ring_gap_chance_no_single_point_arcs(self):
        """No arc produced by gap splitting should have fewer than 2 points."""
        if not self.noise_available:
            pytest.skip("noise library not installed")

        paths = self.gen.generate(
            {
                "ring_count": 15,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "ring_gap_chance": 0.4,
                "noise_amplitude_mm": 0.0,
                "noise_seed": 7,
                "points_per_ring": 64,
            },
            self.canvas,
        )
        for path in paths:
            assert len(path) >= 2, "No arc should have fewer than 2 points"

    # ------------------------------------------------------------------
    # (b) radial_lines creates spokes from centre
    # ------------------------------------------------------------------

    def test_radial_lines_false_produces_only_rings(self):
        """radial_lines=False should produce exactly ring_count paths."""
        ring_count = 8
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "radial_lines": False,
                "noise_amplitude_mm": 0.0,
            },
            self.canvas,
        )
        assert len(paths) == ring_count

    def test_radial_lines_true_adds_spokes(self):
        """radial_lines=True should produce ring_count + radial_line_count paths."""
        ring_count = 8
        radial_count = 6
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 0.0,
                "radial_lines": True,
                "radial_line_count": radial_count,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
            },
            self.canvas,
        )
        assert len(paths) == ring_count + radial_count

    def test_radial_lines_start_at_centre(self):
        """Each radial line should start at the canvas centre."""
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        ring_count = 5
        radial_count = 4
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 0.0,
                "radial_lines": True,
                "radial_line_count": radial_count,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
            },
            self.canvas,
        )
        # Last radial_count paths are the radial lines
        radial_paths = paths[ring_count:]
        for radial in radial_paths:
            x0, y0 = radial[0]
            assert abs(x0 - cx) < 1e-6, f"Radial start x: expected {cx:.4f}, got {x0:.6f}"
            assert abs(y0 - cy) < 1e-6, f"Radial start y: expected {cy:.4f}, got {y0:.6f}"

    def test_radial_lines_have_ring_count_plus_one_points(self):
        """Each radial line should have ring_count + 1 points (centre + one per ring)."""
        ring_count = 5
        radial_count = 3
        paths = self.gen.generate(
            {
                "ring_count": ring_count,
                "ring_spacing_mm": 2.0,
                "ring_shape": "Circle",
                "noise_amplitude_mm": 0.0,
                "radial_lines": True,
                "radial_line_count": radial_count,
                "x_offset_mm": 0.0,
                "y_offset_mm": 0.0,
            },
            self.canvas,
        )
        radial_paths = paths[ring_count:]
        for radial in radial_paths:
            assert len(radial) == ring_count + 1, (
                f"Radial line should have {ring_count + 1} points, got {len(radial)}"
            )

    # ------------------------------------------------------------------
    # (c) Each preset produces at least one path (including Spider Web)
    # ------------------------------------------------------------------

    def test_spider_web_preset_exists_and_generates(self):
        """The 'Spider Web' preset should exist and generate paths."""
        presets = self.gen.get_presets()
        spider_web = next((p for p in presets if p.name == "Spider Web"), None)
        assert spider_web is not None, "Spider Web preset must exist"

        params = {
            "ring_count": 10,
            "ring_spacing_mm": 2.0,
            "ring_shape": "Circle",
            **spider_web.params,
        }
        paths = self.gen.generate(params, self.canvas)
        assert len(paths) > 0, "Spider Web preset must produce at least one path"

    def test_organic_rings_preset_has_gap_chance(self):
        """The 'Organic Rings' preset must include ring_gap_chance > 0."""
        presets = self.gen.get_presets()
        organic = next((p for p in presets if p.name == "Organic Rings"), None)
        assert organic is not None, "Organic Rings preset must exist"
        assert organic.params.get("ring_gap_chance", 0.0) > 0.0, (
            "Organic Rings preset should have ring_gap_chance > 0"
        )
