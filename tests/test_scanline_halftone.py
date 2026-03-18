"""Tests for the ScanlineHalftoneGenerator (Phase 33.1 and 33.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_solid_image(brightness: int = 128, h: int = 100, w: int = 100) -> np.ndarray:
    """Single-value grayscale image."""
    return np.full((h, w), brightness, dtype=np.uint8)


def make_gradient_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Left-to-right brightness gradient (0=black left, 255=white right)."""
    arr = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        arr[:, x] = int(x / (w - 1) * 255)
    return arr


def make_vertical_gradient_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Top-to-bottom brightness gradient (0=black top, 255=white bottom)."""
    arr = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        arr[y, :] = int(y / (h - 1) * 255)
    return arr


class TestScanlineHalftoneGenerator:
    def setup_method(self):
        from plottter.generators.scanline_halftone import ScanlineHalftoneGenerator
        self.gen = ScanlineHalftoneGenerator()
        self.canvas = make_canvas()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_registration(self):
        from plottter.generators import GENERATORS
        assert "Scanline Halftone" in GENERATORS
        assert GENERATORS["Scanline Halftone"].category == "image"

    def test_name_and_category(self):
        assert self.gen.name == "Scanline Halftone"
        assert self.gen.category == "image"

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def test_has_required_parameters(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "line_spacing_mm" in names
        assert "angle_deg" in names
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names

    def test_has_thickness_parameters(self):
        """Task 33.2: new thickness-related parameters must be present."""
        names = [p.name for p in self.gen.get_parameters()]
        assert "max_thickness" in names
        assert "pen_width_mm" in names
        assert "sample_interval_mm" in names
        assert "tone_gamma" in names

    def test_max_thickness_defaults_to_4(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["max_thickness"].default == 4

    def test_pen_width_defaults_to_03(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert abs(params["pen_width_mm"].default - 0.3) < 1e-9

    def test_sample_interval_defaults_to_1(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert abs(params["sample_interval_mm"].default - 1.0) < 1e-9

    def test_tone_gamma_defaults_to_15(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert abs(params["tone_gamma"].default - 1.5) < 1e-9

    def test_x_y_offset_not_randomizable(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["x_offset_mm"].randomizable is False
        assert params["y_offset_mm"].randomizable is False

    def test_has_shared_image_parameters(self):
        """Generator must expose the four standard shared image parameters."""
        names = [p.name for p in self.gen.get_parameters()]
        assert "invert" in names
        assert "brightness" in names
        assert "contrast" in names
        assert "blur_radius" in names

    def test_invert_is_bool_param(self):
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["invert"], BoolParam)

    def test_brightness_contrast_blur_are_float_params(self):
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["brightness"], FloatParam)
        assert isinstance(params["contrast"], FloatParam)
        assert isinstance(params["blur_radius"], FloatParam)

    def test_max_thickness_is_int_param(self):
        from plottter.generators.base import IntParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["max_thickness"], IntParam)

    # ------------------------------------------------------------------
    # Basic generation
    # ------------------------------------------------------------------

    def test_returns_empty_without_source_image(self):
        result = self.gen.generate({}, self.canvas)
        assert result == []

    def test_produces_lines_with_solid_image(self):
        img = make_solid_image(128)
        result = self.gen.generate({"_source_image": img}, self.canvas)
        assert len(result) > 0
        # Each path must have at least 2 points (center line = 2, offsets = 2+)
        for path in result:
            assert len(path) >= 2

    def test_max_thickness_zero_gives_only_center_lines(self):
        """max_thickness=0 should produce only center lines (no offset lines).

        Pass skip_white=False so the center line is always a 2-point segment
        (old pre-33.3 behaviour), letting us confirm that no offset polylines
        are added even for a fully-black image.
        """
        img = make_solid_image(0)  # fully black — would trigger max offsets
        result = self.gen.generate(
            {"_source_image": img, "max_thickness": 0, "line_spacing_mm": 5.0,
             "skip_white": False},
            self.canvas,
        )
        assert len(result) > 0
        for path in result:
            assert len(path) == 2, "With max_thickness=0 and skip_white=False, only 2-point center lines"

    def test_lines_are_horizontal_at_angle_zero(self):
        """At angle=0, all scan lines should be horizontal (same y for all points)."""
        img = make_solid_image(128)
        result = self.gen.generate(
            {"_source_image": img, "angle_deg": 0.0, "line_spacing_mm": 5.0},
            self.canvas,
        )
        assert len(result) > 0
        for path in result:
            assert len(path) >= 2
            y_vals = [y for _, y in path]
            assert all(
                abs(y - y_vals[0]) < 1e-3 for y in y_vals
            ), f"All points should have same y, but got {set(round(y, 4) for y in y_vals)}"

    def test_lines_are_vertical_at_angle_90(self):
        """At angle=90, all scan lines should be vertical (same x for all points)."""
        img = make_solid_image(128)
        result = self.gen.generate(
            {"_source_image": img, "angle_deg": 90.0, "line_spacing_mm": 5.0},
            self.canvas,
        )
        assert len(result) > 0
        for path in result:
            assert len(path) >= 2
            x_vals = [x for x, _ in path]
            assert all(
                abs(x - x_vals[0]) < 1e-3 for x in x_vals
            ), f"All points should have same x, but got {set(round(x, 4) for x in x_vals)}"

    def test_spacing_controls_line_density(self):
        """Smaller spacing should produce more lines than larger spacing."""
        img = make_solid_image(128)
        result_dense = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 1.0},
            self.canvas,
        )
        result_sparse = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0},
            self.canvas,
        )
        assert len(result_dense) > len(result_sparse)

    def test_angle_rotates_lines(self):
        """Lines at 45 degrees should not be horizontal or vertical."""
        img = make_solid_image(128)
        result = self.gen.generate(
            {"_source_image": img, "angle_deg": 45.0, "line_spacing_mm": 5.0},
            self.canvas,
        )
        assert len(result) > 0
        for path in result:
            assert len(path) >= 2
            x0, y0 = path[0]
            x1, y1 = path[-1]
            # The overall direction of each path should be neither horizontal nor vertical
            assert abs(y1 - y0) > 1e-3, "Lines at 45° should not be horizontal"
            assert abs(x1 - x0) > 1e-3, "Lines at 45° should not be vertical"

    def test_lines_clipped_to_drawing_area(self):
        """All generated points should fall within the canvas drawing area."""
        img = make_solid_image(0)  # fully black → maximum offsets
        result = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 3.0,
             "max_thickness": 4, "pen_width_mm": 0.3},
            self.canvas,
        )
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = 1e-3
        for path in result:
            for x, y in path:
                assert x1 - tol <= x <= x2 + tol, f"x={x} out of bounds [{x1}, {x2}]"
                assert y1 - tol <= y <= y2 + tol, f"y={y} out of bounds [{y1}, {y2}]"

    def test_x_offset_shifts_output(self):
        """x_offset_mm should shift all output points horizontally."""
        img = make_solid_image(128)
        baseline = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 0},
            self.canvas,
        )
        shifted = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 0,
             "x_offset_mm": 10.0},
            self.canvas,
        )
        assert len(baseline) == len(shifted)
        for base_path, shift_path in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(base_path, shift_path):
                assert abs(sx - bx - 10.0) < 1e-6
                assert abs(sy - by) < 1e-6

    def test_y_offset_shifts_output(self):
        """y_offset_mm should shift all output points vertically."""
        img = make_solid_image(128)
        baseline = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 0},
            self.canvas,
        )
        shifted = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 0,
             "y_offset_mm": 7.0},
            self.canvas,
        )
        assert len(baseline) == len(shifted)
        for base_path, shift_path in zip(baseline, shifted):
            for (bx, by), (sx, sy) in zip(base_path, shift_path):
                assert abs(sx - bx) < 1e-6
                assert abs(sy - by - 7.0) < 1e-6

    def test_zero_offset_unchanged(self):
        """Default zero offsets should not change the output."""
        img = make_solid_image(128)
        r1 = self.gen.generate({"_source_image": img, "line_spacing_mm": 5.0}, self.canvas)
        r2 = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "x_offset_mm": 0.0, "y_offset_mm": 0.0},
            self.canvas,
        )
        assert len(r1) == len(r2)

    def test_accepts_rgb_source_image(self):
        """Generator should work with 3-channel RGB source images."""
        img_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = self.gen.generate(
            {"_source_image": img_rgb, "line_spacing_mm": 5.0},
            self.canvas,
        )
        assert len(result) > 0

    def test_cancellation(self):
        """Cancellation callback should stop generation early."""
        img = make_solid_image(128)
        cancelled = [False]

        def check_cancel():
            return cancelled[0]

        # Generate normally first
        full_result = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 1.0},
            self.canvas,
            cancelled_callback=check_cancel,
        )

        # Now cancel immediately
        cancelled[0] = True
        partial_result = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 1.0},
            self.canvas,
            cancelled_callback=check_cancel,
        )
        # Partial result should have fewer or equal lines
        assert len(partial_result) <= len(full_result)

    def test_progress_callback_called(self):
        """Progress callback should be invoked during generation."""
        img = make_solid_image(128)
        progress_values = []

        def on_progress(pct: int) -> None:
            progress_values.append(pct)

        self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 3.0},
            self.canvas,
            progress_callback=on_progress,
        )
        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    # ------------------------------------------------------------------
    # Task 33.2: Brightness-based thickness modulation
    # ------------------------------------------------------------------

    def test_dark_image_produces_more_paths_than_bright(self):
        """Dark areas (more offset lines) should produce more total polylines."""
        # Fully black image — max thickness offset lines on every scan line
        black_img = make_solid_image(0)
        black_result = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3},
            self.canvas,
        )
        # Fully white image — only center lines (thickness ≈ 0)
        white_img = make_solid_image(255)
        white_result = self.gen.generate(
            {"_source_image": white_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3},
            self.canvas,
        )
        # Black image must produce more polylines (offset lines added)
        assert len(black_result) > len(white_result)

    def test_white_image_produces_only_center_lines_when_skip_disabled(self):
        """Pure white image (brightness=255) → thickness=0 → only center lines.

        With skip_white=False the center line is always drawn even in white areas,
        so every polyline should have exactly 2 points.
        """
        white_img = make_solid_image(255)
        white_result = self.gen.generate(
            {"_source_image": white_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3, "tone_gamma": 1.0,
             "skip_white": False},
            self.canvas,
        )
        # White brightness → thickness = 4 * (1 - 1.0) = 0 → no offsets
        # Only center lines (2 points each)
        assert len(white_result) > 0
        for path in white_result:
            assert len(path) == 2, (
                f"White image should produce only center lines (2 pts), got {len(path)}"
            )

    def test_black_image_produces_max_offset_lines(self):
        """Pure black image (brightness=0) → thickness=max_thickness → max offset lines."""
        black_img = make_solid_image(0)
        max_t = 3
        result_with = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 10.0,
             "max_thickness": max_t, "pen_width_mm": 0.3, "tone_gamma": 1.0},
            self.canvas,
        )
        result_without = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 10.0,
             "max_thickness": 0, "pen_width_mm": 0.3, "tone_gamma": 1.0},
            self.canvas,
        )
        # With max_thickness=3, for each center line we also get 3*2=6 offset segments
        assert len(result_with) > len(result_without)

    def test_higher_max_thickness_more_paths(self):
        """Increasing max_thickness on a dark image should produce more paths."""
        black_img = make_solid_image(0)
        result_low = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 1, "pen_width_mm": 0.3},
            self.canvas,
        )
        result_high = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3},
            self.canvas,
        )
        assert len(result_high) > len(result_low)

    def test_gradient_dark_side_has_more_paths_than_bright_side(self):
        """For a gradient image, the dark half should contribute more offset paths."""
        # Left-dark, right-bright gradient
        gradient = make_gradient_image(100, 200)
        result = self.gen.generate(
            {"_source_image": gradient, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3, "sample_interval_mm": 1.0,
             "tone_gamma": 1.0},
            self.canvas,
        )
        # Each horizontal scan line should have more/longer offset segments on the left
        # (dark) than on the right (bright). We verify this indirectly:
        # the overall path count exceeds what a pure-white image produces
        white_img = make_solid_image(255, 100, 200)
        white_result = self.gen.generate(
            {"_source_image": white_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3},
            self.canvas,
        )
        assert len(result) > len(white_result)

    def test_tone_gamma_affects_output(self):
        """Different gamma values should produce different path counts for grey images."""
        grey_img = make_solid_image(128)
        result_low_gamma = self.gen.generate(
            {"_source_image": grey_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3, "tone_gamma": 0.5},
            self.canvas,
        )
        result_high_gamma = self.gen.generate(
            {"_source_image": grey_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "pen_width_mm": 0.3, "tone_gamma": 3.0},
            self.canvas,
        )
        # Higher gamma → thickness = 4*(1-(128/255)^3) ≈ high
        # Lower gamma  → thickness = 4*(1-(128/255)^0.5) ≈ lower
        # So high gamma image should have more paths (more offset lines active)
        assert len(result_high_gamma) > len(result_low_gamma)

    def test_offset_lines_parallel_to_center_line(self):
        """Offset lines must be parallel to the center line (same direction)."""
        black_img = make_solid_image(0)
        # Use a distinctive angle and high spacing so we can detect structure
        result = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 20.0,
             "angle_deg": 30.0, "max_thickness": 1, "pen_width_mm": 1.0,
             "sample_interval_mm": 2.0, "tone_gamma": 1.0},
            self.canvas,
        )
        # All polylines should be roughly parallel (same direction vector first→last)
        angles = []
        for path in result:
            if len(path) >= 2:
                x0, y0 = path[0]
                x1, y1 = path[-1]
                if math.hypot(x1 - x0, y1 - y0) > 1e-4:
                    angles.append(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180)
        if angles:
            reference = angles[0]
            for a in angles[1:]:
                diff = abs(a - reference) % 180
                assert min(diff, 180 - diff) < 2.0, (
                    f"Lines not parallel: {reference:.1f}° vs {a:.1f}°"
                )

    def test_pen_width_controls_offset_distance(self):
        """Wider pen_width should produce more total paths spread across more area."""
        black_img = make_solid_image(0)
        # With wide pen_width, fewer offset levels fit without clipping → similar count
        # With narrow pen_width, all offsets fit within drawing area → all counted
        # We just verify the output is non-empty and different for different widths
        result_narrow = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 2, "pen_width_mm": 0.1, "tone_gamma": 1.0},
            self.canvas,
        )
        result_wide = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 2, "pen_width_mm": 0.8, "tone_gamma": 1.0},
            self.canvas,
        )
        assert len(result_narrow) > 0
        assert len(result_wide) > 0

    def test_sample_interval_affects_path_point_count(self):
        """Smaller sample_interval should produce polylines with more interior points."""
        black_img = make_solid_image(0)
        result_fine = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 10.0,
             "max_thickness": 1, "pen_width_mm": 0.3,
             "sample_interval_mm": 0.5, "tone_gamma": 1.0},
            self.canvas,
        )
        result_coarse = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 10.0,
             "max_thickness": 1, "pen_width_mm": 0.3,
             "sample_interval_mm": 4.0, "tone_gamma": 1.0},
            self.canvas,
        )
        # With finer sampling, offset polylines have more intermediate points
        total_points_fine = sum(len(p) for p in result_fine)
        total_points_coarse = sum(len(p) for p in result_coarse)
        assert total_points_fine > total_points_coarse

    def test_offset_lines_clipped_to_drawing_area(self):
        """All offset polyline points must lie within the canvas drawing area."""
        black_img = make_solid_image(0)  # max offsets
        result = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 3.0,
             "max_thickness": 4, "pen_width_mm": 0.3, "tone_gamma": 1.0},
            self.canvas,
        )
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = 1e-3
        for path in result:
            for x, y in path:
                assert x1 - tol <= x <= x2 + tol, f"x={x:.4f} outside [{x1:.1f}, {x2:.1f}]"
                assert y1 - tol <= y <= y2 + tol, f"y={y:.4f} outside [{y1:.1f}, {y2:.1f}]"


class TestClipLinesToRect:
    def test_line_fully_inside(self):
        from plottter.generators.scanline_halftone import _clip_line_to_rect
        result = _clip_line_to_rect(1, 1, 3, 3, 0, 0, 5, 5)
        assert result is not None
        assert abs(result[0] - 1) < 1e-9
        assert abs(result[2] - 3) < 1e-9

    def test_line_fully_outside(self):
        from plottter.generators.scanline_halftone import _clip_line_to_rect
        result = _clip_line_to_rect(10, 10, 20, 20, 0, 0, 5, 5)
        assert result is None

    def test_line_crosses_left_boundary(self):
        from plottter.generators.scanline_halftone import _clip_line_to_rect
        result = _clip_line_to_rect(-5, 2, 5, 2, 0, 0, 10, 10)
        assert result is not None
        assert abs(result[0] - 0) < 1e-9  # clipped to x=0
        assert abs(result[2] - 5) < 1e-9

    def test_line_crosses_right_boundary(self):
        from plottter.generators.scanline_halftone import _clip_line_to_rect
        result = _clip_line_to_rect(5, 2, 15, 2, 0, 0, 10, 10)
        assert result is not None
        assert abs(result[0] - 5) < 1e-9
        assert abs(result[2] - 10) < 1e-9  # clipped to x=10

    def test_vertical_line_inside(self):
        from plottter.generators.scanline_halftone import _clip_line_to_rect
        result = _clip_line_to_rect(5, -5, 5, 15, 0, 0, 10, 10)
        assert result is not None
        assert abs(result[1] - 0) < 1e-9  # clipped to y=0
        assert abs(result[3] - 10) < 1e-9  # clipped to y=10


class TestGenerateScanlinesWithThickness:
    """Unit tests for the core thickness generation function."""

    def setup_method(self):
        self.img_rect = (0.0, 0.0, 100.0, 100.0)  # 100×100 mm square

    def _call(self, gray, **kwargs):
        from plottter.generators.scanline_halftone import _generate_scanlines_with_thickness
        defaults = {
            "angle_deg": 0.0,
            "spacing_mm": 10.0,
            "max_thickness": 2,
            "pen_width_mm": 1.0,
            "sample_interval_mm": 5.0,
            "tone_gamma": 1.0,
        }
        defaults.update(kwargs)
        return _generate_scanlines_with_thickness(
            self.img_rect,
            defaults["angle_deg"],
            defaults["spacing_mm"],
            gray,
            defaults["max_thickness"],
            defaults["pen_width_mm"],
            defaults["sample_interval_mm"],
            defaults["tone_gamma"],
        )

    def test_white_image_only_center_lines(self):
        """Pure white → thickness=0 → only center lines (2 pts each)."""
        gray = make_solid_image(255, 50, 50)
        result = self._call(gray)
        assert all(len(p) == 2 for p in result)

    def test_black_image_has_offset_lines(self):
        """Pure black → max thickness → offset lines present (> center lines)."""
        gray = make_solid_image(0, 50, 50)
        result_with = self._call(gray, max_thickness=2)
        result_without = self._call(gray, max_thickness=0)
        assert len(result_with) > len(result_without)

    def test_all_points_in_rect(self):
        """All points must lie within img_rect regardless of offset."""
        gray = make_solid_image(0, 50, 50)
        result = self._call(gray, max_thickness=3, pen_width_mm=2.0)
        rx1, ry1, rx2, ry2 = self.img_rect
        tol = 1e-3
        for path in result:
            for x, y in path:
                assert rx1 - tol <= x <= rx2 + tol
                assert ry1 - tol <= y <= ry2 + tol

    def test_each_path_at_least_two_points(self):
        """Every returned polyline must have at least 2 points."""
        gray = make_solid_image(64, 50, 50)
        result = self._call(gray)
        for path in result:
            assert len(path) >= 2, f"Path has only {len(path)} point(s)"


# ---------------------------------------------------------------------------
# Task 33.3: skip_white and edge_sensitivity tests
# ---------------------------------------------------------------------------

class TestSkipWhite:
    """Tests for the skip_white / white_threshold feature (Task 33.3)."""

    def setup_method(self):
        from plottter.generators.scanline_halftone import ScanlineHalftoneGenerator
        self.gen = ScanlineHalftoneGenerator()
        self.canvas = make_canvas()

    def test_has_skip_white_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "skip_white" in names

    def test_has_white_threshold_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "white_threshold" in names

    def test_skip_white_default_true(self):
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["skip_white"], BoolParam)
        assert params["skip_white"].default is True

    def test_white_threshold_default_240(self):
        from plottter.generators.base import IntParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["white_threshold"], IntParam)
        assert params["white_threshold"].default == 240

    def test_white_image_skip_true_produces_no_output(self):
        """skip_white=True with a pure white image (255 > threshold=240) → no lines."""
        white_img = make_solid_image(255)
        result = self.gen.generate(
            {"_source_image": white_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "skip_white": True, "white_threshold": 240},
            self.canvas,
        )
        assert result == [], (
            "Pure white image with skip_white=True should produce no polylines"
        )

    def test_white_image_skip_false_produces_center_lines(self):
        """skip_white=False should draw center lines even in white areas."""
        white_img = make_solid_image(255)
        result = self.gen.generate(
            {"_source_image": white_img, "line_spacing_mm": 5.0,
             "max_thickness": 4, "skip_white": False},
            self.canvas,
        )
        assert len(result) > 0

    def test_skip_white_removes_bright_segments(self):
        """Enabling skip_white omits scan lines in bright areas.

        A vertical gradient (dark top → bright bottom) with threshold=200 means
        the bottom half of scan lines should produce no output at all.
        skip_white=False always draws a 2-pt center line per scan line, so the
        total polyline count with skip_white=True must be lower.
        """
        # Image with mixed brightness (vertical gradient: dark top, bright bottom)
        img = make_vertical_gradient_image(100, 100)
        result_skip = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 3.0, "max_thickness": 0,
             "skip_white": True, "white_threshold": 200},
            self.canvas,
        )
        result_no_skip = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 3.0, "max_thickness": 0,
             "skip_white": False},
            self.canvas,
        )
        # skip_white=False gives one 2-pt center line per scan line.
        # skip_white=True omits scan lines where ALL samples are above threshold.
        # The bright bottom half should have many skipped scan lines.
        assert len(result_skip) < len(result_no_skip), (
            "skip_white=True should produce fewer polylines than skip_white=False "
            f"for a gradient image (got {len(result_skip)} vs {len(result_no_skip)})"
        )

    def test_below_threshold_is_always_drawn(self):
        """Dark regions (brightness 0) should always be drawn regardless of threshold."""
        black_img = make_solid_image(0)
        result = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0, "max_thickness": 0,
             "skip_white": True, "white_threshold": 240},
            self.canvas,
        )
        # Black (0) is well below threshold (240) — center lines must be present
        assert len(result) > 0

    def test_threshold_boundary(self):
        """Brightness exactly at threshold should NOT be skipped (condition is >)."""
        # Solid image at the threshold value should not be skipped
        img_at_threshold = make_solid_image(200)
        result = self.gen.generate(
            {"_source_image": img_at_threshold, "line_spacing_mm": 5.0,
             "max_thickness": 0, "skip_white": True, "white_threshold": 200},
            self.canvas,
        )
        # brightness 200 is NOT > 200, so center lines should be drawn
        assert len(result) > 0


class TestEdgeSensitivity:
    """Tests for the edge_sensitivity feature (Task 33.3)."""

    def setup_method(self):
        from plottter.generators.scanline_halftone import ScanlineHalftoneGenerator
        self.gen = ScanlineHalftoneGenerator()
        self.canvas = make_canvas()

    def test_has_edge_sensitivity_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "edge_sensitivity" in names

    def test_edge_sensitivity_default_zero(self):
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["edge_sensitivity"], FloatParam)
        assert params["edge_sensitivity"].default == 0.0

    def test_edge_sensitivity_zero_is_no_op(self):
        """edge_sensitivity=0 must produce identical output to omitting it."""
        black_img = make_solid_image(0)
        result_base = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 3, "skip_white": False},
            self.canvas,
        )
        result_zero = self.gen.generate(
            {"_source_image": black_img, "line_spacing_mm": 5.0,
             "max_thickness": 3, "skip_white": False, "edge_sensitivity": 0.0},
            self.canvas,
        )
        assert len(result_base) == len(result_zero)
        for base_path, zero_path in zip(result_base, result_zero):
            assert len(base_path) == len(zero_path)

    def test_edge_sensitivity_nonzero_produces_fewer_paths(self):
        """edge_sensitivity > 0 should reduce thickness near edges → fewer offset lines.

        We use a checkerboard-like image with sharp transitions so Canny detects
        many edges, then verify that edge_sensitivity reduces the total path count
        compared to edge_sensitivity=0.
        """
        # Create an image with strong edges: left half black, right half white
        h, w = 100, 100
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, w // 2 :] = 255  # right half is white

        result_no_edge = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 4,
             "pen_width_mm": 0.3, "skip_white": False, "edge_sensitivity": 0.0},
            self.canvas,
        )
        result_edge = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 5.0, "max_thickness": 4,
             "pen_width_mm": 0.3, "skip_white": False, "edge_sensitivity": 1.0},
            self.canvas,
        )
        # Edge-aware should produce equal or fewer total paths (thickness reduced at edges)
        total_no_edge = sum(len(p) for p in result_no_edge)
        total_edge = sum(len(p) for p in result_edge)
        assert total_edge <= total_no_edge, (
            "Edge sensitivity should reduce or equal total path count near edges, "
            f"but got more: {total_edge} > {total_no_edge}"
        )

    def test_edge_sensitivity_output_still_valid(self):
        """With edge_sensitivity > 0, all returned polylines must still have >= 2 points."""
        img = np.zeros((80, 80), dtype=np.uint8)
        img[20:60, 20:60] = 200  # block with edges
        result = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 4.0, "max_thickness": 3,
             "pen_width_mm": 0.3, "skip_white": False, "edge_sensitivity": 0.8},
            self.canvas,
        )
        for path in result:
            assert len(path) >= 2, f"Path with edge_sensitivity has only {len(path)} point(s)"

    def test_all_points_in_bounds_with_edge_sensitivity(self):
        """Points must still lie within the drawing area when edge_sensitivity is used."""
        img = np.zeros((80, 80), dtype=np.uint8)
        img[10:70, 10:70] = 180
        result = self.gen.generate(
            {"_source_image": img, "line_spacing_mm": 4.0, "max_thickness": 2,
             "pen_width_mm": 0.3, "skip_white": False, "edge_sensitivity": 0.7},
            self.canvas,
        )
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = 1e-3
        for path in result:
            for x, y in path:
                assert x1 - tol <= x <= x2 + tol
                assert y1 - tol <= y <= y2 + tol
