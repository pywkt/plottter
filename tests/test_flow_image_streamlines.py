"""Tests for the updated flow image streamline tracing algorithm.

Covers:
- Direct Euler integration (streamlines follow vector field smoothly)
- Bidirectional tracing (longer lines than unidirectional)
- Early termination at canvas bounds
- max_length_mm parameter replacing max_steps
- step_size_mm=0.5 default produces smoother curves
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.flow_image import _generate_flow_streamlines, _trace_one_direction
from plottter.models.canvas import Canvas


def _make_canvas(w_mm: float = 100.0, h_mm: float = 100.0) -> Canvas:
    return Canvas(width_mm=w_mm, height_mm=h_mm, margin_mm=0.0)


def _uniform_field(direction_deg: float, h: int = 50, w: int = 50) -> np.ndarray:
    """Create a uniform vector field pointing in a fixed direction."""
    angle = math.radians(direction_deg)
    vx = math.cos(angle)
    vy = math.sin(angle)
    field = np.zeros((h, w, 2), dtype=np.float32)
    field[:, :, 0] = vx
    field[:, :, 1] = vy
    return field


def _black_image(h: int = 50, w: int = 50) -> np.ndarray:
    """Black (all-zero) image — no background pixels, always content."""
    return np.zeros((h, w), dtype=np.uint8)


class TestTraceOneDirection:
    """Unit tests for the _trace_one_direction helper."""

    def test_traces_in_correct_direction(self) -> None:
        """Points should advance in the field direction each step."""
        field = _uniform_field(0.0)  # pointing right
        img = _black_image()
        pts = _trace_one_direction(
            field=field, img=img,
            x0_mm=50.0, y0_mm=50.0,
            direction=+1.0,
            step_size_mm=1.0, max_length_mm=5.0,
            draw_x1=0.0, draw_y1=0.0, draw_x2=100.0, draw_y2=100.0,
            img_w=50, img_h=50, draw_w=100.0, draw_h=100.0,
            skip_background=False, bg_threshold=240.0,
        )
        assert len(pts) > 0
        # All points should have larger x than seed (going right)
        for x, y in pts:
            assert x > 50.0
            assert abs(y - 50.0) < 0.1  # y should barely move

    def test_backward_direction_goes_left(self) -> None:
        """Backward direction (-1) should go opposite to the field."""
        field = _uniform_field(0.0)  # field points right
        img = _black_image()
        pts = _trace_one_direction(
            field=field, img=img,
            x0_mm=50.0, y0_mm=50.0,
            direction=-1.0,
            step_size_mm=1.0, max_length_mm=5.0,
            draw_x1=0.0, draw_y1=0.0, draw_x2=100.0, draw_y2=100.0,
            img_w=50, img_h=50, draw_w=100.0, draw_h=100.0,
            skip_background=False, bg_threshold=240.0,
        )
        assert len(pts) > 0
        for x, y in pts:
            assert x < 50.0  # backward = going left

    def test_stops_at_max_length(self) -> None:
        """Trace should not exceed max_length_mm."""
        field = _uniform_field(0.0)
        img = _black_image()
        step = 0.5
        max_len = 10.0
        pts = _trace_one_direction(
            field=field, img=img,
            x0_mm=50.0, y0_mm=50.0,
            direction=+1.0,
            step_size_mm=step, max_length_mm=max_len,
            draw_x1=0.0, draw_y1=0.0, draw_x2=100.0, draw_y2=100.0,
            img_w=50, img_h=50, draw_w=100.0, draw_h=100.0,
            skip_background=False, bg_threshold=240.0,
        )
        # Total length ≈ len(pts) * step_size ≤ max_len
        total_length = len(pts) * step
        assert total_length <= max_len + step  # allow one step tolerance

    def test_stops_at_canvas_bounds(self) -> None:
        """Trace should stop when it exits canvas bounds."""
        field = _uniform_field(0.0)  # pointing right
        img = _black_image()
        pts = _trace_one_direction(
            field=field, img=img,
            x0_mm=95.0, y0_mm=50.0,  # near right edge
            direction=+1.0,
            step_size_mm=1.0, max_length_mm=200.0,
            draw_x1=0.0, draw_y1=0.0, draw_x2=100.0, draw_y2=100.0,
            img_w=50, img_h=50, draw_w=100.0, draw_h=100.0,
            skip_background=False, bg_threshold=240.0,
        )
        # Should stop quickly since seed is near the right edge
        assert len(pts) < 10

    def test_stops_on_zero_magnitude_field(self) -> None:
        """Trace should stop when field magnitude drops below 0.01."""
        # Create a field with zero magnitude everywhere
        field = np.zeros((50, 50, 2), dtype=np.float32)
        img = _black_image()
        pts = _trace_one_direction(
            field=field, img=img,
            x0_mm=50.0, y0_mm=50.0,
            direction=+1.0,
            step_size_mm=1.0, max_length_mm=50.0,
            draw_x1=0.0, draw_y1=0.0, draw_x2=100.0, draw_y2=100.0,
            img_w=50, img_h=50, draw_w=100.0, draw_h=100.0,
            skip_background=False, bg_threshold=240.0,
        )
        # Should return no points since field magnitude is 0
        assert len(pts) == 0


class TestGenerateFlowStreamlines:
    """Integration tests for _generate_flow_streamlines."""

    def _run_streamlines(
        self,
        num_lines: int = 10,
        step_size_mm: float = 0.5,
        max_length_mm: float = 20.0,
        field_dir_deg: float = 0.0,
        skip_background: bool = False,
    ):
        """Helper: run streamline generation with a uniform right-pointing field."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas()
        img_h, img_w = 50, 50
        img = _black_image(img_h, img_w)

        # Build a synthetic image that produces a known gradient when Sobel is applied.
        # A horizontal gradient image (bright on right, dark on left) will produce
        # a gradient mostly in the X direction.  We test with a real image to ensure
        # the full pipeline works.
        img_gradient = np.tile(
            np.linspace(0, 255, img_w, dtype=np.uint8), (img_h, 1)
        )

        return _generate_flow_streamlines(
            img=img_gradient,
            num_lines=num_lines,
            step_size_mm=step_size_mm,
            max_length_mm=max_length_mm,
            curvature_strength=1.0,
            seed=42,
            skip_background=skip_background,
            bg_threshold=240.0,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

    def test_produces_polylines(self) -> None:
        """Should produce at least some polylines."""
        result = self._run_streamlines(num_lines=20)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_polylines_have_at_least_two_points(self) -> None:
        """Every returned polyline should have at least 2 points."""
        result = self._run_streamlines(num_lines=20)
        for polyline in result:
            assert len(polyline) >= 2

    def test_bidirectional_produces_longer_lines_than_unidirectional(self) -> None:
        """Bidirectional tracing produces longer lines on average.

        We verify by checking that max_length_mm is the total budget (split
        between two directions), so lines can be up to max_length_mm long
        rather than only half that.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas()
        img = np.tile(
            np.linspace(0, 200, 50, dtype=np.uint8), (50, 1)
        )

        # Use half max_length in one direction = half_length per direction
        max_length_mm = 20.0
        step_size_mm = 0.5

        result = _generate_flow_streamlines(
            img=img,
            num_lines=30,
            step_size_mm=step_size_mm,
            max_length_mm=max_length_mm,
            curvature_strength=1.0,
            seed=7,
            skip_background=False,
            bg_threshold=240.0,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        if result:
            # With bidirectional tracing, the max possible points per streamline
            # is max_length_mm / step_size_mm + 1 (for seed point).
            # The minimum should exceed what unidirectional tracing gives
            # (half of max_length_mm / step_size_mm + 1).
            max_pts = max(len(pl) for pl in result)
            # A unidirectional trace of half_length budget would give at most
            # max_length_mm / (2 * step_size_mm) points.
            half_budget_pts = max_length_mm / (2 * step_size_mm)
            # Bidirectional should be able to exceed the half-budget
            assert max_pts > half_budget_pts, (
                f"Expected bidirectional trace to exceed {half_budget_pts:.0f} points, "
                f"got max {max_pts}"
            )

    def test_streamlines_stay_within_canvas_bounds(self) -> None:
        """All points in all streamlines must be within the canvas drawing area."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas(100.0, 100.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        img = np.tile(
            np.linspace(0, 200, 50, dtype=np.uint8), (50, 1)
        )
        result = _generate_flow_streamlines(
            img=img,
            num_lines=50,
            step_size_mm=0.5,
            max_length_mm=40.0,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=240.0,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        for polyline in result:
            for x, y in polyline:
                assert draw_x1 <= x <= draw_x2, f"x={x} out of [{draw_x1}, {draw_x2}]"
                assert draw_y1 <= y <= draw_y2, f"y={y} out of [{draw_y1}, {draw_y2}]"

    def test_smoother_with_smaller_step(self) -> None:
        """step_size=0.5 should produce more points per mm than step_size=1.0."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas()
        img = np.tile(
            np.linspace(0, 200, 50, dtype=np.uint8), (50, 1)
        )
        common = dict(
            img=img, num_lines=20, max_length_mm=10.0,
            curvature_strength=1.0, seed=42,
            skip_background=False, bg_threshold=240.0,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )
        result_fine = _generate_flow_streamlines(step_size_mm=0.5, **common)
        result_coarse = _generate_flow_streamlines(step_size_mm=1.0, **common)

        if result_fine and result_coarse:
            avg_pts_fine = sum(len(pl) for pl in result_fine) / len(result_fine)
            avg_pts_coarse = sum(len(pl) for pl in result_coarse) / len(result_coarse)
            # Smaller step = more points per streamline
            assert avg_pts_fine > avg_pts_coarse

    def test_max_length_mm_parameter_respected(self) -> None:
        """Streamlines should not exceed max_length_mm in total length."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas()
        img = np.tile(
            np.linspace(0, 200, 50, dtype=np.uint8), (50, 1)
        )
        max_length_mm = 10.0
        step_size_mm = 0.5

        result = _generate_flow_streamlines(
            img=img,
            num_lines=20,
            step_size_mm=step_size_mm,
            max_length_mm=max_length_mm,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=240.0,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        for polyline in result:
            # Compute actual arc length
            arc_length = sum(
                math.hypot(polyline[i + 1][0] - polyline[i][0],
                           polyline[i + 1][1] - polyline[i][1])
                for i in range(len(polyline) - 1)
            )
            # Allow one extra step tolerance per direction (2 total)
            assert arc_length <= max_length_mm + 2 * step_size_mm, (
                f"Streamline length {arc_length:.2f}mm exceeds max {max_length_mm}mm"
            )


class TestParameterDefinitions:
    """Verify updated parameter definitions in FlowImageGenerator."""

    def test_max_length_mm_param_exists(self) -> None:
        """max_length_mm should be a defined parameter."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "max_length_mm" in param_names

    def test_max_steps_param_removed(self) -> None:
        """max_steps should no longer be a defined parameter."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "max_steps" not in param_names

    def test_step_size_mm_default_is_half(self) -> None:
        """step_size_mm default should be 0.5."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for p in gen.get_parameters():
            if p.name == "step_size_mm":
                assert p.default == 0.5, f"Expected 0.5, got {p.default}"
                break

    def test_max_length_mm_default(self) -> None:
        """max_length_mm default should be 20.0."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for p in gen.get_parameters():
            if p.name == "max_length_mm":
                assert p.default == 20.0, f"Expected 20.0, got {p.default}"
                break

    def test_presets_use_max_length_mm(self) -> None:
        """All presets should have max_length_mm, not max_steps."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for preset in gen.get_presets():
            assert "max_length_mm" in preset.params, (
                f"Preset '{preset.name}' missing max_length_mm"
            )
            assert "max_steps" not in preset.params, (
                f"Preset '{preset.name}' still has max_steps"
            )
