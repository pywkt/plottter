"""Tests for the updated flow image streamline tracing algorithm.

Covers:
- Direct Euler integration (streamlines follow vector field smoothly)
- Bidirectional tracing (longer lines than unidirectional)
- Early termination at canvas bounds
- max_length_mm parameter replacing max_steps
- step_size_mm=0.5 default produces smoother curves
- Grid-based seeding: uniform coverage, density modulation, brightness threshold
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


def _white_image(h: int = 50, w: int = 50) -> np.ndarray:
    """White (all-255) image — all pixels are background."""
    return np.full((h, w), 255, dtype=np.uint8)


def _gradient_image(h: int = 50, w: int = 50) -> np.ndarray:
    """Left-dark, right-bright horizontal gradient image."""
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def _run_streamlines(
    seed_spacing_mm: float = 5.0,
    step_size_mm: float = 0.5,
    max_length_mm: float = 20.0,
    skip_background: bool = False,
    brightness_threshold: int = 255,
    density_modulation: bool = False,
    img: np.ndarray | None = None,
    canvas: Canvas | None = None,
    seed: int = 42,
):
    """Helper: run streamline generation with optional parameters."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("opencv-python not available")

    if canvas is None:
        canvas = _make_canvas()
    if img is None:
        img = _gradient_image()

    return _generate_flow_streamlines(
        img=img,
        seed_spacing_mm=seed_spacing_mm,
        step_size_mm=step_size_mm,
        max_length_mm=max_length_mm,
        curvature_strength=1.0,
        seed=seed,
        skip_background=skip_background,
        bg_threshold=240.0,
        brightness_threshold=brightness_threshold,
        density_modulation=density_modulation,
        canvas=canvas,
        cancelled_callback=None,
        progress_callback=None,
        vector_field="Gradient",
    )


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

    def test_produces_polylines(self) -> None:
        """Should produce at least some polylines."""
        result = _run_streamlines(seed_spacing_mm=5.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_polylines_have_at_least_two_points(self) -> None:
        """Every returned polyline should have at least 2 points."""
        result = _run_streamlines(seed_spacing_mm=5.0)
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
            seed_spacing_mm=10.0,
            step_size_mm=step_size_mm,
            max_length_mm=max_length_mm,
            curvature_strength=1.0,
            seed=7,
            skip_background=False,
            bg_threshold=240.0,
            brightness_threshold=255,
            density_modulation=False,
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
            seed_spacing_mm=5.0,
            step_size_mm=0.5,
            max_length_mm=40.0,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=240.0,
            brightness_threshold=255,
            density_modulation=False,
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
            img=img, seed_spacing_mm=10.0, max_length_mm=10.0,
            curvature_strength=1.0, seed=42,
            skip_background=False, bg_threshold=240.0,
            brightness_threshold=255, density_modulation=False,
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
            seed_spacing_mm=10.0,
            step_size_mm=step_size_mm,
            max_length_mm=max_length_mm,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=240.0,
            brightness_threshold=255,
            density_modulation=False,
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


class TestGridSeeding:
    """Tests for the new grid-based seeding behavior."""

    def test_grid_produces_uniform_coverage(self) -> None:
        """Grid seeding should distribute seeds more uniformly than random.

        Check that seeds cover the canvas reasonably by verifying that the
        canvas is divided into quadrants and each quadrant has roughly equal
        seed representation in the produced streamlines.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        canvas = _make_canvas(100.0, 100.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2
        mid_y = (draw_y1 + draw_y2) / 2

        # Use a gradient image with actual pixel variation so the vector field is
        # non-trivial and streamlines can be traced.  Set brightness_threshold=255
        # so all seeds survive filtering.
        img = _gradient_image(50, 50)
        result = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=10.0,  # coarse grid → manageable number of seeds
            step_size_mm=1.0,
            max_length_mm=5.0,
            curvature_strength=1.0,
            seed=42,
            skip_background=False,
            bg_threshold=260.0,
            brightness_threshold=255,
            density_modulation=False,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        # Count which quadrant each streamline's first point falls in
        q = [0, 0, 0, 0]  # TL, TR, BL, BR
        for polyline in result:
            x, y = polyline[0]
            if x < mid_x and y < mid_y:
                q[0] += 1
            elif x >= mid_x and y < mid_y:
                q[1] += 1
            elif x < mid_x and y >= mid_y:
                q[2] += 1
            else:
                q[3] += 1

        # Each quadrant should have at least 1 streamline
        assert all(count >= 1 for count in q), (
            f"Uneven quadrant coverage: {q}"
        )

        # Max/min ratio should be reasonable (not all in one quadrant)
        total = sum(q)
        if total > 0:
            max_frac = max(q) / total
            assert max_frac < 0.6, f"Too concentrated in one quadrant: {q}"

    def test_smaller_spacing_produces_more_seeds(self) -> None:
        """seed_spacing_mm=1.0 should produce more seeds than spacing=5.0."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        # Use a gradient image so the vector field is non-trivial and streamlines
        # can be traced; brightness_threshold=255 keeps all seeds
        img = _gradient_image(50, 50)
        canvas = _make_canvas(50.0, 50.0)

        result_dense = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=1.0,
            step_size_mm=0.5,
            max_length_mm=5.0,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=260.0,
            brightness_threshold=255,
            density_modulation=False,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        result_sparse = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=5.0,
            step_size_mm=0.5,
            max_length_mm=5.0,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=260.0,
            brightness_threshold=255,
            density_modulation=False,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        assert len(result_dense) > len(result_sparse), (
            f"Dense ({len(result_dense)}) should exceed sparse ({len(result_sparse)})"
        )

    def test_brightness_threshold_zero_removes_all_seeds_from_white_image(self) -> None:
        """brightness_threshold=0 should remove all seeds from a white image."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        img = _white_image(50, 50)
        canvas = _make_canvas(50.0, 50.0)

        result = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=2.0,
            step_size_mm=0.5,
            max_length_mm=10.0,
            curvature_strength=1.0,
            seed=42,
            skip_background=False,
            bg_threshold=260.0,  # high bg_threshold so skip_background never triggers
            brightness_threshold=0,  # threshold=0 → everything >= 0 is removed
            density_modulation=False,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        assert len(result) == 0, (
            f"Expected 0 streamlines from white image with threshold=0, got {len(result)}"
        )

    def test_density_modulation_produces_more_lines_in_dark_areas(self) -> None:
        """With density_modulation=True, dark areas should get more streamlines."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv-python not available")

        # Left half black (dark), right half grey (bright ~128)
        h, w = 50, 50
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, w // 2:] = 128

        canvas = _make_canvas(100.0, 100.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2

        result = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=2.0,
            step_size_mm=0.5,
            max_length_mm=5.0,
            curvature_strength=1.0,
            seed=0,
            skip_background=False,
            bg_threshold=255.0,
            brightness_threshold=255,
            density_modulation=True,
            canvas=canvas,
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Gradient",
        )

        # Count streamlines in dark (left) vs bright (right) halves
        dark_count = 0
        bright_count = 0
        for polyline in result:
            # Use the first point to determine which half it's in
            x0 = polyline[0][0]
            if x0 < mid_x:
                dark_count += 1
            else:
                bright_count += 1

        # Dark area should have significantly more streamlines
        assert dark_count > bright_count, (
            f"Expected more streamlines in dark areas: dark={dark_count}, bright={bright_count}"
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

    def test_seed_spacing_mm_param_exists(self) -> None:
        """seed_spacing_mm should be a defined parameter."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "seed_spacing_mm" in param_names

    def test_brightness_threshold_param_exists(self) -> None:
        """brightness_threshold should be a defined parameter."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "brightness_threshold" in param_names

    def test_density_modulation_param_exists(self) -> None:
        """density_modulation should be a defined parameter."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "density_modulation" in param_names

    def test_seed_spacing_mm_default(self) -> None:
        """seed_spacing_mm default should be 2.0."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for p in gen.get_parameters():
            if p.name == "seed_spacing_mm":
                assert p.default == 2.0, f"Expected 2.0, got {p.default}"
                break

    def test_brightness_threshold_default(self) -> None:
        """brightness_threshold default should be 230."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for p in gen.get_parameters():
            if p.name == "brightness_threshold":
                assert p.default == 230, f"Expected 230, got {p.default}"
                break

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

    def test_presets_use_seed_spacing_mm_for_flow(self) -> None:
        """All flow presets should have seed_spacing_mm."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for preset in gen.get_presets():
            if preset.params.get("mode") == "flow":
                assert "seed_spacing_mm" in preset.params, (
                    f"Flow preset '{preset.name}' missing seed_spacing_mm"
                )

    def test_presets_have_brightness_threshold(self) -> None:
        """All presets should have brightness_threshold."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for preset in gen.get_presets():
            assert "brightness_threshold" in preset.params, (
                f"Preset '{preset.name}' missing brightness_threshold"
            )

    def test_presets_have_density_modulation(self) -> None:
        """All presets should have density_modulation."""
        from plottter.generators.flow_image import FlowImageGenerator
        gen = FlowImageGenerator()
        for preset in gen.get_presets():
            assert "density_modulation" in preset.params, (
                f"Preset '{preset.name}' missing density_modulation"
            )

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
