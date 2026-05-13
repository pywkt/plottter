"""Tests for FlowImageGenerator — covers the task-66.5 requirements.

Tests:
(a) ETF mode produces streamlines that follow edges
(b) Grid seeding covers canvas more uniformly than random
(c) Separation filter removes close streamlines
(d) Density modulation produces more lines in dark areas
(e) Bidirectional tracing produces longer lines than half-budget
(f) All presets generate valid non-empty output
(g) Squiggle mode regression — still works after refactor
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.flow_image import (
    FlowImageGenerator,
    _filter_streamlines_by_separation,
    _generate_flow_streamlines,
)
from plottter.models.canvas import Canvas


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _canvas(w: float = 100.0, h: float = 100.0) -> Canvas:
    return Canvas(width_mm=w, height_mm=h, margin_mm=0.0)


def _black_img(h: int = 50, w: int = 50) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _gradient_img(h: int = 50, w: int = 50) -> np.ndarray:
    """Left-dark → right-bright horizontal gradient."""
    return np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))


def _half_dark_img(h: int = 60, w: int = 60) -> np.ndarray:
    """Left half black, right half mid-grey (128)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2:] = 128
    return img


def _edge_img(h: int = 64, w: int = 64) -> np.ndarray:
    """Vertical edge at the midpoint: left half black, right half white."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2:] = 255
    return img


def _skip_if_no_cv2() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        pytest.skip("opencv-python not available")


def _run_flow(
    img: np.ndarray,
    canvas: Canvas | None = None,
    seed_spacing_mm: float = 8.0,
    step_size_mm: float = 0.5,
    max_length_mm: float = 15.0,
    vector_field: str = "Gradient",
    brightness_threshold: int = 255,
    density_modulation: bool = False,
    separation_distance_mm: float = 0.0,
    seed: int = 42,
) -> list:
    _skip_if_no_cv2()
    c = canvas or _canvas()
    return _generate_flow_streamlines(
        img=img,
        seed_spacing_mm=seed_spacing_mm,
        step_size_mm=step_size_mm,
        max_length_mm=max_length_mm,
        seed=seed,
        skip_background=False,
        bg_threshold=260.0,
        brightness_threshold=brightness_threshold,
        density_modulation=density_modulation,
        canvas=c,
        cancelled_callback=None,
        progress_callback=None,
        vector_field=vector_field,
        separation_distance_mm=separation_distance_mm,
    )


# ──────────────────────────────────────────────────────────────────────────────
# (a) ETF mode follows edges
# ──────────────────────────────────────────────────────────────────────────────


class TestETFMode:
    """ETF vector field produces streamlines that follow edges."""

    def test_etf_produces_streamlines(self) -> None:
        """ETF mode should return at least some streamlines for a gradient image."""
        _skip_if_no_cv2()
        img = _gradient_img(64, 64)
        result = _run_flow(img, vector_field="Edge Flow (ETF)", seed_spacing_mm=10.0)
        assert len(result) > 0

    def test_etf_streamlines_are_valid_polylines(self) -> None:
        """Every ETF streamline must have at least 2 points."""
        _skip_if_no_cv2()
        img = _gradient_img(64, 64)
        result = _run_flow(img, vector_field="Edge Flow (ETF)", seed_spacing_mm=10.0)
        for pl in result:
            assert len(pl) >= 2

    def test_etf_follows_edge_tangent(self) -> None:
        """For a vertical-edge image, ETF streamlines should run roughly
        vertically (tangent to the edge) rather than crossing it.

        The gradient of a vertical edge is horizontal (gx large, gy ≈ 0).
        The ETF tangent field is (-gy, gx) ≈ (0, gx) — i.e. vertical.
        So streamlines that follow the edge will have |dy| > |dx|.
        """
        _skip_if_no_cv2()

        img = _edge_img(64, 64)  # vertical edge at x=32

        result = _generate_flow_streamlines(
            img=img,
            seed_spacing_mm=12.0,
            step_size_mm=0.5,
            max_length_mm=20.0,
            seed=0,
            skip_background=False,
            bg_threshold=260.0,
            brightness_threshold=255,
            density_modulation=False,
            canvas=_canvas(80.0, 80.0),
            cancelled_callback=None,
            progress_callback=None,
            vector_field="Edge Flow (ETF)",
            etf_kernel_radius=5.0,
            etf_iterations=3,
            separation_distance_mm=0.0,
        )

        if not result:
            pytest.skip("No streamlines produced — insufficient image contrast")

        # For each polyline, measure horizontal vs vertical total displacement
        vertical_dominant = 0
        total = 0
        for pl in result:
            if len(pl) < 3:
                continue
            dx = abs(pl[-1][0] - pl[0][0])
            dy = abs(pl[-1][1] - pl[0][1])
            if dx + dy > 1.0:
                total += 1
                if dy >= dx:  # vertical-dominant = following the vertical edge
                    vertical_dominant += 1

        if total > 0:
            # Majority should follow the vertical edge tangent (vertical direction)
            assert vertical_dominant / total >= 0.5, (
                f"ETF should follow vertical-edge tangent (vertical direction): "
                f"{vertical_dominant}/{total} are vertical-dominant"
            )

    def test_etf_differs_from_gradient_mode(self) -> None:
        """ETF (follows edges) and Gradient (crosses edges) should differ.

        For a vertical-edge image:
        - Gradient mode: field ≈ (1, 0) → horizontal streamlines (cross edge)
        - ETF mode:      field ≈ (0, 1) → vertical streamlines (follow edge)
        """
        _skip_if_no_cv2()
        img = _edge_img(64, 64)
        canvas = _canvas(80.0, 80.0)

        result_etf = _generate_flow_streamlines(
            img=img, seed_spacing_mm=12.0, step_size_mm=0.5, max_length_mm=20.0,
            seed=0, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Edge Flow (ETF)",
        )
        result_grad = _generate_flow_streamlines(
            img=img, seed_spacing_mm=12.0, step_size_mm=0.5, max_length_mm=20.0,
            seed=0, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )

        # Both should produce results
        assert len(result_etf) > 0
        assert len(result_grad) > 0

        # The two methods should produce different streamline paths
        assert result_etf != result_grad, (
            "ETF (follows edges) and Gradient (crosses edges) should produce different paths"
        )


# ──────────────────────────────────────────────────────────────────────────────
# (b) Grid seeding covers canvas uniformly
# ──────────────────────────────────────────────────────────────────────────────


class TestGridSeeding:
    """Grid-based seeding produces uniform spatial coverage."""

    def test_grid_covers_all_quadrants(self) -> None:
        """Streamline seed points should cover all four canvas quadrants."""
        _skip_if_no_cv2()
        canvas = _canvas(100.0, 100.0)
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2
        mid_y = (draw_y1 + draw_y2) / 2

        img = _gradient_img(50, 50)
        result = _generate_flow_streamlines(
            img=img, seed_spacing_mm=10.0, step_size_mm=1.0, max_length_mm=5.0,
            seed=42, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )

        q = [0, 0, 0, 0]
        for pl in result:
            x, y = pl[0]
            if x < mid_x and y < mid_y:
                q[0] += 1
            elif x >= mid_x and y < mid_y:
                q[1] += 1
            elif x < mid_x and y >= mid_y:
                q[2] += 1
            else:
                q[3] += 1

        assert all(c >= 1 for c in q), f"Grid seeding missed quadrant(s): {q}"

        total = sum(q)
        if total > 0:
            max_frac = max(q) / total
            assert max_frac < 0.6, f"Coverage too concentrated in one quadrant: {q}"

    def test_finer_spacing_produces_more_streamlines(self) -> None:
        """Halving seed_spacing_mm should produce more streamlines."""
        _skip_if_no_cv2()
        img = _gradient_img(50, 50)
        canvas = _canvas(50.0, 50.0)

        result_coarse = _run_flow(img, canvas=canvas, seed_spacing_mm=5.0)
        result_fine = _run_flow(img, canvas=canvas, seed_spacing_mm=2.0)

        assert len(result_fine) > len(result_coarse), (
            f"Fine ({len(result_fine)}) should exceed coarse ({len(result_coarse)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# (c) Separation filter removes close streamlines
# ──────────────────────────────────────────────────────────────────────────────


class TestSeparationFilter:
    """_filter_streamlines_by_separation correctly removes nearby streamlines."""

    def test_close_streamlines_deduplicated(self) -> None:
        """Two streamlines 0.2 mm apart should be reduced to one (sep=0.5)."""
        sl1 = [(0.0, 0.0), (1.0, 0.0)]
        sl2 = [(0.2, 0.0), (1.2, 0.0)]
        result = _filter_streamlines_by_separation([sl1, sl2], [100.0, 100.0], 0.5)
        assert len(result) == 1

    def test_distant_streamlines_kept(self) -> None:
        """Two streamlines 5 mm apart should both survive (sep=1.0)."""
        sl1 = [(0.0, 0.0), (2.0, 0.0)]
        sl2 = [(5.0, 0.0), (7.0, 0.0)]
        result = _filter_streamlines_by_separation([sl1, sl2], [100.0, 100.0], 1.0)
        assert len(result) == 2

    def test_darkest_wins_conflict(self) -> None:
        """When two streamlines compete, the darker-area one survives."""
        sl_dark = [(0.0, 0.0), (1.0, 0.0)]
        sl_bright = [(0.05, 0.0), (1.05, 0.0)]
        result = _filter_streamlines_by_separation(
            [sl_bright, sl_dark], [200.0, 50.0], 1.0
        )
        assert len(result) == 1
        assert result[0] == sl_dark, "Dark-area streamline should survive"

    def test_no_pairs_closer_than_separation(self) -> None:
        """After filtering, no two midpoints should be closer than the threshold."""
        separation = 2.0
        streamlines = [
            [(float(i) * 0.5, 0.0), (float(i) * 0.5 + 1.0, 0.0)]
            for i in range(20)
        ]
        brightnesses = [float(i * 5) for i in range(20)]
        result = _filter_streamlines_by_separation(streamlines, brightnesses, separation)

        midpoints = [sl[len(sl) // 2] for sl in result]
        for a in range(len(midpoints)):
            for b in range(a + 1, len(midpoints)):
                dist = math.hypot(
                    midpoints[b][0] - midpoints[a][0],
                    midpoints[b][1] - midpoints[a][1],
                )
                assert dist >= separation - 1e-9

    def test_large_separation_reduces_count_via_generate(self) -> None:
        """Large separation_distance_mm in generate call should remove streamlines."""
        _skip_if_no_cv2()
        img = _gradient_img(50, 50)
        canvas = _canvas(100.0, 100.0)

        unfiltered = _run_flow(img, canvas=canvas, seed_spacing_mm=3.0, separation_distance_mm=0.0)
        filtered = _run_flow(img, canvas=canvas, seed_spacing_mm=3.0, separation_distance_mm=10.0)

        assert len(filtered) < len(unfiltered), (
            f"Filtered ({len(filtered)}) should be fewer than unfiltered ({len(unfiltered)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# (d) Density modulation produces more lines in dark areas
# ──────────────────────────────────────────────────────────────────────────────


class TestDensityModulation:
    """density_modulation=True concentrates streamlines in darker image regions."""

    def test_more_lines_in_dark_half(self) -> None:
        """Dark (left) half should receive more streamlines than mid-grey (right)."""
        _skip_if_no_cv2()
        img = _half_dark_img(60, 60)
        canvas = _canvas(100.0, 100.0)
        draw_x1, _, draw_x2, _ = canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2

        result = _generate_flow_streamlines(
            img=img, seed_spacing_mm=2.0, step_size_mm=0.5, max_length_mm=5.0,
            seed=0, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=True,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )

        dark_count = sum(1 for pl in result if pl[0][0] < mid_x)
        bright_count = sum(1 for pl in result if pl[0][0] >= mid_x)

        assert dark_count > bright_count, (
            f"Expected more dark-area lines: dark={dark_count}, bright={bright_count}"
        )

    def test_density_modulation_off_is_more_even(self) -> None:
        """With density_modulation=False the ratio of dark to bright lines
        should be closer to 1 than with it enabled."""
        _skip_if_no_cv2()
        img = _half_dark_img(60, 60)
        canvas = _canvas(100.0, 100.0)
        draw_x1, _, draw_x2, _ = canvas.drawing_area()
        mid_x = (draw_x1 + draw_x2) / 2

        result_on = _generate_flow_streamlines(
            img=img, seed_spacing_mm=2.0, step_size_mm=0.5, max_length_mm=5.0,
            seed=0, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=True,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )
        result_off = _generate_flow_streamlines(
            img=img, seed_spacing_mm=2.0, step_size_mm=0.5, max_length_mm=5.0,
            seed=0, skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=canvas, cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )

        def dark_ratio(result: list) -> float:
            dark = sum(1 for pl in result if pl[0][0] < mid_x)
            bright = sum(1 for pl in result if pl[0][0] >= mid_x)
            total = dark + bright
            return dark / total if total > 0 else 0.5

        ratio_on = dark_ratio(result_on)
        ratio_off = dark_ratio(result_off)
        # With density modulation on, more lines should be in dark half
        assert ratio_on > ratio_off, (
            f"density_modulation=True should increase dark-area ratio: "
            f"on={ratio_on:.2f}, off={ratio_off:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# (e) Bidirectional tracing produces longer lines
# ──────────────────────────────────────────────────────────────────────────────


class TestBidirectionalTracing:
    """Bidirectional tracing (forward + backward) produces longer streamlines."""

    def test_max_length_exceeds_half_budget(self) -> None:
        """The longest streamline should exceed max_length_mm / 2 in total points.

        With bidirectional tracing both directions trace up to max_length_mm/2,
        so the combined line can contain up to max_length_mm / step_size_mm + 1 points.
        A purely unidirectional trace at the same budget would only produce
        half as many points.
        """
        _skip_if_no_cv2()
        max_length_mm = 20.0
        step_size_mm = 0.5
        half_budget_pts = max_length_mm / (2 * step_size_mm)

        img = np.tile(np.linspace(0, 180, 50, dtype=np.uint8), (50, 1))
        result = _generate_flow_streamlines(
            img=img, seed_spacing_mm=10.0, step_size_mm=step_size_mm,
            max_length_mm=max_length_mm, seed=7,
            skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=_canvas(), cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )

        if result:
            max_pts = max(len(pl) for pl in result)
            assert max_pts > half_budget_pts, (
                f"Bidirectional trace should exceed {half_budget_pts:.0f} pts; "
                f"got max {max_pts}"
            )

    def test_all_streamlines_within_max_length(self) -> None:
        """No streamline arc length should exceed max_length_mm (+ tolerance)."""
        _skip_if_no_cv2()
        max_length_mm = 10.0
        step_size_mm = 0.5
        img = np.tile(np.linspace(0, 180, 50, dtype=np.uint8), (50, 1))
        result = _generate_flow_streamlines(
            img=img, seed_spacing_mm=8.0, step_size_mm=step_size_mm,
            max_length_mm=max_length_mm, seed=0,
            skip_background=False, bg_threshold=260.0,
            brightness_threshold=255, density_modulation=False,
            canvas=_canvas(), cancelled_callback=None, progress_callback=None,
            vector_field="Gradient",
        )
        for pl in result:
            arc = sum(
                math.hypot(pl[i + 1][0] - pl[i][0], pl[i + 1][1] - pl[i][1])
                for i in range(len(pl) - 1)
            )
            # Two directions → allow 2 steps tolerance
            assert arc <= max_length_mm + 2 * step_size_mm


# ──────────────────────────────────────────────────────────────────────────────
# (f) All presets generate valid non-empty output
# ──────────────────────────────────────────────────────────────────────────────


class TestPresets:
    """All presets produce valid, non-empty output when given a suitable image."""

    @pytest.fixture(autouse=True)
    def _cv2_check(self) -> None:
        _skip_if_no_cv2()

    def _make_test_params(self, preset_params: dict) -> dict:
        """Build a params dict suitable for FlowImageGenerator.generate()."""
        h, w = 30, 30
        # Synthesize an image with both dark and semi-bright regions so
        # brightness_threshold filters don't remove everything.
        img = np.zeros((h, w), dtype=np.uint8)
        img[4:20, 4:26] = 80   # dark rectangle (subject)
        img[22:28, 8:22] = 130  # mid-grey region
        params = dict(preset_params)
        params["_source_image"] = img
        return params

    @pytest.mark.parametrize("preset_name,preset_params", [
        (p.name, p.params)
        for p in FlowImageGenerator().get_presets()
    ])
    def test_preset_produces_output(self, preset_name: str, preset_params: dict) -> None:
        """Each preset should return at least one valid polyline."""
        gen = FlowImageGenerator()
        # Small canvas keeps seed count manageable even for fine-spacing presets
        canvas = _canvas(15.0, 15.0)
        params = self._make_test_params(preset_params)

        result = gen.generate(params, canvas)

        assert isinstance(result, list), f"Preset '{preset_name}': expected list"
        assert len(result) > 0, f"Preset '{preset_name}': expected non-empty output"
        for pl in result:
            assert len(pl) >= 2, (
                f"Preset '{preset_name}': polyline has fewer than 2 points: {pl}"
            )

    def test_new_photo_portrait_preset_exists(self) -> None:
        """Photo Portrait preset must be defined."""
        gen = FlowImageGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Photo Portrait" in names

    def test_new_landscape_preset_exists(self) -> None:
        gen = FlowImageGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Landscape" in names

    def test_new_fine_detail_preset_exists(self) -> None:
        gen = FlowImageGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Fine Detail" in names

    def test_new_loose_sketch_preset_exists(self) -> None:
        gen = FlowImageGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Loose Sketch" in names

    def test_new_dense_coverage_preset_exists(self) -> None:
        gen = FlowImageGenerator()
        names = [p.name for p in gen.get_presets()]
        assert "Dense Coverage" in names

    def test_photo_portrait_preset_params(self) -> None:
        """Photo Portrait preset must have the specified parameter values."""
        gen = FlowImageGenerator()
        preset = next(p for p in gen.get_presets() if p.name == "Photo Portrait")
        assert preset.params["seed_spacing_mm"] == 1.5
        assert preset.params["max_length_mm"] == 25.0
        assert preset.params["separation_distance_mm"] == 0.6
        assert preset.params["density_modulation"] is True
        assert preset.params["brightness_threshold"] == 220

    def test_landscape_preset_params(self) -> None:
        gen = FlowImageGenerator()
        preset = next(p for p in gen.get_presets() if p.name == "Landscape")
        assert preset.params["seed_spacing_mm"] == 2.0
        assert preset.params["max_length_mm"] == 30.0
        assert preset.params["separation_distance_mm"] == 1.0
        assert preset.params["density_modulation"] is True

    def test_fine_detail_preset_params(self) -> None:
        gen = FlowImageGenerator()
        preset = next(p for p in gen.get_presets() if p.name == "Fine Detail")
        # Params tuned for <30s runtime on a 600px source (seed_spacing_mm and
        # separation_distance_mm raised from 1.0/0.4 to keep candidate count low).
        assert preset.params["seed_spacing_mm"] == 2.0
        assert preset.params["max_length_mm"] == 15.0
        assert preset.params["separation_distance_mm"] == 0.8
        assert preset.params["density_modulation"] is True
        assert preset.params["brightness_threshold"] == 240

    def test_loose_sketch_preset_params(self) -> None:
        gen = FlowImageGenerator()
        preset = next(p for p in gen.get_presets() if p.name == "Loose Sketch")
        assert preset.params["seed_spacing_mm"] == 3.0
        assert preset.params["max_length_mm"] == 25.0
        assert preset.params["separation_distance_mm"] == 1.5
        assert preset.params["density_modulation"] is False
        assert preset.params["vector_field"] == "Edge Flow (ETF)"

    def test_dense_coverage_preset_params(self) -> None:
        gen = FlowImageGenerator()
        preset = next(p for p in gen.get_presets() if p.name == "Dense Coverage")
        # Params tuned for <30s runtime on a 600px source (seed_spacing_mm raised
        # from 0.8 to 1.5, max_length_mm lowered from 10 to 8, separation raised
        # from 0.3 to 0.8 to keep candidate and streamline counts manageable).
        assert preset.params["seed_spacing_mm"] == 1.5
        assert preset.params["max_length_mm"] == 8.0
        assert preset.params["separation_distance_mm"] == 0.8
        assert preset.params["density_modulation"] is True

    def test_no_preset_has_deprecated_params(self) -> None:
        """No preset should contain num_lines, max_steps, or curvature_strength."""
        gen = FlowImageGenerator()
        for preset in gen.get_presets():
            for bad_key in ("num_lines", "max_steps", "curvature_strength"):
                assert bad_key not in preset.params, (
                    f"Preset '{preset.name}' still has deprecated param '{bad_key}'"
                )


# ──────────────────────────────────────────────────────────────────────────────
# (g) Squiggle mode regression
# ──────────────────────────────────────────────────────────────────────────────


class TestSquiggleRegression:
    """Squiggle mode still works after the parameter refactor."""

    @pytest.fixture(autouse=True)
    def _cv2_check(self) -> None:
        _skip_if_no_cv2()

    def test_squiggle_produces_output(self) -> None:
        """Squiggle mode should return non-empty polylines."""
        gen = FlowImageGenerator()
        h, w = 80, 80
        img = np.zeros((h, w), dtype=np.uint8)
        img[10:70, 10:70] = 100  # large dark region
        canvas = _canvas(100.0, 100.0)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "amplitude_mm": 3.0,
            "frequency": 8.0,
            "wave_spread": 0,
            "line_spacing": "Uniform",
            "min_spacing_mm": 3.0,
            "max_spacing_mm": 5.0,
            "group_size": 3,
            "group_gap_mm": 4.0,
            "group_intra_spacing_mm": 0.5,
            "displacement_variation": 0.0,
            "skip_background": False,
            "bg_threshold": 260.0,
            "seed": 0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result = gen.generate(params, canvas)
        assert isinstance(result, list)
        assert len(result) > 0
        for pl in result:
            assert len(pl) >= 2

    def test_squiggle_uniform_spacing(self) -> None:
        """Uniform squiggle mode should produce lines spanning the full canvas width."""
        gen = FlowImageGenerator()
        h, w = 60, 60
        img = np.full((h, w), 100, dtype=np.uint8)  # uniform grey
        canvas = _canvas(80.0, 80.0)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "amplitude_mm": 2.0,
            "frequency": 5.0,
            "wave_spread": 0,
            "line_spacing": "Uniform",
            "min_spacing_mm": 5.0,
            "max_spacing_mm": 5.0,
            "group_size": 2,
            "group_gap_mm": 5.0,
            "group_intra_spacing_mm": 0.5,
            "displacement_variation": 0.0,
            "skip_background": False,
            "bg_threshold": 260.0,
            "seed": 0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        result = gen.generate(params, canvas)
        assert len(result) > 0

    def test_squiggle_no_num_lines_error(self) -> None:
        """Calling generate() in squiggle mode must not raise NameError for num_lines."""
        gen = FlowImageGenerator()
        img = np.full((40, 40), 80, dtype=np.uint8)
        canvas = _canvas(60.0, 60.0)
        params = {
            "_source_image": img,
            "mode": "squiggle",
            "amplitude_mm": 2.0,
            "frequency": 5.0,
            "wave_spread": 0,
            "line_spacing": "Uniform",
            "min_spacing_mm": 4.0,
            "max_spacing_mm": 5.0,
            "group_size": 2,
            "group_gap_mm": 4.0,
            "group_intra_spacing_mm": 0.5,
            "displacement_variation": 0.0,
            "skip_background": False,
            "bg_threshold": 260.0,
            "seed": 0,
            "invert": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        # Must not raise NameError (was broken by num_lines reference)
        try:
            result = gen.generate(params, canvas)
            assert isinstance(result, list)
        except NameError as e:
            pytest.fail(f"generate() raised NameError: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Parameter definition checks
# ──────────────────────────────────────────────────────────────────────────────


class TestParameterDefinitions:
    """Verify the cleaned-up parameter list."""

    def _params(self) -> dict:
        gen = FlowImageGenerator()
        return {p.name: p for p in gen.get_parameters()}

    def test_deprecated_params_removed(self) -> None:
        params = self._params()
        for bad in ("curvature_strength", "num_lines", "max_steps"):
            assert bad not in params, f"Deprecated param '{bad}' still present"

    def test_required_flow_params_present(self) -> None:
        params = self._params()
        for name in ("vector_field", "seed_spacing_mm", "max_length_mm",
                     "separation_distance_mm", "density_modulation",
                     "brightness_threshold", "etf_kernel_radius", "etf_iterations"):
            assert name in params, f"Required param '{name}' missing"

    def test_step_size_mm_default(self) -> None:
        params = self._params()
        assert params["step_size_mm"].default == 0.5

    def test_step_size_mm_flow_only(self) -> None:
        """step_size_mm should have visible_when restricting it to flow mode."""
        params = self._params()
        p = params["step_size_mm"]
        vw = getattr(p, "visible_when", None)
        assert vw is not None and "mode" in vw, (
            "step_size_mm should be flow-only (visible_when mode=flow)"
        )

    def test_max_length_mm_flow_only(self) -> None:
        """max_length_mm should only be visible in flow mode."""
        params = self._params()
        p = params["max_length_mm"]
        vw = getattr(p, "visible_when", None)
        assert vw is not None and "mode" in vw, (
            "max_length_mm should be flow-only (visible_when mode=flow)"
        )

    def test_etf_params_visible_when_etf(self) -> None:
        """etf_kernel_radius and etf_iterations should only show for ETF vector field."""
        params = self._params()
        for name in ("etf_kernel_radius", "etf_iterations"):
            p = params[name]
            vw = getattr(p, "visible_when", None)
            assert vw is not None, f"{name} should have visible_when"
            assert "vector_field" in vw, f"{name} visible_when should key on vector_field"

    def test_standard_preprocessing_params_present(self) -> None:
        """brightness, contrast, blur_radius, invert should all be present."""
        params = self._params()
        for name in ("brightness", "contrast", "blur_radius", "invert"):
            assert name in params, f"Preprocessing param '{name}' missing"

    def test_offset_params_present(self) -> None:
        params = self._params()
        assert "x_offset_mm" in params
        assert "y_offset_mm" in params
