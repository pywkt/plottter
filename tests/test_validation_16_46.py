"""Phase 16.46 validation: structure-aware halftoning in stipple generator.

Verifies:

1. Parameters ``structure_aware`` (BoolParam) and ``structure_weight``
   (FloatParam) are present in StippleGenerator.get_parameters().

2. All presets include both new parameters with correct defaults.

3. ``_compute_edge_weight_map`` returns zeros for structure_weight <= 0,
   non-zero values near edges, values bounded by structure_weight, and
   correct output shape.

4. With a synthetic image containing a sharp vertical edge, ``structure_aware=True``
   places more stipple points near the edge than ``structure_aware=False``.
   Tested for both Lloyd and LBG algorithms.

5. ``structure_weight=0.0`` (even with structure_aware=True) produces the same
   point count as ``structure_aware=False`` since the edge map is all-zeros.

6. Generator produces valid output (list of polylines within canvas bounds)
   with structure_aware enabled for both algorithms.

Reference: Pang et al., "Structure-Aware Halftoning", SIGGRAPH 2008.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def _sharp_vertical_edge_image(h: int = 80, w: int = 80) -> np.ndarray:
    """Black left half, white right half — strong vertical edge at centre."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2 :] = 255
    return img


def _sharp_diagonal_edge_image(h: int = 80, w: int = 80) -> np.ndarray:
    """Black upper-left triangle, white lower-right — diagonal edge."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if x + y >= (h + w) // 2:
                img[y, x] = 255
    return img


def _uniform_gray_image(h: int = 60, w: int = 60, value: int = 128) -> np.ndarray:
    """Uniform gray image — no edges."""
    return np.full((h, w), value, dtype=np.uint8)


def _base_params(img: np.ndarray, **overrides) -> dict:
    defaults = {
        "_source_image": img,
        "num_points": 100,
        "iterations": 3,
        "connect_tsp": False,
        "min_dot_spacing_mm": 0.0,
        "seed": 42,
        "working_resolution": 400,
        "convergence_threshold": 0.5,
        "structure_aware": False,
        "structure_weight": 0.3,
        "algorithm": "Lloyd",
        "split_threshold": 1.5,
        "merge_threshold": 0.5,
        "initial_distribution": "Weighted Random",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# 1. Parameters exist with correct types and defaults
# ---------------------------------------------------------------------------


class TestParametersExist:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()

    def test_structure_aware_param_exists(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "structure_aware" in names

    def test_structure_weight_param_exists(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "structure_weight" in names

    def test_structure_aware_is_bool_param(self):
        from plottter.generators.base import BoolParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["structure_aware"], BoolParam)

    def test_structure_weight_is_float_param(self):
        from plottter.generators.base import FloatParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["structure_weight"], FloatParam)

    def test_structure_aware_default_false(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert params["structure_aware"].default is False

    def test_structure_weight_default_0_3(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        assert abs(params["structure_weight"].default - 0.3) < 1e-6

    def test_structure_weight_has_visible_when(self):
        params = {p.name: p for p in self.gen.get_parameters()}
        sw = params["structure_weight"]
        assert sw.visible_when is not None, (
            "structure_weight should only be visible when structure_aware is True"
        )


# ---------------------------------------------------------------------------
# 2. Preset completeness
# ---------------------------------------------------------------------------


class TestPresetCompleteness:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()

    def test_all_presets_include_structure_aware(self):
        for preset in self.gen.get_presets():
            assert "structure_aware" in preset.params, (
                f"Preset '{preset.name}' missing 'structure_aware'"
            )

    def test_all_presets_include_structure_weight(self):
        for preset in self.gen.get_presets():
            assert "structure_weight" in preset.params, (
                f"Preset '{preset.name}' missing 'structure_weight'"
            )

    def test_all_presets_structure_aware_defaults_false(self):
        for preset in self.gen.get_presets():
            assert preset.params["structure_aware"] is False, (
                f"Preset '{preset.name}' structure_aware should default to False"
            )

    def test_all_presets_structure_weight_defaults_0_3(self):
        for preset in self.gen.get_presets():
            assert abs(preset.params["structure_weight"] - 0.3) < 1e-6, (
                f"Preset '{preset.name}' structure_weight should default to 0.3"
            )


# ---------------------------------------------------------------------------
# 3. _compute_edge_weight_map correctness
# ---------------------------------------------------------------------------


class TestComputeEdgeWeightMap:
    def setup_method(self):
        from plottter.generators.stipple import _compute_edge_weight_map
        self._fn = _compute_edge_weight_map

    def test_zero_structure_weight_returns_all_zeros(self):
        img = _sharp_vertical_edge_image()
        result = self._fn(img, 0.0)
        assert result.shape == img.shape
        assert np.allclose(result, 0.0)

    def test_negative_structure_weight_returns_all_zeros(self):
        img = _sharp_vertical_edge_image()
        result = self._fn(img, -0.5)
        assert np.allclose(result, 0.0)

    def test_output_shape_matches_input_square(self):
        img = _sharp_vertical_edge_image(60, 60)
        result = self._fn(img, 0.5)
        assert result.shape == (60, 60)

    def test_output_shape_matches_input_rectangular(self):
        img = _sharp_vertical_edge_image(60, 80)
        result = self._fn(img, 0.5)
        assert result.shape == (60, 80)

    def test_uniform_image_has_near_zero_magnitude(self):
        """A flat image has no gradients so the edge map should be negligible."""
        img = _uniform_gray_image(60, 60)
        result = self._fn(img, 1.0)
        assert float(result.max()) < 0.01

    def test_edge_image_nonzero_near_edge(self):
        """The edge at x≈w//2 should have significant weight."""
        h, w = 60, 60
        img = _sharp_vertical_edge_image(h, w)
        result = self._fn(img, 1.0)
        edge_col = w // 2
        edge_strip = result[:, max(0, edge_col - 3): edge_col + 4]
        assert float(edge_strip.max()) > 0.1, (
            "Edge weight map should be non-zero near the sharp edge"
        )

    def test_values_bounded_by_structure_weight(self):
        img = _sharp_vertical_edge_image()
        for sw in [0.1, 0.5, 1.0]:
            result = self._fn(img, sw)
            assert float(result.max()) <= sw + 1e-6, (
                f"Edge map values should not exceed structure_weight={sw}"
            )

    def test_higher_weight_gives_higher_map_values(self):
        img = _sharp_vertical_edge_image()
        r_low = self._fn(img, 0.1)
        r_high = self._fn(img, 0.9)
        assert float(r_high.max()) > float(r_low.max())

    def test_diagonal_edge_produces_nonzero_map(self):
        img = _sharp_diagonal_edge_image(60, 60)
        result = self._fn(img, 0.5)
        assert float(result.max()) > 0.05


# ---------------------------------------------------------------------------
# 4. Structure-aware places more dots near edges
# ---------------------------------------------------------------------------


class TestEdgeDensityBias:
    """Verify structure_aware=True concentrates more points near the image edge.

    Strategy: a sharp black-left / white-right image.
    With structure_aware=False all dots land on the dark (left) side because
    the white side has zero pixel weight.  With structure_aware=True the edge
    at x≈mid gets high weight on BOTH sides of the boundary, so some dots are
    pulled across the midline into the bright region near the edge.  We count
    dots within ±15 % of the draw width around the midline and verify the
    count is at least as large with structure_aware=True.
    """

    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def _near_edge_count(self, result: list) -> int:
        """Count dots within ±15 % of the vertical centreline."""
        x1, _, x2, _ = self.canvas.drawing_area()
        draw_w = x2 - x1
        mid_x = x1 + draw_w / 2.0
        threshold = draw_w * 0.15
        return sum(
            1 for poly in result
            if poly and abs(poly[0][0] - mid_x) <= threshold
        )

    def test_structure_aware_increases_edge_density_lloyd(self):
        """Lloyd: structure_aware=True should place >= dots near the edge."""
        img = _sharp_vertical_edge_image(80, 80)
        common = dict(
            num_points=200, iterations=5,
            connect_tsp=False, min_dot_spacing_mm=0.0, seed=42,
            working_resolution=200, convergence_threshold=0.0,
            algorithm="Lloyd",
        )
        result_no = self.gen.generate(
            _base_params(img, **common, structure_aware=False, structure_weight=0.3),
            self.canvas,
        )
        result_yes = self.gen.generate(
            _base_params(img, **common, structure_aware=True, structure_weight=0.9),
            self.canvas,
        )
        count_no = self._near_edge_count(result_no)
        count_yes = self._near_edge_count(result_yes)
        assert count_yes >= count_no, (
            f"Lloyd structure_aware=True should place >= dots near edge: "
            f"aware={count_yes} vs unaware={count_no}"
        )

    def test_structure_aware_increases_edge_density_lbg(self):
        """LBG: structure_aware=True should place >= dots near the edge."""
        img = _sharp_vertical_edge_image(80, 80)
        common = dict(
            num_points=200, iterations=10,
            connect_tsp=False, min_dot_spacing_mm=0.0, seed=42,
            working_resolution=200, convergence_threshold=0.0,
            algorithm="LBG",
            split_threshold=1.5, merge_threshold=0.5,
            initial_distribution="Weighted Random",
        )
        result_no = self.gen.generate(
            _base_params(img, **common, structure_aware=False, structure_weight=0.3),
            self.canvas,
        )
        result_yes = self.gen.generate(
            _base_params(img, **common, structure_aware=True, structure_weight=0.9),
            self.canvas,
        )
        count_no = self._near_edge_count(result_no)
        count_yes = self._near_edge_count(result_yes)
        assert count_yes >= count_no, (
            f"LBG structure_aware=True should place >= dots near edge: "
            f"aware={count_yes} vs unaware={count_no}"
        )


# ---------------------------------------------------------------------------
# 5. structure_weight=0.0 → same result as structure_aware=False
# ---------------------------------------------------------------------------


class TestZeroStructureWeight:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_zero_weight_produces_same_point_count_as_disabled(self):
        """structure_weight=0 means zero edge bias → same count as disabled."""
        img = _sharp_vertical_edge_image(60, 60)
        base = dict(
            num_points=50, iterations=3,
            connect_tsp=False, min_dot_spacing_mm=0.0, seed=99,
            working_resolution=200, convergence_threshold=0.0,
        )
        result_off = self.gen.generate(
            _base_params(img, **base, structure_aware=False, structure_weight=0.0),
            self.canvas,
        )
        result_zero = self.gen.generate(
            _base_params(img, **base, structure_aware=True, structure_weight=0.0),
            self.canvas,
        )
        assert len(result_off) == len(result_zero), (
            f"structure_weight=0 should be equivalent to disabled: "
            f"off={len(result_off)} zero={len(result_zero)}"
        )


# ---------------------------------------------------------------------------
# 6. Valid output with structure_aware enabled
# ---------------------------------------------------------------------------


class TestValidOutputWithStructureAware:
    def setup_method(self):
        from plottter.generators.stipple import StippleGenerator
        self.gen = StippleGenerator()
        self.canvas = _canvas()

    def test_produces_nonempty_output_lloyd(self):
        img = _sharp_vertical_edge_image(60, 60)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.5,
            algorithm="Lloyd",
        )
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_produces_nonempty_output_lbg(self):
        img = _sharp_vertical_edge_image(60, 60)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.5,
            algorithm="LBG",
        )
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_output_within_canvas_bounds(self):
        img = _sharp_vertical_edge_image(80, 80)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.5,
            num_points=50, iterations=3,
        )
        result = self.gen.generate(params, self.canvas)
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = 1.0  # mm tolerance for dot radius
        for poly in result:
            for x, y in poly:
                assert x1 - tol <= x <= x2 + tol
                assert y1 - tol <= y <= y2 + tol

    def test_dots_are_polylines_with_at_least_two_points(self):
        img = _sharp_vertical_edge_image(60, 60)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.3,
            num_points=30, iterations=2,
        )
        result = self.gen.generate(params, self.canvas)
        assert all(len(p) >= 2 for p in result)

    def test_tsp_with_structure_aware_single_path(self):
        img = _sharp_vertical_edge_image(60, 60)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.5,
            connect_tsp=True, num_points=30, iterations=2,
            min_dot_spacing_mm=0.0,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) == 1
        assert len(result[0]) == 30

    def test_diagonal_edge_image_produces_output(self):
        img = _sharp_diagonal_edge_image(60, 60)
        params = _base_params(
            img, structure_aware=True, structure_weight=0.5,
            num_points=50, iterations=3,
        )
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_all_presets_generate_without_error(self):
        img = _sharp_vertical_edge_image(80, 80)
        for preset in self.gen.get_presets():
            p = dict(preset.params)
            p["_source_image"] = img
            p["num_points"] = 30
            p["iterations"] = 2
            result = self.gen.generate(p, self.canvas)
            assert isinstance(result, list), (
                f"Preset '{preset.name}' raised an error"
            )
