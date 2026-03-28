"""Tests for StippleGenerator TSP presets and TSP path behaviour.

Tests:
(a) TSP mode produces exactly 1 polyline
(b) Polyline length (vertex count) equals the number of stipple points
(c) Existing dot presets remain unchanged (render_mode == "Dots")
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas():
    from plottter.models.canvas import Canvas
    return Canvas(width_mm=200.0, height_mm=200.0, margin_mm=10.0)


def make_gradient_image(w: int = 80, h: int = 80) -> np.ndarray:
    """Left-dark, right-bright gradient for reproducible stipple density."""
    img = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        img[:, col] = int(col / (w - 1) * 255)
    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    from plottter.generators.stipple import StippleGenerator
    return StippleGenerator()


@pytest.fixture
def canvas():
    return make_canvas()


@pytest.fixture
def img():
    return make_gradient_image()


def _base_params(img, **overrides) -> dict:
    base = {
        "_source_image": img,
        "num_points": 30,
        "iterations": 2,
        "min_dot_spacing_mm": 0.0,
        "seed": 42,
        "render_mode": "TSP Path",
        "tsp_optimize": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (a) TSP mode produces exactly 1 polyline
# ---------------------------------------------------------------------------

class TestTSPSinglePolyline:
    def test_tsp_path_produces_single_polyline(self, gen, canvas, img):
        """TSP Path render_mode must return exactly one polyline."""
        params = _base_params(img, num_points=30)
        result = gen.generate(params, canvas)
        assert len(result) == 1, (
            f"TSP Path mode should produce 1 polyline, got {len(result)}"
        )

    def test_tsp_path_single_polyline_small(self, gen, canvas, img):
        """TSP Path with a small number of points still returns 1 polyline."""
        params = _base_params(img, num_points=5)
        result = gen.generate(params, canvas)
        assert len(result) == 1

    def test_tsp_path_single_polyline_medium(self, gen, canvas, img):
        """TSP Path with 50 points returns exactly 1 polyline."""
        params = _base_params(img, num_points=50)
        result = gen.generate(params, canvas)
        assert len(result) == 1

    def test_tsp_path_no_optimize_still_single_polyline(self, gen, canvas, img):
        """tsp_optimize=False must still produce exactly 1 polyline."""
        params = _base_params(img, num_points=30, tsp_optimize=False)
        result = gen.generate(params, canvas)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# (b) Polyline vertex count equals number of stipple points
# ---------------------------------------------------------------------------

class TestTSPVertexCount:
    @pytest.mark.parametrize("n", [10, 25, 50])
    def test_vertex_count_matches_num_points(self, gen, canvas, img, n):
        """TSP path must have exactly num_points vertices."""
        params = _base_params(img, num_points=n)
        result = gen.generate(params, canvas)
        assert len(result) == 1
        assert len(result[0]) == n, (
            f"TSP path should have {n} vertices, got {len(result[0])}"
        )

    def test_vertex_count_with_optimize_false(self, gen, canvas, img):
        """Vertex count is correct when 2-opt optimisation is disabled."""
        n = 20
        params = _base_params(img, num_points=n, tsp_optimize=False)
        result = gen.generate(params, canvas)
        assert len(result[0]) == n


# ---------------------------------------------------------------------------
# (c) Existing dot presets remain unchanged
# ---------------------------------------------------------------------------

DOT_PRESET_NAMES = [
    "Default Stipple",
    "Dense Stipple",
    "Quick Preview",
    "Portrait Photo",
    "LBG Default",
    "LBG Few Seeds",
]


class TestExistingDotPresetsUnchanged:
    def _get_preset(self, gen, name: str):
        for p in gen.get_presets():
            if p.name == name:
                return p
        return None

    @pytest.mark.parametrize("preset_name", DOT_PRESET_NAMES)
    def test_dot_preset_render_mode(self, gen, preset_name):
        """Dot presets must still have render_mode == 'Dots'."""
        preset = self._get_preset(gen, preset_name)
        assert preset is not None, f"Preset '{preset_name}' not found"
        assert preset.params.get("render_mode") == "Dots", (
            f"Preset '{preset_name}' should have render_mode='Dots', "
            f"got {preset.params.get('render_mode')!r}"
        )

    def test_dot_preset_count_unchanged(self, gen):
        """The total number of dot-mode presets must not have decreased."""
        dot_presets = [
            p for p in gen.get_presets()
            if p.params.get("render_mode") == "Dots"
        ]
        assert len(dot_presets) >= len(DOT_PRESET_NAMES), (
            f"Expected at least {len(DOT_PRESET_NAMES)} dot presets, "
            f"got {len(dot_presets)}"
        )


# ---------------------------------------------------------------------------
# New presets: TSP Portrait and TSP Dense
# ---------------------------------------------------------------------------

class TestNewTSPPresets:
    def _get_preset(self, gen, name: str):
        for p in gen.get_presets():
            if p.name == name:
                return p
        return None

    def test_tsp_portrait_exists(self, gen):
        preset = self._get_preset(gen, "TSP Portrait")
        assert preset is not None, "Preset 'TSP Portrait' must exist"

    def test_tsp_portrait_render_mode(self, gen):
        preset = self._get_preset(gen, "TSP Portrait")
        assert preset.params.get("render_mode") == "TSP Path"

    def test_tsp_portrait_num_points(self, gen):
        preset = self._get_preset(gen, "TSP Portrait")
        assert preset.params.get("num_points") == 2000

    def test_tsp_portrait_tsp_optimize(self, gen):
        preset = self._get_preset(gen, "TSP Portrait")
        assert preset.params.get("tsp_optimize") is True

    def test_tsp_dense_exists(self, gen):
        preset = self._get_preset(gen, "TSP Dense")
        assert preset is not None, "Preset 'TSP Dense' must exist"

    def test_tsp_dense_render_mode(self, gen):
        preset = self._get_preset(gen, "TSP Dense")
        assert preset.params.get("render_mode") == "TSP Path"

    def test_tsp_dense_num_points(self, gen):
        preset = self._get_preset(gen, "TSP Dense")
        assert preset.params.get("num_points") == 5000

    def test_tsp_dense_tsp_optimize(self, gen):
        preset = self._get_preset(gen, "TSP Dense")
        assert preset.params.get("tsp_optimize") is True

    def test_all_presets_have_required_keys(self, gen):
        """Every preset must declare render_mode and tsp_optimize."""
        required = {"render_mode", "tsp_optimize"}
        for preset in gen.get_presets():
            missing = required - set(preset.params.keys())
            assert not missing, (
                f"Preset '{preset.name}' is missing keys: {missing}"
            )
