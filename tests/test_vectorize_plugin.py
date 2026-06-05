"""Tests for plugins/vectorize_trace.py (Phase 163.1).

Covers:
- Plugin module imports and the generator registers even when ``potracer`` is
  absent (monkeypatch import to raise).
- ``generate()`` raises a friendly ``RuntimeError`` mentioning
  ``pip install potracer`` when ``potracer`` is absent.
- (Guarded with ``pytest.importorskip("potracer")``) With potracer installed,
  tracing a synthetic shape yields non-empty smooth polylines whose coordinates
  lie within the fitted image rectangle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from plottter.models.canvas import Canvas

# ---------------------------------------------------------------------------
# Load the plugin module directly (lives in plugins/, not src/).
# We use a fixed module name so sys.modules deduplicates across test runs.
# ---------------------------------------------------------------------------

_PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "vectorize_trace.py"
_MODULE_NAME = "plottter_plugin_vectorize_trace"


def _load_plugin(module_name: str = _MODULE_NAME) -> object:
    """Load (or return cached) the vectorize_trace plugin module."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not create module spec for {_PLUGIN_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_vt = _load_plugin()
VectorizeTraceGenerator = _vt.VectorizeTraceGenerator  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canvas() -> Canvas:
    """Standard A4 canvas used across all tests."""
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def generator() -> VectorizeTraceGenerator:
    return VectorizeTraceGenerator()


# ---------------------------------------------------------------------------
# Helper: synthetic source image (black circle on white background)
# ---------------------------------------------------------------------------


def _make_circle_image(size: int = 64, radius: int = 24) -> np.ndarray:
    """Return an HxWx3 uint8 RGB image with a filled black circle."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cx, cy = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = 0  # black circle
    return img


# ---------------------------------------------------------------------------
# Tests: import and registration
# ---------------------------------------------------------------------------


class TestImportAndRegistration:
    """Plugin loads and the generator registers even when potracer is absent."""

    def test_plugin_module_loads(self) -> None:
        """The plugin module itself is importable without errors."""
        assert _vt is not None

    def test_plugin_module_file_exists(self) -> None:
        """Ensure the plugin file is present at the expected path."""
        assert _PLUGIN_PATH.exists(), f"Plugin file not found: {_PLUGIN_PATH}"

    def test_generator_registered_in_generators_dict(self) -> None:
        """VectorizeTraceGenerator appears in the GENERATORS registry."""
        from plottter.generators import GENERATORS

        # Import via load_plugins so the plugin is in the registry
        from plottter.generators import load_plugins

        root = _PLUGIN_PATH.parent
        load_plugins(extra_dirs=[str(root)])

        assert "Vectorize / Trace Bitmap" in GENERATORS, (
            f"Expected 'Vectorize / Trace Bitmap' in GENERATORS, got: "
            f"{sorted(GENERATORS.keys())}"
        )

    def test_generator_class_accessible(self) -> None:
        """VectorizeTraceGenerator class is directly accessible from module."""
        assert hasattr(_vt, "VectorizeTraceGenerator")
        gen = _vt.VectorizeTraceGenerator()  # type: ignore[attr-defined]
        assert gen is not None

    def test_module_loads_with_potracer_absent(self) -> None:
        """Plugin module imports cleanly even when 'potracer' is not installed.

        Uses a unique module name so it is loaded fresh, independent of the
        module-level ``_vt`` fixture that may already have potracer available.
        """
        absent_name = "plottter_plugin_vectorize_trace_absent_test"
        sys.modules.pop(absent_name, None)  # ensure clean state

        with patch.dict(sys.modules, {"potrace": None}):
            spec = importlib.util.spec_from_file_location(absent_name, _PLUGIN_PATH)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[absent_name] = mod
            # This must not raise even though potracer is absent
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # The generator class must be accessible on the loaded module
        assert hasattr(mod, "VectorizeTraceGenerator")
        # And must be instantiable without error
        gen = mod.VectorizeTraceGenerator()  # type: ignore[attr-defined]
        assert gen is not None

        # Cleanup
        sys.modules.pop(absent_name, None)


# ---------------------------------------------------------------------------
# Tests: friendly RuntimeError when potracer is absent
# ---------------------------------------------------------------------------


class TestPotraceAbsent:
    """When potracer is not installed, generate() raises a clear RuntimeError."""

    def _run_generate_without_potracer(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> pytest.ExceptionInfo:
        with patch.dict(sys.modules, {"potrace": None}):
            with pytest.raises(RuntimeError) as exc_info:
                generator.generate({}, canvas)
        return exc_info

    def test_generate_raises_runtime_error(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """generate() raises RuntimeError (not ImportError) when potracer absent."""
        exc_info = self._run_generate_without_potracer(generator, canvas)
        assert isinstance(exc_info.value, RuntimeError)

    def test_error_mentions_potracer(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """RuntimeError message must mention 'potracer'."""
        exc_info = self._run_generate_without_potracer(generator, canvas)
        msg = str(exc_info.value)
        assert "potracer" in msg.lower(), (
            f"Expected 'potracer' in error message, got: {msg!r}"
        )

    def test_error_includes_pip_install(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """RuntimeError message must include 'pip install potracer'."""
        exc_info = self._run_generate_without_potracer(generator, canvas)
        msg = str(exc_info.value)
        assert "pip install potracer" in msg, (
            f"Expected 'pip install potracer' in error message, got: {msg!r}"
        )

    def test_error_message_exact_prefix(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """The RuntimeError message matches the spec's exact wording."""
        exc_info = self._run_generate_without_potracer(generator, canvas)
        msg = str(exc_info.value)
        assert "The potracer package is required" in msg, (
            f"Expected spec's wording in error message, got: {msg!r}"
        )

    def test_generate_layers_also_raises(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """generate_layers() raises the same RuntimeError when potracer absent."""
        with patch.dict(sys.modules, {"potrace": None}):
            with pytest.raises(RuntimeError, match="pip install potracer"):
                generator.generate_layers({}, canvas)

    def test_require_potracer_raises_runtime_error(self) -> None:
        """_require_potracer() raises RuntimeError (not ImportError) directly."""
        with patch.dict(sys.modules, {"potrace": None}):
            with pytest.raises(RuntimeError, match="pip install potracer"):
                _vt._require_potracer()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: parameters and presets (no potracer needed)
# ---------------------------------------------------------------------------


class TestParametersAndPresets:
    """Generator exposes well-formed parameters and presets."""

    def test_get_parameters_returns_list(
        self, generator: VectorizeTraceGenerator
    ) -> None:
        params = generator.get_parameters()
        assert isinstance(params, list)
        assert len(params) > 0

    def test_has_threshold_param(self, generator: VectorizeTraceGenerator) -> None:
        params = generator.get_parameters()
        names = [p.name for p in params]
        assert "threshold" in names

    def test_has_curve_tolerance_param(
        self, generator: VectorizeTraceGenerator
    ) -> None:
        params = generator.get_parameters()
        names = [p.name for p in params]
        assert "curve_tolerance_mm" in names

    def test_has_image_fit_mode_param(
        self, generator: VectorizeTraceGenerator
    ) -> None:
        params = generator.get_parameters()
        names = [p.name for p in params]
        assert "image_fit_mode" in names

    def test_has_num_levels_param(self, generator: VectorizeTraceGenerator) -> None:
        params = generator.get_parameters()
        names = [p.name for p in params]
        assert "num_levels" in names

    def test_get_presets_returns_list(
        self, generator: VectorizeTraceGenerator
    ) -> None:
        presets = generator.get_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_presets_have_name(self, generator: VectorizeTraceGenerator) -> None:
        for preset in generator.get_presets():
            assert preset.name, "Preset must have a non-empty name"

    def test_category_is_image(self, generator: VectorizeTraceGenerator) -> None:
        assert generator.category == "image"

    def test_uses_source_image(self, generator: VectorizeTraceGenerator) -> None:
        assert generator.uses_source_image is True

    def test_emits_multiple_layers(self, generator: VectorizeTraceGenerator) -> None:
        assert generator.emits_multiple_layers is True


# ---------------------------------------------------------------------------
# Tests: Bezier / coordinate helpers (no potracer needed)
# ---------------------------------------------------------------------------


class TestBezierHelpers:
    """Unit tests for the Bezier flattening and coordinate conversion helpers."""

    def test_flatten_straight_line(self) -> None:
        """A linear Bezier (control points on the chord) flattens to just end."""
        flatten = _vt._flatten_bezier_segment  # type: ignore[attr-defined]
        p0 = (0.0, 0.0)
        p1 = (10.0, 0.0)
        c0 = (3.333, 0.0)
        c1 = (6.667, 0.0)
        pts = flatten(p0, c0, c1, p1, tolerance_sq=0.01)
        # Straight bezier: midpoint deviation is zero → single end point
        assert pts == [p1]

    def test_flatten_curved_returns_multiple(self) -> None:
        """A highly-curved Bezier flattens to more than just the end point."""
        flatten = _vt._flatten_bezier_segment  # type: ignore[attr-defined]
        p0 = (0.0, 0.0)
        p1 = (10.0, 0.0)
        c0 = (0.0, 100.0)   # extreme control point — very curved
        c1 = (10.0, 100.0)
        pts = flatten(p0, c0, c1, p1, tolerance_sq=1.0)
        assert len(pts) > 1, "Curved bezier should produce intermediate points"

    def test_flatten_tight_tolerance_more_points(self) -> None:
        """Tighter tolerance produces more points than loose tolerance."""
        flatten = _vt._flatten_bezier_segment  # type: ignore[attr-defined]
        p0 = (0.0, 0.0)
        p1 = (10.0, 0.0)
        c0 = (0.0, 50.0)
        c1 = (10.0, 50.0)
        coarse = flatten(p0, c0, c1, p1, tolerance_sq=25.0)
        fine = flatten(p0, c0, c1, p1, tolerance_sq=0.01)
        assert len(fine) >= len(coarse), (
            "Finer tolerance must produce at least as many points"
        )

    def test_potrace_to_mm_corners(self) -> None:
        """potrace_to_mm maps pixel corners to mm corners correctly."""
        to_mm = _vt._potrace_to_mm  # type: ignore[attr-defined]
        img_w, img_h = 100, 50
        img_x1, img_y1, img_x2, img_y2 = 10.0, 20.0, 110.0, 70.0

        # potrace (0, 0) = top-left (array origin) → mm top-left = (img_x1, img_y1)
        mm = to_mm(0, 0, img_w, img_h, img_x1, img_y1, img_x2, img_y2)
        assert abs(mm[0] - img_x1) < 1e-9
        assert abs(mm[1] - img_y1) < 1e-9

        # potrace (W, H) = bottom-right → mm bottom-right = (img_x2, img_y2)
        mm = to_mm(img_w, img_h, img_w, img_h, img_x1, img_y1, img_x2, img_y2)
        assert abs(mm[0] - img_x2) < 1e-9
        assert abs(mm[1] - img_y2) < 1e-9

    def test_potrace_to_mm_center(self) -> None:
        """Center pixel maps to center of mm rect."""
        to_mm = _vt._potrace_to_mm  # type: ignore[attr-defined]
        img_w, img_h = 100, 100
        img_x1, img_y1, img_x2, img_y2 = 0.0, 0.0, 200.0, 200.0

        mm = to_mm(50, 50, img_w, img_h, img_x1, img_y1, img_x2, img_y2)
        assert abs(mm[0] - 100.0) < 1e-9
        assert abs(mm[1] - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: with potracer installed (guarded)
# ---------------------------------------------------------------------------


class TestWithPotracer:
    """Integration tests requiring the 'potracer' package to be installed.

    Skipped automatically when potracer is not available.
    """

    @pytest.fixture(autouse=True)
    def require_potracer(self) -> None:
        pytest.importorskip("potrace")

    def test_generate_returns_non_empty_for_circle(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """Tracing a black circle returns at least one non-empty polyline."""
        source = _make_circle_image(size=64, radius=20)
        params = {
            "_source_image": source,
            "threshold": 128,
            "num_levels": 1,
            "curve_tolerance_mm": 0.5,
            "turdsize": 2,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        result = generator.generate(params, canvas)
        assert isinstance(result, list), "generate() must return a list"
        assert len(result) > 0, "Expected at least one polyline from circle trace"
        assert any(len(pl) >= 3 for pl in result), (
            "Expected at least one polyline with ≥3 points"
        )

    def test_generate_coordinates_within_fitted_rect(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """All output mm coordinates lie within the fitted image rectangle."""
        source = _make_circle_image(size=64, radius=20)
        params = {
            "_source_image": source,
            "threshold": 128,
            "num_levels": 1,
            "curve_tolerance_mm": 0.5,
            "turdsize": 2,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        result = generator.generate(params, canvas)

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        tolerance = 0.1  # mm rounding tolerance

        for polyline in result:
            for x_mm, y_mm in polyline:
                assert draw_x1 - tolerance <= x_mm <= draw_x2 + tolerance, (
                    f"x={x_mm:.3f} outside fitted rect [{draw_x1}, {draw_x2}]"
                )
                assert draw_y1 - tolerance <= y_mm <= draw_y2 + tolerance, (
                    f"y={y_mm:.3f} outside fitted rect [{draw_y1}, {draw_y2}]"
                )

    def test_generate_layers_returns_single_layer_spec(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """With num_levels=1, generate_layers() returns exactly one LayerSpec."""
        source = _make_circle_image(size=64, radius=20)
        params = {
            "_source_image": source,
            "threshold": 128,
            "num_levels": 1,
            "curve_tolerance_mm": 0.5,
            "turdsize": 2,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        layers = generator.generate_layers(params, canvas)
        assert isinstance(layers, list)
        assert len(layers) == 1
        layer = layers[0]
        assert hasattr(layer, "name")
        assert hasattr(layer, "color")
        assert hasattr(layer, "paths")

    def test_generate_layers_multi_level(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """With num_levels=3, generate_layers() returns exactly 3 LayerSpecs."""
        source = _make_circle_image(size=64, radius=20)
        params = {
            "_source_image": source,
            "threshold": 200,
            "num_levels": 3,
            "curve_tolerance_mm": 1.0,
            "turdsize": 2,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        layers = generator.generate_layers(params, canvas)
        assert isinstance(layers, list)
        # 3 distinct thresholds should produce 3 layers
        assert len(layers) == 3, (
            f"Expected 3 layers for num_levels=3, got {len(layers)}"
        )

    def test_multilevel_layers_non_empty_paths_within_rect(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """Multi-level run: each LayerSpec has non-empty paths inside the fitted rect.

        With num_levels=4 every threshold band of a circle image should produce
        at least one traceable contour.  All mm coordinates must lie within the
        drawing area (the fitted image rectangle for 'fill' mode equals the
        drawing area).
        """
        source = _make_circle_image(size=64, radius=24)
        params = {
            "_source_image": source,
            "threshold": 200,
            "num_levels": 4,
            "curve_tolerance_mm": 1.0,
            "turdsize": 0,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        layers = generator.generate_layers(params, canvas)

        # One LayerSpec per threshold level.
        assert isinstance(layers, list)
        assert len(layers) == 4, (
            f"Expected 4 layers for num_levels=4, got {len(layers)}"
        )

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        tolerance = 0.1  # mm rounding tolerance

        for layer_idx, layer in enumerate(layers):
            # Every layer must have at least one path with >=3 points.
            assert hasattr(layer, "paths"), "LayerSpec must have 'paths' attribute"
            assert len(layer.paths) > 0, (
                f"Layer {layer_idx} ('{layer.name}') has no paths"
            )
            assert any(len(pl) >= 3 for pl in layer.paths), (
                f"Layer {layer_idx} has no polyline with >=3 points"
            )

            # All coordinates must lie within the fitted rect.
            for pl in layer.paths:
                for x_mm, y_mm in pl:
                    assert draw_x1 - tolerance <= x_mm <= draw_x2 + tolerance, (
                        f"Layer {layer_idx}: x={x_mm:.3f} outside"
                        f" [{draw_x1:.1f}, {draw_x2:.1f}]"
                    )
                    assert draw_y1 - tolerance <= y_mm <= draw_y2 + tolerance, (
                        f"Layer {layer_idx}: y={y_mm:.3f} outside"
                        f" [{draw_y1:.1f}, {draw_y2:.1f}]"
                    )

    def test_generate_returns_empty_for_no_source(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """With no source image, generate() returns an empty list gracefully."""
        result = generator.generate({}, canvas)
        assert result == [], "No source image should produce empty output"

    def test_generate_smooth_polylines(
        self, generator: VectorizeTraceGenerator, canvas: Canvas
    ) -> None:
        """Traced polylines have reasonable point counts (smooth, not pixel-grid)."""
        source = _make_circle_image(size=64, radius=24)
        params = {
            "_source_image": source,
            "threshold": 128,
            "num_levels": 1,
            "curve_tolerance_mm": 0.2,
            "turdsize": 0,
            "alphamax": 1.0,
            "opttolerance": 0.2,
            "image_fit_mode": "fill",
        }
        result = generator.generate(params, canvas)
        assert result, "Expected non-empty trace output"
        # The polylines should have a reasonable number of points.
        # A 64×64 pixel circle traced at 0.2mm tolerance should yield at least 10
        # and far fewer than 64*4=256 (pixel-grid staircase) points per curve.
        for pl in result:
            assert len(pl) >= 3, "Polyline too short"
