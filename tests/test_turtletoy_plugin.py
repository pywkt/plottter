"""Tests for the TurtleToy plugin (phase 126.1).

Covers:
- Star sketch (spec §8.1) produces exactly 5 segments grouped into 1 polyline
- All output coords lie within the canvas drawing area bounds
- quickjs-missing path raises a RuntimeError with clear install instructions
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from plottter.models.canvas import Canvas

# ---------------------------------------------------------------------------
# Load the plugin module directly (it lives in plugins/, not src/).
# We use a fixed module name so sys.modules deduplicates across test runs.
# ---------------------------------------------------------------------------

_PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "turtletoy.py"
_MODULE_NAME = "plottter_plugin_turtletoy"


def _load_plugin() -> Any:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not create module spec for {_PLUGIN_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_tt = _load_plugin()
TurtleToyGenerator = _tt.TurtleToyGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canvas() -> Canvas:
    """Standard A4 canvas used across all tests."""
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def generator() -> TurtleToyGenerator:
    return TurtleToyGenerator()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _star_polylines(canvas: Canvas) -> list[list[tuple[float, float]]]:
    """Run the star sketch and return the resulting polylines."""
    gen = TurtleToyGenerator()
    return gen.generate({"code": _tt._STAR_SKETCH}, canvas)


# ---------------------------------------------------------------------------
# Tests: star sketch geometry
# ---------------------------------------------------------------------------


class TestStarSketch:
    """The canonical 5-pointed star (spec §8.1) must produce well-formed output."""

    def test_produces_exactly_5_segments(self, canvas: Canvas) -> None:
        """Five forward() calls → 5 line segments in total."""
        polylines = _star_polylines(canvas)
        total_segments = sum(len(pl) - 1 for pl in polylines)
        assert total_segments == 5, (
            f"Expected 5 segments, got {total_segments} "
            f"(polyline point counts: {[len(pl) for pl in polylines]})"
        )

    def test_produces_exactly_1_polyline(self, canvas: Canvas) -> None:
        """All 5 star segments are consecutive → must merge into 1 polyline."""
        polylines = _star_polylines(canvas)
        assert len(polylines) == 1, (
            f"Expected 1 polyline after grouping, got {len(polylines)}"
        )

    def test_polyline_has_6_points(self, canvas: Canvas) -> None:
        """5 segments = 6 points (start + 4 intermediate + end ≈ start)."""
        polylines = _star_polylines(canvas)
        assert len(polylines[0]) == 6, (
            f"Expected 6 points in polyline, got {len(polylines[0])}"
        )

    def test_all_coords_within_canvas_bounds(self, canvas: Canvas) -> None:
        """Every (x, y) mm coordinate must lie within the printable area."""
        polylines = _star_polylines(canvas)
        left, top, right, bottom = canvas.drawing_area()

        for polyline in polylines:
            for x_mm, y_mm in polyline:
                assert left <= x_mm <= right, (
                    f"x={x_mm:.3f}mm outside printable range [{left}, {right}]"
                )
                assert top <= y_mm <= bottom, (
                    f"y={y_mm:.3f}mm outside printable range [{top}, {bottom}]"
                )

    def test_star_is_closed(self, canvas: Canvas) -> None:
        """The star's last point should approximately equal its first point."""
        polylines = _star_polylines(canvas)
        pts = polylines[0]
        x0, y0 = pts[0]
        xn, yn = pts[-1]
        assert abs(x0 - xn) < 0.1, f"Star not closed in x: {x0:.4f} vs {xn:.4f}"
        assert abs(y0 - yn) < 0.1, f"Star not closed in y: {y0:.4f} vs {yn:.4f}"


# ---------------------------------------------------------------------------
# Tests: quickjs-missing failure mode (spec §7.2)
# ---------------------------------------------------------------------------


class TestQuickjsMissing:
    """When quickjs is not installed, generate() must raise a clear RuntimeError."""

    def test_generate_raises_runtime_error(self, canvas: Canvas) -> None:
        """generate() raises RuntimeError when _QUICKJS_AVAILABLE is False."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                gen.generate({"code": _tt._STAR_SKETCH}, canvas)

        msg = str(exc_info.value)
        assert "quickjs" in msg.lower(), (
            f"Error message should mention 'quickjs', got: {msg!r}"
        )

    def test_error_includes_install_command(self, canvas: Canvas) -> None:
        """The RuntimeError message includes 'pip install quickjs'."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                gen.generate({"code": _tt._STAR_SKETCH}, canvas)

        assert "pip install" in str(exc_info.value), (
            "Error message should include 'pip install' instructions"
        )

    def test_presets_mention_missing_package(self) -> None:
        """get_presets() returns an entry describing the missing dependency."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            presets = gen.get_presets()

        assert len(presets) == 1
        description_or_name = presets[0].description + presets[0].name
        assert "quickjs" in description_or_name.lower(), (
            f"Preset should mention 'quickjs', got name={presets[0].name!r}, "
            f"description={presets[0].description!r}"
        )
