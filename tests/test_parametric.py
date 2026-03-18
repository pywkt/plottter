"""Tests for the ParametricGenerator."""

from __future__ import annotations

import math

import pytest

from plottter.generators.parametric import ParametricGenerator
from plottter.models import Canvas


@pytest.fixture
def canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def generator() -> ParametricGenerator:
    return ParametricGenerator()


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


def test_default_params_generates_paths(generator: ParametricGenerator, canvas: Canvas) -> None:
    params = {
        "x_expr": "sin(3*t)",
        "y_expr": "sin(4*t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": 1000,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result = generator.generate(params, canvas)
    assert len(result) == 1
    assert len(result[0]) > 0


def test_output_within_canvas_bounds(generator: ParametricGenerator, canvas: Canvas) -> None:
    params = {
        "x_expr": "sin(t)",
        "y_expr": "cos(t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": 1000,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result = generator.generate(params, canvas)
    assert result
    x1, y1, x2, y2 = canvas.drawing_area()
    for point in result[0]:
        x, y = point
        assert x1 - 1e-6 <= x <= x2 + 1e-6, f"x={x} out of bounds [{x1}, {x2}]"
        assert y1 - 1e-6 <= y <= y2 + 1e-6, f"y={y} out of bounds [{y1}, {y2}]"


def test_point_count_approximately_matches_num_points(
    generator: ParametricGenerator, canvas: Canvas
) -> None:
    num_points = 2000
    params = {
        "x_expr": "sin(t)",
        "y_expr": "cos(t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": num_points,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result = generator.generate(params, canvas)
    assert result
    # Point count should be close to num_points (within 5%)
    assert abs(len(result[0]) - num_points) / num_points < 0.05


def test_custom_expression_output(generator: ParametricGenerator, canvas: Canvas) -> None:
    """A line along the x-axis (y=0) should produce points with y near canvas center."""
    params = {
        "x_expr": "t",
        "y_expr": "0*t",
        "t_start": -1.0,
        "t_end": 1.0,
        "num_points": 100,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result = generator.generate(params, canvas)
    assert result
    ys = [p[1] for p in result[0]]
    # All y values should be approximately equal (centered on canvas)
    assert max(ys) - min(ys) < 1e-6


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    [
        "Lissajous",
        "Butterfly Curve",
        "Spirograph (Epitrochoid)",
        "Hypotrochoid",
        "Farris / Mystery Curve",
    ],
)
def test_preset_generates_nonempty_paths(
    generator: ParametricGenerator,
    canvas: Canvas,
    preset_name: str,
) -> None:
    presets = {p.name: p.params for p in generator.get_presets()}
    assert preset_name in presets, f"Preset {preset_name!r} not found"
    params = presets[preset_name]
    result = generator.generate(params, canvas)
    assert len(result) >= 1
    assert len(result[0]) > 0


@pytest.mark.parametrize(
    "preset_name",
    [
        "Lissajous",
        "Butterfly Curve",
        "Spirograph (Epitrochoid)",
        "Hypotrochoid",
        "Farris / Mystery Curve",
    ],
)
def test_preset_within_canvas_bounds(
    generator: ParametricGenerator,
    canvas: Canvas,
    preset_name: str,
) -> None:
    presets = {p.name: p.params for p in generator.get_presets()}
    params = presets[preset_name]
    result = generator.generate(params, canvas)
    assert result
    x1, y1, x2, y2 = canvas.drawing_area()
    # Small tolerance for floating-point
    margin = 1.0
    for point in result[0]:
        x, y = point
        assert x1 - margin <= x <= x2 + margin, f"{preset_name}: x={x:.2f} out of [{x1:.2f}, {x2:.2f}]"
        assert y1 - margin <= y <= y2 + margin, f"{preset_name}: y={y:.2f} out of [{y1:.2f}, {y2:.2f}]"


def test_lorenz_attractor_preset(generator: ParametricGenerator, canvas: Canvas) -> None:
    presets = {p.name: p.params for p in generator.get_presets()}
    assert "Lorenz Attractor" in presets
    params = presets["Lorenz Attractor"]
    result = generator.generate(params, canvas)
    assert len(result) == 1
    assert len(result[0]) > 1000


def test_lorenz_within_canvas_bounds(generator: ParametricGenerator, canvas: Canvas) -> None:
    presets = {p.name: p.params for p in generator.get_presets()}
    params = presets["Lorenz Attractor"]
    result = generator.generate(params, canvas)
    assert result
    x1, y1, x2, y2 = canvas.drawing_area()
    margin = 1.0
    for point in result[0]:
        x, y = point
        assert x1 - margin <= x <= x2 + margin
        assert y1 - margin <= y <= y2 + margin


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_rotation_changes_output(generator: ParametricGenerator, canvas: Canvas) -> None:
    base_params = {
        "x_expr": "t",
        "y_expr": "0*t",
        "t_start": -1.0,
        "t_end": 1.0,
        "num_points": 100,
        "scale": 1.0,  # fixed scale so rotation is the only difference
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result_0 = generator.generate(base_params, canvas)[0]

    rotated_params = dict(base_params, rotation_deg=90.0)
    result_90 = generator.generate(rotated_params, canvas)[0]

    # At 90° rotation, what was x variation should now be y variation
    xs_0 = [p[0] for p in result_0]
    xs_90 = [p[0] for p in result_90]
    assert (max(xs_0) - min(xs_0)) > 1e-3
    # After 90° rotation of a horizontal line, x spread should collapse
    assert (max(xs_90) - min(xs_90)) < 1e-3


def test_scale_affects_output_size(generator: ParametricGenerator, canvas: Canvas) -> None:
    params = {
        "x_expr": "sin(t)",
        "y_expr": "cos(t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": 500,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result_auto = generator.generate(params, canvas)[0]

    params_small = dict(params, scale=5.0)
    result_small = generator.generate(params_small, canvas)[0]

    span_auto = max(p[0] for p in result_auto) - min(p[0] for p in result_auto)
    span_small = max(p[0] for p in result_small) - min(p[0] for p in result_small)
    # scale=5 should produce a smaller span than auto-fit (which fills ~90% of canvas)
    assert span_small < span_auto


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_expression_raises(generator: ParametricGenerator, canvas: Canvas) -> None:
    params = {
        "x_expr": "t.real",  # attribute access — not allowed
        "y_expr": "sin(t)",
        "t_start": 0.0,
        "t_end": 1.0,
        "num_points": 100,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    with pytest.raises(ValueError, match="[Ee]xpression"):
        generator.generate(params, canvas)


def test_syntax_error_in_expression_raises(generator: ParametricGenerator, canvas: Canvas) -> None:
    params = {
        "x_expr": "t ++",  # syntax error
        "y_expr": "sin(t)",
        "t_start": 0.0,
        "t_end": 1.0,
        "num_points": 100,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    with pytest.raises(ValueError):
        generator.generate(params, canvas)


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------


def test_parametric_registered_in_registry() -> None:
    from plottter.generators import GENERATORS, get_generators_by_category

    assert "Parametric Curves" in GENERATORS
    math_generators = get_generators_by_category("math")
    assert ParametricGenerator in math_generators


def test_progress_callback_called(generator: ParametricGenerator, canvas: Canvas) -> None:
    progress_values: list[int] = []
    params = {
        "x_expr": "sin(t)",
        "y_expr": "cos(t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": 1000,
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    generator.generate(params, canvas, progress_callback=progress_values.append)
    assert len(progress_values) > 0
    assert progress_values[-1] == 100


def test_cancellation_stops_early(generator: ParametricGenerator, canvas: Canvas) -> None:
    call_count = [0]

    def cancelled() -> bool:
        call_count[0] += 1
        return call_count[0] > 5  # cancel after 5 checks

    params = {
        "x_expr": "sin(t)",
        "y_expr": "cos(t)",
        "t_start": 0.0,
        "t_end": 2 * math.pi,
        "num_points": 50000,  # large, so cancellation matters
        "scale": 0.0,
        "rotation_deg": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    result = generator.generate(params, canvas, cancelled_callback=cancelled)
    # Should return partial result (fewer points than requested)
    total_points = sum(len(p) for p in result)
    assert total_points < 50000
