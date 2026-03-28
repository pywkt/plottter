"""Tests for hatching generator oscillation parameters (task 70.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.hatching import HatchingGenerator
from plottter.models.canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def make_dark_image(h: int = 50, w: int = 50) -> np.ndarray:
    """Solid dark gray image."""
    return np.full((h, w), 50, dtype=np.uint8)


def run_generator(params: dict, canvas: Canvas | None = None) -> list:
    if canvas is None:
        canvas = make_canvas()
    gen = HatchingGenerator()
    base = {
        "mode": "parallel",
        "angle_deg": 0.0,
        "angle2_deg": 90.0,
        "min_spacing_mm": 2.0,
        "max_spacing_mm": 4.0,
        "density_curve": "linear",
        "line_length_mm": 0.0,
        "invert": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "oscillation": False,
        "osc_amplitude": 1.0,
        "osc_wavelength_mm": 2.0,
        "osc_mode": "Sine",
        "_source_image": make_dark_image(),
    }
    base.update(params)
    return gen.generate(base, canvas)


# ---------------------------------------------------------------------------
# Parameter presence tests
# ---------------------------------------------------------------------------


def test_oscillation_param_exists():
    gen = HatchingGenerator()
    param_names = {p.name for p in gen.get_parameters()}
    assert "oscillation" in param_names
    assert "osc_amplitude" in param_names
    assert "osc_wavelength_mm" in param_names
    assert "osc_mode" in param_names


def test_oscillation_param_defaults():
    gen = HatchingGenerator()
    params = {p.name: p for p in gen.get_parameters()}

    osc = params["oscillation"]
    assert osc.default is False

    amp = params["osc_amplitude"]
    assert amp.default == 1.0
    assert amp.min == 0.1
    assert amp.max == 5.0

    wl = params["osc_wavelength_mm"]
    assert wl.default == 2.0
    assert wl.min == 0.5
    assert wl.max == 10.0

    mode = params["osc_mode"]
    assert "Sine" in mode.choices
    assert "Sawtooth" in mode.choices
    assert mode.default == "Sine"


def test_oscillation_sub_params_have_visible_when():
    gen = HatchingGenerator()
    params = {p.name: p for p in gen.get_parameters()}
    for name in ("osc_amplitude", "osc_wavelength_mm", "osc_mode"):
        assert params[name].visible_when is not None, f"{name} should have visible_when"
        assert "oscillation" in params[name].visible_when
        assert True in params[name].visible_when["oscillation"]


# ---------------------------------------------------------------------------
# Regression: oscillation=False produces same output as without the param
# ---------------------------------------------------------------------------


def test_oscillation_false_no_change():
    """oscillation=False must not alter the generated paths."""
    canvas = make_canvas()
    result_no_osc = run_generator({"oscillation": False}, canvas)
    result_default = run_generator({}, canvas)
    # Both should produce the same polylines
    assert len(result_no_osc) == len(result_default)
    for poly_a, poly_b in zip(result_no_osc, result_default):
        assert len(poly_a) == len(poly_b)
        for (x1, y1), (x2, y2) in zip(poly_a, poly_b):
            assert abs(x1 - x2) < 1e-9
            assert abs(y1 - y2) < 1e-9


# ---------------------------------------------------------------------------
# Oscillation=True changes points perpendicular to hatch direction
# ---------------------------------------------------------------------------


def test_oscillation_true_modifies_output():
    """With oscillation=True the points should differ from oscillation=False."""
    canvas = make_canvas()
    result_off = run_generator({"oscillation": False}, canvas)
    # Use wavelength=1.5mm so sample points (step=1mm) don't all land on sine zero-crossings
    result_on = run_generator({"oscillation": True, "osc_amplitude": 2.0, "osc_wavelength_mm": 1.5}, canvas)

    assert len(result_off) > 0, "Expected some hatch lines"
    assert len(result_on) > 0

    # At least some points should differ
    any_diff = False
    for poly_off, poly_on in zip(result_off, result_on):
        for (x1, y1), (x2, y2) in zip(poly_off, poly_on):
            if abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6:
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, "Oscillation=True should displace at least some points"


def test_oscillation_sawtooth_differs_from_sine():
    """Sine and Sawtooth waveforms should produce different paths."""
    canvas = make_canvas()
    # Use wavelength=1.5mm so sample points (step=1mm) don't all land on sine zero-crossings
    sine_result = run_generator({"oscillation": True, "osc_mode": "Sine", "osc_amplitude": 2.0, "osc_wavelength_mm": 1.5}, canvas)
    saw_result = run_generator({"oscillation": True, "osc_mode": "Sawtooth", "osc_amplitude": 2.0, "osc_wavelength_mm": 1.5}, canvas)

    assert len(sine_result) > 0
    assert len(saw_result) > 0

    any_diff = False
    for poly_s, poly_w in zip(sine_result, saw_result):
        for (x1, y1), (x2, y2) in zip(poly_s, poly_w):
            if abs(x1 - x2) > 1e-6 or abs(y1 - y2) > 1e-6:
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, "Sine and Sawtooth should produce different displacements"


def test_cross_mode_oscillation():
    """Oscillation should work in cross hatch mode without errors."""
    result = run_generator({"mode": "cross", "oscillation": True, "osc_amplitude": 1.0})
    assert isinstance(result, list)
    assert len(result) > 0
