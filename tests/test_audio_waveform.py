"""Tests for AudioWaveformGenerator."""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from plottter.generators import GENERATORS
from plottter.generators.audio_waveform import AudioWaveformGenerator
from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine_wav(path: str, duration_sec: float = 2.0, freq: float = 440.0, sample_rate: int = 44100) -> None:
    """Write a mono 16-bit sine WAV to *path*."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory):
    """Provide a temporary 2-second 440 Hz mono sine WAV file."""
    tmp = tmp_path_factory.mktemp("audio")
    wav_path = str(tmp / "sine.wav")
    _make_sine_wav(wav_path)
    return wav_path


@pytest.fixture
def canvas() -> Canvas:
    return Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)


@pytest.fixture
def generator() -> AudioWaveformGenerator:
    return AudioWaveformGenerator()


# ---------------------------------------------------------------------------
# (a) Registry
# ---------------------------------------------------------------------------

def test_registered_in_generators():
    assert "Audio Waveform" in GENERATORS


# ---------------------------------------------------------------------------
# (b) Parameters
# ---------------------------------------------------------------------------

def test_get_parameters_non_empty(generator):
    params = generator.get_parameters()
    assert len(params) > 0
    names = {p.name for p in params}
    assert "audio_file" in names
    assert "mode" in names
    assert "num_rows" in names


# ---------------------------------------------------------------------------
# (c) Presets
# ---------------------------------------------------------------------------

def test_get_presets_returns_at_least_four(generator):
    presets = generator.get_presets()
    assert len(presets) >= 4


# ---------------------------------------------------------------------------
# (d) No file → empty
# ---------------------------------------------------------------------------

def test_generate_no_file_returns_empty(generator, canvas):
    result = generator.generate({"mode": "Ridgeline", "audio_file": ""}, canvas)
    assert result == []


# ---------------------------------------------------------------------------
# (e) Non-existent file → empty
# ---------------------------------------------------------------------------

def test_generate_nonexistent_file_returns_empty(generator, canvas):
    result = generator.generate(
        {"mode": "Ridgeline", "audio_file": "/nonexistent/path/file.wav"},
        canvas,
    )
    assert result == []


# ---------------------------------------------------------------------------
# (f) Ridgeline mode with default params produces output
# ---------------------------------------------------------------------------

def test_ridgeline_default_params_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Ridgeline",
            "audio_file": sine_wav,
            "start_sec": 0.0,
            "duration_sec": 2.0,
        },
        canvas,
    )
    assert len(result) > 0
    # Each item is a list of (x, y) tuples
    for pl in result:
        assert len(pl) >= 2
        for pt in pl:
            assert len(pt) == 2


# ---------------------------------------------------------------------------
# (g) All Ridgeline presets produce non-empty output
# ---------------------------------------------------------------------------

def test_ridgeline_presets_produce_output(generator, canvas, sine_wav):
    presets = generator.get_presets()
    for preset in presets:
        if preset.params.get("mode", "Ridgeline") != "Ridgeline":
            continue
        p = dict(preset.params)
        p["audio_file"] = sine_wav
        p.setdefault("mode", "Ridgeline")
        p.setdefault("duration_sec", 2.0)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# (h) Output polylines are within canvas bounds (+1 mm tolerance)
# ---------------------------------------------------------------------------

def test_output_within_canvas_bounds(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Ridgeline",
            "audio_file": sine_wav,
            "start_sec": 0.0,
            "duration_sec": 2.0,
        },
        canvas,
    )
    assert len(result) > 0
    left, top, right, bottom = canvas.drawing_area()
    tol = 1.0  # 1 mm tolerance
    for pl in result:
        for x, y in pl:
            assert left - tol <= x <= right + tol, f"x={x} out of bounds [{left}, {right}]"
            assert top - tol <= y <= bottom + tol, f"y={y} out of bounds [{top}, {bottom}]"


# ---------------------------------------------------------------------------
# (i) hlr_enabled=False produces output
# ---------------------------------------------------------------------------

def test_hlr_disabled_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Ridgeline",
            "audio_file": sine_wav,
            "hlr_enabled": False,
            "duration_sec": 2.0,
        },
        canvas,
    )
    assert len(result) > 0


# ---------------------------------------------------------------------------
# (j) mirror=True produces output
# ---------------------------------------------------------------------------

def test_mirror_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Ridgeline",
            "audio_file": sine_wav,
            "mirror": True,
            "duration_sec": 2.0,
        },
        canvas,
    )
    assert len(result) > 0


# ---------------------------------------------------------------------------
# (k) Cancellation returns empty list
# ---------------------------------------------------------------------------

def test_cancellation_returns_empty(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Ridgeline",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
        },
        canvas,
        cancelled_callback=lambda: True,
    )
    assert result == []


# ---------------------------------------------------------------------------
# Circular mode tests
# ---------------------------------------------------------------------------

# (a) Circular mode produces a single polyline with ~circle_points (+1 if closed) points

def test_circular_produces_single_polyline(generator, canvas, sine_wav):
    circle_points = 720
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_points": circle_points,
            "circle_closed": True,
        },
        canvas,
    )
    assert len(result) == 1
    # closed: circle_points + 1 points
    assert len(result[0]) == circle_points + 1


def test_circular_open_produces_exact_points(generator, canvas, sine_wav):
    circle_points = 720
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_points": circle_points,
            "circle_closed": False,
        },
        canvas,
    )
    assert len(result) == 1
    assert len(result[0]) == circle_points


# (b) Each circle_source option produces output

@pytest.mark.parametrize("source", ["Waveform", "Envelope", "Spectrum"])
def test_circular_all_sources_produce_output(generator, canvas, sine_wav, source):
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_source": source,
            "circle_points": 360,
        },
        canvas,
    )
    assert len(result) == 1
    assert len(result[0]) > 0


# (c) circle_closed=True: first and last point within 0.01mm

def test_circular_closed_endpoints_match(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_points": 360,
            "circle_closed": True,
        },
        canvas,
    )
    assert len(result) == 1
    poly = result[0]
    x0, y0 = poly[0]
    xn, yn = poly[-1]
    dist = ((x0 - xn) ** 2 + (y0 - yn) ** 2) ** 0.5
    assert dist < 0.01, f"First/last points not close enough: dist={dist}"


# (d) circle_closed=False: first and last point differ

def test_circular_open_endpoints_differ(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_points": 360,
            "circle_closed": False,
        },
        canvas,
    )
    assert len(result) == 1
    poly = result[0]
    x0, y0 = poly[0]
    xn, yn = poly[-1]
    dist = ((x0 - xn) ** 2 + (y0 - yn) ** 2) ** 0.5
    assert dist > 0.01, f"Open loop: first/last points should differ: dist={dist}"


# (e) Circular output is within canvas bounds

def test_circular_output_within_canvas_bounds(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Circular",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "circle_points": 360,
        },
        canvas,
    )
    assert len(result) > 0
    left, top, right, bottom = canvas.drawing_area()
    tol = 1.0  # 1 mm tolerance
    for pl in result:
        for x, y in pl:
            assert left - tol <= x <= right + tol, f"x={x} out of bounds [{left}, {right}]"
            assert top - tol <= y <= bottom + tol, f"y={y} out of bounds [{top}, {bottom}]"


# (f) All Circular presets produce non-empty output

def test_circular_presets_produce_output(generator, canvas, sine_wav):
    presets = generator.get_presets()
    circular_presets = [p for p in presets if p.params.get("mode") == "Circular"]
    assert len(circular_presets) >= 3, "Expected at least 3 Circular presets"
    for preset in circular_presets:
        p = dict(preset.params)
        p["audio_file"] = sine_wav
        p.setdefault("duration_sec", 2.0)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Circular preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# Spiral mode tests
# ---------------------------------------------------------------------------

# (a) Spiral mode produces a single polyline with approximately spiral_points points

def test_spiral_produces_single_polyline(generator, canvas, sine_wav):
    spiral_points = 2000
    result = generator.generate(
        {
            "mode": "Spiral",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "spiral_points": spiral_points,
        },
        canvas,
    )
    assert len(result) == 1
    assert len(result[0]) == spiral_points


# (b) Outward direction: radius generally increases (first 10% vs last 10%)

def test_spiral_outward_radius_increases(generator, canvas, sine_wav):
    spiral_points = 2000
    result = generator.generate(
        {
            "mode": "Spiral",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "spiral_points": spiral_points,
            "spiral_direction": "Outward",
            "spiral_smoothing": 0.0,
        },
        canvas,
    )
    assert len(result) == 1
    poly = result[0]
    left, top, right, bottom = canvas.drawing_area()
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    n = len(poly)
    seg = n // 10
    first_radii = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in poly[:seg]]
    last_radii = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in poly[-seg:]]
    assert np.mean(last_radii) > np.mean(first_radii), (
        f"Outward: mean radius should increase; first={np.mean(first_radii):.2f}, last={np.mean(last_radii):.2f}"
    )


# (c) Inward direction: radius generally decreases

def test_spiral_inward_radius_decreases(generator, canvas, sine_wav):
    spiral_points = 2000
    result = generator.generate(
        {
            "mode": "Spiral",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "spiral_points": spiral_points,
            "spiral_direction": "Inward",
            "spiral_smoothing": 0.0,
        },
        canvas,
    )
    assert len(result) == 1
    poly = result[0]
    left, top, right, bottom = canvas.drawing_area()
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    n = len(poly)
    seg = n // 10
    first_radii = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in poly[:seg]]
    last_radii = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in poly[-seg:]]
    assert np.mean(last_radii) < np.mean(first_radii), (
        f"Inward: mean radius should decrease; first={np.mean(first_radii):.2f}, last={np.mean(last_radii):.2f}"
    )


# (d) Spiral output within canvas bounds

def test_spiral_output_within_canvas_bounds(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Spiral",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "spiral_points": 1000,
        },
        canvas,
    )
    assert len(result) > 0
    left, top, right, bottom = canvas.drawing_area()
    tol = 1.0  # 1 mm tolerance
    for pl in result:
        for x, y in pl:
            assert left - tol <= x <= right + tol, f"x={x} out of bounds [{left}, {right}]"
            assert top - tol <= y <= bottom + tol, f"y={y} out of bounds [{top}, {bottom}]"


# (e) All Spiral presets produce non-empty output

def test_spiral_presets_produce_output(generator, canvas, sine_wav):
    presets = generator.get_presets()
    spiral_presets = [p for p in presets if p.params.get("mode") == "Spiral"]
    assert len(spiral_presets) >= 2, "Expected at least 2 Spiral presets"
    for preset in spiral_presets:
        p = dict(preset.params)
        p["audio_file"] = sine_wav
        p.setdefault("duration_sec", 2.0)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Spiral preset '{preset.name}' produced no output"
