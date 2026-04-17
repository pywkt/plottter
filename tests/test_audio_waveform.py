"""Tests for AudioWaveformGenerator."""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

from plottter.generators import GENERATORS
from plottter.generators.audio_utils import extract_contours, find_interesting_segment
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


def _make_stereo_sine_wav(
    path: str,
    duration_sec: float = 2.0,
    freq_left: float = 440.0,
    freq_right: float = 660.0,
    sample_rate: int = 44100,
) -> None:
    """Write a stereo 16-bit WAV with different sine frequencies per channel."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    left = (np.sin(2 * np.pi * freq_left * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * freq_right * t) * 32767).astype(np.int16)
    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


def _make_varying_stereo_wav(path: str, sample_rate: int = 44100) -> None:
    """Stereo WAV where the left/right frequency ratio changes mid-file.

    First half:  left=440 Hz, right=440 Hz  → Lissajous is a line (1:1)
    Second half: left=440 Hz, right=880 Hz  → Lissajous is a figure-8 (1:2)
    """
    duration = 2.0
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    left_f = np.sin(2 * np.pi * 440.0 * t)
    right_f = np.where(t < 1.0, np.sin(2 * np.pi * 440.0 * t), np.sin(2 * np.pi * 880.0 * t))
    left = (left_f * 32767).astype(np.int16)
    right = (right_f * 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory):
    """Provide a temporary 2-second 440 Hz mono sine WAV file."""
    tmp = tmp_path_factory.mktemp("audio")
    wav_path = str(tmp / "sine.wav")
    _make_sine_wav(wav_path)
    return wav_path


@pytest.fixture(scope="module")
def stereo_wav(tmp_path_factory):
    """Provide a temporary 2-second stereo WAV file (440 Hz left, 660 Hz right)."""
    tmp = tmp_path_factory.mktemp("audio_stereo")
    wav_path = str(tmp / "stereo.wav")
    _make_stereo_sine_wav(wav_path)
    return wav_path


@pytest.fixture(scope="module")
def varying_stereo_wav(tmp_path_factory):
    """Stereo WAV where freq ratio changes at 1s (1:1 → 1:2), giving different Lissajous shapes."""
    tmp = tmp_path_factory.mktemp("audio_varying")
    wav_path = str(tmp / "varying.wav")
    _make_varying_stereo_wav(wav_path)
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


# ---------------------------------------------------------------------------
# Contour mode — unit tests for extract_contours
# ---------------------------------------------------------------------------

# (a) Gaussian bump array produces multiple polylines

def test_extract_contours_gaussian_bump():
    data = np.zeros((100, 100))
    data[50, 50] = 1.0
    data = scipy.ndimage.gaussian_filter(data, sigma=10.0)
    contours = extract_contours(data, num_levels=5, smoothing_sigma=0.0)
    assert len(contours) > 0
    for pl in contours:
        assert len(pl) >= 2
        for pt in pl:
            assert len(pt) == 2


# (b) Uniform array produces no contours

def test_extract_contours_uniform_returns_empty():
    data = np.ones((50, 50))
    contours = extract_contours(data, num_levels=5, smoothing_sigma=0.0)
    assert len(contours) == 0


# ---------------------------------------------------------------------------
# Contour mode — integration tests
# ---------------------------------------------------------------------------

# (c) Contour mode with test WAV produces multiple polylines

def test_contour_mode_produces_multiple_polylines(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Contour",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "contour_levels": 8,
            "contour_smoothing": 1.5,
            "contour_min_length": 0.0,
        },
        canvas,
    )
    assert len(result) > 1
    for pl in result:
        assert len(pl) >= 2


# (d) contour_min_length filtering: with large min_length, fewer paths than with 0

def test_contour_min_length_filters_short_polylines(generator, canvas, sine_wav):
    params_base = {
        "mode": "Contour",
        "audio_file": sine_wav,
        "duration_sec": 2.0,
        "contour_levels": 10,
        "contour_smoothing": 1.0,
    }

    result_no_filter = generator.generate({**params_base, "contour_min_length": 0.0}, canvas)
    result_filtered = generator.generate({**params_base, "contour_min_length": 50.0}, canvas)

    assert len(result_filtered) < len(result_no_filter), (
        f"Expected fewer polylines with min_length=50mm ({len(result_filtered)}) "
        f"than with min_length=0 ({len(result_no_filter)})"
    )


# (e) Contour output is within canvas bounds

def test_contour_output_within_canvas_bounds(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Contour",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "contour_levels": 6,
            "contour_smoothing": 1.5,
            "contour_min_length": 0.0,
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


# (f) All Contour presets produce non-empty output

def test_contour_presets_produce_output(generator, canvas, sine_wav):
    presets = generator.get_presets()
    contour_presets = [p for p in presets if p.params.get("mode") == "Contour"]
    assert len(contour_presets) >= 3, "Expected at least 3 Contour presets"
    for preset in contour_presets:
        p = dict(preset.params)
        p["audio_file"] = sine_wav
        p.setdefault("duration_sec", 2.0)
        p.setdefault("contour_min_length", 0.0)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Contour preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# Frequency Bands mode tests
# ---------------------------------------------------------------------------

# (a) Frequency Bands with 3 bands produces >= 3 polylines

def test_frequency_bands_3_bands_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Frequency Bands",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "band_count": "3 (Bass/Mid/Treble)",
            "band_style": "Stacked Waveforms",
            "band_points": 500,
        },
        canvas,
    )
    assert len(result) >= 3, f"Expected >= 3 polylines for 3-band mode, got {len(result)}"


# (b) Frequency Bands with 5 bands produces >= 5 polylines

def test_frequency_bands_5_bands_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Frequency Bands",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "band_count": "5",
            "band_style": "Stacked Waveforms",
            "band_points": 500,
        },
        canvas,
    )
    assert len(result) >= 5, f"Expected >= 5 polylines for 5-band mode, got {len(result)}"


# (c) Each band_style option produces output

@pytest.mark.parametrize("style", ["Stacked Waveforms", "Stacked Envelopes", "Side by Side"])
def test_frequency_bands_all_styles_produce_output(generator, canvas, sine_wav, style):
    result = generator.generate(
        {
            "mode": "Frequency Bands",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "band_count": "3 (Bass/Mid/Treble)",
            "band_style": style,
            "band_points": 500,
        },
        canvas,
    )
    assert len(result) > 0, f"Style '{style}' produced no output"
    for pl in result:
        assert len(pl) >= 2


# (d) Frequency Bands output within canvas bounds

def test_frequency_bands_output_within_canvas_bounds(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Frequency Bands",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "band_count": "3 (Bass/Mid/Treble)",
            "band_style": "Stacked Waveforms",
            "band_points": 500,
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


# (e) All Frequency Bands presets produce non-empty output

def test_frequency_bands_presets_produce_output(generator, canvas, sine_wav):
    presets = generator.get_presets()
    fb_presets = [p for p in presets if p.params.get("mode") == "Frequency Bands"]
    assert len(fb_presets) >= 3, "Expected at least 3 Frequency Bands presets"
    for preset in fb_presets:
        p = dict(preset.params)
        p["audio_file"] = sine_wav
        p.setdefault("duration_sec", 2.0)
        p.setdefault("band_points", 500)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Frequency Bands preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# find_interesting_segment unit tests
# ---------------------------------------------------------------------------

# (a) find_interesting_segment returns (start, end) with 0 <= start < end <= len(data)

def test_find_interesting_segment_bounds():
    sample_rate = 44100
    data = np.random.randn(sample_rate * 2, 2).astype(np.float64)  # 2s stereo
    start, end = find_interesting_segment(data, sample_rate, segment_duration=0.1)
    assert 0 <= start
    assert start < end
    assert end <= len(data)


# (b) find_interesting_segment selects the loudest segment

def test_find_interesting_segment_picks_loudest():
    sample_rate = 44100
    # 3 seconds mostly silent, loud burst at 2.0–2.1 s
    total = sample_rate * 3
    data = np.zeros((total, 2), dtype=np.float64)
    burst_start = int(2.0 * sample_rate)
    burst_end = int(2.1 * sample_rate)
    data[burst_start:burst_end, :] = 1.0  # loud burst

    start, end = find_interesting_segment(data, sample_rate, segment_duration=0.1, num_candidates=30)
    # The selected segment should overlap the burst region substantially
    assert end > burst_start and start < burst_end, (
        f"Expected segment [{start}, {end}] to overlap burst [{burst_start}, {burst_end}]"
    )


# ---------------------------------------------------------------------------
# Lissajous mode integration tests
# ---------------------------------------------------------------------------

# (c) Lissajous mode with stereo WAV produces a single polyline

def test_lissajous_stereo_produces_single_polyline(generator, canvas, stereo_wav):
    result = generator.generate(
        {
            "mode": "Lissajous",
            "audio_file": stereo_wav,
            "duration_sec": 2.0,
            "liss_segment_sec": 0.05,
            "liss_points": 1000,
            "liss_smoothing": 2.0,
            "liss_auto_segment": True,
        },
        canvas,
    )
    assert len(result) == 1
    assert len(result[0]) == 1000
    for pt in result[0]:
        assert len(pt) == 2


# (d) Lissajous mode with mono WAV still produces output (phase-offset fallback)

def test_lissajous_mono_fallback_produces_output(generator, canvas, sine_wav):
    result = generator.generate(
        {
            "mode": "Lissajous",
            "audio_file": sine_wav,
            "duration_sec": 2.0,
            "liss_segment_sec": 0.05,
            "liss_points": 1000,
            "liss_smoothing": 2.0,
            "liss_auto_segment": True,
        },
        canvas,
    )
    assert len(result) == 1
    assert len(result[0]) > 0


# (e) liss_auto_segment=False with liss_segment_start produces output at a different position
#     Use a time-varying signal where the first half (1:1 ratio) and second half (1:2 ratio)
#     produce genuinely different Lissajous figures.

def test_lissajous_manual_segment_differs_from_auto(generator, canvas, varying_stereo_wav):
    # Manual start at 0.0s → first half (1:1 ratio, line-shaped Lissajous)
    params_early = {
        "mode": "Lissajous",
        "audio_file": varying_stereo_wav,
        "duration_sec": 2.0,
        "liss_segment_sec": 0.05,
        "liss_points": 500,
        "liss_smoothing": 0.0,
        "liss_auto_segment": False,
        "liss_segment_start": 0.0,
    }
    # Manual start at 1.1s → second half (1:2 ratio, figure-8 Lissajous)
    params_late = {
        **params_early,
        "liss_segment_start": 1.1,
    }

    result_early = generator.generate(params_early, canvas)
    result_late = generator.generate(params_late, canvas)

    assert len(result_early) == 1
    assert len(result_late) == 1

    # The two polylines should differ (different Lissajous shapes from different freq ratios)
    pts_early = result_early[0]
    pts_late = result_late[0]
    diffs = [
        abs(pts_early[i][0] - pts_late[i][0]) + abs(pts_early[i][1] - pts_late[i][1])
        for i in range(min(len(pts_early), len(pts_late)))
    ]
    assert max(diffs) > 0.0, "Early and late segments produced identical Lissajous output"


# (f) Lissajous output within canvas bounds

def test_lissajous_output_within_canvas_bounds(generator, canvas, stereo_wav):
    result = generator.generate(
        {
            "mode": "Lissajous",
            "audio_file": stereo_wav,
            "duration_sec": 2.0,
            "liss_segment_sec": 0.05,
            "liss_points": 500,
            "liss_smoothing": 2.0,
            "liss_auto_segment": True,
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


# (g) All Lissajous presets produce non-empty output with stereo input

def test_lissajous_presets_produce_output(generator, canvas, stereo_wav):
    presets = generator.get_presets()
    liss_presets = [p for p in presets if p.params.get("mode") == "Lissajous"]
    assert len(liss_presets) >= 3, "Expected at least 3 Lissajous presets"
    for preset in liss_presets:
        p = dict(preset.params)
        p["audio_file"] = stereo_wav
        p.setdefault("duration_sec", 2.0)
        result = generator.generate(p, canvas)
        assert len(result) > 0, f"Lissajous preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# Helper: chord stereo WAV (220 + 440 + 880 Hz, different amplitudes per channel)
# ---------------------------------------------------------------------------

def _make_chord_stereo_wav(path: str, duration_sec: float = 3.0, sample_rate: int = 44100) -> None:
    """Stereo WAV with a chord (220+440+880 Hz) mixed at different amplitudes per channel."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    # Left channel: emphasise low and high
    left_f = 0.6 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 880.0 * t)
    # Right channel: emphasise mid
    right_f = 0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.8 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t)
    # Normalise to int16
    left = (left_f / np.abs(left_f).max() * 32767).astype(np.int16)
    right = (right_f / np.abs(right_f).max() * 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


@pytest.fixture(scope="module")
def chord_stereo_wav(tmp_path_factory):
    """3-second stereo chord WAV (220+440+880 Hz) for end-to-end integration tests."""
    tmp = tmp_path_factory.mktemp("audio_chord")
    wav_path = str(tmp / "chord.wav")
    _make_chord_stereo_wav(wav_path)
    return wav_path


# ---------------------------------------------------------------------------
# (B) TestAllModesEndToEnd
# ---------------------------------------------------------------------------

ALL_MODES = ["Ridgeline", "Circular", "Spiral", "Contour", "Frequency Bands", "Lissajous"]

_DEFAULT_MODE_PARAMS: dict[str, dict] = {
    "Ridgeline": {"mode": "Ridgeline", "duration_sec": 3.0},
    "Circular": {"mode": "Circular", "duration_sec": 3.0, "circle_points": 360},
    "Spiral": {"mode": "Spiral", "duration_sec": 3.0, "spiral_points": 1000},
    "Contour": {"mode": "Contour", "duration_sec": 3.0, "contour_min_length": 0.0},
    "Frequency Bands": {"mode": "Frequency Bands", "duration_sec": 3.0, "band_points": 300},
    "Lissajous": {"mode": "Lissajous", "duration_sec": 3.0, "liss_points": 500, "liss_auto_segment": True},
}


class TestAllModesEndToEnd:
    """Integration tests: all 6 modes produce non-empty output with a chord stereo WAV."""

    def test_all_modes_produce_output(self, generator, canvas, chord_stereo_wav):
        for mode in ALL_MODES:
            params = dict(_DEFAULT_MODE_PARAMS[mode])
            params["audio_file"] = chord_stereo_wav
            result = generator.generate(params, canvas)
            assert len(result) > 0, f"Mode '{mode}' produced no output"

    def test_all_presets_produce_output(self, generator, canvas, chord_stereo_wav):
        presets = generator.get_presets()
        assert len(presets) > 0, "get_presets() returned no presets"
        for preset in presets:
            p = dict(preset.params)
            p["audio_file"] = chord_stereo_wav
            p.setdefault("duration_sec", 3.0)
            # Ensure min-length filter does not silently discard everything
            p.setdefault("contour_min_length", 0.0)
            result = generator.generate(p, canvas)
            assert len(result) > 0, f"Preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# (C) Progress callback tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Verify progress_callback is invoked correctly for at least 2 modes."""

    def _collect_progress(self, generator, canvas, mode_params, wav_path):
        values = []
        generator.generate(
            {**mode_params, "audio_file": wav_path},
            canvas,
            progress_callback=values.append,
        )
        return values

    def test_ridgeline_progress_callback(self, generator, canvas, sine_wav):
        values = self._collect_progress(
            generator, canvas, {"mode": "Ridgeline", "duration_sec": 2.0}, sine_wav
        )
        assert len(values) >= 3, f"Expected >= 3 progress calls, got {len(values)}"
        for v in values:
            assert 0 <= v <= 100, f"Progress value {v} out of [0, 100]"
        for a, b in zip(values, values[1:]):
            assert a <= b, f"Progress values not non-decreasing: {a} > {b}"

    def test_circular_progress_callback(self, generator, canvas, sine_wav):
        values = self._collect_progress(
            generator, canvas,
            {"mode": "Circular", "duration_sec": 2.0, "circle_points": 360},
            sine_wav,
        )
        assert len(values) >= 3, f"Expected >= 3 progress calls, got {len(values)}"
        for v in values:
            assert 0 <= v <= 100, f"Progress value {v} out of [0, 100]"
        for a, b in zip(values, values[1:]):
            assert a <= b, f"Progress values not non-decreasing: {a} > {b}"


# ---------------------------------------------------------------------------
# (D) Cancellation tests for each mode
# ---------------------------------------------------------------------------

class TestCancellation:
    """Verify that cancelled_callback=lambda: True returns [] for every mode."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_cancel_returns_empty(self, generator, canvas, chord_stereo_wav, mode):
        params = dict(_DEFAULT_MODE_PARAMS[mode])
        params["audio_file"] = chord_stereo_wav
        result = generator.generate(params, canvas, cancelled_callback=lambda: True)
        assert result == [], f"Mode '{mode}' did not return [] when cancelled"


# ---------------------------------------------------------------------------
# (E) Generator metadata verification
# ---------------------------------------------------------------------------

class TestGeneratorMetadata:
    """Verify registry, category, and visible_when parameter references."""

    def test_registry_maps_to_correct_class(self):
        assert GENERATORS["Audio Waveform"] is AudioWaveformGenerator

    def test_category_is_math(self):
        assert AudioWaveformGenerator.category == "math"

    def test_visible_when_keys_reference_valid_params(self):
        gen = AudioWaveformGenerator()
        params = gen.get_parameters()
        param_names = {p.name for p in params}
        for p in params:
            vw = getattr(p, "visible_when", None)
            if vw is None:
                continue
            for key in vw:
                assert key in param_names, (
                    f"Parameter '{p.name}' has visible_when key '{key}' "
                    f"which is not in the parameter list {param_names}"
                )
