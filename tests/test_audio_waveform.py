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
