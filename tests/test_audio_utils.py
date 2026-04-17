"""Tests for plottter.generators.audio_utils."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100  # default sample rate used in fixtures


def _make_sine(freq: float, duration: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    """Return a mono float64 sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _write_wav_int16(path: Path, data: np.ndarray, sr: int = SR) -> None:
    """Write a float64 mono array as int16 WAV."""
    pcm = (data * 32767).astype(np.int16)
    scipy.io.wavfile.write(str(path), sr, pcm)


def _write_wav_stereo_int16(path: Path, data: np.ndarray, sr: int = SR) -> None:
    """Write a float64 mono array as stereo int16 WAV (both channels identical)."""
    pcm_mono = (data * 32767).astype(np.int16)
    stereo = np.column_stack([pcm_mono, pcm_mono])
    scipy.io.wavfile.write(str(path), sr, stereo)


# ---------------------------------------------------------------------------
# (a) load_audio — mono int16 WAV returns float64 array in [-1, 1]
# ---------------------------------------------------------------------------

def test_load_audio_mono_int16_float64_range():
    from plottter.generators.audio_utils import load_audio

    sine = _make_sine(440, 1.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        _write_wav_int16(path, sine)
        sr, data = load_audio(path)
        assert sr == SR
        assert data.dtype == np.float64
        assert data.ndim == 1
        assert data.min() >= -1.0 - 1e-4
        assert data.max() <= 1.0 + 1e-4
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (b) load_audio — stereo int16 WAV, mono=True → 1D array
# ---------------------------------------------------------------------------

def test_load_audio_stereo_mono_downmix():
    from plottter.generators.audio_utils import load_audio

    sine = _make_sine(440, 1.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        _write_wav_stereo_int16(path, sine)
        sr, data = load_audio(path, mono=True)
        assert sr == SR
        assert data.ndim == 1
        assert data.dtype == np.float64
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (c) load_audio — stereo WAV, mono=False → 2D array shape (N, 2)
# ---------------------------------------------------------------------------

def test_load_audio_stereo_keep_channels():
    from plottter.generators.audio_utils import load_audio

    sine = _make_sine(440, 1.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        _write_wav_stereo_int16(path, sine)
        sr, data = load_audio(path, mono=False)
        assert sr == SR
        assert data.ndim == 2
        assert data.shape[1] == 2
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (d) load_audio — segment extraction returns correct number of samples
# ---------------------------------------------------------------------------

def test_load_audio_segment_extraction():
    from plottter.generators.audio_utils import load_audio

    duration = 3.0
    sine = _make_sine(440, duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        _write_wav_int16(path, sine)
        seg_duration = 0.5
        sr, data = load_audio(path, start_sec=1.0, duration_sec=seg_duration)
        expected_samples = int(seg_duration * SR)
        assert len(data) == expected_samples
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (e) load_audio — non-existent file raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_load_audio_file_not_found():
    from plottter.generators.audio_utils import load_audio

    with pytest.raises(FileNotFoundError):
        load_audio("/tmp/definitely_does_not_exist_12345.wav")


# ---------------------------------------------------------------------------
# (f) load_audio — .mp3 without pydub raises ImportError with helpful message
# ---------------------------------------------------------------------------

def test_load_audio_mp3_no_pydub_raises_import_error(monkeypatch, tmp_path):
    from plottter.generators import audio_utils

    # Create a fake .mp3 file so the existence check passes
    mp3_path = tmp_path / "test.mp3"
    mp3_path.write_bytes(b"\x00" * 16)

    # Patch builtins.__import__ to simulate pydub not being installed
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydub":
            raise ImportError("No module named 'pydub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pydub is required"):
        audio_utils.load_audio(mp3_path)


# ---------------------------------------------------------------------------
# (g) compute_spectrogram — Sxx shape (num_rows, *) and values in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_spectrogram_shape_and_range():
    from plottter.generators.audio_utils import compute_spectrogram

    num_rows = 40
    sine = _make_sine(440, 2.0)
    freqs, times, Sxx = compute_spectrogram(
        sine, SR, nperseg=1024, overlap_frac=0.75,
        freq_min=0, freq_max=8000, num_rows=num_rows,
    )
    assert Sxx.shape[0] == num_rows
    assert Sxx.min() >= 0.0 - 1e-9
    assert Sxx.max() <= 1.0 + 1e-9
    assert freqs.shape[0] == num_rows


# ---------------------------------------------------------------------------
# (h) compute_spectrogram — peak energy near 440 Hz for a 440 Hz sine
# ---------------------------------------------------------------------------

def test_compute_spectrogram_peak_at_440hz():
    from plottter.generators.audio_utils import compute_spectrogram

    num_rows = 80
    sine = _make_sine(440, 3.0, amplitude=0.8)
    freqs, times, Sxx = compute_spectrogram(
        sine, SR, nperseg=4096, overlap_frac=0.75,
        freq_min=0, freq_max=8000, num_rows=num_rows,
    )
    # Find the frequency row index with the highest mean energy
    mean_energy = Sxx.mean(axis=1)
    peak_idx = int(np.argmax(mean_energy))
    peak_freq = freqs[peak_idx]

    # Peak should be within ±200 Hz of 440 Hz
    assert abs(peak_freq - 440.0) < 200.0, (
        f"Expected peak near 440 Hz but got {peak_freq:.1f} Hz"
    )


# ---------------------------------------------------------------------------
# (i) split_frequency_bands — edges [250, 4000] → 3 arrays, same length
# ---------------------------------------------------------------------------

def test_split_frequency_bands_count_and_length():
    from plottter.generators.audio_utils import split_frequency_bands

    data = _make_sine(200, 1.0)
    bands = split_frequency_bands(data, SR, band_edges=[250, 4000])
    assert len(bands) == 3
    for band in bands:
        assert len(band) == len(data)


# ---------------------------------------------------------------------------
# (j) split_frequency_bands — 200 Hz sine: bass band has highest RMS energy
# ---------------------------------------------------------------------------

def test_split_frequency_bands_bass_energy():
    from plottter.generators.audio_utils import split_frequency_bands

    # 200 Hz signal → should mostly pass through the low-frequency (bass) band
    data = _make_sine(200, 2.0, amplitude=0.8)
    bands = split_frequency_bands(data, SR, band_edges=[250, 4000])
    rms_values = [float(np.sqrt(np.mean(b ** 2))) for b in bands]
    # Bass band (index 0) should have highest RMS
    assert rms_values[0] == max(rms_values), (
        f"Expected bass band to have highest RMS, got: {rms_values}"
    )


# ---------------------------------------------------------------------------
# (k) compute_envelope — same length as input, all values >= 0
# ---------------------------------------------------------------------------

def test_compute_envelope_length_and_nonneg():
    from plottter.generators.audio_utils import compute_envelope

    data = _make_sine(440, 1.0)
    env = compute_envelope(data, window_size=512)
    assert len(env) == len(data)
    assert np.all(env >= 0.0)


# ---------------------------------------------------------------------------
# (l) compute_envelope — constant-amplitude sine → approximately constant
# ---------------------------------------------------------------------------

def test_compute_envelope_constant_amplitude():
    from plottter.generators.audio_utils import compute_envelope

    # Long enough signal so edge effects are negligible in the middle
    data = _make_sine(440, 3.0, amplitude=0.5)
    window_size = 1024
    env = compute_envelope(data, window_size=window_size)

    # Exclude edges (half a window on each side)
    half_w = window_size // 2
    mid = env[half_w:-half_w]
    expected_rms = 0.5 / np.sqrt(2)  # RMS of sine with amplitude 0.5
    # Allow 10% tolerance
    assert np.abs(mid.mean() - expected_rms) < 0.1 * expected_rms, (
        f"Expected envelope mean ~{expected_rms:.4f}, got {mid.mean():.4f}"
    )
    # Coefficient of variation should be low (< 5%)
    cv = mid.std() / mid.mean()
    assert cv < 0.05, f"Envelope is not approximately constant; CV = {cv:.4f}"


# ---------------------------------------------------------------------------
# (m) get_audio_duration — returns expected duration for a WAV
# ---------------------------------------------------------------------------

def test_get_audio_duration_wav():
    from plottter.generators.audio_utils import get_audio_duration

    expected_duration = 2.0
    sine = _make_sine(440, expected_duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    try:
        _write_wav_int16(path, sine)
        duration = get_audio_duration(path)
        # Allow small rounding difference
        assert abs(duration - expected_duration) < 0.01, (
            f"Expected duration ~{expected_duration}s, got {duration:.4f}s"
        )
    finally:
        path.unlink(missing_ok=True)
