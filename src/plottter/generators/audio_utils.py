"""Audio loading and analysis utilities for audio-driven art generators."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    filepath: str | os.PathLike,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    mono: bool = True,
) -> tuple[int, np.ndarray]:
    """Load an audio file and return (sample_rate, data).

    Parameters
    ----------
    filepath:
        Path to the audio file.
    start_sec:
        Start offset in seconds (default 0.0).
    duration_sec:
        Duration to load in seconds. None means load to end.
    mono:
        If True and the file is stereo, average channels to produce a
        1-D array. If False, stereo data is returned as shape (N, 2).

    Returns
    -------
    sample_rate : int
    data : np.ndarray
        float64 array normalised to [-1, 1].  Shape is (N,) for mono or
        (N, 2) for stereo when mono=False.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ImportError
        When a non-WAV format is requested but pydub is not installed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = filepath.suffix.lower().lstrip(".")

    if ext == "wav":
        return _load_wav(filepath, start_sec, duration_sec, mono)
    else:
        return _load_via_pydub(filepath, ext, start_sec, duration_sec, mono)


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    nperseg: int = 2048,
    overlap_frac: float = 0.75,
    freq_min: float = 0,
    freq_max: float = 8000,
    num_rows: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a normalised spectrogram from mono audio data.

    Parameters
    ----------
    data:
        1-D float64 audio samples normalised to [-1, 1].
    sample_rate:
        Sample rate in Hz.
    nperseg:
        FFT window size.
    overlap_frac:
        Fraction of window overlap [0, 1).
    freq_min, freq_max:
        Frequency range of interest in Hz.
    num_rows:
        Number of frequency rows in the output Sxx.

    Returns
    -------
    freqs : np.ndarray  shape (num_rows,)
    times : np.ndarray  shape (num_time_cols,)
    Sxx   : np.ndarray  shape (num_rows, num_time_cols), values in [0, 1]
    """
    noverlap = int(nperseg * overlap_frac)
    freqs, times, Sxx = scipy.signal.spectrogram(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Convert to dB
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)

    # Crop to [freq_min, freq_max]
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_crop = freqs[freq_mask]
    Sxx_crop = Sxx_db[freq_mask, :]

    # Resample frequency axis to num_rows using index selection
    if len(freqs_crop) >= num_rows:
        indices = np.linspace(0, len(freqs_crop) - 1, num_rows, dtype=int)
    else:
        # Fewer freq bins than num_rows: repeat/interpolate
        indices = np.round(np.linspace(0, len(freqs_crop) - 1, num_rows)).astype(int)
    freqs_out = freqs_crop[indices]
    Sxx_resampled = Sxx_crop[indices, :]

    # Normalise to [0, 1]
    smin = Sxx_resampled.min()
    smax = Sxx_resampled.max()
    if smax > smin:
        Sxx_norm = (Sxx_resampled - smin) / (smax - smin)
    else:
        Sxx_norm = np.zeros_like(Sxx_resampled)

    # Noise floor: set values below 10th percentile to 0
    floor = np.percentile(Sxx_norm, 10)
    Sxx_norm[Sxx_norm < floor] = 0.0

    return freqs_out, times, Sxx_norm


def split_frequency_bands(
    data: np.ndarray,
    sample_rate: int,
    band_edges: list[float],
) -> list[np.ndarray]:
    """Split a signal into frequency bands using zero-phase Butterworth filters.

    Parameters
    ----------
    data:
        1-D audio signal.
    sample_rate:
        Sample rate in Hz.
    band_edges:
        Frequency edges in Hz.  E.g. [250, 4000] yields 3 bands:
        [0–250 Hz], [250–4000 Hz], [4000+ Hz].

    Returns
    -------
    bands : list[np.ndarray]
        One array per band, each the same length as *data*.
    """
    nyq = sample_rate / 2.0
    order = 4
    bands: list[np.ndarray] = []

    def _clamp(f: float) -> float:
        return float(np.clip(f / nyq, 0.001, 0.999))

    # First band: lowpass up to band_edges[0]
    sos = scipy.signal.butter(order, _clamp(band_edges[0]), btype="low", output="sos")
    bands.append(scipy.signal.sosfiltfilt(sos, data))

    # Middle bands: bandpass between adjacent edges
    for i in range(len(band_edges) - 1):
        lo = _clamp(band_edges[i])
        hi = _clamp(band_edges[i + 1])
        sos = scipy.signal.butter(order, [lo, hi], btype="band", output="sos")
        bands.append(scipy.signal.sosfiltfilt(sos, data))

    # Last band: highpass above band_edges[-1]
    sos = scipy.signal.butter(order, _clamp(band_edges[-1]), btype="high", output="sos")
    bands.append(scipy.signal.sosfiltfilt(sos, data))

    return bands


def compute_envelope(data: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """Compute the RMS amplitude envelope of a signal.

    Parameters
    ----------
    data:
        1-D audio signal.
    window_size:
        Sliding window size in samples.

    Returns
    -------
    envelope : np.ndarray
        Same length as *data*, all values >= 0.
    """
    kernel = np.ones(window_size) / window_size
    rms_squared = np.convolve(data ** 2, kernel, mode="same")
    return np.sqrt(rms_squared)


def get_audio_duration(filepath: str | os.PathLike) -> float:
    """Return the duration of an audio file in seconds.

    Parameters
    ----------
    filepath:
        Path to the audio file.

    Returns
    -------
    duration : float
        Duration in seconds.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ImportError
        When pydub is required but not installed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = filepath.suffix.lower().lstrip(".")
    if ext == "wav":
        sample_rate, wav_data = scipy.io.wavfile.read(str(filepath), mmap=True)
        n_samples = wav_data.shape[0]
        return float(n_samples) / float(sample_rate)
    else:
        try:
            from pydub import AudioSegment  # type: ignore[import]
        except ImportError:
            raise ImportError(
                f"pydub is required for .{ext} files. "
                "Install with: pip install pydub (also requires ffmpeg)"
            )
        audio = AudioSegment.from_file(str(filepath))
        return len(audio) / 1000.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_wav(
    filepath: Path,
    start_sec: float,
    duration_sec: float | None,
    mono: bool,
) -> tuple[int, np.ndarray]:
    """Load a WAV file and return (sample_rate, float64 data)."""
    sample_rate, raw = scipy.io.wavfile.read(str(filepath), mmap=True)

    # Normalise dtype to float64 in [-1, 1]
    dtype = raw.dtype
    if dtype == np.int16:
        data = raw.astype(np.float64) / 32768.0
    elif dtype == np.int32:
        data = raw.astype(np.float64) / 2147483648.0
    elif dtype == np.uint8:
        data = (raw.astype(np.float64) - 128.0) / 128.0
    elif dtype in (np.float32, np.float64):
        data = raw.astype(np.float64)
    else:
        # Fallback: normalise by max possible value
        info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
        if info is not None:
            data = raw.astype(np.float64) / float(max(abs(info.min), abs(info.max)))
        else:
            data = raw.astype(np.float64)

    # Slice in time
    data = _slice_samples(data, sample_rate, start_sec, duration_sec)

    # Mono downmix
    if mono and data.ndim == 2:
        data = data.mean(axis=1)

    return sample_rate, data


def _load_via_pydub(
    filepath: Path,
    ext: str,
    start_sec: float,
    duration_sec: float | None,
    mono: bool,
) -> tuple[int, np.ndarray]:
    """Load a non-WAV audio file using pydub."""
    try:
        from pydub import AudioSegment  # type: ignore[import]
    except ImportError:
        raise ImportError(
            f"pydub is required for .{ext} files. "
            "Install with: pip install pydub (also requires ffmpeg)"
        )

    audio = AudioSegment.from_file(str(filepath))

    # Slice by milliseconds
    start_ms = int(start_sec * 1000)
    if duration_sec is not None:
        end_ms = start_ms + int(duration_sec * 1000)
        audio = audio[start_ms:end_ms]
    elif start_ms > 0:
        audio = audio[start_ms:]

    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    # Reshape for stereo
    if audio.channels == 2:
        samples = samples.reshape(-1, 2)

    # Normalise to float64 [-1, 1]
    divisor = float(2 ** (8 * audio.sample_width - 1))
    data = samples.astype(np.float64) / divisor

    # Mono downmix
    if mono and data.ndim == 2:
        data = data.mean(axis=1)

    return sample_rate, data


def _slice_samples(
    data: np.ndarray,
    sample_rate: int,
    start_sec: float,
    duration_sec: float | None,
) -> np.ndarray:
    """Slice a sample array by time and materialise any mmap."""
    if start_sec == 0.0 and duration_sec is None:
        # No slicing needed but still materialise if mmap
        if hasattr(data, '_mmap') or isinstance(data, np.memmap):
            return data.copy()
        return np.array(data, copy=False)

    start_sample = int(start_sec * sample_rate)
    if duration_sec is not None:
        end_sample = start_sample + int(duration_sec * sample_rate)
        sliced = data[start_sample:end_sample]
    else:
        sliced = data[start_sample:]

    # Materialise the mmap slice
    return sliced.copy()
