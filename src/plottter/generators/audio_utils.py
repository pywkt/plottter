"""Audio loading and analysis utilities for audio-driven art generators."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.ndimage
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
        try:
            audio = AudioSegment.from_file(str(filepath))
        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg is required to decode .{ext} files but was not found. "
                "Install with: sudo apt install ffmpeg (Linux), "
                "brew install ffmpeg (macOS), or download from https://ffmpeg.org"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode .{ext} file: {exc}. "
                "Ensure ffmpeg is installed and the file is a valid audio file."
            )
        return len(audio) / 1000.0


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def find_interesting_segment(
    data: np.ndarray,
    sample_rate: int,
    segment_duration: float,
    num_candidates: int = 20,
) -> tuple[int, int]:
    """Find the most energetic segment in audio data.

    Parameters
    ----------
    data:
        Audio data. Shape (N,) for mono or (N, 2) for stereo.
    sample_rate:
        Sample rate in Hz.
    segment_duration:
        Duration of the segment in seconds.
    num_candidates:
        Number of evenly-spaced candidate positions to evaluate.

    Returns
    -------
    (start_sample, end_sample) of the most energetic segment.
    """
    len_data = data.shape[0]
    seg_len = int(segment_duration * sample_rate)
    seg_len = max(1, min(seg_len, len_data))

    candidates = np.linspace(0, max(0, len_data - seg_len), num_candidates, dtype=int)

    best_start = int(candidates[0])
    best_energy = -1.0

    for start in candidates:
        start = int(start)
        end = start + seg_len
        segment = data[start:end]
        if data.ndim == 2:
            # Sum energy across both channels
            energy = float(np.sum(segment ** 2))
        else:
            energy = float(np.sum(segment ** 2))
        if energy > best_energy:
            best_energy = energy
            best_start = start

    return best_start, best_start + seg_len


# ---------------------------------------------------------------------------
# Ridgeline hidden-line-removal
# ---------------------------------------------------------------------------

def _extract_visible_segments(
    x: np.ndarray,
    y: np.ndarray,
    visible_mask: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Extract contiguous visible runs as polylines.

    Parameters
    ----------
    x, y:
        1-D coordinate arrays of the same length.
    visible_mask:
        Boolean array; True where points are visible.

    Returns
    -------
    List of polylines (each a list of (x, y) tuples) with >= 2 points.
    """
    segments: list[list[tuple[float, float]]] = []
    if not np.any(visible_mask):
        return segments

    # Use diff to find run boundaries
    mask_int = visible_mask.astype(int)
    diff = np.diff(mask_int)

    starts: list[int] = []
    ends: list[int] = []

    # Run starts at index where diff == 1 (False→True), shifted by 1
    starts_arr = np.where(diff == 1)[0] + 1
    # Run ends at index where diff == -1 (True→False)
    ends_arr = np.where(diff == -1)[0]

    # Handle edge: visible at the very start
    if visible_mask[0]:
        starts = [0] + list(starts_arr)
    else:
        starts = list(starts_arr)

    # Handle edge: visible at the very end
    if visible_mask[-1]:
        ends = list(ends_arr) + [len(visible_mask) - 1]
    else:
        ends = list(ends_arr)

    for s, e in zip(starts, ends):
        if e - s + 1 >= 2:
            pts = [(float(x[j]), float(y[j])) for j in range(s, e + 1)]
            segments.append(pts)

    return segments


def ridgeline_hlr(
    spectrogram_rows: np.ndarray,
    width: float,
    amplitude_scale: float = 1.0,
    row_spacing: float = 1.0,
    smoothing_sigma: float = 2.0,
    mirror: bool = False,
) -> list[list[tuple[float, float]]]:
    """Ridgeline plot with hidden-line removal (Unknown Pleasures effect).

    Parameters
    ----------
    spectrogram_rows:
        2-D array of shape (num_rows, num_time_cols) with values in [0, 1].
        Row 0 is the front (bottom), last row is the back (top).
    width:
        Horizontal extent of the plot in mm.
    amplitude_scale:
        Vertical scale factor applied to each row's values.
    row_spacing:
        Vertical distance between row baselines.
    smoothing_sigma:
        Sigma for Gaussian smoothing applied to each row (0 = no smoothing).
    mirror:
        If True, also draw downward-facing ridges, symmetric about each baseline.

    Returns
    -------
    List of polylines (list of (x, y) tuples).
    """
    num_rows, num_cols = spectrogram_rows.shape
    x = np.linspace(0.0, width, num_cols)

    polylines: list[list[tuple[float, float]]] = []

    # Upward horizon: tracks the highest y seen so far per column.
    horizon = np.full(num_cols, -np.inf)

    # Downward horizon (mirror mode)
    horizon_down = np.full(num_cols, np.inf) if mirror else None

    for i in range(num_rows):
        baseline = float(i) * row_spacing
        row = spectrogram_rows[i].astype(float)

        if smoothing_sigma > 0:
            smoothed = scipy.ndimage.gaussian_filter1d(row, sigma=smoothing_sigma)
        else:
            smoothed = row

        y = baseline + smoothed * amplitude_scale

        # Upward: visible where y > horizon
        visible = y > horizon
        segs = _extract_visible_segments(x, y, visible)
        polylines.extend(segs)

        # Update horizon: the curve itself plus the baseline fill
        horizon = np.maximum(horizon, y)
        horizon = np.maximum(horizon, baseline)

        # Mirror (downward)
        if mirror and horizon_down is not None:
            y_down = baseline - smoothed * amplitude_scale
            visible_down = y_down < horizon_down
            segs_down = _extract_visible_segments(x, y_down, visible_down)
            polylines.extend(segs_down)
            horizon_down = np.minimum(horizon_down, y_down)
            horizon_down = np.minimum(horizon_down, baseline)

    return polylines


def ridgeline_no_hlr(
    spectrogram_rows: np.ndarray,
    width: float,
    amplitude_scale: float = 1.0,
    row_spacing: float = 1.0,
    smoothing_sigma: float = 2.0,
    mirror: bool = False,
) -> list[list[tuple[float, float]]]:
    """Ridgeline plot without hidden-line removal (every row drawn in full).

    Parameters
    ----------
    spectrogram_rows:
        2-D array of shape (num_rows, num_time_cols) with values in [0, 1].
    width:
        Horizontal extent of the plot in mm.
    amplitude_scale:
        Vertical scale factor.
    row_spacing:
        Vertical distance between row baselines.
    smoothing_sigma:
        Sigma for Gaussian smoothing (0 = no smoothing).
    mirror:
        If True, also draw a downward-facing copy of each row.

    Returns
    -------
    List of polylines; exactly one polyline per row (two when mirror=True).
    """
    num_rows, num_cols = spectrogram_rows.shape
    x = np.linspace(0.0, width, num_cols)
    polylines: list[list[tuple[float, float]]] = []

    for i in range(num_rows):
        baseline = float(i) * row_spacing
        row = spectrogram_rows[i].astype(float)

        if smoothing_sigma > 0:
            smoothed = scipy.ndimage.gaussian_filter1d(row, sigma=smoothing_sigma)
        else:
            smoothed = row

        y = baseline + smoothed * amplitude_scale
        polylines.append([(float(x[j]), float(y[j])) for j in range(num_cols)])

        if mirror:
            y_down = baseline - smoothed * amplitude_scale
            polylines.append([(float(x[j]), float(y_down[j])) for j in range(num_cols)])

    return polylines


# ---------------------------------------------------------------------------
# Spectrogram contour extraction
# ---------------------------------------------------------------------------

def extract_contours(
    data_2d: np.ndarray,
    num_levels: int = 10,
    smoothing_sigma: float = 1.5,
) -> list[list[tuple[float, float]]]:
    """Extract contour polylines from a 2D array.

    Parameters
    ----------
    data_2d:
        2D array to extract contours from.
    num_levels:
        Number of contour levels.
    smoothing_sigma:
        Gaussian smoothing sigma applied before extraction (0 = no smoothing).

    Returns
    -------
    List of polylines, each a list of (x, y) tuples in (column, row) space.
    """
    smoothed = scipy.ndimage.gaussian_filter(data_2d, sigma=smoothing_sigma)

    # If the array is uniform there is nothing to contour
    if smoothed.max() <= smoothed.min():
        return []

    levels = np.linspace(smoothed.min(), smoothed.max(), num_levels + 2)[1:-1]

    polylines: list[list[tuple[float, float]]] = []

    try:
        from skimage.measure import find_contours as _sk_find_contours  # type: ignore[import]

        for level in levels:
            for contour in _sk_find_contours(smoothed, level):
                # skimage returns (row, col); swap to (col, row) = (x, y)
                pts = [(float(c[1]), float(c[0])) for c in contour]
                if len(pts) >= 2:
                    polylines.append(pts)

    except ImportError:
        # Fallback: boundary-extraction + nearest-neighbour ordering
        for level in levels:
            mask = smoothed >= level
            if not np.any(mask) or np.all(mask):
                continue
            eroded = scipy.ndimage.binary_erosion(mask)
            boundary = mask ^ eroded  # XOR → boundary pixels only
            coords = np.argwhere(boundary)  # shape (N, 2), each row is [row, col]
            if len(coords) < 2:
                continue
            polylines.extend(_chain_boundary_points(coords))

    return polylines


def _chain_boundary_points(
    coords: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Order boundary pixels into polylines via greedy nearest-neighbour traversal.

    Parameters
    ----------
    coords:
        Array of shape (N, 2) containing (row, col) coordinates.

    Returns
    -------
    List of polylines in (col, row) = (x, y) coordinates.
    """
    from scipy.spatial import cKDTree  # local import to keep top-level clean

    n = len(coords)
    if n < 2:
        return []

    visited = np.zeros(n, dtype=bool)
    polylines: list[list[tuple[float, float]]] = []
    tree = cKDTree(coords)

    # Maximum step: slightly above sqrt(2) so 8-connected pixels are linked
    max_step = 2.0
    k = min(10, n)

    while True:
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break

        current = int(unvisited[0])
        chain: list[tuple[float, float]] = []

        while not visited[current]:
            visited[current] = True
            row, col = coords[current]
            chain.append((float(col), float(row)))  # x=col, y=row

            dists, idxs = tree.query(coords[current], k=k)

            next_idx = -1
            for d, idx in zip(dists, idxs):
                idx = int(idx)
                if visited[idx]:
                    continue
                if d > max_step:
                    break  # results are distance-sorted; no need to continue
                next_idx = idx
                break

            if next_idx < 0:
                break
            current = next_idx

        if len(chain) >= 2:
            polylines.append(chain)

    return polylines


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

    try:
        audio = AudioSegment.from_file(str(filepath))
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg is required to decode .{ext} files but was not found. "
            "Install with: sudo apt install ffmpeg (Linux), "
            "brew install ffmpeg (macOS), or download from https://ffmpeg.org"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode .{ext} file: {exc}. "
            "Ensure ffmpeg is installed and the file is a valid audio file."
        )

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
