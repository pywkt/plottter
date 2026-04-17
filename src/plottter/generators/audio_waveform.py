"""Audio Waveform Generator — spectrum-driven ridgeline and other modes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal

from plottter.generators import register_generator
from plottter.generators.audio_utils import (
    compute_envelope,
    compute_spectrogram,
    extract_contours,
    load_audio,
    ridgeline_hlr,
    ridgeline_no_hlr,
)
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FileParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


@register_generator
class AudioWaveformGenerator(Generator):
    """Visualise audio files as plottable line art."""

    name = "Audio Waveform"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            FileParam(
                name="audio_file",
                label="Audio File",
                default="",
                filter=(
                    "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;"
                    "WAV Files (*.wav);;"
                    "All Files (*)"
                ),
                description="Path to an audio file",
            ),
            ChoiceParam(
                name="mode",
                label="Visualization",
                choices=["Ridgeline", "Circular", "Spiral", "Contour", "Frequency Bands", "Lissajous"],
                default="Ridgeline",
            ),
            FloatParam(
                name="start_sec",
                label="Start (seconds)",
                min=0.0,
                max=3600.0,
                step=0.1,
                default=0.0,
            ),
            FloatParam(
                name="duration_sec",
                label="Duration (seconds)",
                min=0.1,
                max=60.0,
                step=0.1,
                default=10.0,
            ),
            # --- Ridgeline-specific parameters ---
            IntParam(
                name="num_rows",
                label="Number of Lines",
                min=10,
                max=200,
                step=1,
                default=60,
                visible_when={"mode": ["Ridgeline"]},
            ),
            FloatParam(
                name="amplitude",
                label="Amplitude",
                min=0.1,
                max=5.0,
                step=0.1,
                default=1.5,
                visible_when={"mode": ["Ridgeline"]},
            ),
            FloatParam(
                name="row_spacing",
                label="Line Spacing",
                min=0.5,
                max=5.0,
                step=0.1,
                default=1.5,
                visible_when={"mode": ["Ridgeline"]},
            ),
            FloatParam(
                name="smoothing",
                label="Smoothing",
                min=0.0,
                max=10.0,
                step=0.5,
                default=2.0,
                visible_when={"mode": ["Ridgeline"]},
            ),
            IntParam(
                name="freq_max",
                label="Max Frequency (Hz)",
                min=500,
                max=20000,
                step=100,
                default=8000,
                visible_when={"mode": ["Ridgeline"]},
            ),
            IntParam(
                name="fft_size",
                label="FFT Size",
                min=512,
                max=8192,
                step=512,
                default=2048,
                visible_when={"mode": ["Ridgeline"]},
            ),
            BoolParam(
                name="mirror",
                label="Mirror",
                default=False,
                visible_when={"mode": ["Ridgeline"]},
            ),
            BoolParam(
                name="hlr_enabled",
                label="Hidden Line Removal",
                default=True,
                visible_when={"mode": ["Ridgeline"]},
            ),
            # --- Circular-specific parameters ---
            FloatParam(
                name="circle_amplitude",
                label="Amplitude",
                min=0.01,
                max=0.5,
                step=0.01,
                default=0.2,
                description="Waveform amplitude relative to radius",
                visible_when={"mode": ["Circular"]},
            ),
            IntParam(
                name="circle_points",
                label="Points",
                min=360,
                max=7200,
                step=360,
                default=3600,
                visible_when={"mode": ["Circular"]},
            ),
            FloatParam(
                name="circle_smoothing",
                label="Smoothing",
                min=0.0,
                max=20.0,
                step=1.0,
                default=5.0,
                visible_when={"mode": ["Circular"]},
            ),
            ChoiceParam(
                name="circle_source",
                label="Source",
                choices=["Waveform", "Envelope", "Spectrum"],
                default="Waveform",
                description="Waveform=raw audio, Envelope=amplitude envelope, Spectrum=frequency magnitudes",
                visible_when={"mode": ["Circular"]},
            ),
            BoolParam(
                name="circle_closed",
                label="Close Loop",
                default=True,
                visible_when={"mode": ["Circular"]},
            ),
            # --- Spiral-specific parameters ---
            IntParam(
                name="spiral_turns",
                label="Turns",
                min=1,
                max=30,
                step=1,
                default=8,
                visible_when={"mode": ["Spiral"]},
            ),
            FloatParam(
                name="spiral_amplitude",
                label="Amplitude",
                min=0.01,
                max=0.3,
                step=0.01,
                default=0.05,
                description="Waveform amplitude relative to spiral gap",
                visible_when={"mode": ["Spiral"]},
            ),
            IntParam(
                name="spiral_points",
                label="Points",
                min=1000,
                max=20000,
                step=1000,
                default=7200,
                visible_when={"mode": ["Spiral"]},
            ),
            FloatParam(
                name="spiral_smoothing",
                label="Smoothing",
                min=0.0,
                max=20.0,
                step=1.0,
                default=8.0,
                visible_when={"mode": ["Spiral"]},
            ),
            ChoiceParam(
                name="spiral_source",
                label="Source",
                choices=["Waveform", "Envelope"],
                default="Waveform",
                visible_when={"mode": ["Spiral"]},
            ),
            ChoiceParam(
                name="spiral_direction",
                label="Direction",
                choices=["Outward", "Inward"],
                default="Outward",
                visible_when={"mode": ["Spiral"]},
            ),
            # --- Contour-specific parameters ---
            IntParam(
                name="contour_levels",
                label="Contour Levels",
                min=3,
                max=30,
                step=1,
                default=10,
                visible_when={"mode": ["Contour"]},
            ),
            FloatParam(
                name="contour_smoothing",
                label="Smoothing",
                min=0.0,
                max=5.0,
                step=0.5,
                default=1.5,
                visible_when={"mode": ["Contour"]},
            ),
            IntParam(
                name="contour_freq_max",
                label="Max Frequency (Hz)",
                min=500,
                max=20000,
                step=100,
                default=8000,
                visible_when={"mode": ["Contour"]},
            ),
            IntParam(
                name="contour_fft_size",
                label="FFT Size",
                min=512,
                max=8192,
                step=512,
                default=2048,
                visible_when={"mode": ["Contour"]},
            ),
            FloatParam(
                name="contour_min_length",
                label="Min Contour Length (mm)",
                min=0.0,
                max=20.0,
                step=0.5,
                default=2.0,
                visible_when={"mode": ["Contour"]},
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        def _progress(pct: float) -> None:
            if progress_callback is not None:
                progress_callback(int(pct))

        def _cancelled() -> bool:
            return cancelled_callback is not None and cancelled_callback()

        mode = params.get("mode", "Ridgeline")

        if mode == "Ridgeline":
            return self._generate_ridgeline(params, canvas, _progress, _cancelled)

        if mode == "Circular":
            return self._generate_circular(params, canvas, _progress, _cancelled)

        if mode == "Spiral":
            return self._generate_spiral(params, canvas, _progress, _cancelled)

        if mode == "Contour":
            return self._generate_contour(params, canvas, _progress, _cancelled)

        # Other modes not yet implemented
        return []

    def _generate_ridgeline(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress: Any,
        cancelled: Any,
    ) -> list[Polyline]:
        audio_file = params.get("audio_file", "")
        if not audio_file or not Path(audio_file).is_file():
            return []

        start_sec = float(params.get("start_sec", 0.0))
        duration_sec = float(params.get("duration_sec", 10.0))
        num_rows = int(params.get("num_rows", 60))
        amplitude = float(params.get("amplitude", 1.5))
        row_spacing = float(params.get("row_spacing", 1.5))
        smoothing = float(params.get("smoothing", 2.0))
        freq_max = int(params.get("freq_max", 8000))
        fft_size = int(params.get("fft_size", 2048))
        mirror = bool(params.get("mirror", False))
        hlr_enabled = bool(params.get("hlr_enabled", True))

        # Load audio
        sample_rate, data = load_audio(audio_file, start_sec, duration_sec, mono=True)
        progress(10)

        if cancelled():
            return []

        # Compute spectrogram
        _freqs, _times, Sxx = compute_spectrogram(
            data,
            sample_rate,
            nperseg=fft_size,
            freq_max=freq_max,
            num_rows=num_rows,
        )
        progress(30)

        if cancelled():
            return []

        # Use drawing area width as the horizontal extent for the raw plot
        left, top, right, bottom = canvas.drawing_area()
        draw_width = right - left
        draw_height = bottom - top

        # Generate ridgelines
        if hlr_enabled:
            polylines = ridgeline_hlr(
                Sxx,
                width=draw_width,
                amplitude_scale=amplitude,
                row_spacing=row_spacing,
                smoothing_sigma=smoothing,
                mirror=mirror,
            )
        else:
            polylines = ridgeline_no_hlr(
                Sxx,
                width=draw_width,
                amplitude_scale=amplitude,
                row_spacing=row_spacing,
                smoothing_sigma=smoothing,
                mirror=mirror,
            )

        progress(80)

        if cancelled():
            return []

        if not polylines:
            return []

        # Scale and center output on canvas with 5mm padding
        padding = 5.0
        avail_w = draw_width - 2 * padding
        avail_h = draw_height - 2 * padding

        # Compute bounding box of all polylines
        all_pts = [pt for pl in polylines for pt in pl]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        if bbox_w <= 0 or bbox_h <= 0:
            return []

        # Uniform scale to fit within available area
        scale = min(avail_w / bbox_w, avail_h / bbox_h)

        # Center within drawing area
        scaled_w = bbox_w * scale
        scaled_h = bbox_h * scale
        offset_x = left + padding + (avail_w - scaled_w) / 2.0 - min_x * scale
        offset_y = top + padding + (avail_h - scaled_h) / 2.0 - min_y * scale

        scaled_polylines: list[Polyline] = [
            [(x * scale + offset_x, y * scale + offset_y) for x, y in pl]
            for pl in polylines
        ]

        progress(100)
        return scaled_polylines

    def _generate_circular(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress: Any,
        cancelled: Any,
    ) -> list[Polyline]:
        audio_file = params.get("audio_file", "")
        if not audio_file or not Path(audio_file).is_file():
            return []

        start_sec = float(params.get("start_sec", 0.0))
        duration_sec = float(params.get("duration_sec", 10.0))
        circle_amplitude = float(params.get("circle_amplitude", 0.2))
        circle_points = int(params.get("circle_points", 3600))
        circle_smoothing = float(params.get("circle_smoothing", 5.0))
        circle_source = params.get("circle_source", "Waveform")
        circle_closed = bool(params.get("circle_closed", True))

        # Load mono audio
        _sample_rate, data = load_audio(audio_file, start_sec, duration_sec, mono=True)
        progress(20)

        if cancelled():
            return []

        # Build signal based on source type
        n = len(data)
        out_idx = np.linspace(0, n - 1, circle_points)
        in_idx = np.arange(n)

        if circle_source == "Waveform":
            signal = np.interp(out_idx, in_idx, data)
        elif circle_source == "Envelope":
            envelope = compute_envelope(data)
            signal = np.interp(out_idx, in_idx, envelope)
        else:  # Spectrum
            spectrum = np.abs(np.fft.rfft(data))
            spec_idx = np.linspace(0, len(spectrum) - 1, circle_points)
            signal = np.interp(spec_idx, np.arange(len(spectrum)), spectrum)

        progress(50)

        if cancelled():
            return []

        # Smooth
        if circle_smoothing > 0:
            signal = scipy.ndimage.gaussian_filter1d(signal, sigma=circle_smoothing)

        # Normalise to [-1, 1]
        peak = max(np.abs(signal).max(), 1e-10)
        signal = signal / peak

        # Canvas geometry
        left, top, right, bottom = canvas.drawing_area()
        draw_width = right - left
        draw_height = bottom - top
        center_x = left + draw_width / 2.0
        center_y = top + draw_height / 2.0
        # Shrink base_radius so that even at maximum amplitude the waveform
        # stays within the 5mm padding.
        max_radial_extent = min(draw_width, draw_height) / 2.0 - 5.0
        base_radius = max_radial_extent / (1.0 + circle_amplitude)

        if base_radius <= 0:
            return []

        # Map to circle
        theta = np.linspace(0, 2 * np.pi, circle_points, endpoint=False)
        r = base_radius + circle_amplitude * base_radius * signal
        x = r * np.cos(theta) + center_x
        y = r * np.sin(theta) + center_y

        polyline: Polyline = [(float(x[i]), float(y[i])) for i in range(circle_points)]

        if circle_closed:
            polyline.append(polyline[0])

        progress(100)
        return [polyline]

    def _generate_spiral(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress: Any,
        cancelled: Any,
    ) -> list[Polyline]:
        audio_file = params.get("audio_file", "")
        if not audio_file or not Path(audio_file).is_file():
            return []

        start_sec = float(params.get("start_sec", 0.0))
        duration_sec = float(params.get("duration_sec", 10.0))
        spiral_turns = int(params.get("spiral_turns", 8))
        spiral_amplitude = float(params.get("spiral_amplitude", 0.05))
        spiral_points = int(params.get("spiral_points", 7200))
        spiral_smoothing = float(params.get("spiral_smoothing", 8.0))
        spiral_source = params.get("spiral_source", "Waveform")
        spiral_direction = params.get("spiral_direction", "Outward")

        # Load mono audio
        _sample_rate, data = load_audio(audio_file, start_sec, duration_sec, mono=True)
        progress(20)

        if cancelled():
            return []

        # Build signal based on source type
        n = len(data)
        out_idx = np.linspace(0, n - 1, spiral_points)
        in_idx = np.arange(n)

        if spiral_source == "Envelope":
            envelope = compute_envelope(data)
            signal = np.interp(out_idx, in_idx, envelope)
        else:  # Waveform
            signal = np.interp(out_idx, in_idx, data)

        progress(50)

        if cancelled():
            return []

        # Smooth
        if spiral_smoothing > 0:
            signal = scipy.ndimage.gaussian_filter1d(signal, sigma=spiral_smoothing)

        # Normalise to [-1, 1]
        peak = max(np.abs(signal).max(), 1e-10)
        signal = signal / peak

        # Canvas geometry
        left, top, right, bottom = canvas.drawing_area()
        draw_width = right - left
        draw_height = bottom - top
        center_x = left + draw_width / 2.0
        center_y = top + draw_height / 2.0

        r_outer = min(draw_width, draw_height) / 2.0 - 5.0
        r_inner = r_outer * 0.1

        if r_outer <= r_inner:
            return []

        gap = (r_outer - r_inner) / spiral_turns
        effective_amplitude = min(spiral_amplitude * r_outer, gap * 0.4)

        # Build spiral
        theta = np.linspace(0, 2 * np.pi * spiral_turns, spiral_points)
        r_base = np.linspace(r_inner, r_outer, spiral_points)

        if spiral_direction == "Inward":
            r_base = r_base[::-1]

        r = r_base + effective_amplitude * signal
        x = r * np.cos(theta) + center_x
        y = r * np.sin(theta) + center_y

        polyline: Polyline = [(float(x[i]), float(y[i])) for i in range(spiral_points)]

        progress(100)
        return [polyline]

    def _generate_contour(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress: Any,
        cancelled: Any,
    ) -> list[Polyline]:
        audio_file = params.get("audio_file", "")
        if not audio_file or not Path(audio_file).is_file():
            return []

        start_sec = float(params.get("start_sec", 0.0))
        duration_sec = float(params.get("duration_sec", 10.0))
        contour_levels = int(params.get("contour_levels", 10))
        contour_smoothing = float(params.get("contour_smoothing", 1.5))
        contour_freq_max = int(params.get("contour_freq_max", 8000))
        contour_fft_size = int(params.get("contour_fft_size", 2048))
        contour_min_length = float(params.get("contour_min_length", 2.0))

        # Load audio (mono)
        sample_rate, data = load_audio(audio_file, start_sec, duration_sec, mono=True)
        progress(10)

        if cancelled():
            return []

        # Compute spectrogram at full frequency resolution (no row downsampling)
        noverlap = int(contour_fft_size * 0.75)
        freqs, _times, Sxx_raw = scipy.signal.spectrogram(
            data,
            fs=sample_rate,
            window="hann",
            nperseg=contour_fft_size,
            noverlap=noverlap,
        )

        # Crop to [0, contour_freq_max]
        freq_mask = freqs <= contour_freq_max
        Sxx_crop = Sxx_raw[freq_mask, :]

        # Convert to dB and normalise to [0, 1]
        Sxx_db = 10.0 * np.log10(Sxx_crop + 1e-10)
        smin, smax = Sxx_db.min(), Sxx_db.max()
        if smax <= smin:
            return []
        Sxx_norm = (Sxx_db - smin) / (smax - smin)

        progress(30)

        if cancelled():
            return []

        # Extract contours in (column, row) space
        raw_contours = extract_contours(Sxx_norm, contour_levels, contour_smoothing)

        progress(60)

        if cancelled():
            return []

        if not raw_contours:
            return []

        # Map contour coordinates to drawing-area mm
        left, top, right, bottom = canvas.drawing_area()
        draw_width = right - left
        draw_height = bottom - top
        n_rows, n_cols = Sxx_norm.shape
        col_scale = draw_width / max(n_cols - 1, 1)
        row_scale = draw_height / max(n_rows - 1, 1)

        mapped: list[Polyline] = []
        for contour in raw_contours:
            pl: Polyline = [
                (cx * col_scale, cy * row_scale)
                for cx, cy in contour  # cx=col(time), cy=row(freq)
            ]
            mapped.append(pl)

        # Filter polylines shorter than contour_min_length mm
        if contour_min_length > 0.0:
            filtered: list[Polyline] = []
            for pl in mapped:
                arc = sum(
                    ((pl[i + 1][0] - pl[i][0]) ** 2 + (pl[i + 1][1] - pl[i][1]) ** 2) ** 0.5
                    for i in range(len(pl) - 1)
                )
                if arc >= contour_min_length:
                    filtered.append(pl)
            mapped = filtered

        if not mapped:
            return []

        progress(80)

        if cancelled():
            return []

        # Scale and center on canvas
        all_pts = [pt for pl in mapped for pt in pl]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        if bbox_w <= 0 or bbox_h <= 0:
            return []

        padding = 5.0
        avail_w = draw_width - 2 * padding
        avail_h = draw_height - 2 * padding
        scale = min(avail_w / bbox_w, avail_h / bbox_h)
        scaled_w = bbox_w * scale
        scaled_h = bbox_h * scale
        offset_x = left + padding + (avail_w - scaled_w) / 2.0 - min_x * scale
        offset_y = top + padding + (avail_h - scaled_h) / 2.0 - min_y * scale

        result: list[Polyline] = [
            [(x * scale + offset_x, y * scale + offset_y) for x, y in pl]
            for pl in mapped
        ]

        progress(100)
        return result

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                "Joy Division",
                params={
                    "mode": "Ridgeline",
                    "num_rows": 80,
                    "amplitude": 1.8,
                    "row_spacing": 1.2,
                    "smoothing": 3.0,
                    "hlr_enabled": True,
                    "mirror": False,
                    "freq_max": 5000,
                },
            ),
            Preset(
                "Dense Ridgeline",
                params={
                    "mode": "Ridgeline",
                    "num_rows": 120,
                    "amplitude": 1.0,
                    "row_spacing": 0.8,
                    "smoothing": 2.0,
                },
            ),
            Preset(
                "Wide Ridgeline",
                params={
                    "mode": "Ridgeline",
                    "num_rows": 40,
                    "amplitude": 2.5,
                    "row_spacing": 3.0,
                    "smoothing": 1.0,
                },
            ),
            Preset(
                "Mirror Ridgeline",
                params={
                    "mode": "Ridgeline",
                    "num_rows": 60,
                    "mirror": True,
                    "amplitude": 1.5,
                },
            ),
            Preset(
                "Circular Waveform",
                params={
                    "mode": "Circular",
                    "circle_source": "Waveform",
                    "circle_points": 3600,
                    "circle_amplitude": 0.2,
                    "circle_smoothing": 5.0,
                },
            ),
            Preset(
                "Circular Envelope",
                params={
                    "mode": "Circular",
                    "circle_source": "Envelope",
                    "circle_points": 3600,
                    "circle_amplitude": 0.3,
                    "circle_smoothing": 8.0,
                },
            ),
            Preset(
                "Circular Spectrum",
                params={
                    "mode": "Circular",
                    "circle_source": "Spectrum",
                    "circle_points": 1800,
                    "circle_amplitude": 0.25,
                    "circle_smoothing": 3.0,
                },
            ),
            Preset(
                "Vinyl Spiral",
                params={
                    "mode": "Spiral",
                    "spiral_turns": 12,
                    "spiral_direction": "Outward",
                    "spiral_source": "Envelope",
                    "spiral_amplitude": 0.04,
                    "spiral_smoothing": 10.0,
                },
            ),
            Preset(
                "Tight Spiral",
                params={
                    "mode": "Spiral",
                    "spiral_turns": 20,
                    "spiral_direction": "Outward",
                    "spiral_source": "Waveform",
                    "spiral_amplitude": 0.03,
                    "spiral_smoothing": 6.0,
                },
            ),
            Preset(
                "Topographic Audio",
                params={
                    "mode": "Contour",
                    "contour_levels": 12,
                    "contour_smoothing": 2.0,
                    "contour_freq_max": 6000,
                },
            ),
            Preset(
                "Detailed Contours",
                params={
                    "mode": "Contour",
                    "contour_levels": 20,
                    "contour_smoothing": 1.0,
                    "contour_freq_max": 10000,
                },
            ),
            Preset(
                "Smooth Contours",
                params={
                    "mode": "Contour",
                    "contour_levels": 8,
                    "contour_smoothing": 3.0,
                    "contour_freq_max": 5000,
                },
            ),
        ]
