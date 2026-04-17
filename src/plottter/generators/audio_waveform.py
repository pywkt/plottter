"""Audio Waveform Generator — spectrum-driven ridgeline and other modes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators.audio_utils import (
    compute_spectrogram,
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
        ]
