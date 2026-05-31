"""Pointillist Generator — optical-mixing dot art using per-pen masks.

One layer per palette colour; dots placed with Mitchell's best-candidate
sampler and rendered as polylines ready for pen plotters.
"""

from __future__ import annotations

from typing import Any

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    LayerSpec,
    Parameter,
    Polyline,
    Preset,
)
from plottter.models.canvas import Canvas


def _palette_choices() -> list[str]:
    """Return palette preset names at call time (includes user palettes)."""
    from plottter.color import list_presets

    return [p.name for p in list_presets()]


@register_generator
class PointillistGenerator(Generator):
    """Multi-layer generator that reproduces a colour image as non-overlapping
    dots of pure pen colours, enabling optical colour mixing on paper.
    """

    name = "Pointillist"
    category = "image"
    uses_source_image = True
    uses_color_source = True
    emits_multiple_layers = True

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="palette",
                label="Palette",
                choices=_palette_choices(),
                default="Basic 6",
                randomizable=False,
                description=(
                    "Pen palette used to separate the image into colour layers. "
                    "Each palette entry produces one dot layer."
                ),
            ),
            FloatParam(
                name="density_per_cm2",
                label="Density (dots/cm²)",
                min=10.0,
                max=2000.0,
                step=1.0,
                default=200.0,
                randomizable=False,
                description=(
                    "Target dot density per cm² for a fully-covered pen layer. "
                    "Pens that cover less area receive proportionally fewer dots."
                ),
            ),
            ChoiceParam(
                name="dither",
                label="Dithering",
                choices=["none", "floyd-steinberg", "ordered", "atkinson"],
                default="floyd-steinberg",
                randomizable=False,
                description=(
                    "Dithering method passed to palette separation. "
                    "Floyd-Steinberg is recommended for optical-mixing output."
                ),
            ),
            ChoiceParam(
                name="dot_style",
                label="Dot Style",
                choices=["point", "cross", "circle"],
                default="point",
                randomizable=False,
                description=(
                    "Shape rendered at each dot position. "
                    "'point' draws a minimal pen-down mark; "
                    "'cross' draws two perpendicular strokes; "
                    "'circle' draws a closed circular outline."
                ),
            ),
            FloatParam(
                name="dot_size_mm",
                label="Dot Size (mm)",
                min=0.1,
                max=3.0,
                step=0.1,
                default=0.5,
                randomizable=False,
                visible_when={"dot_style": ["cross", "circle"]},
                description=(
                    "Dot size in mm. Used by 'cross' and 'circle' styles; "
                    "ignored by 'point'."
                ),
            ),
            IntParam(
                name="seed",
                label="Seed",
                min=0,
                max=99999,
                step=1,
                default=0,
                randomizable=True,
                description=(
                    "Random seed for Mitchell's best-candidate sampler. "
                    "Different seeds re-shuffle dot positions without changing density."
                ),
            ),
            BoolParam(
                name="skip_paper_white",
                label="Skip Paper White",
                default=True,
                randomizable=False,
                description=(
                    "When the palette contains #FFFFFF, skip that layer "
                    "(no point plotting white on white paper)."
                ),
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Pointillist Classic",
                params={
                    "palette": "Basic 6",
                    "density_per_cm2": 250.0,
                    "dither": "floyd-steinberg",
                    "dot_style": "point",
                    "dot_size_mm": 0.5,
                    "seed": 0,
                    "skip_paper_white": True,
                },
            ),
            Preset(
                name="Halftone Dots",
                params={
                    "palette": "Copic 12",
                    "density_per_cm2": 600.0,
                    "dither": "ordered",
                    "dot_style": "point",
                    "dot_size_mm": 0.3,
                    "seed": 0,
                    "skip_paper_white": True,
                },
            ),
            Preset(
                name="Big Cross Stipple",
                params={
                    "palette": "Basic 6",
                    "density_per_cm2": 80.0,
                    "dither": "floyd-steinberg",
                    "dot_style": "cross",
                    "dot_size_mm": 1.2,
                    "seed": 0,
                    "skip_paper_white": True,
                },
            ),
            Preset(
                name="Sketchy Mono",
                params={
                    "palette": "Grayscale 5",
                    "density_per_cm2": 300.0,
                    "dither": "none",
                    "dot_style": "point",
                    "dot_size_mm": 0.5,
                    "seed": 0,
                    "skip_paper_white": True,
                },
            ),
        ]

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Return one LayerSpec per non-empty, non-skipped palette colour."""
        from plottter.color import get_preset, palette_separate
        from plottter.generators._pointillist_core import (
            image_to_canvas_mm,
            mitchell_sample,
            render_dots,
        )

        image = params.get("_source_image")
        if image is None:
            return []

        palette = get_preset(str(params.get("palette", "Basic 6")))
        dither = str(params.get("dither", "floyd-steinberg"))
        masks = palette_separate(image, palette, dither=dither)

        left, top, right, bottom = canvas.drawing_area()
        canvas_area_cm2 = ((right - left) * (bottom - top)) / 100.0

        seed = int(params.get("seed", 0))
        style = str(params.get("dot_style", "point"))
        size_mm = float(params.get("dot_size_mm", 0.5))
        density = float(params.get("density_per_cm2", 200.0))
        skip_white = bool(params.get("skip_paper_white", True))

        layer_specs: list[LayerSpec] = []
        for i, (mask, hex_color) in enumerate(masks):
            if skip_white and hex_color.upper() == "#FFFFFF":
                continue
            if not (mask == 255).any():
                continue

            n = int(round(density * canvas_area_cm2 * float((mask == 255).mean())))
            if n <= 0:
                continue

            dots_image = mitchell_sample(mask, n, seed=seed + i, candidates=10)
            dots_mm = image_to_canvas_mm(
                dots_image, mask.shape[:2], canvas.drawing_area()
            )
            paths = render_dots(dots_mm, style=style, size_mm=size_mm)

            layer_specs.append(
                LayerSpec(
                    name=f"Pen {i + 1} ({hex_color})",
                    color=hex_color,
                    paths=paths,
                )
            )

            if progress_callback:
                progress_callback((i + 1) / len(masks))
            if cancelled_callback and cancelled_callback():
                break

        return layer_specs

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Fallback single-layer mode: flatten all layer paths into one list."""
        specs = self.generate_layers(params, canvas, progress_callback, cancelled_callback)
        paths: list[Polyline] = []
        for spec in specs:
            paths.extend(spec.paths)
        return paths
