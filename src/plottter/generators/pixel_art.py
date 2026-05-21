"""PixelArtGenerator — convert a source image to a per-palette-index layer grid.

Each palette colour gets its own layer, filled with hatch lines proportional to
the fill density.  The generator emits one :class:`LayerSpec` per colour index
that appears in the quantised grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._pixel_fills import fill_solid_hatch
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    LayerSpec,
    Parameter,
    Preset,
)
from plottter.models.canvas import Canvas

# Curated palette list — covers the most useful presets for plotter pixel art.
# Uses underscore-form keys so they round-trip cleanly through get_palette().
_PALETTE_CHOICES: list[str] = [
    "grayscale_4",
    "grayscale_2",
    "grayscale_8",
    "grayscale_16",
    "gameboy",
    "gameboy_pocket",
    "super_gameboy",
    "nes",
    "pico8",
    "c64",
    "cga",
    "ega",
    "endesga32",
    "sweetie16",
    "db32",
    "endesga64",
    "resurrect64",
]


@register_generator
class PixelArtGenerator(Generator):
    """Convert a source image to a palette-indexed cell grid, one layer per colour."""

    name = "Pixel Art"
    category = "image"
    uses_source_image = True
    emits_multiple_layers = True

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="grid_width",
                label="Grid Width (cells)",
                min=4,
                max=512,
                step=1,
                default=32,
                description="Number of cells across the drawing area width.",
            ),
            ChoiceParam(
                name="palette",
                label="Palette",
                choices=_PALETTE_CHOICES,
                default="grayscale_4",
                description="Colour palette used to quantise the source image.",
            ),
            ChoiceParam(
                name="cell_fill_style",
                label="Cell Fill Style",
                choices=["solid_hatch"],
                default="solid_hatch",
                description="Fill pattern drawn inside each cell.",
            ),
            FloatParam(
                name="fill_density",
                label="Fill Density",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                description="Hatch line density (0 = sparse, 1 = dense).",
            ),
            BoolParam(
                name="cell_border",
                label="Cell Border",
                default=False,
                description="Draw a rectangular outline around each cell.",
            ),
            FloatParam(
                name="cell_gap_mm",
                label="Cell Gap (mm)",
                min=0.0,
                max=5.0,
                step=0.1,
                default=0.0,
                description="Gap between adjacent cells in mm.",
            ),
        ]

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Game Boy",
                params={
                    "grid_width": 40,
                    "palette": "grayscale_4",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="NES Pixel",
                params={
                    "grid_width": 32,
                    "palette": "nes",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.8,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Return one LayerSpec per palette colour that appears in the grid."""
        from plottter.pixel_art import get_palette, image_to_palette_grid

        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        grid_width = int(params.get("grid_width", 32))
        palette_name = str(params.get("palette", "grayscale_4"))
        fill_style = str(params.get("cell_fill_style", "solid_hatch"))
        density = float(params.get("fill_density", 0.7))
        cell_border = bool(params.get("cell_border", False))
        cell_gap_mm = float(params.get("cell_gap_mm", 0.0))

        palette = get_palette(palette_name)
        indices = image_to_palette_grid(source, palette, grid_width)

        n_rows, n_cols = indices.shape
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1

        # cell_size_mm is the pitch (cell + gap); effective_cell_mm is the fill area.
        cell_size_mm = draw_w / n_cols
        effective_cell_mm = max(0.01, cell_size_mm - cell_gap_mm)

        hex_colors = palette.to_hex_list()
        n_colors = len(hex_colors)

        # Accumulate paths keyed by palette index.
        paths_by_index: dict[int, list] = {}

        total_cells = n_rows * n_cols
        report_every = max(1, total_cells // 20)
        processed = 0

        for r in range(n_rows):
            if cancelled_callback and cancelled_callback():
                break
            for c in range(n_cols):
                idx = int(indices[r, c])
                if idx < 0 or idx >= n_colors:
                    processed += 1
                    continue

                # Top-left corner of this cell's fill area.
                cell_x = draw_x1 + c * cell_size_mm + cell_gap_mm / 2.0
                cell_y = draw_y1 + r * cell_size_mm + cell_gap_mm / 2.0

                cell_paths: list = []

                if cell_border:
                    x0, y0 = cell_x, cell_y
                    x1b, y1b = cell_x + effective_cell_mm, cell_y + effective_cell_mm
                    cell_paths.append(
                        [(x0, y0), (x1b, y0), (x1b, y1b), (x0, y1b), (x0, y0)]
                    )

                if fill_style == "solid_hatch":
                    cell_paths.extend(
                        fill_solid_hatch(cell_x, cell_y, effective_cell_mm, density)
                    )

                if idx not in paths_by_index:
                    paths_by_index[idx] = []
                paths_by_index[idx].extend(cell_paths)

                processed += 1
                if progress_callback and processed % report_every == 0:
                    progress_callback(int(processed / total_cells * 100))

        if progress_callback:
            progress_callback(100)

        # Emit one LayerSpec per used index (sorted for determinism).
        return [
            LayerSpec(
                name=f"Pixel {idx}",
                color=hex_colors[idx],
                paths=paths_by_index[idx],
            )
            for idx in sorted(paths_by_index)
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list:
        """Fallback single-layer mode: flatten all layer paths into one list."""
        specs = self.generate_layers(params, canvas, progress_callback, cancelled_callback)
        paths: list = []
        for spec in specs:
            paths.extend(spec.paths)
        return paths
