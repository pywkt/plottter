"""PixelArtGenerator — convert a source image to a per-palette-index layer grid.

Each palette colour gets its own layer, filled with hatch lines proportional to
the fill density.  The generator emits one :class:`LayerSpec` per colour index
that appears in the quantised grid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._pixel_fills import fill_cross_hatch, fill_dithered_dots, fill_solid_hatch
from plottter.generators._pixel_shapes import cell_polygon, hex_polygon
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

# Curated retro-console palette list — the natural fit for pixel art. Uses
# underscore-form keys so they round-trip cleanly through get_palette().
_RETRO_PALETTE_CHOICES: list[str] = [
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
    "cga_mode4",
    "ega",
    "endesga32",
    "sweetie16",
    "db32",
    "endesga64",
    "resurrect64",
]


def _pen_palette_choices() -> list[str]:
    """Pen-palette preset names (the same set as Color Separation / Pointillist),
    resolved at call time so newly added built-ins appear automatically."""
    from plottter.color import list_presets

    return [p.name for p in list_presets()]


def _resolve_pixel_palette(palette_group: str, retro_name: str, pen_name: str):
    """Resolve the selected palette to a pixel_art Palette object.

    The "Pen Palettes" group reuses the shared color.palettes presets (Basic 6,
    Copic 12, PaperMate InkJoy 30, …) wrapped via the same adapter the Custom
    Palette separator uses; otherwise the vendored retro-console palettes.
    """
    from plottter.pixel_art import get_palette

    if palette_group == "Pen Palettes":
        from plottter.color.palette_separator import as_pixelart_palette
        from plottter.color.palettes import get_preset

        return as_pixelart_palette(get_preset(pen_name))
    return get_palette(retro_name)


@register_generator
class PixelArtGenerator(Generator):
    """Convert a source image to a palette-indexed cell grid, one layer per colour."""

    name = "Pixel Art"
    category = "image"
    uses_source_image = True
    uses_color_source = True
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
                name="palette_group",
                label="Palette Group",
                choices=["Retro Consoles", "Pen Palettes"],
                default="Retro Consoles",
                description=(
                    "Which family of palettes to choose from. 'Retro Consoles' "
                    "are the classic game-system palettes (best suited to pixel "
                    "art); 'Pen Palettes' are the same pen sets used by Color "
                    "Separation and Pointillist (Basic 6, Copic 12, PaperMate "
                    "InkJoy 30, …)."
                ),
            ),
            ChoiceParam(
                name="palette",
                label="Palette",
                choices=_RETRO_PALETTE_CHOICES,
                default="grayscale_4",
                visible_when={"palette_group": ["Retro Consoles"]},
                description="Retro-console colour palette used to quantise the source image.",
            ),
            ChoiceParam(
                name="palette_pen",
                label="Pen Palette",
                choices=_pen_palette_choices(),
                default="Basic 6",
                visible_when={"palette_group": ["Pen Palettes"]},
                description="Pen palette used to quantise the source image — one layer per pen colour.",
            ),
            ChoiceParam(
                name="quantization",
                label="Quantization",
                choices=["nearest", "kmeans", "median_cut", "octree"],
                default="nearest",
                description="Color quantization algorithm used to map pixels to the palette.",
            ),
            ChoiceParam(
                name="color_space",
                label="Color Space",
                choices=["rgb", "lab"],
                default="rgb",
                description="Color space used for distance calculations during quantization.",
            ),
            ChoiceParam(
                name="dithering",
                label="Dithering",
                choices=["none", "floyd_steinberg", "ordered", "atkinson"],
                default="none",
                description="Dithering algorithm applied during palette quantisation.",
            ),
            ChoiceParam(
                name="cell_shape",
                label="Cell Shape",
                choices=["square", "diamond", "octagonal", "circle", "rounded_square", "hex"],
                default="square",
                description="Shape of each cell (clips the fill to the chosen geometry).",
            ),
            ChoiceParam(
                name="cell_fill_style",
                label="Cell Fill Style",
                choices=["solid_hatch", "cross_hatch", "dithered_dots", "point", "none"],
                default="solid_hatch",
                description=(
                    "Fill pattern drawn inside each cell. 'point' drops a single "
                    "pen tap at the cell centre (no back-and-forth fill) — ideal "
                    "for plotting one marker dot per cell; size it with 'Point "
                    "Diameter'. Use 'none' for outline-only rendering."
                ),
            ),
            FloatParam(
                name="point_diameter_mm",
                label="Point Diameter (mm)",
                min=0.1,
                max=10.0,
                step=0.1,
                default=1.0,
                visible_when={"cell_fill_style": ["point"]},
                description=(
                    "Marker/pen tip diameter for the 'point' fill style. The dot "
                    "stays a single pen tap (the plotter just dabs once); this "
                    "value only sets how big that dot is drawn in the preview and "
                    "SVG export. Match it to your real marker tip — e.g. 3.0 for a "
                    "3 mm marker — and set Grid Width so cells are about this size "
                    "for edge-to-edge, non-overlapping dots."
                ),
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
            BoolParam(
                name="force_grayscale",
                label="Force Grayscale",
                default=False,
                description=(
                    "Desaturate the source image before quantizing. "
                    "On colour palettes this restricts output to the "
                    "palette's gray axis (the legacy behaviour)."
                ),
            ),
            BoolParam(
                name="skip_background",
                label="Skip Background",
                default=False,
                description=(
                    "Don't plot cells that quantise to a near-white palette "
                    "colour, leaving the paper bare. Use this when your image "
                    "has a white/removed background and you don't want it filled "
                    "with the nearest pen. Tune the cut-off with 'Background "
                    "Brightness'."
                ),
            ),
            FloatParam(
                name="skip_background_threshold",
                label="Background Brightness",
                min=0.0,
                max=1.0,
                step=0.01,
                default=0.9,
                visible_when={"skip_background": [True]},
                description=(
                    "Brightness cut-off for 'Skip Background' (0 = black, "
                    "1 = white). Any palette colour at or above this brightness "
                    "is treated as background and not plotted. Lower it if some "
                    "light background tones are still being drawn; raise it if "
                    "wanted light colours are being skipped."
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "grid_width": 48,
                    "palette": "grayscale_4",
                    "dithering": "none",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.5,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Game Boy",
                params={
                    "grid_width": 40,
                    "palette": "grayscale_4",
                    "dithering": "none",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Game Boy Portrait",
                params={
                    "grid_width": 80,
                    "palette": "gameboy",
                    "dithering": "floyd_steinberg",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Game Boy Tiny",
                params={
                    "grid_width": 32,
                    "palette": "gameboy",
                    "dithering": "none",
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
                    "dithering": "none",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.8,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="B&W Halftone Dots",
                params={
                    "grid_width": 80,
                    "palette": "grayscale_2",
                    "dithering": "none",
                    "cell_shape": "circle",
                    "cell_fill_style": "dithered_dots",
                    "fill_density": 0.8,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Diamond Dither",
                params={
                    "grid_width": 64,
                    "palette": "grayscale_4",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "diamond",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.55,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Hex Honeycomb",
                params={
                    "grid_width": 32,
                    "palette": "gameboy",
                    "dithering": "none",
                    "cell_shape": "hex",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.6,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="NES Color",
                params={
                    "grid_width": 40,
                    "palette": "nes",
                    "dithering": "none",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.55,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="NES Crosshatched",
                params={
                    "grid_width": 36,
                    "palette": "nes",
                    "dithering": "ordered",
                    "cell_shape": "square",
                    "cell_fill_style": "cross_hatch",
                    "fill_density": 0.30,
                    "cell_border": False,
                    "cell_gap_mm": 0.1,
                },
            ),
            Preset(
                name="SNES Detail",
                params={
                    "grid_width": 80,
                    "palette": "sweetie16",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Genesis Detail",
                params={
                    "grid_width": 80,
                    "palette": "endesga32",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="PICO-8 Sketch",
                params={
                    "grid_width": 64,
                    "palette": "pico8",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.65,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="C64 Classic",
                params={
                    "grid_width": 40,
                    "palette": "c64",
                    "dithering": "none",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.7,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="CGA 4-Color",
                params={
                    "grid_width": 32,
                    "palette": "cga_mode4",
                    "dithering": "ordered",
                    "cell_shape": "square",
                    "cell_fill_style": "cross_hatch",
                    "fill_density": 0.45,
                    "cell_border": False,
                    "cell_gap_mm": 0.1,
                },
            ),
            Preset(
                name="EGA 16-Color",
                params={
                    "grid_width": 56,
                    "palette": "ega",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.6,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Endesga Modern",
                params={
                    "grid_width": 64,
                    "palette": "endesga32",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.65,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Sweetie 16 Sketch",
                params={
                    "grid_width": 48,
                    "palette": "sweetie16",
                    "dithering": "atkinson",
                    "cell_shape": "square",
                    "cell_fill_style": "cross_hatch",
                    "fill_density": 0.35,
                    "cell_border": True,
                    "cell_gap_mm": 0.1,
                },
            ),
            Preset(
                name="B&W Hatch",
                params={
                    "grid_width": 64,
                    "palette": "grayscale_2",
                    "dithering": "none",
                    "cell_shape": "square",
                    "cell_fill_style": "cross_hatch",
                    "fill_density": 0.8,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Grayscale Fine",
                params={
                    "grid_width": 96,
                    "palette": "grayscale_16",
                    "dithering": "floyd_steinberg",
                    "cell_shape": "square",
                    "cell_fill_style": "solid_hatch",
                    "fill_density": 0.6,
                    "cell_border": False,
                    "cell_gap_mm": 0.0,
                },
            ),
            Preset(
                name="Outline Only",
                params={
                    "grid_width": 48,
                    "palette": "grayscale_4",
                    "dithering": "none",
                    "cell_shape": "square",
                    "cell_fill_style": "none",
                    "fill_density": 0.0,
                    "cell_border": True,
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
        from plottter.pixel_art import image_to_palette_grid

        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        grid_width = int(params.get("grid_width", 32))
        palette_group = str(params.get("palette_group", "Retro Consoles"))
        palette_name = str(params.get("palette", "grayscale_4"))
        pen_palette_name = str(params.get("palette_pen", "Basic 6"))
        quantization = str(params.get("quantization", "nearest"))
        color_space = str(params.get("color_space", "rgb"))
        dithering = str(params.get("dithering", "none"))
        cell_shape = str(params.get("cell_shape", "square"))
        fill_style = str(params.get("cell_fill_style", "solid_hatch"))
        density = float(params.get("fill_density", 0.7))
        point_diameter_mm = float(params.get("point_diameter_mm", 1.0))
        cell_border = bool(params.get("cell_border", False))
        cell_gap_mm = float(params.get("cell_gap_mm", 0.0))
        force_grayscale = bool(params.get("force_grayscale", False))
        skip_background = bool(params.get("skip_background", False))
        skip_bg_threshold = float(params.get("skip_background_threshold", 0.9))

        # The hex path samples source[py, px, :3], so it needs 3 channels even
        # in the grayscale case.  Broadcast the luminance back to RGB.
        if force_grayscale and source.ndim == 3:
            from plottter.io.image_import import to_grayscale
            gray = to_grayscale(source)
            source = np.repeat(gray[:, :, None], 3, axis=2)

        palette = _resolve_pixel_palette(palette_group, palette_name, pen_palette_name)

        if cell_shape == "hex":
            return self._generate_hex_layers(
                source=source,
                grid_width=grid_width,
                palette=palette,
                fill_style=fill_style,
                density=density,
                point_diameter_mm=point_diameter_mm,
                skip_background=skip_background,
                skip_bg_threshold=skip_bg_threshold,
                cell_border=cell_border,
                cell_gap_mm=cell_gap_mm,
                canvas=canvas,
                params=params,
                progress_callback=progress_callback,
                cancelled_callback=cancelled_callback,
            )

        indices = image_to_palette_grid(
            source,
            palette,
            grid_width,
            quantization=quantization,
            color_space=color_space,
            dithering=dithering,
        )

        from plottter.generators._helpers import compute_image_rect

        n_rows, n_cols = indices.shape
        img_h, img_w = source.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fit")),
            img_w,
            img_h,
            draw_x1,
            draw_y1,
            draw_x2,
            draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        # In fit/custom mode the rect preserves image aspect, so rect_w/n_cols
        # equals rect_h/n_rows and cells are square. In fill mode rect == drawing
        # area and the rect_w-derived cell size matches the previous behaviour.
        rect_w = img_x2 - img_x1
        cell_size_mm = rect_w / n_cols
        effective_cell_mm = max(0.01, cell_size_mm - cell_gap_mm)

        hex_colors = palette.to_hex_list()
        n_colors = len(hex_colors)

        # Pre-compute perceptual brightness [0, 1] for each palette colour so
        # darker colours receive denser fill and lighter colours receive sparser
        # fill, making different shades visually distinguishable when rendered.
        palette_colors = palette.colors
        color_brightness: list[float] = []
        for pr, pg, pb in palette_colors:
            lum = (0.2126 * pr + 0.7152 * pg + 0.0722 * pb) / 255.0
            color_brightness.append(max(0.0, min(1.0, lum)))

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

                # Skip near-white (background) cells when requested, leaving the
                # paper bare instead of filling it with the nearest pen.
                if skip_background and color_brightness[idx] >= skip_bg_threshold:
                    processed += 1
                    continue

                # Top-left corner of this cell's fill area.
                cell_x = img_x1 + c * cell_size_mm + cell_gap_mm / 2.0
                cell_y = img_y1 + r * cell_size_mm + cell_gap_mm / 2.0

                # Scale density by darkness: black → full density, white → 0.
                cell_density = density * (1.0 - color_brightness[idx])

                cell_paths: list = []

                # A 'point' cell is a single pen tap at the cell centre — no fill
                # geometry and no clip polygon needed.
                if fill_style == "point":
                    px_c = cell_x + effective_cell_mm / 2.0
                    py_c = cell_y + effective_cell_mm / 2.0
                    cell_paths.append([(px_c, py_c), (px_c + 0.01, py_c)])
                    paths_by_index.setdefault(idx, []).extend(cell_paths)
                    processed += 1
                    if progress_callback and processed % report_every == 0:
                        progress_callback(int(processed / total_cells * 100))
                    continue

                # Build the clip polygon once per cell (None for square).
                poly_verts = cell_polygon(cell_shape, cell_x, cell_y, effective_cell_mm)
                if poly_verts is not None:
                    from shapely.geometry import Polygon as _ShapelyPolygon  # lazy
                    poly = _ShapelyPolygon(poly_verts)
                else:
                    poly = None

                if cell_border:
                    x0, y0 = cell_x, cell_y
                    x1b, y1b = cell_x + effective_cell_mm, cell_y + effective_cell_mm
                    cell_paths.append(
                        [(x0, y0), (x1b, y0), (x1b, y1b), (x0, y1b), (x0, y0)]
                    )

                if fill_style == "solid_hatch":
                    cell_paths.extend(
                        fill_solid_hatch(cell_x, cell_y, effective_cell_mm, cell_density, polygon=poly)
                    )
                elif fill_style == "cross_hatch":
                    cell_paths.extend(
                        fill_cross_hatch(cell_x, cell_y, effective_cell_mm, cell_density, polygon=poly)
                    )
                elif fill_style == "dithered_dots":
                    cell_paths.extend(
                        fill_dithered_dots(cell_x, cell_y, effective_cell_mm, cell_density, polygon=poly)
                    )

                if idx not in paths_by_index:
                    paths_by_index[idx] = []
                paths_by_index[idx].extend(cell_paths)

                processed += 1
                if progress_callback and processed % report_every == 0:
                    progress_callback(int(processed / total_cells * 100))

        if progress_callback:
            progress_callback(100)

        # Single-tap 'point' cells carry no size in their geometry, so attach the
        # marker diameter as a render hint: the canvas preview and SVG export use
        # it (round caps) to show each dot at its true size.
        spec_info = {"dot_diameter_mm": point_diameter_mm} if fill_style == "point" else None

        # Emit one LayerSpec per used index (sorted for determinism).
        return [
            LayerSpec(
                name=f"Pixel {idx}",
                color=hex_colors[idx],
                paths=paths_by_index[idx],
                generator_info=spec_info,
            )
            for idx in sorted(paths_by_index)
        ]

    def _generate_hex_layers(
        self,
        source: np.ndarray,
        grid_width: int,
        palette: Any,
        fill_style: str,
        density: float,
        point_diameter_mm: float,
        skip_background: bool,
        skip_bg_threshold: float,
        cell_border: bool,
        cell_gap_mm: float,
        canvas: Canvas,
        params: dict[str, Any],
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Hex grid path: sample the source image at flat-topped hexagon centres.

        Circumradius *s* is derived from canvas width and *grid_width* columns so
        that exactly *grid_width* hexes span the drawing area horizontally.
        Odd-numbered columns are offset downward by half a row height (even-q
        offset layout).
        """
        from shapely.geometry import Polygon as _ShapelyPolygon  # lazy

        from plottter.generators._helpers import compute_image_rect

        img_h_px, img_w_px = source.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fit")),
            img_w_px,
            img_h_px,
            draw_x1,
            draw_y1,
            draw_x2,
            draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )
        rect_w = img_x2 - img_x1
        rect_h = img_y2 - img_y1

        # Circumradius: grid_width flat-top hexes spanning rect_w.
        # Total x-extent = s*(2 + 1.5*(grid_width-1)) = s*(0.5 + 1.5*grid_width)
        s = rect_w / (0.5 + 1.5 * grid_width)
        hex_h = s * math.sqrt(3)  # centre-to-centre row spacing

        palette_colors = palette.colors  # list of (R, G, B) tuples
        hex_colors = palette.to_hex_list()
        n_colors = len(hex_colors)

        # Perceptual brightness per colour (0=black → dense fill, 1=white → sparse).
        color_brightness: list[float] = []
        for pr, pg, pb in palette_colors:
            lum = (0.2126 * pr + 0.7152 * pg + 0.0722 * pb) / 255.0
            color_brightness.append(max(0.0, min(1.0, lum)))

        img_h, img_w = source.shape[:2]

        # Build all hex centres using even-q offset layout, anchored within the
        # fitted image rect (so map preserves source aspect ratio).
        hex_centers: list[tuple[float, float]] = []
        for q in range(grid_width):
            cx = img_x1 + s + q * 1.5 * s
            y_offset = hex_h / 2.0 if (q % 2 == 1) else 0.0
            r = 0
            while True:
                cy = img_y1 + hex_h / 2.0 + y_offset + r * hex_h
                if cy > img_y2 + hex_h / 2.0:
                    break
                hex_centers.append((cx, cy))
                r += 1

        paths_by_index: dict[int, list] = {}
        total_cells = len(hex_centers)
        report_every = max(1, total_cells // 20)
        processed = 0

        for cx, cy in hex_centers:
            if cancelled_callback and cancelled_callback():
                break

            # Sample source image at hex centre (nearest-neighbour).
            px = int((cx - img_x1) / rect_w * img_w)
            py = int((cy - img_y1) / rect_h * img_h)
            px = max(0, min(img_w - 1, px))
            py = max(0, min(img_h - 1, py))
            pixel = source[py, px, :3]

            # Nearest palette colour.
            best_idx = 0
            best_dist = float("inf")
            for j, (pr, pg, pb) in enumerate(palette_colors):
                d = (int(pixel[0]) - pr) ** 2 + (int(pixel[1]) - pg) ** 2 + (int(pixel[2]) - pb) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            idx = best_idx
            if idx < 0 or idx >= n_colors:
                processed += 1
                continue

            # Skip near-white (background) cells when requested, leaving the
            # paper bare instead of filling it with the nearest pen.
            if skip_background and color_brightness[idx] >= skip_bg_threshold:
                processed += 1
                continue

            # A 'point' cell is a single pen tap at the hex centre — no fill
            # geometry and no clip polygon needed.
            if fill_style == "point":
                paths_by_index.setdefault(idx, []).append([(cx, cy), (cx + 0.01, cy)])
                processed += 1
                if progress_callback and processed % report_every == 0:
                    progress_callback(int(processed / total_cells * 100))
                continue

            # Apply gap by shrinking the circumradius.
            effective_s = max(0.01, s - cell_gap_mm / 2.0)
            eff_hex_h = effective_s * math.sqrt(3)

            hex_verts = hex_polygon(cx, cy, effective_s)
            poly = _ShapelyPolygon(hex_verts)

            cell_density = density * (1.0 - color_brightness[idx])
            cell_paths: list = []

            if cell_border:
                # Draw the hex outline as a closed polyline.
                cell_paths.append(hex_verts + [hex_verts[0]])

            # Fill bounding box covers the full hex width/height; polygon clips it.
            box_x = cx - effective_s
            box_y = cy - eff_hex_h / 2.0
            box_size = 2.0 * effective_s

            if fill_style == "solid_hatch":
                cell_paths.extend(fill_solid_hatch(box_x, box_y, box_size, cell_density, polygon=poly))
            elif fill_style == "cross_hatch":
                cell_paths.extend(fill_cross_hatch(box_x, box_y, box_size, cell_density, polygon=poly))
            elif fill_style == "dithered_dots":
                cell_paths.extend(fill_dithered_dots(box_x, box_y, box_size, cell_density, polygon=poly))

            if cell_paths:
                if idx not in paths_by_index:
                    paths_by_index[idx] = []
                paths_by_index[idx].extend(cell_paths)

            processed += 1
            if progress_callback and processed % report_every == 0:
                progress_callback(int(processed / total_cells * 100))

        if progress_callback:
            progress_callback(100)

        spec_info = {"dot_diameter_mm": point_diameter_mm} if fill_style == "point" else None

        return [
            LayerSpec(
                name=f"Pixel {idx}",
                color=hex_colors[idx],
                paths=paths_by_index[idx],
                generator_info=spec_info,
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
