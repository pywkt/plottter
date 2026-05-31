"""CRT TV Generator — imitate cathode-ray-tube visual artefacts with pen marks.

Each palette pen gets a fixed position inside a per-pixel triad cell
(shadow-mask, aperture-grille, or slot-mask layout), with optional
scanline, vignette, and barrel-distortion effects.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from plottter.generators import register_generator
from plottter.generators.base import (
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
class CrtTvGenerator(Generator):
    """Render a colour image as CRT-style subpixel triads, scanlines, and vignette.

    Each palette entry produces one output layer.  Sub-pixel positions follow
    a configurable mask geometry (shadow_mask, aperture_grille, slot_mask).
    """

    name = "CRT TV"
    category = "image"
    uses_source_image = True
    uses_color_source = True
    emits_multiple_layers = True

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

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
                    "Each palette entry produces one subpixel layer."
                ),
            ),
            IntParam(
                name="crt_resolution_w",
                label="CRT Width (pixels)",
                min=40,
                max=800,
                step=1,
                default=160,
                randomizable=False,
                description=(
                    "Output pixel grid width in CRT pixels.  Image is downsampled "
                    "to this width; height is derived from the source aspect ratio.  "
                    "Recommended: 80–320 for A4."
                ),
            ),
            ChoiceParam(
                name="mask_type",
                label="Mask Type",
                choices=["shadow_mask", "aperture_grille", "slot_mask"],
                default="shadow_mask",
                randomizable=False,
                description=(
                    "Sub-pixel layout geometry.  shadow_mask places pens in a "
                    "triangular triad (classic CRT dot pattern).  "
                    "aperture_grille uses vertical stripes (Sony Trinitron "
                    "style).  slot_mask offsets stripes vertically per row "
                    "(brick pattern)."
                ),
            ),
            ChoiceParam(
                name="subpixel_shape",
                label="Subpixel Shape",
                choices=["circle", "cross", "point"],
                default="circle",
                randomizable=False,
                description=(
                    "Shape drawn at each subpixel position.  Same vocabulary as "
                    "PointillistGenerator."
                ),
            ),
            FloatParam(
                name="subpixel_size_mm",
                label="Subpixel Size (mm)",
                min=0.05,
                max=2.0,
                step=0.05,
                default=0.3,
                randomizable=False,
                description=(
                    "Subpixel mark size in mm.  Should be < cell_size / n_pens "
                    "to avoid overlap between adjacent subpixels."
                ),
            ),
            ChoiceParam(
                name="dither",
                label="Dithering",
                choices=["none", "floyd-steinberg", "ordered", "atkinson"],
                default="floyd-steinberg",
                randomizable=False,
                description=(
                    "Dithering passed to palette_separate.  Floyd-Steinberg is "
                    "recommended for soft gradient reproduction."
                ),
            ),
            FloatParam(
                name="scanline_intensity",
                label="Scanline Intensity",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                randomizable=False,
                description=(
                    "Strength of scanline darkening.  0 = no scanlines, "
                    "1 = fully drop targeted rows."
                ),
            ),
            IntParam(
                name="scanline_period",
                label="Scanline Period",
                min=1,
                max=5,
                step=1,
                default=2,
                randomizable=False,
                description=(
                    "Apply scanline darkening every Nth row.  "
                    "2 = classic every-other-row scanline."
                ),
            ),
            FloatParam(
                name="vignette_strength",
                label="Vignette Strength",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.3,
                randomizable=False,
                description=(
                    "Corner darkening.  0 = no vignette, "
                    "1 = full black corners."
                ),
            ),
            FloatParam(
                name="barrel_strength",
                label="Barrel Strength",
                min=0.0,
                max=0.15,
                step=0.01,
                default=0.0,
                randomizable=False,
                description=(
                    "Barrel distortion of the dot grid.  Hard-capped at 0.15 "
                    "because pen plots distort badly under aggressive warps."
                ),
            ),
            FloatParam(
                name="gamma",
                label="Gamma",
                min=0.4,
                max=2.5,
                step=0.05,
                default=1.0,
                randomizable=False,
                description=(
                    "Gamma curve applied before quantisation.  "
                    "> 1 darkens midtones, < 1 lightens."
                ),
            ),
            IntParam(
                name="seed",
                label="Seed",
                min=0,
                max=99999,
                step=1,
                default=0,
                randomizable=False,
                description=(
                    "Random seed for scanline / vignette keep tests.  "
                    "Same seed → deterministic output."
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="NES",
                params={
                    "palette": "Basic 6",
                    "crt_resolution_w": 256,
                    "mask_type": "shadow_mask",
                    "subpixel_shape": "circle",
                    "subpixel_size_mm": 0.25,
                    "dither": "floyd-steinberg",
                    "scanline_intensity": 0.8,
                    "scanline_period": 2,
                    "vignette_strength": 0.2,
                    "barrel_strength": 0.0,
                    "gamma": 1.0,
                    "seed": 0,
                },
            ),
            Preset(
                name="Trinitron",
                params={
                    "palette": "Basic 6",
                    "crt_resolution_w": 320,
                    "mask_type": "aperture_grille",
                    "subpixel_shape": "circle",
                    "subpixel_size_mm": 0.20,
                    "dither": "floyd-steinberg",
                    "scanline_intensity": 0.4,
                    "scanline_period": 2,
                    "vignette_strength": 0.1,
                    "barrel_strength": 0.0,
                    "gamma": 1.0,
                    "seed": 0,
                },
            ),
            Preset(
                name="VGA Monitor",
                params={
                    "palette": "Basic 6",
                    "crt_resolution_w": 320,
                    "mask_type": "slot_mask",
                    "subpixel_shape": "circle",
                    "subpixel_size_mm": 0.25,
                    "dither": "floyd-steinberg",
                    "scanline_intensity": 0.5,
                    "scanline_period": 2,
                    "vignette_strength": 0.2,
                    "barrel_strength": 0.05,
                    "gamma": 1.0,
                    "seed": 0,
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
        """Return one LayerSpec per non-empty pen after CRT processing."""
        from plottter.color import get_preset, palette_separate
        from plottter.generators._crt_core import (
            barrel_warp,
            scanline_mask,
            subpixel_layout,
            vignette_mask,
        )
        from plottter.generators._helpers import compute_image_rect
        from plottter.generators._pointillist_core import render_dots

        image: np.ndarray | None = params.get("_source_image")
        if image is None:
            return []

        # ------------------------------------------------------------------
        # 1. Preprocess — apply gamma to the source image
        # ------------------------------------------------------------------
        gamma = float(params.get("gamma", 1.0))
        if gamma != 1.0:
            lut = (
                np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0
            ).clip(0, 255).astype(np.uint8)
            image = lut[image]

        img_h, img_w = image.shape[:2]

        # ------------------------------------------------------------------
        # 2. Downsample to CRT resolution (height derived from source aspect)
        # ------------------------------------------------------------------
        crt_w = int(params.get("crt_resolution_w", 160))
        crt_h = max(1, round(img_h * crt_w / img_w))
        pil_img = Image.fromarray(image, "RGB")
        pil_small = pil_img.resize((crt_w, crt_h), Image.LANCZOS)
        small_image = np.array(pil_small, dtype=np.uint8)

        # ------------------------------------------------------------------
        # 3. Compute fitted image rect (honouring image_fit_mode — critical to
        #    avoid the PixelArt/Pointillist aspect-ratio bug).
        # ------------------------------------------------------------------
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
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
        rect_x1, rect_y1, rect_x2, rect_y2 = img_rect

        # Cell sizes in mm — one cell = one CRT pixel (may be non-square in fill mode)
        cell_size_x = (rect_x2 - rect_x1) / crt_w
        cell_size_y = (rect_y2 - rect_y1) / crt_h

        # ------------------------------------------------------------------
        # 4. Palette separation on the downsampled image
        # ------------------------------------------------------------------
        palette_name = str(params.get("palette", "Basic 6"))
        palette = get_preset(palette_name)
        dither = str(params.get("dither", "floyd-steinberg"))
        masks = palette_separate(small_image, palette, dither=dither)
        n_pens = len(masks)

        # ------------------------------------------------------------------
        # 5. Scanline + vignette → combined keep-probability map
        # ------------------------------------------------------------------
        scanline_intensity = float(params.get("scanline_intensity", 0.7))
        scanline_period = int(params.get("scanline_period", 2))
        vignette_strength = float(params.get("vignette_strength", 0.3))

        sl_mask = scanline_mask(crt_h, crt_w, scanline_intensity, scanline_period)
        vig_mask = vignette_mask(crt_h, crt_w, vignette_strength)
        combined_mask = (sl_mask * vig_mask).astype(np.float64)

        seed = int(params.get("seed", 0))
        rng = np.random.default_rng(seed)
        rand_vals = rng.random((crt_h, crt_w))

        # keep_map[r, c] = True iff this pixel survives the scanline+vignette test
        keep_map: np.ndarray = rand_vals <= combined_mask

        # ------------------------------------------------------------------
        # 6. Per-pen subpixel emission
        # ------------------------------------------------------------------
        mask_type = str(params.get("mask_type", "shadow_mask"))
        subpixel_shape = str(params.get("subpixel_shape", "circle"))
        subpixel_size_mm = float(params.get("subpixel_size_mm", 0.3))
        barrel_strength = float(params.get("barrel_strength", 0.0))

        # Barrel warp geometry parameters
        centre_mm = (
            (rect_x1 + rect_x2) / 2.0,
            (rect_y1 + rect_y2) / 2.0,
        )
        diag_mm = math.sqrt(
            (rect_x2 - rect_x1) ** 2 + (rect_y2 - rect_y1) ** 2
        )
        max_radius_mm = diag_mm / 2.0 if diag_mm > 0 else 1.0

        layer_specs: list[LayerSpec] = []

        for i, (pen_mask, hex_color) in enumerate(masks):
            # Pixels assigned to this pen that also survive the keep test
            active: np.ndarray = (pen_mask == 255) & keep_map
            rows_arr, cols_arr = np.where(active)
            if len(rows_arr) == 0:
                continue

            # Compute subpixel mm coordinates (vectorised per-pen)
            if mask_type == "slot_mask":
                # y-position depends on row parity; x is uniform
                x_frac = (i + 0.5) / n_pens
                y_fracs = np.where(rows_arr % 2 == 0, 0.35, 0.65)
                cx_mm = rect_x1 + cols_arr * cell_size_x + x_frac * cell_size_x
                cy_mm = rect_y1 + rows_arr * cell_size_y + y_fracs * cell_size_y
            else:
                # shadow_mask and aperture_grille: same offset for all rows
                x_frac, y_frac = subpixel_layout(mask_type, i, n_pens)
                cx_mm = rect_x1 + cols_arr * cell_size_x + x_frac * cell_size_x
                cy_mm = rect_y1 + rows_arr * cell_size_y + y_frac * cell_size_y

            coords_mm = np.stack([cx_mm, cy_mm], axis=1)

            # Apply barrel distortion (no-op when barrel_strength == 0)
            if barrel_strength > 0.0:
                coords_mm = barrel_warp(
                    coords_mm, centre_mm, barrel_strength, max_radius_mm
                )

            paths = render_dots(
                coords_mm, style=subpixel_shape, size_mm=subpixel_size_mm
            )

            if paths:
                layer_specs.append(
                    LayerSpec(
                        name=f"Pen {i + 1} ({hex_color})",
                        color=hex_color,
                        paths=paths,
                    )
                )

            if progress_callback:
                progress_callback((i + 1) / n_pens)
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
        specs = self.generate_layers(
            params, canvas, progress_callback, cancelled_callback
        )
        paths: list[Polyline] = []
        for spec in specs:
            paths.extend(spec.paths)
        return paths
