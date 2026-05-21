"""image_to_palette_grid — convert a source image to a 2D array of palette indices."""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from plottter.pixel_art.converter import ConversionOptions, PixelArtConverter
from plottter.pixel_art.dithering import DitherMethod
from plottter.pixel_art.palettes.palette import Palette
from plottter.pixel_art.quantizer import ColorSpace, QuantizeMethod
from plottter.pixel_art.transparency import AlphaHandling

_QUANTIZE_MAP = {
    "nearest": QuantizeMethod.NEAREST,
    "kmeans": QuantizeMethod.KMEANS,
    "median_cut": QuantizeMethod.MEDIAN_CUT,
    "octree": QuantizeMethod.OCTREE,
}

_COLOR_SPACE_MAP = {
    "rgb": ColorSpace.RGB,
    "lab": ColorSpace.LAB,
}

_DITHER_MAP = {
    "none": DitherMethod.NONE,
    "floyd_steinberg": DitherMethod.FLOYD_STEINBERG,
    "ordered": DitherMethod.ORDERED,
    "atkinson": DitherMethod.ATKINSON,
}

_ALPHA_MAP = {
    "preserve": AlphaHandling.PRESERVE,
    "white": AlphaHandling.REMOVE,
    "skip": AlphaHandling.PRESERVE,  # preserve alpha so we can mark -1 afterward
}


def image_to_palette_grid(
    image: Union[np.ndarray, Image.Image],
    palette: Palette,
    grid_width: int,
    grid_height: int | None = None,
    *,
    quantization: str = "nearest",
    color_space: str = "rgb",
    dithering: str = "none",
    transparency: str = "preserve",
) -> np.ndarray:
    """Convert a source image to a 2D ndarray of palette indices.

    Parameters
    ----------
    image:
        Source image as a PIL Image or numpy array (H×W×3 or H×W×4, uint8).
    palette:
        Target palette.
    grid_width:
        Number of columns in the output grid.
    grid_height:
        Number of rows. None → preserves source aspect ratio.
    quantization:
        Color quantization algorithm: "nearest" | "kmeans" | "median_cut" | "octree".
    color_space:
        Color space for distance calculations: "rgb" | "lab".
    dithering:
        Dithering algorithm: "none" | "floyd_steinberg" | "ordered" | "atkinson".
    transparency:
        How to handle transparency: "preserve" | "white" | "skip".
        "skip" returns -1 for transparent cells.

    Returns
    -------
    indices : ndarray[int32], shape (rows, cols)
        ``indices[r, c]`` is the palette index of the cell at row r, column c.
        -1 indicates a transparent cell (when transparency="skip").
    """
    # --- 1. Coerce input to PIL Image ---
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            pil_image: Image.Image = Image.fromarray(image, "RGBA")
        else:
            pil_image = Image.fromarray(image, "RGB")
    else:
        pil_image = image

    if pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGBA" if "A" in pil_image.mode else "RGB")

    src_w, src_h = pil_image.size

    # --- 2. Compute target height if not given ---
    if grid_height is None:
        grid_height = max(1, round(grid_width * src_h / src_w))

    # --- 3. Validate string args ---
    quantization_lower = quantization.lower()
    if quantization_lower not in _QUANTIZE_MAP:
        raise ValueError(
            f"Unknown quantization '{quantization}'. Valid: {list(_QUANTIZE_MAP)}"
        )
    color_space_lower = color_space.lower()
    if color_space_lower not in _COLOR_SPACE_MAP:
        raise ValueError(
            f"Unknown color_space '{color_space}'. Valid: {list(_COLOR_SPACE_MAP)}"
        )
    dithering_lower = dithering.lower()
    if dithering_lower not in _DITHER_MAP:
        raise ValueError(
            f"Unknown dithering '{dithering}'. Valid: {list(_DITHER_MAP)}"
        )
    transparency_lower = transparency.lower()
    if transparency_lower not in _ALPHA_MAP:
        raise ValueError(
            f"Unknown transparency '{transparency}'. Valid: {list(_ALPHA_MAP)}"
        )

    # --- 4. Build ConversionOptions ---
    alpha_handling = _ALPHA_MAP[transparency_lower]
    background = (255, 255, 255)

    options = ConversionOptions(
        target_width=grid_width,
        target_height=grid_height,
        maintain_aspect=False,
        quantize_method=_QUANTIZE_MAP[quantization_lower],
        color_space=_COLOR_SPACE_MAP[color_space_lower],
        dither_method=_DITHER_MAP[dithering_lower],
        alpha_handling=alpha_handling,
        background_color=background,
        allow_oversized=True,
    )

    # --- 5. Run converter ---
    converter = PixelArtConverter(palette)
    result = converter.convert_image(pil_image, options)
    out_image = result.image

    # --- 6. Build a fast colour→index lookup ---
    palette_colors = palette.colors  # list of (R, G, B)
    color_to_index = {rgb: idx for idx, rgb in enumerate(palette_colors)}

    # --- 7. Extract pixels and build ndarray ---
    # Ensure we work in RGB mode for index lookups
    if out_image.mode == "RGBA":
        # Keep alpha separately for transparency="skip"
        r_ch, g_ch, b_ch, a_ch = out_image.split()
        rgb_image = Image.merge("RGB", (r_ch, g_ch, b_ch))
        alpha_arr = np.array(a_ch, dtype=np.uint8)
    else:
        rgb_image = out_image.convert("RGB") if out_image.mode != "RGB" else out_image
        alpha_arr = None

    pixels = np.array(rgb_image, dtype=np.uint8)  # shape (rows, cols, 3)
    rows, cols, _ = pixels.shape

    indices = np.empty((rows, cols), dtype=np.int32)

    # Vectorised lookup: reshape to (N, 3) and map each pixel
    flat = pixels.reshape(-1, 3)
    for i, (r, g, b) in enumerate(flat):
        rgb_tuple = (int(r), int(g), int(b))
        idx = color_to_index.get(rgb_tuple)
        if idx is None:
            # Nearest fallback (should not normally happen with exact-match palettes)
            best = 0
            best_dist = float("inf")
            for j, (pr, pg, pb) in enumerate(palette_colors):
                dist = (int(r) - pr) ** 2 + (int(g) - pg) ** 2 + (int(b) - pb) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = j
            idx = best
        indices.flat[i] = idx

    # --- 8. Apply transparency="skip" → mark transparent cells as -1 ---
    if transparency_lower == "skip" and alpha_arr is not None:
        indices[alpha_arr == 0] = -1

    return indices
