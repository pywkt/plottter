"""palette_separate() — map an RGB image to per-pen binary masks."""
from __future__ import annotations

import numpy as np
from PIL import Image

from plottter.color.palette import PenPalette


def _as_pixelart_palette(pp: PenPalette):
    """Wrap a PenPalette so pixel_art's quantize_to_palette accepts it."""
    from plottter.pixel_art.palette import Palette, PaletteMetadata

    rgb_tuples = [
        (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        for c in pp.colors
    ]

    class _Adapter(Palette):
        @property
        def colors(self):
            return rgb_tuples

        @property
        def metadata(self):
            return PaletteMetadata(
                name=pp.name,
                system="Custom",
                color_count=len(rgb_tuples),
            )

    return _Adapter()


def palette_separate(
    image: np.ndarray,
    palette: PenPalette,
    dither: str = "none",
    color_space: str = "lab",
) -> list[tuple[np.ndarray, str]]:
    """Separate an RGB image into one binary mask per palette colour.

    Parameters
    ----------
    image:
        RGB image, shape (H, W, 3), dtype uint8.
    palette:
        PenPalette with the target pen colours.
    dither:
        One of "none", "floyd-steinberg", "ordered", "atkinson". Maps to
        DitherMethod values in plottter.pixel_art.dithering.
    color_space:
        "lab" (default, recommended — perceptually uniform) or "rgb"
        (faster, less perceptually accurate). Passed to ColorSpace.

    Returns
    -------
    list of (mask, hex) tuples in palette order. Each mask is uint8
    (H, W) where 255 indicates "this pixel was assigned to this colour"
    and 0 indicates "not this colour". The masks are mutually exclusive
    (each pixel is in exactly one mask) and exhaustive (every pixel is
    in some mask), so `sum_over_masks == 255` everywhere. hex matches
    palette.colors entry-for-entry.
    """
    from plottter.pixel_art.dithering import DitherMethod, DitherOptions, apply_dithering
    from plottter.pixel_art.quantizer import ColorSpace, QuantizeMethod, quantize_to_palette

    # Validate inputs.
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be uint8 RGB (H, W, 3), got shape={image.shape} dtype={image.dtype}"
        )
    if not palette.colors:
        raise ValueError("palette must have at least one colour")

    # Map color_space string to enum.
    cs_map = {"lab": ColorSpace.LAB, "rgb": ColorSpace.RGB}
    if color_space not in cs_map:
        raise ValueError(f"color_space must be 'lab' or 'rgb', got {color_space!r}")
    cs = cs_map[color_space]

    # Map dither string to enum.
    dither_map = {
        "none": DitherMethod.NONE,
        "floyd-steinberg": DitherMethod.FLOYD_STEINBERG,
        "ordered": DitherMethod.ORDERED,
        "atkinson": DitherMethod.ATKINSON,
    }
    if dither not in dither_map:
        raise ValueError(
            f"dither must be one of {list(dither_map)}, got {dither!r}"
        )

    pil_img = Image.fromarray(image, "RGB")
    adapter = _as_pixelart_palette(palette)

    if dither != "none":
        options = DitherOptions(method=dither_map[dither], strength=1.0)
        quantized_pil = apply_dithering(pil_img, adapter, options)
        # apply_dithering may return RGBA if preserve_alpha, strip to RGB.
        if quantized_pil.mode != "RGB":
            quantized_pil = quantized_pil.convert("RGB")
    else:
        quantized_pil = quantize_to_palette(
            pil_img, adapter, method=QuantizeMethod.NEAREST, color_space=cs
        )
        if quantized_pil.mode != "RGB":
            quantized_pil = quantized_pil.convert("RGB")

    quant = np.array(quantized_pil, dtype=np.uint8)  # (H, W, 3)

    masks: list[tuple[np.ndarray, str]] = []
    for hex_color in palette.colors:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        match = (quant[:, :, 0] == r) & (quant[:, :, 1] == g) & (quant[:, :, 2] == b)
        mask = np.where(match, np.uint8(255), np.uint8(0))
        masks.append((mask, hex_color))

    return masks
