"""Luminance / brightness band splitting for color separation."""

from __future__ import annotations

import numpy as np


# Default grayscale hex colors for brightness bands (dark → light)
_DEFAULT_COLORS = ["#000000", "#444444", "#888888", "#BBBBBB", "#FFFFFF"]


def luminance_separate(
    image: np.ndarray,
    num_bands: int = 3,
    thresholds: list[float] | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Split a grayscale or RGB image into *num_bands* brightness bands.

    Parameters
    ----------
    image:
        Input image array.  Can be grayscale (H×W or H×W×1) or RGB (H×W×3),
        dtype uint8.  RGB images are converted to grayscale internally using
        luminance-weighted coefficients.
    num_bands:
        Number of brightness bands, range 2–5.
    thresholds:
        Explicit band boundary values in [0, 255].  Should be a list of
        ``num_bands - 1`` ascending values.  When None, evenly spaced
        thresholds are computed automatically.

    Returns
    -------
    list of (binary_mask, hex_color) tuples — one per band, ordered from
    darkest to lightest.  ``binary_mask`` is a boolean array of shape (H, W).
    ``hex_color`` is a default gray assigned to that band.
    """
    num_bands = max(2, min(5, num_bands))

    # Convert to grayscale
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        # Luminance-weighted conversion
        gray = (
            0.299 * image[:, :, 0].astype(np.float32)
            + 0.587 * image[:, :, 1].astype(np.float32)
            + 0.114 * image[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if thresholds is None:
        # Evenly spaced thresholds in [0, 255]
        step = 256.0 / num_bands
        thresholds = [step * i for i in range(1, num_bands)]
    else:
        if len(thresholds) != num_bands - 1:
            raise ValueError(
                f"Expected {num_bands - 1} threshold(s), got {len(thresholds)}"
            )

    # Build band boundaries: [(low, high), ...]
    boundaries = [0.0] + list(thresholds) + [256.0]

    results: list[tuple[np.ndarray, str]] = []
    gray_f = gray.astype(np.float32)
    for i in range(num_bands):
        low = boundaries[i]
        high = boundaries[i + 1]
        if i == num_bands - 1:
            # Last band is inclusive of 255
            mask = (gray_f >= low) & (gray_f <= 255)
        else:
            mask = (gray_f >= low) & (gray_f < high)
        hex_color = _DEFAULT_COLORS[min(i, len(_DEFAULT_COLORS) - 1)]
        results.append((mask, hex_color))

    return results
