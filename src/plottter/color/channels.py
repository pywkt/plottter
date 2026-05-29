"""RGB and CMYK channel separation for color separation."""

from __future__ import annotations

import numpy as np


def rgb_separate(image: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Split an RGB image into Red, Green, and Blue grayscale intensity channels.

    Parameters
    ----------
    image:
        RGB image array of shape (H, W, 3), dtype uint8.

    Returns
    -------
    list of 3 ``(grayscale_image, hex_color)`` tuples in the order
    (Red, Green, Blue).  Each ``grayscale_image`` is an (H, W) uint8 array
    representing that channel's intensity; ``hex_color`` is the standard
    pure-channel color ("#FF0000", "#00FF00", "#0000FF").
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H×W×3), got shape {image.shape}")

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    return [
        (r, "#FF0000"),
        (g, "#00FF00"),
        (b, "#0000FF"),
    ]


def cmyk_separate(
    image: np.ndarray,
    k_amount: float = 1.0,
) -> list[tuple[np.ndarray, str]]:
    """Separate an RGB image into Cyan, Magenta, Yellow, and Key (Black) channels.

    The conversion follows the standard formula:
    - K  = 1 - max(R, G, B)
    - C  = (1 - R - K) / (1 - K),  0 when K = 1
    - M  = (1 - G - K) / (1 - K),  0 when K = 1
    - Y  = (1 - B - K) / (1 - K),  0 when K = 1

    Parameters
    ----------
    image:
        RGB image array of shape (H, W, 3), dtype uint8.
    k_amount:
        Scaling factor applied to the K (black) channel before quantisation.
        Range ``[0.0, 1.0]``:

        * ``1.0`` (default) — full standard K, suitable for traditional CMYK
          printing where K compensates for the inks' inability to reach
          true black.
        * ``0.0`` — emit K as all zeros, letting CMY alone carry the
          colour information.
        * Intermediate values scale K proportionally.

        Lower values are recommended for plotter line art with multiply-
        blended preview: at full K, the dense black-line layer overwhelms
        the underlying CMY in multiply mode (black × anything = black),
        so non-saturated colours look black instead of their true hue.

    Returns
    -------
    list of 4 ``(grayscale_image, hex_color)`` tuples in the order
    (Cyan, Magenta, Yellow, Key).  Each ``grayscale_image`` is an (H, W)
    uint8 array (0 = no ink, 255 = full ink); ``hex_color`` is the
    standard process color ("#00FFFF", "#FF00FF", "#FFFF00", "#000000").
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H×W×3), got shape {image.shape}")
    k_amount = float(max(0.0, min(1.0, k_amount)))

    img_f = image.astype(np.float32) / 255.0
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    k = 1.0 - np.maximum(np.maximum(r, g), b)  # (H, W)
    denom = 1.0 - k
    # Avoid division by zero for fully-black pixels (k == 1)
    safe_denom = np.where(denom > 0, denom, 1.0)

    c = (1.0 - r - k) / safe_denom
    m = (1.0 - g - k) / safe_denom
    y = (1.0 - b - k) / safe_denom

    # Zero out fully-black pixels where denom == 0
    zero_mask = denom == 0
    c = np.where(zero_mask, 0.0, c)
    m = np.where(zero_mask, 0.0, m)
    y = np.where(zero_mask, 0.0, y)

    # Clamp and convert to uint8 (0 = no ink, 255 = full ink)
    c_u8 = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    m_u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    k_u8 = np.clip(k * 255.0 * k_amount, 0, 255).astype(np.uint8)

    return [
        (c_u8, "#00FFFF"),
        (m_u8, "#FF00FF"),
        (y_u8, "#FFFF00"),
        (k_u8, "#000000"),
    ]
