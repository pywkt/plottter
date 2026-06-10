"""Image import and preprocessing pipeline for Plottter."""

from __future__ import annotations

from typing import Any

import numpy as np


class ImageImportError(Exception):
    """Raised when an image cannot be loaded or processed."""


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_image(filepath: str) -> np.ndarray:
    """Load an image file as an RGB numpy array (H × W × 3, uint8).

    Supports JPG, PNG, WebP, and GIF via Pillow.
    Raises ImageImportError with a descriptive message on failure.
    """
    import os

    ext = os.path.splitext(filepath.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageImportError(
            f"Unsupported image format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        from PIL import Image as PILImage

        pil_img = PILImage.open(filepath)
    except FileNotFoundError:
        raise ImageImportError(f"File not found: {filepath}")
    except Exception as exc:
        raise ImageImportError(f"Failed to load image '{filepath}': {exc}") from exc

    # Ensure RGB; handles palette, RGBA, greyscale, etc.
    pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using ITU-R BT.601 luminance weights.

    Weights: 0.299R + 0.587G + 0.114B.
    Returns a 2D uint8 array (H × W).
    If the input is already 2D (grayscale), it is returned unchanged.
    """
    if image.ndim == 2:
        return image
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------


def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
    """Add *value* (–100 … 100) to all pixel channels.

    value is scaled so ±100 maps to ±255.
    """
    delta = int(value * 2.55)
    arr = image.astype(np.int16) + delta
    return np.clip(arr, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """Adjust contrast using the Photoshop-style formula.

    value in –100 … 100.  0 = no change, +100 = max contrast, –100 = flat grey.
    """
    # Map value (-100..100) → c (-255..255)
    c = value * 2.55
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    arr = image.astype(np.float32)
    arr = factor * (arr - 128.0) + 128.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction.

    gamma=1.0 → no change; gamma<1 brightens, gamma>1 darkens.
    Builds a look-up table for speed.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return table[image]


def apply_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Apply Gaussian blur. *radius* is the sigma in pixels.

    Returns a copy of the input array when radius ≤ 0.
    """
    if radius <= 0:
        return image.copy()
    import cv2

    # ksize must be positive and odd
    ksize = max(1, int(radius * 3) | 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(radius))


def apply_sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    """Sharpen using unsharp masking.

    amount controls the sharpening strength (0 = no change, 1 = strong).
    """
    if amount <= 0:
        return image.copy()
    import cv2

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)
    sharpened = cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_threshold(image: np.ndarray, value: float) -> np.ndarray:
    """Apply a binary threshold.

    Pixels with luminance ≥ value → 255; below → 0.
    If the input is RGB, it is converted to grayscale first.
    Returns a 2D uint8 array.
    """
    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image
    return np.where(gray >= value, np.uint8(255), np.uint8(0)).astype(np.uint8)


def invert_image(image: np.ndarray) -> np.ndarray:
    """Return 255 − image (bitwise NOT, channel-wise)."""
    return (255 - image.astype(np.int16)).clip(0, 255).astype(np.uint8)


def remove_background(image: np.ndarray, tolerance: float = 20.0) -> np.ndarray:
    """Set near-white pixels to white (treat them as empty background).

    Pixels where all channels >= 255 − tolerance are replaced with 255.
    For RGB images this means every channel must meet the brightness threshold.
    For grayscale images the pixel value is compared directly.

    A higher tolerance removes more off-white pixels (e.g. tolerance=50 removes
    pixels with all channels >= 205).  The default of 20 removes only very
    near-white pixels (all channels >= 235).
    """
    result = image.copy()
    threshold = 255.0 - tolerance
    if image.ndim == 2:
        result[image >= threshold] = 255
    else:
        mask = np.all(image >= threshold, axis=2)
        result[mask] = [255, 255, 255]
    return result


def crop_to_aspect(image: np.ndarray, width: float, height: float) -> np.ndarray:
    """Center-crop the image to match *width*:*height* aspect ratio.

    After cropping, the image is resized to (int(width), int(height)) pixels
    using high-quality Lanczos resampling if both dimensions are > 1.
    """
    h, w = image.shape[:2]
    target_ratio = width / height
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-6:
        cropped = image
    elif current_ratio > target_ratio:
        # Wider than target — trim width
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        cropped = image[:, x0 : x0 + new_w]
    else:
        # Taller than target — trim height
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        cropped = image[y0 : y0 + new_h, :]

    out_w, out_h = int(width), int(height)
    if out_w > 1 and out_h > 1:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(cropped)
        pil = pil.resize((out_w, out_h), PILImage.LANCZOS)
        return np.array(pil, dtype=np.uint8)
    return cropped


def downscale_to_max_pixels(image: np.ndarray, max_pixels: int) -> np.ndarray:
    """Downscale *image* so it has at most *max_pixels* pixels, preserving aspect.

    Returns the input unchanged when it already fits (or when ``max_pixels``
    is <= 0, which disables the cap).  The output dimensions depend only on
    the input dimensions and *max_pixels*, so two images of the same size are
    always reduced to the same size — color-separation relies on this so a
    cluster mask and its companion preprocessed image stay aligned.

    Used to bound the cost of color separation and the line generators that
    run on each separated mask: separation masks don't need full photographic
    resolution, and the line generators resample to plot density anyway.
    """
    h, w = image.shape[:2]
    if max_pixels <= 0 or h * w <= max_pixels:
        return image

    import math

    # Floor (not round) so the result never exceeds max_pixels.
    scale = math.sqrt(max_pixels / float(h * w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    from PIL import Image as PILImage

    pil = PILImage.fromarray(image)
    pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil, dtype=image.dtype)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def preprocess(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply all enabled preprocessing steps in order.

    Recognised param keys (all optional):

    ============  ====================  ========================
    Key           Type                  Description
    ============  ====================  ========================
    auto_contrast   bool                  Default True (stretch histogram)
    brightness      float −100 … 100      Default 0 (no change)
    contrast        float −100 … 100      Default 0 (no change)
    gamma           float 0.1 … 5.0       Default 1.0 (no change)
    blur            float ≥ 0             Default 0 (disabled)
    sharpen         float ≥ 0             Default 0 (disabled)
    unsharp_amount  float 0.0 … 5.0       Default 0 (disabled)
    remove_background float|None          Default None (disabled)
    crop_width      float                 Both crop_* required
    crop_height     float                 Both crop_* required
    threshold       float 0-255 | None    Default None (disabled)
    invert          bool                  Default False
    ============  ====================  ========================
    """
    result = image.copy()

    if params.get("auto_contrast", True):
        lo = np.percentile(result, 0.5)
        hi = np.percentile(result, 99.5)
        if hi > lo:
            result = np.clip((result.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    brightness = float(params.get("brightness", 0))
    if brightness != 0:
        result = adjust_brightness(result, brightness)

    contrast = float(params.get("contrast", 0))
    if contrast != 0:
        result = adjust_contrast(result, contrast)

    gamma = float(params.get("gamma", 1.0))
    if gamma != 1.0:
        result = adjust_gamma(result, gamma)

    blur = float(params.get("blur", 0))
    if blur > 0:
        result = apply_blur(result, blur)

    sharpen = float(params.get("sharpen", 0))
    if sharpen > 0:
        result = apply_sharpen(result, sharpen)

    unsharp_amount = float(params.get("unsharp_amount", 0))
    if unsharp_amount > 0:
        import cv2
        blurred = cv2.GaussianBlur(result, (0, 0), 2)
        result = cv2.addWeighted(result, 1 + unsharp_amount, blurred, -unsharp_amount, 0)

    bg_tol = params.get("remove_background", None)
    if bg_tol is not None:
        result = remove_background(result, float(bg_tol))

    crop_w = params.get("crop_width", None)
    crop_h = params.get("crop_height", None)
    if crop_w is not None and crop_h is not None:
        result = crop_to_aspect(result, float(crop_w), float(crop_h))

    threshold = params.get("threshold", None)
    if threshold is not None:
        result = apply_threshold(result, float(threshold))

    if params.get("invert", False):
        result = invert_image(result)

    return result
