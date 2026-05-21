"""Color space conversions and distance calculations.

This module provides functions for converting between color spaces (RGB, Lab)
and calculating color distances using various metrics. These are used by
the quantization and palette matching algorithms.

Color Spaces:
- RGB: Standard 8-bit per channel (0-255)
- Lab: CIE L*a*b* perceptually uniform color space

Distance Metrics:
- Euclidean: Simple squared distance (fast but not perceptually accurate)
- CIEDE2000: Advanced perceptual color difference (slower but accurate)
"""

import math
import re
from typing import List, Tuple

import numpy as np

from plottter.pixel_art.exceptions import ConfigurationError, InvalidPaletteError

# Type aliases for clarity
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
Lab = Tuple[float, float, float]

# D65 illuminant reference values (standard daylight)
D65_XN = 95.047
D65_YN = 100.0
D65_ZN = 108.883


def rgb_to_xyz(rgb: RGB) -> Tuple[float, float, float]:
    """Convert RGB to CIE XYZ color space.

    Uses sRGB to XYZ conversion matrix with gamma correction.

    Args:
        rgb: RGB tuple with values 0-255

    Returns:
        XYZ tuple as floats
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    def linearize(c: float) -> float:
        if c > 0.04045:
            return float(((c + 0.055) / 1.055) ** 2.4)
        return c / 12.92

    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)

    x = r_lin * 41.24564 + g_lin * 35.75761 + b_lin * 18.04375
    y = r_lin * 21.26729 + g_lin * 71.51522 + b_lin * 7.21750
    z = r_lin * 1.93339 + g_lin * 11.91920 + b_lin * 95.03041

    return (x, y, z)


def xyz_to_lab(xyz: Tuple[float, float, float]) -> Lab:
    """Convert CIE XYZ to CIE L*a*b* color space.

    Args:
        xyz: XYZ tuple

    Returns:
        Lab tuple (L: 0-100, a: ~-128-127, b: ~-128-127)
    """
    x = xyz[0] / D65_XN
    y = xyz[1] / D65_YN
    z = xyz[2] / D65_ZN

    def f(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta**3:
            return float(t ** (1.0 / 3.0))
        return t / (3.0 * delta**2) + 4.0 / 29.0

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return (L, a, b)


def rgb_to_lab(rgb: RGB) -> Lab:
    """Convert RGB to CIE L*a*b* color space.

    Args:
        rgb: RGB tuple with values 0-255

    Returns:
        Lab tuple (L: 0-100, a: ~-128-127, b: ~-128-127)
    """
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)


def lab_to_xyz(lab: Lab) -> Tuple[float, float, float]:
    """Convert CIE L*a*b* to CIE XYZ color space.

    Args:
        lab: Lab tuple

    Returns:
        XYZ tuple
    """
    L, a, b = lab

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def f_inv(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta:
            return t**3
        return 3.0 * delta**2 * (t - 4.0 / 29.0)

    x = D65_XN * f_inv(fx)
    y = D65_YN * f_inv(fy)
    z = D65_ZN * f_inv(fz)

    return (x, y, z)


def xyz_to_rgb(xyz: Tuple[float, float, float]) -> RGB:
    """Convert CIE XYZ to RGB color space.

    Args:
        xyz: XYZ tuple

    Returns:
        RGB tuple with values 0-255
    """
    x, y, z = xyz[0] / 100.0, xyz[1] / 100.0, xyz[2] / 100.0

    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def gamma_correct(c: float) -> float:
        if c > 0.0031308:
            return float(1.055 * (c ** (1.0 / 2.4)) - 0.055)
        return 12.92 * c

    r = gamma_correct(r_lin)
    g = gamma_correct(g_lin)
    b = gamma_correct(b_lin)

    r = int(round(max(0.0, min(1.0, r)) * 255))
    g = int(round(max(0.0, min(1.0, g)) * 255))
    b = int(round(max(0.0, min(1.0, b)) * 255))

    return (r, g, b)


def lab_to_rgb(lab: Lab) -> RGB:
    """Convert CIE L*a*b* to RGB color space.

    Args:
        lab: Lab tuple

    Returns:
        RGB tuple with values 0-255 (clamped if out of gamut)
    """
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb(xyz)


def rgb_distance_euclidean(c1: RGB, c2: RGB) -> float:
    """Calculate Euclidean distance in RGB space.

    Args:
        c1: First RGB color
        c2: Second RGB color

    Returns:
        Euclidean distance
    """
    dr = float(c1[0]) - float(c2[0])
    dg = float(c1[1]) - float(c2[1])
    db = float(c1[2]) - float(c2[2])
    return math.sqrt(dr * dr + dg * dg + db * db)


def lab_distance_euclidean(c1: Lab, c2: Lab) -> float:
    """Calculate Euclidean distance in Lab space.

    Args:
        c1: First Lab color
        c2: Second Lab color

    Returns:
        Euclidean distance (Delta E ab)
    """
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def ciede2000(lab1: Lab, lab2: Lab) -> float:
    """Calculate CIEDE2000 color difference.

    Args:
        lab1: First color in Lab space
        lab2: Second color in Lab space

    Returns:
        CIEDE2000 Delta E value (0 = identical, >100 = very different)
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)

    C_bar = (C1 + C2) / 2.0

    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - math.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    def hue_angle(a: float, b: float) -> float:
        if abs(a) < 1e-10 and abs(b) < 1e-10:
            return 0.0
        h = math.degrees(math.atan2(b, a))
        if h < 0:
            h += 360.0
        return h

    h1_prime = hue_angle(a1_prime, b1)
    h2_prime = hue_angle(a2_prime, b2)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0.0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180.0:
            delta_h_prime = diff
        elif diff > 180.0:
            delta_h_prime = diff - 360.0
        else:
            delta_h_prime = diff + 360.0

    delta_H_prime = (
        2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))
    )

    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        h_bar_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360.0:
        h_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        h_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_prime - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_prime - 63.0))
    )

    delta_theta = 30.0 * math.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))

    C_bar_prime_7 = C_bar_prime**7
    RC = 2.0 * math.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0**7))

    L_bar_prime_minus_50_sq = (L_bar_prime - 50.0) ** 2
    SL = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / math.sqrt(20.0 + L_bar_prime_minus_50_sq)

    SC = 1.0 + 0.045 * C_bar_prime

    SH = 1.0 + 0.015 * C_bar_prime * T

    RT = -math.sin(math.radians(2.0 * delta_theta)) * RC

    kL = 1.0
    kC = 1.0
    kH = 1.0

    delta_E = math.sqrt(
        (delta_L_prime / (kL * SL)) ** 2
        + (delta_C_prime / (kC * SC)) ** 2
        + (delta_H_prime / (kH * SH)) ** 2
        + RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )

    return delta_E


def find_nearest_color(
    color: RGB,
    palette: List[RGB],
    method: str = "euclidean",
    color_space: str = "rgb",
) -> Tuple[RGB, int]:
    """Find nearest palette color to input color.

    Args:
        color: RGB color to match
        palette: List of RGB colors in the palette
        method: Distance method ("euclidean" or "ciede2000")
        color_space: Color space for matching ("rgb" or "lab")

    Returns:
        Tuple of (nearest_color, palette_index)

    Raises:
        InvalidPaletteError: If palette is empty
        ConfigurationError: If method or color_space is invalid
    """
    if not palette:
        raise InvalidPaletteError("Palette cannot be empty")

    if color_space not in ("rgb", "lab"):
        raise ConfigurationError(f"Invalid color space: {color_space}")

    if method not in ("euclidean", "ciede2000"):
        raise ConfigurationError(f"Invalid method: {method}")

    if color_space == "lab" or method == "ciede2000":
        color_lab = rgb_to_lab(color)

    min_distance = float("inf")
    nearest_idx = 0

    for i, palette_color in enumerate(palette):
        if method == "ciede2000":
            palette_lab = rgb_to_lab(palette_color)
            distance = ciede2000(color_lab, palette_lab)
        elif color_space == "lab":
            palette_lab = rgb_to_lab(palette_color)
            distance = lab_distance_euclidean(color_lab, palette_lab)
        else:
            distance = rgb_distance_euclidean(color, palette_color)

        if distance < min_distance:
            min_distance = distance
            nearest_idx = i

    return palette[nearest_idx], nearest_idx


def find_nearest_colors_batch(
    colors: np.ndarray,
    palette: np.ndarray,
    color_space: str = "rgb",
) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest palette colors for a batch of colors (vectorized).

    Args:
        colors: (N, 3) array of RGB values
        palette: (P, 3) array of palette RGB values
        color_space: Color space for matching ("rgb" only for batch)

    Returns:
        Tuple of (matched_colors (N, 3), indices (N,))
    """
    if color_space != "rgb":
        raise ConfigurationError("Batch matching only supports RGB color space")

    diff = colors[:, np.newaxis, :].astype(np.float64) - palette[np.newaxis, :, :].astype(
        np.float64
    )
    distances = np.sum(diff**2, axis=2)

    nearest_indices = np.argmin(distances, axis=1)

    matched_colors = palette[nearest_indices]

    return matched_colors, nearest_indices


def hex_to_rgb(hex_color: str) -> RGB:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        RGB tuple

    Raises:
        InvalidPaletteError: If hex_color is not a valid 6-character hex string
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise InvalidPaletteError(f"Invalid hex color length: #{hex_color}")

    if not re.match(r"^[0-9a-fA-F]{6}$", hex_color):
        raise InvalidPaletteError(f"Invalid hex color characters: #{hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def rgb_to_hex(rgb: RGB) -> str:
    """Convert RGB tuple to hex color string.

    Args:
        rgb: RGB tuple

    Returns:
        Hex color string (e.g., "#FF0000")
    """
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
