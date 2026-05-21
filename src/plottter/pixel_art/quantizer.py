"""Color quantization algorithms for pixel art conversion.

This module provides various color quantization methods for reducing
an image's colors to match a specific palette or color count.

Key algorithms:
    NEAREST: Simple nearest-neighbor mapping (fastest)
    KMEANS: K-means clustering with k-means++ initialization
    MEDIAN_CUT: Median cut algorithm for adaptive palettes
    OCTREE: Octree-based quantization (memory efficient)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from plottter.pixel_art.color_utils import (
    ciede2000,
    rgb_to_lab,
)
from plottter.pixel_art.exceptions import InvalidImageError, InvalidPaletteError

logger = logging.getLogger(__name__)
from plottter.pixel_art.palette import RGB, FixedPalette, Palette, PaletteMetadata


class QuantizeMethod(Enum):
    """Available quantization algorithms.

    Attributes:
        NEAREST: Simple nearest-neighbor color mapping.
        KMEANS: K-means clustering with k-means++ initialization.
        MEDIAN_CUT: Median cut algorithm for generating adaptive palettes.
        OCTREE: Octree-based quantization.
    """

    NEAREST = "nearest"
    KMEANS = "kmeans"
    MEDIAN_CUT = "median_cut"
    OCTREE = "octree"


class ColorSpace(Enum):
    """Color space for distance calculations.

    Attributes:
        RGB: Standard RGB color space.
        LAB: CIE L*a*b* color space.
    """

    RGB = "rgb"
    LAB = "lab"


@dataclass
class QuantizeOptions:
    """Options for color quantization.

    Attributes:
        method: Quantization algorithm to use.
        color_space: Color space for distance calculations.
        max_colors: Maximum number of colors (for adaptive quantization).
        kmeans_max_iter: Maximum iterations for k-means.
        kmeans_epsilon: Convergence threshold for k-means.
        use_ciede2000: Use CIEDE2000 for LAB distance (more accurate but slower).
    """

    method: QuantizeMethod = QuantizeMethod.NEAREST
    color_space: ColorSpace = ColorSpace.RGB
    max_colors: Optional[int] = None
    kmeans_max_iter: int = 100
    kmeans_epsilon: float = 0.001
    use_ciede2000: bool = False


def quantize_to_palette(
    image: Image.Image,
    palette: Palette,
    method: QuantizeMethod = QuantizeMethod.NEAREST,
    color_space: ColorSpace = ColorSpace.RGB,
    options: Optional[QuantizeOptions] = None,
) -> Image.Image:
    """Quantize image colors to match a palette.

    Args:
        image: PIL Image to quantize (RGB or RGBA).
        palette: Target palette to map colors to.
        method: Quantization algorithm to use.
        color_space: Color space for distance calculations.
        options: Additional quantization options.

    Returns:
        Quantized PIL Image with colors from palette.

    Raises:
        InvalidPaletteError: If palette is empty.
        InvalidImageError: If image cannot be processed.
    """
    if palette.color_count == 0:
        raise InvalidPaletteError("Cannot quantize to empty palette")

    if options is None:
        options = QuantizeOptions(method=method, color_space=color_space)

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    has_alpha = image.mode == "RGBA"

    pixels = np.array(image)
    original_shape = pixels.shape

    alpha_channel: Optional[np.ndarray] = None
    if has_alpha:
        rgb_pixels = pixels[:, :, :3]
        alpha_channel = pixels[:, :, 3]
    else:
        rgb_pixels = pixels

    palette_colors = np.array(palette.colors, dtype=np.uint8)

    if options.method == QuantizeMethod.NEAREST:
        quantized_rgb = _nearest_quantize(
            rgb_pixels, palette_colors, options.color_space, options.use_ciede2000
        )
    elif options.method == QuantizeMethod.KMEANS:
        quantized_rgb = _kmeans_quantize(rgb_pixels, palette_colors, options)
    else:
        logger.warning(
            "%s quantization method is designed for adaptive palette generation. "
            "When used with a fixed palette, it falls back to NEAREST matching. "
            "Consider using NEAREST or KMEANS method for fixed palettes.",
            options.method.value.upper(),
        )
        quantized_rgb = _nearest_quantize(
            rgb_pixels, palette_colors, options.color_space, options.use_ciede2000
        )

    if has_alpha and alpha_channel is not None:
        result = np.zeros(original_shape, dtype=np.uint8)
        result[:, :, :3] = quantized_rgb
        result[:, :, 3] = alpha_channel
        return Image.fromarray(result, "RGBA")
    else:
        return Image.fromarray(quantized_rgb, "RGB")


def quantize_adaptive(
    image: Image.Image,
    max_colors: int,
    method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT,
) -> Tuple[Image.Image, Palette]:
    """Quantize image to an adaptive palette.

    Args:
        image: PIL Image to quantize.
        max_colors: Maximum number of colors in the adaptive palette.
        method: Quantization algorithm (MEDIAN_CUT or OCTREE).

    Returns:
        Tuple of (quantized image, generated palette).

    Raises:
        InvalidImageError: If max_colors is invalid.
    """
    if max_colors < 1:
        raise InvalidImageError(f"Invalid max_colors: {max_colors}")
    if max_colors > 256:
        max_colors = 256

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    has_alpha = image.mode == "RGBA"
    pixels = np.array(image)

    alpha_channel: Optional[np.ndarray] = None
    if has_alpha:
        rgb_pixels = pixels[:, :, :3]
        alpha_channel = pixels[:, :, 3]
    else:
        rgb_pixels = pixels

    if method == QuantizeMethod.OCTREE:
        quantized_rgb, palette_colors = _octree_quantize(rgb_pixels, max_colors)
    else:
        quantized_rgb, palette_colors = _median_cut_quantize(rgb_pixels, max_colors)

    metadata = PaletteMetadata(
        name="Adaptive",
        system="Generated",
        description=f"Adaptive {len(palette_colors)}-color palette",
    )
    generated_palette = FixedPalette(palette_colors, metadata)

    if has_alpha and alpha_channel is not None:
        result = np.zeros(pixels.shape, dtype=np.uint8)
        result[:, :, :3] = quantized_rgb
        result[:, :, 3] = alpha_channel
        return Image.fromarray(result, "RGBA"), generated_palette
    else:
        return Image.fromarray(quantized_rgb, "RGB"), generated_palette


def _nearest_quantize(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
    color_space: ColorSpace,
    use_ciede2000: bool = False,
) -> np.ndarray:
    """Nearest-neighbor quantization (vectorized)."""
    height, width = pixels.shape[:2]

    flat_pixels = pixels.reshape(-1, 3).astype(np.float32)
    n_pixels = flat_pixels.shape[0]

    if color_space == ColorSpace.LAB:
        palette_lab = np.array([rgb_to_lab(tuple(c)) for c in palette_colors], dtype=np.float32)

        if use_ciede2000:
            batch_size = 1000
            best_indices = np.zeros(n_pixels, dtype=np.int32)

            for start in range(0, n_pixels, batch_size):
                end = min(start + batch_size, n_pixels)
                batch = flat_pixels[start:end]

                for i, pixel in enumerate(batch):
                    pixel_lab = rgb_to_lab(tuple(pixel))
                    min_dist = float("inf")
                    best_idx = 0
                    for j, lab in enumerate(palette_lab):
                        dist = ciede2000(pixel_lab, tuple(lab))
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = j
                    best_indices[start + i] = best_idx
        else:
            pixels_lab = np.array([rgb_to_lab(tuple(p)) for p in flat_pixels], dtype=np.float32)

            diff = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
            distances = np.sum(diff**2, axis=2)

            best_indices = np.argmin(distances, axis=1)
    else:
        palette_float = palette_colors.astype(np.float32)
        diff = flat_pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :]
        distances = np.sum(diff**2, axis=2)

        best_indices = np.argmin(distances, axis=1)

    result = palette_colors[best_indices].reshape(height, width, 3)
    return result


def _kmeans_quantize(
    pixels: np.ndarray,
    palette_colors: np.ndarray,
    options: QuantizeOptions,
) -> np.ndarray:
    """K-means quantization with k-means++ initialization (vectorized)."""
    height, width = pixels.shape[:2]
    flat_pixels = pixels.reshape(-1, 3).astype(np.float32)
    n_pixels = flat_pixels.shape[0]

    if options.color_space == ColorSpace.LAB:
        data = np.array([rgb_to_lab(tuple(p)) for p in flat_pixels], dtype=np.float32)
        centers = np.array([rgb_to_lab(tuple(c)) for c in palette_colors], dtype=np.float32)
    else:
        centers = palette_colors.astype(np.float32)
        data = flat_pixels

    if options.color_space == ColorSpace.LAB and options.use_ciede2000:
        assignments = np.zeros(n_pixels, dtype=np.int32)

        for iteration in range(options.kmeans_max_iter):
            old_assignments = assignments.copy()

            for i in range(n_pixels):
                min_dist = float("inf")
                best_center = 0

                for j, center in enumerate(centers):
                    dist = ciede2000(tuple(data[i]), tuple(center))
                    if dist < min_dist:
                        min_dist = dist
                        best_center = j

                assignments[i] = best_center

            changed = np.sum(assignments != old_assignments)
            if changed == 0 or changed / n_pixels < options.kmeans_epsilon:
                break
    else:
        assignments = np.zeros(n_pixels, dtype=np.int32)

        for iteration in range(options.kmeans_max_iter):
            old_assignments = assignments.copy()

            diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.sum(diff**2, axis=2)

            assignments = np.argmin(distances, axis=1).astype(np.int32)

            changed = np.sum(assignments != old_assignments)
            if changed == 0 or changed / n_pixels < options.kmeans_epsilon:
                break

    result = palette_colors[assignments].reshape(height, width, 3)
    return result


def _median_cut_quantize(
    pixels: np.ndarray,
    max_colors: int,
) -> Tuple[np.ndarray, List[RGB]]:
    """Median cut quantization for adaptive palettes."""
    height, width = pixels.shape[:2]
    flat_pixels = pixels.reshape(-1, 3)

    unique_colors = np.unique(flat_pixels, axis=0)

    if len(unique_colors) <= max_colors:
        early_palette: List[RGB] = [(int(c[0]), int(c[1]), int(c[2])) for c in unique_colors]
        result = _map_to_palette(flat_pixels, np.array(early_palette))
        return result.reshape(height, width, 3), early_palette

    boxes = [unique_colors.copy()]

    while len(boxes) < max_colors:
        max_range = -1
        max_box_idx = 0
        max_axis = 0

        for i, box in enumerate(boxes):
            if len(box) < 2:
                continue
            for axis in range(3):
                range_val = box[:, axis].max() - box[:, axis].min()
                if range_val > max_range:
                    max_range = range_val
                    max_box_idx = i
                    max_axis = axis

        if max_range <= 0:
            break

        box = boxes[max_box_idx]
        sorted_box = box[box[:, max_axis].argsort()]
        median_idx = len(sorted_box) // 2

        box1 = sorted_box[:median_idx]
        box2 = sorted_box[median_idx:]

        boxes[max_box_idx] = box1
        if len(box2) > 0:
            boxes.append(box2)

    palette_colors: List[RGB] = []
    for box in boxes:
        if len(box) > 0:
            mean_color = np.mean(box, axis=0).astype(np.uint8)
            palette_colors.append((int(mean_color[0]), int(mean_color[1]), int(mean_color[2])))

    palette_array = np.array(palette_colors, dtype=np.uint8)
    result = _map_to_palette(flat_pixels, palette_array)

    return result.reshape(height, width, 3), palette_colors


def _octree_quantize(
    pixels: np.ndarray,
    max_colors: int,
) -> Tuple[np.ndarray, List[RGB]]:
    """Octree quantization for adaptive palettes."""
    height, width = pixels.shape[:2]
    flat_pixels = pixels.reshape(-1, 3)

    octree = _OctreeNode()

    for pixel in flat_pixels:
        octree.insert(tuple(pixel))

    while octree.leaf_count > max_colors:
        octree.reduce()

    palette_colors = octree.get_palette()

    if len(palette_colors) == 0:
        mean_color = tuple(np.mean(flat_pixels, axis=0).astype(np.uint8))
        palette_colors = [mean_color]

    palette_array = np.array(palette_colors, dtype=np.uint8)
    result = _map_to_palette(flat_pixels, palette_array)

    return result.reshape(height, width, 3), palette_colors


class _OctreeNode:
    """Node in an octree for color quantization."""

    def __init__(self, level: int = 0, parent: Optional["_OctreeNode"] = None) -> None:
        self.level = level
        self.parent = parent
        self.children: List[Optional["_OctreeNode"]] = [None] * 8
        self.is_leaf = True
        self.pixel_count = 0
        self.red_sum = 0
        self.green_sum = 0
        self.blue_sum = 0

        self._reducible: List[List["_OctreeNode"]] = [[] for _ in range(8)]

    @property
    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1 if self.pixel_count > 0 else 0

        count = 0
        for child in self.children:
            if child is not None:
                count += child.leaf_count
        return count

    def insert(self, color: RGB) -> None:
        self._insert_recursive(color, 0)

    def _insert_recursive(self, color: RGB, level: int) -> None:
        if level >= 8:
            self.is_leaf = True
            self.pixel_count += 1
            self.red_sum += int(color[0])
            self.green_sum += int(color[1])
            self.blue_sum += int(color[2])
            return

        index = self._get_color_index(color, level)

        if self.children[index] is None:
            self.children[index] = _OctreeNode(level + 1, self)
            self.is_leaf = False

        child = self.children[index]
        if child is not None:
            child._insert_recursive(color, level + 1)

    def _get_color_index(self, color: RGB, level: int) -> int:
        shift = 7 - level
        r_bit = (color[0] >> shift) & 1
        g_bit = (color[1] >> shift) & 1
        b_bit = (color[2] >> shift) & 1
        return (r_bit << 2) | (g_bit << 1) | b_bit

    def reduce(self) -> None:
        deepest = self._find_deepest_reducible()
        if deepest is not None:
            deepest._merge_children()

    def _find_deepest_reducible(self) -> Optional["_OctreeNode"]:
        for level in range(7, -1, -1):
            for child in self.children:
                if child is not None:
                    if child.is_leaf:
                        continue
                    all_leaves = True
                    for grandchild in child.children:
                        if grandchild is not None and not grandchild.is_leaf:
                            all_leaves = False
                            break
                    if all_leaves:
                        return child
                    result = child._find_deepest_reducible()
                    if result is not None:
                        return result
        if not self.is_leaf:
            return self
        return None

    def _merge_children(self) -> None:
        self.pixel_count = 0
        self.red_sum = 0
        self.green_sum = 0
        self.blue_sum = 0

        for i, child in enumerate(self.children):
            if child is not None:
                if child.is_leaf:
                    self.pixel_count += child.pixel_count
                    self.red_sum += child.red_sum
                    self.green_sum += child.green_sum
                    self.blue_sum += child.blue_sum
                else:
                    child._merge_children()
                    self.pixel_count += child.pixel_count
                    self.red_sum += child.red_sum
                    self.green_sum += child.green_sum
                    self.blue_sum += child.blue_sum
                self.children[i] = None

        self.is_leaf = True

    def get_palette(self) -> List[RGB]:
        colors: List[RGB] = []
        self._collect_colors(colors)
        return colors

    def _collect_colors(self, colors: List[RGB]) -> None:
        if self.is_leaf:
            if self.pixel_count > 0:
                r = int(self.red_sum / self.pixel_count)
                g = int(self.green_sum / self.pixel_count)
                b = int(self.blue_sum / self.pixel_count)
                colors.append((r, g, b))
        else:
            for child in self.children:
                if child is not None:
                    child._collect_colors(colors)


def _map_to_palette(
    pixels: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Map pixels to nearest palette colors (vectorized)."""
    pixels_float = pixels.astype(np.float32)
    palette_float = palette.astype(np.float32)

    diff = pixels_float[:, np.newaxis, :] - palette_float[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=2)

    best_indices = np.argmin(distances, axis=1)

    return palette[best_indices].astype(np.uint8)


def get_dominant_colors(
    image: Image.Image,
    n_colors: int = 5,
    method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT,
) -> List[RGB]:
    """Extract dominant colors from an image.

    Args:
        image: PIL Image to analyze.
        n_colors: Number of dominant colors to extract.
        method: Quantization method to use.

    Returns:
        List of dominant RGB colors, sorted by frequency.
    """
    if n_colors < 1:
        raise InvalidImageError(f"Invalid n_colors: {n_colors}")

    _, palette = quantize_adaptive(image, n_colors, method)

    return palette.colors
