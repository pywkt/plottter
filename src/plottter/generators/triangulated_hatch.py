"""TriangulatedHatchGenerator — edge-aware seed point placement for Delaunay-based hatching."""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import compute_image_rect
from plottter.generators.base import (
    BoolParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


def _edge_aware_seeds(
    gray: np.ndarray,
    img_rect: tuple[float, float, float, float],
    num_points: int,
    edge_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate seed points with higher density near image edges.

    Parameters
    ----------
    gray:
        Grayscale image array (uint8, shape HxW).
    img_rect:
        Bounding box in mm: (x1, y1, x2, y2).
    num_points:
        Target number of seed points (excluding the 4 corners).
    edge_weight:
        0 = uniform distribution, 1 = all points attracted to edges.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with (x_mm, y_mm) coordinates.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for triangulated hatching. "
            "Install with: pip install opencv-python"
        ) from exc

    img_x1, img_y1, img_x2, img_y2 = img_rect
    img_h, img_w = gray.shape[:2]

    # Step 1: Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Step 2: Blur the edge map to create a smooth density field
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 3)

    # Step 3: Combined weight map: uniform + edge attraction
    # Weights in [1 - edge_weight, 1] — never zero, so rejection sampling always converges.
    weights = (1.0 - edge_weight) + edge_weight * (edge_density / 255.0)

    # Step 4: Rejection sampling — generate candidate points uniformly, accept based on weight
    accepted: list[tuple[float, float]] = []
    batch_size = max(num_points * 4, 1000)

    while len(accepted) < num_points:
        # Generate candidates uniformly in mm space
        xs = rng.uniform(img_x1, img_x2, batch_size)
        ys = rng.uniform(img_y1, img_y2, batch_size)

        # Convert mm coords to pixel indices
        px = ((xs - img_x1) / (img_x2 - img_x1) * (img_w - 1)).astype(int)
        py = ((ys - img_y1) / (img_y2 - img_y1) * (img_h - 1)).astype(int)

        # Clamp to valid pixel range
        px = np.clip(px, 0, img_w - 1)
        py = np.clip(py, 0, img_h - 1)

        # Look up weight for each candidate
        w = weights[py, px]

        # Acceptance probability = w / max(w) — max weight is 1.0
        u = rng.uniform(0.0, 1.0, batch_size)
        mask = u <= w

        for i in np.where(mask)[0]:
            accepted.append((float(xs[i]), float(ys[i])))
            if len(accepted) >= num_points:
                break

    # Trim to exactly num_points
    accepted = accepted[:num_points]

    # Step 5: Add the 4 corners of the image rect so triangulation covers the full area
    corners = [
        (img_x1, img_y1),
        (img_x2, img_y1),
        (img_x1, img_y2),
        (img_x2, img_y2),
    ]
    all_points = corners + accepted

    # Step 6: Return as ndarray of shape (N, 2)
    return np.array(all_points, dtype=np.float64)


@register_generator
class TriangulatedHatchGenerator(Generator):
    """Edge-aware seed point placement for Delaunay-based triangulated hatching."""

    name = "Triangulated Hatching"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="num_points",
                label="Seed Points",
                min=100,
                max=10000,
                step=100,
                default=1000,
                description="Number of seed points to place across the image — more points produce finer triangulation detail",
            ),
            FloatParam(
                name="edge_weight",
                label="Edge Attraction",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.7,
                description="How strongly edges attract triangle vertices — 0 = uniform, 1 = all on edges",
            ),
            FloatParam(
                name="brightness",
                label="Brightness",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image brightness before processing (-100 to +100)",
            ),
            FloatParam(
                name="contrast",
                label="Contrast",
                min=-100.0,
                max=100.0,
                step=1.0,
                default=0.0,
                description="Adjust image contrast before processing (-100 to +100)",
            ),
            FloatParam(
                name="blur_radius",
                label="Blur Radius",
                min=0.0,
                max=20.0,
                step=0.5,
                default=1.0,
                description="Gaussian blur applied before processing — reduces noise",
            ),
            BoolParam(
                name="invert",
                label="Invert Image",
                default=False,
                description="Invert the image before processing",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Horizontal offset applied to the generated output on the canvas page (mm)",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y Offset (mm)",
                min=-500.0,
                max=500.0,
                step=0.5,
                default=0.0,
                randomizable=False,
                description="Vertical offset applied to the generated output on the canvas page (mm)",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "num_points": 1000,
                    "edge_weight": 0.7,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 1.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Uniform Grid",
                params={
                    "num_points": 1000,
                    "edge_weight": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "blur_radius": 0.0,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Edge Emphasis",
                params={
                    "num_points": 2000,
                    "edge_weight": 1.0,
                    "brightness": 0.0,
                    "contrast": 20.0,
                    "blur_radius": 0.5,
                    "invert": False,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        source: np.ndarray | None = params.get("_source_image")
        if source is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
        )

        # Apply preprocessing
        img = source.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        # Convert to grayscale
        if img.ndim == 3:
            try:
                import cv2
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except ImportError:
                gray = img.mean(axis=2).astype(np.uint8)
        else:
            gray = img

        if progress_callback:
            progress_callback(10)

        img_h, img_w = gray.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_rect = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        num_points = int(params.get("num_points", 1000))
        edge_weight = float(params.get("edge_weight", 0.7))
        seed = int(params.get("random_seed", 42))
        rng = np.random.default_rng(seed)

        if progress_callback:
            progress_callback(20)

        _seeds = _edge_aware_seeds(gray, img_rect, num_points, edge_weight, rng)

        if progress_callback:
            progress_callback(100)

        # Scaffold: seed points generated, return empty polylines until triangulation is added
        return []
