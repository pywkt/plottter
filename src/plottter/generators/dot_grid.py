"""DotGridGenerator — grids of shapes with Perlin noise size/rotation/density modulation."""

from __future__ import annotations

import math
from typing import Any

try:
    import noise as _noise_lib
    _NOISE_AVAILABLE = True
except ImportError:
    _NOISE_AVAILABLE = False

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from plottter.generators import register_generator
from plottter.generators._helpers import _load_source_image
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    ImageParam,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Shape helpers — all drawn centred at (0, 0), caller applies transform
# ---------------------------------------------------------------------------

def _circle_at_origin(r: float, sides: int = 16) -> Polyline:
    """Closed polygon approximating a circle of radius r centred at the origin."""
    pts = []
    for i in range(sides + 1):
        angle = _TWO_PI * i / sides
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def _square_at_origin(r: float) -> Polyline:
    """Axis-aligned square with half-side r, centred at the origin."""
    return [(-r, -r), (r, -r), (r, r), (-r, r), (-r, -r)]


def _diamond_at_origin(r: float) -> Polyline:
    """45°-rotated square (diamond) with half-diagonal r, centred at the origin."""
    return [(0.0, -r), (r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]


def _cross_at_origin(r: float) -> list[Polyline]:
    """Two perpendicular lines of half-length r, centred at the origin."""
    return [[(-r, 0.0), (r, 0.0)], [(0.0, -r), (0.0, r)]]


def _star_at_origin(r: float, points: int = 5) -> Polyline:
    """5-pointed star with outer radius r and inner radius r*0.4, centred at the origin."""
    inner_r = r * 0.4
    pts = []
    for i in range(points * 2 + 1):
        angle = _TWO_PI * i / (points * 2) - math.pi / 2
        radius = r if i % 2 == 0 else inner_r
        pts.append((radius * math.cos(angle), radius * math.sin(angle)))
    return pts


def _hexagon_at_origin(r: float) -> Polyline:
    """Regular hexagon with circumradius r, centred at the origin."""
    pts = []
    for i in range(7):
        angle = _TWO_PI * i / 6 - math.pi / 6
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def _transform_paths(
    paths: list[Polyline],
    rot_rad: float,
    cx: float,
    cy: float,
) -> list[Polyline]:
    """Rotate paths around the origin by *rot_rad* radians, then translate to (cx, cy)."""
    if rot_rad == 0.0:
        return [[(x + cx, y + cy) for x, y in path] for path in paths]
    cos_a = math.cos(rot_rad)
    sin_a = math.sin(rot_rad)
    return [
        [(x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy) for x, y in path]
        for path in paths
    ]


@register_generator
class DotGridGenerator(Generator):
    """Generates a regular grid of dot shapes with optional Perlin noise modulation."""

    name = "Dot Grid"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            IntParam(
                name="grid_cols",
                label="Columns",
                min=3,
                max=100,
                step=1,
                default=20,
                description="Number of columns in the grid",
            ),
            IntParam(
                name="grid_rows",
                label="Rows",
                min=3,
                max=100,
                step=1,
                default=20,
                description="Number of rows in the grid",
            ),
            ChoiceParam(
                name="dot_shape",
                label="Dot shape",
                choices=["Circle", "Square", "Diamond", "Cross", "Star", "Hexagon"],
                default="Circle",
                description="Shape drawn at each grid cell",
            ),
            FloatParam(
                name="base_size_mm",
                label="Base size (mm)",
                min=0.5,
                max=20.0,
                step=0.1,
                default=3.0,
                description="Base dot radius/half-size in mm",
            ),
            FloatParam(
                name="spacing_mm",
                label="Grid spacing (mm)",
                min=1.0,
                max=30.0,
                step=0.1,
                default=5.0,
                description="Center-to-center distance between grid cells in mm",
            ),
            FloatParam(
                name="noise_scale",
                label="Noise scale",
                min=0.01,
                max=1.0,
                step=0.01,
                default=0.1,
                description="Scale of Perlin noise field — smaller = larger noise features",
            ),
            FloatParam(
                name="noise_strength",
                label="Noise strength",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.5,
                description="How much noise affects dot size — 0 = uniform, 1 = full range",
            ),
            IntParam(
                name="noise_seed",
                label="Noise seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for noise generation",
            ),
            FloatParam(
                name="min_size_mm",
                label="Min size (mm)",
                min=0.0,
                max=10.0,
                step=0.1,
                default=0.5,
                description="Minimum dot size after noise modulation — dots smaller than this are skipped",
            ),
            FloatParam(
                name="max_size_mm",
                label="Max size (mm)",
                min=0.5,
                max=30.0,
                step=0.1,
                default=8.0,
                description="Maximum dot size after noise modulation",
            ),
            FloatParam(
                name="rotation_noise",
                label="Rotation noise (°)",
                min=0.0,
                max=180.0,
                step=1.0,
                default=0.0,
                description="Max random rotation per dot in degrees — noise drives direction",
            ),
            FloatParam(
                name="jitter_mm",
                label="Position jitter (mm)",
                min=0.0,
                max=10.0,
                step=0.1,
                default=0.0,
                description="Random offset from grid position in mm — breaks rigid grid feel",
            ),
            BoolParam(
                name="filled",
                label="Fill shapes",
                default=False,
                description="Fill shapes with concentric lines instead of a single outline",
            ),
            FloatParam(
                name="pen_width_mm",
                label="Pen width (mm)",
                min=0.1,
                max=1.0,
                step=0.05,
                default=0.3,
                description="Pen width for fill line spacing",
                visible_when={"filled": [True]},
            ),
            FloatParam(
                name="convergence",
                label="Convergence",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.0,
                description=(
                    "Nudge grid points toward darker image areas — "
                    "0 = regular grid, 1 = maximum attraction"
                ),
            ),
            ImageParam(
                name="_source_image",
                label="Source Image",
                randomizable=False,
                description=(
                    "Image used to drive point convergence (dark areas attract grid points). "
                    "Only active when Convergence > 0."
                ),
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
                name="Default Grid",
                params={
                    "grid_cols": 20,
                    "grid_rows": 20,
                    "dot_shape": "Circle",
                    "base_size_mm": 3.0,
                    "spacing_mm": 5.0,
                },
            ),
            Preset(
                name="Halftone Dots",
                params={
                    "dot_shape": "Circle",
                    "noise_strength": 0.8,
                    "base_size_mm": 2.0,
                    "spacing_mm": 4.0,
                    "noise_scale": 0.05,
                },
            ),
            Preset(
                name="Star Field",
                params={
                    "dot_shape": "Star",
                    "noise_strength": 0.6,
                    "rotation_noise": 45.0,
                    "jitter_mm": 1.0,
                    "base_size_mm": 2.5,
                    "spacing_mm": 6.0,
                    "min_size_mm": 0.3,
                },
            ),
            Preset(
                name="Diamond Mosaic",
                params={
                    "dot_shape": "Diamond",
                    "filled": True,
                    "noise_strength": 0.4,
                    "spacing_mm": 4.0,
                    "base_size_mm": 3.5,
                    "pen_width_mm": 0.25,
                },
            ),
            Preset(
                name="Cross Stitch",
                params={
                    "dot_shape": "Cross",
                    "noise_strength": 0.3,
                    "rotation_noise": 15.0,
                    "spacing_mm": 5.0,
                    "base_size_mm": 3.0,
                },
            ),
            Preset(
                name="Hexagon Grid",
                params={
                    "dot_shape": "Hexagon",
                    "noise_strength": 0.5,
                    "base_size_mm": 4.0,
                    "spacing_mm": 5.0,
                    "jitter_mm": 0.5,
                },
            ),
            Preset(
                name="Noise Landscape",
                params={
                    "dot_shape": "Circle",
                    "noise_scale": 0.02,
                    "noise_strength": 1.0,
                    "base_size_mm": 1.0,
                    "max_size_mm": 10.0,
                    "spacing_mm": 5.0,
                    "filled": True,
                    "pen_width_mm": 0.2,
                },
            ),
            Preset(
                name="Convergent Dots",
                params={
                    "dot_shape": "Circle",
                    "convergence": 0.5,
                    "noise_strength": 0.0,
                    "base_size_mm": 2.0,
                    "spacing_mm": 5.0,
                    "grid_cols": 25,
                    "grid_rows": 25,
                },
                description=(
                    "Grid points attracted toward darker areas of a source image.\n"
                    "Requires a source image (_source_image) to drive convergence."
                ),
            ),
            Preset(
                name="Warped Grid",
                params={
                    "dot_shape": "Square",
                    "convergence": 0.8,
                    "noise_strength": 0.2,
                    "base_size_mm": 2.5,
                    "spacing_mm": 5.0,
                    "grid_cols": 25,
                    "grid_rows": 25,
                    "noise_scale": 0.08,
                },
                description=(
                    "Squares warped toward darker image regions with subtle noise distortion.\n"
                    "Requires a source image (_source_image) to drive convergence."
                ),
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        grid_cols = int(params.get("grid_cols", 20))
        grid_rows = int(params.get("grid_rows", 20))
        dot_shape = str(params.get("dot_shape", "Circle"))
        base_size = float(params.get("base_size_mm", 3.0))
        spacing = float(params.get("spacing_mm", 5.0))
        noise_scale = float(params.get("noise_scale", 0.1))
        noise_strength = float(params.get("noise_strength", 0.5))
        noise_seed = int(params.get("noise_seed", 42))
        min_size = float(params.get("min_size_mm", 0.5))
        max_size = float(params.get("max_size_mm", 8.0))
        rotation_noise = float(params.get("rotation_noise", 0.0))
        jitter_mm = float(params.get("jitter_mm", 0.0))
        filled = bool(params.get("filled", False))
        pen_width = float(params.get("pen_width_mm", 0.3))
        convergence = float(params.get("convergence", 0.0))
        source_image = _load_source_image(params.get("_source_image"))
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # --- Precompute convergence maps (Sobel gradient) ---
        _gray_img = None
        _sobel_x = None
        _sobel_y = None
        _conv_img_w = 0
        _conv_img_h = 0

        if convergence > 0.0 and source_image is not None and _NUMPY_AVAILABLE:
            gray = source_image
            if gray.ndim == 3:
                try:
                    import cv2 as _cv2
                    gray = _cv2.cvtColor(gray, _cv2.COLOR_RGB2GRAY)
                except ImportError:
                    gray = gray.mean(axis=2).astype(_np.uint8)
            _gray_img = gray.astype(_np.float32)
            _conv_img_h, _conv_img_w = _gray_img.shape
            try:
                import cv2 as _cv2
                _sobel_x = _cv2.Sobel(_gray_img, _cv2.CV_32F, 1, 0, ksize=3)
                _sobel_y = _cv2.Sobel(_gray_img, _cv2.CV_32F, 0, 1, ksize=3)
            except ImportError:
                # Fallback: numpy gradient (central differences)
                _sobel_y, _sobel_x = _np.gradient(_gray_img)

        # Centre the grid on the drawing area
        grid_w = (grid_cols - 1) * spacing
        grid_h = (grid_rows - 1) * spacing
        origin_x = draw_x1 + (draw_w - grid_w) / 2.0
        origin_y = draw_y1 + (draw_h - grid_h) / 2.0

        use_noise = _NOISE_AVAILABLE and (
            noise_strength > 0.0 or rotation_noise > 0.0 or jitter_mm > 0.0
        )

        result: list[Polyline] = []
        total = grid_cols * grid_rows

        for row in range(grid_rows):
            for col in range(grid_cols):
                if cancelled_callback and cancelled_callback():
                    break

                cx = origin_x + col * spacing
                cy = origin_y + row * spacing

                # ---- size modulation ----
                if use_noise and noise_strength > 0.0:
                    size_noise = _noise_lib.pnoise2(
                        col * noise_scale,
                        row * noise_scale,
                        base=noise_seed,
                    )
                    r = base_size * (1.0 + size_noise * noise_strength)
                else:
                    r = base_size

                # Skip dots smaller than min_size
                if r < min_size:
                    continue

                r = min(max_size, r)

                # ---- rotation modulation ----
                rot_rad = 0.0
                if use_noise and rotation_noise > 0.0:
                    rot_noise = _noise_lib.pnoise2(
                        col * noise_scale + 100.0,
                        row * noise_scale + 100.0,
                        base=noise_seed,
                    )
                    rot_rad = rot_noise * math.radians(rotation_noise)

                # ---- position jitter ----
                cx_final = cx
                cy_final = cy
                if use_noise and jitter_mm > 0.0:
                    dx_noise = _noise_lib.pnoise2(
                        col * noise_scale + 200.0,
                        row * noise_scale + 200.0,
                        base=noise_seed,
                    )
                    dy_noise = _noise_lib.pnoise2(
                        col * noise_scale + 300.0,
                        row * noise_scale + 300.0,
                        base=noise_seed,
                    )
                    cx_final += dx_noise * jitter_mm
                    cy_final += dy_noise * jitter_mm

                # ---- convergence: attract toward darker image areas ----
                if _gray_img is not None and _sobel_x is not None and _sobel_y is not None:
                    # Map mm position to image pixel coordinates
                    px = (cx_final - draw_x1) / draw_w * _conv_img_w
                    py = (cy_final - draw_y1) / draw_h * _conv_img_h
                    px_i = int(max(0, min(_conv_img_w - 1, round(px))))
                    py_i = int(max(0, min(_conv_img_h - 1, round(py))))

                    brightness = float(_gray_img[py_i, px_i])
                    gx_val = float(_sobel_x[py_i, px_i])
                    gy_val = float(_sobel_y[py_i, px_i])
                    gmag = math.hypot(gx_val, gy_val)

                    if gmag > 1e-6:
                        # Gradient points toward brighter areas; negate to move toward darker
                        scale = convergence * spacing * (1.0 - brightness / 255.0)
                        cx_final += (-gx_val / gmag) * scale
                        cy_final += (-gy_val / gmag) * scale

                    # Clamp to canvas drawing area
                    cx_final = max(draw_x1, min(draw_x2, cx_final))
                    cy_final = max(draw_y1, min(draw_y2, cy_final))

                # ---- emit shape(s) ----
                if filled:
                    cur_r = r
                    while cur_r > 0.0:
                        paths = self._shape_paths_at_origin(dot_shape, cur_r)
                        result.extend(_transform_paths(paths, rot_rad, cx_final, cy_final))
                        cur_r -= pen_width
                else:
                    paths = self._shape_paths_at_origin(dot_shape, r)
                    result.extend(_transform_paths(paths, rot_rad, cx_final, cy_final))

                if progress_callback:
                    idx = row * grid_cols + col
                    if idx % 50 == 0:
                        progress_callback(int(idx / total * 100))

            if cancelled_callback and cancelled_callback():
                break

        if progress_callback:
            progress_callback(100)

        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]

        return result

    # ------------------------------------------------------------------
    # Shape dispatch — returns list of Polylines centred at (0, 0)
    # ------------------------------------------------------------------

    def _shape_paths_at_origin(self, shape: str, r: float) -> list[Polyline]:
        """Return paths for *shape* at radius *r*, centred at the origin."""
        if shape == "Square":
            return [_square_at_origin(r)]
        elif shape == "Diamond":
            return [_diamond_at_origin(r)]
        elif shape == "Cross":
            return _cross_at_origin(r)
        elif shape == "Star":
            return [_star_at_origin(r)]
        elif shape == "Hexagon":
            return [_hexagon_at_origin(r)]
        # Default: Circle
        return [_circle_at_origin(r)]
