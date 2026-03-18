"""GridPatternGenerator — sine grids, Truchet tiles, and concentric shapes."""

from __future__ import annotations

import math
import random as _random
from typing import Any

from plottter.generators.base import (
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.generators import register_generator
from plottter.models import Canvas, Polyline

_TWO_PI = 2.0 * math.pi


@register_generator
class GridPatternGenerator(Generator):
    """Generates grid-based patterns: sine-modulated grid, Truchet tiles, concentric shapes."""

    name = "Grid Pattern"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ChoiceParam(
                name="mode",
                label="Mode",
                choices=["Sine Grid", "Truchet Tiles", "Concentric Shapes", "Islamic Tiling", "Celtic Knot"],
                default="Sine Grid",
                description="Type of grid/tile pattern to generate",
                choice_descriptions={
                    "Sine Grid": "Parallel lines modulated by a sine wave — varying amplitude, frequency and direction",
                    "Truchet Tiles": "Random quarter-circle arcs filling a grid — creates flowing maze-like patterns",
                    "Concentric Shapes": "Nested repeated shapes (circles, squares, polygons) radiating outward from center",
                    "Islamic Tiling": "Geometric star patterns based on traditional Islamic art — 6, 8 or 12-pointed stars",
                    "Celtic Knot": "Interlaced strand weaving pattern on a grid with over/under crossings",
                },
            ),
            # --- Sine Grid parameters ---
            IntParam(
                name="line_count",
                label="Line count",
                min=2,
                max=500,
                step=1,
                default=20,
                description="Number of lines in the sine-modulated grid",
            ),
            FloatParam(
                name="line_spacing_mm",
                label="Line spacing (mm)",
                min=0.5,
                max=50.0,
                step=0.5,
                default=10.0,
                description="Spacing between grid lines in millimeters",
            ),
            FloatParam(
                name="amplitude_mm",
                label="Sine amplitude (mm)",
                min=0.0,
                max=100.0,
                step=0.5,
                default=5.0,
                description="Peak amplitude of the sine wave in millimeters — how far each line deviates from straight",
            ),
            FloatParam(
                name="frequency",
                label="Sine frequency",
                min=0.01,
                max=20.0,
                step=0.01,
                default=1.0,
                description="Number of complete sine wave cycles across the canvas",
            ),
            FloatParam(
                name="phase",
                label="Sine phase (radians)",
                min=0.0,
                max=_TWO_PI,
                step=0.01,
                default=0.0,
                description="Phase offset of the sine wave in radians — shifts the wave horizontally",
            ),
            ChoiceParam(
                name="direction",
                label="Grid direction",
                choices=["Horizontal", "Vertical", "Both"],
                default="Horizontal",
                description="Orientation of the sine-modulated grid lines",
            ),
            # --- Truchet Tiles parameters ---
            FloatParam(
                name="tile_size_mm",
                label="Tile size (mm)",
                min=1.0,
                max=50.0,
                step=1.0,
                default=10.0,
                description="Size of each Truchet tile in millimeters",
            ),
            IntParam(
                name="seed",
                label="Random seed",
                min=0,
                max=9999,
                step=1,
                default=42,
                description="Random seed for reproducible tile pattern generation",
            ),
            # --- Concentric Shapes parameters ---
            ChoiceParam(
                name="shape",
                label="Shape",
                choices=["Circle", "Square", "Polygon"],
                default="Circle",
                description="Shape type for concentric repetition",
            ),
            IntParam(
                name="sides",
                label="Polygon sides",
                min=3,
                max=12,
                step=1,
                default=6,
                description="Number of sides for the polygon shape (3 = triangle, 6 = hexagon, etc.)",
            ),
            FloatParam(
                name="spacing_mm",
                label="Shape spacing (mm)",
                min=0.5,
                max=50.0,
                step=0.5,
                default=10.0,
                description="Distance between each concentric repetition of the shape in millimeters",
            ),
            IntParam(
                name="count",
                label="Shape count",
                min=1,
                max=50,
                step=1,
                default=10,
                description="Number of concentric shapes to draw",
            ),
            # --- Islamic Tiling parameters ---
            ChoiceParam(
                name="islamic_type",
                label="Star type",
                choices=["6-Point Stars", "8-Point Stars", "12-Point Stars"],
                default="8-Point Stars",
                description="Style of Islamic geometric star tiling",
            ),
            FloatParam(
                name="star_inset",
                label="Star inset ratio",
                min=0.05,
                max=0.49,
                step=0.01,
                default=0.15,
                description="Controls how pointed the star tips are — smaller values create sharper, more elongated points",
            ),
            # --- Celtic Knot parameters ---
            IntParam(
                name="knot_cols",
                label="Knot columns",
                min=1,
                max=30,
                step=1,
                default=6,
                description="Number of columns in the Celtic knot grid",
            ),
            IntParam(
                name="knot_rows",
                label="Knot rows",
                min=1,
                max=30,
                step=1,
                default=6,
                description="Number of rows in the Celtic knot grid",
            ),
            FloatParam(
                name="gap_mm",
                label="Crossing gap (mm)",
                min=0.1,
                max=5.0,
                step=0.1,
                default=0.8,
                description="Size of the gap at strand crossings in millimeters — larger values make the over/under weaving more visible",
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
                name="Sine Grid (H)",
                params={
                    "mode": "Sine Grid",
                    "line_count": 20,
                    "line_spacing_mm": 10.0,
                    "amplitude_mm": 5.0,
                    "frequency": 1.0,
                    "phase": 0.0,
                    "direction": "Horizontal",
                },
            ),
            Preset(
                name="Sine Grid (Both)",
                params={
                    "mode": "Sine Grid",
                    "line_count": 15,
                    "line_spacing_mm": 12.0,
                    "amplitude_mm": 4.0,
                    "frequency": 2.0,
                    "phase": 0.0,
                    "direction": "Both",
                },
            ),
            Preset(
                name="Truchet Tiles",
                params={
                    "mode": "Truchet Tiles",
                    "tile_size_mm": 10.0,
                    "seed": 42,
                },
            ),
            Preset(
                name="Concentric Circles",
                params={
                    "mode": "Concentric Shapes",
                    "shape": "Circle",
                    "spacing_mm": 10.0,
                    "count": 10,
                },
            ),
            Preset(
                name="Concentric Squares",
                params={
                    "mode": "Concentric Shapes",
                    "shape": "Square",
                    "spacing_mm": 10.0,
                    "count": 10,
                },
            ),
            Preset(
                name="Concentric Hexagons",
                params={
                    "mode": "Concentric Shapes",
                    "shape": "Polygon",
                    "sides": 6,
                    "spacing_mm": 10.0,
                    "count": 8,
                },
            ),
            Preset(
                name="Islamic 6-Point Stars",
                params={
                    "mode": "Islamic Tiling",
                    "islamic_type": "6-Point Stars",
                    "tile_size_mm": 18.0,
                    "star_inset": 0.12,
                },
            ),
            Preset(
                name="Islamic 8-Point Stars",
                params={
                    "mode": "Islamic Tiling",
                    "islamic_type": "8-Point Stars",
                    "tile_size_mm": 20.0,
                    "star_inset": 0.15,
                },
            ),
            Preset(
                name="Islamic 12-Point Stars",
                params={
                    "mode": "Islamic Tiling",
                    "islamic_type": "12-Point Stars",
                    "tile_size_mm": 22.0,
                    "star_inset": 0.15,
                },
            ),
            Preset(
                name="Celtic Plait 6x6",
                params={
                    "mode": "Celtic Knot",
                    "knot_cols": 6,
                    "knot_rows": 6,
                    "tile_size_mm": 12.0,
                    "gap_mm": 0.8,
                },
            ),
            Preset(
                name="Celtic Plait 10x8",
                params={
                    "mode": "Celtic Knot",
                    "knot_cols": 10,
                    "knot_rows": 8,
                    "tile_size_mm": 10.0,
                    "gap_mm": 0.6,
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
        mode = str(params.get("mode", "Sine Grid"))

        if mode == "Sine Grid":
            result = self._generate_sine_grid(params, canvas, progress_callback, cancelled_callback)
        elif mode == "Truchet Tiles":
            result = self._generate_truchet(params, canvas, progress_callback, cancelled_callback)
        elif mode == "Concentric Shapes":
            result = self._generate_concentric(params, canvas, progress_callback, cancelled_callback)
        elif mode == "Islamic Tiling":
            result = self._generate_islamic(params, canvas, progress_callback, cancelled_callback)
        elif mode == "Celtic Knot":
            result = self._generate_celtic(params, canvas, progress_callback, cancelled_callback)
        else:
            return []

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
        return result

    def _generate_sine_grid(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        line_count = int(params.get("line_count", 20))
        line_spacing = float(params.get("line_spacing_mm", 10.0))
        amplitude = float(params.get("amplitude_mm", 5.0))
        frequency = float(params.get("frequency", 1.0))
        phase = float(params.get("phase", 0.0))
        direction = str(params.get("direction", "Horizontal"))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        # Clamp line count so lines fit within the drawing area
        max_h_lines = max(1, int(draw_h / line_spacing))
        max_v_lines = max(1, int(draw_w / line_spacing))
        h_count = min(line_count, max_h_lines)
        v_count = min(line_count, max_v_lines)

        polylines: list[Polyline] = []
        samples = 200  # points per line

        def make_h_line(y_base: float) -> Polyline:
            points: Polyline = []
            for j in range(samples + 1):
                t = j / samples
                x = draw_x1 + t * draw_w
                y = y_base + amplitude * math.sin(frequency * _TWO_PI * t + phase)
                points.append((x, y))
            return points

        def make_v_line(x_base: float) -> Polyline:
            points: Polyline = []
            for j in range(samples + 1):
                t = j / samples
                y = draw_y1 + t * draw_h
                x = x_base + amplitude * math.sin(frequency * _TWO_PI * t + phase)
                points.append((x, y))
            return points

        if direction in ("Horizontal", "Both"):
            for i in range(h_count):
                if cancelled_callback and cancelled_callback():
                    break
                y_base = draw_y1 + (i + 0.5) * line_spacing
                polylines.append(make_h_line(y_base))
                if progress_callback:
                    progress_callback(int(i / h_count * 50))

        if direction in ("Vertical", "Both"):
            for i in range(v_count):
                if cancelled_callback and cancelled_callback():
                    break
                x_base = draw_x1 + (i + 0.5) * line_spacing
                polylines.append(make_v_line(x_base))
                if progress_callback and direction == "Both":
                    progress_callback(50 + int(i / v_count * 50))
                elif progress_callback:
                    progress_callback(int(i / v_count * 100))

        if progress_callback:
            progress_callback(100)

        return polylines

    def _generate_truchet(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        tile_size = float(params.get("tile_size_mm", 10.0))
        seed = int(params.get("seed", 42))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        rng = _random.Random(seed)
        cols = max(1, int((draw_x2 - draw_x1) / tile_size))
        rows = max(1, int((draw_y2 - draw_y1) / tile_size))

        arc_samples = 20  # points per quarter-circle arc
        polylines: list[Polyline] = []
        total = rows * cols

        for row in range(rows):
            for col in range(cols):
                if cancelled_callback and cancelled_callback():
                    break

                tx = draw_x1 + col * tile_size
                ty = draw_y1 + row * tile_size
                flip = rng.choice([False, True])

                # Each tile has two quarter-circle arcs
                # Arc 1: corner (tx, ty) → (tx+tile, ty+tile) if not flipped
                #        corner (tx+tile, ty) → (tx, ty+tile) if flipped
                r = tile_size / 2.0

                if not flip:
                    # Arc from top-left corner
                    cx1, cy1 = tx, ty  # center of arc is at top-left corner
                    arc1 = [
                        (cx1 + r * math.cos(math.radians(a)), cy1 + r * math.sin(math.radians(a)))
                        for a in [i * 90 / arc_samples for i in range(arc_samples + 1)]
                    ]
                    # Arc from bottom-right corner
                    cx2, cy2 = tx + tile_size, ty + tile_size
                    arc2 = [
                        (cx2 + r * math.cos(math.radians(180 + a)), cy2 + r * math.sin(math.radians(180 + a)))
                        for a in [i * 90 / arc_samples for i in range(arc_samples + 1)]
                    ]
                else:
                    # Arc from top-right corner
                    cx1, cy1 = tx + tile_size, ty
                    arc1 = [
                        (cx1 + r * math.cos(math.radians(90 + a)), cy1 + r * math.sin(math.radians(90 + a)))
                        for a in [i * 90 / arc_samples for i in range(arc_samples + 1)]
                    ]
                    # Arc from bottom-left corner
                    cx2, cy2 = tx, ty + tile_size
                    arc2 = [
                        (cx2 + r * math.cos(math.radians(270 + a)), cy2 + r * math.sin(math.radians(270 + a)))
                        for a in [i * 90 / arc_samples for i in range(arc_samples + 1)]
                    ]

                polylines.append(arc1)
                polylines.append(arc2)

                if progress_callback:
                    progress_callback(int((row * cols + col) / total * 100))

        if progress_callback:
            progress_callback(100)

        return polylines

    def _generate_concentric(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        shape = str(params.get("shape", "Circle"))
        sides = int(params.get("sides", 6))
        spacing = float(params.get("spacing_mm", 10.0))
        count = int(params.get("count", 10))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        cx = (draw_x1 + draw_x2) / 2.0
        cy = (draw_y1 + draw_y2) / 2.0

        polylines: list[Polyline] = []

        for i in range(1, count + 1):
            if cancelled_callback and cancelled_callback():
                break

            r = i * spacing

            if shape == "Circle":
                n_pts = max(64, int(r * 4))
                pts: Polyline = [
                    (cx + r * math.cos(_TWO_PI * j / n_pts), cy + r * math.sin(_TWO_PI * j / n_pts))
                    for j in range(n_pts + 1)
                ]
                polylines.append(pts)
            elif shape == "Square":
                half = r
                sq: Polyline = [
                    (cx - half, cy - half),
                    (cx + half, cy - half),
                    (cx + half, cy + half),
                    (cx - half, cy + half),
                    (cx - half, cy - half),
                ]
                polylines.append(sq)
            elif shape == "Polygon":
                n = max(3, sides)
                pts = [
                    (cx + r * math.cos(_TWO_PI * j / n - math.pi / 2), cy + r * math.sin(_TWO_PI * j / n - math.pi / 2))
                    for j in range(n + 1)
                ]
                polylines.append(pts)

            if progress_callback:
                progress_callback(int(i / count * 100))

        if progress_callback:
            progress_callback(100)

        return polylines

    def _generate_islamic(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        """Generate Islamic geometric tiling patterns (6-, 8-, or 12-point star rosettes)."""
        islamic_type = str(params.get("islamic_type", "8-Point Stars"))
        tile_size = float(params.get("tile_size_mm", 20.0))
        star_inset = float(params.get("star_inset", 0.15))

        x1, y1, x2, y2 = canvas.drawing_area()
        polylines: list[Polyline] = []

        if islamic_type == "6-Point Stars":
            # Pointy-top hexagonal grid; each cell contains a hexagram (two overlapping triangles).
            # For pointy-top hexagons: flat-to-flat width = R * sqrt(3), height = 2*R.
            # Grid: column spacing = R * sqrt(3), row spacing = R * 1.5.
            # Odd rows offset right by col_step / 2.
            R = tile_size
            col_step = R * math.sqrt(3)
            row_step = R * 1.5
            outer_r = R * (1.0 - star_inset)

            row_min = int((y1 - R) / row_step) - 1
            row_max = int((y2 + R) / row_step) + 2
            col_min = int((x1 - R) / col_step) - 1
            col_max = int((x2 + R) / col_step) + 2

            total = max(1, (row_max - row_min) * (col_max - col_min))
            done = 0

            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    if cancelled_callback and cancelled_callback():
                        return polylines

                    cx = x1 + col * col_step + (row % 2) * col_step * 0.5
                    cy = y1 + row * row_step

                    # Hexagram: two equilateral triangles.
                    # Triangle 1: vertices at 90°, 210°, 330° (pointy-top, pointing up first).
                    t1: Polyline = []
                    for k in range(3):
                        angle = math.radians(90 + k * 120)
                        t1.append((cx + outer_r * math.cos(angle), cy - outer_r * math.sin(angle)))
                    t1.append(t1[0])
                    polylines.append(t1)

                    # Triangle 2: vertices at 270°, 30°, 150° (pointing down first).
                    t2: Polyline = []
                    for k in range(3):
                        angle = math.radians(270 + k * 120)
                        t2.append((cx + outer_r * math.cos(angle), cy - outer_r * math.sin(angle)))
                    t2.append(t2[0])
                    polylines.append(t2)

                    done += 1
                    if progress_callback:
                        progress_callback(int(done / total * 100))

        elif islamic_type == "8-Point Stars":
            # Square grid; each cell contains an octagram (two overlapping squares).
            T = tile_size
            R = T * (0.5 - star_inset * 0.5)

            col_min = int((x1 - T) / T) - 1
            col_max = int((x2 + T) / T) + 2
            row_min = int((y1 - T) / T) - 1
            row_max = int((y2 + T) / T) + 2

            total = max(1, (row_max - row_min) * (col_max - col_min))
            done = 0

            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    if cancelled_callback and cancelled_callback():
                        return polylines

                    cx = x1 + col * T + T * 0.5
                    cy = y1 + row * T + T * 0.5

                    # Square 1: vertices at 0°, 90°, 180°, 270° (axis-aligned).
                    sq1: Polyline = []
                    for k in range(4):
                        angle = k * math.pi / 2
                        sq1.append((cx + R * math.cos(angle), cy - R * math.sin(angle)))
                    sq1.append(sq1[0])
                    polylines.append(sq1)

                    # Square 2: vertices at 45°, 135°, 225°, 315° (diagonal).
                    sq2: Polyline = []
                    for k in range(4):
                        angle = math.pi / 4 + k * math.pi / 2
                        sq2.append((cx + R * math.cos(angle), cy - R * math.sin(angle)))
                    sq2.append(sq2[0])
                    polylines.append(sq2)

                    done += 1
                    if progress_callback:
                        progress_callback(int(done / total * 100))

        elif islamic_type == "12-Point Stars":
            # Hexagonal grid (same as 6-fold) with dodecagram (two overlapping hexagons).
            T = tile_size
            R = T * (0.5 - star_inset * 0.5)

            col_step = T * math.sqrt(3)
            row_step = T * 1.5

            row_min = int((y1 - T) / row_step) - 1
            row_max = int((y2 + T) / row_step) + 2
            col_min = int((x1 - T) / col_step) - 1
            col_max = int((x2 + T) / col_step) + 2

            total = max(1, (row_max - row_min) * (col_max - col_min))
            done = 0

            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    if cancelled_callback and cancelled_callback():
                        return polylines

                    cx = x1 + col * col_step + (row % 2) * col_step * 0.5
                    cy = y1 + row * row_step

                    # Dodecagram: two regular hexagons rotated 30° relative to each other.
                    # Hexagon 1: vertices at 0°, 60°, 120°, 180°, 240°, 300°.
                    hex1: Polyline = []
                    for k in range(6):
                        angle = k * math.pi / 3
                        hex1.append((cx + R * math.cos(angle), cy - R * math.sin(angle)))
                    hex1.append(hex1[0])
                    polylines.append(hex1)

                    # Hexagon 2: vertices at 30°, 90°, 150°, 210°, 270°, 330°.
                    hex2: Polyline = []
                    for k in range(6):
                        angle = math.pi / 6 + k * math.pi / 3
                        hex2.append((cx + R * math.cos(angle), cy - R * math.sin(angle)))
                    hex2.append(hex2[0])
                    polylines.append(hex2)

                    done += 1
                    if progress_callback:
                        progress_callback(int(done / total * 100))

        if progress_callback:
            progress_callback(100)

        return polylines

    def _generate_celtic(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any,
        cancelled_callback: Any,
    ) -> list[Polyline]:
        """Generate a Celtic diagonal plait (two interlacing diagonal strand sets).

        Two sets of strands cross the grid at ±45°.  At each crossing the "under"
        strand is drawn with a gap around the crossing centre, creating the classic
        over-under interlace pattern.

        Type-A strands go from lower-left to upper-right (NE in screen coords).
          Cells on a Type-A strand share the same value of  col + row.
        Type-B strands go from upper-left to lower-right (SE in screen coords).
          Cells on a Type-B strand share the same value of  col - row.

        At cell (col, row):
          - (col + row) % 2 == 0  →  Type-A is over, Type-B has a gap
          - (col + row) % 2 == 1  →  Type-B is over, Type-A has a gap
        """
        knot_cols = max(1, int(params.get("knot_cols", 6)))
        knot_rows = max(1, int(params.get("knot_rows", 6)))
        T = float(params.get("tile_size_mm", 12.0))
        gap_mm = float(params.get("gap_mm", 0.8))

        x1, y1, x2, y2 = canvas.drawing_area()

        # Centre the knot grid on the drawing area.
        grid_w = knot_cols * T
        grid_h = knot_rows * T
        ox = (x1 + x2 - grid_w) / 2.0
        oy = (y1 + y2 - grid_h) / 2.0

        half = T / 2.0
        # gap_half is the distance from the crossing centre to the gap endpoint,
        # measured along the strand direction.  Each axis component is gap_half/√2.
        gap_half = gap_mm / 2.0
        gx = gap_half / math.sqrt(2)
        gy = gap_half / math.sqrt(2)

        polylines: list[Polyline] = []
        total_cells = knot_cols * knot_rows
        done = 0

        # ── Type-A strands (col + row = s, constant; strands go NE in screen) ──────
        for s in range(knot_cols + knot_rows - 1):
            c_lo = max(0, s - (knot_rows - 1))
            c_hi = min(knot_cols - 1, s)
            if c_lo > c_hi:
                continue

            current: list[tuple[float, float]] = []

            for c in range(c_lo, c_hi + 1):
                r = s - c
                if cancelled_callback and cancelled_callback():
                    return polylines

                cx = ox + (c + 0.5) * T
                cy = oy + (r + 0.5) * T

                # Segment enters from lower-left, exits to upper-right.
                a_enter = (cx - half, cy + half)
                a_exit = (cx + half, cy - half)

                if (c + r) % 2 == 0:  # Type-A is over: draw full segment.
                    if not current:
                        current = [a_enter, a_exit]
                    else:
                        current.append(a_exit)
                else:  # Type-A is under: gap at crossing centre.
                    # Endpoint just before the gap (going NE: x+, y−).
                    gap_before = (cx - gx, cy + gy)
                    # Endpoint just after the gap.
                    gap_after = (cx + gx, cy - gy)

                    if not current:
                        if a_enter != gap_before:
                            current = [a_enter, gap_before]
                    else:
                        current.append(gap_before)

                    if len(current) >= 2:
                        polylines.append(current)
                    current = [gap_after, a_exit]

                done += 1
                if progress_callback:
                    progress_callback(int(done / max(1, total_cells * 2) * 100))

            if len(current) >= 2:
                polylines.append(current)

        # ── Type-B strands (col − row = d, constant; strands go SE in screen) ──────
        for d in range(-(knot_rows - 1), knot_cols):
            c_lo = max(0, d)
            c_hi = min(knot_cols - 1, d + knot_rows - 1)
            if c_lo > c_hi:
                continue

            current = []

            for c in range(c_lo, c_hi + 1):
                r = c - d
                if cancelled_callback and cancelled_callback():
                    return polylines

                cx = ox + (c + 0.5) * T
                cy = oy + (r + 0.5) * T

                # Segment enters from upper-left, exits to lower-right.
                b_enter = (cx - half, cy - half)
                b_exit = (cx + half, cy + half)

                if (c + r) % 2 == 1:  # Type-B is over: draw full segment.
                    if not current:
                        current = [b_enter, b_exit]
                    else:
                        current.append(b_exit)
                else:  # Type-B is under: gap at crossing centre.
                    gap_before = (cx - gx, cy - gy)
                    gap_after = (cx + gx, cy + gy)

                    if not current:
                        if b_enter != gap_before:
                            current = [b_enter, gap_before]
                    else:
                        current.append(gap_before)

                    if len(current) >= 2:
                        polylines.append(current)
                    current = [gap_after, b_exit]

                done += 1
                if progress_callback:
                    progress_callback(int(done / max(1, total_cells * 2) * 100))

            if len(current) >= 2:
                polylines.append(current)

        if progress_callback:
            progress_callback(100)

        return polylines
