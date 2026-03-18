"""LSystemGenerator — L-system string expansion with turtle graphics interpreter."""

from __future__ import annotations

import math
from typing import Any

from plottter.generators.base import (
    ExpressionParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.generators import register_generator
from plottter.models import Canvas, Polyline

# Maximum string length to prevent exponential blowup
_MAX_STRING_LENGTH = 2_000_000


def expand_lsystem(axiom: str, rules: dict[str, str], iterations: int) -> str:
    """Expand an L-system axiom by applying rules for the given number of iterations."""
    current = axiom
    for _ in range(iterations):
        result: list[str] = []
        for ch in current:
            result.append(rules.get(ch, ch))
        current = "".join(result)
        if len(current) > _MAX_STRING_LENGTH:
            current = current[:_MAX_STRING_LENGTH]
            break
    return current


def parse_rules(rules_str: str) -> dict[str, str]:
    """Parse semicolon-separated rules like 'F=F+F-F;G=GG' into a dict."""
    rules: dict[str, str] = {}
    for part in rules_str.split(";"):
        part = part.strip()
        if "=" in part:
            lhs, _, rhs = part.partition("=")
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs:
                rules[lhs] = rhs
    return rules


def turtle_to_polylines(
    lstring: str,
    angle_deg: float,
    step_length: float,
) -> list[list[tuple[float, float]]]:
    """Interpret an L-system string as turtle graphics commands.

    Returns a list of polylines (each polyline is a contiguous drawn segment).
    Turtle commands:
      F, G : move forward and draw
      f, g : move forward without drawing
      + : turn right (clockwise) by angle
      - : turn left (counter-clockwise) by angle
      [ : push state (position + heading)
      ] : pop state
    All other characters are ignored.
    """
    angle_rad = math.radians(angle_deg)
    x, y = 0.0, 0.0
    heading = -math.pi / 2  # start pointing up

    stack: list[tuple[float, float, float]] = []
    polylines: list[list[tuple[float, float]]] = []
    current_line: list[tuple[float, float]] = [(x, y)]
    pen_down = True

    for ch in lstring:
        if ch in ("F", "G"):
            if not pen_down:
                # Resume drawing from current position after a pen-up move
                current_line = [(x, y)]
                pen_down = True
            nx = x + step_length * math.cos(heading)
            ny = y + step_length * math.sin(heading)
            if not current_line:
                current_line = [(x, y)]
            current_line.append((nx, ny))
            x, y = nx, ny
        elif ch in ("f", "g"):
            nx = x + step_length * math.cos(heading)
            ny = y + step_length * math.sin(heading)
            # Pen up move: end current line, start new position
            if len(current_line) > 1:
                polylines.append(current_line)
            current_line = []
            pen_down = False
            x, y = nx, ny
        elif ch == "+":
            heading += angle_rad
        elif ch == "-":
            heading -= angle_rad
        elif ch == "[":
            stack.append((x, y, heading))
        elif ch == "]":
            if stack:
                # Save current line before popping
                if len(current_line) > 1:
                    polylines.append(current_line)
                current_line = []
                x, y, heading = stack.pop()
                # After popping, we're at a new position — start a new line from here
                pen_down = True
        # Ignore all other characters

    if len(current_line) > 1:
        polylines.append(current_line)

    return polylines


@register_generator
class LSystemGenerator(Generator):
    """Generates fractal patterns via L-system string expansion and turtle graphics."""

    name = "L-System / Fractal"
    category = "math"

    def get_parameters(self) -> list[Parameter]:
        return [
            ExpressionParam(
                name="axiom",
                label="Axiom",
                default="F",
                variables=[],
                description="Starting string for the L-system (e.g. 'F'). This is the initial state that gets expanded by the rules",
            ),
            ExpressionParam(
                name="rules",
                label="Rules (e.g. F=F+F-F)",
                default="F=F+F-F+F",
                variables=[],
                description="Rewriting rules in the format 'F=F+F-F' — semicolon-separated for multiple rules. F/G = draw forward, f/g = move without drawing, + = turn right, - = turn left, [ ] = push/pop state",
            ),
            IntParam(
                name="iterations",
                label="Iterations",
                min=1,
                max=10,
                step=1,
                default=4,
                description="Number of times to apply the rules — more iterations create more detail but exponentially more points",
            ),
            FloatParam(
                name="angle_deg",
                label="Turn angle (degrees)",
                min=0.1,
                max=360.0,
                step=0.1,
                default=90.0,
                description="Angle for + (turn right) and - (turn left) commands in degrees",
            ),
            FloatParam(
                name="step_length_mm",
                label="Step length (mm)",
                min=0.01,
                max=100.0,
                step=0.1,
                default=5.0,
                description="Length of each F/G drawing step in millimeters (before auto-fit scaling)",
            ),
            FloatParam(
                name="scale",
                label="Scale (0 = auto-fit)",
                min=0.0,
                max=100.0,
                step=0.01,
                default=0.0,
                description="Manual scale factor (0 = auto-fit to canvas drawing area)",
            ),
            FloatParam(
                name="rotation_deg",
                label="Rotation (degrees)",
                min=-360.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the output by this many degrees around the center",
            ),
            FloatParam(
                name="x_offset_mm",
                label="X offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Shift the output horizontally in millimeters",
            ),
            FloatParam(
                name="y_offset_mm",
                label="Y offset (mm)",
                min=-500.0,
                max=500.0,
                step=1.0,
                default=0.0,
                randomizable=False,
                description="Shift the output vertically in millimeters",
            ),
        ]

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Koch Snowflake",
                params={
                    "axiom": "F--F--F",
                    "rules": "F=F+F--F+F",
                    "iterations": 4,
                    "angle_deg": 60.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Sierpinski Triangle",
                params={
                    "axiom": "F-G-G",
                    "rules": "F=F-G+F+G-F;G=GG",
                    "iterations": 5,
                    "angle_deg": 120.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Dragon Curve",
                params={
                    "axiom": "F",
                    "rules": "F=F+G;G=F-G",
                    "iterations": 10,
                    "angle_deg": 90.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Plant / Tree",
                params={
                    "axiom": "X",
                    "rules": "X=F+[[X]-X]-F[-FX]+X;F=FF",
                    "iterations": 5,
                    "angle_deg": 25.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Hilbert Curve",
                params={
                    "axiom": "A",
                    "rules": "A=-BF+AFA+FB-;B=+AF-BFB-FA+",
                    "iterations": 5,
                    "angle_deg": 90.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
                    "x_offset_mm": 0.0,
                    "y_offset_mm": 0.0,
                },
            ),
            Preset(
                name="Gosper Curve",
                params={
                    "axiom": "F",
                    "rules": "F=F-G--G+F++FF+G-;G=+F-GG--G-F++F+G",
                    "iterations": 4,
                    "angle_deg": 60.0,
                    "step_length_mm": 5.0,
                    "scale": 0.0,
                    "rotation_deg": 0.0,
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
        axiom = str(params.get("axiom", "F"))
        rules_str = str(params.get("rules", "F=F+F-F+F"))
        iterations = int(params.get("iterations", 4))
        angle_deg = float(params.get("angle_deg", 90.0))
        step_length_mm = float(params.get("step_length_mm", 5.0))
        scale = float(params.get("scale", 0.0))
        rotation_deg = float(params.get("rotation_deg", 0.0))
        x_offset_mm = float(params.get("x_offset_mm", 0.0))
        y_offset_mm = float(params.get("y_offset_mm", 0.0))

        if progress_callback:
            progress_callback(5)

        rules = parse_rules(rules_str)
        lstring = expand_lsystem(axiom, rules, iterations)

        if progress_callback:
            progress_callback(20)

        if cancelled_callback and cancelled_callback():
            return []

        raw_polylines = turtle_to_polylines(lstring, angle_deg, step_length_mm)

        if progress_callback:
            progress_callback(60)

        if not raw_polylines:
            return []

        # Collect all points to compute bounding box for auto-fit
        all_points = [pt for pl in raw_polylines for pt in pl]
        if not all_points:
            return []

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        raw_cx = (min(xs) + max(xs)) / 2.0
        raw_cy = (min(ys) + max(ys)) / 2.0

        # Center all polylines
        centered_polylines = [
            [(x - raw_cx, y - raw_cy) for x, y in pl]
            for pl in raw_polylines
        ]

        # Apply rotation
        if rotation_deg != 0.0:
            theta = math.radians(rotation_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            centered_polylines = [
                [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in pl]
                for pl in centered_polylines
            ]

        # Recompute bounding box after rotation
        all_centered = [pt for pl in centered_polylines for pt in pl]
        cxs = [p[0] for p in all_centered]
        cys = [p[1] for p in all_centered]
        span_x = (max(cxs) - min(cxs)) or 1.0
        span_y = (max(cys) - min(cys)) or 1.0

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1

        if scale == 0.0:
            scale_factor = min(draw_w / span_x, draw_h / span_y) * 0.9
        else:
            scale_factor = scale

        canvas_cx = (draw_x1 + draw_x2) / 2.0 + x_offset_mm
        canvas_cy = (draw_y1 + draw_y2) / 2.0 + y_offset_mm

        result: list[Polyline] = [
            [(x * scale_factor + canvas_cx, y * scale_factor + canvas_cy) for x, y in pl]
            for pl in centered_polylines
        ]

        if progress_callback:
            progress_callback(100)

        return result
