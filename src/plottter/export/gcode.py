"""G-code export — for CNC-style plotters (AxiDraw in G-code mode, custom builds)."""

from __future__ import annotations

import os
import re

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project


def _safe_filename(name: str) -> str:
    """Strip characters that are invalid in file names."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return re.sub(r'\s+', '_', sanitized)


def _write_layer_gcode(layer: Layer, settings: dict) -> str:
    """Build a G-code command string for a single layer.

    Args:
        layer: The layer whose paths to export.
        settings: Export options dict.

    Returns:
        G-code string.
    """
    travel_speed: int = int(settings.get("travel_speed", 3000))
    draw_speed: int = int(settings.get("draw_speed", 1000))
    pen_up_angle: int = int(settings.get("pen_up_angle", 0))
    pen_down_angle: int = int(settings.get("pen_down_angle", 90))
    precision: int = int(settings.get("precision", 3))

    lines: list[str] = []

    # Preamble
    lines.append("G90")           # Absolute positioning
    lines.append("G21")           # Units in mm
    lines.append("G28")           # Home
    lines.append(f"M3 S{pen_up_angle}")   # Pen up

    for path in layer.paths:
        if len(path) < 2:
            continue

        x0, y0 = path[0]
        # Rapid move to start of path (pen up)
        lines.append(f"G0 X{x0:.{precision}f} Y{y0:.{precision}f} F{travel_speed}")
        # Pen down
        lines.append(f"M3 S{pen_down_angle}")
        # Draw through all remaining points
        for x, y in path[1:]:
            lines.append(f"G1 X{x:.{precision}f} Y{y:.{precision}f} F{draw_speed}")
        # Pen up after each path
        lines.append(f"M3 S{pen_up_angle}")

    # Epilogue
    lines.append("G28")   # Home at end
    lines.append("M5")    # Disable servo

    return "\n".join(lines) + "\n"


def export_layer_gcode(
    layer: Layer,
    canvas: Canvas,
    filepath: str,
    settings: dict,
) -> None:
    """Export a single *layer* to a G-code file at *filepath*.

    Args:
        layer: The layer to export.
        canvas: Canvas (unused for G-code but kept for API consistency).
        filepath: Destination file path (will be created/overwritten).
        settings: Export options dict with keys:
            - ``travel_speed`` (int, default 3000): Speed when pen is up (mm/min).
            - ``draw_speed`` (int, default 1000): Speed when pen is down (mm/min).
            - ``pen_up_angle`` (int, default 0): Servo angle for pen-up position.
            - ``pen_down_angle`` (int, default 90): Servo angle for pen-down position.
            - ``precision`` (int, default 3): Decimal places for coordinates.
    """
    content = _write_layer_gcode(layer, settings)
    with open(filepath, "w", encoding="ascii") as fh:
        fh.write(content)


def export_all_layers_gcode(
    project: Project,
    output_dir: str,
    settings: dict,
) -> None:
    """Export each visible layer of *project* to its own G-code file inside *output_dir*.

    File naming: ``{project_name}_{layer_number:02d}_{layer_name}.gcode``

    Args:
        project: The project to export.
        output_dir: Directory where G-code files will be written (created if absent).
        settings: Export options dict (same keys as :func:`export_layer_gcode`).
    """
    os.makedirs(output_dir, exist_ok=True)
    project_name = _safe_filename(project.name)
    visible_layers = [lyr for lyr in project.layers if lyr.visible]

    for idx, layer in enumerate(visible_layers, start=1):
        layer_name = _safe_filename(layer.name)
        filename = f"{project_name}_{idx:02d}_{layer_name}.gcode"
        filepath = os.path.join(output_dir, filename)
        content = _write_layer_gcode(layer, settings)
        with open(filepath, "w", encoding="ascii") as fh:
            fh.write(content)
