"""HPGL export — for legacy HP and compatible pen plotters."""

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


def _mm_to_hpgl(x_mm: float, y_mm: float, canvas_height_mm: float) -> tuple[int, int]:
    """Convert mm coordinates to HPGL plotter units.

    HPGL uses 1 unit = 0.025 mm (40 units/mm).
    Y-axis is inverted: origin is bottom-left.
    """
    hx = int(x_mm * 40)
    hy = int((canvas_height_mm - y_mm) * 40)
    return hx, hy


def _write_layer_hpgl(
    layer: Layer,
    canvas: Canvas,
    pen_number: int,
    settings: dict,
) -> str:
    """Build an HPGL command string for a single layer.

    Args:
        layer: The layer to export.
        canvas: Canvas for coordinate conversion.
        pen_number: HPGL pen number (1–8) assigned to this layer.
        settings: Export options (speed, force).

    Returns:
        HPGL command string (no trailing newline).
    """
    h = canvas.height_mm
    lines: list[str] = []

    lines.append("IN;")

    speed = settings.get("speed")
    force = settings.get("force")
    if speed is not None:
        lines.append(f"VS{int(speed)};")
    if force is not None:
        lines.append(f"FS{int(force)};")

    lines.append(f"SP{pen_number};")

    for path in layer.paths:
        if len(path) < 2:
            continue
        # Move to start with pen up
        x0, y0 = _mm_to_hpgl(path[0][0], path[0][1], h)
        lines.append(f"PU{x0},{y0};")
        # Draw through remaining points
        coord_pairs = ",".join(
            f"{_mm_to_hpgl(x, y, h)[0]},{_mm_to_hpgl(x, y, h)[1]}"
            for x, y in path[1:]
        )
        lines.append(f"PD{coord_pairs};")

    lines.append("PU;")
    return "\n".join(lines)


def export_layer_hpgl(
    layer: Layer,
    canvas: Canvas,
    filepath: str,
    settings: dict,
) -> None:
    """Export a single *layer* to an HPGL file at *filepath*.

    Args:
        layer: The layer to export.
        canvas: Canvas that defines paper dimensions and coordinate space.
        filepath: Destination file path (will be created/overwritten).
        settings: Export options dict with keys:
            - ``pen_number`` (int, default 1): HPGL pen number for this layer.
            - ``speed`` (int | None): Optional VS speed command value.
            - ``force`` (int | None): Optional FS force command value.
    """
    pen_number: int = int(settings.get("pen_number", 1))
    content = _write_layer_hpgl(layer, canvas, pen_number, settings)
    with open(filepath, "w", encoding="ascii") as fh:
        fh.write(content)
        fh.write("\n")


def export_all_layers_hpgl(
    project: Project,
    output_dir: str,
    settings: dict,
) -> None:
    """Export each visible layer of *project* to its own HPGL file inside *output_dir*.

    File naming: ``{project_name}_{layer_number:02d}_{layer_name}.plt``
    Pen numbers are assigned sequentially (1-based) per layer.

    Args:
        project: The project to export.
        output_dir: Directory where HPGL files will be written (created if absent).
        settings: Export options dict (same keys as :func:`export_layer_hpgl`).
    """
    os.makedirs(output_dir, exist_ok=True)
    project_name = _safe_filename(project.name)
    visible_layers = [lyr for lyr in project.layers if lyr.visible]

    for idx, layer in enumerate(visible_layers, start=1):
        layer_name = _safe_filename(layer.name)
        filename = f"{project_name}_{idx:02d}_{layer_name}.plt"
        filepath = os.path.join(output_dir, filename)
        layer_settings = dict(settings)
        layer_settings["pen_number"] = idx
        content = _write_layer_hpgl(layer, project.canvas, idx, layer_settings)
        with open(filepath, "w", encoding="ascii") as fh:
            fh.write(content)
            fh.write("\n")
