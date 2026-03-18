"""Mural plotter export — plain-text command format for wall-mounted plotters."""

from __future__ import annotations

import math
import os
import re

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project


def _safe_filename(name: str) -> str:
    """Strip characters that are invalid in file names."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return re.sub(r'\s+', '_', sanitized)


def _build_mural_content(
    layer: Layer,
    canvas: Canvas,
    top_distance: float,
) -> tuple[str, list[str]]:
    """Build Mural command file content for a single layer.

    The Mural coordinate system has:
    - Origin at top-left.
    - Y increases downward (same as Qt/plottter convention).
    - Width derived from pin distance: ``width = top_distance * 0.6``.
    - Home position at ``(width/2, 350)``.

    The drawing (paper area) is centred on the home position:
    - ``x_offset = (width - paper_width) / 2``
    - ``y_offset = 350 - (paper_height / 2)``

    Args:
        layer: The layer to export.
        canvas: Canvas that defines paper dimensions and coordinate space.
        top_distance: Distance between the two wall-mounted anchor pins (mm).

    Returns:
        A ``(content, warnings)`` tuple where *content* is the full file text
        (including headers) and *warnings* is a (possibly empty) list of human-
        readable warning strings for out-of-bounds coordinates.
    """
    width = top_distance * 0.6
    home_y = 350.0
    x_offset = (width - canvas.width_mm) / 2.0
    y_offset = home_y - (canvas.height_mm / 2.0)

    lines: list[str] = []
    warnings: list[str] = []
    total_distance = 0.0
    drawing_height = canvas.height_mm

    # Current pen position for distance tracking (start at home)
    pen_x = width / 2.0
    pen_y = home_y

    def _to_mural(x_mm: float, y_mm: float) -> tuple[float, float]:
        return x_offset + x_mm, y_offset + y_mm

    # --- First pass: collect all commands to compute distance & detect OOB ---
    commands: list[str] = []

    def _move(mx: float, my: float) -> None:
        nonlocal pen_x, pen_y, total_distance
        dx = mx - pen_x
        dy = my - pen_y
        total_distance += math.sqrt(dx * dx + dy * dy)
        pen_x, pen_y = mx, my
        commands.append(f"{mx:.1f} {my:.1f}")
        # Bounds check
        if mx < 0 or mx > width or my < 0:
            warnings.append(
                f"Coordinate ({mx:.1f}, {my:.1f}) is outside the valid "
                f"drawing area (x: 0–{width:.1f}, y ≥ 0)."
            )

    for path in layer.paths:
        if len(path) < 2:
            continue
        commands.append("p0")
        mx0, my0 = _to_mural(path[0][0], path[0][1])
        _move(mx0, my0)
        commands.append("p1")
        for pt in path[1:]:
            mx, my = _to_mural(pt[0], pt[1])
            _move(mx, my)

    commands.append("p0")

    # --- Assemble final content ---
    header = [
        f"d{total_distance:.1f}",
        f"h{drawing_height:.1f}",
    ]
    content = "\n".join(header + commands) + "\n"
    return content, warnings


def export_layer_mural(
    layer: Layer,
    canvas: Canvas,
    filepath: str,
    settings: dict,
) -> list[str]:
    """Export a single *layer* to a Mural plotter file at *filepath*.

    Args:
        layer: The layer to export.
        canvas: Canvas that defines paper dimensions and coordinate space.
        filepath: Destination file path (will be created/overwritten).
        settings: Export options dict with keys:
            - ``top_distance`` (float, default 1025): Distance between anchor
              pins in mm. Drawing area width = ``top_distance * 0.6``.

    Returns:
        A (possibly empty) list of warning strings for out-of-bounds coordinates.
    """
    top_distance: float = float(settings.get("top_distance", 1025.0))
    content, warnings = _build_mural_content(layer, canvas, top_distance)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(content)
    return warnings


def export_all_layers_mural(
    project: Project,
    output_dir: str,
    settings: dict,
) -> list[str]:
    """Export each visible layer of *project* to its own Mural file inside *output_dir*.

    File naming: ``{project_name}_{layer_number:02d}_{layer_name}.mural``

    Args:
        project: The project to export.
        output_dir: Directory where Mural files will be written (created if absent).
        settings: Export options dict (same keys as :func:`export_layer_mural`).

    Returns:
        A (possibly empty) list of warning strings for out-of-bounds coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    project_name = _safe_filename(project.name)
    visible_layers = [lyr for lyr in project.layers if lyr.visible]
    all_warnings: list[str] = []

    for idx, layer in enumerate(visible_layers, start=1):
        layer_name = _safe_filename(layer.name)
        filename = f"{project_name}_{idx:02d}_{layer_name}.mural"
        filepath = os.path.join(output_dir, filename)
        warnings = export_layer_mural(layer, project.canvas, filepath, settings)
        all_warnings.extend(warnings)

    return all_warnings
