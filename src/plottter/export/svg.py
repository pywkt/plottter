"""SVG export — primary export format for plotter art."""

from __future__ import annotations

import os
import re

import svgwrite

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project


def _safe_filename(name: str) -> str:
    """Strip characters that are invalid in file names or XML IDs."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return re.sub(r'\s+', '_', sanitized)


def _add_registration_marks(
    dwg: svgwrite.Drawing,
    canvas: Canvas,
    style: str = "corners",
) -> None:
    """Add registration crosshairs to *dwg* in a <g id="registration"> group.

    Args:
        dwg: The svgwrite Drawing to add marks to.
        canvas: Canvas whose drawing area defines mark placement.
        style: "corners", "center", or "both".
    """
    arm = 3.0  # mm — arm half-length
    sw = 0.1   # mm — stroke width

    reg = dwg.g(id="registration")

    def _cross(cx: float, cy: float) -> None:
        """Add a crosshair centred at (cx, cy)."""
        reg.add(dwg.line(
            start=(f"{cx - arm:.3f}mm", f"{cy:.3f}mm"),
            end=(f"{cx + arm:.3f}mm", f"{cy:.3f}mm"),
            stroke="#000000",
            stroke_width=f"{sw}mm",
        ))
        reg.add(dwg.line(
            start=(f"{cx:.3f}mm", f"{cy - arm:.3f}mm"),
            end=(f"{cx:.3f}mm", f"{cy + arm:.3f}mm"),
            stroke="#000000",
            stroke_width=f"{sw}mm",
        ))

    x0, y0, x1, y1 = canvas.drawing_area()  # drawing area extents

    if style in ("corners", "both"):
        _cross(x0, y0)
        _cross(x1, y0)
        _cross(x0, y1)
        _cross(x1, y1)

    if style in ("center", "both"):
        cx = canvas.width_mm / 2
        cy = canvas.height_mm / 2
        _cross(cx, cy)

    dwg.add(reg)


def _layer_group(
    dwg: svgwrite.Drawing,
    layer: Layer,
    stroke_width: float = 0.3,
) -> svgwrite.container.Group:
    """Build a <g> element with all polylines from *layer*."""
    g = dwg.g(
        id=f"layer_{_safe_filename(layer.name)}",
        stroke=layer.color,
        fill="none",
        **{"stroke-width": f"{stroke_width:.3f}mm"},
    )
    for path in layer.paths:
        if len(path) < 2:
            continue
        points = [(f"{x:.3f}", f"{y:.3f}") for x, y in path]
        g.add(dwg.polyline(points=points))
    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_layer_svg(
    layer: Layer,
    canvas: Canvas,
    filepath: str,
    settings: dict,
) -> None:
    """Export a single *layer* to an SVG file at *filepath*.

    Args:
        layer: The layer to export.
        canvas: Canvas that defines paper dimensions.
        filepath: Destination file path (will be created/overwritten).
        settings: Export options dict with keys:
            - ``registration_marks`` (bool, default True)
            - ``stroke_width`` (float mm, default 0.3)
    """
    stroke_width: float = float(settings.get("stroke_width", 0.3))
    reg_marks: bool = bool(settings.get("registration_marks", True))
    reg_style: str = settings.get("reg_mark_style", "corners")

    dwg = svgwrite.Drawing(
        filename=filepath,
        size=(f"{canvas.width_mm}mm", f"{canvas.height_mm}mm"),
        viewBox=f"0 0 {canvas.width_mm} {canvas.height_mm}",
    )

    if reg_marks:
        _add_registration_marks(dwg, canvas, style=reg_style)

    dwg.add(_layer_group(dwg, layer, stroke_width))
    dwg.save(pretty=True)


def export_all_layers_svg(
    project: Project,
    output_dir: str,
    settings: dict,
) -> None:
    """Export each visible layer of *project* to its own SVG file inside *output_dir*.

    File naming: ``{project_name}_{layer_number:02d}_{layer_name}.svg``

    Args:
        project: The project to export.
        output_dir: Directory where SVG files will be written (must exist or will be created).
        settings: Export options dict (same keys as :func:`export_layer_svg`).
    """
    os.makedirs(output_dir, exist_ok=True)

    stroke_width: float = float(settings.get("stroke_width", 0.3))
    reg_marks: bool = bool(settings.get("registration_marks", True))
    reg_style: str = settings.get("reg_mark_style", "corners")
    project_name = _safe_filename(project.name)

    visible_layers = [lyr for lyr in project.layers if lyr.visible]
    for idx, layer in enumerate(visible_layers, start=1):
        layer_name = _safe_filename(layer.name)
        filename = f"{project_name}_{idx:02d}_{layer_name}.svg"
        filepath = os.path.join(output_dir, filename)

        dwg = svgwrite.Drawing(
            filename=filepath,
            size=(f"{project.canvas.width_mm}mm", f"{project.canvas.height_mm}mm"),
            viewBox=f"0 0 {project.canvas.width_mm} {project.canvas.height_mm}",
        )
        if reg_marks:
            _add_registration_marks(dwg, project.canvas, style=reg_style)
        dwg.add(_layer_group(dwg, layer, stroke_width))
        dwg.save(pretty=True)


def export_combined_svg(
    project: Project,
    filepath: str,
    settings: dict,
) -> None:
    """Export all visible layers into a single SVG file with separate <g> groups.

    Args:
        project: The project to export.
        filepath: Destination SVG file path.
        settings: Export options dict (same keys as :func:`export_layer_svg`).
    """
    stroke_width: float = float(settings.get("stroke_width", 0.3))
    reg_marks: bool = bool(settings.get("registration_marks", True))
    reg_style: str = settings.get("reg_mark_style", "corners")

    dwg = svgwrite.Drawing(
        filename=filepath,
        size=(f"{project.canvas.width_mm}mm", f"{project.canvas.height_mm}mm"),
        viewBox=f"0 0 {project.canvas.width_mm} {project.canvas.height_mm}",
    )

    if reg_marks:
        _add_registration_marks(dwg, project.canvas, style=reg_style)

    for layer in project.layers:
        if layer.visible:
            dwg.add(_layer_group(dwg, layer, stroke_width))

    dwg.save(pretty=True)
