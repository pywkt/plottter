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


def export_layer_specs_svg(
    layer_specs: list,
    canvas: Canvas,
    filepath: str,
    settings: dict,
) -> None:
    """Export a list of LayerSpec objects to a single multi-layer SVG file.

    Each spec is written as a ``<g inkscape:groupmode="layer">`` element so
    Inkscape (and compatible viewers) treats each palette colour as a separate
    layer.

    Args:
        layer_specs: Sequence of objects with ``.name`` (str), ``.color`` (hex
            string), and ``.paths`` (list of polylines).  Compatible with
            :class:`~plottter.generators.base.LayerSpec`.
        canvas: Canvas that defines the paper dimensions.
        filepath: Destination SVG file path (will be created/overwritten).
        settings: Export options dict (same keys as :func:`export_layer_svg`).
    """
    stroke_width: float = float(settings.get("stroke_width", 0.3))
    reg_marks: bool = bool(settings.get("registration_marks", True))
    reg_style: str = settings.get("reg_mark_style", "corners")

    # Build SVG as raw XML strings to avoid svgwrite's attribute validator,
    # which rejects non-SVG-1.1 attributes like inkscape:groupmode.
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
            f' width="{canvas.width_mm}mm"'
            f' height="{canvas.height_mm}mm"'
            f' viewBox="0 0 {canvas.width_mm} {canvas.height_mm}">'
        ),
    ]

    if reg_marks:
        arm = 3.0  # mm — arm half-length
        sw = 0.1   # mm — stroke width
        x0, y0, x1, y1 = canvas.drawing_area()
        lines.append('  <g id="registration">')

        def _cross(cx: float, cy: float) -> None:
            lines.append(
                f'    <line x1="{cx - arm:.3f}" y1="{cy:.3f}"'
                f' x2="{cx + arm:.3f}" y2="{cy:.3f}"'
                f' stroke="#000000" stroke-width="{sw}mm"/>'
            )
            lines.append(
                f'    <line x1="{cx:.3f}" y1="{cy - arm:.3f}"'
                f' x2="{cx:.3f}" y2="{cy + arm:.3f}"'
                f' stroke="#000000" stroke-width="{sw}mm"/>'
            )

        if reg_style in ("corners", "both"):
            _cross(x0, y0)
            _cross(x1, y0)
            _cross(x0, y1)
            _cross(x1, y1)
        if reg_style in ("center", "both"):
            _cross(canvas.width_mm / 2, canvas.height_mm / 2)

        lines.append("  </g>")

    for spec in layer_specs:
        safe_id = _safe_filename(spec.name)
        # Escape XML attribute special characters in the label.
        label = spec.name.replace("&", "&amp;").replace('"', "&quot;")
        lines.append(
            f'  <g id="layer_{safe_id}"'
            f' inkscape:groupmode="layer"'
            f' inkscape:label="{label}"'
            f' stroke="{spec.color}"'
            f' fill="none"'
            f' stroke-width="{stroke_width:.3f}mm">'
        )
        for path in spec.paths:
            if len(path) < 2:
                continue
            pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in path)
            lines.append(f'    <polyline points="{pts}"/>')
        lines.append("  </g>")

    lines.append("</svg>")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
