"""AxiDraw direct plotter control via the official pyaxidraw Python API.

Usage requires the ``pyaxidraw`` package from Evil Mad Scientist::

    pip install pyaxidraw

If pyaxidraw is not installed, all public functions raise ``AxiDrawNotInstalledError``
with installation instructions.
"""

from __future__ import annotations

import io
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.models import Canvas, Layer
    from plottter.models.project import Project


def _safe_xml_id(name: str) -> str:
    """Strip characters that are invalid in XML IDs from a layer name."""
    sanitized = re.sub(r'[\\:*?"<>&|]', "_", name).strip()
    return re.sub(r'\s', '_', sanitized)


class AxiDrawNotInstalledError(RuntimeError):
    """Raised when pyaxidraw is not installed."""


class AxiDrawConnectionError(RuntimeError):
    """Raised when the AxiDraw device cannot be found or connected."""


def _require_axidraw():
    """Import and return the axidraw module, or raise AxiDrawNotInstalledError."""
    try:
        from pyaxidraw import axidraw as _ad  # type: ignore[import-untyped]
        return _ad
    except ImportError:
        raise AxiDrawNotInstalledError(
            "The pyaxidraw package is required for direct plotter control.\n"
            "Install it with:\n"
            "  pip install pyaxidraw\n"
            "For more information visit: https://axidraw.com/doc/py_api/"
        )


def check_axidraw_available() -> bool:
    """Return True if pyaxidraw is installed and importable."""
    try:
        _require_axidraw()
        return True
    except AxiDrawNotInstalledError:
        return False


def run_manual_command(
    command: str,
    settings: dict[str, Any],
) -> None:
    """Run a one-off manual AxiDraw command (raise pen, lower pen, release motors, …).

    Used by the dialog's manual-control buttons so the user can position the
    pen carriage by hand before locking the pen in.

    Parameters
    ----------
    command:
        One of: ``"raise_pen"``, ``"lower_pen"``, ``"disable_xy"`` (release motors),
        ``"enable_xy"`` (re-engage motors). These are pyaxidraw's documented
        manual-mode command names.
    settings:
        Same shape as :func:`plot_svg_string`'s settings. Pen positions
        (``pen_pos_up`` / ``pen_pos_down``), ``model``, ``port``, and
        ``preview`` are honored; other fields are ignored.
    """
    axidraw_module = _require_axidraw()
    ad = axidraw_module.AxiDraw()
    ad.plot_setup()  # No SVG needed for manual mode

    ad.options.mode = "manual"
    ad.options.manual_cmd = command
    ad.options.model = int(settings.get("model", 2))
    ad.options.pen_pos_up = int(settings.get("pen_pos_up", 60))
    ad.options.pen_pos_down = int(settings.get("pen_pos_down", 40))

    port = settings.get("port")
    if port:
        ad.options.port = port
    if bool(settings.get("preview", False)):
        ad.options.preview = True

    try:
        ad.plot_run()
    except Exception as exc:
        msg = str(exc)
        if "unable to find" in msg.lower() or "no axidraw" in msg.lower():
            raise AxiDrawConnectionError(
                "AxiDraw device not found. Make sure the device is connected via USB "
                "and powered on."
            ) from exc
        raise


def plot_svg_string(
    svg_data: str,
    settings: dict[str, Any],
    progress_callback: Any = None,
) -> None:
    """Send an SVG string directly to the AxiDraw plotter.

    Parameters
    ----------
    svg_data:
        Complete SVG document as a string.
    settings:
        Dict with optional keys:
        - ``speed_pendown`` (int 1-100, default 25)
        - ``speed_penup`` (int 1-100, default 75)
        - ``pen_pos_down`` (int 0-100, default 40) — servo position pen-down
        - ``pen_pos_up`` (int 0-100, default 60) — servo position pen-up
        - ``pen_delay_down`` (int ms, default 0)
        - ``pen_delay_up`` (int ms, default 0)
        - ``const_speed`` (bool, default False)
        - ``report_time`` (bool, default False)
        - ``model`` (int 1-6, default 2 — AxiDraw V3/A3)
        - ``port`` (str|None, default None — auto-detect)
        - ``preview`` (bool, default False — dry-run, no device needed)
    progress_callback:
        Optional callable(percent: float) for progress updates.
    """
    axidraw_module = _require_axidraw()

    if progress_callback:
        progress_callback(5)

    ad = axidraw_module.AxiDraw()
    ad.plot_setup(svg_data)

    # Apply settings
    ad.options.speed_pendown = int(settings.get("speed_pendown", 25))
    ad.options.speed_penup = int(settings.get("speed_penup", 75))
    ad.options.pen_pos_down = int(settings.get("pen_pos_down", 40))
    ad.options.pen_pos_up = int(settings.get("pen_pos_up", 60))
    ad.options.pen_delay_down = int(settings.get("pen_delay_down", 0))
    ad.options.pen_delay_up = int(settings.get("pen_delay_up", 0))
    ad.options.const_speed = bool(settings.get("const_speed", False))
    ad.options.report_time = bool(settings.get("report_time", False))
    ad.options.model = int(settings.get("model", 2))

    port = settings.get("port")
    if port:
        ad.options.port = port

    preview = bool(settings.get("preview", False))
    if preview:
        ad.options.preview = True

    if progress_callback:
        progress_callback(10)

    try:
        ad.plot_run()
    except Exception as exc:
        # Wrap device errors in a friendlier message
        msg = str(exc)
        if "unable to find" in msg.lower() or "no axidraw" in msg.lower():
            raise AxiDrawConnectionError(
                "AxiDraw device not found. Make sure the device is connected via USB "
                "and powered on."
            ) from exc
        raise

    if progress_callback:
        progress_callback(100)


def plot_project_layer(
    layer: "Layer",
    canvas: "Canvas",
    settings: dict[str, Any],
    progress_callback: Any = None,
) -> None:
    """Plot a single layer directly to the AxiDraw.

    Generates a temporary SVG from the layer and sends it to the plotter.
    """
    svg_data = _layer_to_svg_string(layer, canvas, settings)

    plot_svg_string(svg_data, settings, progress_callback)


def plot_project(
    project: "Project",
    layer_ids: list[str] | None,
    settings: dict[str, Any],
    progress_callback: Any = None,
) -> None:
    """Plot one or more layers from a project to the AxiDraw.

    Parameters
    ----------
    project:
        The project to plot.
    layer_ids:
        List of layer IDs to include; None means all visible layers.
    settings:
        Same settings dict as :func:`plot_svg_string`, plus:
        - ``per_layer_pause`` (bool, default False) — pause between layers
    progress_callback:
        Optional callable(percent: float).
    """
    if layer_ids is None:
        layers = [lyr for lyr in project.layers if lyr.visible]
    else:
        layers = [lyr for lyr in project.layers if lyr.id in layer_ids]

    if not layers:
        return

    per_layer_pause = bool(settings.get("per_layer_pause", False))

    if not per_layer_pause:
        # Plot all layers as a single combined SVG job
        svg_data = project_to_svg_string(project, layer_ids, settings)
        plot_svg_string(svg_data, settings, progress_callback)
    else:
        # Plot each layer separately
        n = len(layers)
        for idx, layer in enumerate(layers):
            if progress_callback:
                progress_callback(int(idx / n * 90))
            svg_data = _layer_to_svg_string(layer, project.canvas, settings)
            plot_svg_string(svg_data, settings)
        if progress_callback:
            progress_callback(100)


# ---------------------------------------------------------------------------
# Internal SVG helpers
# ---------------------------------------------------------------------------

def _layer_to_svg_string(layer: "Layer", canvas: "Canvas", settings: dict) -> str:
    """Convert a single Layer to an in-memory SVG string."""
    import svgwrite
    stroke_width = float(settings.get("stroke_width_mm", 0.3))

    dwg = svgwrite.Drawing(
        filename="plot.svg",
        size=(f"{canvas.width_mm}mm", f"{canvas.height_mm}mm"),
        viewBox=f"0 0 {canvas.width_mm} {canvas.height_mm}",
    )

    g = dwg.g(stroke=layer.color, fill="none",
              stroke_width=f"{stroke_width:.3f}mm")
    for path in layer.paths:
        if len(path) < 2:
            continue
        points = [(f"{x:.3f}", f"{y:.3f}") for x, y in path]
        g.add(dwg.polyline(points=points))
    dwg.add(g)

    buf = io.StringIO()
    dwg.write(buf)
    return buf.getvalue()


def project_to_svg_string(
    project: "Project",
    layer_ids: list[str] | None,
    settings: dict,
) -> str:
    """Convert project layers to a combined in-memory SVG string."""
    import svgwrite
    stroke_width = float(settings.get("stroke_width_mm", 0.3))
    canvas = project.canvas

    dwg = svgwrite.Drawing(
        filename="plot.svg",
        size=(f"{canvas.width_mm}mm", f"{canvas.height_mm}mm"),
        viewBox=f"0 0 {canvas.width_mm} {canvas.height_mm}",
    )

    for layer in project.layers:
        if not layer.visible:
            continue
        if layer_ids is not None and layer.id not in layer_ids:
            continue
        g = dwg.g(
            id=f"layer_{_safe_xml_id(layer.name)}",
            stroke=layer.color,
            fill="none",
            stroke_width=f"{stroke_width:.3f}mm",
        )
        for path in layer.paths:
            if len(path) < 2:
                continue
            points = [(f"{x:.3f}", f"{y:.3f}") for x, y in path]
            g.add(dwg.polyline(points=points))
        dwg.add(g)

    buf = io.StringIO()
    dwg.write(buf)
    return buf.getvalue()
