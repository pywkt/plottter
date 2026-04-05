"""ExportPlugin ABC and registry for custom export format plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.models.canvas import Canvas
    from plottter.models.path import Polyline

#: Registry of all registered export plugins, keyed by plugin name.
EXPORT_PLUGINS: dict[str, type[ExportPlugin]] = {}


def register_export_plugin(cls: type[ExportPlugin]) -> type[ExportPlugin]:
    """Decorator to register an :class:`ExportPlugin` subclass by its name."""
    EXPORT_PLUGINS[cls.name] = cls
    return cls


class ExportPlugin(ABC):
    """Abstract base class for custom export format plugins.

    An export plugin writes polyline data to a file in a custom format and
    appears in the Export dialog's format dropdown alongside SVG, HPGL,
    G-code, and Mural.

    Subclass this and optionally decorate with :func:`register_export_plugin`.
    Any subclass with a non-empty ``name`` attribute is auto-registered by the
    plugin loader when the file is loaded.  Place the file in any of the
    plugin directories scanned by :mod:`plottter.generators.plugin_loader`.

    Example::

        from plottter.export.plugin import ExportPlugin, register_export_plugin

        @register_export_plugin
        class CSVExportPlugin(ExportPlugin):
            name = "CSV"
            file_extension = ".csv"
            description = "Export paths as CSV coordinates (x_mm, y_mm per row)."

            def export(self, paths_by_layer, canvas, file_path):
                with open(file_path, "w") as f:
                    f.write("layer,color,x_mm,y_mm\\n")
                    for layer_name, hex_color, paths in paths_by_layer:
                        for path in paths:
                            for x, y in path:
                                f.write(f"{layer_name},{hex_color},{x:.4f},{y:.4f}\\n")
    """

    #: Display name shown in the Export dialog format dropdown (must be unique).
    name: str = ""

    #: File extension including the leading dot, e.g. ``".dxf"``.
    #: Used to auto-complete file names in the Export dialog.
    file_extension: str = ""

    #: Short description shown as a tooltip in the Export dialog.
    description: str = ""

    @abstractmethod
    def export(
        self,
        paths_by_layer: list[tuple[str, str, list[Polyline]]],
        canvas: Canvas,
        file_path: str,
    ) -> None:
        """Write *paths_by_layer* to *file_path*.

        Parameters
        ----------
        paths_by_layer:
            List of ``(layer_name, hex_color, paths)`` tuples.
            ``hex_color`` is a CSS hex string like ``"#ff0000"``.
            ``paths`` is a list of polylines; each polyline is a list of
            ``(x_mm, y_mm)`` tuples.
        canvas:
            Canvas with paper dimensions (``width_mm``, ``height_mm``,
            ``margin_mm``).
        file_path:
            Absolute or relative path to write to.  The parent directory is
            guaranteed to exist.  Any existing file should be overwritten.
        """
