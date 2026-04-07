"""_FileOpsMixin — file open/save/export and recent-projects handling."""

from __future__ import annotations

import os

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from plottter.models import Layer, Project


class _FileOpsMixin:
    """Mixin providing file I/O operations for MainWindow."""

    # ------------------------------------------------------------------
    # File menu actions
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        if not self._prompt_save_if_modified():
            return
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dialog = NewProjectDialog(self)
        if dialog.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        canvas = dialog.get_canvas()
        project = Project(name="Untitled", canvas=canvas)
        project.add_layer(Layer(name="Layer 1", color="#000000"))
        self._current_file = None
        self._controller.new_project(project)

    def _on_open(self) -> None:
        if not self._prompt_save_if_modified():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", self.last_file_dir(), "Plottter Files (*.plottter);;All Files (*)"
        )
        if not path:
            return
        try:
            from plottter.io.project_file import load_project
            project = load_project(path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.load_project(project)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening File", str(exc))

    def _on_save(self) -> None:
        if self._current_file:
            self._save_to(self._current_file)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", self.last_file_dir(), "Plottter Files (*.plottter);;All Files (*)"
        )
        if path:
            if not path.endswith(".plottter"):
                path += ".plottter"
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            from plottter.io.project_file import save_project
            save_project(self._controller.current_project, path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.mark_saved()
        except Exception as exc:
            QMessageBox.critical(self, "Error Saving File", str(exc))

    def _on_export_current(self) -> None:
        from plottter.gui.dialogs.export import ExportDialog
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._do_export(settings, "current")

    def _on_export_all(self) -> None:
        from plottter.gui.dialogs.export import ExportDialog
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._do_export(settings, settings.get("layer_mode", "all_separate"))

    def _do_export(self, settings: dict, mode: str) -> None:
        project = self._controller.current_project
        path = settings.get("output_path", "")
        if not path:
            QMessageBox.warning(self, "Export", "Please specify an output path.")
            return
        fmt = settings.get("format", "SVG")
        if mode == "current":
            layer_id = self._controller.active_layer_id
            active = self._controller.get_layer(layer_id) if layer_id else None
            if active is None:
                QMessageBox.warning(self, "Export", "No active layer to export.")
                return
        try:
            if fmt == "SVG":
                from plottter.export.svg import (
                    export_layer_svg,
                    export_all_layers_svg,
                    export_combined_svg,
                )
                if mode == "current":
                    export_layer_svg(active, project.canvas, path, settings)
                elif mode == "all_separate":
                    export_all_layers_svg(project, path, settings)
                else:
                    export_combined_svg(project, path, settings)
            elif fmt == "HPGL":
                from plottter.export.hpgl import (
                    export_layer_hpgl,
                    export_all_layers_hpgl,
                )
                if mode == "current":
                    export_layer_hpgl(active, project.canvas, path, settings)
                else:
                    export_all_layers_hpgl(project, path, settings)
            elif fmt == "G-code":
                from plottter.export.gcode import (
                    export_layer_gcode,
                    export_all_layers_gcode,
                )
                if mode == "current":
                    export_layer_gcode(active, project.canvas, path, settings)
                else:
                    export_all_layers_gcode(project, path, settings)
            elif fmt == "Mural":
                from plottter.export.mural import (
                    export_layer_mural,
                    export_all_layers_mural,
                )
                if mode == "current":
                    mural_warnings = export_layer_mural(active, project.canvas, path, settings)
                else:
                    mural_warnings = export_all_layers_mural(project, path, settings)
                if mural_warnings:
                    unique = list(dict.fromkeys(mural_warnings))
                    warn_text = "\n".join(unique[:10])
                    if len(unique) > 10:
                        warn_text += f"\n… and {len(unique) - 10} more"
                    QMessageBox.warning(
                        self,
                        "Mural Export — Out-of-Bounds Coordinates",
                        f"Some coordinates fall outside the valid drawing area:\n\n{warn_text}",
                    )
            else:
                # Export plugin
                from plottter.export.plugin import EXPORT_PLUGINS
                plugin_cls = EXPORT_PLUGINS.get(fmt)
                if plugin_cls is None:
                    QMessageBox.warning(self, "Export", f"Unknown export format: {fmt}")
                    return
                plugin = plugin_cls()
                if mode == "current":
                    if active is None:
                        QMessageBox.warning(self, "Export", "No active layer to export.")
                        return
                    plugin.export(
                        [(active.name, active.color, active.paths)],
                        project.canvas,
                        path,
                    )
                elif mode == "all_separate":
                    for layer in project.layers:
                        if not layer.paths:
                            continue
                        base_p, ext_p = os.path.splitext(path)
                        if not ext_p and plugin_cls.file_extension:
                            ext_p = plugin_cls.file_extension
                        safe_name = layer.name.replace(
                            os.sep, "_"
                        ).replace("/", "_")
                        layer_path = f"{base_p}_{safe_name}{ext_p}"
                        plugin.export(
                            [(layer.name, layer.color, layer.paths)],
                            project.canvas,
                            layer_path,
                        )
                else:  # all_combined
                    paths_by_layer = [
                        (layer.name, layer.color, layer.paths)
                        for layer in project.layers
                    ]
                    plugin.export(paths_by_layer, project.canvas, path)
            QMessageBox.information(self, "Export", f"Exported successfully to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Recent projects
    # ------------------------------------------------------------------

    def last_file_dir(self) -> str:
        """Return the last used file dialog directory (for open/save dialogs)."""
        settings = QSettings("Plottter", "Plottter")
        return settings.value("last_file_dir", "") or ""

    def save_last_file_dir(self, path: str) -> None:
        """Persist the directory of *path* so the next file dialog opens there."""
        settings = QSettings("Plottter", "Plottter")
        settings.setValue("last_file_dir", os.path.dirname(path))

    def _recent_projects(self) -> list[str]:
        settings = QSettings("Plottter", "Plottter")
        return list(settings.value("recent_projects", []) or [])

    def _add_recent_project(self, path: str) -> None:
        settings = QSettings("Plottter", "Plottter")
        recent: list[str] = list(settings.value("recent_projects", []) or [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]
        settings.setValue("recent_projects", recent)
        self._rebuild_recent_menu()

    def _open_recent_project(self, path: str) -> None:
        if not self._prompt_save_if_modified():
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{path}")
            return
        try:
            from plottter.io.project_file import load_project
            project = load_project(path)
            self._current_file = path
            self.save_last_file_dir(path)
            self._add_recent_project(path)
            self._controller.load_project(project)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening File", str(exc))

    def _clear_recent_projects(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.remove("recent_projects")
        self._rebuild_recent_menu()

    def _prompt_save_if_modified(self) -> bool:
        """Ask the user to save if there are unsaved changes. Returns True to proceed."""
        if not self._controller.modified:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before proceeding?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
            return not self._controller.modified  # True if save succeeded
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel
