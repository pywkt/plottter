"""_CanvasOpsMixin — canvas settings, rotation, preferences, jitter."""

from __future__ import annotations

import copy

from PyQt6.QtWidgets import QMessageBox

from plottter.models import Canvas


class _CanvasOpsMixin:
    """Mixin providing canvas-related operations for MainWindow."""

    def _on_canvas_settings(self) -> None:
        from plottter.gui.dialogs.new_project import NewProjectDialog
        project = self._controller.current_project
        old_canvas = project.canvas
        dialog = NewProjectDialog(self, initial_canvas=old_canvas)
        if dialog.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        new_canvas = dialog.get_canvas()

        # Only offer scaling when there is art to scale
        layers_with_paths = [layer for layer in project.layers if layer.paths]
        scale_art = False
        if layers_with_paths:
            reply = QMessageBox.question(
                self,
                "Scale Art to New Canvas?",
                "Scale existing art to fit the new canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            scale_art = reply == QMessageBox.StandardButton.Yes

        if not scale_art:
            self._controller.set_canvas(new_canvas)
            return

        # Pre-compute scaled paths and updated generator_info for each layer
        from plottter.processing.scale import scale_paths_to_canvas
        old_left, old_top, old_right, old_bottom = old_canvas.drawing_area()
        new_left, new_top, new_right, new_bottom = new_canvas.drawing_area()
        old_draw_w = old_right - old_left
        old_draw_h = old_bottom - old_top
        new_draw_w = new_right - new_left
        new_draw_h = new_bottom - new_top
        sx = new_draw_w / old_draw_w if old_draw_w else 1.0
        sy = new_draw_h / old_draw_h if old_draw_h else 1.0

        scale_data = []
        for layer in project.layers:
            if not layer.paths:
                continue
            old_paths = [list(p) for p in layer.paths]
            new_paths = scale_paths_to_canvas(layer.paths, old_canvas, new_canvas)
            old_gen_info = layer.generator_info
            new_gen_info = None
            if old_gen_info is not None and isinstance(old_gen_info.get("params"), dict):
                params = old_gen_info["params"]
                if "x_offset_mm" in params or "y_offset_mm" in params:
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "x_offset_mm" in params:
                        new_gen_info["params"]["x_offset_mm"] = params["x_offset_mm"] * sx
                    if "y_offset_mm" in params:
                        new_gen_info["params"]["y_offset_mm"] = params["y_offset_mm"] * sy
                elif (
                    old_gen_info.get("mode") == "3D Scene"
                    and ("pos_x" in params or "pos_y" in params)
                ):
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "pos_x" in params:
                        new_gen_info["params"]["pos_x"] = params["pos_x"] * sx
                    if "pos_y" in params:
                        new_gen_info["params"]["pos_y"] = params["pos_y"] * sy
            scale_data.append(
                (
                    layer.id,
                    old_paths,
                    new_paths,
                    copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
                    new_gen_info,
                )
            )

        # Push canvas change + all path scalings as one undoable macro
        from plottter.gui.commands import MoveLayerCommand, SetCanvasCommand
        self._controller.undo_stack.beginMacro("Canvas Resize with Scale")
        try:
            cmd = SetCanvasCommand(
                self._controller, new_canvas, copy.copy(old_canvas), description="Canvas Settings"
            )
            self._controller.undo_stack.push(cmd)
            for layer_id, old_paths, new_paths, old_gen_info, new_gen_info in scale_data:
                cmd = MoveLayerCommand(
                    self._controller,
                    layer_id,
                    new_paths,
                    old_paths,
                    new_gen_info,
                    old_gen_info,
                )
                self._controller.undo_stack.push(cmd)
        finally:
            self._controller.undo_stack.endMacro()

    def _ask_rotate_scale_mode(self) -> str | None:
        """Show a dialog asking how to handle art when rotating the canvas.

        Returns one of ``"stretch"``, ``"keep_aspect"``, ``"none"``, or
        ``None`` if the user cancelled.
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Rotate Canvas — Scale Art?")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("How should existing art be handled after rotation?"))
        rb_stretch = QRadioButton("Scale to fit  (stretch to fill new dimensions)")
        rb_keep = QRadioButton("Scale to fit, keep aspect  (uniform scale, centered)")
        rb_none = QRadioButton("Don't scale  (keep art at original mm positions)")
        rb_keep.setChecked(True)
        layout.addWidget(rb_stretch)
        layout.addWidget(rb_keep)
        layout.addWidget(rb_none)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        if rb_stretch.isChecked():
            return "stretch"
        if rb_keep.isChecked():
            return "keep_aspect"
        return "none"

    def _on_rotate_canvas(self) -> None:
        project = self._controller.current_project
        old_canvas = project.canvas
        new_canvas = Canvas(
            width_mm=old_canvas.height_mm,
            height_mm=old_canvas.width_mm,
            margin_mm=old_canvas.margin_mm,
            paper_preset=old_canvas.paper_preset,
        )

        layers_with_paths = [layer for layer in project.layers if layer.paths]

        scale_mode = "none"  # default when no art exists
        if layers_with_paths:
            result = self._ask_rotate_scale_mode()
            if result is None:
                return
            scale_mode = result

        if scale_mode == "none":
            self._controller.set_canvas(new_canvas, description="Rotate Canvas")
            return

        # Build scaled path data for each layer
        from plottter.processing.scale import scale_paths_to_canvas, scale_paths_keep_aspect

        old_left, old_top, old_right, old_bottom = old_canvas.drawing_area()
        new_left, new_top, new_right, new_bottom = new_canvas.drawing_area()
        old_draw_w = old_right - old_left
        old_draw_h = old_bottom - old_top
        new_draw_w = new_right - new_left
        new_draw_h = new_bottom - new_top

        if scale_mode == "stretch":
            sx = new_draw_w / old_draw_w if old_draw_w else 1.0
            sy = new_draw_h / old_draw_h if old_draw_h else 1.0
        else:  # keep_aspect
            s = min(new_draw_w / old_draw_w, new_draw_h / old_draw_h) if old_draw_w and old_draw_h else 1.0
            sx = sy = s

        scale_data = []
        for layer in project.layers:
            if not layer.paths:
                continue
            old_paths = [list(p) for p in layer.paths]
            if scale_mode == "stretch":
                new_paths = scale_paths_to_canvas(layer.paths, old_canvas, new_canvas)
            else:
                new_paths = scale_paths_keep_aspect(layer.paths, old_canvas, new_canvas)
            old_gen_info = layer.generator_info
            new_gen_info = None
            if old_gen_info is not None and isinstance(old_gen_info.get("params"), dict):
                params = old_gen_info["params"]
                if "x_offset_mm" in params or "y_offset_mm" in params:
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "x_offset_mm" in params:
                        new_gen_info["params"]["x_offset_mm"] = params["x_offset_mm"] * sx
                    if "y_offset_mm" in params:
                        new_gen_info["params"]["y_offset_mm"] = params["y_offset_mm"] * sy
                elif (
                    old_gen_info.get("mode") == "3D Scene"
                    and ("pos_x" in params or "pos_y" in params)
                ):
                    new_gen_info = copy.deepcopy(old_gen_info)
                    if "pos_x" in params:
                        new_gen_info["params"]["pos_x"] = params["pos_x"] * sx
                    if "pos_y" in params:
                        new_gen_info["params"]["pos_y"] = params["pos_y"] * sy
            scale_data.append((
                layer.id,
                old_paths,
                new_paths,
                copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
                new_gen_info,
            ))

        from plottter.gui.commands import MoveLayerCommand, SetCanvasCommand
        self._controller.undo_stack.beginMacro("Rotate Canvas with Scale")
        try:
            cmd = SetCanvasCommand(
                self._controller, new_canvas, copy.copy(old_canvas), description="Rotate Canvas"
            )
            self._controller.undo_stack.push(cmd)
            for layer_id, old_paths, new_paths, old_gen_info, new_gen_info in scale_data:
                cmd = MoveLayerCommand(
                    self._controller,
                    layer_id,
                    new_paths,
                    old_paths,
                    new_gen_info,
                    old_gen_info,
                )
                self._controller.undo_stack.push(cmd)
        finally:
            self._controller.undo_stack.endMacro()

    def _on_preferences(self) -> None:
        from plottter.gui.dialogs.preferences import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec()
        # Refresh AI control availability in case the API key was changed
        self._settings_panel.update_ai_availability()

    def _on_jitter_intensity(self) -> None:
        """Open a dialog to set the pen jitter intensity."""
        from PyQt6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self,
            "Pen Jitter Intensity",
            "Intensity (0.1 = subtle wobble, 5.0 = heavy jitter):",
            self._canvas.get_jitter_intensity(),
            0.1,
            5.0,
            1,
        )
        if ok:
            self._canvas.set_jitter_intensity(value)

    def _on_preview_pen_width(self) -> None:
        """Set the on-canvas display stroke width in mm.

        Lets you preview how thick your real pen / marker strokes will be
        relative to the path layout — invaluable for color-separation work
        where the gap between two paired lines needs to exceed the pen
        width to render as two visible lines instead of one fat blob.
        """
        from PyQt6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self,
            "Preview Pen Width",
            "Stroke width in mm (0.3 ≈ fine pen, 1.2 ≈ marker):",
            self._canvas.get_preview_pen_width_mm(),
            0.05,
            5.0,
            2,
        )
        if ok:
            self._canvas.set_preview_pen_width_mm(value)
