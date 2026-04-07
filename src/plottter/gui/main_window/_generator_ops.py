"""_GeneratorOpsMixin — generate, randomize, 3D regen, and layer-move handlers."""

from __future__ import annotations

import copy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog


class _GeneratorOpsMixin:
    """Mixin providing generator-related operations for MainWindow."""

    # ------------------------------------------------------------------
    # Generate menu actions
    # ------------------------------------------------------------------

    def _on_generate_now(self) -> None:
        self._settings_panel.trigger_generate()

    def _on_randomize(self) -> None:
        self._settings_panel.trigger_randomize()

    def _on_surprise_me(self) -> None:
        """Switch to Math Art mode, pick a random generator, randomize params, generate."""
        self._mode_panel.set_mode("Math Art")
        self._settings_panel.on_mode_changed("Math Art")
        self._settings_panel.trigger_surprise_me()

    def _on_browse_presets(self) -> None:
        """Open the preset gallery; apply chosen preset to the settings panel."""
        from PyQt6.QtWidgets import QDialog as _QDialog
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog

        dialog = PresetGalleryDialog(parent=self)
        if dialog.exec() == _QDialog.DialogCode.Accepted:
            gen_cls, preset_name = dialog.selected_preset()
            if gen_cls is not None:
                self._mode_panel.set_mode("Math Art")
                self._settings_panel.on_mode_changed("Math Art")
                self._settings_panel.apply_generator_preset(gen_cls, preset_name)

    # ------------------------------------------------------------------
    # 3D regen
    # ------------------------------------------------------------------

    def _on_regenerate_all_3d(self) -> None:
        """Sequentially regenerate all 3D Scene layers with up-to-date sibling occlusion."""
        # Only flush when the settings panel is actually showing 3D controls.
        # Flushing while the panel is in a different mode (e.g. Math Art) would
        # overwrite a 3D layer's generator_info with the wrong mode's UI state.
        if self._settings_panel.current_mode == "3D Scene":
            self._settings_panel.flush_current_snapshot()

        project = self._controller.current_project
        d3_layers = [
            layer for layer in project.layers
            if isinstance(layer.generator_info, dict)
            and layer.generator_info.get("mode") == "3D Scene"
        ]

        if not d3_layers:
            QMessageBox.information(
                self,
                "Regenerate All 3D Layers",
                "No 3D Scene layers found in the project.",
            )
            return

        n = len(d3_layers)
        self._regen3d_layers = d3_layers
        self._regen3d_idx = 0

        progress = QProgressDialog(
            f"Generating 3D layer 1 of {n}…", "Cancel", 0, n * 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        self._regen3d_progress = progress

        # Wrap all path changes in a single undo macro
        self._controller.undo_stack.beginMacro("Regenerate All 3D Layers")
        self._start_next_3d_regen()

    def _start_next_3d_regen(self) -> None:
        if self._regen3d_idx >= len(self._regen3d_layers):
            self._finish_3d_regen()
            return

        if self._regen3d_progress.wasCanceled():
            self._finish_3d_regen(cancelled=True)
            return

        layer = self._regen3d_layers[self._regen3d_idx]
        n = len(self._regen3d_layers)
        base_progress = self._regen3d_idx * 100

        self._regen3d_progress.setLabelText(
            f"Generating 3D layer {self._regen3d_idx + 1} of {n}: '{layer.name}'…"
        )
        self._regen3d_progress.setValue(base_progress)

        info = layer.generator_info
        params = dict(info.get("params", {}))

        # Inject shared camera from project metadata
        project = self._controller.current_project
        cam = project.metadata.get("scene3d_camera", {})
        if cam:
            params["_camera"] = cam

        # Inject sibling shapes for HLR occlusion using up-to-date generator_info
        params["_sibling_3d_shapes"] = self._settings_panel._build_sibling_3d_shapes(layer.id)

        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.gui.generator_worker import GeneratorWorker

        generator = Scene3DGenerator()
        canvas = project.canvas
        layer_id = layer.id

        worker = GeneratorWorker(generator, params, canvas, parent=self)

        def on_progress(pct: int) -> None:
            self._regen3d_progress.setValue(base_progress + pct)

        def on_finished(paths: list, lid: str = layer_id) -> None:
            self._controller.set_layer_paths(lid, paths, "Regenerate 3D Layer")
            self._regen3d_idx += 1
            self._regen3d_progress.setValue(self._regen3d_idx * 100)
            self._start_next_3d_regen()
            worker.deleteLater()

        def on_error(msg: str) -> None:
            QMessageBox.critical(self, "3D Regeneration Error", msg)
            self._regen3d_idx += 1
            self._regen3d_progress.setValue(self._regen3d_idx * 100)
            self._start_next_3d_regen()
            worker.deleteLater()

        # Disconnect previous layer's cancel connection to avoid stacking
        prev = getattr(self, "_regen3d_worker", None)
        if prev is not None:
            try:
                self._regen3d_progress.canceled.disconnect(prev.cancel)
            except (RuntimeError, TypeError):
                pass
        self._regen3d_progress.canceled.connect(worker.cancel)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._regen3d_worker = worker
        worker.start()

    def _finish_3d_regen(self, cancelled: bool = False) -> None:
        self._controller.undo_stack.endMacro()
        self._regen3d_progress.close()
        n = len(self._regen3d_layers)
        if cancelled:
            done = self._regen3d_idx
            self.statusBar().showMessage(
                f"3D regeneration cancelled after {done}/{n} layers.", 4000
            )
        else:
            self.statusBar().showMessage(
                f"Regenerated {n} 3D layer{'s' if n != 1 else ''} successfully.", 5000
            )

    # ------------------------------------------------------------------
    # Layer move (drag-to-move tool)
    # ------------------------------------------------------------------

    def _on_layer_move_finished(self, dx_mm: float, dy_mm: float) -> None:
        """Apply a completed drag-to-move offset to the active layer's paths.

        If the layer's generator has ``x_offset_mm`` / ``y_offset_mm`` params,
        those are updated in generator_info so re-generating preserves the new
        position.  Both the path translation and the param update are bundled
        into a single undoable ``MoveLayerCommand``.
        """
        layer_id = self._controller.active_layer_id
        if not layer_id:
            return
        layer = self._controller.get_layer(layer_id)
        if layer is None or not layer.paths:
            return

        old_paths = [list(p) for p in layer.paths]
        new_paths = [[(x + dx_mm, y + dy_mm) for x, y in path] for path in layer.paths]

        # Check if the generator exposes x_offset_mm / y_offset_mm params.
        # generator_info may be None if the user hasn't switched away from this
        # layer yet (it's only persisted on layer switch).  Grab a live snapshot
        # from the settings panel in that case.
        old_gen_info = layer.generator_info
        if old_gen_info is None:
            old_gen_info = self._settings_panel._get_settings_snapshot()
            if old_gen_info is not None:
                layer.generator_info = old_gen_info
        new_gen_info: dict | None = None
        if (
            old_gen_info is not None
            and isinstance(old_gen_info.get("params"), dict)
            and "x_offset_mm" in old_gen_info["params"]
            and "y_offset_mm" in old_gen_info["params"]
        ):
            new_gen_info = copy.deepcopy(old_gen_info)
            new_gen_info["params"]["x_offset_mm"] = (
                old_gen_info["params"]["x_offset_mm"] + dx_mm
            )
            new_gen_info["params"]["y_offset_mm"] = (
                old_gen_info["params"]["y_offset_mm"] + dy_mm
            )
        elif (
            old_gen_info is not None
            and old_gen_info.get("mode") == "3D Scene"
            and isinstance(old_gen_info.get("params"), dict)
            and "pos_x" in old_gen_info["params"]
            and "pos_y" in old_gen_info["params"]
        ):
            # 3D Scene: pos_x/pos_y are in 3D world units.  Canvas X maps
            # directly to 3D X; canvas Y increases downward but 3D Y is up,
            # so the sign is inverted.
            new_gen_info = copy.deepcopy(old_gen_info)
            new_gen_info["params"]["pos_x"] = (
                old_gen_info["params"]["pos_x"] + dx_mm
            )
            new_gen_info["params"]["pos_y"] = (
                old_gen_info["params"]["pos_y"] - dy_mm
            )

        from plottter.gui.commands import MoveLayerCommand
        cmd = MoveLayerCommand(
            self._controller,
            layer_id,
            new_paths,
            old_paths,
            new_gen_info,
            copy.deepcopy(old_gen_info) if new_gen_info is not None else None,
        )
        self._controller.undo_stack.push(cmd)
