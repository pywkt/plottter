"""_ProcessingOpsMixin — path processing operations (simplify, merge, weld, optimize, etc.)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from plottter.models import Layer
from plottter.models.path import Polyline

from ._brush_dialog import _BrushDialog
from .workers import _BrushWorker, _OffsetWorker, _OptimizeWorker, _TaperWorker, _WeldWorker


class _ProcessingOpsMixin:
    """Mixin providing path processing operations for MainWindow."""

    def _on_simplify_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        from plottter.gui.dialogs.simplify_dialog import SimplifyDialog
        dialog = SimplifyDialog(list(layer.paths), parent=self)
        if dialog.exec() != SimplifyDialog.DialogCode.Accepted:
            return
        from plottter.processing import simplify_paths
        new_paths = simplify_paths(layer.paths, dialog.get_tolerance())
        self._controller.set_layer_paths(layer.id, new_paths, "Simplify Paths")

    def _on_merge_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        from plottter.gui.dialogs.merge_dialog import MergeDialog
        dialog = MergeDialog(list(layer.paths), parent=self)
        if dialog.exec() != MergeDialog.DialogCode.Accepted:
            return
        from plottter.processing import merge_nearby_paths
        new_paths = merge_nearby_paths(layer.paths, dialog.get_threshold())
        self._controller.set_layer_paths(layer.id, new_paths, "Merge Nearby Paths")

    def _on_clip_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return
        bounds = self._controller.current_project.canvas.drawing_area()
        from plottter.processing import clip_to_bounds
        new_paths = clip_to_bounds(layer.paths, bounds)
        self._controller.set_layer_paths(layer.id, new_paths, "Clip to Canvas")

    def _on_weld_layer(self) -> None:
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None or not layer.paths:
            return

        from plottter.gui.dialogs.weld_dialog import WeldDialog

        dlg = WeldDialog(parent=self)
        if dlg.exec() != WeldDialog.DialogCode.Accepted:
            return
        tolerance_mm = dlg.get_tolerance()

        total = len(layer.paths)
        progress = QProgressDialog(
            f"Removing duplicate segments in '{layer.name}'…", "Cancel", 0, total, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        layer_id = layer.id
        worker = _WeldWorker(paths=list(layer.paths), tolerance_mm=tolerance_mm, parent=self)

        def on_progress(cur: int, tot: int) -> None:
            if tot > 0:
                progress.setValue(cur)

        def on_finished(new_paths: list, before_count: int, after_count: int) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Remove Duplicate Segments")
            removed = before_count - after_count
            self.statusBar().showMessage(
                f"Duplicate-segment removal complete: {before_count} → "
                f"{after_count} paths ({removed} removed).",
                5000,
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Remove Duplicate Segments — Error", msg)
            worker.deleteLater()

        def on_cancelled() -> None:
            worker.cancel()

        def on_weld_cancelled() -> None:
            progress.close()
            self.statusBar().showMessage("Duplicate-segment removal cancelled.", 3000)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.cancelled.connect(on_weld_cancelled)
        worker.error.connect(on_error)
        progress.canceled.connect(on_cancelled)
        self._weld_worker = worker
        worker.start()

    def _on_optimize_layer(self) -> None:
        """Run the full optimization pipeline on the selected layer."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Optimize", "No selected layer to optimize.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Optimize", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.optimize_dialog import OptimizeSettingsDialog

        dlg = OptimizeSettingsDialog(parent=self)
        if dlg.exec() != OptimizeSettingsDialog.DialogCode.Accepted:
            return

        bounds = self._controller.current_project.canvas.drawing_area()
        self._run_optimization([layer], bounds, settings=dlg.get_settings())

    def _on_optimize_all(self) -> None:
        """Run the full optimization pipeline on all unlocked layers."""
        project = self._controller.current_project
        layers = [l for l in project.layers if not l.locked and l.paths]
        if not layers:
            QMessageBox.information(self, "Optimize All", "No unlocked layers with paths.")
            return
        bounds = project.canvas.drawing_area()
        self._run_optimization(layers, bounds)

    def _on_apply_brush_layer(self) -> None:
        """Show the Apply Brush dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Apply Brush", "No selected layer to apply brush to.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Apply Brush", "Selected layer has no paths.")
            return

        brush_type, params = _BrushDialog.run(self, list(layer.paths))
        if brush_type is None or brush_type == "None":
            return  # Cancelled or no-op

        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying '{brush_type}' brush to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _BrushWorker(paths=list(layer.paths), brush_type=brush_type, params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Apply Brush")
            self.statusBar().showMessage(
                f"Brush applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Brush Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._brush_worker = worker
        worker.start()

    def _on_taper_layer(self) -> None:
        """Show the Taper Paths dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Taper Paths", "No selected layer to apply taper to.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Taper Paths", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.taper_dialog import TaperSettingsDialog

        dlg = TaperSettingsDialog(list(layer.paths), parent=self)
        if dlg.exec() != TaperSettingsDialog.DialogCode.Accepted:
            return

        params = dlg.get_params()
        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying taper to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _TaperWorker(paths=list(layer.paths), params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Taper Paths")
            self.statusBar().showMessage(
                f"Taper applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Taper Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._taper_worker = worker
        worker.start()

    def _on_offset_layer(self) -> None:
        """Show the Offset Paths dialog and replace the selected layer's paths."""
        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "Offset Paths", "No selected layer. Please select a layer first.")
            return
        if not layer.paths:
            QMessageBox.information(self, "Offset Paths", "Selected layer has no paths.")
            return

        from plottter.gui.dialogs.offset_dialog import OffsetSettingsDialog
        dlg = OffsetSettingsDialog(list(layer.paths), parent=self)
        if dlg.exec() != OffsetSettingsDialog.DialogCode.Accepted:
            return

        params = dlg.get_params()
        total = len(layer.paths)
        progress = QProgressDialog(
            f"Applying offset to '{layer.name}'…", "", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = _OffsetWorker(paths=list(layer.paths), params=params, parent=self)

        def on_progress(value: int) -> None:
            progress.setValue(value)

        def on_finished(new_paths: list) -> None:
            progress.close()
            self._controller.set_layer_paths(layer_id, new_paths, "Offset Paths")
            self.statusBar().showMessage(
                f"Offset applied: {total} → {len(new_paths)} paths.", 4000
            )
            worker.deleteLater()

        def on_error(msg: str) -> None:
            progress.close()
            QMessageBox.critical(self, "Offset Error", msg)
            worker.deleteLater()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._offset_worker = worker
        worker.start()

    def _on_calibration_plot(self, plot_type: str) -> None:
        """Generate a calibration plot layer for the given plot type."""
        from plottter.calibration import (
            generate_angle_test,
            generate_circle_test,
            generate_fill_density_test,
            generate_line_spacing_test,
            generate_paper_size_sheet,
            generate_registration_test,
        )
        from plottter.models import Layer

        _generators = {
            "Line Spacing Test": generate_line_spacing_test,
            "Circle & Arc Test": generate_circle_test,
            "Angle Test": generate_angle_test,
            "Fill Density Test": generate_fill_density_test,
            "Registration Test": generate_registration_test,
            "Paper Size Alignment": generate_paper_size_sheet,
        }

        canvas = self._controller.current_project.canvas

        # Handle individual paper size requests ("Paper Size: A3", etc.)
        if plot_type.startswith("Paper Size: "):
            paper_name = plot_type.removeprefix("Paper Size: ")
            paths = generate_paper_size_sheet(
                canvas.width_mm, canvas.height_mm, canvas.margin_mm,
                paper_name=paper_name,
            )
        else:
            gen_fn = _generators.get(plot_type)
            if gen_fn is None:
                return
            paths = gen_fn(canvas.width_mm, canvas.height_mm, canvas.margin_mm)

        layer = Layer(name=plot_type, color="#000000")
        self._controller.add_layer(layer)
        self._controller.set_layer_paths(layer.id, paths, "Calibration Plot")
        self._controller.set_active_layer(layer.id)

    def _on_plot_axidraw(self) -> None:
        """Open AxiDraw plot dialog for direct USB plotting."""
        project = self._controller.current_project
        active_id = self._controller.active_layer_id

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dlg = AxiDrawDialog(project, active_layer_id=active_id, parent=self)
        dlg.plot_started.connect(lambda: self.statusBar().showMessage("Plotting…"))
        dlg.plot_finished.connect(lambda: self.statusBar().showMessage("Plot complete."))
        dlg.exec()

    def _run_optimization(
        self,
        layers: list[Layer],
        bounds: tuple[float, float, float, float],
        settings: dict | None = None,
    ) -> None:
        """Run optimization on each layer sequentially (one worker per layer)."""
        self._opt_layers = list(layers)
        self._opt_bounds = bounds
        self._opt_settings = settings  # None → use worker defaults
        self._opt_layer_idx = 0
        self._opt_results: list[tuple[Layer, list[Polyline], float, float, int, int]] = []

        # Range is 0..100*n_layers so within-layer progress drives the bar smoothly
        progress = QProgressDialog(
            "Optimizing paths…", "Cancel", 0, max(1, len(layers)) * 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        self._opt_progress = progress

        self._start_next_opt_layer()

    def _start_next_opt_layer(self) -> None:
        if self._opt_layer_idx >= len(self._opt_layers):
            self._finish_optimization()
            return

        if self._opt_progress.wasCanceled():
            self._finish_optimization(cancelled=True)
            return

        layer = self._opt_layers[self._opt_layer_idx]
        n_layers = len(self._opt_layers)
        base_progress = self._opt_layer_idx * 100

        self._opt_progress.setLabelText(
            f"Optimizing '{layer.name}' ({self._opt_layer_idx + 1}/{n_layers})…\n"
            f"Step: Preprocessing"
        )

        s = self._opt_settings or {}
        worker = _OptimizeWorker(
            paths=list(layer.paths),
            run_weld=s.get("run_weld", False),
            weld_tolerance=s.get("weld_tolerance", 0.1),
            run_simplify=s.get("run_simplify", True),
            simplify_tolerance=s.get("simplify_tolerance", 0.1),
            run_filter=s.get("run_filter", True),
            filter_min_length=s.get("filter_min_length", 0.5),
            run_clip=s.get("run_clip", True),
            clip_bounds=self._opt_bounds,
            run_merge=s.get("run_merge", True),
            merge_threshold=s.get("merge_threshold", 0.5),
            run_2opt=s.get("run_2opt", True),
            run_3opt=s.get("run_3opt", False),
            run_or_opt=s.get("run_or_opt", True),
            num_starts=5,
            parent=self,
        )

        def on_progress(value: int) -> None:
            self._opt_progress.setValue(base_progress + value)
            if value < 10:
                step = "Preprocessing"
            elif value < 35:
                step = "Reordering paths"
            elif value < 55:
                step = "Running 2-opt"
            elif value < 75:
                step = "Running 3-opt"
            else:
                step = "Running Or-opt"
            self._opt_progress.setLabelText(
                f"Optimizing '{layer.name}' ({self._opt_layer_idx + 1}/{n_layers})…\n"
                f"Step: {step}"
            )

        def on_finished(new_paths, before, after, before_lifts, after_lifts):
            self._opt_results.append((layer, new_paths, before, after, before_lifts, after_lifts))
            self._opt_layer_idx += 1
            self._opt_progress.setValue(self._opt_layer_idx * 100)
            self._start_next_opt_layer()
            worker.deleteLater()

        def on_error(msg):
            QMessageBox.critical(self, "Optimization Error", msg)
            self._opt_layer_idx += 1
            self._opt_progress.setValue(self._opt_layer_idx * 100)
            self._start_next_opt_layer()
            worker.deleteLater()

        # Wire cancel button to stop the worker gracefully.
        # Disconnect the previous layer's worker first to avoid accumulating
        # cancel connections across layers (each layer creates a new worker).
        prev_worker = getattr(self, "_opt_worker", None)
        if prev_worker is not None:
            try:
                self._opt_progress.canceled.disconnect(prev_worker.request_stop)
            except (RuntimeError, TypeError):
                pass  # already disconnected or deleted
        self._opt_progress.canceled.connect(worker.request_stop)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.progress.connect(on_progress)
        self._opt_worker = worker
        worker.start()

    def _finish_optimization(self, cancelled: bool = False) -> None:
        self._opt_progress.close()

        if not self._opt_results:
            return

        # Apply results
        for layer, new_paths, _before, _after, _bl, _al in self._opt_results:
            self._controller.set_layer_paths(layer.id, new_paths, "Optimize Paths")

        if cancelled:
            return

        # Build metrics report
        lines = []
        total_before = 0.0
        total_after = 0.0
        total_lifts_before = 0
        total_lifts_after = 0
        for layer, _, before, after, lifts_before, lifts_after in self._opt_results:
            reduction = ((before - after) / before * 100) if before > 0 else 0.0
            lifts_delta = lifts_before - lifts_after
            lines.append(
                f"<b>{layer.name}</b>: {before:.0f} mm → {after:.0f} mm "
                f"({reduction:.1f}% reduction), "
                f"pen lifts {lifts_before} → {lifts_after} ({lifts_delta:+d})"
            )
            total_before += before
            total_after += after
            total_lifts_before += lifts_before
            total_lifts_after += lifts_after

        if len(self._opt_results) > 1:
            total_reduction = (
                ((total_before - total_after) / total_before * 100)
                if total_before > 0 else 0.0
            )
            total_lifts_delta = total_lifts_before - total_lifts_after
            lines.append(
                f"<br><b>Total</b>: {total_before:.0f} mm → {total_after:.0f} mm "
                f"({total_reduction:.1f}% reduction), "
                f"pen lifts {total_lifts_before} → {total_lifts_after} ({total_lifts_delta:+d})"
            )

        QMessageBox.information(
            self,
            "Optimization Complete",
            "<b>Pen-up travel distance &amp; pen lift count:</b><br><br>" + "<br>".join(lines),
        )
