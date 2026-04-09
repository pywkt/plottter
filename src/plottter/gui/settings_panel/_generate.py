"""_GenerateMixin — generation trigger and worker lifecycle methods."""

from __future__ import annotations

import random
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QSpinBox,
)


class _GenerateMixin:
    """Mixin for generation trigger and worker lifecycle methods."""

    @property
    def current_mode(self) -> str:
        """The currently active mode string (e.g. '3D Scene', 'Math Art')."""
        return self._current_mode

    def trigger_generate(self) -> None:
        """Public entry point — can be called from a menu action."""
        self._on_generate()

    def trigger_randomize(self) -> None:
        """Public entry point — can be called from a menu action."""
        self._on_randomize()

    def trigger_surprise_me(self) -> None:
        """Pick a random math generator, randomize its params, and generate."""
        try:
            from plottter.generators import get_generators_by_category
        except ImportError:
            return
        generators = get_generators_by_category("math")
        if not generators:
            return
        gen_cls = random.choice(generators)
        # Find and select this generator in the combo (if currently in Math Art mode)
        for i in range(self._generator_type_combo.count()):
            if self._generator_type_combo.itemData(i) is gen_cls:
                self._generator_type_combo.setCurrentIndex(i)
                break
        else:
            # Force-set the generator regardless of combo state
            self.set_generator(gen_cls())
        self._on_randomize()
        self._on_generate()

    def _on_generate(self) -> None:
        # Flush current UI state to model before reading sibling layers' generator_info
        self.flush_current_snapshot()

        if self._generator is None:
            QMessageBox.warning(
                self,
                "No Generator",
                "Please select a mode and generator first.",
            )
            return

        layer_id = self.current_layer_id()
        if layer_id is None:
            QMessageBox.warning(
                self,
                "No Target Layer",
                "Please add a layer to the project before generating.",
            )
            return

        params = self.get_params()
        canvas = self._controller.current_project.canvas

        from plottter.gui.generator_worker import GeneratorWorker

        if self._worker is not None and self._worker.isRunning() and not self._worker.is_cancelled():
            return  # already running

        # Inject sibling shapes for 3D HLR occlusion
        from plottter.generators.scene3d_generator import Scene3DGenerator
        from plottter.generators.mesh_slicer import MeshSlicerGenerator
        if isinstance(self._generator, (Scene3DGenerator, MeshSlicerGenerator)):
            params["_sibling_3d_shapes"] = self._build_sibling_3d_shapes(layer_id)

        # Inject preprocessed image for Math Art generators that use source image
        if (
            self._current_mode == "Math Art"
            and getattr(self._generator, "uses_source_image", False)
        ):
            params["_source_image"] = self._preprocessed_image

        self._worker = GeneratorWorker(self._generator, params, canvas)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda paths: self._on_generation_finished(paths, layer_id))
        self._worker.metadata_ready.connect(
            lambda meta: self._on_generation_metadata(meta, layer_id)
        )
        self._worker.error.connect(self._on_generation_error)
        self._worker.finished.connect(self._cleanup_generation_ui)
        self._worker.error.connect(self._cleanup_generation_ui)

        self._generate_btn.setEnabled(False)
        self._randomize_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)

        self._worker.start()

    def _on_progress(self, percent: int) -> None:
        self._progress_bar.setValue(percent)

    def _on_generation_finished(self, paths: list, layer_id: str) -> None:
        paths = self._apply_shared_transforms(paths)
        # Apply brush post-processing if a brush type is selected
        brush_widget = self._post_proc_widgets.get("brush_type")
        if isinstance(brush_widget, QComboBox):
            brush_type = brush_widget.currentText()
            if brush_type and brush_type != "None":
                brush_params: dict[str, Any] = {}
                for _bname, _bwidget in self._post_proc_widgets.items():
                    if _bname == "brush_type":
                        continue
                    if isinstance(_bwidget, (QDoubleSpinBox, QSpinBox)):
                        brush_params[_bname] = _bwidget.value()
                    elif isinstance(_bwidget, QComboBox):
                        brush_params[_bname] = _bwidget.currentText()
                try:
                    from plottter.processing.brush import apply_brush
                    paths = apply_brush(paths, brush_type, brush_params)
                except Exception:
                    pass
        self._controller.set_layer_paths(layer_id, paths, "Generate")

        # Auto-regenerate other 3D layers if enabled (task 62.2)
        if (
            self._current_mode == "3D Scene"
            and self._auto_regen_3d_cb.isChecked()
        ):
            self._trigger_auto_regen_siblings(layer_id)

    def _on_generation_metadata(self, meta: dict, source_layer_id: str) -> None:
        """Handle side-channel metadata emitted by GeneratorWorker after generation.

        The auto-created depth map preview layer was removed in task 16.57 — the
        depth map is now a first-class image source and is visible as the canvas
        overlay rather than a separate layer.  This handler is kept as a no-op so
        the GeneratorWorker.metadata_ready signal still has a valid connection.
        """

    def _on_generation_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Generation Error",
            f"An error occurred during generation:\n\n{message}",
        )

    def _cleanup_generation_ui(self, *_args: Any) -> None:
        self._generate_btn.setEnabled(True)
        self._randomize_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._cleanup_generation_ui()

    def _on_randomize(self) -> None:
        """Randomize all parameter values within their ranges."""
        if self._generator is None:
            return
        try:
            from plottter.generators.base import FloatParam, IntParam, ChoiceParam, BoolParam
        except ImportError:
            return

        for param in self._generator.get_parameters():
            if not param.randomizable:
                continue
            widget = self._param_widgets.get(param.name)
            if widget is None:
                continue
            if isinstance(param, FloatParam) and isinstance(widget, QDoubleSpinBox):
                widget.setValue(random.uniform(param.min, param.max))
            elif isinstance(param, IntParam) and isinstance(widget, QSpinBox):
                widget.setValue(random.randint(param.min, param.max))
            elif isinstance(param, ChoiceParam) and isinstance(widget, QComboBox):
                widget.setCurrentIndex(random.randrange(widget.count()))
            elif isinstance(param, BoolParam) and isinstance(widget, QCheckBox):
                widget.setChecked(random.choice([True, False]))
