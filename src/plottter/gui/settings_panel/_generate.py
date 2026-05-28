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

        # Inject preprocessed image for Math Art generators that use source image.
        # Generators that set ``uses_color_source = True`` receive the RGB
        # version when available; others get the grayscale version.
        if (
            self._current_mode == "Math Art"
            and getattr(self._generator, "uses_source_image", False)
        ):
            wants_color = (
                getattr(self._generator, "uses_color_source", False)
                and self._preprocessed_color is not None
            )
            params["_source_image"] = (
                self._preprocessed_color if wants_color else self._preprocessed_image
            )

        # For multi-layer generators, find any prior run of THIS generator
        # anywhere in the project (not just on the active layer — the user
        # may have clicked Generate while still on the original empty
        # "Layer 1"). _on_multilayer_generation_finished uses the captured
        # run id to replace that run instead of appending duplicates.
        if getattr(self._generator, "emits_multiple_layers", False):
            prior_run_id: str | None = None
            for proj_layer in self._controller.current_project.layers:
                info = proj_layer.generator_info
                if (
                    isinstance(info, dict)
                    and info.get("_generator_name") == self._generator.name
                    and info.get("_generator_run_id")
                ):
                    prior_run_id = info["_generator_run_id"]
            self._pending_multilayer_regen_run_id = prior_run_id
            # Capture the settings NOW — the panel currently shows exactly what
            # is being generated. Capturing later in _on_multilayer_generation_finished
            # is unsafe because removing the old run's layers can re-select a
            # non-map layer in the layer panel, which triggers
            # _on_active_layer_changed and snaps the panel to that layer's
            # (single-layer) generator before our capture runs — so we'd store
            # the wrong generator's settings on every new map layer.
            self._pending_multilayer_run_settings = (
                self._capture_multilayer_run_settings()
                if hasattr(self, "_capture_multilayer_run_settings")
                else None
            )

        # Merge dynamic overrides as a reserved param key so generators
        # (and future machinery) can inspect them without touching static params.
        params["_dynamic_overrides"] = dict(self._dynamic_overrides)

        self._worker = GeneratorWorker(self._generator, params, canvas)
        self._worker.progress.connect(self._on_progress)
        if getattr(self._generator, "emits_multiple_layers", False):
            self._worker.layers_finished.connect(
                lambda specs: self._on_multilayer_generation_finished(specs)
            )
            self._worker.layers_finished.connect(self._cleanup_generation_ui)
        else:
            self._worker.finished.connect(lambda paths: self._on_generation_finished(paths, layer_id))
            self._worker.metadata_ready.connect(
                lambda meta: self._on_generation_metadata(meta, layer_id)
            )
            self._worker.finished.connect(self._cleanup_generation_ui)
        self._worker.error.connect(self._on_generation_error)
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

        # Persist _dynamic_overrides into generator_info (spec §5.1).
        # flush_current_snapshot() was already called at the start of
        # _on_generate(); now we stitch in the current overrides so they
        # survive project save and layer-switch restore.
        layer = self._controller.get_layer(layer_id)
        if layer is not None and isinstance(layer.generator_info, dict):
            updated_info = dict(layer.generator_info)
            updated_info["_dynamic_overrides"] = dict(self._dynamic_overrides)
            self._controller.set_layer_generator_info(layer_id, updated_info)

        # Auto-regenerate other 3D layers if enabled (task 62.2)
        if (
            self._current_mode == "3D Scene"
            and self._auto_regen_3d_cb.isChecked()
        ):
            self._trigger_auto_regen_siblings(layer_id)

    def _on_multilayer_generation_finished(self, layer_specs: list) -> None:
        """Handle results from a multi-layer generator.

        Creates one new :class:`~plottter.models.Layer` per
        :class:`~plottter.generators.base.LayerSpec` returned by
        ``generate_layers()``.  Each new layer is tagged with a fresh
        ``_generator_run_id`` inside its ``generator_info`` dict.

        If ``_pending_multilayer_regen_run_id`` was set by ``_on_generate``
        (i.e. the active layer already belonged to a previous run of the same
        multi-layer generator), all layers from that old run are removed first
        so re-generation replaces rather than appends.  The remove + add
        operations are grouped into a single undo macro.
        """
        import uuid
        from plottter.models import Layer

        new_run_id = str(uuid.uuid4())
        old_run_id = getattr(self, "_pending_multilayer_regen_run_id", None)
        self._pending_multilayer_regen_run_id = None
        # Settings were captured in _on_generate, before removing old layers
        # could side-effect the panel into a different mode/generator.
        run_settings = getattr(self, "_pending_multilayer_run_settings", None)
        self._pending_multilayer_run_settings = None

        first_new_layer_id: str | None = None
        macro_name = "Regenerate Layers" if old_run_id else "Generate Layers"
        self._controller.undo_stack.beginMacro(macro_name)
        try:
            # Remove layers that belong to the previous run of this generator.
            if old_run_id:
                old_ids = [
                    layer.id
                    for layer in self._controller.current_project.layers
                    if isinstance(layer.generator_info, dict)
                    and layer.generator_info.get("_generator_run_id") == old_run_id
                ]
                for lid in old_ids:
                    self._controller.remove_layer(lid)

            # Add the freshly generated layers, each tagged with the generator
            # name + new run id (so future regenerate finds them regardless of
            # which layer is active) AND a snapshot of the settings that
            # produced the run (so selecting any run layer can restore the
            # generator/params in the panel — see _apply_multilayer_run_settings).
            generator_name = getattr(self._generator, "name", "")
            for spec in layer_specs:
                gen_info: dict = {
                    "_generator_name": generator_name,
                    "_generator_run_id": new_run_id,
                }
                if run_settings is not None:
                    gen_info["_generator_settings"] = run_settings
                layer = Layer(
                    name=spec.name,
                    color=spec.color,
                    paths=spec.paths,
                    generator_info=gen_info,
                )
                self._controller.add_layer(layer)
                if first_new_layer_id is None:
                    first_new_layer_id = layer.id
        finally:
            self._controller.undo_stack.endMacro()

        # Removing old run layers can re-select a non-run layer in the layer
        # panel (auto-pick on delete), which would leave the panel snapped to
        # that layer's single-layer generator. Re-activate one of the newly
        # created run layers so the panel returns to the multi-layer generator
        # the user just generated with.
        if first_new_layer_id is not None:
            self._controller.set_active_layer(first_new_layer_id)

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
