"""_SnapshotMixin — generator type changes, set_generator, parameter snapshots."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)


class _SnapshotMixin:
    """Mixin for generator type changes, set_generator, and per-layer settings snapshots."""

    def _on_generator_type_changed(self, _index: int = 0) -> None:
        idx = self._generator_type_combo.currentIndex()
        if idx < 0:
            self.set_generator(None)
            return
        gen_cls = self._generator_type_combo.itemData(idx)
        if gen_cls is not None:
            self.set_generator(gen_cls())

    def set_generator(self, generator: Any) -> None:
        """Rebuild the parameter UI for a new generator."""
        # Deactivate FMM pick mode whenever the generator changes
        if self._canvas_ref is not None:
            self._canvas_ref.set_fmm_source_mode(False)
            self._canvas_ref.clear_fmm_source_marker()
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        self._generator = generator
        self._param_widgets.clear()
        self._param_labels.clear()

        # Clear params form
        while self._static_params_layout.rowCount() > 0:
            self._static_params_layout.removeRow(0)

        # Populate preset combo
        self._rebuild_preset_combo()

        if generator is None:
            # Clean up dynamic param state (no generator = no dynamic params)
            self._dynamic_rebuild_timer.stop()
            self._dynamic_param_specs = []
            self._dynamic_param_widgets.clear()
            self._dynamic_param_labels.clear()
            self._dynamic_overrides.clear()
            self._rebuild_dynamic_params()  # clears the dynamic layout
            return

        # Build param widgets
        try:
            from plottter.generators.base import (
                FloatParam, IntParam, ExpressionParam, ChoiceParam, BoolParam,
                StringParam, FontParam, ImageParam, FileParam,
            )
            from plottter.gui.widgets.font_picker import FontPicker
        except ImportError:
            FontParam = None  # type: ignore[assignment,misc]
            FontPicker = None  # type: ignore[assignment,misc]

        for param in generator.get_parameters():
            label = QLabel(param.label)
            if isinstance(param, FloatParam):
                widget: QWidget = QDoubleSpinBox()
                widget.setMinimum(param.min)  # type: ignore[attr-defined]
                widget.setMaximum(param.max)  # type: ignore[attr-defined]
                widget.setSingleStep(param.step)  # type: ignore[attr-defined]
                widget.setValue(param.default)  # type: ignore[attr-defined]
                widget.setDecimals(4)  # type: ignore[attr-defined]
            elif isinstance(param, IntParam):
                widget = QSpinBox()
                widget.setMinimum(param.min)  # type: ignore[attr-defined]
                widget.setMaximum(param.max)  # type: ignore[attr-defined]
                widget.setSingleStep(param.step)  # type: ignore[attr-defined]
                widget.setValue(param.default)  # type: ignore[attr-defined]
            elif isinstance(param, StringParam):
                if param.multiline:
                    widget = QPlainTextEdit(str(param.default))
                    widget.setMinimumHeight(400)  # type: ignore[attr-defined]
                    widget.setMaximumHeight(800)  # type: ignore[attr-defined]
                    # Use monospace font for code editing (covers TurtleToy and similar)
                    from PyQt6.QtGui import QFontDatabase
                    _mono_font = QFontDatabase.systemFont(
                        QFontDatabase.SystemFont.FixedFont
                    )
                    widget.setFont(_mono_font)  # type: ignore[attr-defined]
                else:
                    widget = QLineEdit(str(param.default))
            elif isinstance(param, ExpressionParam):
                widget = QLineEdit(str(param.default))  # type: ignore[attr-defined]
            elif isinstance(param, ChoiceParam):
                widget = QComboBox()
                widget.addItems(param.choices)  # type: ignore[attr-defined]
                idx = param.choices.index(param.default) if param.default in param.choices else 0  # type: ignore[attr-defined]
                widget.setCurrentIndex(idx)  # type: ignore[attr-defined]
                # When a choice changes, update visibility of conditional params
                widget.currentTextChanged.connect(self._update_param_visibility)  # type: ignore[attr-defined]
                # When a choice changes, update tooltip if choice_descriptions provided
                if param.choice_descriptions:
                    _choice_descs = param.choice_descriptions

                    def _make_choice_tooltip_updater(combo: QComboBox, descs: dict[str, str]) -> None:
                        def _update_choice_tooltip(text: str) -> None:
                            tip = descs.get(text, "")
                            combo.setToolTip(tip)
                        combo.currentTextChanged.connect(_update_choice_tooltip)
                        # Set initial tooltip
                        _update_choice_tooltip(combo.currentText())

                    _make_choice_tooltip_updater(widget, _choice_descs)  # type: ignore[arg-type]
            elif isinstance(param, BoolParam):
                widget = QCheckBox()
                widget.setChecked(param.default)  # type: ignore[attr-defined]
                # When a bool changes, update visibility of conditional params
                widget.stateChanged.connect(self._update_param_visibility)  # type: ignore[attr-defined]
            elif FontParam is not None and FontPicker is not None and isinstance(param, FontParam):
                widget = FontPicker()
                if param.default:
                    widget.set_font_path(param.default)  # type: ignore[attr-defined]
            elif isinstance(param, ImageParam):
                from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton
                import functools
                container = QWidget()
                row_layout = QHBoxLayout(container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                line_edit = QLineEdit(str(param.default) if param.default else "")
                browse_btn = QPushButton("Browse…")
                browse_btn.setFixedWidth(70)

                def _browse(le: QLineEdit) -> None:
                    path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select Image",
                        "",
                        "Images (*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tiff);;All Files (*)",
                    )
                    if path:
                        le.setText(path)

                browse_btn.clicked.connect(functools.partial(_browse, line_edit))
                row_layout.addWidget(line_edit)
                row_layout.addWidget(browse_btn)
                # Store container reference so _update_param_visibility can
                # hide/show the whole row (QLineEdit + Browse button) together.
                line_edit.setProperty("_image_container", container)
                if param.description:
                    container.setToolTip(param.description)
                    label.setToolTip(param.description)
                self._param_widgets[param.name] = line_edit
                self._param_labels[param.name] = label
                self._static_params_layout.addRow(label, container)
                continue
            elif isinstance(param, FileParam):
                from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton
                import functools
                container = QWidget()
                row_layout = QHBoxLayout(container)
                row_layout.setContentsMargins(0, 0, 0, 0)
                line_edit = QLineEdit(str(param.default) if param.default else "")
                browse_btn = QPushButton("Browse…")
                browse_btn.setFixedWidth(70)

                def _browse_file(le: QLineEdit, filt: str) -> None:
                    path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select File",
                        "",
                        filt,
                    )
                    if path:
                        le.setText(path)

                browse_btn.clicked.connect(functools.partial(_browse_file, line_edit, param.filter))
                row_layout.addWidget(line_edit)
                row_layout.addWidget(browse_btn)
                line_edit.setProperty("_image_container", container)
                if param.description:
                    container.setToolTip(param.description)
                    label.setToolTip(param.description)
                self._param_widgets[param.name] = line_edit
                self._param_labels[param.name] = label
                self._static_params_layout.addRow(label, container)
                continue
            else:
                widget = QLineEdit(str(getattr(param, "default", "")))

            # Apply description as tooltip on both the widget and its label.
            # For ChoiceParam with choice_descriptions, the combo tooltip is already
            # set per-choice by _make_choice_tooltip_updater; only set the generic
            # description on the label so we don't overwrite the initial choice tooltip.
            if param.description:
                has_choice_descs = isinstance(param, ChoiceParam) and bool(param.choice_descriptions)
                if not has_choice_descs:
                    widget.setToolTip(param.description)
                label.setToolTip(param.description)

            self._param_widgets[param.name] = widget
            self._param_labels[param.name] = label
            self._static_params_layout.addRow(label, widget)

        # If the generator has fmm_source_x_pct / fmm_source_y_pct parameters, inject
        # a "Pick on Canvas" button so the user can click to set the FMM source point.
        self._pick_fmm_source_btn = None
        self._pick_fmm_source_label = None
        if "fmm_source_x_pct" in self._param_widgets:
            from PyQt6.QtWidgets import QPushButton
            btn = QPushButton("Pick on Canvas")
            btn.setToolTip(
                "Click this button, then click on the image to set the FMM wave origin."
            )
            btn.clicked.connect(self._on_pick_fmm_source_clicked)
            lbl = QLabel("")
            self._pick_fmm_source_btn = btn
            self._pick_fmm_source_label = lbl
            self._static_params_layout.addRow(lbl, btn)

            # Keep the canvas marker in sync when the user edits spinboxes directly.
            x_widget = self._param_widgets.get("fmm_source_x_pct")
            y_widget = self._param_widgets.get("fmm_source_y_pct")
            if isinstance(x_widget, QDoubleSpinBox):
                x_widget.valueChanged.connect(self._update_fmm_marker)
            if isinstance(y_widget, QDoubleSpinBox):
                y_widget.valueChanged.connect(self._update_fmm_marker)

        # Apply initial visibility for params with visible_when conditions
        self._update_param_visibility()

        # For Math Art generators: show image source + preprocessing panels when the
        # generator declares uses_source_image = True.
        if self._current_mode == "Math Art":
            has_image_input = getattr(generator, "uses_source_image", False)
            self._image_source_group.setVisible(has_image_input)
            self._preprocessing_group.setVisible(has_image_input)

        # --- Task 135.1: wire debounce re-parse trigger for dynamic params ---
        # Reset dynamic param state for this generator (fresh start)
        self._dynamic_rebuild_timer.stop()
        self._dynamic_param_specs = []
        self._dynamic_param_widgets.clear()
        self._dynamic_param_labels.clear()
        self._dynamic_overrides.clear()

        # If the generator overrides get_dynamic_parameters, connect QPlainTextEdit
        # widgets (e.g. "code" in TurtleToy) to the 500 ms debounce timer.
        try:
            from plottter.generators.base import Generator as _GeneratorBase
            _has_dynamic = (
                type(generator).get_dynamic_parameters
                is not _GeneratorBase.get_dynamic_parameters
            )
        except ImportError:
            _has_dynamic = False

        if _has_dynamic:
            for _dyn_widget in self._param_widgets.values():
                if isinstance(_dyn_widget, QPlainTextEdit):
                    _dyn_widget.textChanged.connect(
                        lambda: self._dynamic_rebuild_timer.start()
                    )

        # Populate dynamic params immediately (synchronous, no debounce)
        self._rebuild_dynamic_params()

    def _update_param_visibility(self, *_args: Any) -> None:
        """Show/hide parameter rows based on their visible_when conditions."""
        if self._generator is None:
            return
        for param in self._generator.get_parameters():
            if param.visible_when is None:
                continue
            # Param is visible only when ALL conditions are satisfied
            visible = True
            for dep_name, allowed_values in param.visible_when.items():
                dep_widget = self._param_widgets.get(dep_name)
                if dep_widget is None:
                    continue
                if isinstance(dep_widget, QComboBox):
                    current = dep_widget.currentText()
                    if current not in allowed_values:
                        visible = False
                        break
                elif isinstance(dep_widget, QCheckBox):
                    current = dep_widget.isChecked()
                    if current not in allowed_values:
                        visible = False
                        break
            label_w = self._param_labels.get(param.name)
            field_w = self._param_widgets.get(param.name)
            if label_w is not None:
                label_w.setVisible(visible)
            if field_w is not None:
                field_w.setVisible(visible)

        # Show/hide the "Pick on Canvas" button based on fmm_source_point == "Custom"
        if self._fmm_btn_alive():
            source_widget = self._param_widgets.get("fmm_source_point")
            show_btn = (
                isinstance(source_widget, QComboBox)
                and source_widget.currentText() == "Custom"
            )
            self._pick_fmm_source_btn.setVisible(show_btn)  # type: ignore[union-attr]
            if self._pick_fmm_source_label is not None:
                self._pick_fmm_source_label.setVisible(show_btn)  # type: ignore[union-attr]
            if not show_btn and self._canvas_ref is not None:
                self._canvas_ref.set_fmm_source_mode(False)
                self._canvas_ref.clear_fmm_source_marker()

    def _update_post_proc_visibility(self, *_args: Any) -> None:
        """Show/hide post-processing parameter rows based on their visible_when conditions."""
        try:
            from plottter.generators.base import Generator as _Gen
            post_proc_params = _Gen.get_post_processing_parameters()
        except ImportError:
            return
        for param in post_proc_params:
            if param.visible_when is None:
                continue
            visible = True
            for dep_name, allowed_values in param.visible_when.items():
                dep_widget = self._post_proc_widgets.get(dep_name)
                if dep_widget is None:
                    continue
                if isinstance(dep_widget, QComboBox):
                    if dep_widget.currentText() not in allowed_values:
                        visible = False
                        break
            label_w = self._post_proc_labels.get(param.name)
            field_w = self._post_proc_widgets.get(param.name)
            if label_w is not None:
                label_w.setVisible(visible)
            if field_w is not None:
                field_w.setVisible(visible)

    def get_params(self) -> dict[str, Any]:
        """Collect current parameter values from the widgets."""
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FontPicker
        except ImportError:
            _FontPicker = None  # type: ignore[assignment,misc]

        result: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                result[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                result[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                result[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif _FontPicker is not None and isinstance(widget, _FontPicker):
                result[name] = widget.font_path()

        # Inject preprocessed image for image generators.  Generators that opt
        # in via ``uses_color_source = True`` receive the RGB version when
        # available (e.g. PixelArt's palette quantizer needs colour info);
        # everything else gets the grayscale version for back-compat.
        if self._current_mode == "Image to Lines" and self._preprocessed_image is not None:
            wants_color = bool(
                self._generator is not None
                and getattr(self._generator, "uses_color_source", False)
                and self._preprocessed_color is not None
            )
            result["_source_image"] = (
                self._preprocessed_color if wants_color else self._preprocessed_image
            )
            result.update(self._get_preprocessing_params())

        # Inject shared camera for 3D Scene generators
        if self._current_mode == "3D Scene":
            result["_camera"] = self._get_camera_dict()

        # Inject fetched map data for Map generators (spec §10.2)
        if self._current_mode == "Map" and getattr(self, "_map_data", None) is not None:
            result["_map_data"] = self._map_data

        # Inject map view for Map generators (spec §6.3)
        if self._current_mode == "Map" and getattr(self, "_map_view", None) is not None:
            result["_map_view"] = dict(self._map_view)

        return result

    def current_layer_id(self) -> str | None:
        """Return the currently selected target layer id."""
        idx = self._layer_combo.currentIndex()
        if idx >= 0:
            return self._layer_combo.itemData(idx)
        return None

    def _capture_multilayer_run_settings(self) -> dict:
        """Capture the panel state that produced a multi-layer run.

        Stored under ``_generator_settings`` in each run layer's
        ``generator_info``. ``_apply_multilayer_run_settings`` consumes this to
        restore the generator + params when the user later selects one of the
        run's layers — so they can tweak settings and re-generate in place,
        instead of seeing the panel stuck on whatever single-layer generator
        was active when they clicked the map layer.
        """
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FP
        except ImportError:
            _FP = None  # type: ignore[assignment,misc]

        params: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                params[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                params[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif _FP is not None and isinstance(widget, _FP):
                params[name] = widget.font_path()

        settings: dict[str, Any] = {
            "mode": self._current_mode,
            "generator_name": (
                self._generator_type_combo.currentText()
                if hasattr(self, "_generator_type_combo")
                else ""
            ),
            "params": params,
        }
        # Map-specific: persist the location so the cache can be reloaded.
        # (_map_data itself is too large to store on each layer; reload from
        # the location-keyed disk cache when the run is later selected.)
        if self._current_mode == "Map" and hasattr(self, "_map_location_edit"):
            settings["location"] = self._map_location_edit.text()
        return settings

    def _apply_multilayer_run_settings(self, info: dict) -> None:
        """Restore the mode/generator/params from a multi-layer run's stored settings.

        Called from ``_on_active_layer_changed`` when the selected layer carries
        both ``_generator_run_id`` and ``_generator_settings``. Switches mode +
        generator, restores widget values, and for Map also reloads the cached
        ``_map_data`` and re-initialises ``_map_view`` so a subsequent Generate
        replaces the run with the user's settings (and pan/zoom intact).
        """
        settings = info.get("_generator_settings")
        if not isinstance(settings, dict):
            return

        mode = settings.get("mode", "")
        if mode and mode != self._current_mode:
            # on_mode_changed runs synchronously and rebuilds the generator combo.
            self.mode_change_requested.emit(mode)

        gen_name = settings.get("generator_name", "")
        if gen_name:
            idx = self._generator_type_combo.findText(gen_name)
            if idx >= 0:
                self._generator_type_combo.blockSignals(True)
                self._generator_type_combo.setCurrentIndex(idx)
                self._generator_type_combo.blockSignals(False)
                # Always rebuild params UI so subsequent widget-set calls hit
                # the correct generator's widgets (mirrors _apply_settings_snapshot).
                self._on_generator_type_changed()

        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FP
        except ImportError:
            _FP = None  # type: ignore[assignment,misc]
        for name, value in settings.get("params", {}).items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                combo_idx = widget.findText(str(value))
                if combo_idx >= 0:
                    widget.setCurrentIndex(combo_idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif _FP is not None and isinstance(widget, _FP):
                widget.set_font_path(str(value))

        # Map-specific restore: location text, cached map_data, and the map view.
        if mode == "Map" and hasattr(self, "_map_location_edit"):
            location = settings.get("location", "")
            self._map_location_edit.setText(location)
            cached = None
            if location:
                try:
                    from plottter.osm.cache import (
                        cache_key as _cache_key,
                        load as _cache_load,
                    )
                    saved_params = settings.get("params", {})
                    radius = float(saved_params.get("radius_km", 1.5))
                    extent = str(saved_params.get("extent_mode", "radius"))
                    cats = (
                        self._get_enabled_map_categories(saved_params)
                        if hasattr(self, "_get_enabled_map_categories")
                        else []
                    )
                    cached = _cache_load(_cache_key(location, radius, extent, cats))
                except Exception:  # noqa: BLE001 — cache errors must not bubble up
                    cached = None
            if cached is not None:
                self._map_data = cached
                if hasattr(self, "_init_map_view_from_data"):
                    self._init_map_view_from_data(cached)
                # Re-push the preview features to the canvas so its
                # _map_features / _map_preview_polylines match the freshly-
                # reloaded MapData. Without this the canvas keeps the
                # features it captured when "Position Map" was first toggled
                # on, and clamp_map_view / preview projection diverge from
                # the panel's current data over time — symptom: after a few
                # regens the pan feels stuck or the rendered crop drifts.
                self._sync_canvas_map_preview()
                if hasattr(self, "_map_status_label"):
                    n = sum(len(v) for v in cached.features.values())
                    self._map_status_label.setText(
                        f"Loaded: {n:,} features (cached)"
                    )
            elif (
                self._map_data is not None
                and getattr(self._map_data, "location", None) == location
            ):
                # Cache key didn't match (e.g. user changed radius between
                # Fetch and Generate, so the cache was written under a
                # different key) but the in-session payload matches the
                # location — keep it so the user can re-Generate immediately.
                if hasattr(self, "_init_map_view_from_data"):
                    self._init_map_view_from_data(self._map_data)
                self._sync_canvas_map_preview()
            elif hasattr(self, "_map_status_label") and location:
                # No cached copy and no matching in-session data — prompt refetch.
                self._map_status_label.setText("Map data not cached — click Fetch")

    def _sync_canvas_map_preview(self) -> None:
        """Re-push the current self._map_data to the canvas so its preview
        features, bounds, and decimated polylines match the enabled
        categories. Safe no-op when the canvas isn't active or there's no
        data yet.

        Filters to the currently-enabled categories so the preview reflects
        what Generate will produce (the user disabling Water shouldn't see
        water polygons in the preview behind their generated roads).
        """
        if self._canvas_ref is None or getattr(self, "_map_data", None) is None:
            return
        if not getattr(self._canvas_ref, "_map_position_active", False):
            # Positioning mode isn't on — no preview to keep in sync. The
            # next "Position Map" click will re-push fresh data anyway.
            return
        try:
            from plottter.osm.geometry import data_bounds

            enabled_cats = (
                self._get_enabled_map_categories(self.get_params())
                if hasattr(self, "_get_enabled_map_categories")
                else None
            )
            if enabled_cats is not None:
                allow = set(enabled_cats)
                enabled_features = [
                    f
                    for cat, flist in self._map_data.features.items()
                    if cat in allow
                    for f in flist
                ]
            else:
                enabled_features = [
                    f for flist in self._map_data.features.values() for f in flist
                ]
            if not enabled_features:
                # Every category disabled — fall back to full extent so
                # bounds aren't degenerate.
                enabled_features = [
                    f for flist in self._map_data.features.values() for f in flist
                ]
                if not enabled_features:
                    return
            bounds = data_bounds(enabled_features)
            self._canvas_ref.set_map_preview_data(
                self._map_data, bounds, enabled_categories=enabled_cats
            )
            if self._map_view is not None:
                self._canvas_ref.update_map_view(dict(self._map_view))
        except Exception:  # noqa: BLE001 — best-effort, never crash the restore
            pass

    def _is_multilayer_run_member(self, layer_id: str | None) -> bool:
        """True if *layer_id* belongs to a multi-layer generator run.

        Layers emitted by a multi-layer generator (Map, Pixel Art) carry a
        ``_generator_run_id`` in their ``generator_info``. Their settings are
        owned by the run as a whole, not by the single-layer snapshot
        mechanism, so the snapshot save MUST skip them — overwriting a run
        member with a single-layer snapshot destroys the ``_generator_run_id``
        link, which makes re-generation append duplicates instead of replacing
        the run (and mislabels the layer as a different generator's output).
        """
        if not layer_id:
            return False
        layer = self._controller.get_layer(layer_id)
        return bool(
            layer is not None
            and isinstance(layer.generator_info, dict)
            and layer.generator_info.get("_generator_run_id")
        )

    def flush_current_snapshot(self) -> None:
        """Save current UI state to the active layer's generator_info."""
        snapshot = self._get_settings_snapshot()
        layer_id = self.current_layer_id()
        if (
            snapshot is not None
            and layer_id
            and not self._is_multilayer_run_member(layer_id)
        ):
            self._controller.set_layer_generator_info(layer_id, snapshot)

    # ------------------------------------------------------------------
    # Per-layer generator settings memory
    # ------------------------------------------------------------------

    def _get_settings_snapshot(self) -> dict | None:
        """Capture the current generator type, params, and transforms as a dict.

        Returns None if there is no active generator (e.g. Color Separation mode).
        """
        if self._generator is None:
            return None
        if self._current_mode not in ("Math Art", "Image to Lines", "3D Scene"):
            return None

        gen_name = self._generator_type_combo.currentText()

        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FontPickerSnap
        except ImportError:
            _FontPickerSnap = None  # type: ignore[assignment,misc]

        params: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                params[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                sentinel = widget.property("_sentinel")
                params[name] = sentinel if sentinel is not None else widget.text()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif _FontPickerSnap is not None and isinstance(widget, _FontPickerSnap):
                params[name] = widget.font_path()

        transforms = {
            "scale": self._transform_scale_spin.value(),
            "rotation": self._transform_rotation_spin.value(),
            "translate_x": self._transform_x_spin.value(),
            "translate_y": self._transform_y_spin.value(),
            "mirror_h": self._mirror_h_check.isChecked(),
            "mirror_v": self._mirror_v_check.isChecked(),
            "n_fold": self._n_fold_spin.value(),
            "tile_rows": self._tile_rows_spin.value(),
            "tile_cols": self._tile_cols_spin.value(),
        }

        snapshot: dict = {
            "generator_name": gen_name,
            "mode": self._current_mode,
            "params": params,
            "transforms": transforms,
            "image_source_type": self._image_source_type,
            "_dynamic_overrides": dict(self._dynamic_overrides),
        }
        # Persist depth map invert state alongside the source type
        try:
            snapshot["depth_map_invert"] = self._depth_invert_check.isChecked()
        except AttributeError:
            pass

        # Persist image size & position settings
        try:
            snapshot["image_fit_mode"] = self._image_fit_mode()
            snapshot["image_width_mm"] = self._image_width_spin.value()
            snapshot["image_height_mm"] = self._image_height_spin.value()
            snapshot["image_offset_x_mm"] = self._image_offset_x_spin.value()
            snapshot["image_offset_y_mm"] = self._image_offset_y_spin.value()
            snapshot["image_lock_aspect"] = self._lock_aspect_check.isChecked()
        except AttributeError:
            pass

        # Persist post-processing (brush) settings
        post_proc_params: dict[str, Any] = {}
        for _ppname, _ppwidget in self._post_proc_widgets.items():
            if isinstance(_ppwidget, (QDoubleSpinBox, QSpinBox)):
                post_proc_params[_ppname] = _ppwidget.value()
            elif isinstance(_ppwidget, QComboBox):
                post_proc_params[_ppname] = _ppwidget.currentText()
        if post_proc_params:
            snapshot["post_processing"] = post_proc_params

        return snapshot

    def _apply_settings_snapshot(self, info: dict) -> None:
        """Apply a saved generator settings snapshot to the UI.

        Restores the mode (via mode_change_requested signal), generator type,
        parameter values, and shared transforms.
        """
        mode = info.get("mode", "")
        gen_name = info.get("generator_name", "")
        params = info.get("params", {})
        transforms = info.get("transforms", {})

        # Switch mode if needed (ModePanel listens to mode_change_requested)
        if mode and mode != self._current_mode:
            self.mode_change_requested.emit(mode)
            # on_mode_changed() is called synchronously (direct connection),
            # which resets the generator combo to the first item.

        # Select the saved generator by name.
        # Always rebuild the parameter UI even if the same generator is already
        # selected — the parameter values differ per layer.
        if gen_name:
            idx = self._generator_type_combo.findText(gen_name)
            if idx >= 0:
                self._generator_type_combo.blockSignals(True)
                self._generator_type_combo.setCurrentIndex(idx)
                self._generator_type_combo.blockSignals(False)
                self._on_generator_type_changed()

        # Restore parameter values
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPApply
        except ImportError:
            _FPApply = None  # type: ignore[assignment,misc]

        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                combo_idx = widget.findText(str(value))
                if combo_idx >= 0:
                    widget.setCurrentIndex(combo_idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif _FPApply is not None and isinstance(widget, _FPApply):
                widget.set_font_path(str(value))

        # Restore shared transform values
        if "scale" in transforms:
            self._transform_scale_spin.setValue(transforms["scale"])
        if "rotation" in transforms:
            self._transform_rotation_spin.setValue(transforms["rotation"])
        if "translate_x" in transforms:
            self._transform_x_spin.setValue(transforms["translate_x"])
        if "translate_y" in transforms:
            self._transform_y_spin.setValue(transforms["translate_y"])
        if "mirror_h" in transforms:
            self._mirror_h_check.setChecked(transforms["mirror_h"])
        if "mirror_v" in transforms:
            self._mirror_v_check.setChecked(transforms["mirror_v"])
        if "n_fold" in transforms:
            self._n_fold_spin.setValue(transforms["n_fold"])
        if "tile_rows" in transforms:
            self._tile_rows_spin.setValue(transforms["tile_rows"])
        if "tile_cols" in transforms:
            self._tile_cols_spin.setValue(transforms["tile_cols"])

        # Re-evaluate conditional visibility after restoring params
        self._update_param_visibility()

        # Restore image source type (file / layer / depth_map)
        src_type = info.get("image_source_type", "file")
        if src_type == "depth_map":
            self._src_type_depth_radio.setChecked(True)
        elif src_type == "layer":
            self._src_type_layer_radio.setChecked(True)
        else:
            self._src_type_file_radio.setChecked(True)

        # Restore depth map invert state
        if "depth_map_invert" in info:
            try:
                self._depth_invert_check.setChecked(bool(info["depth_map_invert"]))
            except AttributeError:
                pass

        # Restore image size & position settings
        try:
            if "image_fit_mode" in info:
                mode = info["image_fit_mode"]
                if mode == "fit":
                    idx = self._image_fit_combo.findText("Fit (Keep Aspect)")
                elif mode == "custom":
                    idx = self._image_fit_combo.findText("Custom Size")
                else:
                    idx = self._image_fit_combo.findText("Fill Canvas")
                if idx >= 0:
                    self._image_fit_combo.blockSignals(True)
                    self._image_fit_combo.setCurrentIndex(idx)
                    self._image_fit_combo.blockSignals(False)
                    self._on_image_fit_mode_changed()
            if "image_width_mm" in info:
                self._image_width_spin.setValue(float(info["image_width_mm"]))
            if "image_height_mm" in info:
                self._image_height_spin.setValue(float(info["image_height_mm"]))
            if "image_offset_x_mm" in info:
                self._image_offset_x_spin.setValue(float(info["image_offset_x_mm"]))
            if "image_offset_y_mm" in info:
                self._image_offset_y_spin.setValue(float(info["image_offset_y_mm"]))
            if "image_lock_aspect" in info:
                self._lock_aspect_check.setChecked(bool(info["image_lock_aspect"]))
        except AttributeError:
            pass

        # Restore post-processing (brush) settings
        post_proc = info.get("post_processing", {})
        for _ppname, _ppvalue in post_proc.items():
            _ppwidget = self._post_proc_widgets.get(_ppname)
            if _ppwidget is None:
                continue
            if isinstance(_ppwidget, (QDoubleSpinBox, QSpinBox)):
                _ppwidget.setValue(_ppvalue)
            elif isinstance(_ppwidget, QComboBox):
                _pp_idx = _ppwidget.findText(str(_ppvalue))
                if _pp_idx >= 0:
                    _ppwidget.setCurrentIndex(_pp_idx)
        self._update_post_proc_visibility()

        # Restore dynamic overrides and rebuild the dynamic-params section
        # (spec §4.5 step 3).  The code widget was just restored above,
        # which armed the 500 ms debounce timer.  We bypass the timer and
        # rebuild synchronously so the saved override values are visible
        # immediately.
        #
        # Two-step process (per spec: "call _rebuild_dynamic_params then
        # write the override values into the widgets"):
        #   1. Clear _dynamic_overrides and rebuild — widgets appear with
        #      their declared defaults.
        #   2. Write saved override values directly into the new widgets and
        #      update _dynamic_overrides so future operations see them.
        saved_overrides = info.get("_dynamic_overrides")
        if isinstance(saved_overrides, dict):
            self._dynamic_overrides.clear()
            self._rebuild_dynamic_params()
            self._dynamic_overrides.update(saved_overrides)
            for _ov_name, _ov_value in saved_overrides.items():
                _ov_widget = self._dynamic_param_widgets.get(_ov_name)
                _ov_param = next(
                    (p for p in self._dynamic_param_specs if p.name == _ov_name),
                    None,
                )
                if _ov_widget is not None and _ov_param is not None:
                    self._set_dynamic_widget_value(_ov_widget, _ov_param, _ov_value)

    def _on_active_layer_changed(self, layer_id: str) -> None:
        """Handle active layer change: save current settings to old layer, restore new."""
        # Deactivate FMM pick mode when switching layers
        if self._canvas_ref is not None:
            self._canvas_ref.set_fmm_source_mode(False)
            self._canvas_ref.clear_fmm_source_marker()
        if self._fmm_btn_alive():
            self._pick_fmm_source_btn.setText("Pick on Canvas")  # type: ignore[union-attr]

        # Save current settings to the layer currently shown in the target combo.
        # Skip layers that belong to a multi-layer run — overwriting their
        # generator_info would destroy the _generator_run_id link (see
        # _is_multilayer_run_member).
        prev_layer_id = self.current_layer_id()
        if (
            prev_layer_id
            and prev_layer_id != layer_id
            and not self._is_multilayer_run_member(prev_layer_id)
        ):
            snapshot = self._get_settings_snapshot()
            if snapshot is not None:
                self._controller.set_layer_generator_info(prev_layer_id, snapshot)

        # Switch the target layer combo to the new active layer (no signal to avoid loop)
        idx = self._layer_combo.findData(layer_id)
        if idx >= 0:
            self._layer_combo.blockSignals(True)
            self._layer_combo.setCurrentIndex(idx)
            self._layer_combo.blockSignals(False)

        # Restore the new layer's saved settings (if any). Multi-layer runs
        # (Map, Pixel Art) take precedence — their settings are stored at
        # generation time on every run layer so selecting any of them brings
        # the panel back to that run's exact configuration.
        new_layer = self._controller.get_layer(layer_id)
        if new_layer is not None and isinstance(new_layer.generator_info, dict):
            info = new_layer.generator_info
            if (
                info.get("_generator_run_id")
                and isinstance(info.get("_generator_settings"), dict)
            ):
                self._apply_multilayer_run_settings(info)
            elif info.get("mode") in ("Math Art", "Image to Lines", "3D Scene"):
                self._apply_settings_snapshot(info)

    def _on_generator_info_changed(self, layer_id: str) -> None:
        """Refresh offset param widgets when generator_info changes for the active layer.

        Called after an undoable operation (e.g. MoveLayerCommand) updates
        ``generator_info`` on the active layer.  Only the ``x_offset_mm`` and
        ``y_offset_mm`` spinboxes are touched — the rest of the UI is left intact
        to avoid disrupting the user's workflow.
        """
        if layer_id != self._controller.active_layer_id:
            return
        layer = self._controller.get_layer(layer_id)
        if layer is None or not isinstance(layer.generator_info, dict):
            return
        params = layer.generator_info.get("params", {})
        for name in ("x_offset_mm", "y_offset_mm", "pos_x", "pos_y"):
            widget = self._param_widgets.get(name)
            if widget is not None and isinstance(widget, QDoubleSpinBox) and name in params:
                widget.blockSignals(True)
                widget.setValue(float(params[name]))
                widget.blockSignals(False)

    def _on_project_loaded(self) -> None:
        """Reset per-layer settings memory tracking when a new project is loaded."""
        # Clear any stale layer selection so _on_active_layer_changed starts fresh
        self._load_camera_from_project()
        self._restore_image_view_from_metadata()

    def _restore_image_view_from_metadata(self) -> None:
        """Re-apply a persisted image_view dict from ``project.metadata``.

        Mirrors the map_view restore path: when a .plottter file is loaded
        that has an ``image_view`` entry, populate the fit-mode combo and
        custom-size / offset spinboxes so the next preview emit places the
        overlay where the user left it.
        """
        try:
            project = self._controller.current_project
            view = project.metadata.get("image_view") if project else None
        except Exception:  # noqa: BLE001
            return
        if not isinstance(view, dict):
            return
        fit_mode = str(view.get("fit_mode", "custom"))
        combo_text = {
            "fit": "Fit (Keep Aspect)",
            "fill": "Fill Canvas",
            "custom": "Custom Size",
        }.get(fit_mode, "Custom Size")
        self._image_fit_combo.blockSignals(True)
        self._image_fit_combo.setCurrentText(combo_text)
        self._image_fit_combo.blockSignals(False)
        for key, spin in (
            ("custom_w_mm", self._image_width_spin),
            ("custom_h_mm", self._image_height_spin),
            ("offset_x_mm", self._image_offset_x_spin),
            ("offset_y_mm", self._image_offset_y_spin),
        ):
            if key in view:
                spin.blockSignals(True)
                spin.setValue(float(view[key]))
                spin.blockSignals(False)
        # Let _on_image_fit_mode_changed surface the right widgets and
        # trigger one debounced preview refresh.
        self._on_image_fit_mode_changed()
