"""_ColorSepMixin — color separation methods."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QMessageBox,
)

from .workers import _AiBgWorker, _AiSegmentWorker


def _separation_mask_to_luminance(mask: np.ndarray, src_img: np.ndarray) -> np.ndarray:
    """Convert a color-separation mask to a luminance grayscale image.

    All line generators interpret their input as **luminance**: high pixel
    value → light → fewer / shorter strokes.  Each color-separation method
    needs different massaging to get there:

    * **K-Means / Luminance** produce a boolean cluster mask.  The original
      grayscale already has luminance semantics, so we just whiten out the
      pixels that don't belong to the cluster.
    * **RGB / CMYK** produce a uint8 channel-intensity image where 255
      means "lots of this channel wanted".  That's the *opposite* of
      luminance, so we invert here.  Without this step every line
      generator drew ink where the channel was absent rather than where
      it was strong — visually a colour-negated plot.
    """
    if mask.dtype == np.bool_:
        if src_img.ndim == 3:
            from plottter.io.image_import import to_grayscale
            gray = to_grayscale(src_img)
        else:
            gray = src_img.copy()
        masked_gray = gray.copy()
        masked_gray[~mask] = 255  # outside-cluster pixels → white
        return masked_gray
    # RGB / CMYK: flip ink-coverage → luminance.
    return (255 - mask).astype(np.uint8)


class _ColorSepMixin:
    """Mixin for color separation methods."""

    def _on_color_sep_method_changed(self, method: str) -> None:
        is_kmeans = method == "K-Means"
        is_lum = method == "Luminance"
        is_rgb = method == "RGB"
        is_cmyk = method == "CMYK"
        is_ai = method == "AI Layer Separation"
        is_palette = method == "Custom Palette"

        self._color_sep_num_colors_spin.setVisible(is_kmeans or is_lum or is_ai)
        self._color_sep_num_colors_label.setVisible(is_kmeans or is_lum or is_ai)
        if is_kmeans:
            self._color_sep_num_colors_spin.setRange(2, 8)
            self._color_sep_num_colors_label.setText("Colors")
        elif is_lum:
            self._color_sep_num_colors_spin.setRange(2, 5)
            self._color_sep_num_colors_label.setText("Bands")
        elif is_ai:
            self._color_sep_num_colors_spin.setRange(2, 8)
            self._color_sep_num_colors_label.setText("Segments")

        # Build channel checkboxes
        self._channel_check_widget.setVisible(is_rgb or is_cmyk)
        # K Amount slider is only meaningful for CMYK separation.
        self._cmyk_k_amount_widget.setVisible(is_cmyk)
        # Palette picker is only shown for Custom Palette mode.
        self._palette_picker_widget.setVisible(is_palette)
        self._palette_dither_combo.setVisible(is_palette)

        if is_palette:
            self._populate_palette_picker()

        layout = self._channel_check_widget.layout()
        # Clear existing checkboxes
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._channel_checks.clear()

        if is_rgb:
            for ch in ("Red", "Green", "Blue"):
                cb = QCheckBox(ch)
                cb.setChecked(True)
                layout.addWidget(cb)
                self._channel_checks[ch] = cb
        elif is_cmyk:
            for ch in ("Cyan", "Magenta", "Yellow", "Key (Black)"):
                cb = QCheckBox(ch)
                cb.setChecked(True)
                layout.addWidget(cb)
                self._channel_checks[ch] = cb

    def _on_cmyk_k_amount_changed(self, _value: float | None = None) -> None:
        """Refresh cached CMYK channel masks when the K Amount spinbox moves.

        ``cmyk_separate`` runs at separation time and the resulting channel
        images are cached in ``_layer_masks`` keyed by layer ID — the layers
        themselves are visible to the user.  Generate Lines reads from this
        cache, never re-running ``cmyk_separate``.  Without this handler,
        moving the K Amount spinbox after Separate had no effect: the K
        layer's cached mask was frozen at the value used during the most
        recent Separate click.

        Approach: keep the visible layers and any user-generated paths
        intact; swap the underlying mask array in-place so the *next*
        Generate Lines uses the new K Amount.  Skips silently when no CMYK
        separation has been done — the spinbox is only meaningful in that
        context anyway.
        """
        if self._last_sep_method != "CMYK":
            return
        if self._cmyk_raw_rgb is None or not self._separated_layer_ids:
            return
        from plottter.color import cmyk_separate

        new_results = cmyk_separate(
            self._cmyk_raw_rgb,
            k_amount=float(self._cmyk_k_amount_spin.value()),
        )
        # Apply the same channel checkbox filter the Separate step used so
        # the post-filter ordering matches ``_separated_layer_ids``.
        channel_names = ["Cyan", "Magenta", "Yellow", "Key (Black)"]
        filtered = [
            (mask, color)
            for (mask, color), ch in zip(new_results, channel_names)
            if ch not in self._channel_checks
            or self._channel_checks[ch].isChecked()
        ]
        for layer_id, (new_mask, _color) in zip(self._separated_layer_ids, filtered):
            cached = self._layer_masks.get(layer_id)
            if cached is None:
                continue
            # Replace the mask, keep the preprocessed image reference.
            self._layer_masks[layer_id] = (new_mask, cached[1])

    def _populate_palette_picker(self, select_name: str | None = None) -> None:
        """Rebuild the palette picker combo from built-in + user palettes.

        Built-in presets appear first in their natural order; user palettes
        follow alphabetically.  If a user palette's name clashes with a
        built-in, it is shown as ``"<name> (user)"`` in the picker while the
        built-in retains the plain name.  The underlying ``PenPalette`` object
        is always stored as the item data (unmodified name).

        Parameters
        ----------
        select_name:
            If given, the item whose display text matches this string will be
            made current after the combo is rebuilt.  Pass the display name
            (with ``" (user)"`` suffix if applicable) rather than the palette's
            raw ``name`` attribute.
        """
        from plottter.color.palette import load_user_palettes
        from plottter.color.palettes import list_presets

        builtin = list_presets()
        builtin_names = {p.name for p in builtin}
        user = load_user_palettes()

        self._palette_picker_combo.blockSignals(True)
        self._palette_picker_combo.clear()

        # Built-in presets first (insertion order).
        for palette in builtin:
            self._palette_picker_combo.addItem(palette.name, palette)

        # User palettes after, with "(user)" suffix on name collision.
        for palette in user:
            display_name = (
                f"{palette.name} (user)"
                if palette.name in builtin_names
                else palette.name
            )
            self._palette_picker_combo.addItem(display_name, palette)

        # Restore selection by display name if requested.
        if select_name is not None:
            for i in range(self._palette_picker_combo.count()):
                if self._palette_picker_combo.itemText(i) == select_name:
                    self._palette_picker_combo.setCurrentIndex(i)
                    break

        self._palette_picker_combo.blockSignals(False)

    def _on_edit_palette(self) -> None:
        """Open the palette editor for the currently selected palette.

        On dialog acceptance, refreshes the palette picker and selects the
        palette that was just saved.
        """
        from plottter.gui.dialogs.palette_editor_dialog import PaletteEditorDialog
        from plottter.color.palettes import list_presets

        current_palette = self._palette_picker_combo.currentData()
        dlg = PaletteEditorDialog(parent=self, initial=current_palette)
        if dlg.exec():
            result = dlg.get_result()
            if result is None:
                return
            # Determine the display name used in the picker (may have "(user)" suffix).
            builtin_names = {p.name for p in list_presets()}
            display_name = (
                f"{result.name} (user)"
                if result.name in builtin_names
                else result.name
            )
            self._populate_palette_picker(select_name=display_name)

    def _on_palette_dither_changed(self, text: str) -> None:
        """Persist the selected dither method to QSettings."""
        from PyQt6.QtCore import QSettings
        QSettings("Plottter", "Plottter").setValue("colorsep/palette_dither", text)

    def _rebuild_color_sep_preset_combo(self) -> None:
        """Rebuild the color separation preset combo based on the selected generator."""
        self._color_sep_preset_combo.blockSignals(True)
        self._color_sep_preset_combo.clear()

        # Always add "Default" as first item with None data
        self._color_sep_preset_combo.addItem("Default", None)

        # Get the currently selected generator class
        gen_cls = self._color_sep_gen_combo.currentData()
        if gen_cls is None:
            self._color_sep_preset_combo.blockSignals(False)
            return

        try:
            # Instantiate the generator to get its presets
            gen_instance = gen_cls()
            presets = gen_instance.get_presets()

            # Add built-in presets
            for preset in presets:
                self._color_sep_preset_combo.addItem(preset.name, preset.params)

            # Load and add user presets
            try:
                from plottter.presets.user_presets import load_user_presets

                user_presets = load_user_presets(gen_cls.name)
                if user_presets:
                    self._color_sep_preset_combo.insertSeparator(
                        self._color_sep_preset_combo.count()
                    )
                    self._color_sep_preset_combo.addItem("— User Presets —")
                    # Make the section header non-selectable
                    header_idx = self._color_sep_preset_combo.count() - 1
                    model = self._color_sep_preset_combo.model()
                    if model is not None:
                        header_item = model.item(header_idx)
                        if header_item is not None:
                            header_item.setFlags(
                                header_item.flags()
                                & ~Qt.ItemFlag.ItemIsEnabled
                                & ~Qt.ItemFlag.ItemIsSelectable
                            )
                    for user_preset in user_presets:
                        self._color_sep_preset_combo.addItem(
                            user_preset.name, user_preset.params
                        )
            except Exception:
                pass  # User presets are optional; ignore failures

        except Exception:
            pass  # If generator instantiation fails, just show Default

        self._color_sep_preset_combo.blockSignals(False)

    def _on_ai_bg_changed(self, state: int) -> None:
        """Handle AI Background Removal toggle: disable manual BG removal when AI is on."""
        ai_on = bool(state)
        self._remove_bg_check.setEnabled(not ai_on)
        if ai_on:
            self._remove_bg_check.setChecked(False)
            self._bg_tolerance_spin.setEnabled(False)
        # Enable Apply button only when checkbox is on and API key is available
        self._apply_ai_bg_btn.setEnabled(ai_on and self._ai_key_available)
        self._on_preprocessing_changed()

    def update_ai_availability(self) -> None:
        """Enable/disable AI controls based on whether a Replicate API key is configured."""
        try:
            from PyQt6.QtCore import QSettings
            from plottter.ai.replicate_client import ReplicateClient
            settings = QSettings("Plottter", "Plottter")
            api_key = settings.value("replicate/api_key", "") or ""
            client = ReplicateClient(api_key=api_key)
            ai_available = client.is_available()
        except Exception:
            ai_available = False

        _no_key_tip = "Enter a Replicate API key in Preferences > AI Integration to enable"

        self._ai_key_available = ai_available
        has_cached_bg = self._ai_bg_rgba is not None

        # Update cached indicator visibility
        self._ai_bg_cached_label.setVisible(has_cached_bg)

        if ai_available:
            self._ai_bg_check.setEnabled(True)
            self._ai_bg_check.setToolTip("")
            self._apply_ai_bg_btn.setEnabled(self._ai_bg_check.isChecked())
            # AI mask generation — disabled in Manual Brush mode since no AI call is needed
            is_manual_mode = self._ai_mask_mode_combo.currentText() == "Manual Brush"
            self._ai_mask_generate_btn.setEnabled(not is_manual_mode)
            self._ai_mask_generate_btn.setToolTip("")
        else:
            # When no API key, allow enabling the checkbox if a cached result is available
            # so the user can activate BG removal without an API call.
            if has_cached_bg:
                self._ai_bg_check.setEnabled(True)
                self._ai_bg_check.setToolTip(
                    "Cached result available — no API key needed to use it"
                )
            else:
                self._ai_bg_check.setChecked(False)
                self._ai_bg_check.setEnabled(False)
                self._ai_bg_check.setToolTip(_no_key_tip)
            self._apply_ai_bg_btn.setEnabled(False)
            # AI mask generation
            self._ai_mask_generate_btn.setEnabled(False)
            self._ai_mask_generate_btn.setToolTip(_no_key_tip)

    def _on_apply_ai_bg(self) -> None:
        """Start a background thread to call AI background removal on the current image."""
        if self._raw_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if self._ai_bg_worker is not None and self._ai_bg_worker.isRunning():
            return  # already running

        from PyQt6.QtCore import QSettings

        settings = QSettings("Plottter", "Plottter")
        api_key = settings.value("replicate/api_key", "") or ""

        source_img = self._raw_image
        if source_img.ndim == 2:
            source_img = np.stack([source_img] * 3, axis=-1)
        elif source_img.ndim == 3 and source_img.shape[2] == 4:
            source_img = source_img[:, :, :3]

        cache_dir = self._get_cache_dir()
        self._apply_ai_bg_btn.setEnabled(False)
        self._ai_bg_worker = _AiBgWorker(api_key=api_key, image=source_img, cache_dir=cache_dir)
        self._ai_bg_worker.finished.connect(self._on_ai_bg_result)
        self._ai_bg_worker.error.connect(self._on_ai_bg_error)
        self._ai_bg_worker.finished.connect(
            lambda _: self._apply_ai_bg_btn.setEnabled(self._ai_key_available and self._ai_bg_check.isChecked())
        )
        self._ai_bg_worker.error.connect(
            lambda _: self._apply_ai_bg_btn.setEnabled(self._ai_key_available and self._ai_bg_check.isChecked())
        )
        self._ai_bg_worker.start()

    def _on_ai_bg_result(self, rgba: "np.ndarray") -> None:
        """Store the AI background removal result and refresh the preview."""
        self._ai_bg_rgba = rgba
        self._ai_bg_cached_label.setVisible(True)
        self._update_image_preview()

    def _on_ai_bg_error(self, msg: str) -> None:
        QMessageBox.critical(self, "AI Background Removal Error", msg)

    def _on_separate(self) -> None:
        """Run color separation and create one layer per cluster/channel."""
        if self._raw_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            from plottter.io.image_import import preprocess
            params = self._get_preprocessing_params()
            # If AI BG removal is active, composite onto white before
            # preprocessing — same logic as _update_image_preview().
            source = self._raw_image
            if (
                self._ai_bg_check.isChecked()
                and self._ai_bg_rgba is not None
            ):
                rgba = self._ai_bg_rgba
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb = rgba[:, :, :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                source = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            preprocessed = preprocess(source, params)
        except Exception as exc:
            QMessageBox.critical(self, "Preprocessing Error", str(exc))
            return

        method = self._color_sep_method_combo.currentText()
        num = self._color_sep_num_colors_spin.value()

        if method == "AI Layer Separation":
            # Network call — run in a background QThread to keep the GUI responsive.
            from PyQt6.QtCore import QSettings
            from plottter.ai.replicate_client import ReplicateClient

            settings = QSettings("Plottter", "Plottter")
            api_key = settings.value("replicate/api_key", "") or ""
            client = ReplicateClient(api_key=api_key)
            if not client.is_available():
                QMessageBox.warning(
                    self,
                    "AI Unavailable",
                    "AI Layer Separation requires a Replicate API key.\n"
                    "Set your Replicate API key in Preferences > AI Integration.",
                )
                return

            source_img = source
            if source_img.ndim == 2:
                source_img = np.stack([source_img] * 3, axis=-1)
            elif source_img.ndim == 3 and source_img.shape[2] == 4:
                source_img = source_img[:, :, :3]

            # Store preprocessed so the finished callback can use it for mask association
            self._ai_sep_preprocessed = preprocessed

            self._separate_btn.setEnabled(False)
            self._color_sep_progress.setMaximum(0)  # indeterminate while waiting for AI
            self._color_sep_progress.setVisible(True)

            self._ai_segment_worker = _AiSegmentWorker(
                api_key=api_key, image=source_img, num_segments=num
            )
            self._ai_segment_worker.progress.connect(
                lambda p: self._color_sep_progress.setValue(p)
            )
            self._ai_segment_worker.finished.connect(
                lambda results: self._on_ai_segment_finished(results, method)
            )
            self._ai_segment_worker.error.connect(self._on_ai_segment_error)
            self._ai_segment_worker.start()
            return  # layer creation happens asynchronously in _on_ai_segment_finished

        # The synchronous separators (K-Means, Luminance, RGB, CMYK, Custom
        # Palette) can take several seconds on large images and block the GUI
        # thread.  Show a busy cursor + indeterminate progress bar + disable
        # the Separate button so the user sees the app is working.  AI gets
        # its own progress already (see the return above).
        self._separate_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(0)  # indeterminate
        self._color_sep_progress.setVisible(True)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()  # flush so the user sees the busy state

        try:
            if method == "K-Means":
                from plottter.color import kmeans_separate
                # K-Means requires an RGB image; apply only spatial transforms
                # (crop/resize) to the raw image, not grayscale conversion or
                # threshold — those would destroy the color information.
                spatial_params = {
                    k: v for k, v in params.items()
                    if k in ("crop_width", "crop_height")
                }
                raw_rgb = preprocess(source, spatial_params)
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                results = kmeans_separate(raw_rgb, num_colors=num)
                layer_names = [f"Cluster {i + 1}" for i in range(len(results))]
            elif method == "Luminance":
                from plottter.color import luminance_separate
                results = luminance_separate(preprocessed, num_bands=num)
                band_names = ["Shadows", "Midtones", "Highlights", "Highlights 2", "Highlights 3"]
                layer_names = [band_names[i] if i < len(band_names) else f"Band {i + 1}" for i in range(len(results))]
            elif method == "RGB":
                from plottter.color import rgb_separate
                # RGB/CMYK separation requires an RGB image, not the
                # grayscale-preprocessed one.  Use source (with BG removal applied).
                raw_rgb = source
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                results = rgb_separate(raw_rgb)
                layer_names = ["Red Channel", "Green Channel", "Blue Channel"]
                channel_names = ["Red", "Green", "Blue"]
                filtered = []
                filtered_names = []
                for i, (mask, color) in enumerate(results):
                    ch = channel_names[i]
                    if ch not in self._channel_checks or self._channel_checks[ch].isChecked():
                        filtered.append((mask, color))
                        filtered_names.append(layer_names[i])
                results = filtered
                layer_names = filtered_names
            elif method == "CMYK":
                from plottter.color import cmyk_separate
                # CMYK separation requires an RGB image.
                raw_rgb = source
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                # Stash the RGB input so changing K Amount can refresh the
                # cached masks in-place without forcing a full re-separate.
                self._cmyk_raw_rgb = raw_rgb
                self._last_sep_method = "CMYK"
                results = cmyk_separate(
                    raw_rgb,
                    k_amount=float(self._cmyk_k_amount_spin.value()),
                )
                layer_names = ["Cyan Channel", "Magenta Channel", "Yellow Channel", "Key (Black) Channel"]
                channel_names_list = ["Cyan", "Magenta", "Yellow", "Key (Black)"]
                filtered = []
                filtered_names = []
                for i, (mask, color) in enumerate(results):
                    ch = channel_names_list[i]
                    if ch not in self._channel_checks or self._channel_checks[ch].isChecked():
                        filtered.append((mask, color))
                        filtered_names.append(layer_names[i])
                results = filtered
                layer_names = filtered_names
            elif method == "Custom Palette":
                from plottter.color import palette_separate
                raw_rgb = source
                if raw_rgb.ndim == 2:
                    raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
                elif raw_rgb.ndim == 3 and raw_rgb.shape[2] == 4:
                    raw_rgb = raw_rgb[:, :, :3]
                palette = self._palette_picker_combo.currentData()
                if palette is None:
                    QMessageBox.warning(self, "No Palette", "Please select a palette.")
                    return
                results = palette_separate(
                    raw_rgb,
                    palette,
                    dither=self._palette_dither_combo.currentText().lower(),
                )
                layer_names = [f"Pen: {color}" for _, color in results]
            else:
                return
        except Exception as exc:
            QMessageBox.critical(self, "Separation Error", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._color_sep_progress.setMaximum(100)
            self._color_sep_progress.setVisible(False)
            self._separate_btn.setEnabled(True)

        self._apply_separation_results(results, layer_names, method, preprocessed)

    def _on_ai_segment_finished(
        self, results: list, method: str
    ) -> None:
        """Called on the main thread when the AI segmentation worker succeeds."""
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)

        layer_names = [f"AI Segment {i + 1}" for i in range(len(results))]
        preprocessed = self._ai_sep_preprocessed
        self._ai_sep_preprocessed = None
        self._apply_separation_results(results, layer_names, method, preprocessed)

    def _on_ai_segment_error(self, msg: str) -> None:
        """Called on the main thread when the AI segmentation worker fails."""
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)
        self._ai_sep_preprocessed = None
        QMessageBox.critical(self, "AI Segmentation Error", msg)

    def _apply_separation_results(
        self,
        results: list,
        layer_names: list,
        method: str,
        preprocessed: "np.ndarray",
    ) -> None:
        """Create layers from separation results (called from both sync and async paths)."""
        # Remove previous separation layers before creating new ones
        self._controller.undo_stack.beginMacro("Separate Into Layers")
        for old_lid in list(self._separated_layer_ids):
            self._controller.remove_layer(old_lid)
            self._layer_masks.pop(old_lid, None)
        self._separated_layer_ids.clear()

        from plottter.models import Layer
        for (mask, hex_color), lname in zip(results, layer_names):
            display_name = f"{lname} — {hex_color}"
            layer = Layer(
                name=display_name,
                color=hex_color,
                generator_info={
                    "type": "color_separation",
                    "method": method,
                },
            )
            added = self._controller.add_layer(layer)
            self._separated_layer_ids.append(added.id)
            self._layer_masks[added.id] = (mask, preprocessed)
        self._controller.undo_stack.endMacro()

        self._gen_lines_btn.setEnabled(len(self._separated_layer_ids) > 0)
        self._gen_lines_selected_btn.setEnabled(len(self._separated_layer_ids) > 0)
        QMessageBox.information(
            self,
            "Color Separation",
            f"Created {len(self._separated_layer_ids)} layer(s) from color separation.",
        )

    def _on_generate_lines(self) -> None:
        """Generate line art for each separated layer using the selected algorithm."""
        if not self._separated_layer_ids:
            return

        idx = self._color_sep_gen_combo.currentIndex()
        if idx < 0:
            return
        gen_cls = self._color_sep_gen_combo.itemData(idx)
        if gen_cls is None:
            return

        canvas = self._controller.current_project.canvas

        # Gather layers with masks
        layers_to_process: list[tuple[str, object, object]] = []
        for lid in self._separated_layer_ids:
            if lid not in self._layer_masks:
                continue
            mask, src_img = self._layer_masks[lid]
            layers_to_process.append((lid, mask, src_img))

        if not layers_to_process:
            return

        from plottter.gui.generator_worker import GeneratorWorker

        self._gen_lines_btn.setEnabled(False)
        self._gen_lines_selected_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(len(layers_to_process))
        self._color_sep_progress.setValue(0)
        self._color_sep_progress.setVisible(True)

        self._lines_queue = list(layers_to_process)
        self._lines_done = 0
        self._lines_canvas = canvas
        self._lines_gen_cls = gen_cls
        self._lines_worker: object = None
        self._controller.undo_stack.beginMacro("Generate Lines")
        self._process_next_lines_layer()

    def _on_generate_lines_selected(self) -> None:
        """Generate line art for only the currently selected layer."""
        if not self._separated_layer_ids:
            return

        # Find which separated layer is currently active
        active_id = self._controller.active_layer_id
        if active_id not in self._separated_layer_ids:
            QMessageBox.warning(
                self,
                "No Separated Layer Selected",
                "Please select one of the separated layers in the layer panel.",
            )
            return

        if active_id not in self._layer_masks:
            return

        idx = self._color_sep_gen_combo.currentIndex()
        if idx < 0:
            return
        gen_cls = self._color_sep_gen_combo.itemData(idx)
        if gen_cls is None:
            return

        canvas = self._controller.current_project.canvas
        mask, src_img = self._layer_masks[active_id]

        self._gen_lines_btn.setEnabled(False)
        self._gen_lines_selected_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(1)
        self._color_sep_progress.setValue(0)
        self._color_sep_progress.setVisible(True)

        self._lines_queue = [(active_id, mask, src_img)]
        self._lines_done = 0
        self._lines_canvas = canvas
        self._lines_gen_cls = gen_cls
        self._lines_worker = None
        self._controller.undo_stack.beginMacro("Generate Lines (Selected)")
        self._process_next_lines_layer()

    def _process_next_lines_layer(self) -> None:
        if not self._lines_queue:
            self._color_sep_progress.setVisible(False)
            self._gen_lines_btn.setEnabled(True)
            self._gen_lines_selected_btn.setEnabled(True)
            self._controller.undo_stack.endMacro()
            return

        layer_id, mask, src_img = self._lines_queue.pop(0)
        import numpy as np

        masked_gray = _separation_mask_to_luminance(mask, src_img)

        gen = self._lines_gen_cls()

        # Check if a preset is selected in the color sep preset combo
        preset_params = self._color_sep_preset_combo.currentData()
        if preset_params is not None:
            # Use preset params as base (copy to avoid mutation)
            gen_params: dict = dict(preset_params)
        else:
            # Default: build params from generator defaults
            gen_params = {}
            for p in gen.get_parameters():
                if hasattr(p, "default"):
                    gen_params[p.name] = p.default

        # Always set _source_image and image placement params regardless of preset
        gen_params["_source_image"] = masked_gray
        gen_params["image_fit_mode"] = self._image_fit_mode()
        fit_mode = gen_params["image_fit_mode"]
        if fit_mode == "custom":
            gen_params["image_width_mm"] = self._image_width_spin.value()
            gen_params["image_height_mm"] = self._image_height_spin.value()
        if fit_mode != "fill":
            gen_params["image_offset_x_mm"] = self._image_offset_x_spin.value()
            gen_params["image_offset_y_mm"] = self._image_offset_y_spin.value()

        from plottter.gui.generator_worker import GeneratorWorker
        worker = GeneratorWorker(gen, gen_params, self._lines_canvas)

        def on_finished(paths, lid=layer_id):
            self._controller.set_layer_paths(lid, paths, "Generate Lines")
            self._lines_done += 1
            self._color_sep_progress.setValue(self._lines_done)
            self._process_next_lines_layer()

        def on_error(msg):
            QMessageBox.warning(self, "Generate Lines Error", msg)
            self._lines_done += 1
            self._color_sep_progress.setValue(self._lines_done)
            self._process_next_lines_layer()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        self._lines_worker = worker
        worker.start()
