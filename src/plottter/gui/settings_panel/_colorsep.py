"""_ColorSepMixin — color separation methods."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QMessageBox,
    QSpinBox,
)

from .workers import _AiBgWorker, _AiSegmentWorker, _SeparationWorker

# Resolution cap (in pixels) applied to the image fed to color separation when
# "Downsample large images" is enabled.  ~2 MP keeps masks sharp enough for any
# plot density while cutting separation + line-generation time on big photos.
_SEPARATION_MAX_PIXELS = 2_000_000


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


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Normalise an image to 3-channel RGB (grayscale -> stacked, RGBA -> RGB)."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


def _filter_channels(
    results: list,
    layer_names: list,
    channel_names: list,
    enabled: dict,
) -> tuple[list, list]:
    """Keep only the (mask, color) entries whose channel checkbox is enabled.

    ``enabled`` maps channel name -> bool (captured from the checkboxes on the
    main thread).  A channel missing from the map is treated as enabled, which
    mirrors the original ``ch not in self._channel_checks`` fallback.
    """
    out: list = []
    out_names: list = []
    for i, (mask, color) in enumerate(results):
        ch = channel_names[i]
        if ch not in enabled or enabled[ch]:
            out.append((mask, color))
            out_names.append(layer_names[i])
    return out, out_names


def _compute_separation(
    method: str,
    *,
    source: np.ndarray,
    params: dict,
    max_px: int,
    num: int,
    thresholds: "list[float] | None",
    enabled_channels: dict,
    k_amount: float,
    dither: str,
    palette,
) -> dict:
    """Run one color-separation method and return a result payload.

    Pure NumPy work with no Qt access, so it is safe to run in a worker thread
    (see :meth:`_ColorSepMixin._on_separate`) and is directly unit-testable.
    All widget-derived values are passed in by the caller.  Returns a dict with
    ``results`` (list of (mask, hex)), ``layer_names``, ``preprocessed`` (the
    image cached as each layer's src_img), and ``cmyk_raw_rgb`` (the RGB input
    for CMYK so the panel can refresh K Amount later, else None).
    """
    from plottter.io.image_import import downscale_to_max_pixels, preprocess

    payload: dict = {"cmyk_raw_rgb": None}
    # The fully preprocessed image is cached as every layer's src_img and is the
    # input for the luminance path.  Computing it here keeps this heavy step
    # (preprocess + LANCZOS downscale) off the GUI thread.
    preprocessed = downscale_to_max_pixels(preprocess(source, params), max_px)
    payload["preprocessed"] = preprocessed
    if method == "K-Means":
        from plottter.color import kmeans_separate
        # K-Means requires an RGB image; apply only spatial transforms
        # (crop/resize), not grayscale conversion or threshold — those would
        # destroy the color information.
        spatial_params = {
            k: v for k, v in params.items() if k in ("crop_width", "crop_height")
        }
        raw_rgb = _ensure_rgb(
            downscale_to_max_pixels(preprocess(source, spatial_params), max_px)
        )
        results = kmeans_separate(raw_rgb, num_colors=num)
        layer_names = [f"Cluster {i + 1}" for i in range(len(results))]
    elif method == "Luminance":
        from plottter.color import luminance_separate
        results = luminance_separate(preprocessed, num_bands=num, thresholds=thresholds)
        band_names = ["Shadows", "Midtones", "Highlights", "Highlights 2", "Highlights 3"]
        layer_names = [
            band_names[i] if i < len(band_names) else f"Band {i + 1}"
            for i in range(len(results))
        ]
    elif method == "RGB":
        from plottter.color import rgb_separate
        raw_rgb = _ensure_rgb(downscale_to_max_pixels(source, max_px))
        results = rgb_separate(raw_rgb)
        results, layer_names = _filter_channels(
            results,
            ["Red Channel", "Green Channel", "Blue Channel"],
            ["Red", "Green", "Blue"],
            enabled_channels,
        )
    elif method == "CMYK":
        from plottter.color import cmyk_separate
        raw_rgb = _ensure_rgb(downscale_to_max_pixels(source, max_px))
        payload["cmyk_raw_rgb"] = raw_rgb
        results = cmyk_separate(raw_rgb, k_amount=k_amount)
        results, layer_names = _filter_channels(
            results,
            ["Cyan Channel", "Magenta Channel", "Yellow Channel", "Key (Black) Channel"],
            ["Cyan", "Magenta", "Yellow", "Key (Black)"],
            enabled_channels,
        )
    elif method == "Custom Palette":
        from plottter.color import palette_separate
        raw_rgb = _ensure_rgb(downscale_to_max_pixels(source, max_px))
        results = palette_separate(raw_rgb, palette, dither=dither)
        layer_names = [f"Pen: {color}" for _, color in results]
    else:
        results, layer_names = [], []
    payload["results"] = results
    payload["layer_names"] = layer_names
    return payload


def _is_near_white_hex(hex_color: str, threshold: int = 240) -> bool:
    """Return True if a ``#RRGGBB`` colour has all channels >= *threshold*.

    Used to detect "background" layers that arise when AI BG removal
    composites the removed region onto pure white and a partitioning
    separator (K-Means / Luminance / Custom Palette) then assigns those
    pixels to a near-white layer.
    """
    if not isinstance(hex_color, str) or len(hex_color) != 7 or hex_color[0] != "#":
        return False
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return False
    return r >= threshold and g >= threshold and b >= threshold


def filter_near_white_layers(
    results: list,
    layer_names: list,
    threshold: int = 240,
) -> tuple[list, list]:
    """Drop entries whose hex colour is near white (per :func:`_is_near_white_hex`).

    Operates on the ``(mask, hex_color)`` tuples and ``layer_names`` list in
    lockstep — both inputs are sliced identically.  Returns new lists; the
    inputs are not mutated.
    """
    filtered_results: list = []
    filtered_names: list = []
    for (mask, hex_color), lname in zip(results, layer_names):
        if _is_near_white_hex(hex_color, threshold=threshold):
            continue
        filtered_results.append((mask, hex_color))
        filtered_names.append(lname)
    return filtered_results, filtered_names


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
        # "Skip near-white layer" is only meaningful for partitioning separators
        # that may assign white pixels to their own layer (the AI-BG-on-white
        # region).  RGB / CMYK / AI already emit zero ink for white pixels.
        if hasattr(self, "_skip_white_layer_check"):
            self._skip_white_layer_check.setVisible(is_kmeans or is_lum or is_palette)

        if is_palette:
            self._populate_palette_picker()

        # Luminance custom-threshold controls follow the method.
        if hasattr(self, "_lum_custom_check"):
            self._update_lum_threshold_visibility()

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

    def _on_skip_white_layer_toggled(self, checked: bool) -> None:
        """Persist the 'Skip near-white layer' checkbox state to QSettings."""
        from PyQt6.QtCore import QSettings
        QSettings("Plottter", "Plottter").setValue(
            "colorsep/skip_white_layer", "true" if checked else "false"
        )

    def _on_downsample_toggled(self, checked: bool) -> None:
        """Persist the 'Downsample large images' checkbox state to QSettings."""
        from PyQt6.QtCore import QSettings
        QSettings("Plottter", "Plottter").setValue(
            "colorsep/downsample", "true" if checked else "false"
        )

    def _separation_max_pixels(self) -> int:
        """Pixel cap for separation inputs, or 0 (no cap) when downsampling off."""
        check = getattr(self, "_downsample_check", None)
        if check is not None and not check.isChecked():
            return 0
        return _SEPARATION_MAX_PIXELS

    # -- Luminance custom thresholds -----------------------------------------

    def _on_lum_custom_toggled(self, checked: bool) -> None:
        """Reveal/hide the manual band-boundary spinboxes (Luminance mode)."""
        from PyQt6.QtCore import QSettings
        QSettings("Plottter", "Plottter").setValue(
            "colorsep/lum_custom", "true" if checked else "false"
        )
        self._update_lum_threshold_visibility()

    def _on_lum_bands_changed(self, _value: int | None = None) -> None:
        """Rebuild the boundary spinboxes when the band count changes."""
        if self._color_sep_method_combo.currentText() != "Luminance":
            return
        if self._lum_custom_check.isChecked():
            self._rebuild_lum_thresholds()

    def _update_lum_threshold_visibility(self) -> None:
        """Show the custom-threshold controls only for Luminance mode."""
        is_lum = self._color_sep_method_combo.currentText() == "Luminance"
        self._lum_custom_check.setVisible(is_lum)
        show_spins = is_lum and self._lum_custom_check.isChecked()
        self._lum_threshold_widget.setVisible(show_spins)
        # Rebuild when the spinbox count is stale (band count changed while the
        # controls were hidden) so it always matches num_bands - 1.
        expected = self._color_sep_num_colors_spin.value() - 1
        if show_spins and len(self._lum_threshold_spins) != expected:
            self._rebuild_lum_thresholds()

    def _rebuild_lum_thresholds(self) -> None:
        """Recreate one spinbox per band boundary (num_bands - 1), seeded with
        the current evenly-spaced defaults so toggling custom on is a no-op
        until the user edits a value."""
        layout = self._lum_threshold_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._lum_threshold_spins = []

        num_bands = self._color_sep_num_colors_spin.value()
        step = 256.0 / num_bands
        for i in range(1, num_bands):
            spin = QSpinBox()
            spin.setRange(1, 254)
            spin.setValue(min(254, max(1, int(round(step * i)))))
            self._lum_threshold_layout.addRow(QLabel(f"Boundary {i}"), spin)
            self._lum_threshold_spins.append(spin)

    def _gather_lum_thresholds(self) -> list[float] | None:
        """Return ascending custom thresholds, or None for even spacing.

        Values are sorted so the boundaries are always valid for
        ``luminance_separate`` even if the user enters them out of order.
        """
        check = getattr(self, "_lum_custom_check", None)
        spins = getattr(self, "_lum_threshold_spins", None)
        if check is None or not check.isChecked() or not spins:
            return None
        # Defensive: if the count is somehow stale, fall back to even spacing
        # rather than handing luminance_separate a wrong-length list.
        if len(spins) != self._color_sep_num_colors_spin.value() - 1:
            return None
        return sorted(float(s.value()) for s in spins)

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
            from plottter.io.image_import import downscale_to_max_pixels, preprocess
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
            # Resolution cap for separation, applied when downsampling is on.
            # Deterministic on input dimensions, so a cluster mask and its
            # companion preprocessed image (same pre-cap size) stay aligned.
            max_px = self._separation_max_pixels()
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

            # Cap to match the (capped) preprocessed image so returned masks
            # stay aligned with it during mask->luminance conversion.
            source_img = downscale_to_max_pixels(source, max_px)
            if source_img.ndim == 2:
                source_img = np.stack([source_img] * 3, axis=-1)
            elif source_img.ndim == 3 and source_img.shape[2] == 4:
                source_img = source_img[:, :, :3]

            # Store preprocessed so the finished callback can use it for mask
            # association.  AI is network-bound, so this local preprocess on the
            # main thread is negligible.
            self._ai_sep_preprocessed = downscale_to_max_pixels(
                preprocess(source, params), max_px
            )

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
        # Palette) can take several seconds on large images.  Capture every
        # widget value here on the main thread, then do the NumPy work in a
        # worker (via the pure _compute_separation) so the GUI stays responsive.
        palette = self._palette_picker_combo.currentData()
        if method == "Custom Palette" and palette is None:
            QMessageBox.warning(self, "No Palette", "Please select a palette.")
            return
        thresholds = self._gather_lum_thresholds()
        enabled_channels = {
            name: cb.isChecked() for name, cb in self._channel_checks.items()
        }
        k_amount = float(self._cmyk_k_amount_spin.value())
        dither = self._palette_dither_combo.currentText().lower()

        def _compute() -> dict:
            return _compute_separation(
                method,
                source=source,
                params=params,
                max_px=max_px,
                num=num,
                thresholds=thresholds,
                enabled_channels=enabled_channels,
                k_amount=k_amount,
                dither=dither,
                palette=palette,
            )

        self._separate_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(0)  # indeterminate
        self._color_sep_progress.setVisible(True)

        worker = _SeparationWorker(_compute)
        worker.finished.connect(
            lambda payload: self._on_separation_finished(payload, method)
        )
        worker.error.connect(self._on_separation_error)
        self._separation_worker = worker
        worker.start()

    def _on_separation_finished(self, payload: dict, method: str) -> None:
        """Apply worker results on the main thread (creates layers, undo-safe)."""
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)
        if payload.get("cmyk_raw_rgb") is not None:
            self._cmyk_raw_rgb = payload["cmyk_raw_rgb"]
            self._last_sep_method = "CMYK"
        worker = self._separation_worker
        self._separation_worker = None
        if worker is not None:
            worker.wait()  # join before dropping the ref (avoid GC mid-run)
        self._apply_separation_results(
            payload["results"], payload["layer_names"], method, payload["preprocessed"]
        )

    def _on_separation_error(self, msg: str) -> None:
        self._separate_btn.setEnabled(True)
        self._color_sep_progress.setMaximum(100)
        self._color_sep_progress.setVisible(False)
        worker = self._separation_worker
        self._separation_worker = None
        if worker is not None:
            worker.wait()
        QMessageBox.critical(self, "Separation Error", str(msg))

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
        # "Skip near-white layer" filter — drops the layer(s) whose representative
        # colour is essentially white.  Only applied for partitioning separators
        # where a white layer is a real risk (K-Means / Luminance / Custom
        # Palette).  RGB / CMYK / AI Layer Separation are unaffected.
        if (
            getattr(self, "_skip_white_layer_check", None) is not None
            and self._skip_white_layer_check.isChecked()
            and method in ("K-Means", "Luminance", "Custom Palette")
        ):
            results, layer_names = filter_near_white_layers(results, layer_names)

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

        self._lines_canvas = canvas
        self._lines_gen_cls = gen_cls
        self._start_line_generation(layers_to_process, "Generate Lines")

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

        self._lines_canvas = canvas
        self._lines_gen_cls = gen_cls
        self._start_line_generation(
            [(active_id, mask, src_img)], "Generate Lines (Selected)"
        )

    def _line_worker_cap(self) -> int:
        """Max concurrent line-generation workers.

        Generators are CPU-bound, so cap at a few threads to gain parallelism
        on multi-core machines without oversubscribing.
        """
        import os

        return max(1, min(4, (os.cpu_count() or 1)))

    def _start_line_generation(self, queue: list, macro_name: str) -> None:
        """Begin generating lines for every layer in *queue*, running several
        workers concurrently to shrink wall-clock time."""
        self._lines_queue = list(queue)
        self._lines_total = len(self._lines_queue)
        self._lines_done = 0
        self._lines_active: dict = {}

        self._gen_lines_btn.setEnabled(False)
        self._gen_lines_selected_btn.setEnabled(False)
        self._color_sep_progress.setMaximum(self._lines_total)
        self._color_sep_progress.setValue(0)
        self._color_sep_progress.setVisible(True)

        self._controller.undo_stack.beginMacro(macro_name)
        self._pump_line_workers()

    def _pump_line_workers(self) -> None:
        """Launch queued layers until the concurrency cap is reached."""
        cap = self._line_worker_cap()
        active = getattr(self, "_lines_active", None)
        if active is None:
            active = self._lines_active = {}
        while self._lines_queue and len(active) < cap:
            self._process_next_lines_layer()
        # Empty queue with nothing running means there was no work — finalize.
        if not self._lines_queue and not active:
            self._finish_line_generation()

    def _process_next_lines_layer(self) -> None:
        """Launch a single line-generation worker for the next queued layer.

        Builds the generator params on the main thread (reading the preset and
        image-placement widgets) and starts a GeneratorWorker.  Completion is
        handled by :meth:`_on_line_worker_done`, which pumps the next layer.
        """
        if not self._lines_queue:
            return

        active = getattr(self, "_lines_active", None)
        if active is None:
            active = self._lines_active = {}

        layer_id, mask, src_img = self._lines_queue.pop(0)

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
        key = id(worker)

        def on_finished(paths, lid=layer_id, k=key):
            self._controller.set_layer_paths(lid, paths, "Generate Lines")
            self._on_line_worker_done(k)

        def on_error(msg, k=key):
            QMessageBox.warning(self, "Generate Lines Error", msg)
            self._on_line_worker_done(k)

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        active[key] = worker
        worker.start()

    def _on_line_worker_done(self, key: int) -> None:
        """One layer finished: update progress, launch the next, finalize at end."""
        active = getattr(self, "_lines_active", {}) or {}
        worker = active.pop(key, None)
        if worker is not None and hasattr(worker, "wait"):
            worker.wait()  # join before dropping the ref (avoid GC mid-run)
        self._lines_done += 1
        self._color_sep_progress.setValue(self._lines_done)
        if self._lines_queue:
            self._pump_line_workers()
        elif not active:
            self._finish_line_generation()

    def _finish_line_generation(self) -> None:
        """Re-enable controls and close the undo macro once all layers are done."""
        self._color_sep_progress.setVisible(False)
        self._gen_lines_btn.setEnabled(True)
        self._gen_lines_selected_btn.setEnabled(True)
        self._controller.undo_stack.endMacro()
