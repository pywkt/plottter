"""_PresetsMixin — preset combo helpers."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
)


class _PresetsMixin:
    """Mixin for preset management helpers."""

    _SAVE_PRESET_ACTION = "Save Current as Preset\u2026"  # "Save Current as Preset…"

    def _rebuild_preset_combo(self) -> None:
        """Repopulate the preset combo from the current generator's presets."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem("Custom")
        if self._generator is not None:
            for preset in self._generator.get_presets():
                idx = self._preset_combo.count()
                self._preset_combo.addItem(preset.name)
                self._preset_combo.setItemData(idx, "builtin", Qt.ItemDataRole.UserRole)
                if preset.description:
                    self._preset_combo.setItemData(idx, preset.description, Qt.ItemDataRole.ToolTipRole)

            # Load and cache user presets for this generator.
            try:
                from plottter.presets.user_presets import load_user_presets
                self._user_presets = load_user_presets(self._generator.name)
            except Exception:
                self._user_presets = []

            if self._user_presets:
                self._preset_combo.insertSeparator(self._preset_combo.count())
                self._preset_combo.addItem("— User Presets —")
                # Make the section header non-selectable.
                header_idx = self._preset_combo.count() - 1
                model = self._preset_combo.model()
                if model is not None:
                    header_item = model.item(header_idx)
                    if header_item is not None:
                        from PyQt6.QtCore import Qt as _Qt
                        header_item.setFlags(
                            header_item.flags()
                            & ~_Qt.ItemFlag.ItemIsEnabled
                            & ~_Qt.ItemFlag.ItemIsSelectable
                        )
                for preset in self._user_presets:
                    idx = self._preset_combo.count()
                    self._preset_combo.addItem(preset.name)
                    # Tag item as a user preset for future context-menu support.
                    self._preset_combo.setItemData(idx, "user", Qt.ItemDataRole.UserRole)

            self._preset_combo.insertSeparator(self._preset_combo.count())
            self._preset_combo.addItem(self._SAVE_PRESET_ACTION)
        else:
            self._user_presets = []
        self._preset_combo.blockSignals(False)

    def _gather_current_params(self) -> dict[str, Any]:
        """Collect serialisable parameter values from the current widgets."""
        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPGather
        except ImportError:
            _FPGather = None  # type: ignore[assignment,misc]

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
            elif _FPGather is not None and isinstance(widget, _FPGather):
                result[name] = widget.font_path()
        return result

    def _save_current_as_preset(self) -> None:
        """Prompt the user for a name and persist the current params as a user preset."""
        if self._generator is None:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        name, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Enter a name for this preset:",
        )
        if not ok or not name.strip():
            # User cancelled — restore combo to "Custom"
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        name = name.strip()
        params = self._gather_current_params()
        # Capture dynamic overrides so the preset can restore them (spec §5.2).
        if self._dynamic_overrides:
            params["_dynamic_overrides"] = dict(self._dynamic_overrides)

        try:
            from plottter.generators.base import Preset
            from plottter.presets.user_presets import save_user_preset
            save_user_preset(self._generator.name, Preset(name=name, params=params))
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save preset: {exc}")
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            return

        # Refresh the combo to include the newly saved user preset.
        self._rebuild_preset_combo()
        # Select the newly saved preset name if it appears in the combo, else "Custom"
        self._preset_combo.blockSignals(True)
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _on_preset_context_menu(self, pos) -> None:
        """Show a context menu with rename/delete for user preset items."""
        idx = self._preset_combo.view().indexAt(pos).row()
        if idx < 0:
            return
        item_data = self._preset_combo.itemData(idx, Qt.ItemDataRole.UserRole)
        if item_data != "user":
            return
        preset_name = self._preset_combo.itemText(idx)
        if not preset_name:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename Preset")
        delete_action = menu.addAction("Delete Preset")
        chosen = menu.exec(self._preset_combo.view().mapToGlobal(pos))
        if chosen is rename_action:
            self._rename_user_preset_action(preset_name)
        elif chosen is delete_action:
            self._delete_user_preset_action(preset_name)

    def _delete_user_preset_action(self, preset_name: str) -> None:
        """Ask for confirmation and delete a user preset."""
        if self._generator is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from plottter.presets.user_presets import delete_user_preset
            delete_user_preset(self._generator.name, preset_name)
        except Exception as exc:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete preset: {exc}")
            return
        self._rebuild_preset_combo()
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _rename_user_preset_action(self, old_name: str) -> None:
        """Prompt the user for a new name and rename a user preset."""
        if self._generator is None:
            return
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Preset",
            "Enter a new name for this preset:",
            text=old_name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == old_name:
            return
        try:
            from plottter.presets.user_presets import rename_user_preset
            rename_user_preset(self._generator.name, old_name, new_name)
        except Exception as exc:
            QMessageBox.warning(self, "Rename Failed", f"Could not rename preset: {exc}")
            return
        self._rebuild_preset_combo()
        self._preset_combo.blockSignals(True)
        idx = self._preset_combo.findText(new_name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)

    def _on_preset_changed(self, preset_name: str) -> None:
        if preset_name == self._SAVE_PRESET_ACTION:
            self._save_current_as_preset()
            return
        if preset_name == "Custom":
            # Reset all expression fields to editable, clear sentinel values, and
            # restore default text so users don't see the "(ODE — not editable)" placeholder.
            if self._generator is not None:
                try:
                    from plottter.generators.base import ExpressionParam
                    param_defaults = {
                        p.name: p.default
                        for p in self._generator.get_parameters()
                        if isinstance(p, ExpressionParam)
                    }
                except ImportError:
                    param_defaults = {}
            else:
                param_defaults = {}
            for name, widget in self._param_widgets.items():
                if isinstance(widget, QLineEdit):
                    if widget.property("_sentinel") is not None:
                        widget.setReadOnly(False)
                        widget.setProperty("_sentinel", None)
                        widget.setPlaceholderText("")
                        if name in param_defaults:
                            widget.setText(str(param_defaults[name]))
            return
        if self._generator is None:
            return
        for preset in self._generator.get_presets():
            if preset.name == preset_name:
                self._apply_preset_params(preset.params)
                return
        # Check user presets if no built-in preset matched.
        for preset in self._user_presets:
            if preset.name == preset_name:
                self._apply_preset_params(preset.params)
                return

    def _apply_preset_params(self, params: dict[str, Any]) -> None:
        # Extract dynamic overrides before applying static params.  Using a
        # copy avoids mutating the caller's dict.
        params = dict(params)
        saved_overrides: dict[str, Any] | None = params.pop("_dynamic_overrides", None)

        # Reset all expression fields to editable before applying preset values.
        for widget in self._param_widgets.values():
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(False)

        try:
            from plottter.gui.widgets.font_picker import FontPicker as _FPPreset
        except ImportError:
            _FPPreset = None  # type: ignore[assignment,misc]

        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                str_val = str(value)
                # Sentinel values (e.g. "__lorenz__") indicate ODE-driven params
                # that are not user-editable. Show a human-readable placeholder.
                if str_val.startswith("__") and str_val.endswith("__"):
                    widget.setText("(ODE — not editable)")
                    widget.setReadOnly(True)
                    widget.setProperty("_sentinel", str_val)
                else:
                    widget.setText(str_val)
                    widget.setProperty("_sentinel", None)
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif _FPPreset is not None and isinstance(widget, _FPPreset):
                widget.set_font_path(str(value))
        self._update_param_visibility()

        # Apply dynamic overrides (spec §5.2).  Preset application is a single
        # discrete event: bypass the 500 ms debounce and rebuild synchronously,
        # then write saved override values into the new dynamic widgets.
        if saved_overrides is not None:
            # Cancel any pending debounce-triggered rebuild.
            try:
                self._dynamic_rebuild_timer.stop()
            except AttributeError:
                pass
            # Clear stale overrides so the rebuild starts from a clean slate.
            self._dynamic_overrides.clear()
            self._rebuild_dynamic_params()
            # Write saved override values into the freshly built widgets.
            self._dynamic_overrides.update(saved_overrides)
            for _ov_name, _ov_value in saved_overrides.items():
                _ov_widget = self._dynamic_param_widgets.get(_ov_name)
                _ov_param = next(
                    (p for p in self._dynamic_param_specs if p.name == _ov_name),
                    None,
                )
                if _ov_widget is not None and _ov_param is not None:
                    self._set_dynamic_widget_value(_ov_widget, _ov_param, _ov_value)

    def apply_generator_preset(self, gen_cls: type, preset_name: str) -> None:
        """Switch to the given generator and apply the named preset.

        Can be called externally (e.g. from PresetGalleryDialog) while the
        settings panel is in Math Art mode.
        """
        # Select the generator in the type combo if it's listed there
        for i in range(self._generator_type_combo.count()):
            if self._generator_type_combo.itemData(i) is gen_cls:
                self._generator_type_combo.setCurrentIndex(i)
                break
        else:
            # Generator not in combo (e.g. mode mismatch) — force-set it
            self.set_generator(gen_cls())
        # Apply the named preset
        idx = self._preset_combo.findText(preset_name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
