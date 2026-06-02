"""NewProjectDialog — paper size and margin selection for new projects."""

from __future__ import annotations

import json

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from plottter.models.canvas import Canvas, PAPER_PRESETS


def _settings() -> QSettings:
    return QSettings("Plottter", "Plottter")


def load_default_canvas() -> Canvas:
    """Return the user's saved default canvas, or A4 portrait if none saved."""
    s = _settings()
    width = s.value("canvas/default_width_mm", type=float)
    height = s.value("canvas/default_height_mm", type=float)
    if not width or not height:
        return Canvas.from_preset("A4", margin=10.0)
    margin = s.value("canvas/default_margin_mm", 10.0, type=float)
    preset = str(s.value("canvas/default_preset", "Custom"))
    return Canvas(
        width_mm=float(width),
        height_mm=float(height),
        margin_mm=float(margin),
        paper_preset=preset,
    )


def save_default_canvas(canvas: Canvas) -> None:
    """Persist this canvas as the default for future new projects."""
    s = _settings()
    s.setValue("canvas/default_preset", canvas.paper_preset)
    s.setValue("canvas/default_width_mm", float(canvas.width_mm))
    s.setValue("canvas/default_height_mm", float(canvas.height_mm))
    s.setValue("canvas/default_margin_mm", float(canvas.margin_mm))


def load_user_presets() -> dict[str, tuple[float, float]]:
    """Return user-saved paper-size presets keyed by display name."""
    raw = _settings().value("canvas/user_presets", "")
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except (ValueError, TypeError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for name, dims in data.items():
        try:
            w, h = float(dims[0]), float(dims[1])
        except (TypeError, ValueError, IndexError):
            continue
        if w > 0 and h > 0:
            out[str(name)] = (w, h)
    return out


def save_user_presets(presets: dict[str, tuple[float, float]]) -> None:
    """Persist the user-saved paper-size presets."""
    payload = {name: [float(w), float(h)] for name, (w, h) in presets.items()}
    _settings().setValue("canvas/user_presets", json.dumps(payload))


class NewProjectDialog(QDialog):
    """Dialog for creating a new project with paper size and margin settings."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_canvas: Canvas | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(320)
        self._unit = "mm"
        self._setup_ui()
        if initial_canvas is not None:
            self._apply_canvas(initial_canvas)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Paper preset
        preset_group = QGroupBox("Paper Size")
        preset_form = QFormLayout(preset_group)

        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(220)
        preset_form.addRow("Preset:", self._preset_combo)

        self._save_preset_btn = QPushButton("Save…")
        self._save_preset_btn.setToolTip(
            "Save the current width and height as a named preset."
        )
        self._delete_preset_btn = QPushButton("Delete")
        self._delete_preset_btn.setToolTip(
            "Delete the currently selected user preset. Built-in presets cannot be deleted."
        )
        preset_btns_widget = QWidget()
        preset_btns_layout = QHBoxLayout(preset_btns_widget)
        preset_btns_layout.setContentsMargins(0, 0, 0, 0)
        preset_btns_layout.addWidget(self._save_preset_btn)
        preset_btns_layout.addWidget(self._delete_preset_btn)
        preset_btns_layout.addStretch()
        preset_form.addRow("", preset_btns_widget)
        self._refresh_preset_combo(select="A4")

        # Orientation toggle (placed between preset and size spinboxes)
        orientation_widget = QWidget()
        orientation_layout = QHBoxLayout(orientation_widget)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        self._portrait_radio = QRadioButton("Portrait")
        self._landscape_radio = QRadioButton("Landscape")
        self._portrait_radio.setChecked(True)
        self._orientation_group = QButtonGroup(self)
        self._orientation_group.addButton(self._portrait_radio)
        self._orientation_group.addButton(self._landscape_radio)
        orientation_layout.addWidget(self._portrait_radio)
        orientation_layout.addWidget(self._landscape_radio)
        orientation_layout.addStretch()
        preset_form.addRow("Orientation:", orientation_widget)

        # Size spinboxes (always visible)
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1.0, 10000.0)
        self._width_spin.setDecimals(1)
        self._width_spin.setSuffix(" mm")

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(1.0, 10000.0)
        self._height_spin.setDecimals(1)
        self._height_spin.setSuffix(" mm")

        preset_form.addRow("Width:", self._width_spin)
        preset_form.addRow("Height:", self._height_spin)

        # Margin
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.0, 100.0)
        self._margin_spin.setDecimals(1)
        self._margin_spin.setValue(10.0)
        self._margin_spin.setSuffix(" mm")
        preset_form.addRow("Margin:", self._margin_spin)

        layout.addWidget(preset_group)

        # Unit toggle
        unit_group = QGroupBox("Units")
        unit_layout = QHBoxLayout(unit_group)
        self._mm_radio = QRadioButton("mm")
        self._in_radio = QRadioButton("inches")
        self._mm_radio.setChecked(True)
        unit_layout.addWidget(self._mm_radio)
        unit_layout.addWidget(self._in_radio)
        layout.addWidget(unit_group)

        # Set-as-default checkbox
        self._set_default_check = QCheckBox("Set as default for new projects")
        self._set_default_check.setToolTip(
            "Save the current paper size, orientation, and margin as the "
            "default canvas used when creating new projects."
        )
        layout.addWidget(self._set_default_check)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire signals
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        self._delete_preset_btn.clicked.connect(self._on_delete_preset)
        self._mm_radio.toggled.connect(self._on_unit_changed)
        self._portrait_radio.toggled.connect(self._on_orientation_toggled)
        self._width_spin.valueChanged.connect(self._on_size_changed)
        self._height_spin.valueChanged.connect(self._on_size_changed)

        # Initialize state
        self._on_preset_changed("A4")

    def _on_preset_changed(self, preset_name: str) -> None:
        is_custom = preset_name == "Custom"
        self._width_spin.setReadOnly(not is_custom)
        self._height_spin.setReadOnly(not is_custom)
        self._delete_preset_btn.setEnabled(preset_name in load_user_presets())

        dims: tuple[float, float] | None = None
        if preset_name in PAPER_PRESETS:
            dims = PAPER_PRESETS[preset_name]
        elif not is_custom:
            user = load_user_presets()
            if preset_name in user:
                dims = user[preset_name]
        if dims is None:
            return

        w, h = dims
        if self._landscape_radio.isChecked():
            w, h = h, w
        if self._unit == "inches":
            w /= 25.4
            h /= 25.4
        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._width_spin.setValue(w)
        self._height_spin.setValue(h)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)

    def _refresh_preset_combo(self, select: str | None = None) -> None:
        """Rebuild the preset combo: built-ins, then user presets, then Custom."""
        prev = select if select is not None else self._preset_combo.currentText()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for name in PAPER_PRESETS:
            self._preset_combo.addItem(name)
        user = load_user_presets()
        if user:
            self._preset_combo.insertSeparator(self._preset_combo.count())
            for name in user:
                self._preset_combo.addItem(name)
        self._preset_combo.insertSeparator(self._preset_combo.count())
        self._preset_combo.addItem("Custom")
        idx = self._preset_combo.findText(prev)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)
        self._delete_preset_btn.setEnabled(
            self._preset_combo.currentText() in load_user_presets()
        )

    def _current_dims_mm(self) -> tuple[float, float]:
        w = self._width_spin.value()
        h = self._height_spin.value()
        if self._unit == "inches":
            w *= 25.4
            h *= 25.4
        # Store the portrait-canonical form (width ≤ height) so the orientation
        # toggle behaves the same way as built-ins on next load.
        if w > h:
            w, h = h, w
        return w, h

    def _on_save_preset(self) -> None:
        existing = load_user_presets()
        name, ok = QInputDialog.getText(
            self, "Save Paper Preset", "Preset name:"
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Save Paper Preset", "Name cannot be empty.")
            return
        if name in PAPER_PRESETS or name == "Custom":
            QMessageBox.warning(
                self,
                "Save Paper Preset",
                f"{name!r} is reserved by a built-in preset. Choose a different name.",
            )
            return
        if name in existing:
            reply = QMessageBox.question(
                self,
                "Overwrite Preset",
                f"A preset named {name!r} already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        existing[name] = self._current_dims_mm()
        save_user_presets(existing)
        self._refresh_preset_combo(select=name)

    def _on_delete_preset(self) -> None:
        name = self._preset_combo.currentText()
        existing = load_user_presets()
        if name not in existing:
            return
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete the user preset {name!r}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        del existing[name]
        save_user_presets(existing)
        self._refresh_preset_combo(select="Custom")

    def _on_orientation_toggled(self, portrait_checked: bool) -> None:
        """Swap spinbox values when orientation is toggled."""
        w = self._width_spin.value()
        h = self._height_spin.value()
        landscape = not portrait_checked
        should_swap = (landscape and w < h) or (not landscape and w > h)
        if should_swap:
            self._width_spin.blockSignals(True)
            self._height_spin.blockSignals(True)
            self._width_spin.setValue(h)
            self._height_spin.setValue(w)
            self._width_spin.blockSignals(False)
            self._height_spin.blockSignals(False)

    def _on_size_changed(self) -> None:
        """Auto-detect orientation from current spinbox values."""
        w = self._width_spin.value()
        h = self._height_spin.value()
        self._portrait_radio.blockSignals(True)
        self._landscape_radio.blockSignals(True)
        if w > h:
            self._landscape_radio.setChecked(True)
        elif h > w:
            self._portrait_radio.setChecked(True)
        self._portrait_radio.blockSignals(False)
        self._landscape_radio.blockSignals(False)

    def _on_unit_changed(self, mm_checked: bool) -> None:
        new_unit = "mm" if mm_checked else "inches"
        if new_unit == self._unit:
            return
        factor = 25.4 if new_unit == "mm" else 1.0 / 25.4
        # Use 3 decimals in inches mode so mm↔inches round-trips preserve
        # enough precision (e.g. 210 mm → 8.268 in → 210.0 mm). Must be set
        # before setValue() or the new value gets rounded to the old precision.
        new_decimals = 1 if new_unit == "mm" else 3
        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._width_spin.setDecimals(new_decimals)
        self._height_spin.setDecimals(new_decimals)
        self._width_spin.setValue(self._width_spin.value() * factor)
        self._height_spin.setValue(self._height_spin.value() * factor)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)
        self._margin_spin.setDecimals(new_decimals)
        self._margin_spin.setValue(self._margin_spin.value() * factor)

        suffix = " mm" if new_unit == "mm" else " in"
        self._width_spin.setSuffix(suffix)
        self._height_spin.setSuffix(suffix)
        self._margin_spin.setSuffix(suffix)
        self._unit = new_unit

    def _apply_canvas(self, canvas: Canvas) -> None:
        """Pre-populate dialog fields from an existing canvas."""
        preset = canvas.paper_preset
        all_presets = [self._preset_combo.itemText(i) for i in range(self._preset_combo.count())]
        if preset not in all_presets:
            preset = "Custom"

        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText(preset)
        self._preset_combo.blockSignals(False)

        is_custom = preset == "Custom"
        self._width_spin.setReadOnly(not is_custom)
        self._height_spin.setReadOnly(not is_custom)

        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._width_spin.setValue(canvas.width_mm)
        self._height_spin.setValue(canvas.height_mm)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)

        self._margin_spin.setValue(canvas.margin_mm)

        self._portrait_radio.blockSignals(True)
        self._landscape_radio.blockSignals(True)
        if canvas.width_mm > canvas.height_mm:
            self._landscape_radio.setChecked(True)
        else:
            self._portrait_radio.setChecked(True)
        self._portrait_radio.blockSignals(False)
        self._landscape_radio.blockSignals(False)

    def should_save_as_default(self) -> bool:
        """Return whether the user ticked 'Set as default for new projects'."""
        return self._set_default_check.isChecked()

    def get_canvas(self) -> Canvas:
        """Return a Canvas from the dialog's current settings."""
        margin_mm = self._margin_spin.value()
        if self._unit == "inches":
            margin_mm *= 25.4

        width_mm = self._width_spin.value()
        height_mm = self._height_spin.value()
        if self._unit == "inches":
            width_mm *= 25.4
            height_mm *= 25.4

        preset = self._preset_combo.currentText()
        paper_preset = preset if preset != "Custom" else "Custom"
        return Canvas(width_mm=width_mm, height_mm=height_mm, margin_mm=margin_mm, paper_preset=paper_preset)
