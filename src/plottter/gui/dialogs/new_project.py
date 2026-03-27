"""NewProjectDialog — paper size and margin selection for new projects."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QRadioButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from plottter.models.canvas import Canvas, PAPER_PRESETS


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
        presets = list(PAPER_PRESETS.keys()) + ["Custom"]
        self._preset_combo.addItems(presets)
        self._preset_combo.setCurrentText("A4")
        preset_form.addRow("Preset:", self._preset_combo)

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

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire signals
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
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

        if not is_custom and preset_name in PAPER_PRESETS:
            w, h = PAPER_PRESETS[preset_name]
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
        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._width_spin.setValue(self._width_spin.value() * factor)
        self._height_spin.setValue(self._height_spin.value() * factor)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)
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
