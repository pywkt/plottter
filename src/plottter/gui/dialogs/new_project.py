"""NewProjectDialog — paper size and margin selection for new projects."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from plottter.models.canvas import Canvas, PAPER_PRESETS


class NewProjectDialog(QDialog):
    """Dialog for creating a new project with paper size and margin settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(320)
        self._unit = "mm"
        self._setup_ui()

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

        # Custom size (hidden unless Custom selected)
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1.0, 10000.0)
        self._width_spin.setDecimals(1)
        self._width_spin.setSuffix(" mm")

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(1.0, 10000.0)
        self._height_spin.setDecimals(1)
        self._height_spin.setSuffix(" mm")

        self._width_label = QLabel("Width:")
        self._height_label = QLabel("Height:")

        preset_form.addRow(self._width_label, self._width_spin)
        preset_form.addRow(self._height_label, self._height_spin)

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

        # Initialize state
        self._on_preset_changed("A4")

    def _on_preset_changed(self, preset_name: str) -> None:
        is_custom = preset_name == "Custom"
        self._width_label.setVisible(is_custom)
        self._width_spin.setVisible(is_custom)
        self._height_label.setVisible(is_custom)
        self._height_spin.setVisible(is_custom)

        if not is_custom and preset_name in PAPER_PRESETS:
            w, h = PAPER_PRESETS[preset_name]
            if self._unit == "inches":
                w /= 25.4
                h /= 25.4
            self._width_spin.setValue(w)
            self._height_spin.setValue(h)

    def _on_unit_changed(self, mm_checked: bool) -> None:
        new_unit = "mm" if mm_checked else "inches"
        if new_unit == self._unit:
            return
        # Convert current values
        factor = 25.4 if new_unit == "mm" else 1.0 / 25.4
        self._width_spin.setValue(self._width_spin.value() * factor)
        self._height_spin.setValue(self._height_spin.value() * factor)
        self._margin_spin.setValue(self._margin_spin.value() * factor)

        suffix = " mm" if new_unit == "mm" else " in"
        self._width_spin.setSuffix(suffix)
        self._height_spin.setSuffix(suffix)
        self._margin_spin.setSuffix(suffix)
        self._unit = new_unit

    def get_canvas(self) -> Canvas:
        """Return a Canvas from the dialog's current settings."""
        preset = self._preset_combo.currentText()
        margin_mm = self._margin_spin.value()
        if self._unit == "inches":
            margin_mm *= 25.4

        if preset != "Custom" and preset in PAPER_PRESETS:
            return Canvas.from_preset(preset, margin=margin_mm)

        width_mm = self._width_spin.value()
        height_mm = self._height_spin.value()
        if self._unit == "inches":
            width_mm *= 25.4
            height_mm *= 25.4
        return Canvas(width_mm=width_mm, height_mm=height_mm, margin_mm=margin_mm, paper_preset="Custom")
