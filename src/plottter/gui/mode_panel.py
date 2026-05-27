"""ModePanel — radio-button selector for the three main modes."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QGroupBox, QRadioButton, QVBoxLayout, QWidget


class ModePanel(QWidget):
    """Displays radio buttons for Math Art / Image to Lines / Color Separation."""

    mode_changed = pyqtSignal(str)

    MODES = ["Math Art", "Image to Lines", "Color Separation", "Mask Paint", "Shape Drawing", "3D Scene", "Map"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group_box = QGroupBox("Mode")
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(6, 6, 6, 6)

        self._button_group = QButtonGroup(self)
        self._radio_buttons: dict[str, QRadioButton] = {}

        for mode in self.MODES:
            btn = QRadioButton(mode)
            self._radio_buttons[mode] = btn
            self._button_group.addButton(btn)
            group_layout.addWidget(btn)

        # Default: Math Art
        self._radio_buttons["Math Art"].setChecked(True)

        self._button_group.buttonClicked.connect(self._on_button_clicked)

        layout.addWidget(group_box)
        layout.addStretch()

    def _on_button_clicked(self, button: QRadioButton) -> None:
        self.mode_changed.emit(button.text())

    def current_mode(self) -> str:
        checked = self._button_group.checkedButton()
        return checked.text() if checked else "Math Art"

    def set_mode(self, mode: str) -> None:
        if mode in self._radio_buttons:
            self._radio_buttons[mode].setChecked(True)
