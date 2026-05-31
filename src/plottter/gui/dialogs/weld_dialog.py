"""WeldDialog — parameter dialog for the Weld Overlapping Paths tool."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QSettings


class WeldDialog(QDialog):
    """Modal dialog for configuring the Remove Duplicate Segments tolerance.

    (Internal name kept as ``WeldDialog`` for compatibility with existing
    imports — the user-facing label is now "Remove Duplicate Segments" since
    the old "Weld" wording was widely misread as "join touching paths into
    one path", which is actually what Merge Nearby Paths does.)
    """

    _SETTINGS_KEY = "tools/weld_tolerance"
    _DEFAULT_TOLERANCE = 0.1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remove Duplicate Segments")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)

        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(0.01, 2.0)
        self._tolerance_spin.setSingleStep(0.05)
        self._tolerance_spin.setDecimals(2)
        self._tolerance_spin.setSuffix(" mm")
        self._tolerance_spin.setToolTip(
            "Two segments within this distance of each other are treated as "
            "duplicates — the later one is dropped so the pen doesn't draw "
            "the same line twice. Higher = more aggressive de-duplication."
        )

        settings = QSettings("Plottter", "Plottter")
        last = settings.value(self._SETTINGS_KEY, self._DEFAULT_TOLERANCE, type=float)
        self._tolerance_spin.setValue(last)

        form.addRow("Overlap tolerance (mm):", self._tolerance_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.setValue(self._SETTINGS_KEY, self._tolerance_spin.value())
        self.accept()

    def get_tolerance(self) -> float:
        """Return the tolerance (mm) chosen by the user."""
        return self._tolerance_spin.value()
