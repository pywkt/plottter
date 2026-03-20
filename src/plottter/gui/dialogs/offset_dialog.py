"""OffsetSettingsDialog — parameter dialog for the Offset Paths tool."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from plottter.models.path import Polyline


class OffsetSettingsDialog(QDialog):
    """Modal dialog for configuring the Offset Paths effect.

    Parameters
    ----------
    paths:
        The paths from the active layer (used to compute the preview count).
    parent:
        Optional parent widget.
    """

    _SETTINGS_GROUP = "tools/offset"
    _PREVIEW_SAMPLE = 200  # max paths used for the preview computation

    # Maps display label → internal value for offset_paths()
    _SIDES_MAP = {
        "Both": "both",
        "Left Only": "left",
        "Right Only": "right",
    }
    _JOIN_MAP = {
        "Round": "round",
        "Mitre": "mitre",
        "Bevel": "bevel",
    }

    def __init__(
        self,
        paths: list[Polyline],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Offset Paths")
        self.setMinimumWidth(380)

        self._sample_paths: list[Polyline] = paths[: self._PREVIEW_SAMPLE]

        settings = QSettings("Plottter", "Plottter")
        settings.beginGroup(self._SETTINGS_GROUP)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)

        # Distance
        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(0.1, 10.0)
        self._distance_spin.setSingleStep(0.1)
        self._distance_spin.setDecimals(2)
        self._distance_spin.setSuffix(" mm")
        self._distance_spin.setToolTip("Offset distance per copy in millimetres.")
        self._distance_spin.setValue(settings.value("distance_mm", 0.5, type=float))
        form.addRow("Distance:", self._distance_spin)

        # Sides
        self._sides_combo = QComboBox()
        self._sides_combo.addItems(list(self._SIDES_MAP.keys()))
        saved_sides = settings.value("sides", "Both", type=str)
        idx = self._sides_combo.findText(saved_sides)
        if idx >= 0:
            self._sides_combo.setCurrentIndex(idx)
        self._sides_combo.setToolTip(
            "Which side(s) of each path to offset.\n"
            "For closed paths: Left = outside, Right = inside."
        )
        form.addRow("Sides:", self._sides_combo)

        # Count
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 10)
        self._count_spin.setValue(settings.value("count", 1, type=int))
        self._count_spin.setToolTip("Number of offset copies per side.")
        form.addRow("Count:", self._count_spin)

        # Join style
        self._join_combo = QComboBox()
        self._join_combo.addItems(list(self._JOIN_MAP.keys()))
        saved_join = settings.value("join_style", "Round", type=str)
        idx = self._join_combo.findText(saved_join)
        if idx >= 0:
            self._join_combo.setCurrentIndex(idx)
        self._join_combo.setToolTip("Corner join style for offset paths.")
        form.addRow("Join style:", self._join_combo)

        # Include original
        self._include_original_check = QCheckBox()
        self._include_original_check.setChecked(
            settings.value("include_original", True, type=bool)
        )
        self._include_original_check.setToolTip(
            "Keep the original paths alongside the offset copies."
        )
        form.addRow("Include original:", self._include_original_check)

        settings.endGroup()

        # Preview label
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire up signals for live preview
        self._distance_spin.valueChanged.connect(self._update_preview)
        self._sides_combo.currentTextChanged.connect(self._update_preview)
        self._count_spin.valueChanged.connect(self._update_preview)
        self._join_combo.currentTextChanged.connect(self._update_preview)
        self._include_original_check.stateChanged.connect(self._update_preview)

        self._update_preview()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        from plottter.processing.offset import offset_paths

        if not self._sample_paths:
            self._preview_label.setText("(no paths to preview)")
            return

        before = len(self._sample_paths)
        try:
            result = offset_paths(
                self._sample_paths,
                **self.get_params(),
            )
        except Exception:
            result = self._sample_paths

        after = len(result)
        self._preview_label.setText(
            f"Estimated output: {before} paths \u2192 ~{after} paths"
        )

    def _on_accept(self) -> None:
        settings = QSettings("Plottter", "Plottter")
        settings.beginGroup(self._SETTINGS_GROUP)
        settings.setValue("distance_mm", self._distance_spin.value())
        settings.setValue("sides", self._sides_combo.currentText())
        settings.setValue("count", self._count_spin.value())
        settings.setValue("join_style", self._join_combo.currentText())
        settings.setValue("include_original", self._include_original_check.isChecked())
        settings.endGroup()
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a dict with the chosen offset parameters."""
        sides_display = self._sides_combo.currentText()
        join_display = self._join_combo.currentText()
        return {
            "distance_mm": self._distance_spin.value(),
            "sides": self._SIDES_MAP.get(sides_display, "both"),
            "count": self._count_spin.value(),
            "join_style": self._JOIN_MAP.get(join_display, "round"),
            "include_original": self._include_original_check.isChecked(),
        }
