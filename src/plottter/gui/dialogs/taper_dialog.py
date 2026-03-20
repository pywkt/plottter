"""TaperSettingsDialog — parameter dialog for the Taper Paths tool."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from plottter.models.path import Polyline


class TaperSettingsDialog(QDialog):
    """Modal dialog for configuring the Taper Paths effect.

    Parameters
    ----------
    paths:
        The paths from the active layer (used only to compute the preview count).
    parent:
        Optional parent widget.
    """

    _SETTINGS_GROUP = "tools/taper"
    _PREVIEW_SAMPLE = 200  # max paths used for the preview computation

    def __init__(
        self,
        paths: list[Polyline],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Taper Paths")
        self.setMinimumWidth(400)

        self._sample_paths: list[Polyline] = paths[: self._PREVIEW_SAMPLE]

        settings = QSettings("Plottter", "Plottter")
        settings.beginGroup(self._SETTINGS_GROUP)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)

        # Max width
        self._max_width_spin = QDoubleSpinBox()
        self._max_width_spin.setRange(0.1, 5.0)
        self._max_width_spin.setSingleStep(0.1)
        self._max_width_spin.setDecimals(2)
        self._max_width_spin.setSuffix(" mm")
        self._max_width_spin.setToolTip("Maximum stroke width at the widest point of the taper.")
        self._max_width_spin.setValue(settings.value("max_width_mm", 1.0, type=float))
        form.addRow("Max width:", self._max_width_spin)

        # Fade fraction
        self._fade_spin = QDoubleSpinBox()
        self._fade_spin.setRange(0.0, 0.5)
        self._fade_spin.setSingleStep(0.05)
        self._fade_spin.setDecimals(2)
        self._fade_spin.setToolTip(
            "Fraction of path length used to fade in/out the width.\n"
            "0.0 = uniform width (no taper). 0.5 = full taper."
        )
        self._fade_spin.setValue(settings.value("fade_fraction", 0.15, type=float))
        form.addRow("Fade fraction:", self._fade_spin)

        # Fill mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Outline", "Filled"])
        saved_mode = settings.value("fill_mode", "Outline", type=str)
        idx = self._mode_combo.findText(saved_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_combo.setToolTip(
            "Outline: two edge polylines per path (fast).\n"
            "Filled: parallel strokes across the width (more ink)."
        )
        form.addRow("Fill mode:", self._mode_combo)

        # Fill spacing (visible only in Filled mode)
        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(0.1, 2.0)
        self._spacing_spin.setSingleStep(0.1)
        self._spacing_spin.setDecimals(2)
        self._spacing_spin.setSuffix(" mm")
        self._spacing_spin.setToolTip("Spacing between parallel fill strokes.")
        self._spacing_spin.setValue(settings.value("fill_spacing_mm", 0.3, type=float))
        self._spacing_label = QLabel("Fill spacing:")
        form.addRow(self._spacing_label, self._spacing_spin)

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

        # Wire up signals
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._max_width_spin.valueChanged.connect(self._update_preview)
        self._fade_spin.valueChanged.connect(self._update_preview)
        self._mode_combo.currentTextChanged.connect(self._update_preview)
        self._spacing_spin.valueChanged.connect(self._update_preview)

        self._on_mode_changed(self._mode_combo.currentText())
        self._update_preview()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        visible = mode == "Filled"
        self._spacing_spin.setVisible(visible)
        self._spacing_label.setVisible(visible)

    def _update_preview(self) -> None:
        from plottter.processing.taper import taper_paths

        if not self._sample_paths:
            self._preview_label.setText("(no paths to preview)")
            return

        before = len(self._sample_paths)
        try:
            result = taper_paths(
                self._sample_paths,
                max_width_mm=self._max_width_spin.value(),
                fade_fraction=self._fade_spin.value(),
                fill_spacing_mm=self._spacing_spin.value(),
                fill_mode=self._mode_combo.currentText().lower(),
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
        settings.setValue("max_width_mm", self._max_width_spin.value())
        settings.setValue("fade_fraction", self._fade_spin.value())
        settings.setValue("fill_mode", self._mode_combo.currentText())
        settings.setValue("fill_spacing_mm", self._spacing_spin.value())
        settings.endGroup()
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a dict with the chosen taper parameters."""
        return {
            "max_width_mm": self._max_width_spin.value(),
            "fade_fraction": self._fade_spin.value(),
            "fill_mode": self._mode_combo.currentText().lower(),
            "fill_spacing_mm": self._spacing_spin.value(),
        }
