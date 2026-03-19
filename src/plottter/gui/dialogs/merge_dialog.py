"""MergeDialog — parameter dialog for the Merge Nearby Paths tool."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from plottter.models.path import Polyline


class MergeDialog(QDialog):
    """Modal dialog for configuring the Merge Nearby Paths threshold.

    Shows a threshold spinbox and a live path-count preview that updates as
    the user adjusts the spinbox.  The last-used threshold is persisted via
    QSettings so it carries over across application restarts.
    """

    _SETTINGS_KEY = "tools/merge_threshold"
    _DEFAULT_THRESHOLD = 0.5
    _PREVIEW_SAMPLE = 500  # max paths used for the preview computation

    def __init__(
        self,
        paths: list[Polyline],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Nearby Paths")
        self.setMinimumWidth(380)

        self._all_paths: list[Polyline] = paths
        self._sample_paths: list[Polyline] = paths[: self._PREVIEW_SAMPLE]

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Threshold spinbox ---
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.1, 10.0)
        self._threshold_spin.setSingleStep(0.1)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setSuffix(" mm")
        self._threshold_spin.setToolTip(
            "Endpoints closer than this distance will be joined into a single path.\n"
            "Higher = more merging, fewer pen lifts."
        )

        # Restore last-used value (fall back to default if none stored)
        settings = QSettings("Plottter", "Plottter")
        last = settings.value(self._SETTINGS_KEY, self._DEFAULT_THRESHOLD, type=float)
        self._threshold_spin.setValue(last)

        form.addRow("Distance threshold (mm):", self._threshold_spin)

        # --- Live preview label ---
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview_label)

        # --- OK / Cancel buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire live update
        self._threshold_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        """Recompute the path-count preview with the current threshold."""
        from plottter.processing.merge import merge_nearby_paths

        if not self._sample_paths:
            self._preview_label.setText("(no paths to preview)")
            return

        threshold = self._threshold_spin.value()
        before = len(self._sample_paths)
        merged = merge_nearby_paths(self._sample_paths, threshold)
        after = len(merged)
        fewer = before - after
        self._preview_label.setText(
            f"{before:,} paths \u2192 {after:,} paths ({fewer:,} fewer pen lifts)"
        )

    def _on_accept(self) -> None:
        """Persist the chosen threshold and close with Accepted result."""
        settings = QSettings("Plottter", "Plottter")
        settings.setValue(self._SETTINGS_KEY, self._threshold_spin.value())
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(self) -> float:
        """Return the threshold (mm) chosen by the user."""
        return self._threshold_spin.value()
