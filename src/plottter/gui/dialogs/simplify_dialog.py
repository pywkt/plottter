"""SimplifyDialog — parameter dialog for the Simplify Paths tool."""

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


class SimplifyDialog(QDialog):
    """Modal dialog for configuring the Simplify Paths tolerance.

    Shows a tolerance spinbox and a live path-count preview that updates as
    the user adjusts the spinbox.  The last-used tolerance is persisted via
    QSettings so it carries over across application restarts.
    """

    _SETTINGS_KEY = "tools/simplify_tolerance"
    _DEFAULT_TOLERANCE = 0.1
    _PREVIEW_SAMPLE = 200  # max paths used for the preview computation

    def __init__(
        self,
        paths: list[Polyline],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simplify Paths")
        self.setMinimumWidth(380)

        # Sample kept for live preview (up to _PREVIEW_SAMPLE paths)
        self._sample_paths: list[Polyline] = paths[: self._PREVIEW_SAMPLE]

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Tolerance spinbox ---
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)

        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(0.01, 5.0)
        self._tolerance_spin.setSingleStep(0.05)
        self._tolerance_spin.setDecimals(2)
        self._tolerance_spin.setSuffix(" mm")
        self._tolerance_spin.setToolTip(
            "Points within this distance of the simplified line are removed.\n"
            "Higher = fewer points, less detail."
        )

        # Restore last-used value (fall back to default if none stored)
        settings = QSettings("Plottter", "Plottter")
        last = settings.value(self._SETTINGS_KEY, self._DEFAULT_TOLERANCE, type=float)
        self._tolerance_spin.setValue(last)

        form.addRow("Tolerance (mm):", self._tolerance_spin)

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
        self._tolerance_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_points(paths: list[Polyline]) -> int:
        return sum(len(p) for p in paths)

    def _update_preview(self) -> None:
        """Recompute the point-count preview with the current tolerance."""
        from plottter.processing.simplify import simplify_paths

        if not self._sample_paths:
            self._preview_label.setText("(no paths to preview)")
            return

        tolerance = self._tolerance_spin.value()
        before = self._count_points(self._sample_paths)
        simplified = simplify_paths(self._sample_paths, tolerance)
        after = self._count_points(simplified)
        reduction = ((before - after) / before * 100) if before > 0 else 0.0
        self._preview_label.setText(
            f"{before:,} points \u2192 {after:,} points ({reduction:.1f}% reduction)"
        )

    def _on_accept(self) -> None:
        """Persist the chosen tolerance and close with Accepted result."""
        settings = QSettings("Plottter", "Plottter")
        settings.setValue(self._SETTINGS_KEY, self._tolerance_spin.value())
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tolerance(self) -> float:
        """Return the tolerance (mm) chosen by the user."""
        return self._tolerance_spin.value()
