"""OptimizeSettingsDialog — parameter dialog for the Optimize Current Layer tool."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_SETTINGS_GROUP = "tools/optimize"

# Default values (single source of truth)
_DEFAULTS = {
    "run_weld": False,
    "weld_tolerance": 0.1,
    "run_simplify": True,
    "simplify_tolerance": 0.1,
    "run_filter": True,
    "filter_min_length": 0.5,
    "run_clip": True,
    "run_merge": True,
    "merge_threshold": 0.5,
    "run_2opt": True,
    "run_3opt": False,
    "run_or_opt": True,
}


class OptimizeSettingsDialog(QDialog):
    """Modal dialog for configuring the full path optimization pipeline.

    Groups controls into Preprocessing, Merge, and Reordering sections.
    All settings are persisted via QSettings under the ``tools/optimize`` group.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimize Layer — Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Preprocessing group ---
        pre_group = QGroupBox("Preprocessing")
        pre_layout = QVBoxLayout(pre_group)
        pre_layout.setSpacing(6)

        self._weld_check = QCheckBox("Weld overlapping paths")
        self._weld_tol_spin = self._make_spin(0.01, 2.0, 0.05, 2, " mm",
            "Segments within this distance are treated as overlapping and removed.")
        pre_layout.addWidget(self._weld_check)
        pre_layout.addLayout(self._indent(self._labeled_spin("Tolerance:", self._weld_tol_spin)))

        self._simplify_check = QCheckBox("Simplify paths")
        self._simplify_tol_spin = self._make_spin(0.01, 2.0, 0.05, 2, " mm",
            "Ramer-Douglas-Peucker tolerance for curve simplification.")
        pre_layout.addWidget(self._simplify_check)
        pre_layout.addLayout(self._indent(self._labeled_spin("Tolerance:", self._simplify_tol_spin)))

        self._filter_check = QCheckBox("Filter short paths")
        self._filter_len_spin = self._make_spin(0.01, 10.0, 0.1, 2, " mm",
            "Paths shorter than this length are removed.")
        pre_layout.addWidget(self._filter_check)
        pre_layout.addLayout(self._indent(self._labeled_spin("Min length:", self._filter_len_spin)))

        self._clip_check = QCheckBox("Clip to canvas")
        self._clip_check.setToolTip("Trim paths that extend beyond the canvas drawing area.")
        pre_layout.addWidget(self._clip_check)

        layout.addWidget(pre_group)

        # --- Merge group ---
        merge_group = QGroupBox("Merge")
        merge_layout = QVBoxLayout(merge_group)
        merge_layout.setSpacing(6)

        self._merge_check = QCheckBox("Merge nearby endpoints")
        self._merge_thresh_spin = self._make_spin(0.01, 10.0, 0.1, 2, " mm",
            "Endpoints closer than this distance are joined into a single path.")
        merge_layout.addWidget(self._merge_check)
        merge_layout.addLayout(self._indent(self._labeled_spin("Threshold:", self._merge_thresh_spin)))

        layout.addWidget(merge_group)

        # --- Reordering group ---
        reorder_group = QGroupBox("Reordering")
        reorder_layout = QVBoxLayout(reorder_group)
        reorder_layout.setSpacing(4)

        self._opt2_check = QCheckBox("2-opt improvement")
        self._opt2_check.setToolTip("Run 2-opt pass to reduce total travel distance.")
        self._opt3_check = QCheckBox("3-opt improvement")
        self._opt3_check.setToolTip(
            "3-opt — finds improvements 2-opt misses, slower. "
            "For stipple/dot art with 1000+ paths."
        )
        self._oropt_check = QCheckBox("Or-opt improvement")
        self._oropt_check.setToolTip("Run Or-opt pass to further reduce travel distance.")
        reorder_layout.addWidget(self._opt2_check)
        reorder_layout.addWidget(self._opt3_check)
        reorder_layout.addWidget(self._oropt_check)

        layout.addWidget(reorder_group)

        # --- Button row ---
        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._restore_defaults)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(restore_btn)
        btn_row.addStretch()
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        # Enable/disable spinboxes based on checkbox state
        self._weld_check.toggled.connect(self._weld_tol_spin.setEnabled)
        self._simplify_check.toggled.connect(self._simplify_tol_spin.setEnabled)
        self._filter_check.toggled.connect(self._filter_len_spin.setEnabled)
        self._merge_check.toggled.connect(self._merge_thresh_spin.setEnabled)

        # Load persisted values (or defaults)
        self._load_settings()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _make_spin(
        self,
        min_val: float,
        max_val: float,
        step: float,
        decimals: int,
        suffix: str,
        tooltip: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setToolTip(tooltip)
        return spin

    def _labeled_spin(self, label: str, spin: QDoubleSpinBox) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow(label, spin)
        return form

    def _indent(self, layout) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addSpacing(20)
        row.addLayout(layout)
        return row

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        s = QSettings("Plottter", "Plottter")
        s.beginGroup(_SETTINGS_GROUP)
        d = _DEFAULTS

        self._weld_check.setChecked(s.value("run_weld", d["run_weld"], type=bool))
        self._weld_tol_spin.setValue(s.value("weld_tolerance", d["weld_tolerance"], type=float))

        self._simplify_check.setChecked(s.value("run_simplify", d["run_simplify"], type=bool))
        self._simplify_tol_spin.setValue(s.value("simplify_tolerance", d["simplify_tolerance"], type=float))

        self._filter_check.setChecked(s.value("run_filter", d["run_filter"], type=bool))
        self._filter_len_spin.setValue(s.value("filter_min_length", d["filter_min_length"], type=float))

        self._clip_check.setChecked(s.value("run_clip", d["run_clip"], type=bool))

        self._merge_check.setChecked(s.value("run_merge", d["run_merge"], type=bool))
        self._merge_thresh_spin.setValue(s.value("merge_threshold", d["merge_threshold"], type=float))

        self._opt2_check.setChecked(s.value("run_2opt", d["run_2opt"], type=bool))
        self._opt3_check.setChecked(s.value("run_3opt", d["run_3opt"], type=bool))
        self._oropt_check.setChecked(s.value("run_or_opt", d["run_or_opt"], type=bool))

        s.endGroup()

        # Sync spinbox enabled state with checkbox state
        self._weld_tol_spin.setEnabled(self._weld_check.isChecked())
        self._simplify_tol_spin.setEnabled(self._simplify_check.isChecked())
        self._filter_len_spin.setEnabled(self._filter_check.isChecked())
        self._merge_thresh_spin.setEnabled(self._merge_check.isChecked())

    def _save_settings(self) -> None:
        s = QSettings("Plottter", "Plottter")
        s.beginGroup(_SETTINGS_GROUP)
        s.setValue("run_weld", self._weld_check.isChecked())
        s.setValue("weld_tolerance", self._weld_tol_spin.value())
        s.setValue("run_simplify", self._simplify_check.isChecked())
        s.setValue("simplify_tolerance", self._simplify_tol_spin.value())
        s.setValue("run_filter", self._filter_check.isChecked())
        s.setValue("filter_min_length", self._filter_len_spin.value())
        s.setValue("run_clip", self._clip_check.isChecked())
        s.setValue("run_merge", self._merge_check.isChecked())
        s.setValue("merge_threshold", self._merge_thresh_spin.value())
        s.setValue("run_2opt", self._opt2_check.isChecked())
        s.setValue("run_3opt", self._opt3_check.isChecked())
        s.setValue("run_or_opt", self._oropt_check.isChecked())
        s.endGroup()

    def _restore_defaults(self) -> None:
        d = _DEFAULTS
        self._weld_check.setChecked(d["run_weld"])
        self._weld_tol_spin.setValue(d["weld_tolerance"])
        self._simplify_check.setChecked(d["run_simplify"])
        self._simplify_tol_spin.setValue(d["simplify_tolerance"])
        self._filter_check.setChecked(d["run_filter"])
        self._filter_len_spin.setValue(d["filter_min_length"])
        self._clip_check.setChecked(d["run_clip"])
        self._merge_check.setChecked(d["run_merge"])
        self._merge_thresh_spin.setValue(d["merge_threshold"])
        self._opt2_check.setChecked(d["run_2opt"])
        self._opt3_check.setChecked(d["run_3opt"])
        self._oropt_check.setChecked(d["run_or_opt"])

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        """Return a dict of all optimization settings chosen by the user."""
        return {
            "run_weld": self._weld_check.isChecked(),
            "weld_tolerance": self._weld_tol_spin.value(),
            "run_simplify": self._simplify_check.isChecked(),
            "simplify_tolerance": self._simplify_tol_spin.value(),
            "run_filter": self._filter_check.isChecked(),
            "filter_min_length": self._filter_len_spin.value(),
            "run_clip": self._clip_check.isChecked(),
            "run_merge": self._merge_check.isChecked(),
            "merge_threshold": self._merge_thresh_spin.value(),
            "run_2opt": self._opt2_check.isChecked(),
            "run_3opt": self._opt3_check.isChecked(),
            "run_or_opt": self._oropt_check.isChecked(),
        }
