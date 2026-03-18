"""ExportDialog — export format and options dialog."""

from __future__ import annotations

import os

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ExportDialog(QDialog):
    """Dialog for configuring SVG, HPGL, and G-code export options."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(400)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Format selector ──────────────────────────────────────────────
        fmt_group = QGroupBox("Format")
        fmt_form = QFormLayout(fmt_group)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["SVG", "HPGL", "G-code", "Mural"])
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        fmt_form.addRow("Format:", self._format_combo)
        layout.addWidget(fmt_group)

        # ── Layer selection ───────────────────────────────────────────────
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QVBoxLayout(layer_group)
        self._current_radio = QRadioButton("Current Layer")
        self._all_sep_radio = QRadioButton("All Layers (Separate Files)")
        self._all_combined_radio = QRadioButton("All Layers (Combined)")
        self._current_radio.setChecked(True)
        self._current_radio.toggled.connect(self._update_path_label)
        self._all_sep_radio.toggled.connect(self._update_path_label)
        layer_layout.addWidget(self._current_radio)
        layer_layout.addWidget(self._all_sep_radio)
        layer_layout.addWidget(self._all_combined_radio)
        layout.addWidget(layer_group)

        # ── Format-specific options (stacked) ────────────────────────────
        self._format_stack = QStackedWidget()

        # Page 0 — SVG options
        svg_widget = QWidget()
        svg_form = QFormLayout(svg_widget)
        self._reg_marks_check = QCheckBox("Include registration marks")
        self._reg_marks_check.setChecked(True)
        svg_form.addRow(self._reg_marks_check)
        self._stroke_width_spin = QDoubleSpinBox()
        self._stroke_width_spin.setRange(0.01, 10.0)
        self._stroke_width_spin.setValue(0.3)
        self._stroke_width_spin.setSingleStep(0.1)
        self._stroke_width_spin.setSuffix(" mm")
        svg_form.addRow("Stroke width:", self._stroke_width_spin)
        self._format_stack.addWidget(svg_widget)   # index 0

        # Page 1 — HPGL options
        hpgl_widget = QWidget()
        hpgl_form = QFormLayout(hpgl_widget)
        self._hpgl_pen_spin = QSpinBox()
        self._hpgl_pen_spin.setRange(1, 8)
        self._hpgl_pen_spin.setValue(1)
        hpgl_form.addRow("Pen number:", self._hpgl_pen_spin)
        self._hpgl_speed_check = QCheckBox("Include speed (VS) command")
        self._hpgl_speed_check.toggled.connect(self._on_hpgl_speed_toggled)
        hpgl_form.addRow(self._hpgl_speed_check)
        self._hpgl_speed_spin = QSpinBox()
        self._hpgl_speed_spin.setRange(1, 127)
        self._hpgl_speed_spin.setValue(20)
        self._hpgl_speed_spin.setEnabled(False)
        hpgl_form.addRow("Speed:", self._hpgl_speed_spin)
        self._hpgl_force_check = QCheckBox("Include force (FS) command")
        self._hpgl_force_check.toggled.connect(self._on_hpgl_force_toggled)
        hpgl_form.addRow(self._hpgl_force_check)
        self._hpgl_force_spin = QSpinBox()
        self._hpgl_force_spin.setRange(1, 127)
        self._hpgl_force_spin.setValue(8)
        self._hpgl_force_spin.setEnabled(False)
        hpgl_form.addRow("Force:", self._hpgl_force_spin)
        self._format_stack.addWidget(hpgl_widget)  # index 1

        # Page 2 — G-code options
        gcode_widget = QWidget()
        gcode_form = QFormLayout(gcode_widget)
        self._gcode_travel_speed_spin = QSpinBox()
        self._gcode_travel_speed_spin.setRange(100, 10000)
        self._gcode_travel_speed_spin.setValue(3000)
        self._gcode_travel_speed_spin.setSuffix(" mm/min")
        gcode_form.addRow("Travel speed:", self._gcode_travel_speed_spin)
        self._gcode_draw_speed_spin = QSpinBox()
        self._gcode_draw_speed_spin.setRange(100, 10000)
        self._gcode_draw_speed_spin.setValue(1000)
        self._gcode_draw_speed_spin.setSuffix(" mm/min")
        gcode_form.addRow("Draw speed:", self._gcode_draw_speed_spin)
        self._gcode_pen_up_spin = QSpinBox()
        self._gcode_pen_up_spin.setRange(0, 180)
        self._gcode_pen_up_spin.setValue(0)
        self._gcode_pen_up_spin.setSuffix("°")
        gcode_form.addRow("Pen-up angle:", self._gcode_pen_up_spin)
        self._gcode_pen_down_spin = QSpinBox()
        self._gcode_pen_down_spin.setRange(0, 180)
        self._gcode_pen_down_spin.setValue(90)
        self._gcode_pen_down_spin.setSuffix("°")
        gcode_form.addRow("Pen-down angle:", self._gcode_pen_down_spin)
        self._format_stack.addWidget(gcode_widget)  # index 2

        # Page 3 — Mural options
        mural_widget = QWidget()
        mural_form = QFormLayout(mural_widget)
        self._mural_pin_distance_spin = QDoubleSpinBox()
        self._mural_pin_distance_spin.setRange(100.0, 5000.0)
        self._mural_pin_distance_spin.setValue(1025.0)
        self._mural_pin_distance_spin.setSingleStep(10.0)
        self._mural_pin_distance_spin.setSuffix(" mm")
        self._mural_pin_distance_spin.setDecimals(1)
        self._mural_pin_distance_spin.setToolTip(
            "Distance between the two wall-mounted anchor pins.\n"
            "Drawing area width = pin distance × 0.6"
        )
        mural_form.addRow("Pin distance:", self._mural_pin_distance_spin)
        # Computed drawing width (informational)
        self._mural_width_label = QLabel()
        self._mural_pin_distance_spin.valueChanged.connect(self._update_mural_width_label)
        mural_form.addRow("Drawing width:", self._mural_width_label)
        self._format_stack.addWidget(mural_widget)  # index 3
        self._update_mural_width_label(self._mural_pin_distance_spin.value())

        fmt_options_group = QGroupBox("Options")
        fo_layout = QVBoxLayout(fmt_options_group)
        fo_layout.addWidget(self._format_stack)
        layout.addWidget(fmt_options_group)

        # ── Output path ───────────────────────────────────────────────────
        path_group = QGroupBox("Output")
        path_layout = QVBoxLayout(path_group)
        self._path_label = QLabel("File path:")
        path_layout.addWidget(self._path_label)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        path_row.addWidget(self._path_edit)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self._browse_btn)
        path_layout.addLayout(path_row)
        layout.addWidget(path_group)

        # ── Dialog buttons ────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Trigger initial state
        self._on_format_changed(self._format_combo.currentText())

    # ------------------------------------------------------------------
    # Internal slots

    def _on_format_changed(self, fmt: str) -> None:
        index = {"SVG": 0, "HPGL": 1, "G-code": 2, "Mural": 3}.get(fmt, 0)
        self._format_stack.setCurrentIndex(index)
        # "All Combined" only makes sense for SVG; hide it for HPGL/G-code/Mural
        combined_visible = fmt == "SVG"
        self._all_combined_radio.setVisible(combined_visible)
        if not combined_visible and self._all_combined_radio.isChecked():
            self._all_sep_radio.setChecked(True)

    def _update_mural_width_label(self, pin_distance: float) -> None:
        width = pin_distance * 0.6
        self._mural_width_label.setText(f"{width:.1f} mm")

    def _on_hpgl_speed_toggled(self, checked: bool) -> None:
        self._hpgl_speed_spin.setEnabled(checked)

    def _on_hpgl_force_toggled(self, checked: bool) -> None:
        self._hpgl_force_spin.setEnabled(checked)

    def _update_path_label(self) -> None:
        if self._all_sep_radio.isChecked():
            self._path_label.setText("Output directory:")
        else:
            self._path_label.setText("File path:")

    def _on_browse(self) -> None:
        fmt = self._format_combo.currentText()
        if self._all_sep_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if path:
                self._path_edit.setText(path)
        else:
            if fmt == "SVG":
                file_filter = "SVG Files (*.svg);;All Files (*)"
            elif fmt == "HPGL":
                file_filter = "HPGL Files (*.plt *.hpgl);;All Files (*)"
            elif fmt == "Mural":
                file_filter = "Mural Files (*.mural);;All Files (*)"
            else:
                file_filter = "G-code Files (*.gcode *.nc);;All Files (*)"
            path, _ = QFileDialog.getSaveFileName(self, "Export As", "", file_filter)
            if path:
                path = self._ensure_extension(path, fmt)
                self._path_edit.setText(path)

    # ------------------------------------------------------------------
    # Extension helpers

    #: Valid extensions per format (lower-case).
    _VALID_EXTENSIONS: dict[str, list[str]] = {
        "SVG": [".svg"],
        "HPGL": [".plt", ".hpgl"],
        "G-code": [".gcode", ".nc"],
        "Mural": [".mural"],
    }

    #: Default (primary) extension per format.
    _DEFAULT_EXTENSION: dict[str, str] = {
        "SVG": ".svg",
        "HPGL": ".plt",
        "G-code": ".gcode",
        "Mural": ".mural",
    }

    @staticmethod
    def _ensure_extension(path: str, fmt: str) -> str:
        """Return *path* with the correct extension for *fmt* appended if missing.

        Rules:
        - If *path* already has a valid extension for *fmt*, return unchanged.
        - If *path* has no extension (empty string from ``os.path.splitext``),
          append the primary extension for *fmt*.
        - If *path* has some other extension (e.g. ``.png``), leave it as-is
          (do not silently overwrite a user-supplied extension).
        """
        valid = ExportDialog._VALID_EXTENSIONS.get(fmt, [])
        default = ExportDialog._DEFAULT_EXTENSION.get(fmt, "")
        _, ext = os.path.splitext(path)
        if ext.lower() in valid:
            return path
        if ext == "":
            return path + default
        # Has an unrecognised extension — leave it untouched.
        return path

    # ------------------------------------------------------------------
    # Settings persistence

    def _load_settings(self) -> None:
        """Restore last-used export settings from QSettings."""
        s = QSettings("Plottter", "Plottter")
        s.beginGroup("export")
        fmt = s.value("format", "SVG")
        idx = self._format_combo.findText(fmt)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)

        layer_mode = s.value("layer_mode", "current")
        if layer_mode == "all_separate":
            self._all_sep_radio.setChecked(True)
        elif layer_mode == "all_combined":
            self._all_combined_radio.setChecked(True)
        else:
            self._current_radio.setChecked(True)

        # SVG
        self._reg_marks_check.setChecked(s.value("svg_reg_marks", True, type=bool))
        sw = s.value("svg_stroke_width", 0.3)
        try:
            self._stroke_width_spin.setValue(float(sw))
        except (TypeError, ValueError):
            pass

        # HPGL
        hpgl_pen = s.value("hpgl_pen", 1)
        try:
            self._hpgl_pen_spin.setValue(int(hpgl_pen))
        except (TypeError, ValueError):
            pass
        self._hpgl_speed_check.setChecked(s.value("hpgl_speed_enabled", False, type=bool))
        hpgl_speed = s.value("hpgl_speed", 20)
        try:
            self._hpgl_speed_spin.setValue(int(hpgl_speed))
        except (TypeError, ValueError):
            pass
        self._hpgl_force_check.setChecked(s.value("hpgl_force_enabled", False, type=bool))
        hpgl_force = s.value("hpgl_force", 8)
        try:
            self._hpgl_force_spin.setValue(int(hpgl_force))
        except (TypeError, ValueError):
            pass

        # G-code
        for attr, key, default in [
            ("_gcode_travel_speed_spin", "gcode_travel_speed", 3000),
            ("_gcode_draw_speed_spin", "gcode_draw_speed", 1000),
            ("_gcode_pen_up_spin", "gcode_pen_up", 0),
            ("_gcode_pen_down_spin", "gcode_pen_down", 90),
        ]:
            val = s.value(key, default)
            try:
                getattr(self, attr).setValue(int(val))
            except (TypeError, ValueError):
                pass

        # Mural
        mural_pin = s.value("mural_pin_distance", 1025.0)
        try:
            self._mural_pin_distance_spin.setValue(float(mural_pin))
        except (TypeError, ValueError):
            pass

        s.endGroup()

    def _save_settings(self) -> None:
        """Persist current export settings to QSettings."""
        s = QSettings("Plottter", "Plottter")
        s.beginGroup("export")
        s.setValue("format", self._format_combo.currentText())

        if self._all_sep_radio.isChecked():
            s.setValue("layer_mode", "all_separate")
        elif self._all_combined_radio.isChecked():
            s.setValue("layer_mode", "all_combined")
        else:
            s.setValue("layer_mode", "current")

        s.setValue("svg_reg_marks", self._reg_marks_check.isChecked())
        s.setValue("svg_stroke_width", self._stroke_width_spin.value())

        s.setValue("hpgl_pen", self._hpgl_pen_spin.value())
        s.setValue("hpgl_speed_enabled", self._hpgl_speed_check.isChecked())
        s.setValue("hpgl_speed", self._hpgl_speed_spin.value())
        s.setValue("hpgl_force_enabled", self._hpgl_force_check.isChecked())
        s.setValue("hpgl_force", self._hpgl_force_spin.value())

        s.setValue("gcode_travel_speed", self._gcode_travel_speed_spin.value())
        s.setValue("gcode_draw_speed", self._gcode_draw_speed_spin.value())
        s.setValue("gcode_pen_up", self._gcode_pen_up_spin.value())
        s.setValue("gcode_pen_down", self._gcode_pen_down_spin.value())

        s.setValue("mural_pin_distance", self._mural_pin_distance_spin.value())

        s.endGroup()

    def accept(self) -> None:
        self._save_settings()
        super().accept()

    # ------------------------------------------------------------------
    # Public API

    def get_settings(self) -> dict:
        """Return a dict of export settings."""
        if self._current_radio.isChecked():
            layer_mode = "current"
        elif self._all_sep_radio.isChecked():
            layer_mode = "all_separate"
        else:
            layer_mode = "all_combined"

        fmt = self._format_combo.currentText()

        output_path = self._path_edit.text()
        if not self._all_sep_radio.isChecked():
            output_path = self._ensure_extension(output_path, fmt)

        base: dict = {
            "format": fmt,
            "layer_mode": layer_mode,
            "output_path": output_path,
        }

        if fmt == "SVG":
            base["registration_marks"] = self._reg_marks_check.isChecked()
            base["stroke_width"] = self._stroke_width_spin.value()
        elif fmt == "HPGL":
            base["pen_number"] = self._hpgl_pen_spin.value()
            if self._hpgl_speed_check.isChecked():
                base["speed"] = self._hpgl_speed_spin.value()
            if self._hpgl_force_check.isChecked():
                base["force"] = self._hpgl_force_spin.value()
        elif fmt == "G-code":
            base["travel_speed"] = self._gcode_travel_speed_spin.value()
            base["draw_speed"] = self._gcode_draw_speed_spin.value()
            base["pen_up_angle"] = self._gcode_pen_up_spin.value()
            base["pen_down_angle"] = self._gcode_pen_down_spin.value()
        elif fmt == "Mural":
            base["top_distance"] = self._mural_pin_distance_spin.value()

        return base
