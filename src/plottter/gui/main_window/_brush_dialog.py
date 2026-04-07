"""_BrushDialog — factory that builds and shows the Apply Brush dialog."""

from __future__ import annotations

from plottter.models.path import Polyline


class _BrushDialog:
    """Factory that builds and shows the Apply Brush dialog.

    Returns (brush_type, params) on accept, or (None, None) on cancel.
    Uses a QStackedWidget so each brush type shows its own parameter page.
    """

    BRUSH_TYPES = ["None", "Stippled", "Multi-Stroke", "Calligraphic"]
    # Maps brush name → stack page index (0 = empty "None" page)
    _PAGE_INDEX = {"None": 0, "Stippled": 1, "Multi-Stroke": 2, "Calligraphic": 3}

    @staticmethod
    def run(parent, sample_paths: list[Polyline]) -> tuple[str | None, dict | None]:
        """Show the dialog and return (brush_type, params) or (None, None)."""
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QComboBox,
            QGroupBox,
            QVBoxLayout,
            QSpinBox,
            QStackedWidget,
            QWidget,
        )
        from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen
        from PyQt6.QtCore import Qt

        dialog = QDialog(parent)
        dialog.setWindowTitle("Apply Brush to Layer")
        dialog.setMinimumWidth(460)

        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(10)

        # --- Brush type selector ---
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Brush type:"))
        type_combo = QComboBox()
        type_combo.addItems(_BrushDialog.BRUSH_TYPES)
        type_layout.addWidget(type_combo)
        type_layout.addStretch()
        main_layout.addLayout(type_layout)

        # --- Stacked parameter pages ---
        stack = QStackedWidget()
        main_layout.addWidget(stack)

        # Page 0: None (empty placeholder)
        stack.addWidget(QWidget())

        # Page 1: Stipple parameters
        stipple_group = QGroupBox("Stipple Parameters")
        stipple_form = QFormLayout(stipple_group)

        spacing_spin = QDoubleSpinBox()
        spacing_spin.setRange(0.1, 50.0)
        spacing_spin.setSingleStep(0.1)
        spacing_spin.setValue(1.0)
        spacing_spin.setSuffix(" mm")
        stipple_form.addRow("Spacing:", spacing_spin)

        size_spin = QDoubleSpinBox()
        size_spin.setRange(0.05, 5.0)
        size_spin.setSingleStep(0.05)
        size_spin.setValue(0.3)
        size_spin.setSuffix(" mm")
        stipple_form.addRow("Dot size:", size_spin)

        randomness_spin = QDoubleSpinBox()
        randomness_spin.setRange(0.0, 1.0)
        randomness_spin.setSingleStep(0.05)
        randomness_spin.setValue(0.2)
        stipple_form.addRow("Randomness:", randomness_spin)

        stack.addWidget(stipple_group)  # index 1

        # Page 2: Multi-Stroke parameters
        multi_group = QGroupBox("Multi-Stroke Parameters")
        multi_form = QFormLayout(multi_group)

        stroke_count_spin = QSpinBox()
        stroke_count_spin.setRange(1, 20)
        stroke_count_spin.setValue(3)
        multi_form.addRow("Stroke count:", stroke_count_spin)

        spread_spin = QDoubleSpinBox()
        spread_spin.setRange(0.0, 10.0)
        spread_spin.setSingleStep(0.1)
        spread_spin.setValue(0.5)
        spread_spin.setSuffix(" mm")
        multi_form.addRow("Spread:", spread_spin)

        stroke_noise_spin = QDoubleSpinBox()
        stroke_noise_spin.setRange(0.0, 1.0)
        stroke_noise_spin.setSingleStep(0.05)
        stroke_noise_spin.setValue(0.3)
        multi_form.addRow("Noise:", stroke_noise_spin)

        stack.addWidget(multi_group)  # index 2

        # Page 3: Calligraphic parameters
        calli_group = QGroupBox("Calligraphic Parameters")
        calli_form = QFormLayout(calli_group)

        nib_angle_spin = QDoubleSpinBox()
        nib_angle_spin.setRange(0.0, 180.0)
        nib_angle_spin.setSingleStep(5.0)
        nib_angle_spin.setValue(45.0)
        nib_angle_spin.setSuffix("°")
        calli_form.addRow("Nib angle:", nib_angle_spin)

        nib_width_spin = QDoubleSpinBox()
        nib_width_spin.setRange(0.1, 20.0)
        nib_width_spin.setSingleStep(0.1)
        nib_width_spin.setValue(1.5)
        nib_width_spin.setSuffix(" mm")
        calli_form.addRow("Max width:", nib_width_spin)

        min_width_spin = QDoubleSpinBox()
        min_width_spin.setRange(0.01, 10.0)
        min_width_spin.setSingleStep(0.05)
        min_width_spin.setValue(0.2)
        min_width_spin.setSuffix(" mm")
        calli_form.addRow("Min width:", min_width_spin)

        stack.addWidget(calli_group)  # index 3

        # --- Preview area ---
        preview_label = QLabel()
        preview_label.setFixedSize(400, 180)
        preview_label.setStyleSheet("border: 1px solid gray; background: white;")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)

        # --------------- helpers ---------------
        def _get_params() -> dict:
            return {
                # Stipple
                "stipple_spacing_mm": spacing_spin.value(),
                "stipple_size_mm": size_spin.value(),
                "stipple_randomness": randomness_spin.value(),
                # Multi-stroke
                "stroke_count": stroke_count_spin.value(),
                "stroke_spread_mm": spread_spin.value(),
                "stroke_noise": stroke_noise_spin.value(),
                # Calligraphic
                "nib_angle": nib_angle_spin.value(),
                "nib_width_mm": nib_width_spin.value(),
                "min_width_mm": min_width_spin.value(),
            }

        def _update_page() -> None:
            brush = type_combo.currentText()
            page = _BrushDialog._PAGE_INDEX.get(brush, 0)
            stack.setCurrentIndex(page)

        def _render_preview() -> None:
            from plottter.processing.brush import apply_brush

            brush = type_combo.currentText()
            params = _get_params()
            preview_input = sample_paths[:50]
            try:
                brushed = apply_brush(preview_input, brush, params)
            except Exception:
                brushed = preview_input

            all_paths = list(brushed)
            if not all_paths:
                preview_label.setText("(no output)")
                return

            # Compute bounding box
            xs = [x for p in all_paths for x, _ in p]
            ys = [y for p in all_paths for _, y in p]
            if not xs:
                preview_label.setText("(no output)")
                return
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w_data = x_max - x_min or 1.0
            h_data = y_max - y_min or 1.0

            pw, ph = 400, 180
            margin = 10
            scale = min((pw - 2 * margin) / w_data, (ph - 2 * margin) / h_data)

            pixmap = QPixmap(pw, ph)
            pixmap.fill(QColor("white"))
            painter = QPainter(pixmap)
            pen = QPen(QColor("black"))
            pen.setWidthF(0.5)
            painter.setPen(pen)

            def to_px(x: float, y: float):
                px = margin + (x - x_min) * scale
                py = ph - margin - (y - y_min) * scale
                return px, py

            for path in all_paths:
                if len(path) < 2:
                    continue
                for i in range(len(path) - 1):
                    x1, y1 = to_px(*path[i])
                    x2, y2 = to_px(*path[i + 1])
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            painter.end()
            preview_label.setPixmap(pixmap)
            preview_label.setText("")

        def _on_change(*_) -> None:
            _update_page()
            _render_preview()

        type_combo.currentIndexChanged.connect(_on_change)
        # Stipple controls
        spacing_spin.valueChanged.connect(_on_change)
        size_spin.valueChanged.connect(_on_change)
        randomness_spin.valueChanged.connect(_on_change)
        # Multi-stroke controls
        stroke_count_spin.valueChanged.connect(_on_change)
        spread_spin.valueChanged.connect(_on_change)
        stroke_noise_spin.valueChanged.connect(_on_change)
        # Calligraphic controls
        nib_angle_spin.valueChanged.connect(_on_change)
        nib_width_spin.valueChanged.connect(_on_change)
        min_width_spin.valueChanged.connect(_on_change)

        # Initial state
        _update_page()
        _render_preview()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None, None

        return type_combo.currentText(), _get_params()
