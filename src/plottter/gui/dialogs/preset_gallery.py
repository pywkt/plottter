"""PresetGalleryDialog — browseable grid of all math art presets with thumbnail previews."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from plottter.models.canvas import Canvas

_THUMB_PX = 120   # thumbnail side in pixels
_THUMB_MM = 100.0  # synthetic canvas size for thumbnail generation (mm)
_COLS = 4          # grid columns


def _render_polylines_to_image(
    polylines: list[list[tuple[float, float]]],
    size_px: int = _THUMB_PX,
    canvas_mm: float = _THUMB_MM,
    color: str = "#000000",
) -> QImage:
    """Render polylines (mm coordinates) onto a QImage (thread-safe)."""
    image = QImage(size_px, size_px, QImage.Format.Format_RGB32)
    image.fill(QColor("#FFFFFF"))
    if not polylines:
        return image

    # Compute bounding box of all points to auto-fit
    all_xs = [p[0] for poly in polylines for p in poly]
    all_ys = [p[1] for poly in polylines for p in poly]
    if not all_xs:
        return image

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0

    padding = 6  # px padding inside thumbnail
    draw_size = size_px - 2 * padding
    scale = min(draw_size / span_x, draw_size / span_y)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(0.7)
    painter.setPen(pen)

    for polyline in polylines:
        if len(polyline) < 2:
            continue
        for i in range(len(polyline) - 1):
            x0 = (polyline[i][0] - min_x) * scale + padding
            y0 = (polyline[i][1] - min_y) * scale + padding
            x1 = (polyline[i + 1][0] - min_x) * scale + padding
            y1 = (polyline[i + 1][1] - min_y) * scale + padding
            painter.drawLine(round(x0), round(y0), round(x1), round(y1))

    painter.end()
    return image


def _render_polylines_to_pixmap(
    polylines: list[list[tuple[float, float]]],
    size_px: int = _THUMB_PX,
    canvas_mm: float = _THUMB_MM,
    color: str = "#000000",
) -> QPixmap:
    """Render polylines (mm coordinates) onto a square QPixmap (main thread only)."""
    return QPixmap.fromImage(
        _render_polylines_to_image(polylines, size_px=size_px, canvas_mm=canvas_mm, color=color)
    )


class _ThumbnailWorker(QThread):
    """Background QThread that generates thumbnails for presets one by one."""

    # (gen_cls, preset_name, is_user, QImage)
    thumbnail_ready = pyqtSignal(object, str, bool, object)
    all_done = pyqtSignal()

    def __init__(
        self,
        presets: list[tuple[Any, Any, bool]],
        canvas_mm: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presets = presets  # list of (gen_cls, preset, is_user)
        self._canvas_mm = canvas_mm
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self.isRunning():
            self.wait(5000)  # Wait for thread to stop so it can be safely destroyed

    def run(self) -> None:
        thumb_canvas = Canvas(
            width_mm=self._canvas_mm,
            height_mm=self._canvas_mm,
            margin_mm=5.0,
            paper_preset="Custom",
        )
        for gen_cls, preset, is_user in self._presets:
            if self._cancelled:
                break
            try:
                generator = gen_cls()
                polylines = generator.generate(preset.params, thumb_canvas)
            except Exception:
                polylines = []
            try:
                # Use QImage (thread-safe); converted to QPixmap in the main thread
                image = _render_polylines_to_image(
                    polylines, size_px=_THUMB_PX, canvas_mm=self._canvas_mm
                )
            except Exception:
                image = QImage(_THUMB_PX, _THUMB_PX, QImage.Format.Format_RGB32)
                image.fill(QColor("#EEEEEE"))
            self.thumbnail_ready.emit(gen_cls, preset.name, is_user, image)
        self.all_done.emit()


class _PresetCard(QFrame):
    """Clickable card showing a preset thumbnail, name, and generator type."""

    clicked = pyqtSignal(object, str)  # (gen_cls, preset_name)
    delete_requested = pyqtSignal(object, str)  # (gen_cls, preset_name) — user presets only

    def __init__(
        self,
        gen_cls: Any,
        preset_name: str,
        generator_display_name: str,
        is_user: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._gen_cls = gen_cls
        self._preset_name = preset_name
        self._is_user = is_user
        self._selected = False
        self._deleted: bool = False

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedWidth(_THUMB_PX + 20)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Thumbnail placeholder
        self._thumb_label = QLabel()
        placeholder = QPixmap(_THUMB_PX, _THUMB_PX)
        placeholder.fill(QColor("#F0F0F0"))
        self._thumb_label.setPixmap(placeholder)
        self._thumb_label.setFixedSize(_THUMB_PX, _THUMB_PX)
        layout.addWidget(self._thumb_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Preset name (bold); add star badge for user presets
        display_name = f"\u2605 {preset_name}" if is_user else preset_name
        name_label = QLabel(display_name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        font = name_label.font()
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)

        # Generator name (smaller, secondary)
        gen_label = QLabel(generator_display_name)
        gen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gen_label.setWordWrap(True)
        layout.addWidget(gen_label)

        # Apply distinct styling and context menu for user presets
        if is_user:
            self._apply_user_style()
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect(self._show_context_menu)

    def _apply_user_style(self) -> None:
        """Apply a subtle amber/gold border to identify user presets."""
        self.setStyleSheet(
            "QFrame { border: 1px solid #B8860B; background-color: #FFFDF0; }"
        )

    def _show_context_menu(self, pos: Any) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Preset")
        action = menu.exec(self.mapToGlobal(pos))
        if action is delete_action:
            self.delete_requested.emit(self._gen_cls, self._preset_name)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._thumb_label.setPixmap(
            pixmap.scaled(
                _THUMB_PX,
                _THUMB_PX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.setStyleSheet(
                "QFrame { border: 2px solid #0078D4; background-color: #E8F4FD; }"
            )
        elif self._is_user:
            self._apply_user_style()
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event: Any) -> None:
        self.clicked.emit(self._gen_cls, self._preset_name)
        super().mousePressEvent(event)


class PresetGalleryDialog(QDialog):
    """Browseable grid of all math art presets with auto-generated thumbnail previews."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preset Gallery")
        self.setMinimumSize(640, 520)
        self.resize(860, 640)

        self._selected_gen_cls: Any = None
        self._selected_preset_name: str = ""
        self._cards: list[_PresetCard] = []
        self._worker: _ThumbnailWorker | None = None
        self._user_section_header: QLabel | None = None

        self._setup_ui()
        self._start_thumbnail_generation()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Status line
        self._status_label = QLabel("Generating previews\u2026")
        layout.addWidget(self._status_label)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll, stretch=1)

        self._populate_cards()

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("Apply Preset")
            ok_btn.setEnabled(False)
            self._ok_btn = ok_btn
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _collect_math_presets(self) -> list[tuple[Any, Any]]:
        """Return list of (gen_cls, preset) for all registered math generators."""
        from plottter.generators import get_generators_by_category

        result: list[tuple[Any, Any]] = []
        for gen_cls in get_generators_by_category("math"):
            try:
                generator = gen_cls()
                for preset in generator.get_presets():
                    result.append((gen_cls, preset))
            except Exception:
                pass
        return result

    def _collect_user_presets_for_math(self) -> list[tuple[Any, Any]]:
        """Return list of (gen_cls, preset) for all user presets of math generators."""
        from plottter.generators import get_generators_by_category
        from plottter.presets.user_presets import load_user_presets

        result: list[tuple[Any, Any]] = []
        for gen_cls in get_generators_by_category("math"):
            try:
                user_presets = load_user_presets(gen_cls.name)
                for preset in user_presets:
                    result.append((gen_cls, preset))
            except Exception:
                pass
        return result

    def _populate_cards(self) -> None:
        """Create placeholder cards for all math presets (built-in and user) in a grid."""
        builtin_presets = self._collect_math_presets()
        user_presets = self._collect_user_presets_for_math()

        row = 0
        col = 0

        # --- Built-in presets ---
        for gen_cls, preset in builtin_presets:
            card = _PresetCard(gen_cls, preset.name, gen_cls.name, is_user=False, parent=self)
            card.clicked.connect(self._on_card_clicked)
            self._grid_layout.addWidget(card, row, col)
            self._cards.append(card)
            col += 1
            if col >= _COLS:
                col = 0
                row += 1

        # --- User presets section ---
        if user_presets:
            # Start a new row for the section header
            if col > 0:
                col = 0
                row += 1

            # Section header spanning all columns
            header = QLabel("\u2014 User Presets \u2014")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_font = header.font()
            header_font.setBold(True)
            header.setFont(header_font)
            header.setStyleSheet(
                "QLabel { color: #7A5C00; padding: 8px 0; "
                "border-bottom: 1px solid #C8A800; background-color: #FFFBEA; }"
            )
            self._grid_layout.addWidget(header, row, 0, 1, _COLS)
            self._user_section_header = header
            row += 1

            for gen_cls, preset in user_presets:
                card = _PresetCard(
                    gen_cls, preset.name, gen_cls.name, is_user=True, parent=self
                )
                card.clicked.connect(self._on_card_clicked)
                card.delete_requested.connect(self._on_delete_user_preset)
                self._grid_layout.addWidget(card, row, col)
                self._cards.append(card)
                col += 1
                if col >= _COLS:
                    col = 0
                    row += 1

    # ------------------------------------------------------------------
    # Thumbnail generation
    # ------------------------------------------------------------------

    def _start_thumbnail_generation(self) -> None:
        builtin = [(gc, p, False) for gc, p in self._collect_math_presets()]
        user = [(gc, p, True) for gc, p in self._collect_user_presets_for_math()]
        all_presets = builtin + user
        self._worker = _ThumbnailWorker(all_presets, _THUMB_MM, parent=None)
        self._worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._worker.all_done.connect(self._on_thumbnails_done)
        self._worker.start()

    def _on_thumbnail_ready(
        self, gen_cls: Any, preset_name: str, is_user: bool, image: Any
    ) -> None:
        pixmap = QPixmap.fromImage(image) if isinstance(image, QImage) else image
        for card in self._cards:
            if (
                card._gen_cls is gen_cls
                and card._preset_name == preset_name
                and card._is_user == is_user
            ):
                card.set_thumbnail(pixmap)
                break

    def _on_thumbnails_done(self) -> None:
        visible_count = sum(1 for c in self._cards if not c._deleted)
        self._status_label.setText(
            f"{visible_count} preset{'s' if visible_count != 1 else ''} available"
            " \u2014 click a preset to select it"
        )

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------

    def _on_card_clicked(self, gen_cls: Any, preset_name: str) -> None:
        for card in self._cards:
            card.set_selected(False)
        for card in self._cards:
            if card._gen_cls is gen_cls and card._preset_name == preset_name:
                card.set_selected(True)
                break
        self._selected_gen_cls = gen_cls
        self._selected_preset_name = preset_name
        if hasattr(self, "_ok_btn"):
            self._ok_btn.setEnabled(True)

    def selected_preset(self) -> tuple[Any, str] | tuple[None, None]:
        """Return (gen_cls, preset_name) of the accepted selection, or (None, None)."""
        if self._selected_gen_cls is not None and self._selected_preset_name:
            return self._selected_gen_cls, self._selected_preset_name
        return None, None

    # ------------------------------------------------------------------
    # User preset deletion
    # ------------------------------------------------------------------

    def _on_delete_user_preset(self, gen_cls: Any, preset_name: str) -> None:
        """Confirm and delete a user preset, then hide its card."""
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete user preset \u201c{preset_name}\u201d?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from plottter.presets.user_presets import delete_user_preset

        try:
            delete_user_preset(gen_cls.name, preset_name)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Delete Failed",
                f"Could not delete preset \u201c{preset_name}\u201d:\n{exc}",
            )
            return

        # Hide the deleted card and deselect if it was selected
        for card in self._cards:
            if card._gen_cls is gen_cls and card._preset_name == preset_name and card._is_user:
                card._deleted = True
                card.hide()
                if (
                    self._selected_gen_cls is gen_cls
                    and self._selected_preset_name == preset_name
                ):
                    self._selected_gen_cls = None
                    self._selected_preset_name = ""
                    if hasattr(self, "_ok_btn"):
                        self._ok_btn.setEnabled(False)
                break

        # Hide user section header if no user preset cards remain visible
        if self._user_section_header is not None:
            any_user_visible = any(
                c._is_user and not c._deleted for c in self._cards
            )
            if not any_user_visible:
                self._user_section_header.hide()

        # Update status count
        visible_count = sum(1 for c in self._cards if not c._deleted)
        self._status_label.setText(
            f"{visible_count} preset{'s' if visible_count != 1 else ''} available"
            " \u2014 click a preset to select it"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)
