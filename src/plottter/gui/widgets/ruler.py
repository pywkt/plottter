"""Ruler widgets for the canvas (Phase 169).

Two thin widgets that frame the canvas and show a millimetre scale:

- :class:`RulerWidget` — a 24 px-thick strip (horizontal across the top,
  vertical down the left) painting minor/major ticks, integer-mm labels and a
  cursor-marker line.
- :class:`RulerCorner` — the 24×24 px top-left box that just reads ``"mm"``.

The widgets are *passive*: they hold a reference to the :class:`CanvasWidget`
and read its view transform through the public ``view_zoom()`` /
``view_pan_offset()`` accessors. Tick selection is delegated to the pure,
unit-testable :func:`choose_tick_steps`.

See ``specs/canvas-performance.md`` §10.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

# Fixed thickness of the ruler strips and the corner box, in device px.
RULER_THICKNESS_PX = 24

# Candidate step sizes (mm), smallest first.
_TICK_STEPS_MM = (1, 2, 5, 10, 20, 50, 100, 200)

# Minimum on-screen spacing (px) before a step is considered "readable".
_MIN_MAJOR_SPACING_PX = 48.0
_MIN_MINOR_SPACING_PX = 4.0

# Colours.
_BG_COLOR = QColor(245, 245, 245)
_BORDER_COLOR = QColor(180, 180, 180)
_TICK_COLOR = QColor(110, 110, 110)
_LABEL_COLOR = QColor(70, 70, 70)
_MARKER_COLOR = QColor(220, 40, 40)


def choose_tick_steps(zoom_px_per_mm: float) -> tuple[float, float]:
    """Pick (major_mm, minor_mm) tick spacing for the given zoom.

    ``major`` is the smallest step in ``[1, 2, 5, 10, 20, 50, 100, 200]`` whose
    on-screen size (``step * zoom``) is at least 48 px (falling back to 200 if
    none qualifies). ``minor`` is ``major / 5`` when that subdivision is at
    least 4 px on screen, otherwise ``minor == major`` (no minor ticks).

    See ``specs/canvas-performance.md`` §10.2.
    """
    if not zoom_px_per_mm or zoom_px_per_mm <= 0:
        zoom_px_per_mm = 1.0

    major = float(_TICK_STEPS_MM[-1])
    for step in _TICK_STEPS_MM:
        if step * zoom_px_per_mm >= _MIN_MAJOR_SPACING_PX:
            major = float(step)
            break

    minor_candidate = major / 5.0
    if minor_candidate * zoom_px_per_mm >= _MIN_MINOR_SPACING_PX:
        minor = minor_candidate
    else:
        minor = major

    return (major, minor)


class RulerWidget(QWidget):
    """A thin millimetre ruler aligned with the canvas along one axis.

    ``orientation`` is :data:`Qt.Orientation.Horizontal` (top, spans canvas x)
    or :data:`Qt.Orientation.Vertical` (left, spans canvas y). The widget shares
    the canvas's pixel coordinate along its long axis, so a point at widget
    pixel ``p`` corresponds to the same canvas pixel ``p``.
    """

    def __init__(
        self,
        orientation: Qt.Orientation,
        canvas: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._orientation = orientation
        self._canvas = canvas
        self._marker_mm: float | None = None

        if orientation == Qt.Orientation.Horizontal:
            self.setFixedHeight(RULER_THICKNESS_PX)
        else:
            self.setFixedWidth(RULER_THICKNESS_PX)

        font = QFont(self.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2.0))
        self._label_font = font

    # -- public API ----------------------------------------------------

    def set_canvas(self, canvas: QWidget) -> None:
        """Attach the canvas whose transform this ruler mirrors."""
        self._canvas = canvas
        self.update()

    def set_marker_mm(self, x_mm: float, y_mm: float) -> None:
        """Position the cursor marker from a mouse position in mm."""
        value = x_mm if self._orientation == Qt.Orientation.Horizontal else y_mm
        if self._marker_mm != value:
            self._marker_mm = value
            self.update()

    def clear_marker(self) -> None:
        """Hide the cursor marker (e.g. when the mouse leaves the canvas)."""
        if self._marker_mm is not None:
            self._marker_mm = None
            self.update()

    # -- transform helpers ---------------------------------------------

    def _view(self) -> tuple[float, float] | None:
        """Return (zoom, pan_along_axis) or None if no canvas is attached."""
        if self._canvas is None:
            return None
        zoom = float(self._canvas.view_zoom())
        if zoom <= 0:
            return None
        pan = self._canvas.view_pan_offset()
        pan_axis = pan.x() if self._orientation == Qt.Orientation.Horizontal else pan.y()
        return (zoom, float(pan_axis))

    def _mm_to_px(self, mm: float, zoom: float, pan_axis: float) -> float:
        return mm * zoom + pan_axis

    def _px_to_mm(self, px: float, zoom: float, pan_axis: float) -> float:
        return (px - pan_axis) / zoom

    # -- painting ------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG_COLOR)

        horizontal = self._orientation == Qt.Orientation.Horizontal
        long_px = self.width() if horizontal else self.height()

        view = self._view()
        if view is None or long_px <= 0:
            self._paint_border(painter)
            painter.end()
            return

        zoom, pan_axis = view
        major, minor = choose_tick_steps(zoom)

        mm_low = self._px_to_mm(0.0, zoom, pan_axis)
        mm_high = self._px_to_mm(float(long_px), zoom, pan_axis)
        if mm_low > mm_high:
            mm_low, mm_high = mm_high, mm_low

        ratio = max(1, int(round(major / minor))) if minor > 0 else 1
        n_start = math.floor(mm_low / minor)
        n_end = math.ceil(mm_high / minor)

        major_len = RULER_THICKNESS_PX * 2.0 / 3.0
        minor_len = RULER_THICKNESS_PX * 1.0 / 3.0

        painter.setFont(self._label_font)
        for n in range(n_start, n_end + 1):
            mm = n * minor
            px = self._mm_to_px(mm, zoom, pan_axis)
            is_major = (n % ratio == 0)
            length = major_len if is_major else minor_len
            self._draw_tick(painter, px, length, horizontal)
            if is_major:
                self._draw_label(painter, px, mm, horizontal)

        if self._marker_mm is not None:
            marker_px = self._mm_to_px(self._marker_mm, zoom, pan_axis)
            if -1.0 <= marker_px <= long_px + 1.0:
                self._draw_marker(painter, marker_px, horizontal)

        self._paint_border(painter)
        painter.end()

    def _draw_tick(
        self, painter: QPainter, pos: float, length: float, horizontal: bool
    ) -> None:
        painter.setPen(QPen(_TICK_COLOR, 1))
        if horizontal:
            # Ticks hang up from the bottom edge (adjacent to the canvas).
            y1 = RULER_THICKNESS_PX
            painter.drawLine(QPointF(pos, y1), QPointF(pos, y1 - length))
        else:
            # Ticks hang in from the right edge (adjacent to the canvas).
            x1 = RULER_THICKNESS_PX
            painter.drawLine(QPointF(x1, pos), QPointF(x1 - length, pos))

    def _draw_label(
        self, painter: QPainter, pos: float, mm: float, horizontal: bool
    ) -> None:
        painter.setPen(QPen(_LABEL_COLOR, 1))
        text = str(int(round(mm)))
        if horizontal:
            rect = QRectF(pos + 2.0, 0.0, 60.0, RULER_THICKNESS_PX)
            painter.drawText(
                rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop), text
            )
        else:
            rect = QRectF(1.0, pos + 1.0, RULER_THICKNESS_PX - 2.0, 12.0)
            painter.drawText(
                rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop), text
            )

    def _draw_marker(self, painter: QPainter, pos: float, horizontal: bool) -> None:
        painter.setPen(QPen(_MARKER_COLOR, 1))
        if horizontal:
            painter.drawLine(QPointF(pos, 0.0), QPointF(pos, RULER_THICKNESS_PX))
        else:
            painter.drawLine(QPointF(0.0, pos), QPointF(RULER_THICKNESS_PX, pos))

    def _paint_border(self, painter: QPainter) -> None:
        painter.setPen(QPen(_BORDER_COLOR, 1))
        r = self.rect()
        if self._orientation == Qt.Orientation.Horizontal:
            painter.drawLine(r.bottomLeft(), r.bottomRight())
        else:
            painter.drawLine(r.topRight(), r.bottomRight())


class RulerCorner(QWidget):
    """The fixed 24×24 px top-left corner box, labelled ``"mm"``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(RULER_THICKNESS_PX, RULER_THICKNESS_PX)
        font = QFont(self.font())
        font.setPointSizeF(max(6.0, font.pointSizeF() - 2.0))
        self._label_font = font

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG_COLOR)
        painter.setFont(self._label_font)
        painter.setPen(QPen(_LABEL_COLOR, 1))
        painter.drawText(
            self.rect(), int(Qt.AlignmentFlag.AlignCenter), "mm"
        )
        painter.setPen(QPen(_BORDER_COLOR, 1))
        r = self.rect()
        painter.drawLine(r.bottomLeft(), r.bottomRight())
        painter.drawLine(r.topRight(), r.bottomRight())
        painter.end()
