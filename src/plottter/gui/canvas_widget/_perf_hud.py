"""Env-gated paint-time performance HUD for CanvasWidget (spec §5.1).

Constructed only when ``PLOTTTER_PERF_HUD=1`` at widget construction; the
widget keeps ``self._perf_hud = None`` otherwise, so the cost in the common
case is a single ``if self._perf_hud is None`` per paint.

``paintEvent`` times its whole body with ``time.perf_counter()`` plus named
sections (``layers``, ``mask``, ``overlays``, …) via the ``section(name)``
context manager. When the HUD is disabled, ``paintEvent`` uses
``_null_section`` instead, which yields a no-op context manager so the section
``with`` blocks add no measurable overhead. The HUD never calls
``update()`` itself — it draws only when something else triggers a repaint.
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager, nullcontext

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter


def _null_section(_name: str):
    """No-op stand-in for ``PerfHud.section`` used when the HUD is disabled."""
    return nullcontext()


class PerfHud:
    """Accumulates per-frame and per-section paint timings and draws them."""

    MAXLEN = 30
    #: Sections rendered (in this order) when present in the current frame.
    SECTION_ORDER = ("layers", "mask", "overlays", "blit")

    def __init__(self) -> None:
        self._frame_times: deque[float] = deque(maxlen=self.MAXLEN)
        self._sections: dict[str, float] = {}
        self._frame_start: float | None = None
        self._last_frame_ms: float = 0.0

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Reset per-section accumulators and start the frame timer."""
        self._sections = {}
        self._frame_start = time.perf_counter()

    @contextmanager
    def section(self, name: str):
        """Time the wrapped block, accumulating ms under *name* for this frame."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._sections[name] = self._sections.get(name, 0.0) + elapsed_ms

    def end_frame(self) -> None:
        """Stop the frame timer and record the elapsed body time."""
        if self._frame_start is None:
            return
        self._last_frame_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self._frame_times.append(self._last_frame_ms)

    def _avg_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _format_line(self) -> str:
        parts = [
            f"paint {self._last_frame_ms:.1f}ms (avg {self._avg_ms():.1f})"
        ]
        for name in self.SECTION_ORDER:
            if name in self._sections:
                parts.append(f"{name} {self._sections[name]:.1f}")
        fps = 1000.0 / self._last_frame_ms if self._last_frame_ms > 0.0 else 0.0
        parts.append(f"fps~{fps:.0f}")
        return " | ".join(parts)

    def draw(self, painter: QPainter) -> None:
        """Draw the HUD line top-left of the widget. Call after ``end_frame``."""
        text = self._format_line()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )
        painter.setOpacity(1.0)
        font = painter.font()
        font.setPointSize(9)
        font.setFamily("monospace")
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text)
        text_h = metrics.height()
        painter.fillRect(
            QRectF(4.0, 4.0, text_w + 8.0, text_h + 4.0),
            QColor(0, 0, 0, 180),
        )
        painter.setPen(QColor("#00FF88"))
        painter.drawText(QPointF(8.0, 6.0 + metrics.ascent()), text)
        painter.restore()
