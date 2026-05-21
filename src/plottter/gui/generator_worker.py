"""QThread worker for running generators off the main thread."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from plottter.generators.base import Generator
from plottter.models import Canvas, Polyline


class GeneratorWorker(QThread):
    """Runs a Generator.generate() call in a background thread.

    Signals:
        progress(int): emitted periodically with 0–100 completion percent.
        finished(list): emitted with the list[Polyline] result on success.
        error(str): emitted with an error message on failure.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    layers_finished = pyqtSignal(list)  # emitted instead of finished for multi-layer generators
    metadata_ready = pyqtSignal(dict)  # emitted after finished if params contain side-channel data
    error = pyqtSignal(str)

    def __init__(
        self,
        generator: Generator,
        params: dict[str, Any],
        canvas: Canvas,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._generator = generator
        self._params = params
        self._canvas = canvas
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation — the generator checks this between iterations."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            if getattr(self._generator, "emits_multiple_layers", False):
                layer_specs = self._generator.generate_layers(
                    self._params,
                    self._canvas,
                    progress_callback=self._emit_progress,
                    cancelled_callback=self.is_cancelled,
                )
                if not self._cancelled:
                    self.layers_finished.emit(layer_specs)
            else:
                result: list[Polyline] = self._generator.generate(
                    self._params,
                    self._canvas,
                    progress_callback=self._emit_progress,
                    cancelled_callback=self.is_cancelled,
                )
                if not self._cancelled:
                    self.finished.emit(result)
                    # Emit any side-channel metadata written into params by the generator
                    # (e.g. "_depth_map_result" from ContourGenerator's AI depth mode).
                    metadata: dict = {}
                    depth_map = self._params.get("_depth_map_result")
                    if depth_map is not None:
                        metadata["depth_map"] = depth_map
                    if metadata:
                        self.metadata_ready.emit(metadata)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _emit_progress(self, percent: int) -> None:
        self.progress.emit(max(0, min(100, int(percent))))
