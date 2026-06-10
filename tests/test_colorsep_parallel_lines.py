"""Tests for the parallel line-generation pump in the Color Separation panel.

Line art for the separated layers now runs several GeneratorWorkers
concurrently (capped) instead of strictly one-at-a-time. These tests drive the
pump with a fake worker (fired synchronously) to verify: the concurrency cap is
respected, every layer is processed, and the undo macro is closed exactly once.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from plottter.gui.settings_panel._colorsep import _ColorSepMixin


class _Sig:
    def __init__(self) -> None:
        self._slot = None

    def connect(self, slot) -> None:
        self._slot = slot

    def emit(self, *args) -> None:
        if self._slot is not None:
            self._slot(*args)


class _FakeGenWorker:
    """Captures connections and lets the test fire completion synchronously."""

    instances: list = []

    def __init__(self, gen, params, canvas) -> None:
        self.finished = _Sig()
        self.error = _Sig()
        self.params = params
        self.started = False
        _FakeGenWorker.instances.append(self)

    def start(self) -> None:
        self.started = True

    def wait(self) -> None:
        pass


class _FakeGen:
    name = "Fake"

    def get_parameters(self):
        return []


class _Combo:
    def __init__(self, data) -> None:
        self._data = data

    def currentIndex(self) -> int:
        return 0

    def itemData(self, _idx) -> object:
        return self._data

    def currentData(self):
        return None  # no preset


class _Btn:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, v) -> None:
        self.enabled = v


class _Progress:
    def __init__(self) -> None:
        self.value = 0
        self.maximum = 0
        self.visible = False

    def setMaximum(self, v) -> None:
        self.maximum = v

    def setValue(self, v) -> None:
        self.value = v

    def setVisible(self, v) -> None:
        self.visible = v


class _UndoStack:
    def __init__(self) -> None:
        self.begin = 0
        self.end = 0

    def beginMacro(self, _name) -> None:
        self.begin += 1

    def endMacro(self) -> None:
        self.end += 1


class _Project:
    canvas = object()


class _Ctrl:
    def __init__(self) -> None:
        self.undo_stack = _UndoStack()
        self.current_project = _Project()
        self.set_calls: list = []

    def set_layer_paths(self, lid, paths, _desc) -> None:
        self.set_calls.append((lid, paths))


class _Panel(_ColorSepMixin):
    def __init__(self, n_layers: int) -> None:
        self._controller = _Ctrl()
        self._separated_layer_ids = [f"L{i}" for i in range(n_layers)]
        mask = np.zeros((8, 8), dtype=bool)
        src = np.zeros((8, 8), dtype=np.uint8)
        self._layer_masks = {lid: (mask, src) for lid in self._separated_layer_ids}
        self._color_sep_gen_combo = _Combo(_FakeGen)
        self._color_sep_preset_combo = _Combo(None)
        self._gen_lines_btn = _Btn()
        self._gen_lines_selected_btn = _Btn()
        self._color_sep_progress = _Progress()

    def _image_fit_mode(self) -> str:
        return "fill"


def _run(n_layers: int):
    panel = _Panel(n_layers)
    _FakeGenWorker.instances = []
    with patch(
        "plottter.gui.generator_worker.GeneratorWorker", _FakeGenWorker
    ):
        panel._on_generate_lines()
        cap = panel._line_worker_cap()
        # Initially, min(cap, n) workers should be running concurrently.
        started = [w for w in _FakeGenWorker.instances if w.started]
        assert len(started) == min(cap, n_layers)

        # Fire completions until all are drained; the pump should keep the
        # number of new launches flowing and finalize exactly once.
        fired = 0
        while fired < len(_FakeGenWorker.instances):
            w = _FakeGenWorker.instances[fired]
            w.finished.emit([[(0.0, 0.0), (1.0, 1.0)]])
            fired += 1
    return panel


def test_all_layers_processed_and_macro_closed_once():
    panel = _run(5)
    # One worker created per layer, each produced one set_layer_paths call.
    assert len(_FakeGenWorker.instances) == 5
    assert len(panel._controller.set_calls) == 5
    assert panel._controller.undo_stack.begin == 1
    assert panel._controller.undo_stack.end == 1
    # Controls re-enabled and progress hidden at the end.
    assert panel._gen_lines_btn.enabled
    assert not panel._color_sep_progress.visible


def test_concurrency_capped():
    panel = _Panel(10)
    _FakeGenWorker.instances = []
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeGenWorker):
        panel._on_generate_lines()
        cap = panel._line_worker_cap()
        # No more than `cap` workers run before any completion fires.
        assert len(_FakeGenWorker.instances) == min(cap, 10)


def test_single_layer_runs_and_finalizes():
    panel = _run(1)
    assert len(_FakeGenWorker.instances) == 1
    assert len(panel._controller.set_calls) == 1
    assert panel._controller.undo_stack.end == 1
