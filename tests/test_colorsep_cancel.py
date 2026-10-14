"""Cancel button on the Color Separation page.

Image to Lines has had a Cancel button for its single GeneratorWorker; the
Color Separation page had none, so a slow line-art algorithm run across the
separated layers could only be waited out.  One button now covers both
background phases of the page:

* **Generate Lines** — the queue is dropped, running workers are asked to
  stop, the undo macro closes once, and late signals from cancelled workers
  are ignored.
* **Separate into Layers** — the NumPy work can't be interrupted, so the
  worker is orphaned (result discarded on arrival) and the controls come
  back immediately; a fresh separation started meanwhile still applies.

Part 1 drives the pump with stub workers; Part 2 runs real QThreads.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from plottter.generators.base import Generator
from plottter.gui.settings_panel._colorsep import _ColorSepMixin


# ----------------------------------------------------------------------
# Part 1: pump bookkeeping with stub workers
# ----------------------------------------------------------------------


class _Sig:
    def __init__(self) -> None:
        self._slot = None

    def connect(self, slot) -> None:
        self._slot = slot

    def emit(self, *args) -> None:
        if self._slot is not None:
            self._slot(*args)


class _FakeWorker:
    instances: list = []

    def __init__(self, gen, params, canvas) -> None:
        self.finished = _Sig()
        self.error = _Sig()
        self.layers_finished = _Sig()
        self.cancelled = False
        self.running = True
        _FakeWorker.instances.append(self)

    def start(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    def isRunning(self) -> bool:
        return self.running


class _FakeGen:
    name = "Fake"

    def get_parameters(self):
        return []


class _Combo:
    def __init__(self, data) -> None:
        self._data = data

    def currentIndex(self) -> int:
        return 0

    def itemData(self, _idx):
        return self._data

    def currentData(self):
        return None


class _Btn:
    def __init__(self) -> None:
        self.enabled = True
        self.visible = False

    def setEnabled(self, v) -> None:
        self.enabled = v

    def setVisible(self, v) -> None:
        self.visible = v


class _Progress(_Btn):
    def setMaximum(self, v) -> None:
        pass

    def setValue(self, v) -> None:
        self.value = v


class _UndoStack:
    def __init__(self) -> None:
        self.begin = 0
        self.end = 0

    def beginMacro(self, _name) -> None:
        self.begin += 1

    def endMacro(self) -> None:
        self.end += 1


class _Ctrl:
    def __init__(self) -> None:
        self.undo_stack = _UndoStack()
        self.current_project = type("P", (), {"canvas": object()})()
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
        self._color_sep_cancel_btn = _Btn()
        self._separate_btn = _Btn()
        self._separation_worker = None
        self._ai_segment_worker = None

    def _image_fit_mode(self) -> str:
        return "fill"

    def _line_worker_cap(self) -> int:
        return 2


@pytest.fixture
def running_pump():
    """Six layers queued, two workers in flight, first one already finished."""
    panel = _Panel(6)
    _FakeWorker.instances = []
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeWorker):
        panel._on_generate_lines()
        _FakeWorker.instances[0].finished.emit([[(0.0, 0.0), (1.0, 1.0)]])
        yield panel


def test_cancel_button_shown_only_while_lines_run(running_pump):
    panel = running_pump
    assert panel._color_sep_cancel_btn.visible
    panel._on_color_sep_cancel()
    assert not panel._color_sep_cancel_btn.visible
    assert not panel._color_sep_progress.visible


def test_cancel_drops_queue_and_cancels_active_workers(running_pump):
    panel = running_pump
    assert len(_FakeWorker.instances) == 3  # 2 initial + 1 pumped after a finish
    panel._on_color_sep_cancel()

    assert panel._lines_queue == []
    assert panel._lines_active == {}
    assert [w.cancelled for w in _FakeWorker.instances] == [False, True, True]
    # No further workers were launched for the remaining queued layers.
    assert len(_FakeWorker.instances) == 3
    # Cancelled workers are parked until their thread exits.
    assert set(panel._color_sep_orphans) == set(_FakeWorker.instances[1:])


def test_cancel_keeps_finished_layers_and_closes_macro_once(running_pump):
    panel = running_pump
    panel._on_color_sep_cancel()
    assert panel._controller.set_calls == [("L0", [[(0.0, 0.0), (1.0, 1.0)]])]
    assert panel._controller.undo_stack.begin == 1
    assert panel._controller.undo_stack.end == 1
    assert panel._gen_lines_btn.enabled
    assert panel._gen_lines_selected_btn.enabled
    assert not panel._lines_running


def test_late_signals_from_cancelled_workers_are_ignored(running_pump):
    panel = running_pump
    panel._on_color_sep_cancel()
    w1, w2 = _FakeWorker.instances[1], _FakeWorker.instances[2]
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as warn:
        w1.finished.emit([[(5.0, 5.0), (6.0, 6.0)]])
        w2.error.emit("boom")
        w2.layers_finished.emit([])
    warn.assert_not_called()
    assert len(panel._controller.set_calls) == 1  # still only L0
    assert panel._controller.undo_stack.end == 1  # macro not closed again


def test_second_cancel_is_a_no_op(running_pump):
    panel = running_pump
    panel._on_color_sep_cancel()
    panel._on_color_sep_cancel()
    assert panel._controller.undo_stack.end == 1


def test_new_run_after_cancel_works_normally(running_pump):
    panel = running_pump
    panel._on_color_sep_cancel()
    before = len(_FakeWorker.instances)
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeWorker):
        panel._on_generate_lines()
        i = before  # the pump launches new workers as earlier ones finish
        while i < len(_FakeWorker.instances):
            _FakeWorker.instances[i].finished.emit([])
            i += 1
    assert not panel._lines_running
    assert panel._controller.undo_stack.begin == 2
    assert panel._controller.undo_stack.end == 2
    assert len(panel._controller.set_calls) == 1 + 6


def test_cancel_with_nothing_running_only_resets_ui():
    panel = _Panel(1)
    panel._separate_btn.enabled = False
    panel._color_sep_cancel_btn.visible = True
    panel._on_color_sep_cancel()
    assert panel._separate_btn.enabled
    assert not panel._color_sep_cancel_btn.visible
    assert panel._controller.undo_stack.end == 0


# ----------------------------------------------------------------------
# Part 2: real QThreads through the real panel
# ----------------------------------------------------------------------


class _SlowGen(Generator):
    """Spins until cancelled (or ~5 s), polling the flag like real generators."""

    name = "Slow Test Gen"
    category = "image"
    description = ""

    def get_parameters(self):
        return []

    def get_presets(self):
        return []

    def generate(self, params, canvas, progress_callback=None, cancelled_callback=None):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cancelled_callback and cancelled_callback():
                return []
            time.sleep(0.01)
        return [[(0.0, 0.0), (1.0, 1.0)]]


def _make_project():
    from plottter.models import Canvas, Layer, Project

    proj = Project(name="Test", canvas=Canvas.from_preset("A4"))
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def panel(qapp, qtbot):
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    sp = SettingsPanel(ProjectController(_make_project()))
    qtbot.addWidget(sp)
    sp.on_mode_changed("Color Separation")
    return sp


def _gradient(h=120, w=160):
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(ramp[None, :], h, axis=0)[:, :, None].repeat(3, axis=2)


def _separate(panel, qtbot, n=3):
    panel._raw_image = _gradient()
    panel._color_sep_method_combo.setCurrentText("Luminance")
    panel._color_sep_num_colors_spin.setValue(n)
    with patch("PyQt6.QtWidgets.QMessageBox.information"):
        panel._on_separate()
        qtbot.waitUntil(lambda: len(panel._separated_layer_ids) == n, timeout=10000)


def test_cancel_stops_real_line_workers_and_reaps_threads(panel, qtbot):
    _separate(panel, qtbot)
    panel._color_sep_gen_combo.addItem(_SlowGen.name, _SlowGen)
    panel._color_sep_gen_combo.setCurrentIndex(panel._color_sep_gen_combo.count() - 1)

    panel._on_generate_lines()
    assert not panel._color_sep_cancel_btn.isHidden()
    assert panel._lines_running
    qtbot.waitUntil(lambda: all(w.isRunning() for w in panel._lines_active.values()))
    workers = list(panel._lines_active.values())
    assert workers

    panel._color_sep_cancel_btn.click()

    assert not panel._lines_running
    assert panel._gen_lines_btn.isEnabled()
    assert all(w.is_cancelled() for w in workers)
    # Threads exit promptly (they poll the flag) and the reaper drops them.
    qtbot.waitUntil(lambda: not panel._color_sep_orphans, timeout=5000)
    assert all(not w.isRunning() for w in workers)
    # Nothing was written to the layers and the macro is closed: the stack
    # holds the separation plus the (empty) Generate Lines macro, and a new
    # push lands on top instead of inside a still-open macro.
    for lid in panel._separated_layer_ids:
        assert panel._controller.current_project.get_layer(lid).paths == []
    stack = panel._controller.undo_stack
    assert stack.index() == 2
    panel._controller.set_layer_paths(panel._separated_layer_ids[0], [[(0.0, 0.0), (2.0, 2.0)]], "x")
    assert stack.index() == 3


def test_cancel_discards_slow_separation_result(panel, qtbot):
    from plottter.gui.settings_panel import _colorsep as mod

    real = mod._compute_separation
    calls = []

    def slow(*a, **kw):
        calls.append(1)
        time.sleep(0.6)
        return real(*a, **kw)

    panel._raw_image = _gradient()
    panel._color_sep_method_combo.setCurrentText("Luminance")
    panel._color_sep_num_colors_spin.setValue(3)
    with patch.object(mod, "_compute_separation", slow), patch(
        "PyQt6.QtWidgets.QMessageBox.information"
    ) as info:
        panel._on_separate()
        assert not panel._separate_btn.isEnabled()
        worker = panel._separation_worker
        assert worker is not None and worker.isRunning()

        panel._color_sep_cancel_btn.click()
        assert panel._separate_btn.isEnabled()
        assert panel._separation_worker is None
        assert worker in panel._color_sep_orphans

        qtbot.waitUntil(lambda: not panel._color_sep_orphans, timeout=5000)
        qtbot.wait(50)  # let any queued finished() delivery run
    assert calls == [1]
    assert panel._separated_layer_ids == []
    info.assert_not_called()


def test_separation_started_after_cancel_still_applies(panel, qtbot):
    from plottter.gui.settings_panel import _colorsep as mod

    real = mod._compute_separation
    n_calls = [0]

    def first_slow(*a, **kw):
        n_calls[0] += 1
        if n_calls[0] == 1:
            time.sleep(0.6)
        return real(*a, **kw)

    panel._raw_image = _gradient()
    panel._color_sep_method_combo.setCurrentText("Luminance")
    panel._color_sep_num_colors_spin.setValue(3)
    with patch.object(mod, "_compute_separation", first_slow), patch(
        "PyQt6.QtWidgets.QMessageBox.information"
    ) as info:
        panel._on_separate()
        stale = panel._separation_worker
        panel._color_sep_cancel_btn.click()

        panel._color_sep_num_colors_spin.setValue(4)
        panel._on_separate()
        assert panel._separation_worker is not stale
        qtbot.waitUntil(lambda: len(panel._separated_layer_ids) == 4, timeout=10000)
        qtbot.waitUntil(lambda: not panel._color_sep_orphans, timeout=5000)
        qtbot.wait(50)
    # The stale 3-band result never replaced the live 4-band layers.
    assert len(panel._separated_layer_ids) == 4
    assert info.call_count == 1
