"""Multi-layer generators and the Color Separation line pump.

Pixel Art / Pointillist / CRT TV set ``emits_multiple_layers`` so
GeneratorWorker emits ``layers_finished`` instead of ``finished``. The
colorsep pump used to connect only ``finished`` + ``error``, so choosing one
of them left the buttons disabled and the undo macro open for the rest of
the session.  Two guards now exist:

1. They are filtered out of the Line-Art Algorithm combo.
2. If one reaches the worker launch anyway (e.g. a plugin), the pump
   connects ``layers_finished`` and flattens the specs into the mask layer.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from plottter.generators.base import Generator, LayerSpec
from plottter.gui.settings_panel._colorsep import _ColorSepMixin


# ----------------------------------------------------------------------
# Part 1: dropdown excludes multi-layer generators
# ----------------------------------------------------------------------


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


def test_dropdown_excludes_multilayer_generators(panel):
    from plottter.generators import get_generators_by_category

    combo = panel._color_sep_gen_combo
    listed = {combo.itemData(i) for i in range(combo.count())}
    image_gens = get_generators_by_category("image")
    multi = {g for g in image_gens if getattr(g, "emits_multiple_layers", False)}
    single = {g for g in image_gens if not getattr(g, "emits_multiple_layers", False)}

    assert multi, "expected at least one multi-layer image generator to exist"
    assert listed & multi == set()
    assert single <= listed
    assert all(not getattr(g, "emits_multiple_layers", False) for g in listed)


# ----------------------------------------------------------------------
# Part 2: worker guard flattens layer specs and drains the pump
# ----------------------------------------------------------------------


class _Sig:
    def __init__(self):
        self._slot = None

    def connect(self, slot):
        self._slot = slot

    def emit(self, *a):
        if self._slot:
            self._slot(*a)


class _FakeWorker:
    instances: list = []

    def __init__(self, gen, params, canvas):
        self.finished = _Sig()
        self.error = _Sig()
        self.layers_finished = _Sig()
        self.params = params
        _FakeWorker.instances.append(self)

    def start(self):
        pass

    def wait(self):
        pass


class _MultiGen(Generator):
    name = "Fake Multi"
    category = "image"
    emits_multiple_layers = True

    def get_parameters(self):
        return []

    def get_presets(self):
        return []

    def generate(self, params, canvas, progress_callback=None, cancelled_callback=None):
        return []

    def generate_layers(self, params, canvas, progress_callback=None, cancelled_callback=None):
        return []


class _Combo:
    def __init__(self, data):
        self._data = data

    def currentIndex(self):
        return 0

    def itemData(self, _i):
        return self._data

    def currentData(self):
        return None


class _Btn:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = v

    def setVisible(self, v):
        self.visible = v


class _Progress:
    def setMaximum(self, v):
        pass

    def setValue(self, v):
        pass

    def setVisible(self, v):
        self.visible = v


class _UndoStack:
    def __init__(self):
        self.begin = 0
        self.end = 0

    def beginMacro(self, _n):
        self.begin += 1

    def endMacro(self):
        self.end += 1


class _Ctrl:
    def __init__(self):
        self.undo_stack = _UndoStack()
        self.current_project = type("P", (), {"canvas": object()})()
        self.set_calls: list = []

    def set_layer_paths(self, lid, paths, _desc):
        self.set_calls.append((lid, paths))


class _Panel(_ColorSepMixin):
    def __init__(self):
        self._controller = _Ctrl()
        self._separated_layer_ids = ["L0"]
        self._layer_masks = {
            "L0": (np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=np.uint8))
        }
        self._color_sep_gen_combo = _Combo(_MultiGen)
        self._color_sep_preset_combo = _Combo(None)
        self._gen_lines_btn = _Btn()
        self._gen_lines_selected_btn = _Btn()
        self._color_sep_progress = _Progress()
        self._color_sep_cancel_btn = _Btn()

    def _image_fit_mode(self):
        return "fill"


def test_multilayer_worker_output_is_flattened_and_pump_drains():
    panel = _Panel()
    _FakeWorker.instances = []
    specs = [
        LayerSpec(name="a", color="#000", paths=[[(0.0, 0.0), (1.0, 0.0)]]),
        LayerSpec(name="b", color="#f00", paths=[[(0.0, 1.0), (1.0, 1.0)], [(2.0, 2.0), (3.0, 3.0)]]),
    ]
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeWorker):
        panel._on_generate_lines()
        assert not panel._gen_lines_btn.enabled
        _FakeWorker.instances[0].layers_finished.emit(specs)

    # All three paths landed on the single mask layer.
    assert panel._controller.set_calls == [("L0", [p for s in specs for p in s.paths])]
    # Pump drained: buttons back on, macro opened once and closed once.
    assert panel._gen_lines_btn.enabled
    assert panel._controller.undo_stack.begin == 1
    assert panel._controller.undo_stack.end == 1
    assert not panel._lines_active


def test_multilayer_worker_empty_specs_still_drains():
    panel = _Panel()
    _FakeWorker.instances = []
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeWorker):
        panel._on_generate_lines()
        _FakeWorker.instances[0].layers_finished.emit([])
    assert panel._controller.set_calls == [("L0", [])]
    assert panel._controller.undo_stack.end == 1
    assert panel._gen_lines_btn.enabled
