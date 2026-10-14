"""Tests for the Color Separation "Edit…" line-parameter override.

Covers:
* ParamForm — widget generation, value round-trip, visible_when.
* ColorSepParamsDialog — preset prefill, reset, save-as-preset.
* Panel wiring — override set/cleared, "(modified)" marker, metadata
  persistence and restore on project load.
* Worker precedence — override beats preset beats defaults.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QSpinBox

from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Preset,
    StringParam,
)


class _ToyGen(Generator):
    name = "Toy Lines"
    category = "image"
    uses_source_image = True

    def get_parameters(self):
        return [
            FloatParam("spacing", "Spacing", min=0.1, max=10.0, step=0.1, default=1.0),
            IntParam("count", "Count", min=1, max=100, default=10),
            ChoiceParam("style", "Style", choices=["plain", "wavy"], default="plain"),
            FloatParam(
                "amplitude", "Amplitude", min=0.0, max=5.0, default=0.5,
                visible_when={"style": ["wavy"]},
            ),
            BoolParam("invert", "Invert", default=False),
            StringParam("note", "Note", default="hi"),
        ]

    def get_presets(self):
        return [Preset("Dense", {"spacing": 0.3, "count": 50, "style": "wavy"})]

    def generate(self, params, canvas, progress_callback=None, cancelled_callback=None):
        return []


# ----------------------------------------------------------------------
# ParamForm
# ----------------------------------------------------------------------


def test_param_form_builds_widgets_and_reads_defaults(qtbot):
    from plottter.gui.widgets.param_form import ParamForm

    form = ParamForm(_ToyGen().get_parameters())
    qtbot.addWidget(form)

    assert isinstance(form.widget("spacing"), QDoubleSpinBox)
    assert isinstance(form.widget("count"), QSpinBox)
    assert isinstance(form.widget("style"), QComboBox)
    assert isinstance(form.widget("invert"), QCheckBox)
    assert form.values() == {
        "spacing": 1.0,
        "count": 10,
        "style": "plain",
        "amplitude": 0.5,
        "invert": False,
        "note": "hi",
    }


def test_param_form_set_values_round_trips_and_ignores_unknown(qtbot):
    from plottter.gui.widgets.param_form import ParamForm

    form = ParamForm(_ToyGen().get_parameters())
    qtbot.addWidget(form)
    form.set_values({"spacing": 2.5, "count": 3, "style": "wavy", "invert": True, "bogus": 1})
    vals = form.values()
    assert vals["spacing"] == 2.5
    assert vals["count"] == 3
    assert vals["style"] == "wavy"
    assert vals["invert"] is True
    assert "bogus" not in vals


def test_param_form_visible_when_follows_choice(qtbot):
    from plottter.gui.widgets.param_form import ParamForm

    form = ParamForm(_ToyGen().get_parameters())
    qtbot.addWidget(form)
    form.show()
    amp_label = form._labels["amplitude"]
    assert amp_label.isHidden()  # style defaults to "plain"
    form.widget("style").setCurrentText("wavy")
    assert not amp_label.isHidden()
    form.set_values({"style": "plain"})
    assert amp_label.isHidden()


# ----------------------------------------------------------------------
# ColorSepParamsDialog
# ----------------------------------------------------------------------


def _base_for(gen, preset):
    from plottter.gui.dialogs.colorsep_params_dialog import default_params_for

    base = default_params_for(gen)
    base.update(preset.params)
    return base


def test_dialog_prefills_preset_over_defaults_and_resets(qtbot):
    from plottter.gui.dialogs.colorsep_params_dialog import ColorSepParamsDialog

    gen = _ToyGen()
    preset = gen.get_presets()[0]
    base = _base_for(gen, preset)
    dlg = ColorSepParamsDialog(gen, preset.name, base)
    qtbot.addWidget(dlg)

    p = dlg.params()
    assert p["spacing"] == 0.3 and p["count"] == 50 and p["style"] == "wavy"
    assert p["invert"] is False  # default survives alongside preset values

    dlg._form.set_values({"spacing": 9.0})
    assert dlg.params()["spacing"] == 9.0
    dlg._on_reset()
    assert dlg.params()["spacing"] == 0.3


def test_dialog_initial_params_override_preset(qtbot):
    from plottter.gui.dialogs.colorsep_params_dialog import ColorSepParamsDialog

    gen = _ToyGen()
    preset = gen.get_presets()[0]
    base = _base_for(gen, preset)
    dlg = ColorSepParamsDialog(gen, preset.name, base, initial_params={"count": 7})
    qtbot.addWidget(dlg)
    assert dlg.params()["count"] == 7
    assert dlg.params()["spacing"] == 0.3


def test_dialog_save_as_preset_writes_user_preset_and_accepts(qtbot):
    from plottter.gui.dialogs.colorsep_params_dialog import ColorSepParamsDialog

    gen = _ToyGen()
    preset = gen.get_presets()[0]
    dlg = ColorSepParamsDialog(gen, preset.name, _base_for(gen, preset))
    qtbot.addWidget(dlg)
    dlg._form.set_values({"count": 11})

    saved: list = []
    with patch(
        "PyQt6.QtWidgets.QInputDialog.getText", return_value=("My Tweak", True)
    ), patch(
        "plottter.presets.user_presets.save_user_preset",
        side_effect=lambda gname, p: saved.append((gname, p)),
    ):
        dlg._on_save_as_preset()

    assert dlg.saved_preset_name == "My Tweak"
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert saved[0][0] == "Toy Lines"
    assert saved[0][1].name == "My Tweak"
    assert saved[0][1].params["count"] == 11


def test_dialog_save_as_preset_cancelled_is_noop(qtbot):
    from plottter.gui.dialogs.colorsep_params_dialog import ColorSepParamsDialog

    gen = _ToyGen()
    preset = gen.get_presets()[0]
    dlg = ColorSepParamsDialog(gen, preset.name, _base_for(gen, preset))
    qtbot.addWidget(dlg)
    with patch("PyQt6.QtWidgets.QInputDialog.getText", return_value=("", False)):
        dlg._on_save_as_preset()
    assert dlg.saved_preset_name is None
    assert dlg.result() != QDialog.DialogCode.Accepted


# ----------------------------------------------------------------------
# Panel wiring
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

    controller = ProjectController(_make_project())
    sp = SettingsPanel(controller)
    qtbot.addWidget(sp)
    sp.on_mode_changed("Color Separation")
    # Put the toy generator into the algorithm combo so tests are independent
    # of the real image generators' parameter sets.
    sp._color_sep_gen_combo.addItem(_ToyGen.name, _ToyGen)
    sp._color_sep_gen_combo.setCurrentIndex(sp._color_sep_gen_combo.count() - 1)
    return sp


class _FakeDialog:
    """Stand-in for ColorSepParamsDialog driven by class attributes."""

    accepted = True
    result_params: dict = {}
    saved_name: str | None = None
    last_kwargs: dict = {}

    def __init__(self, gen, preset_name, base, initial_params=None, parent=None):
        _FakeDialog.last_kwargs = {
            "gen": gen,
            "preset_name": preset_name,
            "base": dict(base),
            "initial_params": initial_params,
        }
        self.saved_preset_name = _FakeDialog.saved_name
        self.DialogCode = QDialog.DialogCode

    def exec(self):
        return QDialog.DialogCode.Accepted if _FakeDialog.accepted else QDialog.DialogCode.Rejected

    def params(self):
        return dict(_FakeDialog.result_params)


def _edit_with(panel, params, accepted=True, saved_name=None):
    _FakeDialog.accepted = accepted
    _FakeDialog.result_params = params
    _FakeDialog.saved_name = saved_name
    with patch(
        "plottter.gui.dialogs.colorsep_params_dialog.ColorSepParamsDialog", _FakeDialog
    ):
        panel._on_edit_color_sep_params()


def test_edit_sets_override_marker_and_metadata(panel):
    combo = panel._color_sep_preset_combo
    combo.setCurrentText("Dense")

    tweaked = {"spacing": 0.3, "count": 50, "style": "wavy", "amplitude": 2.0,
               "invert": False, "note": "hi"}
    _edit_with(panel, tweaked)

    assert _FakeDialog.last_kwargs["preset_name"] == "Dense"
    assert _FakeDialog.last_kwargs["base"]["spacing"] == 0.3
    assert _FakeDialog.last_kwargs["base"]["invert"] is False
    assert panel._color_sep_param_override == tweaked
    assert combo.currentText() == "Dense (modified)"

    meta = panel._controller.current_project.metadata["colorsep_line_params"]
    assert meta == {"generator": "Toy Lines", "preset": "Dense", "params": tweaked}


def test_edit_cancel_leaves_state_untouched(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    _edit_with(panel, {"count": 1}, accepted=False)
    assert panel._color_sep_param_override is None
    assert panel._color_sep_preset_combo.currentText() == "Dense"
    assert "colorsep_line_params" not in panel._controller.current_project.metadata


def test_edit_unchanged_values_do_not_create_override(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    _edit_with(panel, {"count": 0})  # prime last_kwargs with the base dict
    base = _FakeDialog.last_kwargs["base"]
    _edit_with(panel, dict(base))
    assert panel._color_sep_param_override is None
    assert panel._color_sep_preset_combo.currentText() == "Dense"


def test_changing_preset_clears_override(panel):
    combo = panel._color_sep_preset_combo
    combo.setCurrentText("Dense")
    _edit_with(panel, {"count": 2})
    assert panel._color_sep_param_override is not None

    combo.setCurrentIndex(0)  # "Default"
    assert panel._color_sep_param_override is None
    assert all(not combo.itemText(i).endswith("(modified)") for i in range(combo.count()))
    assert "colorsep_line_params" not in panel._controller.current_project.metadata


def test_changing_algorithm_clears_override(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    _edit_with(panel, {"count": 2})
    assert panel._color_sep_param_override is not None
    panel._color_sep_gen_combo.setCurrentIndex(0)
    assert panel._color_sep_param_override is None


def test_save_as_preset_selects_new_preset_and_drops_override(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    new_preset = Preset("My Tweak", {"spacing": 0.3, "count": 50, "style": "wavy", "amplitude": 2.0})
    with patch(
        "plottter.presets.user_presets.load_user_presets",
        side_effect=lambda gname: [new_preset] if gname == "Toy Lines" else [],
    ):
        _edit_with(panel, dict(new_preset.params), saved_name="My Tweak")
        combo = panel._color_sep_preset_combo
        assert combo.currentText() == "My Tweak"
        assert combo.currentData() == new_preset.params
    assert panel._color_sep_param_override is None
    assert "colorsep_line_params" not in panel._controller.current_project.metadata


def test_restore_from_project_metadata(panel):
    from plottter.models import Project, Canvas

    tweaked = {"spacing": 0.3, "count": 50, "style": "wavy", "amplitude": 3.0,
               "invert": True, "note": "x"}
    proj = Project(name="Loaded", canvas=Canvas.from_preset("A4"))
    proj.metadata["colorsep_line_params"] = {
        "generator": "Toy Lines", "preset": "Dense", "params": tweaked,
    }
    # Start from a different algorithm/preset so the restore has to move both.
    panel._color_sep_gen_combo.setCurrentIndex(0)
    panel._controller.load_project(proj)

    assert panel._color_sep_gen_combo.currentData() is _ToyGen
    assert panel._color_sep_preset_combo.currentText() == "Dense (modified)"
    assert panel._color_sep_param_override == tweaked
    # Restore must not rewrite the metadata it just read.
    assert proj.metadata["colorsep_line_params"]["params"] == tweaked


def test_loading_project_without_entry_clears_stale_override(panel):
    from plottter.models import Project, Canvas

    panel._color_sep_preset_combo.setCurrentText("Dense")
    _edit_with(panel, {"count": 2})
    assert panel._color_sep_param_override is not None

    panel._controller.load_project(Project(name="Clean", canvas=Canvas.from_preset("A4")))
    assert panel._color_sep_param_override is None
    assert not panel._color_sep_preset_combo.currentText().endswith("(modified)")


# ----------------------------------------------------------------------
# Worker precedence
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
        self.params = params
        _FakeWorker.instances.append(self)

    def start(self):
        pass

    def wait(self):
        pass


def _gen_lines_params(panel, override):
    """Run Generate Lines for one fake separated layer and return worker params."""
    panel._separated_layer_ids = ["L0"]
    panel._layer_masks = {"L0": (np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=np.uint8))}
    panel._color_sep_param_override = override
    _FakeWorker.instances = []
    with patch("plottter.gui.generator_worker.GeneratorWorker", _FakeWorker):
        panel._on_generate_lines()
        _FakeWorker.instances[0].finished.emit([])
    return _FakeWorker.instances[0].params


def test_worker_uses_override_over_preset(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    p = _gen_lines_params(panel, {"spacing": 7.0, "count": 2, "style": "plain"})
    assert p["spacing"] == 7.0 and p["count"] == 2 and p["style"] == "plain"
    assert "_source_image" in p and "image_fit_mode" in p


def test_worker_falls_back_to_preset_without_override(panel):
    panel._color_sep_preset_combo.setCurrentText("Dense")
    p = _gen_lines_params(panel, None)
    assert p["spacing"] == 0.3 and p["count"] == 50


def test_worker_falls_back_to_defaults_without_preset(panel):
    panel._color_sep_preset_combo.setCurrentIndex(0)  # Default
    p = _gen_lines_params(panel, None)
    assert p["spacing"] == 1.0 and p["count"] == 10
