"""Tests for AxiDraw multi-layer plotting (per-layer SVG + dialog wiring).

Covers:
- ``project_to_layer_svg_list`` returns one SVG per selected layer.
- Visibility filter (``layer_ids=None`` skips hidden layers, explicit ids do not).
- ``project_to_svg_string`` honours the same explicit-overrides-visibility rule.
- ``AxiDrawDialog`` builds correct layer summaries and enables the pause
  checkbox only when 2+ layers will be plotted.
"""

from __future__ import annotations

import pytest

from plottter.export.axidraw import (
    project_to_layer_svg_list,
    project_to_svg_string,
)
from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(n_layers: int = 3, hide_indices: tuple[int, ...] = ()) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="MultiLayer", canvas=canvas)
    for i in range(n_layers):
        layer = Layer(name=f"L{i}", color=f"#{i*40:02x}{i*40:02x}{i*40:02x}")
        layer.paths = [[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]]
        if i in hide_indices:
            layer.visible = False
        proj.add_layer(layer)
    return proj


# ---------------------------------------------------------------------------
# project_to_layer_svg_list
# ---------------------------------------------------------------------------

class TestLayerSvgList:
    def test_returns_one_svg_per_visible_layer_when_ids_none(self):
        proj = _make_project(n_layers=3, hide_indices=(1,))
        jobs = project_to_layer_svg_list(proj, None, {"stroke_width_mm": 0.3})
        assert [name for name, _, _ in jobs] == ["L0", "L2"]
        assert all("<svg" in svg for _, _, svg in jobs)

    def test_explicit_ids_override_visibility(self):
        """Selecting a hidden layer by id should still produce its SVG."""
        proj = _make_project(n_layers=3, hide_indices=(0, 1, 2))
        hidden_id = proj.layers[1].id
        jobs = project_to_layer_svg_list(
            proj, [hidden_id], {"stroke_width_mm": 0.3}
        )
        assert len(jobs) == 1
        assert jobs[0][0] == "L1"

    def test_empty_when_no_visible_layers(self):
        proj = _make_project(n_layers=2, hide_indices=(0, 1))
        jobs = project_to_layer_svg_list(proj, None, {})
        assert jobs == []

    def test_each_svg_carries_layer_color(self):
        proj = _make_project(n_layers=2)
        proj.layers[0].color = "#ff0000"
        proj.layers[1].color = "#00ff00"
        jobs = project_to_layer_svg_list(proj, None, {})
        assert "#ff0000" in jobs[0][2]
        assert "#00ff00" in jobs[1][2]


# ---------------------------------------------------------------------------
# project_to_svg_string explicit-overrides-visibility rule
# ---------------------------------------------------------------------------

class TestCombinedSvgVisibilityRule:
    def test_ids_none_skips_hidden(self):
        proj = _make_project(n_layers=2, hide_indices=(1,))
        svg = project_to_svg_string(proj, None, {})
        assert "layer_L0" in svg
        assert "layer_L1" not in svg

    def test_explicit_ids_include_hidden_layer(self):
        proj = _make_project(n_layers=2, hide_indices=(1,))
        hidden_id = proj.layers[1].id
        svg = project_to_svg_string(proj, [hidden_id], {})
        assert "layer_L1" in svg
        assert "layer_L0" not in svg


# ---------------------------------------------------------------------------
# AxiDrawDialog: layer-scope summary + pause-checkbox enabled state
# ---------------------------------------------------------------------------

class TestAxiDrawDialogLayerScope:
    def test_all_visible_default_and_pause_enabled(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=3)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)

        assert dlg._scope_all_radio.isChecked()
        assert dlg._pause_check.isEnabled()
        summary = dlg._layer_summary_label.text()
        assert "3 layers" in summary

    def test_active_only_with_three_layers_disables_pause(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=3)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[1].id)
        qtbot.addWidget(dlg)

        dlg._scope_active_radio.setChecked(True)
        # _update_layer_summary should fire from the radio toggle signal.
        summary = dlg._layer_summary_label.text()
        assert "1 layer" in summary
        assert "L1" in summary
        assert not dlg._pause_check.isEnabled()
        assert not dlg._pause_check.isChecked()

    def test_active_only_plots_hidden_active_layer(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=2, hide_indices=(0,))
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)

        dlg._scope_active_radio.setChecked(True)
        layers = dlg._layers_to_plot()
        assert [lyr.name for lyr in layers] == ["L0"]

    def test_active_only_disabled_when_no_active_layer(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=2)
        dlg = AxiDrawDialog(proj, active_layer_id=None)
        qtbot.addWidget(dlg)

        assert not dlg._scope_active_radio.isEnabled()

    def test_single_visible_layer_disables_pause(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=3, hide_indices=(1, 2))
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)

        assert not dlg._pause_check.isEnabled()
        summary = dlg._layer_summary_label.text()
        assert "1 layer" in summary


# ---------------------------------------------------------------------------
# AxiDrawDialog: model selection persists across dialog instances
# ---------------------------------------------------------------------------

class TestAxiDrawModelPersistence:
    def test_model_selection_is_remembered(self, qtbot):
        from PyQt6.QtCore import QSettings

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        key = AxiDrawDialog._MODEL_SETTINGS_KEY
        settings = QSettings("Plottter", "Plottter")
        original = settings.value(key)
        try:
            proj = _make_project(n_layers=1)

            # First dialog: user picks "AxiDraw SE/A2" (index 5).
            dlg1 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg1)
            dlg1._model_combo.setCurrentIndex(5)

            # Second dialog should open with that choice already selected.
            dlg2 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg2)
            assert dlg2._model_combo.currentIndex() == 5
        finally:
            if original is None:
                settings.remove(key)
            else:
                settings.setValue(key, original)

    def test_defaults_to_v3_a3_when_unset(self, qtbot):
        from PyQt6.QtCore import QSettings

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        key = AxiDrawDialog._MODEL_SETTINGS_KEY
        settings = QSettings("Plottter", "Plottter")
        original = settings.value(key)
        try:
            settings.remove(key)
            proj = _make_project(n_layers=1)
            dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg)
            assert dlg._model_combo.currentIndex() == AxiDrawDialog._DEFAULT_MODEL_INDEX
        finally:
            if original is not None:
                settings.setValue(key, original)


# ---------------------------------------------------------------------------
# AxiDrawDialog: live pressure nudge
# ---------------------------------------------------------------------------

class TestAxiDrawPressureNudge:
    def _make_dialog(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        # Don't touch hardware: record the re-lower request instead of running it.
        dlg._lower_calls = []
        dlg._run_manual = lambda cmd: dlg._lower_calls.append(cmd)
        return dlg

    def test_more_pressure_lowers_pen_down_value_and_relowers(self, qtbot):
        dlg = self._make_dialog(qtbot)
        dlg._pen_pos_down.setValue(40)
        dlg._more_pressure_btn.click()
        assert dlg._pen_pos_down.value() == 40 - dlg._PRESSURE_STEP
        assert dlg._lower_calls == ["lower_pen"]

    def test_less_pressure_raises_pen_down_value(self, qtbot):
        dlg = self._make_dialog(qtbot)
        dlg._pen_pos_down.setValue(40)
        dlg._less_pressure_btn.click()
        assert dlg._pen_pos_down.value() == 40 + dlg._PRESSURE_STEP
        assert dlg._lower_calls == ["lower_pen"]

    def test_nudge_clamps_and_does_not_relower_at_limit(self, qtbot):
        dlg = self._make_dialog(qtbot)
        dlg._pen_pos_down.setValue(0)
        dlg._nudge_pressure(-dlg._PRESSURE_STEP)  # already at minimum
        assert dlg._pen_pos_down.value() == 0
        assert dlg._lower_calls == []


# ---------------------------------------------------------------------------
# AxiDrawDialog: default settings
# ---------------------------------------------------------------------------

class TestAxiDrawDefaults:
    def test_pen_down_delay_defaults_to_settle_time(self, qtbot):
        """A short pen-down delay avoids stroke-start skipping out of the box."""
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        assert dlg._pen_delay_down.value() == 125
        assert dlg._build_settings()["pen_delay_down"] == 125
