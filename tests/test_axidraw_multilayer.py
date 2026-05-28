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


class TestLayerNameXmlIds:
    """Layer names with characters invalid in XML ids (e.g. the Map generator's
    "Roads (minor)") must not crash svgwrite's id validation."""

    def test_parenthesised_layer_name_produces_valid_id(self):
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Map", canvas=canvas)
        proj.add_layer(Layer(name="Roads (minor)", color="#444444",
                             paths=[[(10.0, 10.0), (50.0, 50.0)]]))
        proj.add_layer(Layer(name="Roads (major)", color="#000000",
                             paths=[[(20.0, 20.0), (60.0, 40.0)]]))
        ids = [l.id for l in proj.layers]

        # Must not raise (regression: parentheses previously hit svgwrite's
        # id validator and crashed the AxiDraw plot).
        svg = project_to_svg_string(proj, ids, {})
        assert 'id="layer_Roads_minor"' in svg
        assert 'id="layer_Roads_major"' in svg
        assert "(" not in svg.split("<path", 1)[0]  # no '(' in the <g> id attrs

    def test_safe_xml_id_allowlist(self):
        from plottter.export.axidraw import _safe_xml_id

        assert _safe_xml_id("Roads (minor)") == "Roads_minor"
        assert _safe_xml_id("© / x*?:") == "x"
        assert _safe_xml_id("a.b-c_d") == "a.b-c_d"  # valid chars preserved
        assert _safe_xml_id("(((") == ""             # caller's 'layer_' prefix keeps it valid


# ---------------------------------------------------------------------------
# project_to_svg_string explicit-overrides-visibility rule
# ---------------------------------------------------------------------------

class TestPlotOrientation:
    def test_no_flip_keeps_coordinates(self):
        proj = _make_project(n_layers=1)
        proj.layers[0].paths = [[(0.0, 0.0), (10.0, 20.0)]]
        svg = project_to_svg_string(proj, None, {})
        assert "0.000,0.000" in svg
        assert "10.000,20.000" in svg

    def test_flip_x_mirrors_horizontally(self):
        proj = _make_project(n_layers=1)  # A4: 210 x 297
        w = proj.canvas.width_mm
        proj.layers[0].paths = [[(0.0, 0.0), (10.0, 20.0)]]
        svg = project_to_svg_string(proj, None, {"flip_x": True})
        # x is mirrored (width - x); y is unchanged.
        assert f"{w - 0.0:.3f},0.000" in svg
        assert f"{w - 10.0:.3f},20.000" in svg

    def test_flip_y_mirrors_vertically(self):
        proj = _make_project(n_layers=1)  # A4: 210 x 297
        h = proj.canvas.height_mm
        proj.layers[0].paths = [[(0.0, 0.0), (10.0, 20.0)]]
        svg = project_to_svg_string(proj, None, {"flip_y": True})
        assert f"0.000,{h - 0.0:.3f}" in svg
        assert f"10.000,{h - 20.0:.3f}" in svg

    def test_rotate_180_flips_both(self):
        proj = _make_project(n_layers=1)
        w, h = proj.canvas.width_mm, proj.canvas.height_mm
        proj.layers[0].paths = [[(10.0, 20.0), (30.0, 40.0)]]
        svg = project_to_svg_string(proj, None, {"flip_x": True, "flip_y": True})
        assert f"{w - 10.0:.3f},{h - 20.0:.3f}" in svg
        assert f"{w - 30.0:.3f},{h - 40.0:.3f}" in svg

    def test_per_layer_svg_also_flips(self):
        proj = _make_project(n_layers=1)
        w = proj.canvas.width_mm
        proj.layers[0].paths = [[(0.0, 5.0), (10.0, 5.0)]]
        jobs = project_to_layer_svg_list(proj, None, {"flip_x": True})
        assert f"{w - 0.0:.3f},5.000" in jobs[0][2]


class TestPlotBedSize:
    def test_no_bed_uses_canvas_size(self):
        proj = _make_project(n_layers=1)  # A4: 210 x 297
        svg = project_to_svg_string(proj, None, {})
        assert "viewBox=\"0 0 210.0 297.0\"" in svg

    def test_bed_size_pads_document_but_keeps_coords(self):
        proj = _make_project(n_layers=1)  # A4 canvas: 210 x 297
        proj.layers[0].paths = [[(0.0, 0.0), (10.0, 20.0)]]
        # Bed = A2 (420 x 594): document grows, coordinates are unchanged.
        svg = project_to_svg_string(
            proj, None, {"bed_width_mm": 420.0, "bed_height_mm": 594.0}
        )
        assert "viewBox=\"0 0 420.0 594.0\"" in svg
        assert "0.000,0.000" in svg          # top-left stays at the origin
        assert "10.000,20.000" in svg        # not scaled or shifted

    def test_bed_never_shrinks_below_canvas(self):
        proj = _make_project(n_layers=1)  # A4 canvas: 210 x 297
        # A smaller bed must not clip the canvas — falls back to canvas size.
        svg = project_to_svg_string(
            proj, None, {"bed_width_mm": 100.0, "bed_height_mm": 100.0}
        )
        assert "viewBox=\"0 0 210.0 297.0\"" in svg

    def test_per_layer_svg_honours_bed_size(self):
        proj = _make_project(n_layers=1)
        jobs = project_to_layer_svg_list(
            proj, None, {"bed_width_mm": 420.0, "bed_height_mm": 594.0}
        )
        assert "viewBox=\"0 0 420.0 594.0\"" in jobs[0][2]


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


class TestAxiDrawOrientation:
    def test_flip_x_choice_maps_into_settings(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        dlg._orientation_combo.setCurrentIndex(1)  # Flip horizontal (X)
        s = dlg._build_settings()
        assert s["flip_x"] is True
        assert s["flip_y"] is False

    def test_rotate_180_maps_both_flips(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        dlg._orientation_combo.setCurrentIndex(3)  # Rotate 180°
        s = dlg._build_settings()
        assert s["flip_x"] is True
        assert s["flip_y"] is True

    def test_orientation_is_remembered(self, qtbot):
        from PyQt6.QtCore import QSettings

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        key = AxiDrawDialog._ORIENTATION_SETTINGS_KEY
        settings = QSettings("Plottter", "Plottter")
        original = settings.value(key)
        try:
            proj = _make_project(n_layers=1)
            dlg1 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg1)
            dlg1._orientation_combo.setCurrentIndex(1)  # Flip X

            dlg2 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg2)
            assert dlg2._orientation_combo.currentIndex() == 1
        finally:
            if original is None:
                settings.remove(key)
            else:
                settings.setValue(key, original)


class TestAxiDrawBedSize:
    def test_match_canvas_default_yields_no_bed(self, qtbot):
        from PyQt6.QtCore import QSettings

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        key = AxiDrawDialog._BED_SETTINGS_KEY
        settings = QSettings("Plottter", "Plottter")
        original = settings.value(key)
        try:
            settings.remove(key)
            proj = _make_project(n_layers=1)
            dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg)
            assert dlg._bed_combo.currentText() == AxiDrawDialog._BED_MATCH_CANVAS
            s = dlg._build_settings()
            assert s["bed_width_mm"] is None
            assert s["bed_height_mm"] is None
        finally:
            if original is not None:
                settings.setValue(key, original)

    def test_selecting_a2_maps_to_bed_dimensions(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        from plottter.models.canvas import PAPER_PRESETS

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        idx = dlg._bed_combo.findText("A2")
        assert idx >= 0
        dlg._bed_combo.setCurrentIndex(idx)
        s = dlg._build_settings()
        assert (s["bed_width_mm"], s["bed_height_mm"]) == (
            float(PAPER_PRESETS["A2"][0]),
            float(PAPER_PRESETS["A2"][1]),
        )

    def test_bed_choice_is_remembered(self, qtbot):
        from PyQt6.QtCore import QSettings

        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        key = AxiDrawDialog._BED_SETTINGS_KEY
        settings = QSettings("Plottter", "Plottter")
        original = settings.value(key)
        try:
            proj = _make_project(n_layers=1)
            dlg1 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg1)
            dlg1._bed_combo.setCurrentIndex(dlg1._bed_combo.findText("A2"))

            dlg2 = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
            qtbot.addWidget(dlg2)
            assert dlg2._bed_combo.currentText() == "A2"
        finally:
            if original is None:
                settings.remove(key)
            else:
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


class TestAxiDrawReturnHome:
    def test_return_home_button_issues_walk_home(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        # Record the command instead of touching hardware.
        calls = []
        dlg._run_manual = lambda cmd: calls.append(cmd)
        dlg._return_home_btn.click()
        assert calls == ["walk_home"]

    def test_return_home_is_a_managed_manual_button(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        # Must be disabled along with the other manual buttons during commands.
        assert dlg._return_home_btn in dlg._manual_buttons
        dlg._set_manual_buttons_enabled(False)
        assert not dlg._return_home_btn.isEnabled()


# ---------------------------------------------------------------------------
# _PlotWorker software pause
# ---------------------------------------------------------------------------

class TestPlotWorkerPause:
    def test_pause_calls_transmit_pause_request(self):
        from plottter.gui.dialogs.axidraw_dialog import _PlotWorker

        class FakeAd:
            def __init__(self):
                self.paused = False

            def transmit_pause_request(self):
                self.paused = True

        worker = _PlotWorker("<svg/>", {})
        fake = FakeAd()
        worker._store_ad(fake)
        worker.pause()
        assert fake.paused is True

    def test_pause_before_plotting_is_noop(self):
        from plottter.gui.dialogs.axidraw_dialog import _PlotWorker

        worker = _PlotWorker("<svg/>", {})
        worker.pause()  # _ad is None — must not raise


# ---------------------------------------------------------------------------
# AxiDrawDialog: pause / resume UI state machine
# ---------------------------------------------------------------------------

class TestAxiDrawPauseResume:
    def _dlg(self, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

        proj = _make_project(n_layers=1)
        dlg = AxiDrawDialog(proj, active_layer_id=proj.layers[0].id)
        qtbot.addWidget(dlg)
        return dlg

    def test_initial_state_hides_pause_and_resume(self, qtbot):
        dlg = self._dlg(qtbot)
        assert dlg._pause_btn.isHidden()
        assert dlg._resume_btn.isHidden()
        assert dlg._plot_btn.isEnabled()

    def test_plotting_state_shows_pause(self, qtbot):
        dlg = self._dlg(qtbot)
        dlg._set_plot_ui_state("plotting", allow_pause=True)
        assert not dlg._pause_btn.isHidden()
        assert dlg._pause_btn.isEnabled()
        assert dlg._resume_btn.isHidden()
        assert not dlg._plot_btn.isEnabled()

    def test_multilayer_plotting_hides_pause(self, qtbot):
        dlg = self._dlg(qtbot)
        dlg._set_plot_ui_state("plotting", allow_pause=False)
        assert dlg._pause_btn.isHidden()

    def test_paused_state_stores_resume_and_shows_resume_button(self, qtbot, monkeypatch):
        import plottter.gui.dialogs.axidraw_dialog as mod
        monkeypatch.setattr(mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))
        dlg = self._dlg(qtbot)
        dlg._on_plot_paused("<svg>resume-data</svg>")
        assert dlg._resume_svg == "<svg>resume-data</svg>"
        assert not dlg._resume_btn.isHidden()
        assert dlg._resume_btn.isEnabled()
        assert dlg._pause_btn.isHidden()
        assert not dlg._plot_btn.isEnabled()

    def test_pause_without_resume_data_returns_to_idle(self, qtbot, monkeypatch):
        import plottter.gui.dialogs.axidraw_dialog as mod
        monkeypatch.setattr(mod.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        dlg = self._dlg(qtbot)
        dlg._on_plot_paused("")
        assert dlg._resume_svg is None
        assert dlg._resume_btn.isHidden()
        assert dlg._plot_btn.isEnabled()

    def test_finished_clears_resume_and_returns_idle(self, qtbot, monkeypatch):
        import plottter.gui.dialogs.axidraw_dialog as mod
        monkeypatch.setattr(mod.QMessageBox, "information", staticmethod(lambda *a, **k: None))
        dlg = self._dlg(qtbot)
        dlg._resume_svg = "leftover"
        dlg._on_plot_finished()
        assert dlg._resume_svg is None
        assert dlg._plot_btn.isEnabled()
        assert dlg._resume_btn.isHidden()

    def test_resume_is_noop_without_resume_svg(self, qtbot):
        dlg = self._dlg(qtbot)
        dlg._resume_svg = None
        dlg._on_resume()
        assert dlg._worker is None


# ---------------------------------------------------------------------------
# plot_svg_string return value (preview mode, no hardware)
# ---------------------------------------------------------------------------

class TestPlotSvgStringOutcome:
    def test_preview_run_reports_not_paused(self):
        pytest.importorskip("pyaxidraw")
        from plottter.export.axidraw import PlotOutcome, plot_svg_string

        proj = _make_project(n_layers=1)
        proj.layers[0].paths = [[(10.0, 10.0), (30.0, 30.0)]]
        svg = project_to_svg_string(proj, None, {})
        outcome = plot_svg_string(svg, {"preview": True})
        assert isinstance(outcome, PlotOutcome)
        assert outcome.paused is False
        assert outcome.resume_svg is None


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
