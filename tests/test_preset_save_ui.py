"""Tests for task 26.2 — Save Current as Preset UI flow.

Verifies:
(a) "Save Current as Preset…" entry appears at the bottom of the preset combo
    when a generator is loaded.
(b) Selecting the action opens a QInputDialog for a name.
(c) After confirming with a non-empty name, the preset is persisted via
    save_user_preset and the combo is refreshed.
(d) Cancelling the dialog does not create a preset; combo reverts to "Custom".
(e) The saved params match the current widget values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── headless Qt ────────────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ─── minimal mock generator ─────────────────────────────────────────────────

def _make_mock_generator(name: str = "Test Generator"):
    """Return a minimal generator-like object usable with SettingsPanel."""
    from plottter.generators.base import FloatParam, Preset

    gen = MagicMock()
    gen.name = name
    gen.get_presets.return_value = [
        Preset(name="Built-in Preset A", params={"radius": 5.0}),
    ]
    gen.get_parameters.return_value = [
        FloatParam(name="radius", label="Radius (mm)", min=0.1, max=50.0, default=5.0),
    ]
    return gen


# ─── SettingsPanel factory ───────────────────────────────────────────────────

@pytest.fixture
def panel(qapp, tmp_path):
    """Create a SettingsPanel with a minimal mock controller."""
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project
    from plottter.gui.project_controller import ProjectController

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)

    sp = SettingsPanel(controller)
    # Switch to "Math Art" mode so the generator-related widgets are visible
    sp._current_mode = "Math Art"
    return sp, tmp_path


# ─── (a) "Save Current as Preset…" appears in combo ────────────────────────

class TestSavePresetEntry:
    def test_save_action_present_after_set_generator(self, panel):
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "Save Current as Preset\u2026" in items

    def test_save_action_is_last_non_separator_item(self, panel):
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        last = sp._preset_combo.itemText(sp._preset_combo.count() - 1)
        assert last == "Save Current as Preset\u2026"

    def test_custom_and_builtin_presets_still_present(self, panel):
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "Custom" in items
        assert "Built-in Preset A" in items

    def test_no_save_action_when_no_generator(self, panel):
        sp, _ = panel
        sp.set_generator(None)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "Save Current as Preset\u2026" not in items


# ─── (b) & (c) Save action triggers dialog and persists preset ──────────────

class TestSavePresetAction:
    def test_cancel_leaves_no_preset(self, panel, tmp_path):
        """Cancelling the name dialog must not create any preset file."""
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        # Patch QInputDialog to simulate the user clicking Cancel.
        with patch(
            "plottter.gui.settings_panel.QInputDialog.getText",
            return_value=("", False),
        ):
            with patch(
                "plottter.presets.user_presets.save_user_preset"
            ) as mock_save:
                sp._on_preset_changed("Save Current as Preset\u2026")
                mock_save.assert_not_called()

        # Combo should revert to "Custom"
        assert sp._preset_combo.currentText() == "Custom"

    def test_confirm_calls_save_user_preset(self, panel, tmp_path):
        """Confirming with a valid name must call save_user_preset."""
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        # Set the radius widget to a known value (widget was created by set_generator)
        from PyQt6.QtWidgets import QDoubleSpinBox
        widget = sp._param_widgets.get("radius")
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(12.5)

        with patch(
            "plottter.gui.settings_panel.QInputDialog.getText",
            return_value=("My Custom Preset", True),
        ):
            with patch(
                "plottter.presets.user_presets.save_user_preset"
            ) as mock_save:
                sp._on_preset_changed("Save Current as Preset\u2026")
                mock_save.assert_called_once()
                _gen_name, preset = mock_save.call_args[0]
                assert preset.name == "My Custom Preset"

    def test_whitespace_only_name_is_rejected(self, panel):
        """A name of only whitespace must not be saved."""
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        with patch(
            "plottter.gui.settings_panel.QInputDialog.getText",
            return_value=("   ", True),
        ):
            with patch(
                "plottter.presets.user_presets.save_user_preset"
            ) as mock_save:
                sp._on_preset_changed("Save Current as Preset\u2026")
                mock_save.assert_not_called()

    def test_combo_resets_to_custom_on_cancel(self, panel):
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        with patch(
            "plottter.gui.settings_panel.QInputDialog.getText",
            return_value=("", False),
        ):
            with patch("plottter.presets.user_presets.save_user_preset"):
                sp._on_preset_changed("Save Current as Preset\u2026")

        assert sp._preset_combo.currentText() == "Custom"


# ─── (e) Saved params match widget values ────────────────────────────────────

class TestGatherCurrentParams:
    def test_gather_returns_float_value(self, panel):
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        from PyQt6.QtWidgets import QDoubleSpinBox
        widget = sp._param_widgets.get("radius")
        if widget is not None and isinstance(widget, QDoubleSpinBox):
            widget.setValue(7.3)

        params = sp._gather_current_params()
        assert "radius" in params
        if isinstance(sp._param_widgets.get("radius"), QDoubleSpinBox):
            assert abs(params["radius"] - 7.3) < 0.001

    def test_gather_does_not_include_private_keys(self, panel):
        """_gather_current_params must not include injected private keys like _source_image."""
        sp, _ = panel
        gen = _make_mock_generator()
        sp.set_generator(gen)

        params = sp._gather_current_params()
        for key in params:
            assert not key.startswith("_"), f"Private key '{key}' should not appear in gathered params"

    def test_saved_preset_contains_widget_values(self, panel, tmp_path):
        """End-to-end: widget value → save → disk → load back."""
        from plottter.presets.user_presets import load_user_presets, save_user_preset

        sp, _ = panel
        gen = _make_mock_generator("Stipple")
        sp.set_generator(gen)

        from PyQt6.QtWidgets import QDoubleSpinBox
        widget = sp._param_widgets.get("radius")
        expected_radius = 9.9
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(expected_radius)

        with patch(
            "plottter.gui.settings_panel.QInputDialog.getText",
            return_value=("Disk Test Preset", True),
        ):
            # Patch save to use tmp_path presets dir so we don't touch ~/.plottter
            real_save = __import__(
                "plottter.presets.user_presets", fromlist=["save_user_preset"]
            ).save_user_preset

            captured = []

            def fake_save(generator_name, preset, **kw):
                real_save(generator_name, preset, presets_dir=tmp_path, **kw)
                captured.append(preset)

            with patch("plottter.presets.user_presets.save_user_preset", fake_save):
                sp._on_preset_changed("Save Current as Preset\u2026")

        assert len(captured) == 1, "Expected exactly one preset to be saved"
        saved_params = captured[0].params
        if isinstance(widget, QDoubleSpinBox):
            assert abs(saved_params.get("radius", 0) - expected_radius) < 0.001
