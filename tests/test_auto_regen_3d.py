"""Tests for "Auto-regenerate other 3D layers" feature (task 62.2).

Covers:
(a) With auto-regenerate on, generating one 3D layer also regenerates siblings.
(b) With auto-regenerate off, only the active layer is generated.
(c) Setting persists across app restarts (QSettings round-trip).
"""

from __future__ import annotations

import time

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CAM = {
    "azimuth": 30.0,
    "elevation": 20.0,
    "distance": 8.0,
    "look_at_x": 0.0,
    "look_at_y": 0.0,
    "look_at_z": 0.0,
    "fov": 45.0,
    "projection": "perspective",
}

_DUMMY_PATHS = [[(0.0, 0.0), (99.0, 99.0)]]


def _cube_info(pos_x: float = 0.0) -> dict:
    return {
        "mode": "3D Scene",
        "generator": "3D Scene",
        "params": {
            "shape_type": "Cube",
            "hlr_enabled": False,
            "chop_step": 0.5,
            "pos_x": pos_x,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "scale_z": 1.0,
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
        },
    }


def _make_two_cube_project():
    canvas = Canvas.from_preset("A4")
    proj = Project(name="AutoRegen3D", canvas=canvas)
    proj.metadata["scene3d_camera"] = CAM

    layer1 = Layer(name="Cube A", color="#000000")
    layer1.paths = list(_DUMMY_PATHS)
    layer1.generator_info = _cube_info(pos_x=0.0)

    layer2 = Layer(name="Cube B", color="#FF0000")
    layer2.paths = list(_DUMMY_PATHS)
    layer2.generator_info = _cube_info(pos_x=3.0)

    proj.add_layer(layer1)
    proj.add_layer(layer2)
    return proj, layer1.id, layer2.id


def _build_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    return panel


def _wait_auto_regen_done(panel, qtbot, timeout: float = 10.0) -> None:
    """Block until the auto-regen chain finishes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        qtbot.wait(100)
        # Chain is done when _auto_regen_layers is empty
        if not panel._auto_regen_layers:
            break


# ---------------------------------------------------------------------------
# (a) Auto-regenerate ON — sibling gets regenerated
# ---------------------------------------------------------------------------

class TestAutoRegenOn:
    def test_sibling_layer_regenerated(self, qtbot):
        """After generating layer1, layer2 should get fresh (non-dummy) paths."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        # Enable the auto-regen checkbox
        panel._auto_regen_3d_cb.setChecked(True)
        # Switch to 3D mode so _current_mode is correct
        panel._current_mode = "3D Scene"

        # Simulate _on_generation_finished for layer1 with dummy replacement
        new_paths_for_l1 = [[(1.0, 2.0), (3.0, 4.0)]]
        panel._on_generation_finished(new_paths_for_l1, lid1)

        # Wait for the auto-regen chain to complete
        _wait_auto_regen_done(panel, qtbot)

        layer2 = ctrl.get_layer(lid2)
        assert layer2 is not None
        # The sibling's paths should have been updated (no longer dummy)
        assert layer2.paths != _DUMMY_PATHS, \
            "layer2 (sibling) should be regenerated when auto-regen is ON"

    def test_sibling_paths_non_empty(self, qtbot):
        """The regenerated sibling should have non-empty paths."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(True)
        panel._current_mode = "3D Scene"

        panel._on_generation_finished([[(5.0, 5.0), (6.0, 6.0)]], lid1)
        _wait_auto_regen_done(panel, qtbot)

        layer2 = ctrl.get_layer(lid2)
        assert layer2.paths, "sibling layer should have non-empty paths after auto-regen"

    def test_generated_layer_itself_not_re_regenerated(self, qtbot):
        """The layer that was just generated (layer1) should NOT be re-queued."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(True)
        panel._current_mode = "3D Scene"

        explicit_l1_paths = [[(7.0, 8.0), (9.0, 10.0)]]
        panel._on_generation_finished(explicit_l1_paths, lid1)
        _wait_auto_regen_done(panel, qtbot)

        layer1 = ctrl.get_layer(lid1)
        # layer1 should retain the paths set by _on_generation_finished
        assert layer1.paths == explicit_l1_paths, \
            "layer1 should not be re-regenerated by auto-regen"

    def test_status_message_shown(self, qtbot):
        """A status message should be shown when auto-regen starts."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(True)
        panel._current_mode = "3D Scene"

        messages: list[str] = []

        # Patch window().statusBar().showMessage to capture messages
        from PyQt6.QtWidgets import QMainWindow, QStatusBar
        fake_status = QStatusBar()
        fake_status.showMessage = lambda msg, *args: messages.append(msg)
        fake_win = QMainWindow()
        fake_win.setStatusBar(fake_status)
        panel.setParent(fake_win)

        panel._on_generation_finished([[(0.0, 0.0), (1.0, 1.0)]], lid1)
        _wait_auto_regen_done(panel, qtbot)

        assert any("Auto-regenerating" in m for m in messages), \
            f"Expected 'Auto-regenerating' in status messages, got: {messages}"


# ---------------------------------------------------------------------------
# (b) Auto-regenerate OFF — only the active layer is generated
# ---------------------------------------------------------------------------

class TestAutoRegenOff:
    def test_sibling_not_touched(self, qtbot):
        """With auto-regen disabled, sibling layer must retain its original paths."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(False)  # explicitly OFF
        panel._current_mode = "3D Scene"

        panel._on_generation_finished([[(1.0, 2.0), (3.0, 4.0)]], lid1)

        # Give a moment — no background work should happen
        qtbot.wait(300)

        layer2 = ctrl.get_layer(lid2)
        assert layer2.paths == _DUMMY_PATHS, \
            "sibling layer should NOT be touched when auto-regen is OFF"

    def test_auto_regen_chain_empty(self, qtbot):
        """_auto_regen_layers should be empty (no chain started)."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(False)
        panel._current_mode = "3D Scene"

        panel._on_generation_finished([[(1.0, 2.0), (3.0, 4.0)]], lid1)
        qtbot.wait(200)

        assert panel._auto_regen_layers == [], \
            "_auto_regen_layers should remain empty when feature is disabled"

    def test_non_3d_mode_no_regen(self, qtbot):
        """Even with checkbox checked, non-3D mode should not trigger auto-regen."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)

        panel = _build_panel(ctrl, qtbot)
        panel._auto_regen_3d_cb.setChecked(True)
        panel._current_mode = "Math Art"  # Not 3D Scene

        panel._on_generation_finished([[(1.0, 2.0), (3.0, 4.0)]], lid1)
        qtbot.wait(200)

        assert panel._auto_regen_layers == [], \
            "auto-regen should not start outside 3D Scene mode"


# ---------------------------------------------------------------------------
# (c) QSettings persistence
# ---------------------------------------------------------------------------

class TestQSettingsPersistence:
    def test_setting_saved_when_checked(self, qtbot):
        """Toggling the checkbox on should persist True to QSettings."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        panel = _build_panel(ctrl, qtbot)

        # Ensure it starts unchecked then set it
        panel._auto_regen_3d_cb.setChecked(False)
        panel._auto_regen_3d_cb.setChecked(True)

        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        val = settings.value("3d/auto_regenerate", False, type=bool)
        assert val is True, "QSettings should store True after checking the box"

    def test_setting_saved_when_unchecked(self, qtbot):
        """Toggling the checkbox off should persist False to QSettings."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        panel = _build_panel(ctrl, qtbot)

        panel._auto_regen_3d_cb.setChecked(True)
        panel._auto_regen_3d_cb.setChecked(False)

        from PyQt6.QtCore import QSettings
        settings = QSettings("Plottter", "Plottter")
        val = settings.value("3d/auto_regenerate", True, type=bool)
        assert val is False, "QSettings should store False after unchecking the box"

    def test_checkbox_exists_and_is_qcheckbox(self, qtbot):
        """The auto-regen checkbox must exist on the SettingsPanel."""
        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        panel = _build_panel(ctrl, qtbot)

        from PyQt6.QtWidgets import QCheckBox
        assert hasattr(panel, "_auto_regen_3d_cb"), \
            "SettingsPanel should have _auto_regen_3d_cb attribute"
        assert isinstance(panel._auto_regen_3d_cb, QCheckBox), \
            "_auto_regen_3d_cb should be a QCheckBox"

    def test_checkbox_default_is_false(self, qtbot):
        """Default value (when QSettings has no stored value) should be False."""
        from PyQt6.QtCore import QSettings
        # Clear any previously stored value for this test
        settings = QSettings("Plottter", "Plottter")
        settings.remove("3d/auto_regenerate")

        proj, lid1, lid2 = _make_two_cube_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        panel = _build_panel(ctrl, qtbot)

        assert not panel._auto_regen_3d_cb.isChecked(), \
            "Auto-regenerate checkbox should default to unchecked"
