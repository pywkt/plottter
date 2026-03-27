"""Tests for "Regenerate All 3D Layers" action (task 62.1).

Covers:
(a) Two overlapping 3D cubes — action regenerates both layers.
(b) Progress dialog is shown during regeneration.
(c) Undo reverts all layers to pre-regeneration state (single macro).
(d) Non-3D layers are skipped.
(e) Menu action exists with correct shortcut.
"""

from __future__ import annotations

import time

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared camera / 3D params
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


def _cube_info(pos_x: float = 0.0) -> dict:
    """Minimal 3D generator_info for a cube — HLR disabled for speed."""
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


def _non_3d_info() -> dict:
    return {
        "mode": "Math Art",
        "generator": "Parametric Curve",
        "params": {},
    }


# Dummy paths that are clearly NOT the result of a 3D Cube generation
# (a 3D cube produces ~12 polylines, not a single two-point segment)
_DUMMY_PATHS = [[(0.0, 0.0), (99.0, 99.0)]]


def _make_project_two_cubes():
    """Two 3D cube layers as the only layers."""
    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test3D", canvas=canvas)
    proj.metadata["scene3d_camera"] = CAM

    layer1 = Layer(name="Cube Front", color="#000000")
    layer1.paths = list(_DUMMY_PATHS)
    layer1.generator_info = _cube_info(pos_x=0.0)

    layer2 = Layer(name="Cube Back", color="#FF0000")
    layer2.paths = list(_DUMMY_PATHS)
    layer2.generator_info = _cube_info(pos_x=3.0)

    proj.add_layer(layer1)
    proj.add_layer(layer2)
    return proj, layer1.id, layer2.id


def _make_project_mixed():
    """One 3D layer + one non-3D layer."""
    canvas = Canvas.from_preset("A4")
    proj = Project(name="Mixed", canvas=canvas)
    proj.metadata["scene3d_camera"] = CAM

    layer_3d = Layer(name="3D Cube", color="#000000")
    layer_3d.paths = list(_DUMMY_PATHS)
    layer_3d.generator_info = _cube_info()

    layer_other = Layer(name="Math Art Layer", color="#00FF00")
    layer_other.paths = [[(5.0, 5.0), (10.0, 10.0)]]
    layer_other.generator_info = _non_3d_info()

    proj.add_layer(layer_3d)
    proj.add_layer(layer_other)
    return proj, layer_3d.id, layer_other.id


def _make_project_no_3d():
    """Project with no 3D layers."""
    canvas = Canvas.from_preset("A4")
    proj = Project(name="No3D", canvas=canvas)
    layer = Layer(name="Sketch", color="#000000")
    layer.paths = [[(0.0, 0.0), (1.0, 0.0)]]
    proj.add_layer(layer)
    return proj, layer.id


def _build_win(controller, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


def _wait_regen_done(win, qtbot, timeout: float = 10.0) -> None:
    """Block until all 3D layers have been regenerated (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        qtbot.wait(100)
        if hasattr(win, "_regen3d_layers") and win._regen3d_idx >= len(win._regen3d_layers):
            break


# ---------------------------------------------------------------------------
# (a) Two 3D cubes — both layers are regenerated
# ---------------------------------------------------------------------------

class TestTwoCubesRegenerated:
    def test_both_layers_get_new_paths(self, qtbot):
        proj, lid1, lid2 = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        # Make the first layer active so flush_current_snapshot sees a 3D layer
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)
        win = _build_win(ctrl, qtbot)

        win._on_regenerate_all_3d()
        _wait_regen_done(win, qtbot)

        layer1 = ctrl.get_layer(lid1)
        layer2 = ctrl.get_layer(lid2)
        assert layer1 is not None and layer1.paths != _DUMMY_PATHS, \
            "layer1 should be regenerated with fresh paths"
        assert layer2 is not None and layer2.paths != _DUMMY_PATHS, \
            "layer2 should be regenerated with fresh paths"

    def test_regenerated_paths_are_non_empty(self, qtbot):
        proj, lid1, lid2 = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)
        win = _build_win(ctrl, qtbot)

        win._on_regenerate_all_3d()
        _wait_regen_done(win, qtbot)

        assert ctrl.get_layer(lid1).paths, "layer1 should have non-empty paths after regen"
        assert ctrl.get_layer(lid2).paths, "layer2 should have non-empty paths after regen"


# ---------------------------------------------------------------------------
# (b) Progress dialog is shown
# ---------------------------------------------------------------------------

class TestProgressDialog:
    def test_progress_dialog_created(self, qtbot):
        proj, lid1, lid2 = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)
        win = _build_win(ctrl, qtbot)

        win._on_regenerate_all_3d()

        # _regen3d_progress should be set immediately after the call
        assert hasattr(win, "_regen3d_progress"), "progress dialog attribute should exist"
        from PyQt6.QtWidgets import QProgressDialog
        assert isinstance(win._regen3d_progress, QProgressDialog)

        # Wait for completion
        _wait_regen_done(win, qtbot)


# ---------------------------------------------------------------------------
# (c) Undo reverts all layers at once
# ---------------------------------------------------------------------------

class TestUndoMacro:
    def test_single_undo_reverts_both_layers(self, qtbot):
        proj, lid1, lid2 = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)
        win = _build_win(ctrl, qtbot)

        initial_paths1 = [list(p) for p in ctrl.get_layer(lid1).paths]
        initial_paths2 = [list(p) for p in ctrl.get_layer(lid2).paths]

        win._on_regenerate_all_3d()
        _wait_regen_done(win, qtbot)

        # Confirm paths changed
        assert ctrl.get_layer(lid1).paths != initial_paths1
        assert ctrl.get_layer(lid2).paths != initial_paths2

        # Single undo should revert both (macro)
        assert ctrl.undo_stack.canUndo()
        ctrl.undo_stack.undo()

        assert ctrl.get_layer(lid1).paths == initial_paths1, \
            "layer1 should revert to initial paths after undo"
        assert ctrl.get_layer(lid2).paths == initial_paths2, \
            "layer2 should revert to initial paths after undo"

    def test_undo_text_contains_3d(self, qtbot):
        proj, lid1, lid2 = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid1)
        win = _build_win(ctrl, qtbot)

        win._on_regenerate_all_3d()
        _wait_regen_done(win, qtbot)

        assert ctrl.undo_stack.canUndo()
        assert "3D" in ctrl.undo_stack.undoText() or "Regenerate" in ctrl.undo_stack.undoText()


# ---------------------------------------------------------------------------
# (d) Non-3D layers are skipped
# ---------------------------------------------------------------------------

class TestNon3dLayersSkipped:
    def test_non_3d_layer_not_modified(self, qtbot):
        proj, lid_3d, lid_other = _make_project_mixed()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        # Set the 3D layer as active so flush_current_snapshot won't corrupt it
        ctrl.set_active_layer(lid_3d)
        win = _build_win(ctrl, qtbot)

        other_paths_before = [list(p) for p in ctrl.get_layer(lid_other).paths]

        win._on_regenerate_all_3d()
        _wait_regen_done(win, qtbot)

        other_layer = ctrl.get_layer(lid_other)
        assert other_layer is not None
        assert other_layer.paths == other_paths_before, "non-3D layer must not be modified"

    def test_only_3d_layers_are_queued(self, qtbot):
        proj, lid_3d, lid_other = _make_project_mixed()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid_3d)
        win = _build_win(ctrl, qtbot)

        win._on_regenerate_all_3d()

        assert len(win._regen3d_layers) == 1
        assert win._regen3d_layers[0].id == lid_3d

        _wait_regen_done(win, qtbot)

    def test_no_3d_layers_shows_info(self, qtbot, monkeypatch):
        proj, lid = _make_project_no_3d()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        win = _build_win(ctrl, qtbot)

        shown = []
        monkeypatch.setattr(
            "plottter.gui.main_window.QMessageBox.information",
            lambda *a, **kw: shown.append(a),
        )

        win._on_regenerate_all_3d()

        assert len(shown) == 1, "should show info dialog when no 3D layers exist"
        # No regen state should be created
        assert not hasattr(win, "_regen3d_layers") or not win._regen3d_layers


# ---------------------------------------------------------------------------
# (e) Menu action exists with correct shortcut
# ---------------------------------------------------------------------------

class TestMenuAction:
    def test_action_attribute_exists(self, qtbot):
        proj, *_ = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        win = _build_win(ctrl, qtbot)
        assert hasattr(win, "_act_regen_all_3d")

    def test_shortcut_is_ctrl_shift_g(self, qtbot):
        from PyQt6.QtGui import QKeySequence
        proj, *_ = _make_project_two_cubes()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        win = _build_win(ctrl, qtbot)
        expected = QKeySequence("Ctrl+Shift+G").toString()
        actual = win._act_regen_all_3d.shortcut().toString()
        assert actual == expected, f"expected Ctrl+Shift+G, got {actual!r}"
