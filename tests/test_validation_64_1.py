"""Tests for task 64.1 — Scale art to fit when resizing canvas.

Covers:
(a) Changing A4 to A3 with scale=Yes keeps art proportionally scaled
(b) scale=No keeps art at original mm positions
(c) undo restores both canvas and paths
(d) empty project skips the question
(e) offset params in generator_info are updated
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from plottter.models import Canvas, Layer, Project
from plottter.processing.scale import scale_paths_to_canvas


# ---------------------------------------------------------------------------
# Shared canvas presets
# ---------------------------------------------------------------------------

_A4 = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0, paper_preset="A4")
_A3 = Canvas(width_mm=297.0, height_mm=420.0, margin_mm=10.0, paper_preset="A3")

# A4 draw area: left=10, top=10, right=200, bottom=287 → w=190, h=277
# A3 draw area: left=10, top=10, right=287, bottom=410 → w=277, h=400
_SX_A4_A3 = 277.0 / 190.0
_SY_A4_A3 = 400.0 / 277.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_with_paths(extra_gen_info: dict | None = None) -> tuple[Project, str]:
    """A4 project with one layer containing a path and optional generator_info."""
    proj = Project(name="Test", canvas=copy.copy(_A4))
    layer = Layer(name="Layer 1", color="#000000")
    layer.paths = [[(10.0, 10.0), (200.0, 287.0)]]
    if extra_gen_info is not None:
        layer.generator_info = extra_gen_info
    proj.add_layer(layer)
    return proj, layer.id


def _project_empty() -> tuple[Project, str]:
    """A4 project with one layer containing no paths."""
    proj = Project(name="Test", canvas=copy.copy(_A4))
    layer = Layer(name="Layer 1", color="#000000")
    proj.add_layer(layer)
    return proj, layer.id


def _make_controller(proj: Project):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(proj)


def _make_win(controller, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


def _mock_new_project_dlg(new_canvas: Canvas):
    """Build a mock NewProjectDialog class that accepts and returns new_canvas."""
    from PyQt6.QtWidgets import QDialog
    mock_cls = MagicMock()
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    mock_cls.DialogCode.Accepted = QDialog.DialogCode.Accepted
    mock_inst.exec.return_value = QDialog.DialogCode.Accepted
    mock_inst.get_canvas.return_value = new_canvas
    return mock_cls


def _run_canvas_settings(win, new_canvas: Canvas, scale_yes: bool) -> None:
    """Invoke _on_canvas_settings with both dialogs mocked."""
    mock_cls = _mock_new_project_dlg(new_canvas)
    mock_qmb = MagicMock()
    # Set up StandardButton so Yes != No (they are different attributes)
    mock_qmb.question.return_value = (
        mock_qmb.StandardButton.Yes if scale_yes else mock_qmb.StandardButton.No
    )
    with patch("plottter.gui.dialogs.new_project.NewProjectDialog", mock_cls), \
         patch("plottter.gui.main_window.QMessageBox", mock_qmb):
        win._on_canvas_settings()


# ---------------------------------------------------------------------------
# (a) A4 to A3 with scale=Yes keeps art proportionally scaled
# ---------------------------------------------------------------------------


class TestScaleYes:
    """Spec (a): scale=Yes transforms paths proportionally to the new drawing area."""

    def test_canvas_updated_to_new_size(self, qtbot):
        """Canvas dimensions change to A3 after confirming resize."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 420.0) < 1e-6

    def test_draw_area_origin_maps_to_new_origin(self, qtbot):
        """Top-left of old drawing area maps to top-left of new drawing area."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        first_pt = layer.paths[0][0]
        # A4 origin (10,10) → A3 origin (10,10)
        assert abs(first_pt[0] - 10.0) < 1e-6
        assert abs(first_pt[1] - 10.0) < 1e-6

    def test_paths_match_scale_paths_to_canvas(self, qtbot):
        """Scaled paths match output of scale_paths_to_canvas helper."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        expected = scale_paths_to_canvas(original_paths, _A4, _A3)
        for new_poly, exp_poly in zip(layer.paths, expected):
            for (nx, ny), (ex, ey) in zip(new_poly, exp_poly):
                assert abs(nx - ex) < 1e-6, f"x mismatch: {nx} != {ex}"
                assert abs(ny - ey) < 1e-6, f"y mismatch: {ny} != {ey}"


# ---------------------------------------------------------------------------
# (b) scale=No keeps art at original mm positions
# ---------------------------------------------------------------------------


class TestScaleNo:
    """Spec (b): scale=No keeps art at original mm positions."""

    def test_paths_unchanged_when_scale_no(self, qtbot):
        """Paths are not modified when user declines scaling."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=False)

        layer = ctrl.get_layer(lid)
        assert layer.paths == original_paths

    def test_canvas_updated_even_when_scale_no(self, qtbot):
        """Canvas is still changed to A3 even when user declines scaling."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=False)

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 420.0) < 1e-6


# ---------------------------------------------------------------------------
# (c) undo restores both canvas and paths
# ---------------------------------------------------------------------------


class TestUndo:
    """Spec (c): undo restores both canvas and paths to pre-resize state."""

    def test_undo_restores_canvas(self, qtbot):
        """After a single undo, canvas reverts to original A4 dimensions."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)
        ctrl.undo_stack.undo()

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 210.0) < 1e-6
        assert abs(canvas.height_mm - 297.0) < 1e-6

    def test_undo_restores_paths(self, qtbot):
        """After a single undo, paths revert to original positions."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)
        ctrl.undo_stack.undo()

        layer = ctrl.get_layer(lid)
        assert layer is not None
        assert layer.paths == original_paths

    def test_undo_is_single_step(self, qtbot):
        """Canvas and path changes are bundled into one undo step (macro)."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        # One undo should revert BOTH canvas and paths
        assert ctrl.undo_stack.canUndo()
        ctrl.undo_stack.undo()

        canvas = ctrl.current_project.canvas
        layer = ctrl.get_layer(lid)
        assert abs(canvas.width_mm - 210.0) < 1e-6, "Canvas not reverted after single undo"
        assert layer.paths == original_paths, "Paths not reverted after single undo"

        # Now there should be nothing left to undo (only one macro was pushed)
        assert not ctrl.undo_stack.canUndo()


# ---------------------------------------------------------------------------
# (d) empty project skips the question
# ---------------------------------------------------------------------------


class TestEmptyProjectSkipsQuestion:
    """Spec (d): QMessageBox.question is not shown when no layers have paths."""

    def test_no_question_shown_for_empty_project(self, qtbot):
        """QMessageBox.question is never called when all layers are empty."""
        proj, lid = _project_empty()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        mock_cls = _mock_new_project_dlg(copy.copy(_A3))
        mock_qmb = MagicMock()
        with patch("plottter.gui.dialogs.new_project.NewProjectDialog", mock_cls), \
             patch("plottter.gui.main_window.QMessageBox", mock_qmb):
            win._on_canvas_settings()

        mock_qmb.question.assert_not_called()

    def test_canvas_still_updated_for_empty_project(self, qtbot):
        """Canvas is updated even when there are no paths."""
        proj, lid = _project_empty()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=False)

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6


# ---------------------------------------------------------------------------
# (e) offset params in generator_info are updated
# ---------------------------------------------------------------------------


class TestGeneratorInfoOffsets:
    """Spec (e): x_offset_mm/y_offset_mm and 3D pos_x/pos_y in generator_info are scaled."""

    def _offset_gen_info(self, offset_x: float, offset_y: float) -> dict:
        return {
            "mode": "Math Art",
            "generator_name": "Parametric Curve",
            "params": {
                "x_offset_mm": offset_x,
                "y_offset_mm": offset_y,
            },
            "transforms": {},
        }

    def test_x_offset_mm_scaled(self, qtbot):
        """x_offset_mm is multiplied by the x scale factor (new_draw_w / old_draw_w)."""
        gen_info = self._offset_gen_info(20.0, 0.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        new_x = layer.generator_info["params"]["x_offset_mm"]
        assert abs(new_x - 20.0 * _SX_A4_A3) < 1e-6

    def test_y_offset_mm_scaled(self, qtbot):
        """y_offset_mm is multiplied by the y scale factor (new_draw_h / old_draw_h)."""
        gen_info = self._offset_gen_info(0.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        new_y = layer.generator_info["params"]["y_offset_mm"]
        assert abs(new_y - 30.0 * _SY_A4_A3) < 1e-6

    def test_offsets_unchanged_when_scale_no(self, qtbot):
        """generator_info params are not modified when user declines scaling."""
        gen_info = self._offset_gen_info(20.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=False)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        params = layer.generator_info["params"]
        assert abs(params["x_offset_mm"] - 20.0) < 1e-6
        assert abs(params["y_offset_mm"] - 30.0) < 1e-6

    def test_undo_restores_gen_info_offsets(self, qtbot):
        """Undo reverts generator_info offsets to original values."""
        gen_info = self._offset_gen_info(20.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)
        ctrl.undo_stack.undo()

        layer = ctrl.get_layer(lid)
        assert layer is not None
        params = layer.generator_info["params"]
        assert abs(params["x_offset_mm"] - 20.0) < 1e-6
        assert abs(params["y_offset_mm"] - 30.0) < 1e-6

    def _3d_gen_info(self, pos_x: float, pos_y: float) -> dict:
        return {
            "mode": "3D Scene",
            "generator_name": "3D Scene",
            "params": {
                "shape_type": "Cube",
                "cube_size": 2.0,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": 0.0,
            },
            "transforms": {},
        }

    def test_3d_pos_x_scaled(self, qtbot):
        """pos_x in 3D layer generator_info is scaled by x scale factor."""
        gen_info = self._3d_gen_info(15.0, 0.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        new_pos_x = layer.generator_info["params"]["pos_x"]
        assert abs(new_pos_x - 15.0 * _SX_A4_A3) < 1e-6

    def test_3d_pos_y_scaled(self, qtbot):
        """pos_y in 3D layer generator_info is scaled by y scale factor."""
        gen_info = self._3d_gen_info(0.0, 25.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        new_pos_y = layer.generator_info["params"]["pos_y"]
        assert abs(new_pos_y - 25.0 * _SY_A4_A3) < 1e-6

    def test_3d_pos_unchanged_when_scale_no(self, qtbot):
        """3D pos_x/pos_y are not modified when user declines scaling."""
        gen_info = self._3d_gen_info(15.0, 25.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=False)

        layer = ctrl.get_layer(lid)
        assert layer is not None
        params = layer.generator_info["params"]
        assert abs(params["pos_x"] - 15.0) < 1e-6
        assert abs(params["pos_y"] - 25.0) < 1e-6

    def test_undo_restores_3d_pos(self, qtbot):
        """Undo reverts 3D pos_x/pos_y to original values."""
        gen_info = self._3d_gen_info(15.0, 25.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_canvas_settings(win, copy.copy(_A3), scale_yes=True)
        ctrl.undo_stack.undo()

        layer = ctrl.get_layer(lid)
        assert layer is not None
        params = layer.generator_info["params"]
        assert abs(params["pos_x"] - 15.0) < 1e-6
        assert abs(params["pos_y"] - 25.0) < 1e-6
