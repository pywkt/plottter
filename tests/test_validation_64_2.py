"""Tests for task 64.2 — Scale option when rotating canvas.

Covers:
(a) Rotating A4 portrait → landscape with "Scale to fit" stretches art to fill
(b) "Keep aspect" scales uniformly and centers
(c) "Don't scale" leaves art at original positions
(d) undo reverts everything (canvas + paths in a single step)
(e) empty project skips the dialog
(f) offset params in generator_info are updated for each scale mode
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from plottter.models import Canvas, Layer, Project
from plottter.processing.scale import scale_paths_keep_aspect, scale_paths_to_canvas


# ---------------------------------------------------------------------------
# Canvas presets
# ---------------------------------------------------------------------------

# A4 portrait
_A4P = Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0, paper_preset="A4")
# A4 landscape (rotated)
_A4L = Canvas(width_mm=297.0, height_mm=210.0, margin_mm=10.0, paper_preset="A4")

# A4 portrait draw area: w=190, h=277
# A4 landscape draw area: w=277, h=190
_SX_STRETCH = 277.0 / 190.0
_SY_STRETCH = 190.0 / 277.0
_SCALE_KEEP = min(277.0 / 190.0, 190.0 / 277.0)  # = 190/277


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_with_paths(extra_gen_info: dict | None = None) -> tuple[Project, str]:
    """A4 portrait project with one layer containing a path."""
    proj = Project(name="Test", canvas=copy.copy(_A4P))
    layer = Layer(name="Layer 1", color="#000000")
    layer.paths = [[(10.0, 10.0), (200.0, 287.0)]]
    if extra_gen_info is not None:
        layer.generator_info = extra_gen_info
    proj.add_layer(layer)
    return proj, layer.id


def _project_empty() -> tuple[Project, str]:
    """A4 portrait project with one layer containing no paths."""
    proj = Project(name="Test", canvas=copy.copy(_A4P))
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


def _run_rotate(win, scale_mode: str | None) -> None:
    """Invoke _on_rotate_canvas with _ask_rotate_scale_mode mocked to return scale_mode."""
    with patch.object(win, "_ask_rotate_scale_mode", return_value=scale_mode):
        win._on_rotate_canvas()


# ---------------------------------------------------------------------------
# (a) "Scale to fit" — stretch art to fill new dimensions
# ---------------------------------------------------------------------------

class TestScaleStretch:
    """scale_mode='stretch': art is non-uniformly scaled to fill the new drawing area."""

    def test_canvas_rotated(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 210.0) < 1e-6

    def test_paths_match_scale_paths_to_canvas(self, qtbot):
        """Stretched paths match scale_paths_to_canvas(old→new)."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, "stretch")

        layer = ctrl.get_layer(lid)
        expected = scale_paths_to_canvas(original_paths, _A4P, _A4L)
        for new_poly, exp_poly in zip(layer.paths, expected):
            for (nx, ny), (ex, ey) in zip(new_poly, exp_poly):
                assert abs(nx - ex) < 1e-6
                assert abs(ny - ey) < 1e-6

    def test_draw_area_origin_maps_to_new_origin(self, qtbot):
        """Top-left of old drawing area maps to top-left of new drawing area."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")

        layer = ctrl.get_layer(lid)
        first_pt = layer.paths[0][0]
        assert abs(first_pt[0] - 10.0) < 1e-6
        assert abs(first_pt[1] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# (b) "Keep aspect" — uniform scale, centered
# ---------------------------------------------------------------------------

class TestScaleKeepAspect:
    """scale_mode='keep_aspect': art is uniformly scaled and centered."""

    def test_canvas_rotated(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "keep_aspect")

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 210.0) < 1e-6

    def test_paths_match_scale_paths_keep_aspect(self, qtbot):
        """Paths match scale_paths_keep_aspect(old→new)."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, "keep_aspect")

        layer = ctrl.get_layer(lid)
        expected = scale_paths_keep_aspect(original_paths, _A4P, _A4L)
        for new_poly, exp_poly in zip(layer.paths, expected):
            for (nx, ny), (ex, ey) in zip(new_poly, exp_poly):
                assert abs(nx - ex) < 1e-6
                assert abs(ny - ey) < 1e-6

    def test_uniform_scale_applied(self, qtbot):
        """Both axes use the same scale factor (min of the two ratios)."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        # Place a point at old drawing-area origin (10, 10)
        ctrl.get_layer(lid).paths = [[(10.0, 10.0)]]

        _run_rotate(win, "keep_aspect")

        # With A4P→A4L, scale = min(277/190, 190/277) = 190/277
        # offset_x = 10 + (277 - 190*(190/277))/2 ≈ 83.34
        # offset_y = 10 + (190 - 277*(190/277))/2 = 10 + 0 = 10
        layer = ctrl.get_layer(lid)
        pt = layer.paths[0][0]
        expected_x = 10.0 + (277.0 - 190.0 * _SCALE_KEEP) / 2.0
        expected_y = 10.0  # height is the constraining axis → no vertical offset
        assert abs(pt[0] - expected_x) < 1e-4
        assert abs(pt[1] - expected_y) < 1e-4


# ---------------------------------------------------------------------------
# (c) "Don't scale" — art stays at original mm positions
# ---------------------------------------------------------------------------

class TestDontScale:
    """scale_mode='none': paths are unchanged, only canvas is rotated."""

    def test_canvas_rotated(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "none")

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 210.0) < 1e-6

    def test_paths_unchanged(self, qtbot):
        """Paths are not modified when scale_mode is 'none'."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, "none")

        layer = ctrl.get_layer(lid)
        assert layer.paths == original_paths


# ---------------------------------------------------------------------------
# (d) Cancel — nothing changes
# ---------------------------------------------------------------------------

class TestCancel:
    """Returning None from the dialog (cancel) leaves canvas and paths untouched."""

    def test_cancel_leaves_canvas_unchanged(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, None)  # None = cancelled

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 210.0) < 1e-6
        assert abs(canvas.height_mm - 297.0) < 1e-6

    def test_cancel_leaves_paths_unchanged(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, None)

        layer = ctrl.get_layer(lid)
        assert layer.paths == original_paths


# ---------------------------------------------------------------------------
# (e) Empty project — dialog not shown, canvas still rotated
# ---------------------------------------------------------------------------

class TestEmptyProject:
    """When no layers have paths, the scale dialog is not shown."""

    def test_dialog_not_shown_for_empty_project(self, qtbot):
        proj, lid = _project_empty()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        called = []
        original = win._ask_rotate_scale_mode
        win._ask_rotate_scale_mode = lambda: called.append(1) or "none"
        win._on_rotate_canvas()
        # _ask_rotate_scale_mode should NOT have been called
        assert len(called) == 0

    def test_canvas_still_rotated_for_empty_project(self, qtbot):
        proj, lid = _project_empty()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        with patch.object(win, "_ask_rotate_scale_mode", return_value="none"):
            win._on_rotate_canvas()

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 297.0) < 1e-6
        assert abs(canvas.height_mm - 210.0) < 1e-6


# ---------------------------------------------------------------------------
# (f) Undo reverts canvas + paths in a single step
# ---------------------------------------------------------------------------

class TestUndo:
    """Undo reverts both canvas and paths to pre-rotation state."""

    def test_undo_restores_canvas(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")
        ctrl.undo_stack.undo()

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 210.0) < 1e-6
        assert abs(canvas.height_mm - 297.0) < 1e-6

    def test_undo_restores_paths(self, qtbot):
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, "stretch")
        ctrl.undo_stack.undo()

        layer = ctrl.get_layer(lid)
        assert layer.paths == original_paths

    def test_undo_is_single_step(self, qtbot):
        """Canvas and path changes are bundled into one undo macro."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)
        original_paths = copy.deepcopy(ctrl.get_layer(lid).paths)

        _run_rotate(win, "keep_aspect")

        assert ctrl.undo_stack.canUndo()
        ctrl.undo_stack.undo()

        canvas = ctrl.current_project.canvas
        layer = ctrl.get_layer(lid)
        assert abs(canvas.width_mm - 210.0) < 1e-6
        assert layer.paths == original_paths
        assert not ctrl.undo_stack.canUndo()

    def test_undo_none_is_single_step(self, qtbot):
        """'Don't scale' rotation is also a single undoable step."""
        proj, lid = _project_with_paths()
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "none")

        assert ctrl.undo_stack.canUndo()
        ctrl.undo_stack.undo()

        canvas = ctrl.current_project.canvas
        assert abs(canvas.width_mm - 210.0) < 1e-6
        assert not ctrl.undo_stack.canUndo()


# ---------------------------------------------------------------------------
# (g) Generator info offsets scaled correctly
# ---------------------------------------------------------------------------

class TestGeneratorInfoOffsets:
    """x_offset_mm / y_offset_mm and 3D pos_x/pos_y are updated per scale mode."""

    def _offset_gen_info(self, ox: float, oy: float) -> dict:
        return {
            "mode": "Math Art",
            "generator_name": "Parametric Curve",
            "params": {"x_offset_mm": ox, "y_offset_mm": oy},
            "transforms": {},
        }

    def test_stretch_x_offset_scaled(self, qtbot):
        gen_info = self._offset_gen_info(20.0, 0.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")

        new_x = ctrl.get_layer(lid).generator_info["params"]["x_offset_mm"]
        assert abs(new_x - 20.0 * _SX_STRETCH) < 1e-6

    def test_stretch_y_offset_scaled(self, qtbot):
        gen_info = self._offset_gen_info(0.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")

        new_y = ctrl.get_layer(lid).generator_info["params"]["y_offset_mm"]
        assert abs(new_y - 30.0 * _SY_STRETCH) < 1e-6

    def test_keep_aspect_offsets_use_uniform_scale(self, qtbot):
        gen_info = self._offset_gen_info(20.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "keep_aspect")

        params = ctrl.get_layer(lid).generator_info["params"]
        assert abs(params["x_offset_mm"] - 20.0 * _SCALE_KEEP) < 1e-6
        assert abs(params["y_offset_mm"] - 30.0 * _SCALE_KEEP) < 1e-6

    def test_none_offsets_unchanged(self, qtbot):
        gen_info = self._offset_gen_info(20.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "none")

        params = ctrl.get_layer(lid).generator_info["params"]
        assert abs(params["x_offset_mm"] - 20.0) < 1e-6
        assert abs(params["y_offset_mm"] - 30.0) < 1e-6

    def test_undo_restores_offsets(self, qtbot):
        gen_info = self._offset_gen_info(20.0, 30.0)
        proj, lid = _project_with_paths(extra_gen_info=gen_info)
        ctrl = _make_controller(proj)
        win = _make_win(ctrl, qtbot)

        _run_rotate(win, "stretch")
        ctrl.undo_stack.undo()

        params = ctrl.get_layer(lid).generator_info["params"]
        assert abs(params["x_offset_mm"] - 20.0) < 1e-6
        assert abs(params["y_offset_mm"] - 30.0) < 1e-6
