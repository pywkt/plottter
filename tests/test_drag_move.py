"""Tests for drag-to-move tool (task 16.59).

Covers:
- set_drag_move_active(True) sets _drag_move_active flag
- set_drag_move_active(True) disables _mask_paint_active and _shape_draw_active
- set_drag_move_active(False) resets state
- _on_layer_move_finished handler applies coordinate translation via set_layer_paths
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project(with_paths: bool = False) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    if with_paths:
        layer.paths = [[(0.0, 0.0), (10.0, 0.0)], [(5.0, 5.0), (15.0, 5.0)]]
    proj.add_layer(layer)
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def controller_with_paths(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project(with_paths=True))


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget
    w = CanvasWidget(controller)
    w.resize(800, 600)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def main_window(controller_with_paths, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller_with_paths)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ===========================================================================
# 1. set_drag_move_active(True) sets _drag_move_active flag
# ===========================================================================


class TestSetDragMoveActiveTrue:
    def test_flag_is_set(self, canvas_widget):
        assert canvas_widget._drag_move_active is False
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._drag_move_active is True

    def test_is_drag_move_active_returns_true(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget.is_drag_move_active() is True

    def test_start_position_cleared(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._drag_move_start_mm is None

    def test_offset_reset(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._drag_move_offset_mm == (0.0, 0.0)


# ===========================================================================
# 2. set_drag_move_active(True) disables mask_paint and shape_draw
# ===========================================================================


class TestDragMoveDisablesConflictingModes:
    def test_disables_mask_paint(self, canvas_widget):
        canvas_widget._mask_paint_active = True
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._mask_paint_active is False

    def test_disables_shape_draw(self, canvas_widget):
        canvas_widget._shape_draw_active = True
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._shape_draw_active is False

    def test_both_disabled_simultaneously(self, canvas_widget):
        canvas_widget._mask_paint_active = True
        canvas_widget._shape_draw_active = True
        canvas_widget.set_drag_move_active(True)
        assert canvas_widget._mask_paint_active is False
        assert canvas_widget._shape_draw_active is False


# ===========================================================================
# 3. set_drag_move_active(False) resets state
# ===========================================================================


class TestSetDragMoveActiveFalse:
    def test_flag_cleared(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        canvas_widget.set_drag_move_active(False)
        assert canvas_widget._drag_move_active is False

    def test_is_drag_move_active_returns_false(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        canvas_widget.set_drag_move_active(False)
        assert canvas_widget.is_drag_move_active() is False

    def test_start_cleared_on_deactivate(self, canvas_widget):
        canvas_widget.set_drag_move_active(True)
        canvas_widget._drag_move_start_mm = (5.0, 5.0)
        canvas_widget.set_drag_move_active(False)
        assert canvas_widget._drag_move_start_mm is None

    def test_does_not_re_enable_mask_paint(self, canvas_widget):
        """Deactivating drag-move must NOT re-enable mask_paint that was already off."""
        canvas_widget._mask_paint_active = False
        canvas_widget.set_drag_move_active(True)
        canvas_widget.set_drag_move_active(False)
        assert canvas_widget._mask_paint_active is False


# ===========================================================================
# 4. _on_layer_move_finished translates paths via set_layer_paths
# ===========================================================================


class TestOnLayerMoveFinished:
    def test_paths_translated_by_offset(self, main_window, controller_with_paths):
        """Calling _on_layer_move_finished(dx, dy) shifts all path coordinates."""
        layer = controller_with_paths.current_project.active_layer
        assert layer is not None
        original_paths = [list(p) for p in layer.paths]

        dx, dy = 10.0, 5.0
        main_window._on_layer_move_finished(dx, dy)

        updated_layer = controller_with_paths.current_project.active_layer
        for orig_path, new_path in zip(original_paths, updated_layer.paths):
            for (ox, oy), (nx, ny) in zip(orig_path, new_path):
                assert nx == pytest.approx(ox + dx)
                assert ny == pytest.approx(oy + dy)

    def test_no_active_layer_is_noop(self, main_window, controller_with_paths):
        """If no active layer, the handler must return without error."""
        controller_with_paths._active_layer_id = None
        # Should not raise
        main_window._on_layer_move_finished(5.0, 3.0)

    def test_empty_paths_is_noop(self, main_window, controller_with_paths):
        """If the active layer has no paths, handler returns without error."""
        layer = controller_with_paths.current_project.active_layer
        controller_with_paths._raw_set_layer_paths(layer.id, [])
        # Should not raise
        main_window._on_layer_move_finished(5.0, 3.0)

    def test_zero_offset_leaves_paths_unchanged(self, main_window, controller_with_paths):
        """A (0, 0) offset should leave all coordinates identical."""
        layer = controller_with_paths.current_project.active_layer
        original_paths = [list(p) for p in layer.paths]

        main_window._on_layer_move_finished(0.0, 0.0)

        updated_layer = controller_with_paths.current_project.active_layer
        for orig_path, new_path in zip(original_paths, updated_layer.paths):
            for orig_pt, new_pt in zip(orig_path, new_path):
                assert new_pt[0] == pytest.approx(orig_pt[0])
                assert new_pt[1] == pytest.approx(orig_pt[1])

    def test_translation_is_undoable(self, main_window, controller_with_paths):
        """set_layer_paths is called with undo description 'Move Layer'."""
        layer = controller_with_paths.current_project.active_layer
        original_first_pt = layer.paths[0][0]

        main_window._on_layer_move_finished(20.0, 10.0)

        # Undo should revert to original coordinates
        controller_with_paths.undo_stack.undo()
        reverted_layer = controller_with_paths.current_project.active_layer
        assert reverted_layer.paths[0][0][0] == pytest.approx(original_first_pt[0])
        assert reverted_layer.paths[0][0][1] == pytest.approx(original_first_pt[1])


# ===========================================================================
# 5. Toolbar action exists and is checkable
# ===========================================================================


class TestToolbarMoveAction:
    def test_act_drag_move_exists(self, main_window):
        assert hasattr(main_window, "_act_drag_move")

    def test_act_drag_move_is_checkable(self, main_window):
        assert main_window._act_drag_move.isCheckable()

    def test_checking_action_activates_canvas(self, main_window):
        main_window._act_drag_move.setChecked(True)
        assert main_window._canvas.is_drag_move_active() is True

    def test_unchecking_action_deactivates_canvas(self, main_window):
        main_window._act_drag_move.setChecked(True)
        main_window._act_drag_move.setChecked(False)
        assert main_window._canvas.is_drag_move_active() is False
