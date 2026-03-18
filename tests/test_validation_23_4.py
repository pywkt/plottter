"""Tests for task 23.4 — Fix move sync for layers without persisted generator_info.

Covers:
(a) When generator_info is None and the settings panel snapshot has x_offset_mm /
    y_offset_mm, _on_layer_move_finished updates the offset params.
(b) Undo restores both paths AND offset params (back to snapshot with x_offset=0).
(c) When generator_info is None AND the settings panel snapshot is also None (no
    generator active), paths are still translated and no error occurs.
(d) The live snapshot is persisted to layer.generator_info before pushing the undo
    command (so subsequent drags accumulate correctly).
(e) Same scenario works for a 3D Scene generator snapshot (which gained x/y offset
    params in task 23.1).
(f) Same scenario works for an image-based generator snapshot (FlowImage, which
    gained x/y offset params in task 23.2).
(g) When generator_info is already set (the normal case), the live snapshot fallback
    is NOT used — existing generator_info is used as-is.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(generator_name: str, mode: str, x_offset: float = 0.0, y_offset: float = 0.0) -> dict:
    return {
        "generator_name": generator_name,
        "mode": mode,
        "params": {
            "x_offset_mm": x_offset,
            "y_offset_mm": y_offset,
            "text": "Hello",  # extra param that should be preserved
        },
        "transforms": {},
    }


def _make_project_with_null_gen_info() -> Project:
    """Layer with paths but generator_info=None (not yet switched away)."""
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    layer.paths = [[(0.0, 0.0), (10.0, 0.0)], [(5.0, 5.0), (15.0, 5.0)]]
    layer.generator_info = None
    proj.add_layer(layer)
    return proj


def _make_project_with_gen_info(x_offset: float = 0.0, y_offset: float = 0.0) -> Project:
    """Layer with paths and pre-set generator_info (normal case)."""
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    layer.paths = [[(0.0, 0.0), (10.0, 0.0)], [(5.0, 5.0), (15.0, 5.0)]]
    layer.generator_info = _make_snapshot("Text", "Math Art", x_offset, y_offset)
    proj.add_layer(layer)
    return proj


@pytest.fixture
def controller_null(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project_with_null_gen_info())


@pytest.fixture
def controller_with_gen_info(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project_with_gen_info())


@pytest.fixture
def main_window_null(controller_null, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller_null)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


@pytest.fixture
def main_window_with_gen_info(controller_with_gen_info, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller_with_gen_info)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ===========================================================================
# (a) Live snapshot fallback: generator_info is None, snapshot has offsets
# ===========================================================================


class TestLiveSnapshotFallback:
    def test_x_offset_updated_when_gen_info_is_none(self, main_window_null, controller_null):
        """When generator_info is None and snapshot has x_offset_mm, it's updated by drag."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(20.0, 0.0)

        layer = controller_null.current_project.active_layer
        assert layer is not None
        assert layer.generator_info is not None
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(20.0)

    def test_y_offset_updated_when_gen_info_is_none(self, main_window_null, controller_null):
        """When generator_info is None and snapshot has y_offset_mm, it's updated by drag."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(0.0, 15.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(15.0)

    def test_both_offsets_updated_when_gen_info_is_none(self, main_window_null, controller_null):
        """Both x and y offsets are updated when generator_info was None."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(5.0, 8.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(5.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(8.0)

    def test_paths_also_translated_with_null_gen_info(self, main_window_null, controller_null):
        """Paths are translated even when starting from generator_info=None."""
        layer = controller_null.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        snapshot = _make_snapshot("Text", "Math Art")
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(10.0, 5.0)

        layer = controller_null.current_project.active_layer
        for orig_path, new_path in zip(original_paths, layer.paths):
            for (ox, oy), (nx, ny) in zip(orig_path, new_path):
                assert nx == pytest.approx(ox + 10.0)
                assert ny == pytest.approx(oy + 5.0)

    def test_extra_params_preserved_from_snapshot(self, main_window_null, controller_null):
        """Non-offset params from the live snapshot are preserved in generator_info."""
        snapshot = _make_snapshot("Text", "Math Art")
        snapshot["params"]["text"] = "World"
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(5.0, 0.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["text"] == "World"
        assert layer.generator_info["generator_name"] == "Text"


# ===========================================================================
# (b) Undo restores paths AND offset params after fallback
# ===========================================================================


class TestUndoAfterFallback:
    def test_undo_restores_paths(self, main_window_null, controller_null):
        """Undo reverts path coordinates to pre-drag state."""
        layer = controller_null.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        snapshot = _make_snapshot("Text", "Math Art")
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(20.0, 10.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        for orig_path, reverted_path in zip(original_paths, layer.paths):
            for (ox, oy), (rx, ry) in zip(orig_path, reverted_path):
                assert rx == pytest.approx(ox)
                assert ry == pytest.approx(oy)

    def test_undo_restores_x_offset_to_zero(self, main_window_null, controller_null):
        """Undo reverts x_offset_mm to the original snapshot value (0.0)."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(20.0, 0.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(0.0)

    def test_undo_restores_y_offset_to_zero(self, main_window_null, controller_null):
        """Undo reverts y_offset_mm to the original snapshot value (0.0)."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(0.0, 15.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(0.0)

    def test_redo_reapplies_offsets_after_undo(self, main_window_null, controller_null):
        """Redo after undo reapplies the updated offset values."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(20.0, 0.0)
        controller_null.undo_stack.undo()
        controller_null.undo_stack.redo()

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(20.0)


# ===========================================================================
# (c) Snapshot is None — only path translation, no error
# ===========================================================================


class TestNullSnapshotFallback:
    def test_paths_translated_when_snapshot_is_none(self, main_window_null, controller_null):
        """When both generator_info and snapshot are None, paths are still translated."""
        layer = controller_null.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=None)

        main_window_null._on_layer_move_finished(10.0, 5.0)

        layer = controller_null.current_project.active_layer
        for orig_path, new_path in zip(original_paths, layer.paths):
            for (ox, oy), (nx, ny) in zip(orig_path, new_path):
                assert nx == pytest.approx(ox + 10.0)
                assert ny == pytest.approx(oy + 5.0)

    def test_generator_info_stays_none_when_snapshot_is_none(self, main_window_null, controller_null):
        """When snapshot is None, generator_info is not set (remains None)."""
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=None)

        main_window_null._on_layer_move_finished(10.0, 5.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info is None

    def test_no_error_when_snapshot_returns_none(self, main_window_null):
        """No exception is raised when the settings panel returns None."""
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=None)
        # Should not raise
        main_window_null._on_layer_move_finished(5.0, 3.0)

    def test_undo_still_works_when_snapshot_is_none(self, main_window_null, controller_null):
        """Undo reverts path translation even when no generator_info was set."""
        layer = controller_null.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=None)
        main_window_null._on_layer_move_finished(10.0, 5.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        for orig_path, reverted_path in zip(original_paths, layer.paths):
            for (ox, oy), (rx, ry) in zip(orig_path, reverted_path):
                assert rx == pytest.approx(ox)
                assert ry == pytest.approx(oy)


# ===========================================================================
# (d) Live snapshot is persisted to layer.generator_info before undo push
# ===========================================================================


class TestSnapshotPersistedBeforeUndo:
    def test_snapshot_is_set_on_layer_before_command_push(self, main_window_null, controller_null):
        """After drag, layer.generator_info is set to the snapshot (or updated snapshot)."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(10.0, 0.0)

        layer = controller_null.current_project.active_layer
        # After drag, generator_info should be set (with updated offset)
        assert layer.generator_info is not None
        assert "x_offset_mm" in layer.generator_info["params"]

    def test_second_drag_uses_persisted_gen_info(self, main_window_null, controller_null):
        """Second drag accumulates correctly because first drag persisted generator_info."""
        snapshot = _make_snapshot("Text", "Math Art", x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        # First drag: should fallback to snapshot and set x_offset=10
        main_window_null._on_layer_move_finished(10.0, 0.0)

        # Second drag: generator_info is now set (x_offset=10), should go to 15
        main_window_null._on_layer_move_finished(5.0, 0.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(15.0)


# ===========================================================================
# (e) 3D Scene generator snapshot (added x/y offset in 23.1)
# ===========================================================================


class TestLiveSnapshotFallbackWithScene3D:
    def _make_3d_snapshot(self, x_offset: float = 0.0, y_offset: float = 0.0) -> dict:
        return {
            "generator_name": "3D Scene",
            "mode": "3D Scene",
            "params": {
                "x_offset_mm": x_offset,
                "y_offset_mm": y_offset,
                "shape_type": "Sphere",
                "sphere_detail": 24,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
            },
            "transforms": {},
        }

    def test_3d_offsets_updated_from_null_gen_info(self, main_window_null, controller_null):
        """3D Scene generator: offset params updated via live snapshot fallback."""
        snapshot = self._make_3d_snapshot(x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(30.0, 15.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(30.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(15.0)

    def test_3d_undo_reverts_offsets(self, main_window_null, controller_null):
        """Undo restores x/y offset to 0 for 3D Scene generator."""
        snapshot = self._make_3d_snapshot()
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(20.0, 10.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(0.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(0.0)

    def test_3d_other_params_not_modified(self, main_window_null, controller_null):
        """3D shape params (pos_x etc.) are preserved after drag."""
        snapshot = self._make_3d_snapshot()
        snapshot["params"]["pos_x"] = 5.0
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(10.0, 0.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["pos_x"] == pytest.approx(5.0)


# ===========================================================================
# (f) Image-based generator snapshot (FlowImage gained x/y offsets in 23.2)
# ===========================================================================


class TestLiveSnapshotFallbackWithImageGenerator:
    def _make_flow_snapshot(self, x_offset: float = 0.0, y_offset: float = 0.0) -> dict:
        return {
            "generator_name": "Flow Image",
            "mode": "Image to Lines",
            "params": {
                "x_offset_mm": x_offset,
                "y_offset_mm": y_offset,
                "num_lines": 100,
                "amplitude": 5.0,
                "mode": "squiggle",
            },
            "transforms": {},
        }

    def test_flow_image_offsets_updated_from_null_gen_info(self, main_window_null, controller_null):
        """FlowImage generator: offset params updated via live snapshot fallback."""
        snapshot = self._make_flow_snapshot(x_offset=0.0, y_offset=0.0)
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(12.0, 7.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(12.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(7.0)

    def test_flow_image_undo_reverts_offsets(self, main_window_null, controller_null):
        """Undo restores x/y offset to 0 for FlowImage generator."""
        snapshot = self._make_flow_snapshot()
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(12.0, 7.0)
        controller_null.undo_stack.undo()

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(0.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(0.0)

    def test_flow_image_extra_params_preserved(self, main_window_null, controller_null):
        """Non-offset params like num_lines and amplitude are preserved after drag."""
        snapshot = self._make_flow_snapshot()
        snapshot["params"]["num_lines"] = 200
        main_window_null._settings_panel._get_settings_snapshot = MagicMock(return_value=snapshot)

        main_window_null._on_layer_move_finished(5.0, 0.0)

        layer = controller_null.current_project.active_layer
        assert layer.generator_info["params"]["num_lines"] == 200


# ===========================================================================
# (g) Normal case: generator_info already set — live snapshot NOT used
# ===========================================================================


class TestExistingGenInfoNotOverridden:
    def test_existing_gen_info_x_offset_updated(self, main_window_with_gen_info, controller_with_gen_info):
        """When generator_info is already set, x_offset_mm is updated from it (not snapshot)."""
        layer = controller_with_gen_info.current_project.active_layer
        original_x = layer.generator_info["params"]["x_offset_mm"]

        # Mock snapshot to return different values — should NOT be used
        different_snapshot = _make_snapshot("Different", "Math Art", x_offset=999.0, y_offset=999.0)
        main_window_with_gen_info._settings_panel._get_settings_snapshot = MagicMock(
            return_value=different_snapshot
        )

        main_window_with_gen_info._on_layer_move_finished(10.0, 0.0)

        layer = controller_with_gen_info.current_project.active_layer
        # Should use existing gen_info, not the mocked snapshot
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(original_x + 10.0)
        # generator_name should still be "Text" (from original gen_info)
        assert layer.generator_info["generator_name"] == "Text"

    def test_live_snapshot_not_called_when_gen_info_exists(self, main_window_with_gen_info):
        """_get_settings_snapshot is not called if generator_info is already set."""
        mock_snapshot = MagicMock(return_value=_make_snapshot("Text", "Math Art"))
        main_window_with_gen_info._settings_panel._get_settings_snapshot = mock_snapshot

        main_window_with_gen_info._on_layer_move_finished(10.0, 0.0)

        # The mock should NOT have been called
        mock_snapshot.assert_not_called()

    def test_undo_restores_existing_gen_info(self, main_window_with_gen_info, controller_with_gen_info):
        """Undo reverts x_offset_mm to original value from existing generator_info."""
        layer = controller_with_gen_info.current_project.active_layer
        original_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_gen_info._on_layer_move_finished(10.0, 0.0)
        controller_with_gen_info.undo_stack.undo()

        layer = controller_with_gen_info.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(original_x)
