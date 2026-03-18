"""Tests for task 22.1 — Sync x/y offset parameters when layer is dragged.

Covers:
(a) Drag a layer whose generator has x_offset_mm / y_offset_mm — verify
    generator_info params are updated by the drag delta.
(b) Undo the drag — verify both paths AND offset parameters revert.
(c) Drag a layer whose generator has no offset params (e.g. flow image) —
    verify no error occurs and path translation still works.
(d) Multiple consecutive drags accumulate offsets correctly.
(e) MoveLayerCommand redo/undo correctly updates paths + generator_info.
(f) generator_info_changed signal fires when _raw_set_layer_generator_info is called.
"""

from __future__ import annotations

import copy

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project_with_gen_info(
    x_offset: float = 0.0, y_offset: float = 0.0, has_offsets: bool = True
) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    layer = Layer(name="Text Layer", color="#000000")
    layer.paths = [[(10.0, 20.0), (30.0, 20.0)]]
    if has_offsets:
        layer.generator_info = {
            "generator_name": "Text",
            "mode": "Math Art",
            "params": {
                "x_offset_mm": x_offset,
                "y_offset_mm": y_offset,
                "text": "Hello",
            },
            "transforms": {},
        }
    else:
        # Flow Image generator — no x_offset_mm / y_offset_mm
        layer.generator_info = {
            "generator_name": "Flow Image",
            "mode": "Image to Lines",
            "params": {
                "num_lines": 100,
                "amplitude": 5.0,
            },
            "transforms": {},
        }
    proj.add_layer(layer)
    return proj


@pytest.fixture
def controller_with_offsets(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project_with_gen_info())


@pytest.fixture
def controller_no_offsets(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project_with_gen_info(has_offsets=False))


@pytest.fixture
def main_window_with_offsets(controller_with_offsets, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller_with_offsets)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


@pytest.fixture
def main_window_no_offsets(controller_no_offsets, qtbot):
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller_no_offsets)
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ===========================================================================
# (a) Drag with offset params — generator_info is updated
# ===========================================================================


class TestDragUpdatesGeneratorInfo:
    def test_x_offset_updated_by_drag_delta(self, main_window_with_offsets, controller_with_offsets):
        """After drag(20, 0), x_offset_mm in generator_info increments by 20."""
        layer = controller_with_offsets.current_project.active_layer
        assert layer is not None
        old_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(20.0, 0.0)

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(old_x + 20.0)

    def test_y_offset_updated_by_drag_delta(self, main_window_with_offsets, controller_with_offsets):
        """After drag(0, 15), y_offset_mm in generator_info increments by 15."""
        layer = controller_with_offsets.current_project.active_layer
        old_y = layer.generator_info["params"]["y_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(0.0, 15.0)

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(old_y + 15.0)

    def test_both_offsets_updated_together(self, main_window_with_offsets, controller_with_offsets):
        """Both x and y offsets are updated in a single drag."""
        layer = controller_with_offsets.current_project.active_layer
        old_x = layer.generator_info["params"]["x_offset_mm"]
        old_y = layer.generator_info["params"]["y_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(-5.0, 8.0)

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(old_x - 5.0)
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(old_y + 8.0)

    def test_paths_are_also_translated(self, main_window_with_offsets, controller_with_offsets):
        """The drag also translates all path coordinates."""
        layer = controller_with_offsets.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_with_offsets._on_layer_move_finished(10.0, 5.0)

        layer = controller_with_offsets.current_project.active_layer
        for orig_path, new_path in zip(original_paths, layer.paths):
            for (ox, oy), (nx, ny) in zip(orig_path, new_path):
                assert nx == pytest.approx(ox + 10.0)
                assert ny == pytest.approx(oy + 5.0)

    def test_other_params_not_modified(self, main_window_with_offsets, controller_with_offsets):
        """Only x_offset_mm and y_offset_mm change; other params are untouched."""
        main_window_with_offsets._on_layer_move_finished(5.0, 5.0)
        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["text"] == "Hello"
        assert layer.generator_info["generator_name"] == "Text"


# ===========================================================================
# (b) Undo reverts both paths AND offset parameters
# ===========================================================================


class TestUndoDragRevertsOffsets:
    def test_undo_restores_x_offset(self, main_window_with_offsets, controller_with_offsets):
        """Undo reverts x_offset_mm to its pre-drag value."""
        layer = controller_with_offsets.current_project.active_layer
        original_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(20.0, 0.0)
        controller_with_offsets.undo_stack.undo()

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(original_x)

    def test_undo_restores_y_offset(self, main_window_with_offsets, controller_with_offsets):
        """Undo reverts y_offset_mm to its pre-drag value."""
        layer = controller_with_offsets.current_project.active_layer
        original_y = layer.generator_info["params"]["y_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(0.0, 15.0)
        controller_with_offsets.undo_stack.undo()

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(original_y)

    def test_undo_restores_paths(self, main_window_with_offsets, controller_with_offsets):
        """Undo also reverts path coordinates to their pre-drag values."""
        layer = controller_with_offsets.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_with_offsets._on_layer_move_finished(20.0, 10.0)
        controller_with_offsets.undo_stack.undo()

        layer = controller_with_offsets.current_project.active_layer
        for orig_path, reverted_path in zip(original_paths, layer.paths):
            for (ox, oy), (rx, ry) in zip(orig_path, reverted_path):
                assert rx == pytest.approx(ox)
                assert ry == pytest.approx(oy)

    def test_redo_reapplies_offsets(self, main_window_with_offsets, controller_with_offsets):
        """Redo after undo reapplies the updated offsets."""
        layer = controller_with_offsets.current_project.active_layer
        original_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(20.0, 0.0)
        controller_with_offsets.undo_stack.undo()
        controller_with_offsets.undo_stack.redo()

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(original_x + 20.0)


# ===========================================================================
# (c) Drag a layer with no offset params — only path translation happens
# ===========================================================================


class TestDragWithoutOffsetParams:
    def test_paths_translated_without_error(self, main_window_no_offsets, controller_no_offsets):
        """Flow Image layer (no offset params) drag translates paths without error."""
        layer = controller_no_offsets.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_no_offsets._on_layer_move_finished(10.0, 5.0)

        layer = controller_no_offsets.current_project.active_layer
        for orig_path, new_path in zip(original_paths, layer.paths):
            for (ox, oy), (nx, ny) in zip(orig_path, new_path):
                assert nx == pytest.approx(ox + 10.0)
                assert ny == pytest.approx(oy + 5.0)

    def test_generator_info_unchanged_when_no_offset_params(
        self, main_window_no_offsets, controller_no_offsets
    ):
        """For a generator without offset params, generator_info is not modified."""
        layer = controller_no_offsets.current_project.active_layer
        original_info = copy.deepcopy(layer.generator_info)

        main_window_no_offsets._on_layer_move_finished(10.0, 5.0)

        layer = controller_no_offsets.current_project.active_layer
        assert layer.generator_info == original_info

    def test_undo_still_reverts_paths(self, main_window_no_offsets, controller_no_offsets):
        """Undo works even when no offset params were updated."""
        layer = controller_no_offsets.current_project.active_layer
        original_paths = copy.deepcopy(layer.paths)

        main_window_no_offsets._on_layer_move_finished(10.0, 5.0)
        controller_no_offsets.undo_stack.undo()

        layer = controller_no_offsets.current_project.active_layer
        for orig_path, reverted_path in zip(original_paths, layer.paths):
            for (ox, oy), (rx, ry) in zip(orig_path, reverted_path):
                assert rx == pytest.approx(ox)
                assert ry == pytest.approx(oy)


# ===========================================================================
# (d) Multiple drags accumulate offsets
# ===========================================================================


class TestMultipleDragsAccumulate:
    def test_two_drags_accumulate_x(self, main_window_with_offsets, controller_with_offsets):
        """Drag 10mm right, then 5mm left → net x_offset = +5mm from initial."""
        layer = controller_with_offsets.current_project.active_layer
        start_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(10.0, 0.0)
        main_window_with_offsets._on_layer_move_finished(-5.0, 0.0)

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(start_x + 5.0)

    def test_two_drags_accumulate_y(self, main_window_with_offsets, controller_with_offsets):
        """Two vertical drags accumulate correctly in y_offset_mm."""
        layer = controller_with_offsets.current_project.active_layer
        start_y = layer.generator_info["params"]["y_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(0.0, 3.0)
        main_window_with_offsets._on_layer_move_finished(0.0, 7.0)

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["y_offset_mm"] == pytest.approx(start_y + 10.0)

    def test_undo_one_of_two_drags(self, main_window_with_offsets, controller_with_offsets):
        """Undoing one drag of two leaves the other drag's offsets intact."""
        layer = controller_with_offsets.current_project.active_layer
        start_x = layer.generator_info["params"]["x_offset_mm"]

        main_window_with_offsets._on_layer_move_finished(10.0, 0.0)
        main_window_with_offsets._on_layer_move_finished(5.0, 0.0)

        controller_with_offsets.undo_stack.undo()  # undo second drag

        layer = controller_with_offsets.current_project.active_layer
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(start_x + 10.0)


# ===========================================================================
# (e) MoveLayerCommand directly
# ===========================================================================


class TestMoveLayerCommand:
    def _make_controller(self, qapp):
        from plottter.gui.project_controller import ProjectController
        return ProjectController(_make_project_with_gen_info())

    def test_redo_updates_paths(self, qapp):
        from plottter.gui.commands import MoveLayerCommand
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(_make_project_with_gen_info())
        layer = ctrl.current_project.active_layer
        old_paths = [list(p) for p in layer.paths]
        new_paths = [[(x + 5.0, y + 3.0) for x, y in p] for p in old_paths]
        old_info = copy.deepcopy(layer.generator_info)
        new_info = copy.deepcopy(old_info)
        new_info["params"]["x_offset_mm"] += 5.0
        new_info["params"]["y_offset_mm"] += 3.0

        cmd = MoveLayerCommand(ctrl, layer.id, new_paths, old_paths, new_info, old_info)
        cmd.redo()

        layer = ctrl.current_project.get_layer(layer.id)
        assert layer.paths[0][0][0] == pytest.approx(old_paths[0][0][0] + 5.0)
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(
            old_info["params"]["x_offset_mm"] + 5.0
        )

    def test_undo_restores_paths_and_info(self, qapp):
        from plottter.gui.commands import MoveLayerCommand
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(_make_project_with_gen_info())
        layer = ctrl.current_project.active_layer
        old_paths = [list(p) for p in layer.paths]
        new_paths = [[(x + 5.0, y + 3.0) for x, y in p] for p in old_paths]
        old_info = copy.deepcopy(layer.generator_info)
        new_info = copy.deepcopy(old_info)
        new_info["params"]["x_offset_mm"] += 5.0
        new_info["params"]["y_offset_mm"] += 3.0

        cmd = MoveLayerCommand(ctrl, layer.id, new_paths, old_paths, new_info, old_info)
        cmd.redo()
        cmd.undo()

        layer = ctrl.current_project.get_layer(layer.id)
        assert layer.paths[0][0][0] == pytest.approx(old_paths[0][0][0])
        assert layer.generator_info["params"]["x_offset_mm"] == pytest.approx(
            old_info["params"]["x_offset_mm"]
        )

    def test_no_gen_info_update_when_none(self, qapp):
        """When new_generator_info is None, generator_info stays unchanged after redo."""
        from plottter.gui.commands import MoveLayerCommand
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(_make_project_with_gen_info())
        layer = ctrl.current_project.active_layer
        old_paths = [list(p) for p in layer.paths]
        new_paths = [[(x + 5.0, y) for x, y in p] for p in old_paths]
        original_info = copy.deepcopy(layer.generator_info)

        cmd = MoveLayerCommand(ctrl, layer.id, new_paths, old_paths, None, None)
        cmd.redo()

        layer = ctrl.current_project.get_layer(layer.id)
        assert layer.generator_info == original_info


# ===========================================================================
# (f) generator_info_changed signal fires from _raw_set_layer_generator_info
# ===========================================================================


class TestGeneratorInfoChangedSignal:
    def test_signal_emitted_on_raw_set(self, qapp, qtbot):
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(_make_project_with_gen_info())
        layer = ctrl.current_project.active_layer

        emitted_ids = []
        ctrl.generator_info_changed.connect(emitted_ids.append)

        new_info = copy.deepcopy(layer.generator_info)
        ctrl._raw_set_layer_generator_info(layer.id, new_info)

        assert layer.id in emitted_ids

    def test_signal_not_emitted_by_set_layer_generator_info(self, qapp):
        """The public set_layer_generator_info (used for layer switches) does NOT
        emit generator_info_changed — only the _raw_ variant does."""
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(_make_project_with_gen_info())
        layer = ctrl.current_project.active_layer

        emitted_ids = []
        ctrl.generator_info_changed.connect(emitted_ids.append)

        ctrl.set_layer_generator_info(layer.id, layer.generator_info)

        assert emitted_ids == []
