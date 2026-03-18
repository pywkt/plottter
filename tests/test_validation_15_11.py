"""Validation tests for task 15.11 — Undo/Redo system.

Exercises undo/redo for every undoable action:
  - add layer
  - remove layer
  - reorder layer
  - generate (set_layer_paths with content)
  - clear (set_layer_paths with empty list)
  - merge layers
  - duplicate layer
  - layer property changes (name, color, visible, locked, opacity)
  - canvas change
  - color separation (macro)

Also verifies:
  - redo after undo restores the exact state
  - undo stack clears on project load
  - undo stack clears on new project
"""

from __future__ import annotations

import copy
import math
import sys

import pytest

from plottter.models import Canvas, Layer, Project
from plottter.models.path import Polyline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_polyline(n: int = 5, offset: float = 0.0) -> Polyline:
    """Return a synthetic n-point polyline."""
    return [(offset + i * 5.0, offset + math.sin(i) * 10.0) for i in range(n)]


def _make_paths(num_paths: int = 3) -> list[Polyline]:
    return [_make_polyline(5, float(i * 20)) for i in range(num_paths)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(qapp):
    return qapp


@pytest.fixture
def canvas():
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def project(canvas):
    p = Project(name="Test Project", canvas=canvas)
    p.add_layer(Layer(name="Layer 1", color="#000000"))
    return p


@pytest.fixture
def controller(project, app):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(project)


# ---------------------------------------------------------------------------
# 1. Add Layer — undo removes the added layer; redo puts it back
# ---------------------------------------------------------------------------

class TestAddLayerUndo:
    def test_undo_removes_added_layer(self, controller):
        initial_count = len(controller.current_project.layers)
        layer = controller.add_layer()
        assert len(controller.current_project.layers) == initial_count + 1

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count
        assert controller.get_layer(layer.id) is None

    def test_redo_restores_added_layer(self, controller):
        initial_count = len(controller.current_project.layers)
        layer = controller.add_layer()
        layer_id = layer.id

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count

        controller.undo_stack.redo()
        assert len(controller.current_project.layers) == initial_count + 1
        assert controller.get_layer(layer_id) is not None

    def test_undo_redo_preserves_layer_name(self, controller):
        layer = controller.add_layer()
        # The layer name after undo/redo is the name assigned by add_layer
        name = layer.name
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        restored = controller.get_layer(layer.id)
        assert restored is not None
        assert restored.name == name

    def test_undo_redo_preserves_layer_color(self, controller):
        layer = controller.add_layer()
        color = layer.color
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        restored = controller.get_layer(layer.id)
        assert restored.color == color


# ---------------------------------------------------------------------------
# 2. Remove Layer — undo restores removed layer at exact position
# ---------------------------------------------------------------------------

class TestRemoveLayerUndo:
    def test_undo_restores_removed_layer(self, controller):
        # Add a second layer so we have 2 layers
        layer_b = controller.add_layer()
        initial_ids = [l.id for l in controller.current_project.layers]

        controller.remove_layer(layer_b.id)
        assert controller.get_layer(layer_b.id) is None

        controller.undo_stack.undo()
        assert controller.get_layer(layer_b.id) is not None

    def test_undo_restores_layer_at_original_index(self, controller):
        # Layer 0: "Layer 1", add Layer B at index 1
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id
        # Position 1 = index 1
        assert controller.current_project.layers[1].id == layer_b_id

        controller.undo_stack.undo()  # undo the add
        # Now undo add to get 1 layer again
        # Remove layer 1 (the original one)
        layer1_id = controller.current_project.layers[0].id
        controller.remove_layer(layer1_id)
        assert len(controller.current_project.layers) == 0

        controller.undo_stack.undo()
        # Should restore the removed layer at index 0
        assert len(controller.current_project.layers) == 1
        assert controller.current_project.layers[0].id == layer1_id

    def test_undo_restores_layer_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(3)
        controller.set_layer_paths(layer_id, paths)

        controller.remove_layer(layer_id)
        assert controller.get_layer(layer_id) is None

        controller.undo_stack.undo()
        restored = controller.get_layer(layer_id)
        assert restored is not None
        assert len(restored.paths) == 3

    def test_redo_removes_layer_again(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.add_layer()  # ensure we can remove without going to 0

        controller.remove_layer(layer_id)
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id) is not None

        controller.undo_stack.redo()
        assert controller.get_layer(layer_id) is None

    def test_undo_restores_layer_properties(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_name(layer_id, "My Art Layer")
        controller.set_layer_color(layer_id, "#FF0000")
        controller.add_layer()  # need 2 layers so remove doesn't fail

        controller.remove_layer(layer_id)
        controller.undo_stack.undo()

        restored = controller.get_layer(layer_id)
        assert restored is not None
        assert restored.name == "My Art Layer"
        assert restored.color == "#FF0000"


# ---------------------------------------------------------------------------
# 3. Reorder Layer — undo restores original order; redo reorders again
# ---------------------------------------------------------------------------

class TestReorderLayerUndo:
    def test_undo_restores_order(self, controller):
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id
        layer1_id = controller.current_project.layers[0].id

        # Move layer_b from index 1 to index 0
        controller.reorder_layer(layer_b_id, 0)
        assert controller.current_project.layers[0].id == layer_b_id

        controller.undo_stack.undo()
        assert controller.current_project.layers[0].id == layer1_id

    def test_redo_restores_reorder(self, controller):
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id

        controller.reorder_layer(layer_b_id, 0)
        controller.undo_stack.undo()

        # After undo, layer_b should be at index 1 again
        assert controller.current_project.layers[1].id == layer_b_id

        controller.undo_stack.redo()
        assert controller.current_project.layers[0].id == layer_b_id

    def test_multiple_reorders_undo_in_order(self, controller):
        layer_b = controller.add_layer()
        layer_c = controller.add_layer()
        layer_b_id = layer_b.id
        layer_c_id = layer_c.id

        # Record initial state
        initial_ids = [l.id for l in controller.current_project.layers]

        controller.reorder_layer(layer_b_id, 0)
        controller.reorder_layer(layer_c_id, 0)

        # Undo both in reverse
        controller.undo_stack.undo()
        controller.undo_stack.undo()

        current_ids = [l.id for l in controller.current_project.layers]
        assert current_ids == initial_ids


# ---------------------------------------------------------------------------
# 4. Generate (set_layer_paths with content) — undo clears; redo restores
# ---------------------------------------------------------------------------

class TestGenerateUndo:
    def test_undo_clears_generated_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(3)

        controller.set_layer_paths(layer_id, paths, description="Generate Parametric")
        assert len(controller.get_layer(layer_id).paths) == 3

        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 0

    def test_redo_restores_generated_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(3)

        controller.set_layer_paths(layer_id, paths, description="Generate")
        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 0

        controller.undo_stack.redo()
        assert len(controller.get_layer(layer_id).paths) == 3

    def test_undo_restores_previous_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        original_paths = _make_paths(2)
        new_paths = _make_paths(5)

        controller.set_layer_paths(layer_id, original_paths, description="First Generate")
        controller.set_layer_paths(layer_id, new_paths, description="Second Generate")
        assert len(controller.get_layer(layer_id).paths) == 5

        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 2

    def test_generated_path_coordinates_match_after_redo(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = [[(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]]

        controller.set_layer_paths(layer_id, paths)
        controller.undo_stack.undo()
        controller.undo_stack.redo()

        restored_paths = controller.get_layer(layer_id).paths
        assert len(restored_paths) == 1
        assert restored_paths[0] == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


# ---------------------------------------------------------------------------
# 5. Clear (set_layer_paths with empty list)
# ---------------------------------------------------------------------------

class TestClearPathsUndo:
    def test_undo_restores_cleared_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(4)

        controller.set_layer_paths(layer_id, paths)
        controller.set_layer_paths(layer_id, [], description="Clear Layer")
        assert len(controller.get_layer(layer_id).paths) == 0

        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 4

    def test_redo_clears_again(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(4)

        controller.set_layer_paths(layer_id, paths)
        controller.set_layer_paths(layer_id, [], description="Clear Layer")
        controller.undo_stack.undo()  # restores paths
        assert len(controller.get_layer(layer_id).paths) == 4

        controller.undo_stack.redo()  # clears again
        assert len(controller.get_layer(layer_id).paths) == 0

    def test_undo_after_clear_preserves_exact_path_data(self, controller):
        layer_id = controller.current_project.layers[0].id
        original_path = [(1.0, 2.0), (3.0, 4.0)]
        controller.set_layer_paths(layer_id, [original_path])
        controller.set_layer_paths(layer_id, [])

        controller.undo_stack.undo()
        restored = controller.get_layer(layer_id).paths
        assert len(restored) == 1
        assert restored[0] == original_path


# ---------------------------------------------------------------------------
# 6. Merge Layers — undo restores source layers; redo re-merges
# ---------------------------------------------------------------------------

class TestMergeLayersUndo:
    def test_undo_restores_source_layers(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id

        controller.merge_layers([layer_a_id, layer_b_id])
        assert len(controller.current_project.layers) == 1
        # Both source layers gone
        assert controller.get_layer(layer_a_id) is None
        assert controller.get_layer(layer_b_id) is None

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == 2
        assert controller.get_layer(layer_a_id) is not None
        assert controller.get_layer(layer_b_id) is not None

    def test_redo_re_merges(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id

        merged = controller.merge_layers([layer_a_id, layer_b_id])
        merged_id = merged.id

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == 2

        controller.undo_stack.redo()
        assert len(controller.current_project.layers) == 1
        assert controller.get_layer(merged_id) is not None

    def test_undo_restores_source_layers_at_correct_positions(self, controller):
        # layers: [A, B, C]
        layer_a_id = controller.current_project.layers[0].id
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id
        layer_c = controller.add_layer()
        layer_c_id = layer_c.id

        # Merge A and C (non-adjacent)
        controller.merge_layers([layer_a_id, layer_c_id])

        controller.undo_stack.undo()
        ids = [l.id for l in controller.current_project.layers]
        assert ids[0] == layer_a_id
        assert ids[1] == layer_b_id
        assert ids[2] == layer_c_id

    def test_undo_restores_source_layer_paths(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        paths_a = _make_paths(2)
        controller.set_layer_paths(layer_a_id, paths_a)

        layer_b = controller.add_layer()
        layer_b_id = layer_b.id
        paths_b = _make_paths(3)
        controller.set_layer_paths(layer_b_id, paths_b)

        controller.merge_layers([layer_a_id, layer_b_id])
        controller.undo_stack.undo()

        assert len(controller.get_layer(layer_a_id).paths) == 2
        assert len(controller.get_layer(layer_b_id).paths) == 3

    def test_merged_layer_combines_paths(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        paths_a = _make_paths(2)
        controller.set_layer_paths(layer_a_id, paths_a)

        layer_b = controller.add_layer()
        layer_b_id = layer_b.id
        paths_b = _make_paths(3)
        controller.set_layer_paths(layer_b_id, paths_b)

        merged = controller.merge_layers([layer_a_id, layer_b_id])
        assert len(merged.paths) == 5


# ---------------------------------------------------------------------------
# 7. Duplicate Layer — undo removes duplicate; redo restores it
# ---------------------------------------------------------------------------

class TestDuplicateLayerUndo:
    def test_undo_removes_duplicate(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(2)
        controller.set_layer_paths(layer_id, paths)

        initial_count = len(controller.current_project.layers)
        dup = controller.duplicate_layer(layer_id)
        assert len(controller.current_project.layers) == initial_count + 1

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count
        assert controller.get_layer(dup.id) is None

    def test_redo_restores_duplicate(self, controller):
        layer_id = controller.current_project.layers[0].id
        initial_count = len(controller.current_project.layers)

        dup = controller.duplicate_layer(layer_id)
        dup_id = dup.id

        controller.undo_stack.undo()
        controller.undo_stack.redo()

        assert len(controller.current_project.layers) == initial_count + 1
        assert controller.get_layer(dup_id) is not None

    def test_duplicate_preserves_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(3)
        controller.set_layer_paths(layer_id, paths)

        dup = controller.duplicate_layer(layer_id)
        assert len(dup.paths) == 3

    def test_duplicate_is_independent_after_redo(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(2)
        controller.set_layer_paths(layer_id, paths)

        dup = controller.duplicate_layer(layer_id)
        dup_id = dup.id

        controller.undo_stack.undo()
        controller.undo_stack.redo()

        restored_dup = controller.get_layer(dup_id)
        assert restored_dup is not None
        assert len(restored_dup.paths) == 2


# ---------------------------------------------------------------------------
# 8. Layer Property Changes — name, color, visible, locked, opacity
# ---------------------------------------------------------------------------

class TestLayerNameUndo:
    def test_undo_restores_original_name(self, controller):
        layer_id = controller.current_project.layers[0].id
        original = controller.get_layer(layer_id).name

        controller.set_layer_name(layer_id, "New Name")
        assert controller.get_layer(layer_id).name == "New Name"

        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).name == original

    def test_redo_applies_name_change(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_name(layer_id, "New Name")
        controller.undo_stack.undo()

        controller.undo_stack.redo()
        assert controller.get_layer(layer_id).name == "New Name"


class TestLayerColorUndo:
    def test_undo_restores_original_color(self, controller):
        layer_id = controller.current_project.layers[0].id
        original = controller.get_layer(layer_id).color

        controller.set_layer_color(layer_id, "#FF0000")
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).color == original

    def test_redo_applies_color_change(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_color(layer_id, "#00FF00")
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        assert controller.get_layer(layer_id).color == "#00FF00"


class TestLayerVisibilityUndo:
    def test_undo_restores_visibility(self, controller):
        layer_id = controller.current_project.layers[0].id
        original = controller.get_layer(layer_id).visible

        controller.set_layer_visible(layer_id, not original)
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).visible == original

    def test_redo_applies_visibility(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_visible(layer_id, False)
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        assert controller.get_layer(layer_id).visible is False


class TestLayerLockUndo:
    def test_undo_restores_lock(self, controller):
        layer_id = controller.current_project.layers[0].id
        original = controller.get_layer(layer_id).locked

        controller.set_layer_locked(layer_id, not original)
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).locked == original

    def test_redo_applies_lock(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_locked(layer_id, True)
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        assert controller.get_layer(layer_id).locked is True


class TestLayerOpacityUndo:
    def test_undo_restores_opacity(self, controller):
        layer_id = controller.current_project.layers[0].id
        original = controller.get_layer(layer_id).opacity

        controller.set_layer_opacity(layer_id, 0.5)
        controller.undo_stack.undo()
        assert abs(controller.get_layer(layer_id).opacity - original) < 1e-6

    def test_redo_applies_opacity(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 0.3)
        controller.undo_stack.undo()
        controller.undo_stack.redo()
        assert abs(controller.get_layer(layer_id).opacity - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# 9. Canvas Change — undo restores original canvas; redo applies new
# ---------------------------------------------------------------------------

class TestCanvasChangeUndo:
    def test_undo_restores_original_canvas(self, controller):
        original_canvas = controller.current_project.canvas
        original_width = original_canvas.width_mm

        new_canvas = Canvas.from_preset("A3", margin=15.0)
        controller.set_canvas(new_canvas)
        assert controller.current_project.canvas.width_mm == new_canvas.width_mm

        controller.undo_stack.undo()
        assert abs(controller.current_project.canvas.width_mm - original_width) < 1e-6

    def test_redo_applies_canvas_change(self, controller):
        new_canvas = Canvas.from_preset("A3", margin=15.0)
        controller.set_canvas(new_canvas)
        controller.undo_stack.undo()

        controller.undo_stack.redo()
        assert abs(controller.current_project.canvas.width_mm - new_canvas.width_mm) < 1e-6

    def test_undo_restores_canvas_margins(self, controller):
        original_margin = controller.current_project.canvas.margin_mm

        new_canvas = Canvas(
            width_mm=210.0, height_mm=297.0, margin_mm=25.0, paper_preset="A4"
        )
        controller.set_canvas(new_canvas)
        controller.undo_stack.undo()

        assert abs(controller.current_project.canvas.margin_mm - original_margin) < 1e-6

    def test_undo_restores_canvas_paper_preset(self, controller):
        original_preset = controller.current_project.canvas.paper_preset

        new_canvas = Canvas.from_preset("A3", margin=10.0)
        controller.set_canvas(new_canvas)
        controller.undo_stack.undo()

        assert controller.current_project.canvas.paper_preset == original_preset


# ---------------------------------------------------------------------------
# 10. Color Separation (macro) — entire separation undone in one step
# ---------------------------------------------------------------------------

class TestColorSeparationUndo:
    """Color separation adds multiple layers inside a beginMacro/endMacro block.
    A single undo() call should reverse the entire separation atomically."""

    def _simulate_separation(self, controller, num_layers: int = 3) -> list[str]:
        """Simulate the settings panel's _on_separate() macro pattern."""
        controller.undo_stack.beginMacro("Separate Into Layers")
        added_ids = []
        for i in range(num_layers):
            layer = controller.add_layer(
                Layer(name=f"Cluster {i + 1}", color=f"#{i * 80:02X}0000")
            )
            added_ids.append(layer.id)
        controller.undo_stack.endMacro()
        return added_ids

    def test_undo_removes_all_separation_layers(self, controller):
        initial_count = len(controller.current_project.layers)
        added_ids = self._simulate_separation(controller, num_layers=3)
        assert len(controller.current_project.layers) == initial_count + 3

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count
        for layer_id in added_ids:
            assert controller.get_layer(layer_id) is None

    def test_redo_restores_all_separation_layers(self, controller):
        initial_count = len(controller.current_project.layers)
        added_ids = self._simulate_separation(controller, num_layers=3)

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count

        controller.undo_stack.redo()
        assert len(controller.current_project.layers) == initial_count + 3
        for layer_id in added_ids:
            assert controller.get_layer(layer_id) is not None

    def test_kmeans_style_separation_undoable(self, controller):
        """Simulate K-means separation creating layers with colored names."""
        initial_count = len(controller.current_project.layers)
        colors = ["#FF0000", "#00FF00", "#0000FF"]

        controller.undo_stack.beginMacro("Separate Into Layers")
        layer_ids = []
        for i, color in enumerate(colors):
            layer = controller.add_layer(
                Layer(name=f"Cluster {i + 1} — {color}", color=color)
            )
            layer_ids.append(layer.id)
        controller.undo_stack.endMacro()

        assert len(controller.current_project.layers) == initial_count + 3

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count

    def test_rgb_separation_creates_3_layers_undoable(self, controller):
        initial_count = len(controller.current_project.layers)
        rgb_colors = [("#FF0000", "Red Channel"), ("#00FF00", "Green Channel"), ("#0000FF", "Blue Channel")]

        controller.undo_stack.beginMacro("Separate Into Layers")
        layer_ids = []
        for color, name in rgb_colors:
            layer = controller.add_layer(Layer(name=name, color=color))
            layer_ids.append(layer.id)
        controller.undo_stack.endMacro()

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count
        for lid in layer_ids:
            assert controller.get_layer(lid) is None

    def test_cmyk_separation_creates_4_layers_undoable(self, controller):
        initial_count = len(controller.current_project.layers)
        cmyk = [
            ("#00FFFF", "Cyan"), ("#FF00FF", "Magenta"),
            ("#FFFF00", "Yellow"), ("#000000", "Key/Black"),
        ]

        controller.undo_stack.beginMacro("Separate Into Layers")
        layer_ids = []
        for color, name in cmyk:
            layer = controller.add_layer(Layer(name=name, color=color))
            layer_ids.append(layer.id)
        controller.undo_stack.endMacro()

        assert len(controller.current_project.layers) == initial_count + 4

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count

        controller.undo_stack.redo()
        assert len(controller.current_project.layers) == initial_count + 4


# ---------------------------------------------------------------------------
# 11. Undo stack clears on project load
# ---------------------------------------------------------------------------

class TestUndoStackClearsOnLoad:
    def test_stack_empty_after_load_project(self, controller):
        # Perform some undoable actions
        controller.add_layer()
        controller.add_layer()
        assert controller.undo_stack.canUndo()

        # Load a fresh project
        new_canvas = Canvas.from_preset("A4", margin=10.0)
        new_project = Project(name="New", canvas=new_canvas)
        new_project.add_layer(Layer(name="L1"))
        controller.load_project(new_project)

        # Stack should be cleared
        assert not controller.undo_stack.canUndo()
        assert not controller.undo_stack.canRedo()

    def test_stack_empty_after_new_project(self, controller):
        controller.add_layer()
        assert controller.undo_stack.canUndo()

        new_canvas = Canvas.from_preset("A3", margin=5.0)
        new_project = Project(name="Fresh", canvas=new_canvas)
        new_project.add_layer(Layer(name="Base"))
        controller.new_project(new_project)

        assert not controller.undo_stack.canUndo()
        assert not controller.undo_stack.canRedo()

    def test_actions_after_load_are_undoable(self, controller):
        # Do some actions, load a project, do more actions — only new ones undoable
        controller.add_layer()
        controller.add_layer()

        new_canvas = Canvas.from_preset("A4", margin=10.0)
        new_project = Project(name="Clean", canvas=new_canvas)
        new_project.add_layer(Layer(name="L1"))
        controller.load_project(new_project)

        # Now do a new action
        added = controller.add_layer()
        assert controller.undo_stack.canUndo()
        controller.undo_stack.undo()
        assert controller.get_layer(added.id) is None

        # Stack should be empty now (no pre-load actions)
        assert not controller.undo_stack.canUndo()

    def test_modified_flag_reset_after_load(self, controller):
        controller.add_layer()
        assert controller.modified

        new_canvas = Canvas.from_preset("A4", margin=10.0)
        new_project = Project(name="Loaded", canvas=new_canvas)
        new_project.add_layer(Layer(name="L1"))
        controller.load_project(new_project)

        assert not controller.modified

    def test_modified_flag_reset_after_new_project(self, controller):
        controller.add_layer()
        assert controller.modified

        new_canvas = Canvas.from_preset("A4", margin=10.0)
        new_project = Project(name="Fresh", canvas=new_canvas)
        new_project.add_layer(Layer(name="L1"))
        controller.new_project(new_project)

        assert not controller.modified


# ---------------------------------------------------------------------------
# 12. Redo after undo restores exact state
# ---------------------------------------------------------------------------

class TestRedoAfterUndoRestoresExactState:
    def test_layer_count_exact_after_undo_redo(self, controller):
        initial = len(controller.current_project.layers)

        l1 = controller.add_layer()
        l2 = controller.add_layer()
        assert len(controller.current_project.layers) == initial + 2

        controller.undo_stack.undo()
        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial

        controller.undo_stack.redo()
        controller.undo_stack.redo()
        assert len(controller.current_project.layers) == initial + 2

    def test_path_data_exact_after_undo_redo(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = [[(1.1, 2.2), (3.3, 4.4)], [(5.5, 6.6), (7.7, 8.8)]]

        controller.set_layer_paths(layer_id, paths)
        controller.undo_stack.undo()
        controller.undo_stack.redo()

        restored = controller.get_layer(layer_id).paths
        assert restored == paths

    def test_all_properties_exact_after_undo_redo(self, controller):
        layer_id = controller.current_project.layers[0].id

        controller.set_layer_name(layer_id, "Precise Name")
        controller.set_layer_color(layer_id, "#ABCDEF")
        controller.set_layer_visible(layer_id, False)
        controller.set_layer_locked(layer_id, True)
        controller.set_layer_opacity(layer_id, 0.42)

        # Undo all 5 changes
        for _ in range(5):
            controller.undo_stack.undo()

        # Redo all 5 changes
        for _ in range(5):
            controller.undo_stack.redo()

        layer = controller.get_layer(layer_id)
        assert layer.name == "Precise Name"
        assert layer.color == "#ABCDEF"
        assert layer.visible is False
        assert layer.locked is True
        assert abs(layer.opacity - 0.42) < 1e-6

    def test_canvas_exact_after_undo_redo(self, controller):
        new_canvas = Canvas(
            width_mm=297.0, height_mm=420.0, margin_mm=12.5, paper_preset="A3"
        )
        controller.set_canvas(new_canvas)
        controller.undo_stack.undo()
        controller.undo_stack.redo()

        c = controller.current_project.canvas
        assert abs(c.width_mm - 297.0) < 1e-6
        assert abs(c.height_mm - 420.0) < 1e-6
        assert abs(c.margin_mm - 12.5) < 1e-6
        assert c.paper_preset == "A3"

    def test_layer_order_exact_after_undo_redo(self, controller):
        layer_b = controller.add_layer()
        layer_c = controller.add_layer()
        layer_b_id = layer_b.id
        layer_c_id = layer_c.id

        layer1_id = controller.current_project.layers[0].id

        # Reorder: move C to front
        controller.reorder_layer(layer_c_id, 0)
        ids_after_reorder = [l.id for l in controller.current_project.layers]

        controller.undo_stack.undo()
        controller.undo_stack.redo()

        ids_after_redo = [l.id for l in controller.current_project.layers]
        assert ids_after_redo == ids_after_reorder


# ---------------------------------------------------------------------------
# 13. Stack state (canUndo / canRedo) is correct throughout
# ---------------------------------------------------------------------------

class TestStackState:
    def test_can_undo_after_action(self, controller):
        assert not controller.undo_stack.canUndo()
        controller.add_layer()
        assert controller.undo_stack.canUndo()

    def test_cannot_redo_before_undo(self, controller):
        controller.add_layer()
        assert not controller.undo_stack.canRedo()

    def test_can_redo_after_undo(self, controller):
        controller.add_layer()
        controller.undo_stack.undo()
        assert controller.undo_stack.canRedo()

    def test_cannot_redo_after_new_action(self, controller):
        controller.add_layer()
        controller.undo_stack.undo()
        assert controller.undo_stack.canRedo()

        # Performing a new action clears redo history
        controller.add_layer()
        assert not controller.undo_stack.canRedo()

    def test_stack_depth_matches_actions(self, controller):
        # n pushes → stack depth n
        controller.add_layer()
        controller.add_layer()
        controller.add_layer()
        assert controller.undo_stack.count() == 3

    def test_undo_decrements_index(self, controller):
        controller.add_layer()
        controller.add_layer()
        idx_before = controller.undo_stack.index()
        controller.undo_stack.undo()
        assert controller.undo_stack.index() == idx_before - 1

    def test_redo_increments_index(self, controller):
        controller.add_layer()
        controller.undo_stack.undo()
        idx_before = controller.undo_stack.index()
        controller.undo_stack.redo()
        assert controller.undo_stack.index() == idx_before + 1


# ---------------------------------------------------------------------------
# 14. Modified flag changes correctly around undo/redo
# ---------------------------------------------------------------------------

class TestModifiedFlagWithUndoRedo:
    def test_modified_after_action(self, controller):
        assert not controller.modified
        controller.add_layer()
        assert controller.modified

    def test_modified_remains_after_undo(self, controller):
        """After undoing back to original state, modified may still be True."""
        controller.add_layer()
        controller.undo_stack.undo()
        # The modified flag is driven by QUndoStack.cleanChanged; after undo it
        # goes clean only if the stack is back at the save-point (setClean pos).
        # Since we never called mark_saved(), the "clean" position is index 0.
        # After undoing back to index 0 the stack considers itself clean → not modified.
        assert not controller.modified

    def test_modified_after_save_then_edit(self, controller):
        controller.add_layer()
        controller.mark_saved()
        assert not controller.modified

        controller.add_layer()
        assert controller.modified

    def test_not_modified_after_undo_to_save_point(self, controller):
        controller.add_layer()
        controller.mark_saved()
        assert not controller.modified

        controller.add_layer()
        assert controller.modified

        controller.undo_stack.undo()
        # Back to the saved state
        assert not controller.modified

    def test_modified_after_redo_past_save_point(self, controller):
        controller.add_layer()
        controller.mark_saved()
        assert not controller.modified

        controller.add_layer()
        controller.undo_stack.undo()
        assert not controller.modified

        controller.undo_stack.redo()
        assert controller.modified


# ---------------------------------------------------------------------------
# 15. Complex scenario: interleaved undo/redo across multiple action types
# ---------------------------------------------------------------------------

class TestComplexUndoRedoScenarios:
    def test_generate_then_clear_then_undo_undo(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(3)

        controller.set_layer_paths(layer_id, paths, description="Generate")
        controller.set_layer_paths(layer_id, [], description="Clear")

        # Undo clear → should restore generated paths
        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 3

        # Undo generate → should restore empty
        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 0

    def test_add_generate_remove_undo_chain(self, controller):
        initial_count = len(controller.current_project.layers)

        new_layer = controller.add_layer()
        layer_id = new_layer.id

        paths = _make_paths(2)
        controller.set_layer_paths(layer_id, paths)

        controller.remove_layer(layer_id)
        assert controller.get_layer(layer_id) is None

        # Undo remove
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id) is not None
        assert len(controller.get_layer(layer_id).paths) == 2

        # Undo set_layer_paths
        controller.undo_stack.undo()
        assert len(controller.get_layer(layer_id).paths) == 0

        # Undo add
        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count

    def test_redo_after_partial_undo(self, controller):
        layer_id = controller.current_project.layers[0].id

        controller.set_layer_name(layer_id, "Step 1")
        controller.set_layer_name(layer_id, "Step 2")
        controller.set_layer_name(layer_id, "Step 3")

        # Undo 2 steps
        controller.undo_stack.undo()
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).name == "Step 1"

        # Redo 1 step
        controller.undo_stack.redo()
        assert controller.get_layer(layer_id).name == "Step 2"

        # Now make a new action — this invalidates the remaining redo history
        controller.set_layer_name(layer_id, "Branch")
        assert not controller.undo_stack.canRedo()
        assert controller.get_layer(layer_id).name == "Branch"

    def test_duplicate_then_edit_duplicate_undo(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = _make_paths(2)
        controller.set_layer_paths(layer_id, paths)

        dup = controller.duplicate_layer(layer_id)
        dup_id = dup.id

        controller.set_layer_name(dup_id, "Duplicate Renamed")

        # Undo the rename
        controller.undo_stack.undo()
        assert controller.get_layer(dup_id).name != "Duplicate Renamed"

        # Undo the duplicate
        controller.undo_stack.undo()
        assert controller.get_layer(dup_id) is None

        # Original layer still intact
        assert controller.get_layer(layer_id) is not None
        assert len(controller.get_layer(layer_id).paths) == 2

    def test_canvas_change_interleaved_with_layer_ops(self, controller):
        new_canvas = Canvas.from_preset("A3", margin=5.0)
        controller.set_canvas(new_canvas)

        layer_b = controller.add_layer()
        layer_b_id = layer_b.id

        controller.set_canvas(Canvas.from_preset("Letter", margin=10.0))

        # Undo canvas change → back to A3
        controller.undo_stack.undo()
        assert abs(controller.current_project.canvas.width_mm - new_canvas.width_mm) < 1e-6

        # Undo add layer → layer_b gone
        controller.undo_stack.undo()
        assert controller.get_layer(layer_b_id) is None

        # Undo first canvas change → back to original A4
        controller.undo_stack.undo()
        assert controller.current_project.canvas.paper_preset == "A4"

    def test_merge_then_generate_on_merged_undo(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        layer_b = controller.add_layer()
        layer_b_id = layer_b.id

        merged = controller.merge_layers([layer_a_id, layer_b_id])
        merged_id = merged.id

        paths = _make_paths(4)
        controller.set_layer_paths(merged_id, paths)

        # Undo generate → merged layer now empty
        controller.undo_stack.undo()
        assert len(controller.get_layer(merged_id).paths) == 0

        # Undo merge → originals restored, merged gone
        controller.undo_stack.undo()
        assert controller.get_layer(merged_id) is None
        assert controller.get_layer(layer_a_id) is not None
        assert controller.get_layer(layer_b_id) is not None
