"""GUI walkthrough tests — automated end-to-end exercise of all layer management
and project features described in task 15.3.

Covers:
- Create project / set paper size
- Add / rename / reorder / merge / duplicate / delete layers
- Assign pen colors
- Toggle visibility / opacity / lock
- Status bar label content after each operation
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project
from plottter.models.canvas import PAPER_PRESETS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(qapp):
    """Ensure a QApplication exists (pytest-qt provides qapp automatically)."""
    return qapp


@pytest.fixture
def project():
    """A fresh A4 project with one layer."""
    canvas = Canvas.from_preset("A4", margin=10.0)
    p = Project(name="Test Project", canvas=canvas)
    p.add_layer(Layer(name="Layer 1", color="#000000"))
    return p


@pytest.fixture
def controller(project, qapp):
    """A ProjectController wrapping the test project."""
    from plottter.gui.project_controller import ProjectController
    return ProjectController(project)


@pytest.fixture
def main_window(controller, qtbot):
    """A fully constructed MainWindow (not shown)."""
    from plottter.gui.main_window import MainWindow
    win = MainWindow(controller)
    # Prevent the "unsaved changes?" QMessageBox from blocking headless teardown.
    win._prompt_save_if_modified = lambda: True
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# 1. Project creation and paper size
# ---------------------------------------------------------------------------

class TestProjectCreation:
    def test_default_project_has_a4_canvas(self, project):
        assert project.canvas.width_mm == pytest.approx(210.0)
        assert project.canvas.height_mm == pytest.approx(297.0)
        assert project.canvas.paper_preset == "A4"

    def test_new_project_dialog_returns_a4_canvas(self, qapp):
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dlg = NewProjectDialog()
        canvas = dlg.get_canvas()
        assert canvas.width_mm == pytest.approx(210.0)
        assert canvas.height_mm == pytest.approx(297.0)
        assert canvas.paper_preset == "A4"

    def test_new_project_dialog_preset_a3(self, qapp):
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dlg = NewProjectDialog()
        dlg._preset_combo.setCurrentText("A3")
        canvas = dlg.get_canvas()
        assert canvas.width_mm == pytest.approx(297.0)
        assert canvas.height_mm == pytest.approx(420.0)
        assert canvas.paper_preset == "A3"

    def test_new_project_dialog_letter_preset(self, qapp):
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dlg = NewProjectDialog()
        dlg._preset_combo.setCurrentText("Letter")
        canvas = dlg.get_canvas()
        assert canvas.width_mm == pytest.approx(215.9)
        assert canvas.height_mm == pytest.approx(279.4)
        assert canvas.paper_preset == "Letter"

    def test_new_project_dialog_custom_size(self, qapp):
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dlg = NewProjectDialog()
        dlg._preset_combo.setCurrentText("Custom")
        dlg._width_spin.setValue(300.0)
        dlg._height_spin.setValue(400.0)
        dlg._margin_spin.setValue(15.0)
        canvas = dlg.get_canvas()
        assert canvas.width_mm == pytest.approx(300.0)
        assert canvas.height_mm == pytest.approx(400.0)
        assert canvas.margin_mm == pytest.approx(15.0)
        assert canvas.paper_preset == "Custom"

    def test_new_project_dialog_unit_toggle_to_inches(self, qapp):
        from plottter.gui.dialogs.new_project import NewProjectDialog
        dlg = NewProjectDialog()
        # Switch to inches — values should convert
        dlg._in_radio.setChecked(True)
        canvas = dlg.get_canvas()
        # A4 in inches: 210/25.4 ≈ 8.268, 297/25.4 ≈ 11.693 but stored as mm
        assert canvas.width_mm == pytest.approx(210.0, abs=0.5)
        assert canvas.height_mm == pytest.approx(297.0, abs=0.5)

    def test_controller_new_project_emits_project_loaded(self, controller, qtbot):
        new_canvas = Canvas.from_preset("A3", margin=10.0)
        new_project = Project(name="New", canvas=new_canvas)
        new_project.add_layer(Layer(name="Layer 1"))
        with qtbot.waitSignal(controller.project_loaded, timeout=1000):
            controller.new_project(new_project)

    def test_controller_set_canvas_emits_canvas_changed(self, controller, qtbot):
        new_canvas = Canvas.from_preset("A3", margin=10.0)
        with qtbot.waitSignal(controller.canvas_changed, timeout=1000):
            controller.set_canvas(new_canvas)

    def test_set_canvas_updates_project(self, controller):
        new_canvas = Canvas.from_preset("A3", margin=10.0)
        controller.set_canvas(new_canvas)
        assert controller.current_project.canvas.paper_preset == "A3"
        assert controller.current_project.canvas.width_mm == pytest.approx(297.0)


# ---------------------------------------------------------------------------
# 2. Layer management
# ---------------------------------------------------------------------------

class TestLayerManagement:
    def test_add_layer_increases_count(self, controller):
        initial = len(controller.current_project.layers)
        controller.add_layer()
        assert len(controller.current_project.layers) == initial + 1

    def test_add_layer_emits_signal(self, controller, qtbot):
        with qtbot.waitSignal(controller.layer_added, timeout=1000):
            controller.add_layer()

    def test_add_layer_default_name(self, controller):
        initial_count = len(controller.current_project.layers)
        layer = controller.add_layer()
        assert layer.name == f"Layer {initial_count + 1}"

    def test_add_named_layer(self, controller):
        layer = controller.add_layer(Layer(name="Custom Name", color="#ff0000"))
        assert layer.name == "Custom Name"
        assert layer.color == "#ff0000"

    def test_rename_layer(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_name(layer_id, "Renamed Layer")
        layer = controller.get_layer(layer_id)
        assert layer is not None
        assert layer.name == "Renamed Layer"

    def test_rename_layer_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_name(layer_id, "New Name")

    def test_rename_layer_is_undoable(self, controller):
        layer_id = controller.current_project.layers[0].id
        original_name = controller.get_layer(layer_id).name
        controller.set_layer_name(layer_id, "Renamed")
        assert controller.get_layer(layer_id).name == "Renamed"
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).name == original_name

    def test_delete_layer(self, controller):
        controller.add_layer()
        layer_id = controller.current_project.layers[0].id
        controller.remove_layer(layer_id)
        assert controller.get_layer(layer_id) is None

    def test_delete_layer_emits_signal(self, controller, qtbot):
        controller.add_layer()
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_removed, timeout=1000):
            controller.remove_layer(layer_id)

    def test_delete_layer_is_undoable(self, controller):
        controller.add_layer()
        layer_id = controller.current_project.layers[0].id
        controller.remove_layer(layer_id)
        assert controller.get_layer(layer_id) is None
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id) is not None

    def test_duplicate_layer(self, controller):
        layer_id = controller.current_project.layers[0].id
        original_count = len(controller.current_project.layers)
        new_layer = controller.duplicate_layer(layer_id)
        assert len(controller.current_project.layers) == original_count + 1
        assert new_layer.id != layer_id

    def test_duplicate_layer_copies_color(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_color(layer_id, "#ff0000")
        new_layer = controller.duplicate_layer(layer_id)
        assert new_layer.color == "#ff0000"

    def test_duplicate_layer_copies_paths(self, controller):
        layer_id = controller.current_project.layers[0].id
        paths = [[(0.0, 0.0), (10.0, 10.0)]]
        controller.set_layer_paths(layer_id, paths)
        new_layer = controller.duplicate_layer(layer_id)
        assert new_layer.path_count() == 1

    def test_reorder_layer_move_up(self, controller):
        controller.add_layer(Layer(name="Layer B"))
        layers = controller.current_project.layers
        assert layers[0].name == "Layer 1"
        assert layers[1].name == "Layer B"
        layer_b_id = layers[1].id
        controller.reorder_layer(layer_b_id, 0)
        layers = controller.current_project.layers
        assert layers[0].name == "Layer B"
        assert layers[1].name == "Layer 1"

    def test_reorder_layer_emits_signal(self, controller, qtbot):
        controller.add_layer(Layer(name="Layer B"))
        layer_b_id = controller.current_project.layers[1].id
        with qtbot.waitSignal(controller.layers_reordered, timeout=1000):
            controller.reorder_layer(layer_b_id, 0)

    def test_reorder_layer_is_undoable(self, controller):
        controller.add_layer(Layer(name="Layer B"))
        layers = controller.current_project.layers
        layer_b_id = layers[1].id
        controller.reorder_layer(layer_b_id, 0)
        assert controller.current_project.layers[0].name == "Layer B"
        controller.undo_stack.undo()
        assert controller.current_project.layers[0].name == "Layer 1"

    def test_merge_two_layers(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        paths_a = [[(0.0, 0.0), (5.0, 5.0)]]
        controller.set_layer_paths(layer_a_id, paths_a)

        controller.add_layer(Layer(name="Layer B"))
        layer_b_id = controller.current_project.layers[1].id
        paths_b = [[(10.0, 10.0), (20.0, 20.0)]]
        controller.set_layer_paths(layer_b_id, paths_b)

        merged = controller.merge_layers([layer_a_id, layer_b_id])
        # Merged layer has combined paths
        assert merged.path_count() == 2
        # Original layers removed
        remaining_ids = [l.id for l in controller.current_project.layers]
        assert layer_a_id not in remaining_ids
        assert layer_b_id not in remaining_ids

    def test_merge_three_layers(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        controller.add_layer(Layer(name="Layer B"))
        controller.add_layer(Layer(name="Layer C"))
        layers = controller.current_project.layers
        ids = [l.id for l in layers]
        merged = controller.merge_layers(ids[:3])
        assert len(controller.current_project.layers) == 1
        assert merged.path_count() == 0  # all layers empty

    def test_merge_is_undoable(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        controller.add_layer(Layer(name="Layer B"))
        layer_b_id = controller.current_project.layers[1].id
        initial_count = len(controller.current_project.layers)

        controller.merge_layers([layer_a_id, layer_b_id])
        assert len(controller.current_project.layers) == initial_count - 1

        controller.undo_stack.undo()
        assert len(controller.current_project.layers) == initial_count


# ---------------------------------------------------------------------------
# 3. Pen color assignment
# ---------------------------------------------------------------------------

class TestPenColorAssignment:
    def test_set_layer_color(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_color(layer_id, "#ff0000")
        layer = controller.get_layer(layer_id)
        assert layer.color == "#ff0000"

    def test_set_layer_color_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_color(layer_id, "#00ff00")

    def test_set_layer_color_is_undoable(self, controller):
        layer_id = controller.current_project.layers[0].id
        original_color = controller.get_layer(layer_id).color
        controller.set_layer_color(layer_id, "#0000ff")
        assert controller.get_layer(layer_id).color == "#0000ff"
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).color == original_color

    def test_multiple_layers_different_colors(self, controller):
        layer_a_id = controller.current_project.layers[0].id
        controller.add_layer(Layer(name="Layer B"))
        layer_b_id = controller.current_project.layers[1].id

        controller.set_layer_color(layer_a_id, "#ff0000")
        controller.set_layer_color(layer_b_id, "#0000ff")

        assert controller.get_layer(layer_a_id).color == "#ff0000"
        assert controller.get_layer(layer_b_id).color == "#0000ff"


# ---------------------------------------------------------------------------
# 4. Visibility, opacity, lock toggles
# ---------------------------------------------------------------------------

class TestLayerToggles:
    def test_toggle_visibility_off(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_visible(layer_id, False)
        assert controller.get_layer(layer_id).visible is False

    def test_toggle_visibility_on(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_visible(layer_id, False)
        controller.set_layer_visible(layer_id, True)
        assert controller.get_layer(layer_id).visible is True

    def test_toggle_visibility_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_visible(layer_id, False)

    def test_toggle_visibility_is_undoable(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_visible(layer_id, False)
        assert controller.get_layer(layer_id).visible is False
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).visible is True

    def test_toggle_lock_on(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_locked(layer_id, True)
        assert controller.get_layer(layer_id).locked is True

    def test_toggle_lock_off(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_locked(layer_id, True)
        controller.set_layer_locked(layer_id, False)
        assert controller.get_layer(layer_id).locked is False

    def test_toggle_lock_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_locked(layer_id, True)

    def test_toggle_lock_is_undoable(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_locked(layer_id, True)
        assert controller.get_layer(layer_id).locked is True
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).locked is False

    def test_set_opacity_full(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 1.0)
        assert controller.get_layer(layer_id).opacity == pytest.approx(1.0)

    def test_set_opacity_half(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 0.5)
        assert controller.get_layer(layer_id).opacity == pytest.approx(0.5)

    def test_set_opacity_zero(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 0.0)
        assert controller.get_layer(layer_id).opacity == pytest.approx(0.0)

    def test_set_opacity_clamps_above_one(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 1.5)
        assert controller.get_layer(layer_id).opacity == pytest.approx(1.0)

    def test_set_opacity_clamps_below_zero(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, -0.5)
        assert controller.get_layer(layer_id).opacity == pytest.approx(0.0)

    def test_set_opacity_emits_signal(self, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_opacity(layer_id, 0.75)

    def test_set_opacity_is_undoable(self, controller):
        layer_id = controller.current_project.layers[0].id
        controller.set_layer_opacity(layer_id, 0.3)
        assert controller.get_layer(layer_id).opacity == pytest.approx(0.3)
        controller.undo_stack.undo()
        assert controller.get_layer(layer_id).opacity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Status bar updates
# ---------------------------------------------------------------------------

class TestStatusBarUpdates:
    def test_status_canvas_label_shows_dimensions(self, main_window, controller):
        text = main_window._status_canvas.text()
        assert "210" in text
        assert "297" in text
        assert "A4" in text

    def test_status_canvas_updates_on_canvas_change(self, main_window, controller, qtbot):
        new_canvas = Canvas.from_preset("A3", margin=10.0)
        with qtbot.waitSignal(controller.canvas_changed, timeout=1000):
            controller.set_canvas(new_canvas)
        text = main_window._status_canvas.text()
        assert "297" in text
        assert "420" in text
        assert "A3" in text

    def test_status_paths_shows_initial_zero(self, main_window):
        text = main_window._status_paths.text()
        # Should show "Paths: 0" when no paths exist
        assert "Paths:" in text
        assert "0" in text

    def test_status_updates_when_layer_added(self, main_window, controller, qtbot):
        # Adding a layer should trigger _update_status_bar
        with qtbot.waitSignal(controller.layer_added, timeout=1000):
            controller.add_layer()
        # Status bar canvas label still shows canvas info
        text = main_window._status_canvas.text()
        assert "210" in text

    def test_status_updates_on_paths_changed(self, main_window, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        paths = [
            [(0.0, 0.0), (100.0, 0.0)],
            [(0.0, 10.0), (100.0, 10.0)],
        ]
        with qtbot.waitSignal(controller.paths_changed, timeout=1000):
            controller.set_layer_paths(layer_id, paths)
        text = main_window._status_paths.text()
        assert "Paths:" in text
        assert "2" in text

    def test_status_paths_excludes_hidden_layers(self, main_window, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        paths = [[(0.0, 0.0), (10.0, 0.0)]]
        controller.set_layer_paths(layer_id, paths)

        # Hide the layer — path count should drop to 0
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_visible(layer_id, False)
        text = main_window._status_paths.text()
        assert "Paths:" in text
        assert "0" in text

    def test_status_paths_returns_when_layer_reshown(self, main_window, controller, qtbot):
        layer_id = controller.current_project.layers[0].id
        paths = [[(0.0, 0.0), (10.0, 0.0)]]
        controller.set_layer_paths(layer_id, paths)
        controller.set_layer_visible(layer_id, False)

        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            controller.set_layer_visible(layer_id, True)
        text = main_window._status_paths.text()
        assert "Paths:" in text
        assert "1" in text

    def test_status_canvas_shows_custom_dimensions(self, main_window, controller, qtbot):
        new_canvas = Canvas(width_mm=300.0, height_mm=400.0, margin_mm=5.0, paper_preset="Custom")
        with qtbot.waitSignal(controller.canvas_changed, timeout=1000):
            controller.set_canvas(new_canvas)
        text = main_window._status_canvas.text()
        assert "300" in text
        assert "400" in text
        assert "Custom" in text

    def test_window_title_shows_modified_star(self, main_window, controller, qtbot):
        with qtbot.waitSignal(controller.modified_changed, timeout=1000):
            controller.add_layer()
        assert "*" in main_window.windowTitle()

    def test_window_title_shows_project_name(self, main_window):
        assert "Test Project" in main_window.windowTitle()


# ---------------------------------------------------------------------------
# 6. Layer panel widget integration
# ---------------------------------------------------------------------------

class TestLayerPanelIntegration:
    def test_layer_panel_shows_all_layers(self, main_window, controller):
        panel = main_window._layer_panel
        controller.add_layer(Layer(name="Layer B"))
        # Panel rebuilds automatically via signals; check list count
        assert panel._list.count() == 2

    def test_layer_panel_add_button(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        initial_count = len(controller.current_project.layers)
        with qtbot.waitSignal(controller.layer_added, timeout=1000):
            panel._add_btn.click()
        assert len(controller.current_project.layers) == initial_count + 1

    def test_layer_panel_delete_button_removes_second_layer(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        controller.add_layer(Layer(name="Layer B"))
        # Select second layer
        panel._list.setCurrentRow(1)
        with qtbot.waitSignal(controller.layer_removed, timeout=1000):
            panel._del_btn.click()
        assert len(controller.current_project.layers) == 1

    def test_layer_panel_delete_button_blocked_on_last_layer(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        # Select the only layer
        panel._list.setCurrentRow(0)
        initial_count = len(controller.current_project.layers)
        # Clicking delete on last layer should show a warning and NOT remove it
        # We can't easily intercept QMessageBox without patching, so just confirm count unchanged
        # by calling the internal handler directly
        from unittest.mock import patch
        with patch("plottter.gui.layer_panel.QMessageBox.warning"):
            panel._on_delete()
        assert len(controller.current_project.layers) == initial_count

    def test_layer_panel_duplicate_button(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        panel._list.setCurrentRow(0)
        with qtbot.waitSignal(controller.layer_added, timeout=1000):
            panel._dup_btn.click()
        assert len(controller.current_project.layers) == 2

    def test_layer_panel_move_up_button(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        controller.add_layer(Layer(name="Layer B"))
        panel._list.setCurrentRow(1)  # select Layer B
        with qtbot.waitSignal(controller.layers_reordered, timeout=1000):
            panel._up_btn.click()
        assert controller.current_project.layers[0].name == "Layer B"

    def test_layer_panel_move_down_button(self, main_window, controller, qtbot):
        panel = main_window._layer_panel
        controller.add_layer(Layer(name="Layer B"))
        panel._list.setCurrentRow(0)  # select Layer 1
        with qtbot.waitSignal(controller.layers_reordered, timeout=1000):
            panel._down_btn.click()
        assert controller.current_project.layers[1].name == "Layer 1"

    def test_layer_item_visibility_toggle(self, main_window, controller, qtbot):
        """Clicking the visibility button on a layer item toggles visibility."""
        panel = main_window._layer_panel
        item = panel._list.item(0)
        from plottter.gui.layer_panel import _LayerItem
        widget = panel._list.itemWidget(item)
        assert isinstance(widget, _LayerItem)
        layer_id = widget.layer_id
        assert controller.get_layer(layer_id).visible is True
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            widget._vis_btn.click()
        assert controller.get_layer(layer_id).visible is False

    def test_layer_item_lock_toggle(self, main_window, controller, qtbot):
        """Clicking the lock button on a layer item toggles lock."""
        panel = main_window._layer_panel
        item = panel._list.item(0)
        from plottter.gui.layer_panel import _LayerItem
        widget = panel._list.itemWidget(item)
        assert isinstance(widget, _LayerItem)
        layer_id = widget.layer_id
        assert controller.get_layer(layer_id).locked is False
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            widget._lock_btn.click()
        assert controller.get_layer(layer_id).locked is True

    def test_layer_item_name_edit(self, main_window, controller, qtbot):
        """Double-clicking the name field enters edit mode and renames the layer.

        The display widget is a QLabel; entering edit mode swaps in a
        QLineEdit for inline rename.  We test the behavioural contract —
        edit mode swaps which widget commits the new name to the model —
        rather than Qt show/hide flags, which behave oddly for widgets
        whose parent window has never been shown.
        """
        panel = main_window._layer_panel
        item = panel._list.item(0)
        from plottter.gui.layer_panel import _LayerItem
        widget = panel._list.itemWidget(item)
        assert isinstance(widget, _LayerItem)
        layer_id = widget.layer_id
        original_name = widget._name_label.text()

        widget._enter_edit_mode()
        # In edit mode the QLineEdit holds the same text as the label did.
        assert widget._name_edit.text() == original_name

        widget._name_edit.setText("Edited Name")
        with qtbot.waitSignal(controller.layer_changed, timeout=1000):
            widget._name_edit.editingFinished.emit()
        assert controller.get_layer(layer_id).name == "Edited Name"
        # After commit, the display label reflects the new name.
        assert widget._name_label.text() == "Edited Name"

    def test_layer_item_name_edit_blank_reverts(self, main_window, controller):
        """Editing the name to blank reverts to the original name."""
        panel = main_window._layer_panel
        item = panel._list.item(0)
        from plottter.gui.layer_panel import _LayerItem
        widget = panel._list.itemWidget(item)
        assert isinstance(widget, _LayerItem)
        layer_id = widget.layer_id
        original_name = controller.get_layer(layer_id).name
        widget._enter_edit_mode()
        widget._name_edit.setText("")
        widget._name_edit.editingFinished.emit()
        # Label is restored from the captured original name.
        assert widget._name_label.text() == original_name
        # Model name was not changed (blank input is dropped).
        assert controller.get_layer(layer_id).name == original_name


# ---------------------------------------------------------------------------
# 7. End-to-end workflow scenario
# ---------------------------------------------------------------------------

class TestEndToEndWorkflow:
    """Simulate a complete user session: create project, build layers, modify, verify."""

    def test_full_session(self, qapp, qtbot):
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.main_window import MainWindow

        # 1. Create a new project (A4)
        canvas = Canvas.from_preset("A4", margin=10.0)
        project = Project(name="My Art", canvas=canvas)
        project.add_layer(Layer(name="Background", color="#000000"))
        controller = ProjectController(project)
        win = MainWindow(controller)
        qtbot.addWidget(win)

        # Verify initial state
        assert len(project.layers) == 1
        assert "210" in win._status_canvas.text()

        # 2. Add two more layers
        controller.add_layer(Layer(name="Foreground", color="#ff0000"))
        controller.add_layer(Layer(name="Detail", color="#0000ff"))
        assert len(project.layers) == 3

        # 3. Add paths to each layer
        layer_bg = project.layers[0]
        layer_fg = project.layers[1]
        layer_detail = project.layers[2]
        controller.set_layer_paths(layer_bg.id, [[(0.0, 0.0), (210.0, 297.0)]])
        controller.set_layer_paths(layer_fg.id, [
            [(10.0, 10.0), (50.0, 50.0)],
            [(50.0, 50.0), (100.0, 100.0)],
        ])
        controller.set_layer_paths(layer_detail.id, [
            [(5.0, 5.0), (15.0, 15.0)],
        ])

        # Status should show Paths: 4
        assert "Paths:" in win._status_paths.text()
        assert "4" in win._status_paths.text()

        # 4. Hide the detail layer — path count should drop
        controller.set_layer_visible(layer_detail.id, False)
        assert "3" in win._status_paths.text()

        # 5. Reorder: move Foreground to top
        controller.reorder_layer(layer_fg.id, 0)
        assert project.layers[0].name == "Foreground"

        # 6. Rename Background to "Base"
        controller.set_layer_name(layer_bg.id, "Base")
        assert controller.get_layer(layer_bg.id).name == "Base"

        # 7. Change canvas to A3
        new_canvas = Canvas.from_preset("A3", margin=15.0)
        controller.set_canvas(new_canvas)
        assert "297" in win._status_canvas.text()
        assert "420" in win._status_canvas.text()

        # 8. Duplicate foreground layer
        dup = controller.duplicate_layer(layer_fg.id)
        assert len(project.layers) == 4
        assert dup.path_count() == 2

        # 9. Merge Background + Detail into one layer
        merged = controller.merge_layers([layer_bg.id, layer_detail.id])
        assert len(project.layers) == 3
        assert merged.path_count() == 2  # 1 bg + 1 detail path

        # 10. Lock the merged layer
        controller.set_layer_locked(merged.id, True)
        assert controller.get_layer(merged.id).locked is True

        # 11. Set opacity of foreground
        controller.set_layer_opacity(layer_fg.id, 0.8)
        assert controller.get_layer(layer_fg.id).opacity == pytest.approx(0.8)

        # 12. Undo back to check undo chain is intact (just verify canUndo)
        assert controller.undo_stack.canUndo()

        # Window title shows modified
        assert "*" in win.windowTitle()
        assert "My Art" in win.windowTitle()
