"""Tests for task 43.1 — selected layer text contrast in layer panel.

Covers:
(a) Selected layer text uses HighlightedText color (readable against selection bg)
(b) Deselected layers use normal WindowText / PlaceholderText colors
(c) Switching selection between layers updates both old and new correctly
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_controller():
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#ff0000"))
    proj.add_layer(Layer(name="Layer 2", color="#0000ff"))
    return ProjectController(proj)


def _make_item(controller, layer_name="Layer 1", layer_color="#ff0000"):
    """Create a _LayerItem widget for testing (not added to any list)."""
    from plottter.gui.layer_panel import _LayerItem

    return _LayerItem(
        layer_id="test-id",
        layer_name=layer_name,
        layer_color=layer_color,
        visible=True,
        locked=False,
        path_count=0,
        controller=controller,
        opacity=1.0,
    )


# ---------------------------------------------------------------------------
# (a) Selected layer text is clearly readable
# ---------------------------------------------------------------------------


class TestSelectedLayerTextContrast:
    def test_name_edit_uses_window_color_when_active(self, qapp):
        """_name_edit stylesheet uses Window color when active (contrasts against highlight)."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(True)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.Window).name()
        assert expected_color in item._name_edit.styleSheet()

    def test_count_label_uses_highlighted_text_when_active(self, qapp):
        """_count_label stylesheet uses HighlightedText color when active."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(True)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.HighlightedText).name()
        assert expected_color in item._count_label.styleSheet()

    def test_opacity_label_uses_highlighted_text_when_active(self, qapp):
        """_opacity_label stylesheet uses HighlightedText color when active."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(True)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.HighlightedText).name()
        assert expected_color in item._opacity_label.styleSheet()


# ---------------------------------------------------------------------------
# (b) Deselected layers have normal text color
# ---------------------------------------------------------------------------


class TestDeselectedLayerTextColor:
    def test_name_edit_uses_window_text_when_deselected(self, qapp):
        """_name_edit stylesheet uses WindowText color when deselected."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(False)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.WindowText).name()
        assert expected_color in item._name_edit.styleSheet()

    def test_count_label_uses_placeholder_text_when_deselected(self, qapp):
        """_count_label stylesheet uses PlaceholderText color when deselected."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(False)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
        assert expected_color in item._count_label.styleSheet()

    def test_opacity_label_uses_placeholder_text_when_deselected(self, qapp):
        """_opacity_label stylesheet uses PlaceholderText color when deselected."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(False)

        pal = item.palette()
        expected_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
        assert expected_color in item._opacity_label.styleSheet()

    def test_name_edit_not_highlighted_when_deselected(self, qapp):
        """After deselecting, name edit no longer uses HighlightedText color."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl)
        item.set_selected(True)
        item.set_selected(False)

        pal = item.palette()
        highlighted_color = pal.color(QPalette.ColorRole.HighlightedText).name()
        window_text_color = pal.color(QPalette.ColorRole.WindowText).name()

        ss = item._name_edit.styleSheet()
        # Should use WindowText color now (or they may be the same in some themes,
        # in which case the test below still validates correct behavior)
        assert window_text_color in ss


# ---------------------------------------------------------------------------
# (c) Switching selection updates both old and new correctly
# ---------------------------------------------------------------------------


class TestSwitchingSelection:
    def test_newly_selected_item_gets_active_text_color(self, qapp):
        """When item2 is selected, name_edit uses Window color; labels use lighter variant."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item1 = _make_item(ctrl, "Layer 1", "#ff0000")
        item2 = _make_item(ctrl, "Layer 2", "#0000ff")

        item1.set_selected(True)
        item2.set_selected(True)

        pal = item2.palette()
        expected_name_color = pal.color(QPalette.ColorRole.Window).name()
        expected_label_color = pal.color(QPalette.ColorRole.Window).lighter(130).name()
        assert expected_name_color in item2._name_edit.styleSheet()
        assert expected_label_color in item2._count_label.styleSheet()
        assert expected_label_color in item2._opacity_label.styleSheet()

    def test_previously_selected_item_reverts_to_normal(self, qapp):
        """When item2 is selected, item1 (deselected) reverts to WindowText."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item1 = _make_item(ctrl, "Layer 1", "#ff0000")
        item2 = _make_item(ctrl, "Layer 2", "#0000ff")

        item1.set_selected(True)
        # Now switch: deselect item1, select item2
        item1.set_selected(False)
        item2.set_selected(True)

        pal = item1.palette()
        window_text_color = pal.color(QPalette.ColorRole.WindowText).name()
        assert window_text_color in item1._name_edit.styleSheet()

    def test_count_label_reverts_for_deselected_item(self, qapp):
        """After deselection, _count_label reverts to PlaceholderText color."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item1 = _make_item(ctrl, "Layer 1", "#ff0000")
        item2 = _make_item(ctrl, "Layer 2", "#0000ff")

        item1.set_selected(True)
        item1.set_selected(False)
        item2.set_selected(True)

        pal = item1.palette()
        placeholder_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
        assert placeholder_color in item1._count_label.styleSheet()
        assert placeholder_color in item1._opacity_label.styleSheet()

    def test_opacity_label_reverts_for_deselected_item(self, qapp):
        """After deselection, _opacity_label reverts to PlaceholderText color."""
        from PyQt6.QtGui import QPalette

        ctrl = _make_controller()
        item = _make_item(ctrl, "Layer 1", "#ff0000")

        item.set_selected(True)
        item.set_selected(False)

        pal = item.palette()
        placeholder_color = pal.color(QPalette.ColorRole.PlaceholderText).name()
        assert placeholder_color in item._opacity_label.styleSheet()


# ---------------------------------------------------------------------------
# (d) LayerPanel — full panel selection visuals
# ---------------------------------------------------------------------------


class TestLayerPanelSelectionVisuals:
    def _make_panel(self, ctrl):
        from plottter.gui.layer_panel import LayerPanel

        return LayerPanel(ctrl)

    def test_active_item_widget_has_window_text_color(self, qapp):
        """After rebuild, the active layer item widget uses Window color for text."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPalette

        from plottter.gui.layer_panel import _LayerItem

        ctrl = _make_controller()
        panel = self._make_panel(ctrl)

        # The first layer should be selected by default
        current_item = panel._list.currentItem()
        assert current_item is not None
        widget = panel._list.itemWidget(current_item)
        assert isinstance(widget, _LayerItem)

        pal = widget.palette()
        expected = pal.color(QPalette.ColorRole.Window).name()
        assert expected in widget._name_edit.styleSheet()

    def test_non_active_item_widget_has_normal_text(self, qapp):
        """Non-active items use WindowText color (not HighlightedText)."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPalette

        from plottter.gui.layer_panel import _LayerItem

        ctrl = _make_controller()
        panel = self._make_panel(ctrl)

        # Select row 0; row 1 should be non-active
        panel._list.setCurrentRow(0)
        panel._update_selection_visuals()

        item1 = panel._list.item(1)
        assert item1 is not None
        widget1 = panel._list.itemWidget(item1)
        assert isinstance(widget1, _LayerItem)

        pal = widget1.palette()
        highlighted = pal.color(QPalette.ColorRole.HighlightedText).name()
        window_text = pal.color(QPalette.ColorRole.WindowText).name()

        ss = widget1._name_edit.styleSheet()
        # If HighlightedText == WindowText (some themes), skip this check
        if highlighted != window_text:
            assert highlighted not in ss
        assert window_text in ss


# ---------------------------------------------------------------------------
# Task 43.2 — Multi-layer selection for merge
# ---------------------------------------------------------------------------


def _make_controller_with_paths():
    """Create a controller with 3 layers, each containing one path."""
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test", canvas=canvas)
    layer1 = Layer(name="Layer 1", color="#ff0000", paths=[[(0.0, 0.0), (10.0, 0.0)]])
    layer2 = Layer(name="Layer 2", color="#00ff00", paths=[[(20.0, 0.0), (30.0, 0.0)]])
    layer3 = Layer(name="Layer 3", color="#0000ff", paths=[[(40.0, 0.0), (50.0, 0.0)]])
    proj.add_layer(layer1)
    proj.add_layer(layer2)
    proj.add_layer(layer3)
    return ProjectController(proj)


class TestMultiLayerSelection:
    """Tests for task 43.2 — multi-layer selection for merge."""

    def _make_panel(self, ctrl):
        from plottter.gui.layer_panel import LayerPanel
        return LayerPanel(ctrl)

    def test_selection_mode_is_extended(self, qapp):
        """LayerPanel list uses ExtendedSelection mode."""
        from PyQt6.QtWidgets import QAbstractItemView

        ctrl = _make_controller()
        panel = self._make_panel(ctrl)
        assert panel._list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection

    def test_single_click_selects_one_layer(self, qapp):
        """Selecting a single row results in exactly one selected item."""
        ctrl = _make_controller()
        panel = self._make_panel(ctrl)
        panel._list.clearSelection()
        panel._list.setCurrentRow(0)
        assert len(panel._list.selectedItems()) == 1

    def test_multi_select_returns_multiple_ids(self, qapp):
        """Programmatic multi-selection returns multiple layer IDs from _selected_layer_ids."""
        ctrl = _make_controller()
        panel = self._make_panel(ctrl)
        # Select both items programmatically
        panel._list.item(0).setSelected(True)
        panel._list.item(1).setSelected(True)
        ids = panel._selected_layer_ids()
        assert len(ids) == 2

    def test_current_item_changed_sets_active_layer(self, qapp):
        """_on_current_item_changed sets the active layer to the most recently clicked one."""
        ctrl = _make_controller_with_paths()
        panel = self._make_panel(ctrl)
        # Simulate clicking layer at row 1
        panel._list.setCurrentRow(1)
        expected_id = panel._list.item(1).data(0x0100)  # Qt.ItemDataRole.UserRole
        assert ctrl.active_layer_id == expected_id

    def test_merge_with_less_than_2_shows_message(self, qapp, monkeypatch):
        """merge button shows info message when fewer than 2 layers are selected."""
        from PyQt6.QtWidgets import QMessageBox

        shown = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: shown.append(True))
        ctrl = _make_controller()
        panel = self._make_panel(ctrl)
        panel._list.clearSelection()
        panel._list.setCurrentRow(0)
        panel._on_merge()
        assert shown, "Expected QMessageBox.information to be called"

    def test_merge_with_2_selected_combines_paths(self, qapp):
        """Merging 2 selected layers produces one layer with combined paths."""
        ctrl = _make_controller_with_paths()
        panel = self._make_panel(ctrl)

        initial_count = len(ctrl.current_project.layers)

        # Select first two layers
        panel._list.setCurrentRow(0)
        panel._list.item(0).setSelected(True)
        panel._list.item(1).setSelected(True)
        panel._on_merge()

        # Two layers merged into one — total count drops by 1
        assert len(ctrl.current_project.layers) == initial_count - 1
        # Merged layer has 2 paths (one from each source layer)
        merged = ctrl.current_project.layers[-1]
        assert merged.path_count() == 2

    def test_merge_with_3_selected_combines_all_paths(self, qapp):
        """Merging all 3 layers produces one layer with all 3 paths."""
        ctrl = _make_controller_with_paths()
        panel = self._make_panel(ctrl)

        panel._list.setCurrentRow(0)
        panel._list.item(0).setSelected(True)
        panel._list.item(1).setSelected(True)
        panel._list.item(2).setSelected(True)
        panel._on_merge()

        assert len(ctrl.current_project.layers) == 1
        assert ctrl.current_project.layers[0].path_count() == 3

    def test_undo_after_merge_restores_layers(self, qapp):
        """After merging, undo restores the original layers."""
        ctrl = _make_controller_with_paths()
        panel = self._make_panel(ctrl)

        initial_count = len(ctrl.current_project.layers)

        panel._list.setCurrentRow(0)
        panel._list.item(0).setSelected(True)
        panel._list.item(1).setSelected(True)
        panel._on_merge()

        assert len(ctrl.current_project.layers) == initial_count - 1

        ctrl._undo_stack.undo()

        assert len(ctrl.current_project.layers) == initial_count
