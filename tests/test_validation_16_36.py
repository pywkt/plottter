"""Phase 16.36 validation: Interactive Shape Drawing mode.

Verifies:
1. Drawing a rectangle produces a closed polyline (last point == first point).
2. Hatching fill produces hatch lines clipped to the shape boundary.
3. Appending multiple shapes accumulates paths on the layer.
4. Undo removes only the last drawn shape (not all paths).
5. Stroke toggle controls whether the outline polyline is included in new_paths.
6. ModePanel includes "Shape Drawing" in MODES list.
7. CanvasWidget shape_drawn signal is defined.
8. AppendPathsCommand appends and undos correctly.
9. ProjectController.add_paths_to_layer appends without replacing.
10. Concentric fill produces rings inside the shape boundary.
"""

from __future__ import annotations

import math
import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project(num_layers: int = 1) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    for i in range(num_layers):
        proj.add_layer(Layer(name=f"Layer {i + 1}", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project(num_layers=2))


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget
    w = CanvasWidget(controller)
    w.resize(800, 600)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def settings_panel(controller, canvas_widget, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    sp = SettingsPanel(controller)
    sp.set_canvas(canvas_widget)
    sp.resize(400, 900)
    sp.show()
    qtbot.addWidget(sp)
    return sp


# ---------------------------------------------------------------------------
# ModePanel
# ---------------------------------------------------------------------------


class TestModePanel:
    def test_shape_drawing_in_modes_list(self, qapp):
        from plottter.gui.mode_panel import ModePanel
        assert "Shape Drawing" in ModePanel.MODES

    def test_mode_panel_emits_shape_drawing(self, qapp, qtbot):
        from plottter.gui.mode_panel import ModePanel
        panel = ModePanel()
        qtbot.addWidget(panel)
        received: list[str] = []
        panel.mode_changed.connect(received.append)
        panel.set_mode("Shape Drawing")
        panel._radio_buttons["Shape Drawing"].click()
        assert "Shape Drawing" in received

    def test_current_mode_shape_drawing(self, qapp, qtbot):
        from plottter.gui.mode_panel import ModePanel
        panel = ModePanel()
        qtbot.addWidget(panel)
        panel.set_mode("Shape Drawing")
        assert panel.current_mode() == "Shape Drawing"


# ---------------------------------------------------------------------------
# CanvasWidget shape drawing API
# ---------------------------------------------------------------------------


class TestCanvasWidgetShapeDrawing:
    def test_shape_drawn_signal_exists(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        # signal should exist and be connectable
        received: list = []
        canvas.shape_drawn.connect(received.append)
        # Emit manually to verify
        canvas.shape_drawn.emit([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        assert len(received) == 1
        assert received[0] == [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]

    def test_set_shape_draw_active_enables_mode(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        canvas.set_shape_draw_active(True)
        assert canvas._shape_draw_active is True

    def test_set_shape_draw_active_disables_mask_paint(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        canvas.set_mask_paint_active(True)
        canvas.set_shape_draw_active(True)
        assert canvas._mask_paint_active is False

    def test_set_shape_draw_tool(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget, ShapeDrawTool
        canvas = CanvasWidget(controller)
        canvas.set_shape_draw_tool("ellipse")
        assert canvas._shape_draw_tool == ShapeDrawTool.ELLIPSE

    def test_set_shape_draw_color(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        canvas.set_shape_draw_color("#FF0000")
        assert canvas._shape_draw_color == "#FF0000"

    def test_make_rectangle_polyline_closed(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        pts = canvas._sd_make_rectangle_polyline(0, 0, 10, 20)
        assert len(pts) == 5
        assert pts[0] == pts[-1], "Rectangle polyline must be closed"
        # Check all four corners present
        xs = {p[0] for p in pts}
        ys = {p[1] for p in pts}
        assert 0.0 in xs and 10.0 in xs
        assert 0.0 in ys and 20.0 in ys

    def test_make_ellipse_polyline_closed(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        pts = canvas._sd_make_ellipse_polyline(0, 0, 20, 10)
        assert pts[0] == pts[-1], "Ellipse polyline must be closed"
        assert len(pts) > 10

    def test_set_shape_draw_active_cancels_in_progress(self, qapp, controller):
        from plottter.gui.canvas_widget import CanvasWidget
        canvas = CanvasWidget(controller)
        canvas.set_shape_draw_active(True)
        # Simulate in-progress polygon
        canvas._sd_polygon_vertices = [(0, 0), (10, 0)]
        canvas._sd_line_vertices = [(5, 5)]
        canvas.set_shape_draw_active(False)
        assert canvas._sd_polygon_vertices == []
        assert canvas._sd_line_vertices == []


# ---------------------------------------------------------------------------
# AppendPathsCommand (commands.py)
# ---------------------------------------------------------------------------


class TestAppendPathsCommand:
    def test_append_adds_to_layer(self, controller):
        layer = controller.current_project.layers[0]
        existing = [[(0.0, 0.0), (5.0, 5.0)]]
        layer.paths = existing

        new_paths = [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0), (4.0, 4.0)]]
        controller.add_paths_to_layer(layer.id, new_paths, "Draw Shape")

        assert len(layer.paths) == 3
        assert layer.paths[1] == new_paths[0]
        assert layer.paths[2] == new_paths[1]

    def test_undo_removes_only_appended_paths(self, controller):
        layer = controller.current_project.layers[0]
        existing = [[(0.0, 0.0), (5.0, 5.0)]]
        layer.paths = list(existing)

        new_paths = [[(10.0, 10.0), (20.0, 20.0)]]
        controller.add_paths_to_layer(layer.id, new_paths, "Draw Shape")
        assert len(layer.paths) == 2

        # Undo
        controller.undo_stack.undo()
        assert len(layer.paths) == 1
        assert layer.paths[0] == existing[0]

    def test_multiple_appends_accumulate(self, controller):
        layer = controller.current_project.layers[0]
        layer.paths = []

        controller.add_paths_to_layer(layer.id, [[(0, 0), (1, 1)]], "Shape 1")
        controller.add_paths_to_layer(layer.id, [[(2, 2), (3, 3)]], "Shape 2")
        controller.add_paths_to_layer(layer.id, [[(4, 4), (5, 5)]], "Shape 3")
        assert len(layer.paths) == 3

        # Undo only shape 3
        controller.undo_stack.undo()
        assert len(layer.paths) == 2

        # Undo shape 2
        controller.undo_stack.undo()
        assert len(layer.paths) == 1

    def test_append_empty_paths_is_noop(self, controller):
        layer = controller.current_project.layers[0]
        layer.paths = [[(0, 0), (1, 1)]]
        controller.add_paths_to_layer(layer.id, [], "Empty")
        # Should not push a command for empty paths
        assert len(layer.paths) == 1

    def test_append_marks_project_modified(self, controller):
        layer = controller.current_project.layers[0]
        assert not controller.modified
        controller.add_paths_to_layer(layer.id, [[(0, 0), (5, 5)]], "Test")
        assert controller.modified

    def test_paths_changed_signal_emitted(self, controller, qtbot):
        layer = controller.current_project.layers[0]
        with qtbot.waitSignal(controller.paths_changed, timeout=1000) as blocker:
            controller.add_paths_to_layer(layer.id, [[(0, 0), (5, 5)]], "Test")
        assert blocker.args == [layer.id]


# ---------------------------------------------------------------------------
# ProjectController.add_paths_to_layer
# ---------------------------------------------------------------------------


class TestAddPathsToLayer:
    def test_does_not_replace_existing_paths(self, controller):
        layer = controller.current_project.layers[0]
        layer.paths = [[(0.0, 0.0), (5.0, 5.0)], [(1.0, 0.0), (6.0, 5.0)]]
        controller.add_paths_to_layer(layer.id, [[(10.0, 10.0), (15.0, 15.0)]])
        assert len(layer.paths) == 3

    def test_invalid_layer_id_is_noop(self, controller):
        # Should not raise
        controller.add_paths_to_layer("nonexistent-id", [[(0, 0), (1, 1)]])

    def test_redo_after_undo(self, controller):
        layer = controller.current_project.layers[0]
        layer.paths = []
        controller.add_paths_to_layer(layer.id, [[(0, 0), (5, 5)]], "Draw")
        controller.undo_stack.undo()
        assert len(layer.paths) == 0
        controller.undo_stack.redo()
        assert len(layer.paths) == 1


# ---------------------------------------------------------------------------
# Fill functions (from contour.py)
# ---------------------------------------------------------------------------


class TestFillFunctions:
    def _square_polygon(self):
        """Return a 10×10 square starting at (5,5)."""
        return [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0), (5.0, 5.0)]

    def test_hatch_fill_produces_lines(self):
        from plottter.generators.contour import _fill_polygon_hatch
        pts = self._square_polygon()
        lines = _fill_polygon_hatch(pts, [], 45.0, 1.0)
        assert len(lines) > 0
        for line in lines:
            assert len(line) >= 2

    def test_hatch_fill_clipped_to_boundary(self):
        """All hatch line endpoints should lie within the polygon bounding box."""
        from plottter.generators.contour import _fill_polygon_hatch
        pts = self._square_polygon()
        lines = _fill_polygon_hatch(pts, [], 0.0, 1.0)
        for line in lines:
            for x, y in line:
                assert 4.9 <= x <= 15.1, f"x={x} out of boundary"
                assert 4.9 <= y <= 15.1, f"y={y} out of boundary"

    def test_concentric_fill_produces_rings(self):
        from plottter.generators.contour import _fill_polygon_concentric
        pts = self._square_polygon()
        rings = _fill_polygon_concentric(pts, [], 1.0)
        assert len(rings) > 0

    def test_concentric_rings_shrink_inward(self):
        """Concentric rings should all lie inside the original polygon."""
        from plottter.generators.contour import _fill_polygon_concentric
        from shapely.geometry import Polygon, Point
        pts = self._square_polygon()
        rings = _fill_polygon_concentric(pts, [], 0.5)
        outer = Polygon(pts)
        for ring in rings:
            for x, y in ring:
                assert outer.contains(Point(x, y)) or outer.boundary.distance(Point(x, y)) < 0.1


# ---------------------------------------------------------------------------
# SettingsPanel: Shape Drawing group
# ---------------------------------------------------------------------------


class TestSettingsPanelShapeDrawing:
    def test_shape_draw_group_exists(self, settings_panel):
        assert hasattr(settings_panel, "_shape_draw_group")

    def test_shape_draw_group_visible_in_shape_mode(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        assert settings_panel._shape_draw_group.isVisible()

    def test_shape_draw_group_hidden_in_math_art_mode(self, settings_panel):
        settings_panel.on_mode_changed("Math Art")
        assert not settings_panel._shape_draw_group.isVisible()

    def test_fill_spacing_hidden_when_no_fill(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        settings_panel._sd_fill_combo.setCurrentText("None")
        assert not settings_panel._sd_fill_spacing_spin.isVisible()
        assert not settings_panel._sd_fill_spacing_label.isVisible()

    def test_fill_spacing_shown_when_hatching(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        settings_panel._sd_fill_combo.setCurrentText("Hatching")
        assert settings_panel._sd_fill_spacing_spin.isVisible()

    def test_fill_angle_shown_for_hatching(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        settings_panel._sd_fill_combo.setCurrentText("Hatching")
        assert settings_panel._sd_fill_angle_spin.isVisible()

    def test_fill_angle_shown_for_crosshatch(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        settings_panel._sd_fill_combo.setCurrentText("Cross-hatch")
        assert settings_panel._sd_fill_angle_spin.isVisible()

    def test_fill_angle_hidden_for_concentric(self, settings_panel):
        settings_panel.on_mode_changed("Shape Drawing")
        settings_panel._sd_fill_combo.setCurrentText("Concentric")
        assert not settings_panel._sd_fill_angle_spin.isVisible()

    def test_sd_tool_combo_has_all_tools(self, settings_panel):
        tools = [settings_panel._sd_tool_combo.itemText(i)
                 for i in range(settings_panel._sd_tool_combo.count())]
        assert "Rectangle" in tools
        assert "Ellipse" in tools
        assert "Polygon" in tools
        assert "Freehand" in tools
        assert "Line/Polyline" in tools

    def test_stroke_checkbox_default_true(self, settings_panel):
        assert settings_panel._sd_stroke_check.isChecked()

    def test_smooth_spin_default_zero(self, settings_panel):
        assert settings_panel._sd_smooth_spin.value() == 0

    def test_on_shape_drawn_stroke_only(self, settings_panel, controller):
        """With stroke=True and fill=None, shape_drawn produces exactly the outline."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        # Set up: stroke on, no fill
        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("None")

        # Pick the target layer
        settings_panel._sd_target_layer_combo.blockSignals(True)
        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break
        settings_panel._sd_target_layer_combo.blockSignals(False)

        rect = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        settings_panel._on_shape_drawn(rect)

        assert len(layer.paths) == 1
        assert layer.paths[0] == rect

    def test_on_shape_drawn_no_stroke(self, settings_panel, controller):
        """With stroke=False and no fill, shape_drawn produces no paths (empty → skipped)."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(False)
        settings_panel._sd_fill_combo.setCurrentText("None")

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        rect = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        settings_panel._on_shape_drawn(rect)

        assert len(layer.paths) == 0

    def test_on_shape_drawn_hatching_adds_fill(self, settings_panel, controller):
        """Hatching fill adds more than just the stroke outline."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("Hatching")
        settings_panel._sd_fill_spacing_spin.setValue(2.0)

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        # A 20×20 square — should produce several hatch lines
        rect = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)]
        settings_panel._on_shape_drawn(rect)

        # Should have the stroke outline plus hatch lines
        assert len(layer.paths) > 1

    def test_on_shape_drawn_multiple_shapes_accumulate(self, settings_panel, controller):
        """Each shape_drawn call appends to layer without replacing."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("None")

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        rect1 = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        rect2 = [(10.0, 10.0), (15.0, 10.0), (15.0, 15.0), (10.0, 15.0), (10.0, 10.0)]

        settings_panel._on_shape_drawn(rect1)
        settings_panel._on_shape_drawn(rect2)

        assert len(layer.paths) == 2

    def test_on_shape_drawn_undo_removes_last_shape(self, settings_panel, controller):
        """Undo after drawing two shapes removes only the second one."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("None")

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        rect1 = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        rect2 = [(10.0, 10.0), (15.0, 10.0), (15.0, 15.0), (10.0, 15.0), (10.0, 10.0)]

        settings_panel._on_shape_drawn(rect1)
        settings_panel._on_shape_drawn(rect2)
        assert len(layer.paths) == 2

        controller.undo_stack.undo()
        assert len(layer.paths) == 1
        # rect1 should remain
        assert layer.paths[0] == rect1

    def test_on_shape_drawn_degenerate_open_polyline_no_fill(self, settings_panel, controller):
        """An open polyline (line tool) with no fill still adds the stroke."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("Hatching")

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        # Open polyline (line tool output — first != last)
        line = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        settings_panel._on_shape_drawn(line)

        # Since it's not closed, fill should be skipped. Stroke should be present.
        assert len(layer.paths) == 1
        assert layer.paths[0] == line

    def test_on_shape_drawn_ignores_too_short_polyline(self, settings_panel, controller):
        """A single-point or empty polyline is silently ignored."""
        layer = controller.current_project.layers[0]
        layer.paths = []

        settings_panel._sd_stroke_check.setChecked(True)
        settings_panel._sd_fill_combo.setCurrentText("None")

        for i in range(settings_panel._sd_target_layer_combo.count()):
            if settings_panel._sd_target_layer_combo.itemData(i) == layer.id:
                settings_panel._sd_target_layer_combo.setCurrentIndex(i)
                break

        settings_panel._on_shape_drawn([])
        settings_panel._on_shape_drawn([(5.0, 5.0)])
        assert len(layer.paths) == 0
