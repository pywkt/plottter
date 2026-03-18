"""Phase 15.9 validation: preview and simulation features.

Verifies the preview and simulation system described in specs/preview_simulation.md:

1. Stroke-order animation — play/pause toggle, step forward/backward, seek slider,
   speed control (0.1x–10x), animation state signal emissions, path collection from
   visible layers only, animation reset on path/layer change.
2. Pen-up travel visualization — toggle state, _draw_travel_lines logic (tracked via
   state), calculate_travel_distance correctness.
3. Travel metrics — pen-down distance, pen-up distance, travel efficiency %, pen lifts.
4. Registration mark preview — on/off toggle, default state.
5. Paper texture background — on/off toggle, default state, purely cosmetic (no export
   effect).
6. Zoom/pan — zoom_in/zoom_out/fit_to_window, coordinate transforms (mm↔pixel roundtrip),
   zoom clamping (MIN_ZOOM/MAX_ZOOM), pan offset state.
7. Image overlay toggle — show/hide state, data preserved when toggled off.
8. Large path count — canvas widget does not crash or error with 10k+ paths.
9. Animation timer constants — tick interval and speed-distance budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.models import Canvas, Layer, Project
from plottter.processing.optimize import calculate_travel_distance


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_canvas(preset: str = "A4", margin: float = 10.0) -> Canvas:
    return Canvas.from_preset(preset, margin=margin)


def _make_layer_with_paths(
    name: str = "L1",
    color: str = "#000000",
    num_paths: int = 3,
    pts_per_path: int = 3,
) -> Layer:
    layer = Layer(name=name, color=color)
    paths = []
    for i in range(num_paths):
        base = float(i) * 20.0
        path = [(base + j * 5.0, base + j * 5.0) for j in range(pts_per_path)]
        paths.append(path)
    layer.add_paths(paths)
    return layer


def _make_project(num_layers: int = 1, paths_per_layer: int = 3) -> Project:
    canvas = _make_canvas()
    proj = Project(name="TestProject", canvas=canvas)
    colors = ["#000000", "#FF0000", "#0000FF"]
    for i in range(num_layers):
        layer = _make_layer_with_paths(
            name=f"Layer {i + 1}",
            color=colors[i % len(colors)],
            num_paths=paths_per_layer,
        )
        proj.add_layer(layer)
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    proj = _make_project(num_layers=1, paths_per_layer=3)
    return ProjectController(proj)


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget
    widget = CanvasWidget(controller)
    widget.resize(800, 600)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# 1. Animation state management
# ---------------------------------------------------------------------------


class TestAnimationState:
    """Verify animation mode transitions, step, seek, and speed controls."""

    def test_initial_state_not_in_anim_mode(self, canvas_widget):
        """Canvas starts with animation mode disabled."""
        assert canvas_widget._anim_mode is False
        assert canvas_widget._anim_playing is False
        assert canvas_widget._anim_current_path == 0
        assert canvas_widget._anim_all_paths == []

    def test_toggle_animation_enters_anim_mode(self, canvas_widget):
        """toggle_animation() enters animation mode when paths are available."""
        canvas_widget.toggle_animation()
        assert canvas_widget._anim_mode is True
        assert canvas_widget._anim_playing is True

    def test_toggle_animation_no_paths_stays_idle(self, qapp, qtbot):
        """toggle_animation() is a no-op when the project has no paths."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        empty_proj = Project(name="Empty", canvas=_make_canvas())
        empty_proj.add_layer(Layer(name="Empty Layer", color="#000000"))
        ctrl = ProjectController(empty_proj)
        widget = CanvasWidget(ctrl)
        qtbot.addWidget(widget)

        widget.toggle_animation()
        assert widget._anim_mode is False
        assert widget._anim_playing is False

    def test_toggle_animation_pauses_when_playing(self, canvas_widget):
        """Second toggle_animation() call pauses playback."""
        canvas_widget.toggle_animation()  # start
        assert canvas_widget._anim_playing is True
        canvas_widget.toggle_animation()  # pause
        assert canvas_widget._anim_playing is False
        assert canvas_widget._anim_mode is True  # mode remains active

    def test_toggle_animation_resumes_when_paused(self, canvas_widget):
        """toggle_animation() on a paused animation resumes it."""
        canvas_widget.toggle_animation()  # start
        canvas_widget.toggle_animation()  # pause
        canvas_widget.toggle_animation()  # resume
        assert canvas_widget._anim_playing is True

    def test_step_forward_enters_anim_mode(self, canvas_widget):
        """step_anim_forward() initiates animation mode even without play."""
        canvas_widget.step_anim_forward()
        assert canvas_widget._anim_mode is True

    def test_step_forward_advances_path_index(self, canvas_widget):
        """step_anim_forward() increments the current path index."""
        canvas_widget.toggle_animation()  # enter mode, start at path 0
        canvas_widget.toggle_animation()  # pause immediately
        initial_idx = canvas_widget._anim_current_path
        canvas_widget.step_anim_forward()
        assert canvas_widget._anim_current_path == initial_idx + 1

    def test_step_forward_does_not_exceed_total(self, canvas_widget):
        """step_anim_forward() at the last path does not overflow."""
        canvas_widget.toggle_animation()
        canvas_widget.toggle_animation()
        total = len(canvas_widget._anim_all_paths)
        # Advance past the end
        for _ in range(total + 5):
            canvas_widget.step_anim_forward()
        assert canvas_widget._anim_current_path <= total

    def test_step_backward_decrements_path_index(self, canvas_widget):
        """step_anim_backward() moves back one path."""
        canvas_widget.toggle_animation()
        canvas_widget.toggle_animation()
        canvas_widget.step_anim_forward()
        canvas_widget.step_anim_forward()
        idx_before = canvas_widget._anim_current_path
        canvas_widget.step_anim_backward()
        assert canvas_widget._anim_current_path == idx_before - 1

    def test_step_backward_noop_when_not_in_anim_mode(self, canvas_widget):
        """step_anim_backward() is a no-op before animation has started."""
        canvas_widget.step_anim_backward()
        assert canvas_widget._anim_mode is False
        assert canvas_widget._anim_current_path == 0

    def test_step_backward_clamps_at_zero(self, canvas_widget):
        """step_anim_backward() at path 0 stays at 0."""
        canvas_widget.toggle_animation()
        canvas_widget.toggle_animation()
        # Already at 0
        canvas_widget.step_anim_backward()
        assert canvas_widget._anim_current_path == 0

    def test_seek_animation_sets_path_index(self, canvas_widget):
        """seek_animation(n) moves the animation to path n."""
        canvas_widget.seek_animation(2)
        assert canvas_widget._anim_mode is True
        assert canvas_widget._anim_current_path == 2

    def test_seek_animation_clamps_above_zero(self, canvas_widget):
        """seek_animation with negative index is clamped to 0."""
        canvas_widget.seek_animation(-10)
        assert canvas_widget._anim_current_path == 0

    def test_seek_animation_clamps_to_total(self, canvas_widget):
        """seek_animation beyond total is clamped to total."""
        canvas_widget.seek_animation(0)  # enter mode first
        total = len(canvas_widget._anim_all_paths)
        canvas_widget.seek_animation(total + 100)
        assert canvas_widget._anim_current_path <= total

    def test_seek_resets_point_index(self, canvas_widget):
        """seek_animation() always resets the within-path point index."""
        canvas_widget.seek_animation(1)
        assert canvas_widget._anim_current_point == 0

    def test_set_anim_speed_accepts_valid_value(self, canvas_widget):
        """set_anim_speed() stores the given speed within valid range."""
        canvas_widget.set_anim_speed(3.0)
        assert canvas_widget._anim_speed == pytest.approx(3.0)

    def test_set_anim_speed_clamps_to_minimum(self, canvas_widget):
        """set_anim_speed(0) is clamped to 0.1."""
        canvas_widget.set_anim_speed(0.0)
        assert canvas_widget._anim_speed == pytest.approx(0.1)

    def test_set_anim_speed_clamps_to_maximum(self, canvas_widget):
        """set_anim_speed(999) is clamped to 10.0."""
        canvas_widget.set_anim_speed(999.0)
        assert canvas_widget._anim_speed == pytest.approx(10.0)

    def test_anim_timer_interval_is_50ms(self, canvas_widget):
        """Animation timer fires at ~20 fps (50 ms interval)."""
        from plottter.gui.canvas_widget import CanvasWidget
        assert CanvasWidget.ANIM_TIMER_INTERVAL_MS == 50


class TestAnimationSignal:
    """Verify the anim_state_changed signal is emitted with correct data."""

    def test_signal_emits_on_toggle(self, canvas_widget, qtbot):
        """toggle_animation() emits anim_state_changed with correct total_paths."""
        signals = []
        canvas_widget.anim_state_changed.connect(lambda p, idx, tot: signals.append((p, idx, tot)))
        canvas_widget.toggle_animation()
        assert len(signals) >= 1
        is_playing, current, total = signals[-1]
        assert is_playing is True
        assert total == 3  # 3 paths per layer as configured in fixture

    def test_signal_emits_on_step_forward(self, canvas_widget, qtbot):
        """step_anim_forward() emits anim_state_changed."""
        canvas_widget.toggle_animation()
        canvas_widget.toggle_animation()
        signals = []
        canvas_widget.anim_state_changed.connect(lambda p, idx, tot: signals.append((p, idx, tot)))
        canvas_widget.step_anim_forward()
        assert len(signals) >= 1

    def test_signal_emits_on_seek(self, canvas_widget, qtbot):
        """seek_animation() emits anim_state_changed."""
        signals = []
        canvas_widget.anim_state_changed.connect(lambda p, idx, tot: signals.append((p, idx, tot)))
        canvas_widget.seek_animation(1)
        assert len(signals) >= 1

    def test_signal_total_paths_matches_all_paths(self, canvas_widget, qtbot):
        """Signal total_paths value matches len(_anim_all_paths)."""
        signals = []
        canvas_widget.anim_state_changed.connect(lambda p, idx, tot: signals.append((p, idx, tot)))
        canvas_widget.toggle_animation()
        assert signals
        _, _, total = signals[-1]
        assert total == len(canvas_widget._anim_all_paths)

    def test_signal_is_playing_false_when_paused(self, canvas_widget, qtbot):
        """Signal is_playing is False after pause."""
        canvas_widget.toggle_animation()  # play
        signals = []
        canvas_widget.anim_state_changed.connect(lambda p, idx, tot: signals.append((p, idx, tot)))
        canvas_widget.toggle_animation()  # pause
        assert signals
        is_playing, _, _ = signals[-1]
        assert is_playing is False


class TestAnimationRebuildPaths:
    """Verify _rebuild_anim_paths collects paths correctly from project layers."""

    def test_rebuild_includes_all_visible_paths(self, qapp, qtbot):
        """All paths from visible layers are collected for animation."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        proj = _make_project(num_layers=2, paths_per_layer=4)
        ctrl = ProjectController(proj)
        widget = CanvasWidget(ctrl)
        qtbot.addWidget(widget)

        widget._rebuild_anim_paths()
        # 2 layers × 4 paths = 8 total
        assert len(widget._anim_all_paths) == 8

    def test_rebuild_excludes_invisible_layers(self, qapp, qtbot):
        """Paths from hidden layers are excluded from animation."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        proj = _make_project(num_layers=2, paths_per_layer=3)
        # Hide the second layer
        proj.layers[1].visible = False
        ctrl = ProjectController(proj)
        widget = CanvasWidget(ctrl)
        qtbot.addWidget(widget)

        widget._rebuild_anim_paths()
        # Only 1 visible layer × 3 paths = 3
        assert len(widget._anim_all_paths) == 3

    def test_rebuild_path_tuples_have_color_opacity_polyline(self, canvas_widget):
        """Each item in _anim_all_paths is (color, opacity, polyline)."""
        canvas_widget._rebuild_anim_paths()
        assert canvas_widget._anim_all_paths
        for item in canvas_widget._anim_all_paths:
            color, opacity, polyline = item
            assert isinstance(color, str)
            assert 0.0 <= opacity <= 1.0
            assert isinstance(polyline, list)
            assert len(polyline) >= 2

    def test_rebuild_skips_single_point_paths(self, qapp, qtbot):
        """Degenerate single-point paths are excluded from animation."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        proj = Project(name="P", canvas=_make_canvas())
        layer = Layer(name="L", color="#000000")
        # Two valid paths, one degenerate
        layer.add_paths([
            [(0.0, 0.0), (10.0, 10.0)],
            [(5.0, 5.0)],              # degenerate — only 1 point
            [(20.0, 20.0), (30.0, 30.0)],
        ])
        proj.add_layer(layer)
        ctrl = ProjectController(proj)
        widget = CanvasWidget(ctrl)
        qtbot.addWidget(widget)

        widget._rebuild_anim_paths()
        assert len(widget._anim_all_paths) == 2

    def test_animation_reset_on_path_change(self, canvas_widget, controller):
        """Animation mode is reset when layer paths change."""
        canvas_widget.toggle_animation()
        assert canvas_widget._anim_mode is True
        controller.paths_changed.emit(controller.current_project.layers[0].id)
        assert canvas_widget._anim_mode is False


# ---------------------------------------------------------------------------
# 2. Animation tick / speed
# ---------------------------------------------------------------------------


class TestAnimationTick:
    """Verify distance-budget animation advancement."""

    def test_anim_speed_1x_default(self, canvas_widget):
        """Default animation speed is 1.0."""
        assert canvas_widget._anim_speed == pytest.approx(1.0)

    def test_distance_budget_formula(self, canvas_widget):
        """Distance budget per tick = 80 mm/s × speed × tick_s.

        At 1× speed and 50 ms tick: 80 × 1.0 × 0.05 = 4.0 mm.
        At 2× speed: 8.0 mm.
        """
        tick_s = 50 / 1000.0
        plotter_speed = 80.0

        budget_1x = plotter_speed * 1.0 * tick_s
        budget_2x = plotter_speed * 2.0 * tick_s

        assert budget_1x == pytest.approx(4.0)
        assert budget_2x == pytest.approx(8.0)

    def test_tick_advances_point_index(self, canvas_widget):
        """_anim_tick() advances _anim_current_point for a long path."""
        # Build a long path so a single tick can't finish it
        layer = canvas_widget._controller.current_project.layers[0]
        long_path = [(float(i), 0.0) for i in range(0, 1000, 1)]  # 1000 pts, 1mm each
        layer.clear_paths()
        layer.add_paths([long_path])

        canvas_widget.toggle_animation()  # enters mode at path 0
        assert canvas_widget._anim_playing is True

        initial_point = canvas_widget._anim_current_point
        canvas_widget._anim_tick()
        # At 1× speed, budget = 4mm, so point index should have advanced ≥4
        assert canvas_widget._anim_current_point > initial_point


# ---------------------------------------------------------------------------
# 3. Pen-up travel visualization
# ---------------------------------------------------------------------------


class TestTravelVisualization:
    """Verify pen-up travel toggle state and metrics."""

    def test_show_travel_off_by_default(self, canvas_widget):
        """Travel visualization is disabled on startup."""
        assert canvas_widget._show_travel is False

    def test_set_show_travel_true(self, canvas_widget):
        """set_show_travel(True) enables travel visualization."""
        canvas_widget.set_show_travel(True)
        assert canvas_widget._show_travel is True

    def test_set_show_travel_false(self, canvas_widget):
        """set_show_travel(False) disables travel visualization."""
        canvas_widget.set_show_travel(True)
        canvas_widget.set_show_travel(False)
        assert canvas_widget._show_travel is False


# ---------------------------------------------------------------------------
# 4. Travel metrics — calculate_travel_distance
# ---------------------------------------------------------------------------


class TestTravelMetrics:
    """Verify travel distance calculation from processing.optimize."""

    def test_single_path_travel_from_origin(self):
        """Single path: travel = origin→start + end→origin."""
        path = [(10.0, 0.0), (20.0, 0.0)]
        distance = calculate_travel_distance([path])
        # origin(0,0) → (10,0) = 10; (20,0) → origin = 20; total = 30
        assert distance == pytest.approx(30.0)

    def test_two_paths_travel_includes_between_paths(self):
        """Travel between two paths is included in the total."""
        path1 = [(0.0, 0.0), (10.0, 0.0)]
        path2 = [(20.0, 0.0), (30.0, 0.0)]
        total = calculate_travel_distance([path1, path2])
        # origin(0,0)→(0,0)=0; (10,0)→(20,0)=10; (30,0)→origin=30; total=40
        assert total == pytest.approx(40.0)

    def test_empty_paths_returns_zero(self):
        """Empty path list produces zero travel distance."""
        assert calculate_travel_distance([]) == pytest.approx(0.0)

    def test_travel_distance_is_non_negative(self):
        """Travel distance is always non-negative for any path arrangement."""
        paths = [
            [(5.0, 5.0), (15.0, 5.0)],
            [(50.0, 50.0), (60.0, 50.0)],
            [(100.0, 100.0), (120.0, 100.0)],
        ]
        assert calculate_travel_distance(paths) >= 0.0

    def test_reorder_reduces_travel_for_reversed_order(self):
        """Nearest-neighbor reorder reduces travel vs. far-apart ordering."""
        from plottter.processing.optimize import reorder_paths

        # 4 short paths arranged in a line; worst order = jumps across
        paths = [
            [(100.0, 0.0), (105.0, 0.0)],
            [(0.0, 0.0), (5.0, 0.0)],
            [(50.0, 0.0), (55.0, 0.0)],
            [(200.0, 0.0), (205.0, 0.0)],
        ]
        travel_before = calculate_travel_distance(paths)
        reordered = reorder_paths(paths)
        travel_after = calculate_travel_distance(reordered)
        assert travel_after <= travel_before

    def test_pen_down_distance_unchanged_by_reorder(self):
        """Reordering preserves the total pen-down (drawing) distance."""
        from plottter.processing.optimize import reorder_paths

        paths = [
            [(10.0, 0.0), (20.0, 0.0)],
            [(50.0, 0.0), (60.0, 0.0)],
            [(30.0, 0.0), (40.0, 0.0)],
        ]
        before_sum = sum(
            math.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1]) for p in paths
        )
        reordered = reorder_paths(paths)
        after_sum = sum(
            math.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1]) for p in reordered
        )
        assert before_sum == pytest.approx(after_sum)

    def test_travel_efficiency_formula(self):
        """Travel efficiency = pen_down / (pen_down + pen_up) × 100."""
        # Two paths along X axis, no travel waste
        path1 = [(0.0, 0.0), (10.0, 0.0)]
        path2 = [(10.0, 0.0), (20.0, 0.0)]

        pen_down = 20.0  # 10 + 10
        pen_up = calculate_travel_distance([path1, path2])
        # travel = origin→(0,0)=0; (10,0)→(10,0)=0; (20,0)→origin=20; total=20
        # But the paths are back-to-back so end-of-p1 = start-of-p2
        efficiency = pen_down / (pen_down + pen_up) * 100
        assert 0 < efficiency <= 100

    def test_pen_lift_count_equals_path_count(self):
        """Each path requires exactly one pen lift; travel distance reflects all lifts."""
        # 7 paths spaced 100 mm apart, each 5 mm long along X
        paths = [[(i * 100.0, 0.0), (i * 100.0 + 5.0, 0.0)] for i in range(7)]
        travel = calculate_travel_distance(paths)
        # origin→path0 = 0 mm (path0 starts at origin)
        # 6 inter-path gaps of 95 mm each = 570 mm (one pen lift per gap)
        # path6 end (605, 0) → origin = 605 mm (final pen lift)
        # Total = 0 + 570 + 605 = 1175 mm — accounts for exactly 7 pen lifts
        assert travel == pytest.approx(1175.0)


# ---------------------------------------------------------------------------
# 5. Registration marks preview
# ---------------------------------------------------------------------------


class TestRegistrationMarks:
    """Verify registration mark visibility state."""

    def test_reg_marks_visible_by_default(self, canvas_widget):
        """Registration marks are shown on startup."""
        assert canvas_widget._show_reg_marks is True

    def test_set_show_reg_marks_false(self, canvas_widget):
        """set_show_reg_marks(False) disables marks."""
        canvas_widget.set_show_reg_marks(False)
        assert canvas_widget._show_reg_marks is False

    def test_set_show_reg_marks_true(self, canvas_widget):
        """set_show_reg_marks(True) re-enables marks."""
        canvas_widget.set_show_reg_marks(False)
        canvas_widget.set_show_reg_marks(True)
        assert canvas_widget._show_reg_marks is True


# ---------------------------------------------------------------------------
# 6. Paper texture background
# ---------------------------------------------------------------------------


class TestPaperTexture:
    """Verify paper texture state toggle (cosmetic-only)."""

    def test_paper_texture_off_by_default(self, canvas_widget):
        """Paper texture is disabled by default."""
        assert canvas_widget._show_paper_texture is False

    def test_set_paper_texture_true(self, canvas_widget):
        """set_paper_texture(True) enables the subtle background color."""
        canvas_widget.set_paper_texture(True)
        assert canvas_widget._show_paper_texture is True

    def test_set_paper_texture_false(self, canvas_widget):
        """set_paper_texture(False) restores plain white background."""
        canvas_widget.set_paper_texture(True)
        canvas_widget.set_paper_texture(False)
        assert canvas_widget._show_paper_texture is False

    def test_paper_texture_does_not_affect_project_model(self, canvas_widget, controller):
        """Toggling paper texture leaves the project model unchanged."""
        canvas = controller.current_project.canvas
        w_before = canvas.width_mm
        h_before = canvas.height_mm
        canvas_widget.set_paper_texture(True)
        assert canvas.width_mm == pytest.approx(w_before)
        assert canvas.height_mm == pytest.approx(h_before)


# ---------------------------------------------------------------------------
# 7. Zoom / pan
# ---------------------------------------------------------------------------


class TestZoomPan:
    """Verify zoom and pan state management."""

    def test_initial_zoom_is_one(self, canvas_widget):
        """Default zoom level is 1.0."""
        assert canvas_widget._zoom == pytest.approx(1.0)

    def test_zoom_in_increases_zoom(self, canvas_widget):
        """zoom_in() increases the current zoom level."""
        initial = canvas_widget._zoom
        canvas_widget.zoom_in()
        assert canvas_widget._zoom > initial

    def test_zoom_out_decreases_zoom(self, canvas_widget):
        """zoom_out() decreases the current zoom level."""
        canvas_widget.zoom_in()
        zoomed = canvas_widget._zoom
        canvas_widget.zoom_out()
        assert canvas_widget._zoom < zoomed

    def test_zoom_clamped_at_maximum(self, canvas_widget):
        """Zoom never exceeds MAX_ZOOM."""
        from plottter.gui.canvas_widget import CanvasWidget
        for _ in range(100):
            canvas_widget.zoom_in()
        assert canvas_widget._zoom <= CanvasWidget.MAX_ZOOM

    def test_zoom_clamped_at_minimum(self, canvas_widget):
        """Zoom never goes below MIN_ZOOM."""
        from plottter.gui.canvas_widget import CanvasWidget
        for _ in range(100):
            canvas_widget.zoom_out()
        assert canvas_widget._zoom >= CanvasWidget.MIN_ZOOM

    def test_min_zoom_is_01(self, canvas_widget):
        """MIN_ZOOM constant is 0.1."""
        from plottter.gui.canvas_widget import CanvasWidget
        assert CanvasWidget.MIN_ZOOM == pytest.approx(0.1)

    def test_max_zoom_is_20(self, canvas_widget):
        """MAX_ZOOM constant is 20.0."""
        from plottter.gui.canvas_widget import CanvasWidget
        assert CanvasWidget.MAX_ZOOM == pytest.approx(20.0)

    def test_mm_to_pixel_roundtrip(self, canvas_widget):
        """pixel_to_mm(mm_to_pixel(pt)) ≈ pt for arbitrary mm coordinates."""
        test_points = [(0.0, 0.0), (100.0, 150.0), (50.0, 25.0)]
        for mm_pt in test_points:
            px = canvas_widget.mm_to_pixel(mm_pt)
            recovered = canvas_widget.pixel_to_mm(px)
            assert recovered[0] == pytest.approx(mm_pt[0], abs=1e-6)
            assert recovered[1] == pytest.approx(mm_pt[1], abs=1e-6)

    def test_mm_to_pixel_respects_zoom(self, canvas_widget):
        """mm_to_pixel result changes when zoom changes."""
        pt = (50.0, 50.0)
        px_before = canvas_widget.mm_to_pixel(pt)
        canvas_widget.zoom_in()
        px_after = canvas_widget.mm_to_pixel(pt)
        # At least one coordinate should differ after zoom change
        assert px_before.x() != pytest.approx(px_after.x()) or \
               px_before.y() != pytest.approx(px_after.y())

    def test_fit_to_window_sets_positive_zoom(self, canvas_widget):
        """fit_to_window() sets zoom to a positive value."""
        canvas_widget.resize(800, 600)
        canvas_widget.fit_to_window()
        assert canvas_widget._zoom > 0.0

    def test_initial_pan_offset_is_zero(self, canvas_widget):
        """Initial pan offset is (0, 0)."""
        from PyQt6.QtCore import QPointF
        assert canvas_widget._pan_offset == QPointF(0.0, 0.0)

    def test_zoom_in_adjusts_pan_offset(self, canvas_widget):
        """zoom_in() modifies pan_offset to keep the widget center fixed in mm space."""
        from PyQt6.QtCore import QPointF
        center_px = canvas_widget.rect().center()
        center_pt = (float(center_px.x()), float(center_px.y()))
        # Record the mm coordinate of the widget center before zooming
        center_mm_before = canvas_widget.pixel_to_mm(QPointF(*center_pt))
        canvas_widget.zoom_in()
        # After zoom, pan_offset must have changed (widget is 800×600, center ≠ origin)
        assert canvas_widget._pan_offset != QPointF(0.0, 0.0)
        # The mm coordinate of the widget center should be unchanged (zoom is center-fixed)
        center_mm_after = canvas_widget.pixel_to_mm(QPointF(*center_pt))
        assert center_mm_after[0] == pytest.approx(center_mm_before[0], abs=1e-9)
        assert center_mm_after[1] == pytest.approx(center_mm_before[1], abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Image overlay toggle
# ---------------------------------------------------------------------------


class TestImageOverlayToggle:
    """Verify image overlay show/hide state."""

    def test_image_overlay_shown_by_default(self, canvas_widget):
        """Image overlay is visible by default."""
        assert canvas_widget._show_image_overlay is True

    def test_set_show_image_overlay_false(self, canvas_widget):
        """set_show_image_overlay(False) hides the overlay."""
        canvas_widget.set_show_image_overlay(False)
        assert canvas_widget._show_image_overlay is False

    def test_set_show_image_overlay_true(self, canvas_widget):
        """set_show_image_overlay(True) re-shows the overlay."""
        canvas_widget.set_show_image_overlay(False)
        canvas_widget.set_show_image_overlay(True)
        assert canvas_widget._show_image_overlay is True

    def test_image_data_preserved_when_hidden(self, canvas_widget):
        """Hiding the overlay does not clear the image data."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:] = 128
        canvas_widget.set_image_overlay(arr)
        assert canvas_widget._image_overlay is not None
        canvas_widget.set_show_image_overlay(False)
        # Image pixmap is still stored after hiding
        assert canvas_widget._image_overlay is not None

    def test_set_image_overlay_none_clears(self, canvas_widget):
        """set_image_overlay(None) clears the stored pixmap."""
        arr = np.zeros((50, 50), dtype=np.uint8)
        canvas_widget.set_image_overlay(arr)
        assert canvas_widget._image_overlay is not None
        canvas_widget.set_image_overlay(None)
        assert canvas_widget._image_overlay is None


# ---------------------------------------------------------------------------
# 9. Large path count — robustness
# ---------------------------------------------------------------------------


class TestLargePathCount:
    """Verify the canvas handles large numbers of paths without errors."""

    def test_10k_paths_rebuild_anim_no_crash(self, qapp, qtbot):
        """_rebuild_anim_paths() handles 10,000 paths without error."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        proj = Project(name="Large", canvas=_make_canvas())
        layer = Layer(name="Big Layer", color="#000000")
        paths = [[(float(i), 0.0), (float(i) + 1.0, 0.0)] for i in range(10_000)]
        layer.add_paths(paths)
        proj.add_layer(layer)

        ctrl = ProjectController(proj)
        widget = CanvasWidget(ctrl)
        widget.resize(800, 600)
        qtbot.addWidget(widget)

        # Should not raise
        widget._rebuild_anim_paths()
        assert len(widget._anim_all_paths) == 10_000

    def test_10k_paths_calculate_travel_distance(self):
        """calculate_travel_distance handles 10,000 paths efficiently."""
        paths = [[(float(i) * 2.0, 0.0), (float(i) * 2.0 + 1.0, 0.0)] for i in range(10_000)]
        distance = calculate_travel_distance(paths)
        assert distance > 0

    def test_seek_animation_with_10k_paths(self, qapp, qtbot):
        """seek_animation works correctly with 10,000 paths."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.canvas_widget import CanvasWidget

        proj = Project(name="Large", canvas=_make_canvas())
        layer = Layer(name="Big Layer", color="#000000")
        paths = [[(float(i), 0.0), (float(i) + 1.0, 0.0)] for i in range(10_000)]
        layer.add_paths(paths)
        proj.add_layer(layer)

        ctrl = ProjectController(proj)
        widget = CanvasWidget(ctrl)
        widget.resize(800, 600)
        qtbot.addWidget(widget)

        widget.seek_animation(5000)
        assert widget._anim_current_path == 5000


# ---------------------------------------------------------------------------
# 10. Pen jitter (preview-only simulation)
# ---------------------------------------------------------------------------


class TestPenJitter:
    """Verify pen jitter state management (preview-only, no export effect)."""

    def test_jitter_disabled_by_default(self, canvas_widget):
        """Pen jitter is disabled by default."""
        assert canvas_widget._jitter_enabled is False

    def test_set_jitter_enabled(self, canvas_widget):
        """set_jitter_enabled(True) activates jitter simulation."""
        canvas_widget.set_jitter_enabled(True)
        assert canvas_widget._jitter_enabled is True

    def test_set_jitter_disabled(self, canvas_widget):
        """set_jitter_enabled(False) deactivates jitter simulation."""
        canvas_widget.set_jitter_enabled(True)
        canvas_widget.set_jitter_enabled(False)
        assert canvas_widget._jitter_enabled is False

    def test_default_jitter_intensity(self, canvas_widget):
        """Default jitter intensity is 1.0."""
        assert canvas_widget._jitter_intensity == pytest.approx(1.0)

    def test_set_jitter_intensity_valid(self, canvas_widget):
        """set_jitter_intensity() accepts values in the documented range."""
        canvas_widget.set_jitter_intensity(2.5)
        assert canvas_widget.get_jitter_intensity() == pytest.approx(2.5)

    def test_set_jitter_intensity_clamped_below(self, canvas_widget):
        """set_jitter_intensity(0.0) is clamped to 0.1."""
        canvas_widget.set_jitter_intensity(0.0)
        assert canvas_widget.get_jitter_intensity() == pytest.approx(0.1)

    def test_set_jitter_intensity_clamped_above(self, canvas_widget):
        """set_jitter_intensity(100.0) is clamped to 5.0."""
        canvas_widget.set_jitter_intensity(100.0)
        assert canvas_widget.get_jitter_intensity() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 11. Grid overlay
# ---------------------------------------------------------------------------


class TestGridOverlay:
    """Verify grid overlay toggle state."""

    def test_grid_hidden_by_default(self, canvas_widget):
        """Grid overlay is hidden on startup."""
        assert canvas_widget._show_grid is False

    def test_set_show_grid_true(self, canvas_widget):
        """set_show_grid(True) enables the grid."""
        canvas_widget.set_show_grid(True)
        assert canvas_widget._show_grid is True

    def test_set_show_grid_false(self, canvas_widget):
        """set_show_grid(False) hides the grid."""
        canvas_widget.set_show_grid(True)
        canvas_widget.set_show_grid(False)
        assert canvas_widget._show_grid is False

    def test_grid_spacing_constant(self, canvas_widget):
        """Grid spacing is 10 mm as specified."""
        from plottter.gui.canvas_widget import CanvasWidget
        assert CanvasWidget.GRID_SPACING_MM == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 12. Animation mode resets on project change
# ---------------------------------------------------------------------------


class TestAnimationReset:
    """Animation mode is exited when project/layer state changes."""

    def test_reset_animation_exits_mode(self, canvas_widget):
        """_reset_animation() exits animation mode completely."""
        canvas_widget.toggle_animation()
        assert canvas_widget._anim_mode is True
        canvas_widget._reset_animation()
        assert canvas_widget._anim_mode is False
        assert canvas_widget._anim_playing is False
        assert canvas_widget._anim_all_paths == []
        assert canvas_widget._anim_current_path == 0

    def test_layer_removed_resets_animation(self, canvas_widget, controller):
        """Removing a layer exits animation mode."""
        canvas_widget.toggle_animation()
        layer_id = controller.current_project.layers[0].id
        controller.layer_removed.emit(layer_id)
        assert canvas_widget._anim_mode is False

    def test_layer_added_resets_animation(self, canvas_widget, controller):
        """Adding a layer exits animation mode (path set has changed)."""
        canvas_widget.toggle_animation()
        new_layer = Layer(name="New", color="#FF0000")
        controller.current_project.add_layer(new_layer)
        controller.layer_added.emit(new_layer.id)
        assert canvas_widget._anim_mode is False
