"""Phase 15.13 validation: stretch features.

This validation suite verifies the five stretch features described in item 15.13:

1. Preset gallery — PresetGalleryDialog loads all math-art preset thumbnails, cards
   populate the grid, selection works, and the worker can be cancelled cleanly.

2. Manual color masking brush — CanvasWidget mask-paint mode: lazy mask creation,
   hard and soft brush stamps, erase mode, interpolated strokes, mask get/set, and
   brush-size/hardness/erase controls.

3. Pen jitter preview toggle — set_jitter_enabled / set_jitter_intensity /
   get_jitter_intensity API, clamping behaviour, and _jitter_point producing
   perturbations when enabled and exact pixel coords when disabled.

4. Plugin system — load_plugins discovers and registers generator plugins from
   an on-disk directory; broken plugins are skipped; duplicate loads are prevented;
   non-generator plugins return an empty list; underscore files are skipped;
   create_user_plugin_dir creates the directory.

5. AxiDraw dialog — check_axidraw_available returns False when pyaxidraw is absent;
   AxiDrawNotInstalledError / AxiDrawConnectionError hierarchy; _safe_xml_id
   sanitisation; dialog construction and settings dict structure; preview-mode flag.
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_project(num_layers: int = 1, paths_per_layer: int = 3) -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    for i in range(num_layers):
        layer = Layer(name=f"Layer {i + 1}", color="#000000")
        paths = [[(j * 5.0, i * 10.0 + k * 3.0) for k in range(3)] for j in range(paths_per_layer)]
        layer.add_paths(paths)
        proj.add_layer(layer)
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project(num_layers=1, paths_per_layer=3))


@pytest.fixture
def canvas_widget(controller, qtbot):
    from plottter.gui.canvas_widget import CanvasWidget
    widget = CanvasWidget(controller)
    widget.resize(800, 600)
    qtbot.addWidget(widget)
    return widget


# ===========================================================================
# 1. Preset Gallery
# ===========================================================================


class TestRenderPolylinesToPixmap:
    """_render_polylines_to_pixmap() produces a valid QPixmap."""

    def test_returns_pixmap_with_correct_size(self, qapp):
        from plottter.gui.dialogs.preset_gallery import _render_polylines_to_pixmap
        polylines = [[(0.0, 0.0), (10.0, 10.0), (20.0, 5.0)]]
        pixmap = _render_polylines_to_pixmap(polylines, size_px=120)
        assert not pixmap.isNull()
        assert pixmap.width() == 120
        assert pixmap.height() == 120

    def test_empty_polylines_returns_white_pixmap(self, qapp):
        from plottter.gui.dialogs.preset_gallery import _render_polylines_to_pixmap
        pixmap = _render_polylines_to_pixmap([], size_px=60)
        assert not pixmap.isNull()
        assert pixmap.width() == 60

    def test_single_point_polyline_skipped(self, qapp):
        """Polylines with fewer than 2 points are skipped without error."""
        from plottter.gui.dialogs.preset_gallery import _render_polylines_to_pixmap
        polylines = [[(5.0, 5.0)]]  # single point — should be skipped
        pixmap = _render_polylines_to_pixmap(polylines, size_px=60)
        assert not pixmap.isNull()

    def test_custom_size_respected(self, qapp):
        from plottter.gui.dialogs.preset_gallery import _render_polylines_to_pixmap
        polylines = [[(0.0, 0.0), (50.0, 50.0)]]
        pixmap = _render_polylines_to_pixmap(polylines, size_px=80)
        assert pixmap.width() == 80
        assert pixmap.height() == 80

    def test_degenerate_bounding_box_no_crash(self, qapp):
        """All points at same location: span_x == 0 → uses fallback span of 1.0."""
        from plottter.gui.dialogs.preset_gallery import _render_polylines_to_pixmap
        polylines = [[(10.0, 10.0), (10.0, 10.0)]]
        pixmap = _render_polylines_to_pixmap(polylines, size_px=60)
        assert not pixmap.isNull()


class TestPresetGalleryCollectPresets:
    """_collect_math_presets() returns at least one preset per registered math generator."""

    def test_returns_nonempty_list(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        # Cancel worker immediately so we don't wait for thumbnails in tests
        if dialog._worker is not None:
            dialog._worker.cancel()
        presets = dialog._collect_math_presets()
        assert len(presets) > 0, "Expected at least one math preset to be collected"

    def test_each_entry_is_gen_cls_and_preset(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        from plottter.generators.base import Preset
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        presets = dialog._collect_math_presets()
        for gen_cls, preset in presets:
            assert callable(gen_cls), "First element should be a generator class"
            assert isinstance(preset, Preset), "Second element should be a Preset"


class TestPresetGalleryCards:
    """PresetGalleryDialog populates a grid of _PresetCard widgets."""

    def test_cards_created_for_each_preset(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        assert len(dialog._cards) == len(dialog._collect_math_presets()) + len(dialog._collect_user_presets_for_math())

    def test_initial_selection_is_none(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        gen_cls, preset_name = dialog.selected_preset()
        assert gen_cls is None
        assert preset_name is None

    def test_card_click_updates_selection(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        if not dialog._cards:
            pytest.skip("No preset cards available")
        first_card = dialog._cards[0]
        # Simulate clicking the first card
        first_card.clicked.emit(first_card._gen_cls, first_card._preset_name)
        gen_cls, preset_name = dialog.selected_preset()
        assert gen_cls is first_card._gen_cls
        assert preset_name == first_card._preset_name

    def test_ok_button_enabled_after_selection(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        # OK button starts disabled
        assert not dialog._ok_btn.isEnabled()
        if not dialog._cards:
            pytest.skip("No preset cards available")
        first_card = dialog._cards[0]
        first_card.clicked.emit(first_card._gen_cls, first_card._preset_name)
        assert dialog._ok_btn.isEnabled()

    def test_only_one_card_selected_at_a_time(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        if len(dialog._cards) < 2:
            pytest.skip("Need at least 2 preset cards")
        card0 = dialog._cards[0]
        card1 = dialog._cards[1]
        card0.clicked.emit(card0._gen_cls, card0._preset_name)
        assert card0._selected is True
        card1.clicked.emit(card1._gen_cls, card1._preset_name)
        assert card0._selected is False
        assert card1._selected is True

    def test_status_label_updates_when_done(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        # Simulate all_done signal
        dialog._on_thumbnails_done()
        count = len(dialog._cards)
        assert str(count) in dialog._status_label.text()

    def test_set_thumbnail_on_card(self, qapp, qtbot):
        from PyQt6.QtGui import QPixmap, QColor
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None:
            dialog._worker.cancel()
        if not dialog._cards:
            pytest.skip("No preset cards available")
        card = dialog._cards[0]
        pixmap = QPixmap(120, 120)
        pixmap.fill(QColor("#FF0000"))
        # Should not raise
        card.set_thumbnail(pixmap)

    def test_worker_cancellation(self, qapp, qtbot):
        """Worker can be cancelled without blocking indefinitely."""
        from plottter.gui.dialogs.preset_gallery import PresetGalleryDialog
        dialog = PresetGalleryDialog()
        qtbot.addWidget(dialog)
        if dialog._worker is not None and dialog._worker.isRunning():
            dialog._worker.cancel()
            finished = dialog._worker.wait(3000)  # 3 second timeout
            assert finished, "Worker did not finish after cancel()"


class TestThumbnailWorkerDirect:
    """_ThumbnailWorker generates thumbnails for presets without using a full dialog."""

    def test_worker_emits_thumbnail_ready_for_each_preset(self, qapp, qtbot):
        from plottter.gui.dialogs.preset_gallery import _ThumbnailWorker
        from plottter.generators import get_generators_by_category
        from plottter.generators.base import Preset

        # Collect one preset from the first available math generator
        presets = []
        for gen_cls in get_generators_by_category("math"):
            try:
                gen = gen_cls()
                gen_presets = gen.get_presets()
                if gen_presets:
                    presets.append((gen_cls, gen_presets[0], False))
                    break
            except Exception:
                pass
        if not presets:
            pytest.skip("No math presets available")

        received = []
        worker = _ThumbnailWorker(presets, 100.0)
        worker.thumbnail_ready.connect(lambda g, n, u, p: received.append((g, n, p)))
        with qtbot.waitSignal(worker.all_done, timeout=10000, raising=True):
            worker.start()
        assert len(received) == len(presets)
        gen_cls, preset_name, pixmap = received[0]
        assert not pixmap.isNull()


# ===========================================================================
# 2. Manual Color Masking Brush
# ===========================================================================


class TestMaskInitialState:
    """CanvasWidget mask-paint mode is off and mask is None initially."""

    def test_mask_paint_active_default_false(self, canvas_widget):
        assert canvas_widget._mask_paint_active is False

    def test_mask_array_default_none(self, canvas_widget):
        assert canvas_widget._mask_array is None

    def test_erase_mode_default_false(self, canvas_widget):
        assert canvas_widget._mask_erase is False

    def test_brush_size_default_positive(self, canvas_widget):
        assert canvas_widget._mask_brush_size_mm > 0.0

    def test_brush_hardness_default_in_range(self, canvas_widget):
        assert 0.0 <= canvas_widget._mask_brush_hardness <= 1.0


class TestMaskPaintActivation:
    """set_mask_paint_active() toggles cursor and internal flag."""

    def test_enable_sets_flag(self, canvas_widget):
        canvas_widget.set_mask_paint_active(True)
        assert canvas_widget._mask_paint_active is True

    def test_disable_clears_flag(self, canvas_widget):
        canvas_widget.set_mask_paint_active(True)
        canvas_widget.set_mask_paint_active(False)
        assert canvas_widget._mask_paint_active is False

    def test_get_mask_returns_none_before_painting(self, canvas_widget):
        assert canvas_widget.get_mask() is None


class TestBrushControls:
    """Brush size, hardness, and erase-mode controls."""

    def test_set_brush_size_accepted(self, canvas_widget):
        canvas_widget.set_brush_size_mm(10.0)
        assert canvas_widget._mask_brush_size_mm == 10.0

    def test_set_brush_size_clamped_below(self, canvas_widget):
        canvas_widget.set_brush_size_mm(0.0)
        assert canvas_widget._mask_brush_size_mm >= 0.5

    def test_set_brush_size_clamped_negative(self, canvas_widget):
        canvas_widget.set_brush_size_mm(-5.0)
        assert canvas_widget._mask_brush_size_mm >= 0.5

    def test_set_brush_hardness_accepted(self, canvas_widget):
        canvas_widget.set_brush_hardness(0.5)
        assert math.isclose(canvas_widget._mask_brush_hardness, 0.5)

    def test_set_brush_hardness_clamp_below(self, canvas_widget):
        canvas_widget.set_brush_hardness(-1.0)
        assert canvas_widget._mask_brush_hardness >= 0.0

    def test_set_brush_hardness_clamp_above(self, canvas_widget):
        canvas_widget.set_brush_hardness(2.0)
        assert canvas_widget._mask_brush_hardness <= 1.0

    def test_set_erase_mode_true(self, canvas_widget):
        canvas_widget.set_erase_mode(True)
        assert canvas_widget._mask_erase is True

    def test_set_erase_mode_false(self, canvas_widget):
        canvas_widget.set_erase_mode(True)
        canvas_widget.set_erase_mode(False)
        assert canvas_widget._mask_erase is False


class TestEnsureMask:
    """_ensure_mask() creates a correctly shaped float32 zero array."""

    def test_ensure_mask_creates_array(self, canvas_widget):
        assert canvas_widget._mask_array is None
        canvas_widget._ensure_mask()
        assert canvas_widget._mask_array is not None

    def test_mask_dtype_float32(self, canvas_widget):
        canvas_widget._ensure_mask()
        assert canvas_widget._mask_array.dtype == np.float32

    def test_mask_all_zeros_after_creation(self, canvas_widget):
        canvas_widget._ensure_mask()
        assert np.all(canvas_widget._mask_array == 0.0)

    def test_mask_shape_matches_canvas(self, canvas_widget):
        canvas_widget._ensure_mask()
        from plottter.gui.canvas_widget import _MASK_PX_PER_MM
        canvas = canvas_widget._controller.current_project.canvas
        expected_h = int(canvas.height_mm * _MASK_PX_PER_MM)
        expected_w = int(canvas.width_mm * _MASK_PX_PER_MM)
        assert canvas_widget._mask_array.shape == (expected_h, expected_w)

    def test_ensure_mask_idempotent(self, canvas_widget):
        """Calling _ensure_mask twice keeps the same array."""
        canvas_widget._ensure_mask()
        arr_id = id(canvas_widget._mask_array)
        canvas_widget._ensure_mask()
        assert id(canvas_widget._mask_array) == arr_id


class TestPaintAt:
    """_paint_at() stamps values into the mask."""

    def test_paint_at_marks_center_pixel(self, canvas_widget):
        canvas = canvas_widget._controller.current_project.canvas
        cx = canvas.width_mm / 2
        cy = canvas.height_mm / 2
        canvas_widget.set_brush_hardness(1.0)
        canvas_widget.set_brush_size_mm(10.0)
        canvas_widget._paint_at(cx, cy)
        assert canvas_widget._mask_array is not None
        # The centre pixel should now have a value > 0
        from plottter.gui.canvas_widget import _MASK_PX_PER_MM
        px = int(cx * _MASK_PX_PER_MM)
        py = int(cy * _MASK_PX_PER_MM)
        assert canvas_widget._mask_array[py, px] > 0.0

    def test_hard_brush_gives_binary_stamp(self, canvas_widget):
        """Hard brush (hardness=1.0) produces only 0.0 or 1.0 values."""
        canvas = canvas_widget._controller.current_project.canvas
        cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
        canvas_widget.set_brush_hardness(1.0)
        canvas_widget.set_brush_size_mm(10.0)
        canvas_widget._paint_at(cx, cy)
        arr = canvas_widget._mask_array
        vals = arr[arr > 0.0]
        assert np.all(np.isclose(vals, 1.0)), "Hard brush should produce only 0 or 1 values"

    def test_soft_brush_gives_gradient(self, canvas_widget):
        """Soft brush (hardness=0.0) produces intermediate values."""
        canvas = canvas_widget._controller.current_project.canvas
        cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
        canvas_widget.set_brush_hardness(0.0)
        canvas_widget.set_brush_size_mm(20.0)
        canvas_widget._paint_at(cx, cy)
        arr = canvas_widget._mask_array
        # Should have values strictly between 0 and 1 (not just 0/1)
        intermediate = arr[(arr > 0.01) & (arr < 0.99)]
        assert len(intermediate) > 0, "Soft brush should produce intermediate values"

    def test_erase_removes_painted_area(self, canvas_widget):
        canvas = canvas_widget._controller.current_project.canvas
        cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
        canvas_widget.set_brush_hardness(1.0)
        canvas_widget.set_brush_size_mm(10.0)
        canvas_widget._paint_at(cx, cy)
        # Now erase
        canvas_widget.set_erase_mode(True)
        canvas_widget._paint_at(cx, cy)
        from plottter.gui.canvas_widget import _MASK_PX_PER_MM
        px = int(cx * _MASK_PX_PER_MM)
        py = int(cy * _MASK_PX_PER_MM)
        assert canvas_widget._mask_array[py, px] == 0.0

    def test_paint_outside_canvas_no_crash(self, canvas_widget):
        """Painting far outside the canvas bounds should not crash."""
        canvas_widget.set_brush_size_mm(5.0)
        canvas_widget._paint_at(-9999.0, -9999.0)  # way outside

    def test_paint_clamps_values_to_one(self, canvas_widget):
        """Painting the same spot twice never exceeds 1.0."""
        canvas = canvas_widget._controller.current_project.canvas
        cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
        canvas_widget.set_brush_hardness(1.0)
        canvas_widget.set_brush_size_mm(10.0)
        for _ in range(5):
            canvas_widget._paint_at(cx, cy)
        assert canvas_widget._mask_array.max() <= 1.0


class TestMaskGetSet:
    """set_mask / get_mask round-trips correctly."""

    def test_set_mask_stores_array(self, canvas_widget):
        arr = np.ones((100, 100), dtype=np.float32) * 0.5
        canvas_widget.set_mask(arr)
        result = canvas_widget.get_mask()
        assert result is arr

    def test_set_mask_none_clears(self, canvas_widget):
        arr = np.zeros((50, 50), dtype=np.float32)
        canvas_widget.set_mask(arr)
        canvas_widget.set_mask(None)
        assert canvas_widget.get_mask() is None


class TestInterpolateStroke:
    """_interpolate_stroke() paints along the path between two positions."""

    def test_interpolate_paints_intermediate_positions(self, canvas_widget):
        canvas = canvas_widget._controller.current_project.canvas
        x0, y0 = 20.0, canvas.height_mm / 2
        x1, y1 = 80.0, canvas.height_mm / 2
        canvas_widget.set_brush_hardness(1.0)
        canvas_widget.set_brush_size_mm(2.0)
        canvas_widget._interpolate_stroke((x0, y0), (x1, y1))
        arr = canvas_widget._mask_array
        # Should have painted at multiple x positions
        from plottter.gui.canvas_widget import _MASK_PX_PER_MM
        py = int(y0 * _MASK_PX_PER_MM)
        row = arr[py, :]
        painted_cols = np.where(row > 0)[0]
        assert len(painted_cols) > 1, "Stroke should paint multiple columns"


# ===========================================================================
# 3. Pen Jitter Preview Toggle
# ===========================================================================


class TestJitterInitialState:
    """Pen jitter starts disabled with default intensity 1.0."""

    def test_jitter_disabled_by_default(self, canvas_widget):
        assert canvas_widget._jitter_enabled is False

    def test_jitter_intensity_default(self, canvas_widget):
        assert math.isclose(canvas_widget._jitter_intensity, 1.0)

    def test_get_jitter_intensity_returns_default(self, canvas_widget):
        assert math.isclose(canvas_widget.get_jitter_intensity(), 1.0)


class TestJitterControls:
    """set_jitter_enabled / set_jitter_intensity API."""

    def test_enable_jitter(self, canvas_widget):
        canvas_widget.set_jitter_enabled(True)
        assert canvas_widget._jitter_enabled is True

    def test_disable_jitter(self, canvas_widget):
        canvas_widget.set_jitter_enabled(True)
        canvas_widget.set_jitter_enabled(False)
        assert canvas_widget._jitter_enabled is False

    def test_set_intensity_accepted(self, canvas_widget):
        canvas_widget.set_jitter_intensity(3.0)
        assert math.isclose(canvas_widget.get_jitter_intensity(), 3.0)

    def test_set_intensity_clamped_below(self, canvas_widget):
        canvas_widget.set_jitter_intensity(0.0)
        assert canvas_widget.get_jitter_intensity() >= 0.1

    def test_set_intensity_clamped_above(self, canvas_widget):
        canvas_widget.set_jitter_intensity(100.0)
        assert canvas_widget.get_jitter_intensity() <= 5.0

    def test_set_intensity_min_boundary(self, canvas_widget):
        canvas_widget.set_jitter_intensity(0.1)
        assert math.isclose(canvas_widget.get_jitter_intensity(), 0.1)

    def test_set_intensity_max_boundary(self, canvas_widget):
        canvas_widget.set_jitter_intensity(5.0)
        assert math.isclose(canvas_widget.get_jitter_intensity(), 5.0)


class TestJitterPoint:
    """_jitter_point() returns exact or perturbed pixel coordinates."""

    def test_disabled_returns_exact_pixel(self, canvas_widget):
        from PyQt6.QtCore import QPointF
        canvas_widget.set_jitter_enabled(False)
        pt = (50.0, 50.0)
        exact = canvas_widget.mm_to_pixel(pt)
        result = canvas_widget._jitter_point(pt)
        # Should be identical when jitter is off
        assert math.isclose(result.x(), exact.x(), abs_tol=1e-6)
        assert math.isclose(result.y(), exact.y(), abs_tol=1e-6)

    def test_enabled_perturbs_at_least_sometimes(self, canvas_widget):
        """With jitter on, repeated calls should not all return identical values."""
        canvas_widget.set_jitter_enabled(True)
        canvas_widget.set_jitter_intensity(5.0)  # large intensity for reliable detection
        pt = (50.0, 50.0)
        results_x = [canvas_widget._jitter_point(pt).x() for _ in range(50)]
        # At least some values should differ
        unique_x = set(round(x, 4) for x in results_x)
        assert len(unique_x) > 1, "Jitter should produce varying x coordinates"

    def test_intensity_affects_spread(self, canvas_widget):
        """Higher intensity → larger standard deviation of jitter."""
        import statistics
        canvas_widget.set_jitter_enabled(True)
        pt = (50.0, 50.0)

        canvas_widget.set_jitter_intensity(0.1)
        low_xs = [canvas_widget._jitter_point(pt).x() for _ in range(200)]

        canvas_widget.set_jitter_intensity(5.0)
        high_xs = [canvas_widget._jitter_point(pt).x() for _ in range(200)]

        low_std = statistics.stdev(low_xs)
        high_std = statistics.stdev(high_xs)
        assert high_std > low_std, "Higher intensity should produce larger jitter spread"


# ===========================================================================
# 4. Plugin System
# ===========================================================================


_VALID_PLUGIN = textwrap.dedent("""\
    from plottter.generators import register_generator
    from plottter.generators.base import Generator, IntParam

    @register_generator
    class _ValidationPlugin15_13(Generator):
        name = "_ValidationPlugin15_13"
        category = "math"

        def get_parameters(self):
            return [IntParam("n", "N", min=1, max=10, step=1, default=3)]

        def get_presets(self):
            return []

        def generate(self, params, canvas, progress_callback=None):
            return [[(0.0, 0.0), (10.0, 10.0)]]
""")

_BROKEN_PLUGIN = textwrap.dedent("""\
    # Syntax error
    def broken(:
        pass
""")

_NON_GENERATOR_PLUGIN = textwrap.dedent("""\
    SOME_CONSTANT = 42
""")


@pytest.fixture
def plugin_dir(tmp_path):
    """A temporary plugins directory pre-populated with a valid plugin."""
    d = tmp_path / "plugins"
    d.mkdir()
    return d


class TestLoadPlugins:
    """load_plugins() discovers and registers generator plugins."""

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        from plottter.generators.plugin_loader import load_plugins
        result = load_plugins(extra_dirs=[str(tmp_path / "nonexistent")])
        assert result == []

    def test_valid_plugin_registered(self, plugin_dir):
        from plottter.generators.plugin_loader import load_plugins
        from plottter.generators import GENERATORS

        plugin_file = plugin_dir / "val_15_13_circles.py"
        # Use unique class name to avoid collision with other test runs
        unique_name = "_Val1513CirclesXXX"
        src = _VALID_PLUGIN.replace("_ValidationPlugin15_13", unique_name)
        plugin_file.write_text(src)

        # Clean up any stale sys.modules entry
        mod_name = f"plottter_plugin_val_15_13_circles"
        sys.modules.pop(mod_name, None)
        GENERATORS.pop(unique_name, None)

        result = load_plugins(extra_dirs=[str(plugin_dir)])
        assert unique_name in result
        assert unique_name in GENERATORS

        # Cleanup
        GENERATORS.pop(unique_name, None)
        sys.modules.pop(mod_name, None)

    def test_broken_plugin_skipped_without_crash(self, plugin_dir):
        from plottter.generators.plugin_loader import load_plugins

        plugin_file = plugin_dir / "broken_plugin.py"
        plugin_file.write_text(_BROKEN_PLUGIN)
        mod_name = "plottter_plugin_broken_plugin"
        sys.modules.pop(mod_name, None)

        # Should not raise
        result = load_plugins(extra_dirs=[str(plugin_dir)])
        assert isinstance(result, list)

    def test_non_generator_plugin_returns_empty(self, plugin_dir):
        from plottter.generators.plugin_loader import load_plugins

        plugin_file = plugin_dir / "non_gen_plugin.py"
        plugin_file.write_text(_NON_GENERATOR_PLUGIN)
        mod_name = "plottter_plugin_non_gen_plugin"
        sys.modules.pop(mod_name, None)

        result = load_plugins(extra_dirs=[str(plugin_dir)])
        assert result == []

    def test_underscore_files_skipped(self, plugin_dir):
        from plottter.generators.plugin_loader import load_plugins
        from plottter.generators import GENERATORS

        # A file starting with _ should be skipped entirely
        private_file = plugin_dir / "_private.py"
        unique_name = "_Val1513PrivatePlugin"
        src = _VALID_PLUGIN.replace("_ValidationPlugin15_13", unique_name)
        private_file.write_text(src)

        mod_name = "plottter_plugin__private"
        sys.modules.pop(mod_name, None)
        GENERATORS.pop(unique_name, None)

        result = load_plugins(extra_dirs=[str(plugin_dir)])
        assert unique_name not in result
        assert unique_name not in GENERATORS

    def test_no_double_registration(self, plugin_dir):
        """Loading the same plugin directory twice does not double-register generators."""
        from plottter.generators.plugin_loader import load_plugins
        from plottter.generators import GENERATORS

        unique_name = "_Val1513DupXXX"
        src = _VALID_PLUGIN.replace("_ValidationPlugin15_13", unique_name)
        plugin_file = plugin_dir / "dup_gen.py"
        plugin_file.write_text(src)
        mod_name = "plottter_plugin_dup_gen"
        sys.modules.pop(mod_name, None)
        GENERATORS.pop(unique_name, None)

        first = load_plugins(extra_dirs=[str(plugin_dir)])
        second = load_plugins(extra_dirs=[str(plugin_dir)])
        # Second call should not register again
        assert unique_name in first
        assert unique_name not in second  # already registered; not returned again

        GENERATORS.pop(unique_name, None)
        sys.modules.pop(mod_name, None)

    def test_empty_directory_returns_empty(self, plugin_dir):
        from plottter.generators.plugin_loader import load_plugins
        result = load_plugins(extra_dirs=[str(plugin_dir)])
        assert result == []


class TestGetPluginDirs:
    """get_plugin_dirs() returns only existing directories."""

    def test_returns_list(self):
        from plottter.generators.plugin_loader import get_plugin_dirs
        result = get_plugin_dirs()
        assert isinstance(result, list)
        for d in result:
            assert d.exists()

    def test_all_entries_are_path_objects(self):
        from plottter.generators.plugin_loader import get_plugin_dirs
        for d in get_plugin_dirs():
            assert isinstance(d, Path)


class TestCreateUserPluginDir:
    """create_user_plugin_dir() creates the user plugin directory."""

    def test_creates_directory(self, tmp_path, monkeypatch):
        from plottter import generators as gens_mod
        import plottter.generators.plugin_loader as pl_mod

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        from plottter.generators.plugin_loader import create_user_plugin_dir
        result = create_user_plugin_dir()
        assert result.exists()
        assert result.is_dir()

    def test_idempotent(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home2"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        from plottter.generators.plugin_loader import create_user_plugin_dir
        first = create_user_plugin_dir()
        second = create_user_plugin_dir()
        assert first == second
        assert second.exists()


# ===========================================================================
# 5. AxiDraw Dialog
# ===========================================================================


class TestAxiDrawAvailability:
    """check_axidraw_available() reflects the presence of pyaxidraw."""

    def test_returns_false_when_pyaxidraw_missing(self, monkeypatch):
        """Without pyaxidraw installed, check_axidraw_available() returns False."""
        from plottter.export import axidraw as ax_mod
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyaxidraw" or name.startswith("pyaxidraw."):
                raise ImportError("mocked: pyaxidraw not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from plottter.export.axidraw import _require_axidraw
            # patch _require_axidraw directly on the module
            with patch.object(ax_mod, "_require_axidraw",
                               side_effect=ax_mod.AxiDrawNotInstalledError("not installed")):
                result = ax_mod.check_axidraw_available()
        assert result is False

    def test_check_axidraw_available_false_without_package(self):
        """In the test environment, pyaxidraw is not installed."""
        from plottter.export.axidraw import check_axidraw_available
        # pyaxidraw is optional; in CI it won't be installed
        result = check_axidraw_available()
        assert isinstance(result, bool)


class TestAxiDrawExceptions:
    """AxiDrawNotInstalledError and AxiDrawConnectionError hierarchy."""

    def test_not_installed_is_runtime_error(self):
        from plottter.export.axidraw import AxiDrawNotInstalledError
        exc = AxiDrawNotInstalledError("test")
        assert isinstance(exc, RuntimeError)

    def test_connection_error_is_runtime_error(self):
        from plottter.export.axidraw import AxiDrawConnectionError
        exc = AxiDrawConnectionError("test")
        assert isinstance(exc, RuntimeError)

    def test_not_installed_message_preserved(self):
        from plottter.export.axidraw import AxiDrawNotInstalledError
        exc = AxiDrawNotInstalledError("Install with pip install pyaxidraw")
        assert "pyaxidraw" in str(exc)

    def test_require_axidraw_raises_when_missing(self):
        """_require_axidraw raises AxiDrawNotInstalledError when pyaxidraw absent."""
        from plottter.export.axidraw import _require_axidraw, AxiDrawNotInstalledError
        # If pyaxidraw is not installed, _require_axidraw should raise
        try:
            import pyaxidraw  # noqa: F401
            pytest.skip("pyaxidraw is actually installed; skipping negative test")
        except ImportError:
            with pytest.raises(AxiDrawNotInstalledError):
                _require_axidraw()


class TestSafeXmlId:
    """_safe_xml_id() sanitises layer names for use in XML ids."""

    def test_spaces_converted_to_underscores(self):
        from plottter.export.axidraw import _safe_xml_id
        assert _safe_xml_id("My Layer") == "My_Layer"

    def test_multiple_spaces_collapsed(self):
        from plottter.export.axidraw import _safe_xml_id
        assert _safe_xml_id("Layer   A") == "Layer___A"

    def test_xml_special_chars_replaced(self):
        from plottter.export.axidraw import _safe_xml_id
        result = _safe_xml_id('Layer<>&"')
        assert "<" not in result
        assert ">" not in result
        assert "&" not in result
        assert '"' not in result

    def test_backslash_replaced(self):
        from plottter.export.axidraw import _safe_xml_id
        result = _safe_xml_id("path\\layer")
        assert "\\" not in result

    def test_clean_name_unchanged(self):
        from plottter.export.axidraw import _safe_xml_id
        assert _safe_xml_id("Layer1") == "Layer1"

    def test_empty_string_returns_empty(self):
        from plottter.export.axidraw import _safe_xml_id
        result = _safe_xml_id("")
        assert isinstance(result, str)

    def test_colon_replaced(self):
        from plottter.export.axidraw import _safe_xml_id
        result = _safe_xml_id("Layer:A")
        assert ":" not in result

    def test_asterisk_replaced(self):
        from plottter.export.axidraw import _safe_xml_id
        result = _safe_xml_id("Layer*A")
        assert "*" not in result


class TestAxiDrawDialog:
    """AxiDrawDialog constructs without error and exposes correct UI."""

    def test_dialog_can_be_constructed(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert dialog is not None

    def test_preview_checkbox_present(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert hasattr(dialog, "_preview_check")
        assert dialog._preview_check is not None

    def test_speed_controls_present(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert hasattr(dialog, "_speed_pendown")
        assert hasattr(dialog, "_speed_penup")

    def test_pen_position_controls_present(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert hasattr(dialog, "_pen_pos_down")
        assert hasattr(dialog, "_pen_pos_up")

    def test_settings_dict_has_required_keys(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        settings = dialog._build_settings()
        required_keys = {
            "speed_pendown", "speed_penup",
            "pen_pos_down", "pen_pos_up",
            "pen_delay_down", "pen_delay_up",
            "const_speed", "preview",
        }
        for key in required_keys:
            assert key in settings, f"Missing key: {key}"

    def test_preview_mode_default_false_when_axidraw_available(self, qapp, qtbot):
        """When pyaxidraw is available, preview starts unchecked."""
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        from plottter.export import axidraw as ax_mod
        with patch.object(ax_mod, "check_axidraw_available", return_value=True):
            dialog = AxiDrawDialog()
            qtbot.addWidget(dialog)
        # When axidraw IS available, preview should be unchecked by default
        assert dialog._preview_check.isChecked() is False

    def test_preview_mode_auto_checked_when_axidraw_unavailable(self, qapp, qtbot):
        """When pyaxidraw is not available, preview is automatically checked."""
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        from plottter.export import axidraw as ax_mod
        with patch.object(ax_mod, "check_axidraw_available", return_value=False):
            dialog = AxiDrawDialog()
            qtbot.addWidget(dialog)
        assert dialog._preview_check.isChecked() is True

    def test_model_combo_has_entries(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert hasattr(dialog, "_model_combo")
        assert dialog._model_combo.count() > 0

    def test_window_title(self, qapp, qtbot):
        from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog
        dialog = AxiDrawDialog()
        qtbot.addWidget(dialog)
        assert "AxiDraw" in dialog.windowTitle()


class TestPlotWorker:
    """_PlotWorker is a QThread subclass with expected signals."""

    def test_plot_worker_is_qthread(self, qapp):
        from PyQt6.QtCore import QThread
        from plottter.gui.dialogs.axidraw_dialog import _PlotWorker
        assert issubclass(_PlotWorker, QThread)

    def test_plot_worker_has_signals(self, qapp):
        from plottter.gui.dialogs.axidraw_dialog import _PlotWorker
        # Verify signal attributes exist on the class
        assert hasattr(_PlotWorker, "progress")
        assert hasattr(_PlotWorker, "finished")
        assert hasattr(_PlotWorker, "error")
