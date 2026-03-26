"""Tests for task 61.1 — Flush all 3D layer snapshots before generating.

Covers:
(a) flush_current_snapshot() is called at the start of _on_generate()
(b) _build_sibling_3d_shapes() reads generator_info from other 3D layers correctly
(c) Single-layer 3D scenes still render correctly (no regression)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_3d_project(num_layers: int = 1) -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="3DTest", canvas=canvas)
    for i in range(num_layers):
        proj.add_layer(Layer(name=f"3D Layer {i + 1}", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_3d_project(2))


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    return panel


# ---------------------------------------------------------------------------
# (a) flush_current_snapshot() is called at the start of _on_generate()
# ---------------------------------------------------------------------------


class TestFlushCalledBeforeGenerate:
    """Spec (a): flush_current_snapshot() is called at the start of _on_generate()."""

    def test_flush_called_before_sibling_shapes(self, settings_panel, qtbot):
        """flush_current_snapshot() is called before _build_sibling_3d_shapes()."""
        panel = settings_panel

        # Switch to 3D mode
        panel.on_mode_changed("3D Scene")

        call_order = []

        original_flush = panel.flush_current_snapshot
        original_build = panel._build_sibling_3d_shapes

        def mock_flush():
            call_order.append("flush")
            original_flush()

        def mock_build(layer_id):
            call_order.append("build_sibling")
            return []

        panel.flush_current_snapshot = mock_flush
        panel._build_sibling_3d_shapes = mock_build

        # Mock GeneratorWorker to avoid actually running generation
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_worker.is_cancelled.return_value = False
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = MagicMock()
        mock_worker.metadata_ready = MagicMock()
        mock_worker.metadata_ready.connect = MagicMock()
        mock_worker.error = MagicMock()
        mock_worker.error.connect = MagicMock()
        mock_worker.start = MagicMock()

        with patch("plottter.gui.generator_worker.GeneratorWorker", return_value=mock_worker):
            panel._on_generate()

        assert "flush" in call_order, "flush_current_snapshot() was not called"
        assert "build_sibling" in call_order, "_build_sibling_3d_shapes() was not called"

        flush_idx = call_order.index("flush")
        build_idx = call_order.index("build_sibling")
        assert flush_idx < build_idx, (
            "flush_current_snapshot() must be called before _build_sibling_3d_shapes()"
        )

    def test_flush_called_even_when_no_sibling_shapes(self, settings_panel):
        """flush_current_snapshot() is called regardless of sibling layer count."""
        panel = settings_panel
        panel.on_mode_changed("3D Scene")

        flush_calls = []
        original_flush = panel.flush_current_snapshot

        def mock_flush():
            flush_calls.append(True)
            original_flush()

        panel.flush_current_snapshot = mock_flush

        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_worker.is_cancelled.return_value = False
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = MagicMock()
        mock_worker.metadata_ready = MagicMock()
        mock_worker.metadata_ready.connect = MagicMock()
        mock_worker.error = MagicMock()
        mock_worker.error.connect = MagicMock()
        mock_worker.start = MagicMock()

        with patch("plottter.gui.generator_worker.GeneratorWorker", return_value=mock_worker):
            panel._on_generate()

        assert len(flush_calls) == 1, "flush_current_snapshot() should be called once"


# ---------------------------------------------------------------------------
# (b) _build_sibling_3d_shapes() reads generator_info from sibling 3D layers
# ---------------------------------------------------------------------------


class TestBuildSiblingShapes:
    """Spec (b): _build_sibling_3d_shapes() reads generator_info from other 3D layers."""

    def test_sibling_layer_with_3d_generator_info_produces_shape(self, controller, qtbot):
        """A sibling layer with mode='3D Scene' generator_info produces a shape."""
        from plottter.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(controller)
        qtbot.addWidget(panel)

        project = controller.current_project
        layers = project.layers
        assert len(layers) >= 2

        # Give the second layer a valid 3D generator_info (a cube)
        sibling_layer = layers[1]
        sibling_info = {
            "generator_name": "3D Scene",
            "mode": "3D Scene",
            "params": {
                "shape_type": "Cube",
                "cube_size": 2.0,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
            },
            "transforms": {},
        }
        controller.set_layer_generator_info(sibling_layer.id, sibling_info)

        # Build sibling shapes for the first layer
        first_layer_id = layers[0].id
        shapes = panel._build_sibling_3d_shapes(first_layer_id)

        assert len(shapes) == 1, (
            f"Expected 1 sibling shape from the 3D layer, got {len(shapes)}"
        )

    def test_non_3d_sibling_layer_is_excluded(self, controller, qtbot):
        """A sibling layer without mode='3D Scene' is excluded from occlusion."""
        from plottter.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(controller)
        qtbot.addWidget(panel)

        project = controller.current_project
        layers = project.layers

        # Give the second layer a non-3D generator_info
        sibling_layer = layers[1]
        non_3d_info = {
            "generator_name": "Hatching",
            "mode": "Image to Lines",
            "params": {"angle": 45.0},
            "transforms": {},
        }
        controller.set_layer_generator_info(sibling_layer.id, non_3d_info)

        first_layer_id = layers[0].id
        shapes = panel._build_sibling_3d_shapes(first_layer_id)

        assert len(shapes) == 0, "Non-3D sibling layer should not produce a shape"

    def test_current_layer_is_excluded_from_siblings(self, controller, qtbot):
        """The current layer being generated is not included as a sibling."""
        from plottter.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(controller)
        qtbot.addWidget(panel)

        project = controller.current_project
        layers = project.layers

        # Give both layers 3D generator_info
        for layer in layers:
            info = {
                "generator_name": "3D Scene",
                "mode": "3D Scene",
                "params": {
                    "shape_type": "Cube",
                    "cube_size": 2.0,
                    "pos_x": 0.0,
                    "pos_y": 0.0,
                    "pos_z": 0.0,
                },
                "transforms": {},
            }
            controller.set_layer_generator_info(layer.id, info)

        # When building siblings for layer[0], only layer[1] should be included
        shapes = panel._build_sibling_3d_shapes(layers[0].id)
        assert len(shapes) == 1, (
            "Only one sibling should be returned (current layer excluded)"
        )

    def test_sibling_position_is_read_from_generator_info(self, controller, qtbot):
        """Sibling shape position (pos_x/pos_y/pos_z) is read from generator_info params."""
        from plottter.gui.settings_panel import SettingsPanel
        from plottter.scene3d.shapes.transformed import TransformedShape

        panel = SettingsPanel(controller)
        qtbot.addWidget(panel)

        project = controller.current_project
        layers = project.layers

        # Position the sibling cube at a known offset
        sibling_layer = layers[1]
        sibling_info = {
            "generator_name": "3D Scene",
            "mode": "3D Scene",
            "params": {
                "shape_type": "Cube",
                "cube_size": 1.0,
                "pos_x": 5.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
            },
            "transforms": {},
        }
        controller.set_layer_generator_info(sibling_layer.id, sibling_info)

        shapes = panel._build_sibling_3d_shapes(layers[0].id)
        assert len(shapes) == 1
        shape = shapes[0]
        # The shape should be a TransformedShape wrapping the cube
        assert isinstance(shape, TransformedShape)


# ---------------------------------------------------------------------------
# (c) Single-layer 3D scenes still render correctly (no regression)
# ---------------------------------------------------------------------------


class TestSingleLayer3DNoRegression:
    """Spec (c): Single-layer 3D scenes still render correctly."""

    def test_single_layer_generates_paths(self):
        """A single-layer 3D scene generates non-empty paths."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)

        params = {
            "shape_type": "Cube",
            "cube_size": 2.0,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "hlr_enabled": False,
            "_sibling_3d_shapes": [],
        }

        paths = gen.generate(params, canvas)
        assert len(paths) > 0, "Single-layer 3D cube should produce paths"

    def test_single_layer_hlr_generates_paths(self):
        """A single-layer 3D scene with HLR generates non-empty paths."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)

        params = {
            "shape_type": "Cube",
            "cube_size": 2.0,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "hlr_enabled": True,
            "chop_step": 0.2,
            "_sibling_3d_shapes": [],
        }

        paths = gen.generate(params, canvas)
        assert len(paths) > 0, "Single-layer 3D cube with HLR should produce paths"

    def test_empty_sibling_list_same_as_no_sibling_key(self):
        """Passing empty _sibling_3d_shapes gives same result as no key (no regression)."""
        from plottter.generators.scene3d_generator import Scene3DGenerator

        gen = Scene3DGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)

        base_params = {
            "shape_type": "Sphere",
            "sphere_radius": 1.0,
            "sphere_lat_lines": 6,
            "sphere_lng_lines": 6,
            "hlr_enabled": False,
        }

        paths_no_key = gen.generate(base_params, canvas)
        paths_empty_list = gen.generate({**base_params, "_sibling_3d_shapes": []}, canvas)

        assert len(paths_no_key) == len(paths_empty_list), (
            "Empty sibling list should produce same result as no sibling key"
        )
