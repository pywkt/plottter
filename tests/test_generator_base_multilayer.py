"""Tests for multi-layer generator support (task 118.1).

Covers:
- LayerSpec dataclass attributes
- Generator.emits_multiple_layers class attribute default
- Generator.generate_layers() raises NotImplementedError by default
- A concrete multi-layer generator routes through the multi-layer code path
- _on_multilayer_generation_finished calls ProjectController.add_layer per spec
- GeneratorWorker emits layers_finished (not finished) for multi-layer generators
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from plottter.generators.base import Generator, LayerSpec
from plottter.models.canvas import Canvas
from plottter.models.layer import Layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


class _DummyMultiLayerGenerator(Generator):
    """Minimal multi-layer generator for testing."""

    name = "DummyMultiLayer"
    category = "Test"
    emits_multiple_layers = True

    def get_parameters(self):
        return []

    def get_presets(self):
        return []

    def generate(self, params, canvas, progress_callback=None, cancelled_callback=None):
        # Should not be called when emits_multiple_layers is True
        raise AssertionError("generate() should not be called for multi-layer generators")

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        return [
            LayerSpec(name="Red Lines", color="#ff0000", paths=[[(0, 0), (10, 10)]]),
            LayerSpec(name="Blue Lines", color="#0000ff", paths=[[(5, 5), (15, 15)]]),
        ]


class _DummySingleLayerGenerator(Generator):
    """Minimal single-layer generator to verify backward compatibility."""

    name = "DummySingleLayer"
    category = "Test"

    def get_parameters(self):
        return []

    def get_presets(self):
        return []

    def generate(self, params, canvas, progress_callback=None, cancelled_callback=None):
        return [[(0, 0), (1, 1)]]


# ---------------------------------------------------------------------------
# LayerSpec
# ---------------------------------------------------------------------------

class TestLayerSpec:
    def test_fields(self):
        spec = LayerSpec(name="Test", color="#123456", paths=[[(0, 0), (1, 1)]])
        assert spec.name == "Test"
        assert spec.color == "#123456"
        assert spec.paths == [[(0, 0), (1, 1)]]

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(LayerSpec)


# ---------------------------------------------------------------------------
# Generator ABC changes
# ---------------------------------------------------------------------------

class TestGeneratorABCExtensions:
    def test_emits_multiple_layers_default_false(self):
        assert _DummySingleLayerGenerator.emits_multiple_layers is False

    def test_emits_multiple_layers_true_on_multilayer(self):
        assert _DummyMultiLayerGenerator.emits_multiple_layers is True

    def test_generate_layers_raises_not_implemented_by_default(self):
        gen = _DummySingleLayerGenerator()
        with pytest.raises(NotImplementedError):
            gen.generate_layers({}, make_canvas())

    def test_generate_layers_returns_layer_specs(self):
        gen = _DummyMultiLayerGenerator()
        specs = gen.generate_layers({}, make_canvas())
        assert len(specs) == 2
        assert all(isinstance(s, LayerSpec) for s in specs)
        assert specs[0].name == "Red Lines"
        assert specs[1].color == "#0000ff"


# ---------------------------------------------------------------------------
# _on_multilayer_generation_finished
# ---------------------------------------------------------------------------

class TestMultilayerGenerationFinished:
    """Verify the settings panel handler calls add_layer once per LayerSpec."""

    def _make_mixin_instance(self):
        """Build a minimal object with _on_multilayer_generation_finished."""
        from plottter.gui.settings_panel._generate import _GenerateMixin

        mock_controller = MagicMock()
        # Capture add_layer calls so we can inspect created Layer objects
        added_layers: list[Layer] = []

        def fake_add_layer(layer):
            added_layers.append(layer)
            return layer

        mock_controller.add_layer.side_effect = fake_add_layer

        # Create a bare instance without calling __init__ (which needs Qt widgets).
        # The handler reads several panel-state attributes that the real __init__
        # would have set up; stub them here so the test stays focused on the
        # add_layer behaviour it actually asserts on:
        #   _worker — checked + waited on before the macro runs (b8a80ab added
        #     this to fix a regen race; None means the wait() branch is skipped).
        #   _generator — looked up via getattr(self._generator, "name", "") when
        #     tagging layers with their source generator's name.
        # `_pending_multilayer_*` attributes are accessed via getattr-with-default
        # in the handler, so they don't need stubbing here.
        obj = object.__new__(_GenerateMixin)
        obj._controller = mock_controller
        obj._worker = None
        obj._generator = None
        return obj, mock_controller, added_layers

    def test_calls_add_layer_for_each_spec(self):
        obj, mock_controller, added_layers = self._make_mixin_instance()

        specs = [
            LayerSpec(name="Layer A", color="#ff0000", paths=[[(0, 0), (1, 1)]]),
            LayerSpec(name="Layer B", color="#00ff00", paths=[[(2, 2), (3, 3)]]),
            LayerSpec(name="Layer C", color="#0000ff", paths=[]),
        ]

        obj._on_multilayer_generation_finished(specs)

        assert mock_controller.add_layer.call_count == 3
        assert added_layers[0].name == "Layer A"
        assert added_layers[0].color == "#ff0000"
        assert added_layers[0].paths == [[(0, 0), (1, 1)]]
        assert added_layers[1].name == "Layer B"
        assert added_layers[2].name == "Layer C"

    def test_empty_specs_no_add_layer_calls(self):
        obj, mock_controller, _ = self._make_mixin_instance()
        obj._on_multilayer_generation_finished([])
        mock_controller.add_layer.assert_not_called()


# ---------------------------------------------------------------------------
# GeneratorWorker multi-layer routing (requires Qt)
# ---------------------------------------------------------------------------

class TestGeneratorWorkerMultiLayer:
    def test_worker_emits_layers_finished_for_multilayer_generator(self, qapp):
        from plottter.gui.generator_worker import GeneratorWorker

        gen = _DummyMultiLayerGenerator()
        canvas = make_canvas()
        worker = GeneratorWorker(gen, {}, canvas)

        received_specs: list = []
        finished_called = []

        worker.layers_finished.connect(lambda specs: received_specs.extend(specs))
        worker.finished.connect(lambda _: finished_called.append(True))

        worker.run()  # run synchronously (not via start())

        assert len(received_specs) == 2
        assert isinstance(received_specs[0], LayerSpec)
        assert not finished_called, "finished should NOT be emitted for multi-layer generators"

    def test_worker_emits_finished_for_single_layer_generator(self, qapp):
        from plottter.gui.generator_worker import GeneratorWorker

        gen = _DummySingleLayerGenerator()
        canvas = make_canvas()
        worker = GeneratorWorker(gen, {}, canvas)

        received_paths: list = []
        layers_finished_called = []

        worker.finished.connect(lambda paths: received_paths.extend(paths))
        worker.layers_finished.connect(lambda _: layers_finished_called.append(True))

        worker.run()

        assert len(received_paths) == 1
        assert not layers_finished_called, "layers_finished should NOT be emitted for single-layer generators"
