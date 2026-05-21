"""Tests for PixelArtGenerator re-generation flow (task 119.2).

Verifies that running a multi-layer generator a second time *replaces* the
layers from the first run instead of appending to them.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.base import LayerSpec
from plottter.generators.pixel_art import PixelArtGenerator
from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project() -> tuple[Project, str]:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="RegenTest", canvas=canvas)
    layer = Layer(name="Layer 1")
    proj.add_layer(layer)
    return proj, layer.id


def _build_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    return panel


def _make_source_image(rows: int = 8, cols: int = 8) -> np.ndarray:
    """Return a tiny uint8 RGB image with two distinct luminance zones."""
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    img[: rows // 2, :] = 0      # black top half
    img[rows // 2 :, :] = 255    # white bottom half
    return img


def _run_pixel_art_sync(
    canvas: Canvas,
    grid_width: int = 4,
    fill_density: float = 0.7,
    cell_border: bool = False,
) -> list[LayerSpec]:
    """Run PixelArtGenerator synchronously and return its LayerSpec list."""
    gen = PixelArtGenerator()
    params = {
        "_source_image": _make_source_image(),
        "grid_width": grid_width,
        "palette": "grayscale_4",
        "dithering": "none",
        "cell_fill_style": "solid_hatch",
        "fill_density": fill_density,
        "cell_border": cell_border,
        "cell_gap_mm": 0.0,
    }
    return gen.generate_layers(params, canvas)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPixelArtRegeneration:
    """Second run of PixelArtGenerator must replace, not append, first-run layers."""

    def test_second_run_replaces_first_run_layers(self, qtbot):
        """After re-generating with different params, old layers are gone; only new remain.

        Uses fill_density=0.3 for run 1 and fill_density=0.9 for run 2 to ensure
        the params actually differ, matching the task requirement of "different params".
        """
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        canvas = proj.canvas

        # Run 1: sparse fill
        specs_1 = _run_pixel_art_sync(canvas, fill_density=0.3)
        assert specs_1, "PixelArtGenerator must produce at least one layer spec"

        panel._pending_multilayer_regen_run_id = None
        panel._on_multilayer_generation_finished(specs_1)

        first_run_layers = [
            layer
            for layer in ctrl.current_project.layers
            if isinstance(layer.generator_info, dict)
            and "_generator_run_id" in layer.generator_info
        ]
        assert first_run_layers, "First run must produce tagged layers"

        old_run_id = first_run_layers[0].generator_info["_generator_run_id"]
        assert all(
            l.generator_info["_generator_run_id"] == old_run_id
            for l in first_run_layers
        ), "All first-run layers must share the same _generator_run_id"

        # Run 2: dense fill — different params from run 1
        specs_2 = _run_pixel_art_sync(canvas, fill_density=0.9)
        panel._pending_multilayer_regen_run_id = old_run_id
        panel._on_multilayer_generation_finished(specs_2)

        final_tagged = [
            layer
            for layer in ctrl.current_project.layers
            if isinstance(layer.generator_info, dict)
            and "_generator_run_id" in layer.generator_info
        ]
        final_run_ids = {l.generator_info["_generator_run_id"] for l in final_tagged}

        assert old_run_id not in final_run_ids, (
            "Old run layers must be removed on re-generation"
        )
        assert len(final_run_ids) == 1, "All second-run layers must share one new run ID"
        assert next(iter(final_run_ids)) != old_run_id

    def test_layer_count_stable_after_regen(self, qtbot):
        """Total pixel-art layer count must not grow when re-generating."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        canvas = proj.canvas
        specs = _run_pixel_art_sync(canvas)
        assert specs

        # First run
        panel._pending_multilayer_regen_run_id = None
        panel._on_multilayer_generation_finished(specs)
        count_after_run1 = len(ctrl.current_project.layers)

        run1_run_id = next(
            l.generator_info["_generator_run_id"]
            for l in ctrl.current_project.layers
            if isinstance(l.generator_info, dict) and "_generator_run_id" in l.generator_info
        )

        # Second run — different fill_density param
        specs2 = _run_pixel_art_sync(canvas, fill_density=0.9)
        panel._pending_multilayer_regen_run_id = run1_run_id
        panel._on_multilayer_generation_finished(specs2)

        count_after_run2 = len(ctrl.current_project.layers)
        assert count_after_run2 == count_after_run1, (
            f"Re-generation must replace layers, not append: "
            f"expected {count_after_run1}, got {count_after_run2}"
        )

    def test_first_run_does_not_remove_unrelated_layers(self, qtbot):
        """A first-time generate (no prior run ID) must leave pre-existing layers intact."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        canvas = proj.canvas
        specs = _run_pixel_art_sync(canvas)
        assert specs

        # First run — no pending run ID
        panel._pending_multilayer_regen_run_id = None
        panel._on_multilayer_generation_finished(specs)

        plain_layers = [
            l for l in ctrl.current_project.layers if l.generator_info is None
        ]
        assert len(plain_layers) >= 1, (
            "Pre-existing plain layer must not be removed by a first-time generate"
        )

    def test_run_id_uniqueness_across_runs(self, qtbot):
        """Each generation run must produce a distinct _generator_run_id UUID."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController
        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        canvas = proj.canvas
        specs = _run_pixel_art_sync(canvas)
        assert specs

        run_ids: list[str] = []

        panel._pending_multilayer_regen_run_id = None
        panel._on_multilayer_generation_finished(specs)
        run_ids.append(
            next(
                l.generator_info["_generator_run_id"]
                for l in ctrl.current_project.layers
                if isinstance(l.generator_info, dict) and "_generator_run_id" in l.generator_info
            )
        )

        # Re-run with different params (cell_border enabled)
        specs_b = _run_pixel_art_sync(canvas, cell_border=True)
        panel._pending_multilayer_regen_run_id = run_ids[0]
        panel._on_multilayer_generation_finished(specs_b)
        run_ids.append(
            next(
                l.generator_info["_generator_run_id"]
                for l in ctrl.current_project.layers
                if isinstance(l.generator_info, dict) and "_generator_run_id" in l.generator_info
            )
        )

        assert run_ids[0] != run_ids[1], "Each generation run must get a fresh UUID"
