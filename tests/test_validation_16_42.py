"""Phase 16.42 validation: Invert Mask button in Mask Paint mode.

Verifies:
1. invert_mask() produces the complement of the current mask.
2. Inverting twice restores the original mask.
3. Undo after invert restores the previous mask state.
4. Inverting when no mask is set initialises a full mask (all 1.0).
5. The "Invert Mask" button is present in the settings panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4")
    proj = Project(name="TestProject", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


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
    qtbot.addWidget(sp)
    return sp


# ---------------------------------------------------------------------------
# 1. invert_mask() produces the complement
# ---------------------------------------------------------------------------


class TestInvertMaskComplement:
    """Painted regions become unpainted and vice versa after invert."""

    def _seed_mask(self, canvas_widget, value: float) -> np.ndarray:
        """Create a small test mask with a known pattern and load it."""
        mask = np.full((50, 50), value, dtype=np.float32)
        # Make top-left quadrant different so we have a non-uniform mask
        mask[:25, :25] = 1.0 - value
        canvas_widget.set_mask(mask.copy())
        return mask

    def test_painted_areas_become_unpainted(self, canvas_widget):
        original = self._seed_mask(canvas_widget, 0.0)
        canvas_widget.invert_mask()
        result = canvas_widget.get_mask()
        assert result is not None
        np.testing.assert_allclose(result, 1.0 - original, atol=1e-6)

    def test_unpainted_areas_become_painted(self, canvas_widget):
        original = self._seed_mask(canvas_widget, 1.0)
        canvas_widget.invert_mask()
        result = canvas_widget.get_mask()
        assert result is not None
        np.testing.assert_allclose(result, 1.0 - original, atol=1e-6)

    def test_returns_before_and_after_tuple(self, canvas_widget):
        original = self._seed_mask(canvas_widget, 0.3)
        before, after = canvas_widget.invert_mask()
        assert before is not None
        assert after is not None
        np.testing.assert_allclose(before, original, atol=1e-6)
        np.testing.assert_allclose(after, 1.0 - original, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Inverting twice restores the original
# ---------------------------------------------------------------------------


class TestInvertMaskTwice:
    """Two successive inversions round-trip back to the original mask."""

    def test_double_invert_restores_original(self, canvas_widget):
        mask = np.random.default_rng(42).random((50, 50)).astype(np.float32)
        canvas_widget.set_mask(mask.copy())
        canvas_widget.invert_mask()
        canvas_widget.invert_mask()
        result = canvas_widget.get_mask()
        assert result is not None
        np.testing.assert_allclose(result, mask, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Undo after invert restores previous state
# ---------------------------------------------------------------------------


class TestInvertMaskUndo:
    """MaskPaintCommand pushed after invert_mask() correctly undoes the operation."""

    def test_undo_restores_original_mask(self, canvas_widget, controller):
        from plottter.gui.commands import MaskPaintCommand

        original = np.random.default_rng(7).random((30, 30)).astype(np.float32)
        canvas_widget.set_mask(original.copy())

        before, after = canvas_widget.invert_mask()
        cmd = MaskPaintCommand(canvas_widget, before, after, "Invert Mask")
        controller.undo_stack.push(cmd)

        # After push, mask should be the inverted state (after)
        np.testing.assert_allclose(canvas_widget.get_mask(), after, atol=1e-6)

        # Undo should restore the original
        controller.undo_stack.undo()
        np.testing.assert_allclose(canvas_widget.get_mask(), original, atol=1e-6)

    def test_redo_after_undo_reapplies_invert(self, canvas_widget, controller):
        from plottter.gui.commands import MaskPaintCommand

        original = np.random.default_rng(13).random((30, 30)).astype(np.float32)
        canvas_widget.set_mask(original.copy())

        before, after = canvas_widget.invert_mask()
        cmd = MaskPaintCommand(canvas_widget, before, after, "Invert Mask")
        controller.undo_stack.push(cmd)

        controller.undo_stack.undo()
        controller.undo_stack.redo()

        np.testing.assert_allclose(canvas_widget.get_mask(), after, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Inverting with no prior mask
# ---------------------------------------------------------------------------


class TestInvertMaskNone:
    """Inverting when _mask_array is None initialises a full mask (all 1.0)."""

    def test_invert_none_produces_full_mask(self, canvas_widget):
        canvas_widget.clear_mask()
        assert canvas_widget.get_mask() is None

        canvas_widget.invert_mask()
        result = canvas_widget.get_mask()
        assert result is not None
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_invert_none_before_is_none(self, canvas_widget):
        canvas_widget.clear_mask()
        before, after = canvas_widget.invert_mask()
        assert before is None
        assert after is not None


# ---------------------------------------------------------------------------
# 5. "Invert Mask" button present in settings panel
# ---------------------------------------------------------------------------


class TestInvertMaskButton:
    """The settings panel exposes an _invert_mask_btn widget."""

    def test_invert_mask_btn_exists(self, settings_panel):
        assert hasattr(settings_panel, "_invert_mask_btn")

    def test_invert_mask_btn_text(self, settings_panel):
        assert settings_panel._invert_mask_btn.text() == "Invert Mask"

    def test_invert_mask_btn_click_inverts_canvas(self, settings_panel, canvas_widget):
        """Clicking the button calls invert_mask on the canvas."""
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        canvas_widget.set_mask(mask.copy())

        # Enter Mask Paint mode so the button is wired up
        settings_panel.on_mode_changed("Mask Paint")
        settings_panel._invert_mask_btn.click()

        result = canvas_widget.get_mask()
        assert result is not None
        np.testing.assert_allclose(result, 1.0 - mask, atol=1e-6)
