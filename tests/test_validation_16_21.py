"""Phase 16.21 validation: Default Mask Paint mode to manual brush.

Verifies:
1. "Manual Brush" is the first entry in the AI mask mode combo box.
2. When "Manual Brush" is selected, brush controls are enabled and canvas has
   brush painting active.
3. When any AI prompt mode (Point Prompt, Box Prompt, Text Prompt) is selected,
   brush controls are grayed out (disabled) and brush painting is inactive.
4. Instructions label is hidden in Manual Brush mode and visible in Point/Box modes.
5. Entering Mask Paint mode defaults to Manual Brush (brush active immediately).
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
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


# ===========================================================================
# 1. Combo contents and default
# ===========================================================================


class TestComboContents:
    """AI mask mode combo has Manual Brush as the first item."""

    def test_manual_brush_is_first_item(self, settings_panel):
        combo = settings_panel._ai_mask_mode_combo
        assert combo.itemText(0) == "Manual Brush"

    def test_point_prompt_still_present(self, settings_panel):
        combo = settings_panel._ai_mask_mode_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "Point Prompt" in texts

    def test_box_prompt_still_present(self, settings_panel):
        combo = settings_panel._ai_mask_mode_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "Box Prompt" in texts

    def test_text_prompt_still_present(self, settings_panel):
        combo = settings_panel._ai_mask_mode_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "Text Prompt" in texts

    def test_default_selection_is_manual_brush(self, settings_panel):
        """Index 0 is selected by default, which is Manual Brush."""
        combo = settings_panel._ai_mask_mode_combo
        assert combo.currentText() == "Manual Brush"


# ===========================================================================
# 2. Brush controls enabled in Manual Brush mode
# ===========================================================================


class TestBrushControlsManualMode:
    """Brush size, hardness, and erase are enabled when Manual Brush is selected."""

    def test_brush_size_enabled(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert settings_panel._brush_size_spin.isEnabled()

    def test_brush_hardness_enabled(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert settings_panel._brush_hardness_slider.isEnabled()

    def test_erase_check_enabled(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert settings_panel._erase_check.isEnabled()


# ===========================================================================
# 3. Brush controls disabled in AI prompt modes
# ===========================================================================


class TestBrushControlsAIModes:
    """Brush controls are disabled when Point, Box, or Text Prompt is selected."""

    @pytest.mark.parametrize("ai_mode", ["Point Prompt", "Box Prompt", "Text Prompt"])
    def test_brush_size_disabled(self, settings_panel, ai_mode):
        settings_panel._ai_mask_mode_combo.setCurrentText(ai_mode)
        assert not settings_panel._brush_size_spin.isEnabled()

    @pytest.mark.parametrize("ai_mode", ["Point Prompt", "Box Prompt", "Text Prompt"])
    def test_brush_hardness_disabled(self, settings_panel, ai_mode):
        settings_panel._ai_mask_mode_combo.setCurrentText(ai_mode)
        assert not settings_panel._brush_hardness_slider.isEnabled()

    @pytest.mark.parametrize("ai_mode", ["Point Prompt", "Box Prompt", "Text Prompt"])
    def test_erase_check_disabled(self, settings_panel, ai_mode):
        settings_panel._ai_mask_mode_combo.setCurrentText(ai_mode)
        assert not settings_panel._erase_check.isEnabled()


# ===========================================================================
# 4. Canvas interaction based on mode
# ===========================================================================


class TestCanvasInteraction:
    """Canvas brush painting and AI mask mode toggled correctly per selection."""

    def _enter_mask_paint(self, settings_panel):
        settings_panel.on_mode_changed("Mask Paint")

    def test_manual_brush_activates_canvas_painting(self, settings_panel, canvas_widget):
        self._enter_mask_paint(settings_panel)
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert canvas_widget._mask_paint_active is True
        assert canvas_widget._ai_mask_mode is None

    def test_point_prompt_disables_canvas_painting(self, settings_panel, canvas_widget):
        self._enter_mask_paint(settings_panel)
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        assert canvas_widget._mask_paint_active is False
        assert canvas_widget._ai_mask_mode == "point"

    def test_box_prompt_disables_canvas_painting(self, settings_panel, canvas_widget):
        self._enter_mask_paint(settings_panel)
        settings_panel._ai_mask_mode_combo.setCurrentText("Box Prompt")
        assert canvas_widget._mask_paint_active is False
        assert canvas_widget._ai_mask_mode == "box"

    def test_text_prompt_disables_canvas_painting(self, settings_panel, canvas_widget):
        self._enter_mask_paint(settings_panel)
        settings_panel._ai_mask_mode_combo.setCurrentText("Text Prompt")
        assert canvas_widget._mask_paint_active is False
        assert canvas_widget._ai_mask_mode is None


# ===========================================================================
# 5. Entering Mask Paint mode defaults to brush active
# ===========================================================================


class TestMaskPaintModeEntry:
    """Entering Mask Paint mode with default combo (Manual Brush) activates brush."""

    def test_entering_mask_paint_activates_brush(self, settings_panel, canvas_widget):
        settings_panel.on_mode_changed("Mask Paint")
        assert canvas_widget._mask_paint_active is True

    def test_leaving_mask_paint_deactivates_brush(self, settings_panel, canvas_widget):
        settings_panel.on_mode_changed("Mask Paint")
        assert canvas_widget._mask_paint_active is True
        settings_panel.on_mode_changed("Math Art")
        assert canvas_widget._mask_paint_active is False


# ===========================================================================
# 6. Instructions label visibility
# ===========================================================================


class TestInstructionsVisibility:
    """Instructions label is hidden for Manual Brush / Text, visible for Point / Box."""

    def test_instructions_hidden_for_manual_brush(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_instructions.isVisible()

    def test_instructions_visible_for_point_prompt(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        assert settings_panel._ai_mask_instructions.isVisible()

    def test_instructions_visible_for_box_prompt(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Box Prompt")
        assert settings_panel._ai_mask_instructions.isVisible()

    def test_instructions_hidden_for_text_prompt(self, settings_panel):
        settings_panel._ai_mask_mode_combo.setCurrentText("Text Prompt")
        assert not settings_panel._ai_mask_instructions.isVisible()


# ===========================================================================
# 7. Generate Mask button disabled in Manual Brush mode
# ===========================================================================


class TestGenerateMaskButtonState:
    """Generate Mask button is disabled in Manual Brush mode, enabled in AI modes."""

    def test_generate_btn_disabled_when_manual_and_ai_available(self, settings_panel):
        """Button is disabled in Manual Brush mode even if AI key is available."""
        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

    @pytest.mark.parametrize("ai_mode", ["Point Prompt", "Box Prompt", "Text Prompt"])
    def test_generate_btn_enabled_for_ai_modes_when_key_available(self, settings_panel, ai_mode):
        """Button is enabled for AI modes when the API key is available."""
        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText(ai_mode)
        assert settings_panel._ai_mask_generate_btn.isEnabled()

    @pytest.mark.parametrize("ai_mode", ["Point Prompt", "Box Prompt", "Text Prompt"])
    def test_generate_btn_disabled_for_ai_modes_when_no_key(self, settings_panel, ai_mode):
        """Button remains disabled for AI modes when no API key is set."""
        settings_panel._ai_key_available = False
        settings_panel._ai_mask_mode_combo.setCurrentText(ai_mode)
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

    def test_switching_from_manual_to_ai_mode_enables_btn(self, settings_panel):
        """Switching away from Manual Brush to an AI mode re-enables the button."""
        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_generate_btn.isEnabled()
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        assert settings_panel._ai_mask_generate_btn.isEnabled()

    def test_switching_from_ai_mode_to_manual_disables_btn(self, settings_panel):
        """Switching from an AI mode to Manual Brush disables the button."""
        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        assert settings_panel._ai_mask_generate_btn.isEnabled()
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_generate_btn.isEnabled()


# ===========================================================================
# 8. _on_ai_mask_generate is a no-op in Manual Brush mode
# ===========================================================================


class TestGenerateMaskNoOpInManualMode:
    """Calling _on_ai_mask_generate while in Manual Brush mode starts no worker."""

    def test_generate_does_not_start_worker_in_manual_brush(self, settings_panel):
        """_on_ai_mask_generate returns early without creating a worker."""
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        # No worker should be running before the call
        assert settings_panel._ai_mask_worker is None
        # Call the generate handler directly
        settings_panel._on_ai_mask_generate()
        # Worker should still be None — method returned early
        assert settings_panel._ai_mask_worker is None


# ===========================================================================
# 9. Async worker callbacks respect current mode when re-enabling button
# ===========================================================================


class TestAsyncCallbacksRespectMode:
    """_on_ai_mask_result and _on_ai_mask_error must not unconditionally enable
    the Generate Mask button — they should respect the current combo mode."""

    def test_result_callback_keeps_btn_disabled_in_manual_mode(self, settings_panel):
        """If user switches to Manual Brush while worker runs, result callback
        must leave the button disabled."""
        import numpy as np

        settings_panel._ai_key_available = True
        # Simulate user switching to Manual Brush after worker was dispatched
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

        # Simulate worker finishing with a dummy mask
        dummy_mask = np.zeros((10, 10), dtype=np.uint8)
        settings_panel._on_ai_mask_result(dummy_mask)

        # Button must remain disabled — we're in Manual Brush mode
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

    def test_error_callback_keeps_btn_disabled_in_manual_mode(self, settings_panel, monkeypatch):
        """If user switches to Manual Brush while worker runs, error callback
        must leave the button disabled."""
        # Suppress the QMessageBox.critical dialog
        import plottter.gui.settings_panel as _sp_mod
        monkeypatch.setattr(_sp_mod.QMessageBox, "critical", lambda *a, **kw: None)

        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Manual Brush")
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

        settings_panel._on_ai_mask_error("some error")

        # Button must remain disabled — we're in Manual Brush mode
        assert not settings_panel._ai_mask_generate_btn.isEnabled()

    def test_result_callback_enables_btn_in_ai_mode(self, settings_panel):
        """If user is in an AI mode when the worker finishes, result callback
        re-enables the button."""
        import numpy as np

        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        # Manually disable (simulating an in-progress worker)
        settings_panel._ai_mask_generate_btn.setEnabled(False)

        dummy_mask = np.zeros((10, 10), dtype=np.uint8)
        settings_panel._on_ai_mask_result(dummy_mask)

        assert settings_panel._ai_mask_generate_btn.isEnabled()

    def test_error_callback_enables_btn_in_ai_mode(self, settings_panel, monkeypatch):
        """If user is in an AI mode when the worker errors, error callback
        re-enables the button."""
        import plottter.gui.settings_panel as _sp_mod
        monkeypatch.setattr(_sp_mod.QMessageBox, "critical", lambda *a, **kw: None)

        settings_panel._ai_key_available = True
        settings_panel._ai_mask_mode_combo.setCurrentText("Point Prompt")
        settings_panel._ai_mask_generate_btn.setEnabled(False)

        settings_panel._on_ai_mask_error("boom")

        assert settings_panel._ai_mask_generate_btn.isEnabled()
