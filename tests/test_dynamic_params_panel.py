"""Tests for task 134.1 — three named sub-layouts in the parameter container.

Covers:
(a) _static_params_layout exists and is a QFormLayout within _params_group
(b) _dynamic_params_layout exists and is a QFormLayout within _params_group
(c) _post_proc_layout exists and is a QFormLayout within _post_proc_group
(d) _rebuild_dynamic_params() stub exists and clears the dynamic layout
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


@pytest.fixture
def panel(qapp):
    """Create a SettingsPanel with a minimal controller."""
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)
    return SettingsPanel(controller)


# ---------------------------------------------------------------------------
# test_layout_split_creates_three_sub_layouts
# ---------------------------------------------------------------------------


class TestLayoutSplitCreatesThreeSubLayouts:
    def test_static_params_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_static_params_layout")
        assert isinstance(panel._static_params_layout, QFormLayout)

    def test_dynamic_params_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_dynamic_params_layout")
        assert isinstance(panel._dynamic_params_layout, QFormLayout)

    def test_post_proc_layout_exists(self, panel):
        from PyQt6.QtWidgets import QFormLayout

        assert hasattr(panel, "_post_proc_layout")
        assert isinstance(panel._post_proc_layout, QFormLayout)

    def test_static_and_dynamic_within_params_group(self, panel):
        """Both static and dynamic layouts are nested inside _params_group."""
        params_group = panel._params_group
        # Walk the layout hierarchy: params_group → outer VBoxLayout → sub-layouts
        outer = params_group.layout()
        assert outer is not None, "_params_group should have a layout"

        sub_layouts = []
        for i in range(outer.count()):
            item = outer.itemAt(i)
            if item is not None:
                sub_layouts.append(item.layout())

        assert panel._static_params_layout in sub_layouts
        assert panel._dynamic_params_layout in sub_layouts

    def test_post_proc_layout_within_post_proc_group(self, panel):
        """_post_proc_layout is the layout of _post_proc_group."""
        assert panel._post_proc_group.layout() is panel._post_proc_layout

    def test_dynamic_layout_starts_empty(self, panel):
        """_dynamic_params_layout begins with zero rows (nothing wired yet)."""
        assert panel._dynamic_params_layout.rowCount() == 0

    def test_rebuild_dynamic_params_stub_exists(self, panel):
        """_rebuild_dynamic_params() is callable and clears the dynamic layout."""
        assert callable(getattr(panel, "_rebuild_dynamic_params", None))
        # Should not raise even when layout is already empty
        panel._rebuild_dynamic_params()
        assert panel._dynamic_params_layout.rowCount() == 0


# ---------------------------------------------------------------------------
# test_overrides_merged_into_params (task 134.2)
# ---------------------------------------------------------------------------


class TestDynamicOverridesMergedIntoParams:
    """_dynamic_overrides are passed into params under the reserved key."""

    def test_overrides_merged_into_params(self, panel):
        """When _on_generate fires, params['_dynamic_overrides'] equals the panel's
        stored _dynamic_overrides dict (a shallow copy)."""
        from unittest.mock import MagicMock, patch

        # Minimal stub generator — no special flags set
        mock_gen = MagicMock()
        mock_gen.name = "MockGen"
        mock_gen.emits_multiple_layers = False
        mock_gen.uses_source_image = False
        panel._generator = mock_gen
        panel._current_mode = "Math Art"

        # Put some overrides in the panel state
        panel._dynamic_overrides = {"speed": 7, "density": 0.25}

        captured: dict = {}

        def fake_worker(gen, params, canvas):
            captured["params"] = dict(params)  # snapshot before .start()
            w = MagicMock()
            w.isRunning.return_value = False
            w.is_cancelled.return_value = False
            return w

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker",
            side_effect=fake_worker,
        ):
            panel._on_generate()

        assert "_dynamic_overrides" in captured["params"], (
            "params must contain the '_dynamic_overrides' key"
        )
        assert captured["params"]["_dynamic_overrides"] == {"speed": 7, "density": 0.25}

    def test_empty_overrides_when_no_dynamic_params(self, panel):
        """A generator that provides no dynamic params receives _dynamic_overrides={}."""
        from unittest.mock import MagicMock, patch

        mock_gen = MagicMock()
        mock_gen.name = "StaticGen"
        mock_gen.emits_multiple_layers = False
        mock_gen.uses_source_image = False
        panel._generator = mock_gen
        panel._current_mode = "Math Art"
        panel._dynamic_overrides = {}  # empty — as if get_dynamic_parameters() returned []

        captured: dict = {}

        def fake_worker(gen, params, canvas):
            captured["params"] = dict(params)
            w = MagicMock()
            w.isRunning.return_value = False
            w.is_cancelled.return_value = False
            return w

        with patch(
            "plottter.gui.generator_worker.GeneratorWorker",
            side_effect=fake_worker,
        ):
            panel._on_generate()

        assert captured["params"]["_dynamic_overrides"] == {}


# ---------------------------------------------------------------------------
# Helpers / dummy generator for task 135.1 tests
# ---------------------------------------------------------------------------

def _make_dyn_code(var_name: str, kind: str = "int", default: int = 5) -> str:
    """Return a one-liner adjustable-var declaration for the given name/kind."""
    if kind == "int":
        return f"const {var_name} = {default}; // min=0,max=100\n"
    if kind == "float":
        return f"const {var_name} = 0.5; // min=0.0,max=1.0\n"
    return ""


class _TestDynGenerator:
    """Minimal generator that surfaces adjustable variables from a 'code' field.

    Mirrors what TurtleToyGenerator does: parses the code string with
    parse_adjustable_vars and maps each AdjustableVar to a Parameter.
    """

    name = "_TestDynGen"
    emits_multiple_layers = False
    uses_source_image = False

    def get_presets(self) -> list:
        return []

    def get_parameters(self):
        from plottter.generators.base import StringParam

        return [
            StringParam(
                name="code",
                label="Code",
                default="",
                multiline=True,
            )
        ]

    def get_dynamic_parameters(self, static_param_values: dict) -> list:
        from plottter.generators._adjustable_vars import parse_adjustable_vars
        from plottter.generators.base import FloatParam, IntParam

        code: str = static_param_values.get("code", "")
        if not code:
            return []
        result = []
        for v in parse_adjustable_vars(code):
            if v.kind == "int":
                result.append(
                    IntParam(
                        name=v.name,
                        label=v.name,
                        default=int(v.default) if v.default is not None else 0,
                        min=int(v.min) if v.min is not None else 0,
                        max=int(v.max) if v.max is not None else 100,
                    )
                )
            elif v.kind == "float":
                result.append(
                    FloatParam(
                        name=v.name,
                        label=v.name,
                        default=float(v.default) if v.default is not None else 0.0,
                        min=float(v.min) if v.min is not None else 0.0,
                        max=float(v.max) if v.max is not None else 1.0,
                    )
                )
        return result


@pytest.fixture
def dyn_panel(qapp):
    """SettingsPanel with _TestDynGenerator already set."""
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="DynTest", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)
    p = SettingsPanel(controller)
    gen = _TestDynGenerator()
    p.set_generator(gen)
    return p


# ---------------------------------------------------------------------------
# TestDebounceRebuildDynamicParams  (task 135.1)
# ---------------------------------------------------------------------------


class TestDebounceRebuildDynamicParams:
    """Verify that editing the code widget triggers _rebuild_dynamic_params via
    the 500 ms debounce timer and correctly adds/removes/renames dynamic param
    widgets in _dynamic_params_layout."""

    def _code_widget(self, panel):
        """Return the QPlainTextEdit for the 'code' static param."""
        from PyQt6.QtWidgets import QPlainTextEdit

        w = panel._param_widgets.get("code")
        assert w is not None, "panel must have a 'code' widget"
        assert isinstance(w, QPlainTextEdit)
        return w

    def _row_labels(self, panel) -> list[str]:
        """Return the label texts currently visible in _dynamic_params_layout."""
        from PyQt6.QtWidgets import QFormLayout, QLabel

        layout: QFormLayout = panel._dynamic_params_layout
        labels = []
        for row in range(layout.rowCount()):
            item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            if item is not None:
                w = item.widget()
                if isinstance(w, QLabel):
                    labels.append(w.text())
        return labels

    # ------------------------------------------------------------------

    def test_initial_dynamic_layout_empty(self, dyn_panel):
        """After set_generator with empty code, dynamic layout has no rows."""
        assert dyn_panel._dynamic_params_layout.rowCount() == 0

    def test_variable_added(self, qtbot, dyn_panel):
        """When the code gains an adjustable var, _rebuild_dynamic_params fires
        (after 600 ms) and inserts a row for it."""
        code_w = self._code_widget(dyn_panel)
        code_w.setPlainText(_make_dyn_code("speed"))
        qtbot.wait(600)  # let 500 ms debounce fire

        labels = self._row_labels(dyn_panel)
        assert "speed" in labels, f"expected 'speed' in {labels}"
        assert dyn_panel._dynamic_params_layout.rowCount() == 1

    def test_variable_removed(self, qtbot, dyn_panel):
        """When an existing adjustable var is deleted from the code, its row
        disappears and its override is dropped."""
        code_w = self._code_widget(dyn_panel)
        # Add then remove
        code_w.setPlainText(_make_dyn_code("speed"))
        qtbot.wait(600)
        assert dyn_panel._dynamic_params_layout.rowCount() == 1

        code_w.setPlainText("")
        qtbot.wait(600)

        assert dyn_panel._dynamic_params_layout.rowCount() == 0
        assert "speed" not in dyn_panel._dynamic_overrides

    def test_variable_renamed(self, qtbot, dyn_panel):
        """Replacing 'speed' with 'density' removes the old row and adds a new
        one; neither old widget nor override remains."""
        code_w = self._code_widget(dyn_panel)
        code_w.setPlainText(_make_dyn_code("speed"))
        qtbot.wait(600)
        assert "speed" in self._row_labels(dyn_panel)

        code_w.setPlainText(_make_dyn_code("density"))
        qtbot.wait(600)

        labels = self._row_labels(dyn_panel)
        assert "density" in labels, f"expected 'density' in {labels}"
        assert "speed" not in labels, f"'speed' should be gone, got {labels}"
        assert "speed" not in dyn_panel._dynamic_overrides

    def test_two_variables_added(self, qtbot, dyn_panel):
        """Two variables in the code produce two rows in the dynamic layout."""
        code_w = self._code_widget(dyn_panel)
        code = _make_dyn_code("alpha") + _make_dyn_code("beta")
        code_w.setPlainText(code)
        qtbot.wait(600)

        labels = self._row_labels(dyn_panel)
        assert "alpha" in labels
        assert "beta" in labels
        assert dyn_panel._dynamic_params_layout.rowCount() == 2

    def test_value_preserved_across_rebuild(self, qtbot, dyn_panel):
        """A user-set widget value is preserved when the code is changed in a
        way that keeps the same variable name and type (same-kind diff)."""
        from PyQt6.QtWidgets import QSpinBox

        code_w = self._code_widget(dyn_panel)
        code_w.setPlainText(_make_dyn_code("speed"))
        qtbot.wait(600)

        # Manually set the dynamic spinbox to a non-default value
        spin = dyn_panel._dynamic_param_widgets.get("speed")
        assert isinstance(spin, QSpinBox)
        spin.setValue(42)
        # Wait for the valueChanged → _dynamic_overrides update
        qtbot.wait(50)
        assert dyn_panel._dynamic_overrides.get("speed") == 42

        # Add another variable (speed stays); the rebuild should restore 42
        code_w.setPlainText(_make_dyn_code("speed") + _make_dyn_code("density"))
        qtbot.wait(600)

        new_spin = dyn_panel._dynamic_param_widgets.get("speed")
        assert isinstance(new_spin, QSpinBox)
        assert new_spin.value() == 42, f"expected 42, got {new_spin.value()}"

    def test_textbox_focus_preserved(self, qtbot, dyn_panel):
        """Focus remains on the code textbox across debounce-triggered rebuilds.

        Pre-populates the code widget with an adjustable-var declaration so
        the dynamic layout has a row.  Then gives focus to the code QPlainTextEdit
        and simulates rapid typing via keyClicks.  After the 500 ms debounce
        timer fires _rebuild_dynamic_params the textbox must still have
        keyboard focus (spec §4.3).
        """
        code_w = self._code_widget(dyn_panel)

        # Show the panel so widgets can receive/report keyboard focus.
        dyn_panel.show()
        qtbot.wait(50)

        # Pre-populate with a variable declaration so the first rebuild is
        # meaningful (adds a row).  Wait for it to settle.
        code_w.setPlainText(_make_dyn_code("speed"))
        qtbot.wait(600)

        # Now give focus to the code textbox — this is the UX state we want
        # to preserve across subsequent rebuilds.
        code_w.setFocus()
        qtbot.wait(20)
        assert code_w.hasFocus(), "Precondition: code_w must have focus before rebuild"

        # Simulate rapid typing: keyClicks adds plain alphabetic text, which
        # fires textChanged and starts the 500 ms debounce timer.
        qtbot.keyClicks(code_w, "extra")

        # Wait past the debounce + a comfortable buffer.
        qtbot.wait(600)

        assert code_w.hasFocus(), (
            "Code textbox must retain focus after debounce-triggered "
            "dynamic params rebuild"
        )
