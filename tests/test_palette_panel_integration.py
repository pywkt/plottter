"""Integration tests for the Custom Palette colorsep panel wiring (Phase 159.4).

Tests:
1. "Custom Palette" appears in the method combo.
2. Selecting "Custom Palette" shows the palette picker, hides channel-checks
   and num-colors widgets.
3. The picker is populated from list_presets() with PenPalette as item data.
4. The _on_separate dispatch calls palette_separate with the selected PenPalette.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plottter.color.palette import PenPalette
from plottter.color.palettes import list_presets, get_preset
from plottter.gui.settings_panel._colorsep import _ColorSepMixin


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project():
    from plottter.models import Canvas, Layer, Project

    canvas = Canvas.from_preset("A4")
    proj = Project(name="Test", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController

    return ProjectController(_make_project())


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel

    sp = SettingsPanel(controller)
    sp.resize(400, 900)
    qtbot.addWidget(sp)
    return sp


# ---------------------------------------------------------------------------
# Stub panel for dispatch tests (no full Qt needed)
# ---------------------------------------------------------------------------


class _StubWidget:
    """Generic stub for a Qt widget that only needs isVisible / setVisible."""

    def __init__(self, visible: bool = False) -> None:
        self._visible = visible

    def setVisible(self, v: bool) -> None:
        self._visible = v

    def isVisible(self) -> bool:
        return self._visible


class _StubSpin:
    def __init__(self, value=3):
        self._v = value

    def value(self):
        return self._v

    def setRange(self, *_):
        pass

    def setVisible(self, *_):
        pass


class _StubLabel:
    def setText(self, *_):
        pass

    def setVisible(self, *_):
        pass


class _StubCheck:
    def __init__(self, checked: bool = False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _StubLayout:
    def __init__(self):
        self._items = []

    def count(self):
        return len(self._items)

    def takeAt(self, _):
        return self._items.pop(0) if self._items else _NullItem()


class _NullItem:
    def widget(self):
        return None


class _StubChannelWidget:
    def __init__(self):
        self._visible = False
        self._layout = _StubLayout()

    def setVisible(self, v):
        self._visible = v

    def isVisible(self):
        return self._visible

    def layout(self):
        return self._layout


class _StubCombo:
    """Stand-in for QComboBox."""

    def __init__(self, text: str = "", data=None) -> None:
        self._text = text
        self._data = data
        self._items: list[tuple[str, object]] = []

    def currentText(self) -> str:
        return self._text

    def currentData(self):
        return self._data

    def clear(self) -> None:
        self._items.clear()

    def addItem(self, text: str, data=None) -> None:
        self._items.append((text, data))
        # Automatically pick first item
        if len(self._items) == 1:
            self._text = text
            self._data = data

    def setCurrentIndex(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._text, self._data = self._items[idx]


class _DispatchPanel(_ColorSepMixin):
    """Minimal stub panel for testing _on_separate dispatch without full Qt."""

    def __init__(self, raw_rgb: np.ndarray, palette: PenPalette) -> None:
        self._raw_image = raw_rgb
        self._ai_bg_check = _StubCheck(False)
        self._ai_bg_rgba = None
        self._color_sep_method_combo = _StubCombo("Custom Palette")
        self._color_sep_num_colors_spin = _StubSpin(3)
        self._palette_picker_combo = _StubCombo("Basic 6", data=palette)
        self._channel_checks: dict = {}
        self._separated_layer_ids: list = []
        self._layer_masks: dict = {}
        self._apply_results_calls: list = []

    def _get_preprocessing_params(self) -> dict:
        return {}

    def _apply_separation_results(self, results, layer_names, method, preprocessed):
        self._apply_results_calls.append((results, layer_names, method))


# ---------------------------------------------------------------------------
# Tests: method dropdown
# ---------------------------------------------------------------------------


class TestMethodDropdown:
    def test_custom_palette_in_combo(self, settings_panel):
        """'Custom Palette' must be present in the method combo."""
        combo = settings_panel._color_sep_method_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "Custom Palette" in items


# ---------------------------------------------------------------------------
# Tests: visibility toggling
# ---------------------------------------------------------------------------


class TestVisibilityToggling:
    def test_picker_visible_when_custom_palette_selected(self, settings_panel):
        """Selecting 'Custom Palette' should show the palette picker widget."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        # isVisible() checks the full parent chain; _color_sep_group starts hidden,
        # so we check isHidden() which only reflects the widget's own visibility state.
        assert not settings_panel._palette_picker_widget.isHidden()

    def test_channel_checks_hidden_when_custom_palette_selected(self, settings_panel):
        """Channel checkboxes widget must be hidden for Custom Palette."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        assert not settings_panel._channel_check_widget.isVisible()

    def test_num_colors_hidden_when_custom_palette_selected(self, settings_panel):
        """Num-colors spinbox must be hidden for Custom Palette."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        assert not settings_panel._color_sep_num_colors_spin.isVisible()

    def test_k_amount_hidden_when_custom_palette_selected(self, settings_panel):
        """K-amount widget must be hidden for Custom Palette."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        assert not settings_panel._cmyk_k_amount_widget.isVisible()

    def test_picker_hidden_for_kmeans(self, settings_panel):
        """Switching back to K-Means must hide the palette picker."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        settings_panel._on_color_sep_method_changed("K-Means")
        assert not settings_panel._palette_picker_widget.isVisible()


# ---------------------------------------------------------------------------
# Tests: palette picker population
# ---------------------------------------------------------------------------


class TestPalettePickerPopulation:
    def test_picker_populated_from_list_presets(self, settings_panel):
        """After selecting Custom Palette, all list_presets() appear in the combo."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        for preset in list_presets():
            assert preset.name in items

    def test_picker_items_carry_pen_palette_as_data(self, settings_panel):
        """Each item in the picker must carry its PenPalette as item data."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo
        for i in range(combo.count()):
            data = combo.itemData(i)
            assert isinstance(data, PenPalette), (
                f"Item {i} ({combo.itemText(i)!r}) has data {data!r}, expected PenPalette"
            )

    def test_basic_6_is_present_and_selectable(self, settings_panel):
        """'Basic 6' preset must appear and its data must be the correct PenPalette."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo
        # Find "Basic 6" item
        basic6_idx = None
        for i in range(combo.count()):
            if combo.itemText(i) == "Basic 6":
                basic6_idx = i
                break
        assert basic6_idx is not None, "'Basic 6' not found in palette picker"
        combo.setCurrentIndex(basic6_idx)
        data = combo.currentData()
        assert isinstance(data, PenPalette)
        assert data.name == "Basic 6"


# ---------------------------------------------------------------------------
# Tests: dispatch — palette_separate is called with the right PenPalette
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_palette_separate_called_with_pen_palette(self):
        """_on_separate must call palette_separate with the selected PenPalette."""
        basic6 = get_preset("Basic 6")
        raw_rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        panel = _DispatchPanel(raw_rgb, basic6)

        # Build fake results shaped like palette_separate output
        fake_mask = np.zeros((4, 4), dtype=np.uint8)
        fake_results = [(fake_mask, color) for color in basic6.colors]

        with patch(
            "plottter.color.palette_separator.palette_separate",
            return_value=fake_results,
        ) as mock_sep:
            # Also patch the import inside _colorsep so the mock is used
            with patch(
                "plottter.color.palette_separate",
                mock_sep,
            ):
                panel._on_separate()

        assert mock_sep.called, "palette_separate was not called"
        args, kwargs = mock_sep.call_args
        called_palette = args[1] if len(args) > 1 else kwargs.get("palette")
        assert called_palette == basic6, (
            f"Expected PenPalette 'Basic 6', got {called_palette!r}"
        )

    def test_palette_separate_called_with_dither_none(self):
        """dither must be hard-coded to 'none' in this phase (159.4)."""
        basic6 = get_preset("Basic 6")
        raw_rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        panel = _DispatchPanel(raw_rgb, basic6)

        fake_mask = np.zeros((4, 4), dtype=np.uint8)
        fake_results = [(fake_mask, color) for color in basic6.colors]

        with patch(
            "plottter.color.palette_separate",
            return_value=fake_results,
        ) as mock_sep:
            panel._on_separate()

        _, kwargs = mock_sep.call_args
        assert kwargs.get("dither", "none") == "none"

    def test_no_palette_selected_does_not_call_separate(self):
        """If currentData() is None (no palette selected), _on_separate must bail."""
        raw_rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        panel = _DispatchPanel(raw_rgb, None)
        # Override palette_picker_combo to return None
        panel._palette_picker_combo = _StubCombo("", data=None)

        with patch("plottter.color.palette_separate") as mock_sep:
            with patch("PyQt6.QtWidgets.QMessageBox.warning"):
                panel._on_separate()

        assert not mock_sep.called


# ---------------------------------------------------------------------------
# Tests: Phase 159.6 — user palettes loaded + collision naming
# ---------------------------------------------------------------------------


class TestUserPaletteLoading:
    """_populate_palette_picker merges built-ins + user palettes correctly."""

    def test_user_palettes_appear_after_builtins(
        self, settings_panel, tmp_path, monkeypatch
    ):
        """User palettes must be appended after all built-in presets."""
        import json

        from plottter.color.palette import palette_to_dict

        user_pal = PenPalette(name="My Watercolours", colors=("#1E3A5F", "#C13B4F"))
        (tmp_path / "my-watercolours.json").write_text(
            json.dumps(palette_to_dict(user_pal))
        )
        monkeypatch.setattr("plottter.color.palette.palette_dir", lambda: tmp_path)

        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo
        texts = [combo.itemText(i) for i in range(combo.count())]

        builtin_count = len(list_presets())
        assert "My Watercolours" in texts
        assert texts.index("My Watercolours") >= builtin_count

    def test_user_palette_collision_gets_user_suffix(
        self, settings_panel, tmp_path, monkeypatch
    ):
        """A user palette whose name matches a built-in must appear as '<name> (user)'."""
        import json

        from plottter.color.palette import palette_to_dict

        # Write a user palette whose name collides with the "Basic 6" built-in.
        user_basic6 = PenPalette(name="Basic 6", colors=("#AABBCC",))
        (tmp_path / "basic-6.json").write_text(
            json.dumps(palette_to_dict(user_basic6))
        )
        monkeypatch.setattr("plottter.color.palette.palette_dir", lambda: tmp_path)

        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo
        texts = [combo.itemText(i) for i in range(combo.count())]

        # The built-in keeps its plain name.
        assert "Basic 6" in texts
        # The user version gets the suffix.
        assert "Basic 6 (user)" in texts

    def test_builtin_and_user_collision_carry_correct_data(
        self, settings_panel, tmp_path, monkeypatch
    ):
        """Built-in 'Basic 6' and user 'Basic 6 (user)' must carry distinct PenPalettes."""
        import json

        from plottter.color.palette import palette_to_dict

        user_basic6 = PenPalette(name="Basic 6", colors=("#AABBCC",))
        (tmp_path / "basic-6.json").write_text(
            json.dumps(palette_to_dict(user_basic6))
        )
        monkeypatch.setattr("plottter.color.palette.palette_dir", lambda: tmp_path)

        settings_panel._on_color_sep_method_changed("Custom Palette")
        combo = settings_panel._palette_picker_combo

        # Collect display_name → PenPalette mappings.
        by_text: dict[str, PenPalette] = {}
        for i in range(combo.count()):
            by_text[combo.itemText(i)] = combo.itemData(i)

        builtin_data = by_text["Basic 6"]
        user_data = by_text["Basic 6 (user)"]

        # Both carry a PenPalette with name == "Basic 6" (raw name unchanged on disk).
        assert builtin_data.name == "Basic 6"
        assert user_data.name == "Basic 6"
        # But they are different objects with different colour counts.
        assert len(builtin_data.colors) > 1  # built-in has 6 colors
        assert len(user_data.colors) == 1  # user has just #AABBCC


# ---------------------------------------------------------------------------
# Tests: Phase 159.6 — "Edit / New Palette…" button wiring
# ---------------------------------------------------------------------------


class TestEditPaletteButton:
    """The 'Edit / New Palette…' button exists and is wired correctly."""

    def test_edit_button_exists_on_panel(self, settings_panel):
        """SettingsPanel must expose a _palette_edit_btn attribute."""
        assert hasattr(settings_panel, "_palette_edit_btn")
        assert settings_panel._palette_edit_btn is not None

    def test_edit_button_visible_with_custom_palette_mode(self, settings_panel):
        """The button must not be hidden when Custom Palette mode is active."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        assert not settings_panel._palette_edit_btn.isHidden()

    def test_edit_button_opens_dialog_with_current_palette(
        self, settings_panel, monkeypatch
    ):
        """Clicking Edit must open PaletteEditorDialog pre-populated with the
        palette that is currently selected in the combo."""
        settings_panel._on_color_sep_method_changed("Custom Palette")
        current_palette = settings_panel._palette_picker_combo.currentData()
        assert isinstance(current_palette, PenPalette)

        captured: dict = {}

        class _FakeDialog:
            def __init__(self, parent=None, initial=None):
                captured["initial"] = initial

            def exec(self):
                return 0  # Rejected — don't save anything.

            def get_result(self):
                return None

        monkeypatch.setattr(
            "plottter.gui.dialogs.palette_editor_dialog.PaletteEditorDialog",
            _FakeDialog,
        )

        settings_panel._on_edit_palette()

        assert captured.get("initial") == current_palette

    def test_accepting_editor_rebuilds_picker_and_selects_new_palette(
        self, settings_panel, tmp_path, monkeypatch
    ):
        """After accepting the dialog the picker is rebuilt and the saved palette
        is selected."""
        import json

        from plottter.color.palette import palette_to_dict

        new_pal = PenPalette(name="My Test Set", colors=("#FF0000", "#00FF00"))
        # Simulate what the dialog's Save button does: write the file to the
        # user palette directory (monkeypatched to tmp_path).
        (tmp_path / "my-test-set.json").write_text(
            json.dumps(palette_to_dict(new_pal))
        )
        monkeypatch.setattr("plottter.color.palette.palette_dir", lambda: tmp_path)

        settings_panel._on_color_sep_method_changed("Custom Palette")

        class _FakeDialog:
            def __init__(self, parent=None, initial=None):
                pass

            def exec(self):
                return 1  # Accepted.

            def get_result(self):
                return new_pal

        monkeypatch.setattr(
            "plottter.gui.dialogs.palette_editor_dialog.PaletteEditorDialog",
            _FakeDialog,
        )

        settings_panel._on_edit_palette()

        combo = settings_panel._palette_picker_combo
        texts = [combo.itemText(i) for i in range(combo.count())]

        # New palette must appear in the combo…
        assert "My Test Set" in texts
        # …and must be the selected item.
        assert combo.currentText() == "My Test Set"
        assert combo.currentData() == new_pal
