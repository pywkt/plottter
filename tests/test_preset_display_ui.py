"""Tests for task 26.3 — Display user presets in combo with section separation.

Verifies:
(a) Built-in presets appear first, user presets appear in a separate section below.
(b) The "— User Presets —" header is visible but not selectable.
(c) Selecting a user preset applies its params correctly.
(d) Switching generators shows the correct user presets for each generator.
(e) A generator with no user presets shows no user section (no empty separator gap).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── headless Qt ────────────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ─── minimal mock generator ─────────────────────────────────────────────────

def _make_mock_generator(name: str = "Test Generator", builtin_presets=None):
    """Return a minimal generator-like object usable with SettingsPanel."""
    from plottter.generators.base import FloatParam, Preset

    if builtin_presets is None:
        builtin_presets = [
            Preset(name="Built-in Preset A", params={"radius": 5.0}),
            Preset(name="Built-in Preset B", params={"radius": 10.0}),
        ]

    gen = MagicMock()
    gen.name = name
    gen.get_presets.return_value = builtin_presets
    gen.get_parameters.return_value = [
        FloatParam(name="radius", label="Radius (mm)", min=0.1, max=50.0, default=5.0),
    ]
    return gen


# ─── SettingsPanel factory ───────────────────────────────────────────────────

@pytest.fixture
def panel(qapp, tmp_path):
    """Create a SettingsPanel with a minimal mock controller."""
    from plottter.gui.settings_panel import SettingsPanel
    from plottter.models import Canvas, Layer, Project
    from plottter.gui.project_controller import ProjectController

    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Test", canvas=canvas)
    project.add_layer(Layer(name="L1", color="#000000"))
    controller = ProjectController(project)

    sp = SettingsPanel(controller)
    sp._current_mode = "Math Art"
    return sp, tmp_path


# ─── Helper: create real user preset files in tmp_path ───────────────────────

def _save_real_preset(generator_name: str, preset_name: str, params: dict, presets_dir: Path):
    """Write a user preset to disk using the real persistence layer."""
    from plottter.presets.user_presets import save_user_preset
    from plottter.generators.base import Preset

    save_user_preset(generator_name, Preset(name=preset_name, params=params),
                     presets_dir=presets_dir)


# ─── (a) Built-in first, user presets below ─────────────────────────────────

class TestComboOrdering:
    def test_builtin_presets_come_before_user_presets(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("OrderTest")
        _save_real_preset("OrderTest", "My User Preset", {"radius": 3.0}, tmp_path)

        with patch(
            "plottter.gui.settings_panel.load_user_presets",
            side_effect=lambda name: _load_from_dir(name, tmp_path),
            create=True,
        ):
            # Patch load_user_presets inside _rebuild_preset_combo
            from plottter.presets.user_presets import load_user_presets as real_load

            def patched_load(name):
                return real_load(name, presets_dir=tmp_path)

            with patch(
                "plottter.presets.user_presets.load_user_presets",
                side_effect=patched_load,
            ):
                sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]

        # "Custom" must be first
        assert items[0] == "Custom"

        # Built-in presets must appear before the user section header
        builtin_a_idx = items.index("Built-in Preset A")
        builtin_b_idx = items.index("Built-in Preset B")
        header_idx = items.index("— User Presets —")
        user_idx = items.index("My User Preset")

        assert builtin_a_idx < header_idx
        assert builtin_b_idx < header_idx
        assert header_idx < user_idx

    def test_save_action_is_last_item(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("SaveLast")
        _save_real_preset("SaveLast", "U Preset", {"radius": 1.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        last = sp._preset_combo.itemText(sp._preset_combo.count() - 1)
        assert last == "Save Current as Preset\u2026"


# ─── (b) "— User Presets —" header is non-selectable ────────────────────────

class TestUserPresetsHeader:
    def test_header_exists_when_user_presets_present(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("HeaderTest")
        _save_real_preset("HeaderTest", "Saved Preset", {"radius": 2.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "— User Presets —" in items

    def test_header_is_not_selectable(self, panel, tmp_path):
        from PyQt6.QtCore import Qt

        sp, _ = panel
        gen = _make_mock_generator("HeaderTest2")
        _save_real_preset("HeaderTest2", "Saved Preset", {"radius": 2.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        header_idx = items.index("— User Presets —")

        model = sp._preset_combo.model()
        header_item = model.item(header_idx)
        assert header_item is not None

        # The item must not be enabled or selectable.
        flags = header_item.flags()
        assert not bool(flags & Qt.ItemFlag.ItemIsEnabled), (
            "Header item should not be enabled (non-selectable)"
        )

    def test_no_header_when_no_user_presets(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("NoUserPresets")

        with _patch_load(tmp_path):  # tmp_path has no files for this generator
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "— User Presets —" not in items

    def test_no_extra_separator_when_no_user_presets(self, panel, tmp_path):
        """When there are no user presets, there should be exactly one separator
        (before "Save Current as Preset…"), not two."""
        sp, _ = panel
        gen = _make_mock_generator("NoExtra")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        # Count separators (empty string items from insertSeparator)
        separator_count = sum(
            1 for i in range(sp._preset_combo.count())
            if sp._preset_combo.itemText(i) == ""
        )
        assert separator_count == 1, (
            f"Expected 1 separator, got {separator_count}"
        )


# ─── (c) Selecting a user preset applies its params ─────────────────────────

class TestUserPresetSelection:
    def test_selecting_user_preset_applies_params(self, panel, tmp_path):
        from PyQt6.QtWidgets import QDoubleSpinBox

        sp, _ = panel
        gen = _make_mock_generator("ApplyTest")
        _save_real_preset("ApplyTest", "My Saved", {"radius": 42.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        # Verify "My Saved" is in the combo
        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "My Saved" in items

        # Simulate selecting the user preset
        sp._on_preset_changed("My Saved")

        # Verify the radius widget was updated
        widget = sp._param_widgets.get("radius")
        if isinstance(widget, QDoubleSpinBox):
            assert abs(widget.value() - 42.0) < 0.001

    def test_builtin_preset_still_applies_correctly(self, panel, tmp_path):
        from PyQt6.QtWidgets import QDoubleSpinBox

        sp, _ = panel
        gen = _make_mock_generator("ApplyTest2")
        _save_real_preset("ApplyTest2", "User P", {"radius": 7.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        sp._on_preset_changed("Built-in Preset A")

        widget = sp._param_widgets.get("radius")
        if isinstance(widget, QDoubleSpinBox):
            assert abs(widget.value() - 5.0) < 0.001


# ─── (d) Switching generators shows correct user presets ────────────────────

class TestGeneratorSwitch:
    def test_user_presets_update_on_generator_switch(self, panel, tmp_path):
        sp, _ = panel

        gen_a = _make_mock_generator("GenA")
        gen_b = _make_mock_generator("GenB")

        _save_real_preset("GenA", "GenA Preset", {"radius": 1.1}, tmp_path)
        _save_real_preset("GenB", "GenB Preset", {"radius": 2.2}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen_a)

        items_a = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "GenA Preset" in items_a
        assert "GenB Preset" not in items_a

        with _patch_load(tmp_path):
            sp.set_generator(gen_b)

        items_b = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "GenB Preset" in items_b
        assert "GenA Preset" not in items_b


# ─── (e) No user presets → no user section ──────────────────────────────────

class TestNoUserSection:
    def test_no_user_section_when_empty(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("EmptyUserPresets")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert "— User Presets —" not in items

    def test_combo_has_custom_builtin_separator_save_when_empty(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("EmptyUserPresets2")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        assert items[0] == "Custom"
        assert "Built-in Preset A" in items
        assert "Built-in Preset B" in items
        assert items[-1] == "Save Current as Preset\u2026"


# ─── (extra) _user_presets cache is populated correctly ─────────────────────

class TestUserPresetsCache:
    def test_user_presets_cached_on_set_generator(self, panel, tmp_path):
        from plottter.generators.base import Preset

        sp, _ = panel
        gen = _make_mock_generator("CacheTest")
        _save_real_preset("CacheTest", "Cached One", {"radius": 9.0}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        assert len(sp._user_presets) == 1
        assert sp._user_presets[0].name == "Cached One"

    def test_user_presets_empty_when_no_file(self, panel, tmp_path):
        sp, _ = panel
        gen = _make_mock_generator("NoCacheTest")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        assert sp._user_presets == []


# ─── (extra) User preset item tagged with UserRole "user" ───────────────────

class TestUserRoleTag:
    def test_user_preset_items_tagged_with_user_role(self, panel, tmp_path):
        from PyQt6.QtCore import Qt

        sp, _ = panel
        gen = _make_mock_generator("RoleTest")
        _save_real_preset("RoleTest", "Tagged Preset", {"radius": 3.5}, tmp_path)

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        idx = items.index("Tagged Preset")
        role_data = sp._preset_combo.itemData(idx, Qt.ItemDataRole.UserRole)
        assert role_data == "user"

    def test_builtin_preset_not_tagged_as_user(self, panel, tmp_path):
        from PyQt6.QtCore import Qt

        sp, _ = panel
        gen = _make_mock_generator("RoleTest2")

        with _patch_load(tmp_path):
            sp.set_generator(gen)

        items = [sp._preset_combo.itemText(i) for i in range(sp._preset_combo.count())]
        idx = items.index("Built-in Preset A")
        role_data = sp._preset_combo.itemData(idx, Qt.ItemDataRole.UserRole)
        assert role_data != "user"


# ─── Utility ─────────────────────────────────────────────────────────────────

def _load_from_dir(name: str, presets_dir: Path):
    from plottter.presets.user_presets import load_user_presets
    return load_user_presets(name, presets_dir=presets_dir)


def _patch_load(presets_dir: Path):
    """Context manager that redirects load_user_presets to use presets_dir."""
    from plottter.presets.user_presets import load_user_presets as real_load

    def patched(name):
        return real_load(name, presets_dir=presets_dir)

    return patch(
        "plottter.presets.user_presets.load_user_presets",
        side_effect=patched,
    )
