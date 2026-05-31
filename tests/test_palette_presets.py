"""Tests for built-in palette presets (Phase 159.3)."""

import pytest

from plottter.color.palette import PenPalette
from plottter.color.palettes import PALETTE_PRESETS, list_presets, get_preset
from plottter.color.palettes import basic_6, copic_12, sakura_metallic_5, grayscale_5


# ---------------------------------------------------------------------------
# Individual module imports
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_basic_6_imports(self):
        assert hasattr(basic_6, "PALETTE")
        assert isinstance(basic_6.PALETTE, PenPalette)

    def test_copic_12_imports(self):
        assert hasattr(copic_12, "PALETTE")
        assert isinstance(copic_12.PALETTE, PenPalette)

    def test_sakura_metallic_5_imports(self):
        assert hasattr(sakura_metallic_5, "PALETTE")
        assert isinstance(sakura_metallic_5.PALETTE, PenPalette)

    def test_grayscale_5_imports(self):
        assert hasattr(grayscale_5, "PALETTE")
        assert isinstance(grayscale_5.PALETTE, PenPalette)


# ---------------------------------------------------------------------------
# Pen counts from spec §6
# ---------------------------------------------------------------------------

class TestPresetCounts:
    def test_basic_6_has_six_pens(self):
        assert basic_6.PALETTE.count == 6

    def test_copic_12_has_twelve_pens(self):
        assert copic_12.PALETTE.count == 12

    def test_sakura_metallic_5_has_five_pens(self):
        assert sakura_metallic_5.PALETTE.count == 5

    def test_grayscale_5_has_five_pens(self):
        assert grayscale_5.PALETTE.count == 5


# ---------------------------------------------------------------------------
# list_presets()
# ---------------------------------------------------------------------------

class TestListPresets:
    def test_returns_all_builtin_presets(self):
        presets = list_presets()
        assert len(presets) == len(PALETTE_PRESETS)
        assert len(presets) >= 7  # baseline: don't accidentally lose presets

    def test_all_items_are_pen_palettes(self):
        for p in list_presets():
            assert isinstance(p, PenPalette)

    def test_contains_expected_names(self):
        names = {p.name for p in list_presets()}
        assert "Basic 6" in names
        assert "Copic 12" in names
        assert "Sakura Metallic 5" in names
        assert "Grayscale 5" in names
        assert "RYBK 4" in names
        assert "CMYKOG 6" in names
        assert "Risograph 6" in names

    def test_order_matches_palette_presets_dict(self):
        assert list_presets() == list(PALETTE_PRESETS.values())


# ---------------------------------------------------------------------------
# get_preset() — case-insensitivity
# ---------------------------------------------------------------------------

class TestGetPreset:
    def test_exact_name_lookup(self):
        p = get_preset("Basic 6")
        assert p.name == "Basic 6"

    def test_lowercase_lookup(self):
        assert get_preset("basic 6") == get_preset("Basic 6")

    def test_uppercase_lookup(self):
        assert get_preset("BASIC 6") == get_preset("Basic 6")

    def test_mixed_case_copic(self):
        assert get_preset("copic 12") == get_preset("Copic 12")

    def test_mixed_case_sakura(self):
        assert get_preset("sakura metallic 5") == get_preset("Sakura Metallic 5")

    def test_mixed_case_grayscale(self):
        assert get_preset("GRAYSCALE 5") == get_preset("Grayscale 5")

    def test_leading_trailing_whitespace_stripped(self):
        assert get_preset("  basic 6  ") == get_preset("Basic 6")

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            get_preset("Nonexistent Palette")


# ---------------------------------------------------------------------------
# Palette_presets re-export from plottter.color
# ---------------------------------------------------------------------------

class TestColorPackageReexport:
    def test_get_preset_importable_from_color(self):
        from plottter.color import get_preset as gp  # noqa: F401
        assert callable(gp)

    def test_list_presets_importable_from_color(self):
        from plottter.color import list_presets as lp  # noqa: F401
        assert callable(lp)

    def test_palette_presets_importable_from_color(self):
        from plottter.color import PALETTE_PRESETS as pp  # noqa: F401
        assert isinstance(pp, dict)
        assert len(pp) == len(PALETTE_PRESETS)


# ---------------------------------------------------------------------------
# Individual palette validity
# ---------------------------------------------------------------------------

class TestPaletteValidity:
    @pytest.mark.parametrize("palette", list_presets())
    def test_all_colors_are_valid_hex(self, palette: PenPalette):
        import re
        pattern = re.compile(r"^#[0-9A-F]{6}$")
        for color in palette.colors:
            assert pattern.match(color), f"{palette.name}: invalid hex {color!r}"

    @pytest.mark.parametrize("palette", list_presets())
    def test_colors_are_uppercase(self, palette: PenPalette):
        for color in palette.colors:
            assert color == color.upper(), f"{palette.name}: color {color!r} not uppercase"

    @pytest.mark.parametrize("palette", list_presets())
    def test_name_is_non_empty(self, palette: PenPalette):
        assert palette.name.strip() != ""

    def test_copic_has_source_url(self):
        p = get_preset("Copic 12")
        assert p.source.startswith("http"), (
            "Copic 12 preset must document a source URL in the `source` field"
        )
