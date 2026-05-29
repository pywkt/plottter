"""End-to-end tests for the Hershey font catalog + shim API.

Smoke-tests every shipped font: parses, has at least the basic Latin set,
and renders a sample string without raising.  Also verifies the legacy
alias contract that keeps old projects loading.
"""

from __future__ import annotations

import pytest

from plottter.fonts.hershey import (
    CAP_HEIGHT,
    DEFAULT_FONT_NAME,
    FONTS,
    choices_for_param,
    entries_by_category,
    get_entry,
    glyph_strokes,
    list_categories,
    list_entries,
    list_names,
    load_font,
    resolve_name,
)
from plottter.fonts.hershey.catalog import _LEGACY_ALIASES


# ---------------------------------------------------------------------------
# Catalog structure
# ---------------------------------------------------------------------------


def test_default_font_resolves_to_a_real_entry():
    assert DEFAULT_FONT_NAME in list_names()
    entry = get_entry(DEFAULT_FONT_NAME)
    assert entry.name == DEFAULT_FONT_NAME
    assert entry.path.exists()


def test_catalog_has_all_expected_categories():
    cats = list_categories()
    for required in (
        "EMS Modern",
        "Hershey Sans",
        "Hershey Serif",
        "Hershey Script",
        "Hershey Gothic",
        "Symbols",
        "Custom",
    ):
        assert required in cats, f"missing category {required!r}"


def test_entries_by_category_round_trip():
    grouped = entries_by_category()
    total = sum(len(v) for v in grouped.values())
    assert total == len(list_entries())


def test_every_entry_has_a_data_file():
    for entry in list_entries():
        assert entry.path.exists(), f"missing font file: {entry.path}"


# ---------------------------------------------------------------------------
# Per-font parsing + Latin-set coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list_names())
def test_font_parses_and_has_basic_glyphs(name):
    font = load_font(name)
    # Symbol fonts (Greek, Cyrillic, Japanese, Math, …) intentionally omit
    # the Latin alphabet; only assert latin coverage on text-style faces.
    entry = get_entry(name)
    if entry.category in {"Symbols"}:
        # Even symbol fonts ship at least some glyphs.
        assert len(font.glyphs) >= 10, f"{name}: only {len(font.glyphs)} glyphs"
        return
    # All "text" fonts must cover ASCII letters + digits + a space.
    assert " " in font.glyphs
    for ch in "ABCabc0123":
        assert ch in font.glyphs, f"{name}: missing {ch!r}"


@pytest.mark.parametrize("name", list_names())
def test_font_metrics_are_sane(name):
    m = load_font(name).metrics
    assert m.units_per_em > 0
    assert m.cap_height > 0
    assert m.x_height >= 0
    assert m.ascent > 0
    assert m.descent <= 0


@pytest.mark.parametrize("name", list_names())
def test_font_renders_sample_via_shim(name):
    entry = get_entry(name)
    sample = "Hi 1" if entry.category == "Symbols" else "Hi 1!"
    # Skip the assertion for symbol fonts that don't contain ASCII at all —
    # the shim's "?" fallback would also be missing.
    if entry.category == "Symbols":
        font = load_font(name)
        if "?" not in font.glyphs and not any(c in font.glyphs for c in sample):
            pytest.skip(f"{name} has no ASCII coverage and no '?' fallback")
    for ch in sample:
        left, right, strokes = glyph_strokes(ch, name)
        assert right >= left  # advance is non-negative
        for stroke in strokes:
            assert len(stroke) >= 2  # shim filters zero-length strokes


# ---------------------------------------------------------------------------
# Legacy alias contract — old projects must keep loading.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias,target", list(_LEGACY_ALIASES.items()))
def test_legacy_alias_resolves_to_real_font(alias, target):
    assert resolve_name(alias) == target
    # And produces non-empty output for a basic letter
    _l, _r, strokes = glyph_strokes("A", alias)
    assert len(strokes) >= 1


def test_unknown_font_falls_back_to_default():
    # get_entry should never raise on unknown input — it falls back so old
    # projects referencing a deleted font keep opening.
    entry = get_entry("ThisFontDoesNotExist")
    assert entry.name == DEFAULT_FONT_NAME


# ---------------------------------------------------------------------------
# FONTS proxy — preserves the dict-like API of the old hand-coded module.
# ---------------------------------------------------------------------------


def test_fonts_proxy_membership():
    assert "EMSReadability" in FONTS
    assert "Simplex" in FONTS  # legacy alias
    assert "Bogus" not in FONTS


def test_fonts_proxy_iteration_includes_all_canonical_names():
    names = list(iter(FONTS))
    assert "EMSReadability" in names
    assert "HersheySans1" in names


# ---------------------------------------------------------------------------
# choices_for_param helper — what the UI dropdowns consume.
# ---------------------------------------------------------------------------


def test_choices_for_param_includes_default_and_legacy():
    choices, descriptions = choices_for_param()
    assert DEFAULT_FONT_NAME in choices
    for legacy in _LEGACY_ALIASES:
        assert legacy in choices
        assert "Legacy alias" in descriptions[legacy]


def test_choices_for_param_descriptions_cover_every_choice():
    choices, descriptions = choices_for_param()
    for name in choices:
        assert name in descriptions, f"{name} missing description"


def test_choices_for_param_can_exclude_legacy():
    choices, _ = choices_for_param(include_legacy_aliases=False)
    for legacy in _LEGACY_ALIASES:
        assert legacy not in choices


# ---------------------------------------------------------------------------
# Specific feature checks driven by the upgrade goals.
# ---------------------------------------------------------------------------


def test_default_font_has_degree_sign():
    """``calibration.py`` switched away from a ``"d"`` hack to a real °.

    Guard against future font swaps that lose the glyph.
    """
    font = load_font(DEFAULT_FONT_NAME)
    assert "°" in font.glyphs


def test_default_font_has_latin1_diacritics():
    """The map labels caller relies on EMSReadability for international names."""
    font = load_font(DEFAULT_FONT_NAME)
    for ch in "äöüß":
        assert ch in font.glyphs, f"missing diacritic: {ch}"


def test_shim_glyph_strokes_uses_legacy_cap_height():
    """The shim scales every font into the 21-unit Hershey system used by the
    existing callers (``text.py`` divides ``font_size_mm`` by ``CAP_HEIGHT``).

    Real typefaces routinely overshoot cap-height for visual balance — the
    apex of an ``A`` may sit a few units above 21 — so the legitimate
    ceiling is the font's *ascent*, not its cap-height.  ``EMSReadability``
    declares ascent = 800 units / cap_height 500, so the apex can reach
    ``21 * 800/500 ≈ 33.6`` in shim units.
    """
    font = load_font("EMSReadability")
    ascent_in_legacy_units = CAP_HEIGHT * (font.metrics.ascent / font.metrics.cap_height)

    _l, _r, strokes = glyph_strokes("A", "EMSReadability")
    ys = [py for poly in strokes for _, py in poly]
    assert ys, "no strokes returned for A"
    assert max(ys) <= ascent_in_legacy_units + 0.1
    assert min(ys) >= -2.0  # A has no descender


def test_shim_returns_centred_bearings():
    """Old convention: left < 0 < right, with the pen at the glyph centre."""
    left, right, _ = glyph_strokes("H", "EMSReadability")
    assert left < 0
    assert right > 0
    assert abs(left + right) < 1e-6  # symmetric around origin
