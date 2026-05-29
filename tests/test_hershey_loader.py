"""Tests for the SVG-font parser at :mod:`plottter.fonts.hershey.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from plottter.fonts.hershey.loader import Font, _parse_path, load_svg_font


# ---------------------------------------------------------------------------
# _parse_path — the path-tokeniser is the part most likely to break on
# third-party fonts, so cover its edge cases directly.
# ---------------------------------------------------------------------------


def test_parse_path_empty_returns_no_polylines():
    assert _parse_path("") == ()
    assert _parse_path("   ") == ()


def test_parse_path_single_stroke():
    polys = _parse_path("M 0 0 L 10 0 L 10 10")
    assert polys == (((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),)


def test_parse_path_pen_up_splits_strokes():
    polys = _parse_path("M 0 0 L 5 5 M 10 10 L 15 15")
    assert polys == (
        ((0.0, 0.0), (5.0, 5.0)),
        ((10.0, 10.0), (15.0, 15.0)),
    )


def test_parse_path_drops_single_point_subpaths():
    # An unterminated M (no following L) would render as a zero-length stroke;
    # the parser must drop it.
    polys = _parse_path("M 0 0 M 10 10 L 20 20")
    assert polys == (((10.0, 10.0), (20.0, 20.0)),)


def test_parse_path_implicit_repeats_after_m():
    # SVG path grammar: coords after an M are implicitly Ls.
    polys = _parse_path("M 0 0 10 0 20 0")
    assert polys == (((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),)


def test_parse_path_negative_and_float_coords():
    polys = _parse_path("M -1.5 2.25 L 3 -4.5")
    assert polys == (((-1.5, 2.25), (3.0, -4.5)),)


def test_parse_path_relative_lowercase_commands():
    # m / l are relative; we never emit them in the vendored fonts but the
    # parser should tolerate them for user-supplied SVGs.
    polys = _parse_path("M 10 10 l 5 0 l 0 5")
    assert polys == (((10.0, 10.0), (15.0, 10.0), (15.0, 15.0)),)


# ---------------------------------------------------------------------------
# load_svg_font — exercise against a small inline fixture so the test
# stays valid even if the vendored data changes.
# ---------------------------------------------------------------------------


_FIXTURE_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <font horiz-adv-x="500" id="Tiny">
      <font-face font-family="Tiny" units-per-em="1000"
                 ascent="800" descent="-200"
                 cap-height="500" x-height="300"/>
      <missing-glyph horiz-adv-x="500"/>
      <glyph unicode=" " glyph-name="space" horiz-adv-x="250"/>
      <glyph unicode="A" glyph-name="A" horiz-adv-x="400"
             d="M 0 0 L 200 500 L 400 0 M 100 200 L 300 200"/>
      <glyph unicode="?" glyph-name="question" horiz-adv-x="300"
             d="M 0 400 L 200 400 L 200 200 L 100 200 M 100 50 L 100 0"/>
    </font>
  </defs>
</svg>
"""


@pytest.fixture
def tiny_font(tmp_path) -> Font:
    p = tmp_path / "Tiny.svg"
    p.write_text(_FIXTURE_SVG)
    return load_svg_font(p)


def test_load_metrics(tiny_font):
    m = tiny_font.metrics
    assert m.units_per_em == 1000.0
    assert m.ascent == 800.0
    assert m.descent == -200.0
    assert m.cap_height == 500.0
    assert m.x_height == 300.0
    assert m.default_advance == 500.0


def test_load_glyphs(tiny_font):
    assert "A" in tiny_font.glyphs
    assert "?" in tiny_font.glyphs
    # Space synthesised from the empty-d glyph
    assert " " in tiny_font.glyphs
    assert tiny_font.glyphs[" "].advance == 250.0
    assert tiny_font.glyphs[" "].strokes == ()


def test_glyph_A_has_two_strokes(tiny_font):
    g = tiny_font.glyphs["A"]
    assert g.advance == 400.0
    # Triangle outline + cross-bar = 2 strokes
    assert len(g.strokes) == 2
    assert g.strokes[0] == ((0.0, 0.0), (200.0, 500.0), (400.0, 0.0))
    assert g.strokes[1] == ((100.0, 200.0), (300.0, 200.0))


def test_glyph_lookup_misses_return_none(tiny_font):
    assert tiny_font.glyph("Q") is None


def test_load_synthesises_space_when_absent(tmp_path):
    # Remove the explicit space glyph and confirm a synthetic one appears.
    src = _FIXTURE_SVG.replace(
        '<glyph unicode=" " glyph-name="space" horiz-adv-x="250"/>', ""
    )
    p = tmp_path / "NoSpace.svg"
    p.write_text(src)
    font = load_svg_font(p)
    assert " " in font.glyphs
    # Falls back to the font's default horiz-adv-x
    assert font.glyphs[" "].advance == font.metrics.default_advance


def test_load_rejects_font_with_no_glyphs(tmp_path):
    p = tmp_path / "Empty.svg"
    p.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg">
        <defs><font><font-face units-per-em="1000" cap-height="500"/></font></defs>
        </svg>"""
    )
    with pytest.raises(ValueError, match="no <glyph"):
        load_svg_font(p)


def test_load_rejects_font_face_missing(tmp_path):
    p = tmp_path / "NoFace.svg"
    p.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg">
        <defs><font><glyph unicode="A" d="M 0 0 L 1 1"/></font></defs>
        </svg>"""
    )
    with pytest.raises(ValueError, match="font-face"):
        load_svg_font(p)


def test_load_uses_stem_as_default_name(tmp_path):
    p = tmp_path / "MyFont.svg"
    p.write_text(_FIXTURE_SVG)
    font = load_svg_font(p)
    assert font.name == "MyFont"
    assert font.family == "Tiny"
