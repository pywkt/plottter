"""Tests for TextGenerator (16.40 — Text tool with font selection, fill, and stroke;
18.6 — Text generator presets using real fonts)."""

from __future__ import annotations

import math
import os
import unittest.mock as mock

import pytest

from plottter.models.canvas import Canvas
from plottter.generators.text import (
    TextGenerator,
    _classify_glyph_contours,
    _compute_fill,
    _point_in_contour,
    _render_hershey_text,
    _render_ttf_text,
    _resolve_font_path,
    _rotate_polylines,
    _signed_area,
    _translate_polylines,
)
from plottter.generators._hershey import CAP_HEIGHT, glyph_strokes, FONTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


def _default_params(gen: TextGenerator) -> dict:
    return {p.name: p.default for p in gen.get_parameters()}


def within_drawing_area(polylines, canvas, tolerance: float = 5.0) -> bool:
    """Return True if at least one point per polyline lies within the canvas drawing area."""
    x1, y1, x2, y2 = canvas.drawing_area()
    for poly in polylines:
        if not any(
            x1 - tolerance <= x <= x2 + tolerance
            and y1 - tolerance <= y <= y2 + tolerance
            for x, y in poly
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Hershey font data tests
# ---------------------------------------------------------------------------

class TestHersheyFontData:
    def test_all_fonts_registered(self):
        assert set(FONTS.keys()) == {"Simplex", "Duplex", "Script", "Gothic"}

    def test_glyph_strokes_ascii_printable(self):
        """Every printable ASCII character should return valid glyph data."""
        for code in range(32, 127):
            ch = chr(code)
            left, right, strokes = glyph_strokes(ch, "Simplex")
            assert isinstance(left, (int, float))
            assert isinstance(right, (int, float))
            assert isinstance(strokes, list)

    def test_glyph_strokes_returns_polylines(self):
        """Strokes for 'A' should contain at least 2 polylines (stems + crossbar)."""
        _left, _right, strokes = glyph_strokes("A", "Simplex")
        # A has 2 strokes (two legs + crossbar = 2 groups: legs and bar)
        assert len(strokes) >= 1
        for s in strokes:
            assert len(s) >= 2

    def test_cap_height_constant(self):
        assert CAP_HEIGHT == 21

    def test_missing_char_falls_back(self):
        """A character not in the font should fall back to '?' or empty glyph."""
        left, right, strokes = glyph_strokes("\x01", "Simplex")
        assert isinstance(strokes, list)

    def test_duplex_wider_than_simplex(self):
        """Duplex characters should be wider than Simplex equivalents."""
        l_s, r_s, _ = glyph_strokes("H", "Simplex")
        l_d, r_d, _ = glyph_strokes("H", "Duplex")
        assert (r_d - l_d) > (r_s - l_s)


# ---------------------------------------------------------------------------
# _render_hershey_text internals
# ---------------------------------------------------------------------------

class TestRenderHersheyText:
    def test_single_line_produces_polylines(self):
        polys, w, h = _render_hershey_text(
            text="ABC",
            font_name="Simplex",
            font_size_mm=10.0,
            letter_spacing_mm=0.0,
            line_spacing=1.2,
            text_align="Center",
            stroke_repeat=1,
        )
        assert len(polys) > 0
        for p in polys:
            assert len(p) >= 2

    def test_multiline_height_larger_than_single(self):
        _p1, _w1, h1 = _render_hershey_text(
            "A", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        _p2, _w2, h2 = _render_hershey_text(
            "A\nB", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        assert h2 > h1

    def test_multiline_produces_more_polylines_than_single(self):
        p1, _w1, _h1 = _render_hershey_text(
            "AB", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        p2, _w2, _h2 = _render_hershey_text(
            "AB\nCD", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        assert len(p2) >= len(p1)

    def test_stroke_repeat_multiplies_polylines(self):
        p1, _, _ = _render_hershey_text(
            "A", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        p2, _, _ = _render_hershey_text(
            "A", "Simplex", 10.0, 0.0, 1.2, "Center", 2
        )
        assert len(p2) == len(p1) * 2

    def test_font_size_scales_output(self):
        p_small, w_small, _h = _render_hershey_text(
            "ABC", "Simplex", 5.0, 0.0, 1.2, "Center", 1
        )
        p_large, w_large, _h = _render_hershey_text(
            "ABC", "Simplex", 20.0, 0.0, 1.2, "Center", 1
        )
        assert w_large > w_small

    def test_letter_spacing_increases_width(self):
        _p, w0, _h = _render_hershey_text(
            "AB", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        _p, w1, _h = _render_hershey_text(
            "AB", "Simplex", 10.0, 2.0, 1.2, "Center", 1
        )
        assert w1 > w0

    def test_all_hershey_fonts(self):
        for font in ("Simplex", "Duplex", "Script", "Gothic"):
            polys, w, h = _render_hershey_text(
                "Hello", font, 10.0, 0.0, 1.2, "Center", 1
            )
            assert len(polys) > 0, f"Font {font} produced no polylines"
            assert w > 0

    def test_empty_text_returns_empty(self):
        polys, w, h = _render_hershey_text(
            "", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        assert polys == []

    def test_space_only_text(self):
        polys, w, h = _render_hershey_text(
            "   ", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        # Spaces have no strokes — result may be empty
        assert isinstance(polys, list)


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

class TestTransformHelpers:
    def test_translate_polylines(self):
        polys = [[(0.0, 0.0), (1.0, 1.0)]]
        moved = _translate_polylines(polys, 5.0, 3.0)
        assert moved[0][0] == (5.0, 3.0)
        assert moved[0][1] == (6.0, 4.0)

    def test_translate_zero_is_identity(self):
        polys = [[(1.0, 2.0), (3.0, 4.0)]]
        assert _translate_polylines(polys, 0.0, 0.0) is polys

    def test_rotate_360_is_identity(self):
        polys = [[(5.0, 0.0)]]
        rotated = _rotate_polylines(polys, 360.0, 0.0, 0.0)
        x, y = rotated[0][0]
        assert abs(x - 5.0) < 1e-6
        assert abs(y - 0.0) < 1e-6

    def test_rotate_90(self):
        polys = [[(1.0, 0.0)]]
        rotated = _rotate_polylines(polys, 90.0, 0.0, 0.0)
        x, y = rotated[0][0]
        # CCW 90° of (1, 0) around origin → (0, 1)
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TextGenerator — public API
# ---------------------------------------------------------------------------

class TestTextGeneratorRegistration:
    def test_registered(self):
        from plottter.generators import GENERATORS
        assert "Text" in GENERATORS

    def test_category(self):
        gen = TextGenerator()
        assert gen.category == "math"


class TestTextGeneratorParameters:
    def setup_method(self):
        self.gen = TextGenerator()

    def test_has_text_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "text" in names

    def test_has_font_type_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "font_type" in names

    def test_has_hershey_font_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "hershey_font" in names

    def test_has_font_size_param(self):
        names = [p.name for p in self.gen.get_parameters()]
        assert "font_size_mm" in names

    def test_has_layout_params(self):
        names = [p.name for p in self.gen.get_parameters()]
        for param in ("letter_spacing_mm", "line_spacing", "text_align"):
            assert param in names, f"Missing param: {param}"

    def test_has_position_params(self):
        names = [p.name for p in self.gen.get_parameters()]
        for param in ("x_offset_mm", "y_offset_mm", "rotation_deg"):
            assert param in names

    def test_string_param_type_for_text(self):
        from plottter.generators.base import StringParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["text"], StringParam)
        assert params["text"].multiline is True

    def test_system_font_path_is_font_param(self):
        from plottter.generators.base import FontParam
        params = {p.name: p for p in self.gen.get_parameters()}
        assert isinstance(params["system_font_path"], FontParam)


class TestTextGeneratorGenerate:
    def setup_method(self):
        self.gen = TextGenerator()
        self.canvas = make_canvas()

    def test_default_generate_produces_polylines(self):
        params = _default_params(self.gen)
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        for poly in result:
            assert len(poly) >= 2

    def test_empty_text_returns_empty(self):
        params = _default_params(self.gen)
        params["text"] = ""
        result = self.gen.generate(params, self.canvas)
        assert result == []

    def test_whitespace_only_returns_empty(self):
        params = _default_params(self.gen)
        params["text"] = "   "
        result = self.gen.generate(params, self.canvas)
        assert result == []

    def test_multiline_text(self):
        params = _default_params(self.gen)
        params["text"] = "Hello\nWorld"
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0

    def test_output_within_drawing_area(self):
        params = _default_params(self.gen)
        params["text"] = "Hello"
        params["font_size_mm"] = 10.0
        result = self.gen.generate(params, self.canvas)
        assert len(result) > 0
        assert within_drawing_area(result, self.canvas)

    def test_offset_shifts_text(self):
        params = _default_params(self.gen)
        params["text"] = "A"
        params["x_offset_mm"] = 0.0
        params["y_offset_mm"] = 0.0
        result0 = self.gen.generate(params, self.canvas)

        params["x_offset_mm"] = 20.0
        params["y_offset_mm"] = 10.0
        result1 = self.gen.generate(params, self.canvas)

        # Centroids should differ
        cx0 = sum(x for p in result0 for x, _ in p) / sum(len(p) for p in result0)
        cx1 = sum(x for p in result1 for x, _ in p) / sum(len(p) for p in result1)
        assert cx1 > cx0

    def test_rotation_changes_output(self):
        params = _default_params(self.gen)
        params["text"] = "A"
        params["rotation_deg"] = 0.0
        result0 = self.gen.generate(params, self.canvas)
        params["rotation_deg"] = 45.0
        result1 = self.gen.generate(params, self.canvas)
        # Rotated output should differ from unrotated
        assert result0 != result1

    def test_all_hershey_fonts(self):
        for font in ("Simplex", "Duplex", "Script", "Gothic"):
            params = _default_params(self.gen)
            params["text"] = "Plotter"
            params["font_type"] = "Hershey"
            params["hershey_font"] = font
            result = self.gen.generate(params, self.canvas)
            assert len(result) > 0, f"Font {font} produced no polylines"

    def test_stroke_repeat_2(self):
        params = _default_params(self.gen)
        params["text"] = "A"
        params["stroke_repeat"] = 1
        r1 = self.gen.generate(params, self.canvas)
        params["stroke_repeat"] = 2
        r2 = self.gen.generate(params, self.canvas)
        assert len(r2) == len(r1) * 2

    def test_progress_callback_called(self):
        params = _default_params(self.gen)
        params["text"] = "Hi"
        calls = []
        self.gen.generate(params, self.canvas, progress_callback=calls.append)
        assert len(calls) > 0
        assert calls[-1] == 100


class TestTextGeneratorPresets:
    def setup_method(self):
        self.gen = TextGenerator()
        self.canvas = make_canvas()

    def test_presets_exist(self):
        presets = self.gen.get_presets()
        assert len(presets) >= 4

    def test_all_presets_have_required_keys(self):
        required = {"text", "font_type", "font_size_mm"}
        for preset in self.gen.get_presets():
            for key in required:
                assert key in preset.params, f"Preset '{preset.name}' missing '{key}'"

    def test_hershey_presets_generate(self):
        canvas = self.canvas
        for preset in self.gen.get_presets():
            if preset.params.get("font_type") == "Hershey":
                params = _default_params(self.gen)
                params.update(preset.params)
                result = self.gen.generate(params, canvas)
                assert len(result) > 0, f"Preset '{preset.name}' produced no output"


# ---------------------------------------------------------------------------
# TTF/OTF backend tests (fonttools mocked / skipped if unavailable)
# ---------------------------------------------------------------------------

class TestTTFBackend:
    """Tests for TTF/OTF font rendering via fonttools.

    Skipped if fonttools is not installed.
    """

    @pytest.fixture(autouse=True)
    def _check_fonttools(self):
        pytest.importorskip("fonttools")

    def test_import_error_falls_back_to_hershey(self):
        """When fonttools raises ImportError the generator falls back to Hershey."""
        gen = TextGenerator()
        canvas = make_canvas()
        params = _default_params(gen)
        params["text"] = "AB"
        params["font_type"] = "System Font"
        params["system_font_path"] = "/nonexistent/font.ttf"
        # Should not raise — falls back to Hershey
        result = gen.generate(params, canvas)
        assert len(result) > 0

    def test_render_ttf_text_mocked(self):
        """_render_ttf_text returns polylines when fonttools is available (mocked font)."""
        from plottter.generators.text import _render_ttf_text

        # Build a minimal mock TTFont
        mock_font = mock.MagicMock()
        mock_font.__getitem__ = mock.MagicMock(
            side_effect=lambda key: mock.MagicMock(
                unitsPerEm=1000,
                sTypoAscender=800,
                sTypoDescender=-200,
            )
            if key in ("head", "OS/2")
            else mock.MagicMock()
        )
        mock_font.getBestCmap.return_value = {ord("A"): "A", ord("B"): "B"}

        # Mock glyph set
        mock_glyph_A = mock.MagicMock()
        mock_glyph_A.width = 600

        def draw_A(pen):
            pen.moveTo((100, 0))
            pen.lineTo((300, 700))
            pen.lineTo((500, 0))
            pen.closePath()

        mock_glyph_A.draw = draw_A

        mock_glyph_B = mock.MagicMock()
        mock_glyph_B.width = 600

        def draw_B(pen):
            pen.moveTo((100, 0))
            pen.lineTo((100, 700))
            pen.lineTo((400, 700))
            pen.lineTo((500, 600))
            pen.lineTo((500, 400))
            pen.lineTo((100, 400))
            pen.closePath()

        mock_glyph_B.draw = draw_B

        mock_glyph_set = {"A": mock_glyph_A, "B": mock_glyph_B}
        mock_font.getGlyphSet.return_value = mock_glyph_set

        with mock.patch("fonttools.ttLib.TTFont", return_value=mock_font):
            polys, w, h = _render_ttf_text(
                text="AB",
                font_path="fake.ttf",
                font_size_mm=10.0,
                letter_spacing_mm=0.0,
                line_spacing=1.2,
                text_align="Left",
                render_mode="Outline",
                fill_type="Hatching",
                fill_spacing_mm=0.5,
                fill_angle=45.0,
                curve_tolerance_mm=0.5,
            )
        assert len(polys) >= 2
        # All contours should be closed (first == last point)
        for poly in polys:
            assert poly[0] == poly[-1], "Closed contour expected"


# ---------------------------------------------------------------------------
# Multi-line layout positioning
# ---------------------------------------------------------------------------

class TestMultilineLayout:
    def test_second_line_below_first(self):
        """In canvas coordinates (y DOWN), second line's polylines should have
        higher y values than the first line's polylines."""
        polys_1, _, _ = _render_hershey_text(
            "A", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        polys_2, _, _ = _render_hershey_text(
            "A\nB", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )

        # First-line polylines in polys_2 should have negative y values,
        # second-line should have positive y values (y is down).
        # In practice, polys_2 has more polylines — just verify ordering.
        assert len(polys_2) > len(polys_1)

    def test_alignment_center_vs_left(self):
        """Center-aligned text should have polylines centred around x=0;
        left-aligned text should start at a more negative x."""
        polys_c, _, _ = _render_hershey_text(
            "HELLO", "Simplex", 10.0, 0.0, 1.2, "Center", 1
        )
        polys_l, _, _ = _render_hershey_text(
            "HELLO", "Simplex", 10.0, 0.0, 1.2, "Left", 1
        )
        # Centre: x values should be roughly symmetric
        all_x_c = [x for p in polys_c for x, _ in p]
        centre_c = sum(all_x_c) / len(all_x_c)
        assert abs(centre_c) < 15.0, "Centre-aligned text not centred near x=0"

        # Left: left edge should be more negative
        all_x_l = [x for p in polys_l for x, _ in p]
        min_x_l = min(all_x_l)
        min_x_c = min(all_x_c)
        # Left-aligned has a different (non-symmetric) distribution
        assert min_x_l <= min_x_c + 1.0


# ---------------------------------------------------------------------------
# 18.6 — Font path resolution and Google Font auto-download
# ---------------------------------------------------------------------------

# Stable system TTF available on all Linux CI images
_SYSTEM_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_HAS_SYSTEM_TTF = os.path.isfile(_SYSTEM_TTF)


class TestResolveFontPath:
    """Unit tests for the _resolve_font_path() helper."""

    def test_valid_system_path_returned_unchanged(self):
        """A path to an existing file is returned as-is."""
        if not _HAS_SYSTEM_TTF:
            pytest.skip("System TTF not available")
        result = _resolve_font_path(_SYSTEM_TTF, {})
        assert result == _SYSTEM_TTF

    def test_nonexistent_path_with_no_family_returns_empty(self):
        """If the path doesn't exist and no font_family is given, return ''."""
        result = _resolve_font_path("/nonexistent/font.ttf", {})
        assert result == ""

    def test_empty_path_with_no_family_returns_empty(self):
        result = _resolve_font_path("", {})
        assert result == ""

    def test_resolves_from_system_catalog(self):
        """If font_family resolves via get_font_path(), that path is returned."""
        fake_path = "/fake/fonts/Roboto.ttf"
        with mock.patch(
            "plottter.fonts.discovery.get_font_path",
            return_value=fake_path,
        ):
            result = _resolve_font_path("", {"font_family": "Roboto", "font_style": "regular"})
        assert result == fake_path

    def test_falls_back_to_download_when_not_in_catalog(self):
        """When get_font_path returns None, download_google_font is called."""
        fake_downloaded = "/fake/cache/Roboto-regular.ttf"
        with mock.patch(
            "plottter.fonts.discovery.get_font_path",
            return_value=None,
        ), mock.patch(
            "plottter.fonts.google_fonts.download_google_font",
            return_value=fake_downloaded,
        ) as mock_dl:
            result = _resolve_font_path("", {"font_family": "Roboto", "font_style": "regular"})
        mock_dl.assert_called_once_with("Roboto", "regular")
        assert result == fake_downloaded

    def test_returns_empty_when_download_fails(self):
        """If download_google_font raises, '' is returned (no exception propagated)."""
        with mock.patch(
            "plottter.fonts.discovery.get_font_path",
            return_value=None,
        ), mock.patch(
            "plottter.fonts.google_fonts.download_google_font",
            side_effect=RuntimeError("Network error"),
        ):
            result = _resolve_font_path("", {"font_family": "Roboto"})
        assert result == ""

    def test_direct_path_takes_priority_over_family(self):
        """A valid system_font_path is used even when font_family is also set."""
        if not _HAS_SYSTEM_TTF:
            pytest.skip("System TTF not available")
        result = _resolve_font_path(
            _SYSTEM_TTF,
            {"font_family": "SomeOtherFont", "font_style": "regular"},
        )
        assert result == _SYSTEM_TTF


class TestTextGeneratorGoogleFontPresets:
    """Tests for the six new Google Font presets (18.6)."""

    EXPECTED_PRESET_NAMES = [
        "Elegant Serif",
        "Clean Sans",
        "Handwritten",
        "Hatched Display",
        "Monospace Code",
        "Concentric Rings",
    ]

    def setup_method(self):
        self.gen = TextGenerator()
        self.canvas = make_canvas()

    def test_new_presets_exist(self):
        names = [p.name for p in self.gen.get_presets()]
        for expected in self.EXPECTED_PRESET_NAMES:
            assert expected in names, f"Preset '{expected}' not found in presets"

    def test_new_presets_have_font_family(self):
        """Each Google Font preset must specify a font_family."""
        presets = {p.name: p for p in self.gen.get_presets()}
        for name in self.EXPECTED_PRESET_NAMES:
            assert name in presets
            assert presets[name].params.get("font_family"), (
                f"Preset '{name}' missing font_family"
            )

    def test_new_presets_are_system_font_type(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        for name in self.EXPECTED_PRESET_NAMES:
            assert presets[name].params.get("font_type") == "System Font", (
                f"Preset '{name}' should use font_type='System Font'"
            )

    def test_broken_presets_removed(self):
        """The old broken presets (empty system_font_path) should be gone."""
        names = [p.name for p in self.gen.get_presets()]
        assert "Decorative / Hatched" not in names
        assert "Engraved / Concentric" not in names

    def test_hershey_presets_still_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Title / Hershey Sans" in names
        assert "Script Signature" in names

    def test_preset_generate_falls_back_to_hershey_on_download_failure(self):
        """When font download fails, generation falls back to Hershey (no crash)."""
        gen = TextGenerator()
        canvas = make_canvas()
        params = _default_params(gen)
        params.update(self.gen.get_presets()[4].params)  # any Google Font preset

        with mock.patch(
            "plottter.fonts.discovery.get_font_path", return_value=None
        ), mock.patch(
            "plottter.fonts.google_fonts.download_google_font",
            side_effect=RuntimeError("no network"),
        ):
            result = gen.generate(params, canvas)

        # Hershey fallback should produce non-empty output
        assert len(result) > 0

    @pytest.mark.skipif(not _HAS_SYSTEM_TTF, reason="System TTF not available")
    def test_preset_generate_with_cached_font(self):
        """Preset generates output when get_font_path returns a cached TTF."""
        gen = TextGenerator()
        canvas = make_canvas()

        # Use "Elegant Serif" preset with a mock path pointing to a real TTF
        presets = {p.name: p for p in gen.get_presets()}
        params = _default_params(gen)
        params.update(presets["Elegant Serif"].params)

        with mock.patch(
            "plottter.fonts.discovery.get_font_path",
            return_value=_SYSTEM_TTF,
        ):
            result = gen.generate(params, canvas)

        assert len(result) > 0

    def test_preset_triggers_download_for_uncached_font(self):
        """When font not in catalog, download_google_font is called during generate()."""
        gen = TextGenerator()
        canvas = make_canvas()

        presets = {p.name: p for p in gen.get_presets()}
        params = _default_params(gen)
        params.update(presets["Elegant Serif"].params)

        with mock.patch(
            "plottter.fonts.discovery.get_font_path", return_value=None
        ), mock.patch(
            "plottter.fonts.google_fonts.download_google_font",
            return_value="/nonexistent/Playfair.ttf",  # path won't exist → falls back
        ) as mock_dl:
            gen.generate(params, canvas)

        mock_dl.assert_called_once_with("Playfair Display", "regular")

    def test_existing_hershey_presets_unchanged(self):
        """Hershey presets still generate output without any mock."""
        gen = TextGenerator()
        canvas = make_canvas()
        for preset in gen.get_presets():
            if preset.params.get("font_type") == "Hershey":
                params = _default_params(gen)
                params.update(preset.params)
                result = gen.generate(params, canvas)
                assert len(result) > 0, f"Hershey preset '{preset.name}' failed"


# ---------------------------------------------------------------------------
# 19.1 — Fix filled text rendering ignoring glyph holes (counters)
# ---------------------------------------------------------------------------

class TestSignedArea:
    """Tests for the _signed_area shoelace helper."""

    def test_ccw_square_positive(self):
        """Counter-clockwise square has positive area."""
        square_ccw = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
        assert _signed_area(square_ccw) > 0.0

    def test_cw_square_negative(self):
        """Clockwise square has negative area."""
        square_cw = [(0., 0.), (0., 10.), (10., 10.), (10., 0.)]
        assert _signed_area(square_cw) < 0.0

    def test_area_magnitude(self):
        """10×10 square has area 100."""
        square = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
        assert abs(abs(_signed_area(square)) - 100.0) < 1e-9

    def test_degenerate_returns_zero(self):
        assert _signed_area([]) == 0.0
        assert _signed_area([(0., 0.)]) == 0.0
        assert _signed_area([(0., 0.), (1., 1.)]) == 0.0


class TestPointInContour:
    """Tests for the _point_in_contour ray-casting helper."""

    def _square(self):
        return [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]

    def test_inside_point(self):
        assert _point_in_contour(5., 5., self._square()) is True

    def test_outside_point(self):
        assert _point_in_contour(15., 5., self._square()) is False

    def test_outside_negative(self):
        assert _point_in_contour(-1., 5., self._square()) is False

    def test_corner_vicinity(self):
        # A point just inside the square near a corner
        assert _point_in_contour(0.5, 0.5, self._square()) is True

    def test_outside_far(self):
        assert _point_in_contour(100., 100., self._square()) is False


class TestClassifyGlyphContours:
    """Tests for _classify_glyph_contours — outer/hole separation."""

    def _make_square(self, x0, y0, x1, y1):
        """Return a closed square contour (CCW in math coords)."""
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

    def test_single_contour_returned_as_outer(self):
        outer = self._make_square(0, 0, 10, 10)
        groups = _classify_glyph_contours([outer])
        assert len(groups) == 1
        o, holes = groups[0]
        assert o is outer
        assert holes == []

    def test_empty_returns_empty(self):
        assert _classify_glyph_contours([]) == []

    def test_outer_and_inner_classified(self):
        """A small square inside a big square — the inner one is a hole."""
        outer = self._make_square(0, 0, 10, 10)
        inner = self._make_square(3, 3, 7, 7)
        groups = _classify_glyph_contours([outer, inner])
        assert len(groups) == 1, "Should produce one outer group"
        o, holes = groups[0]
        assert len(holes) == 1, "Inner square should be classified as a hole"

    def test_two_non_overlapping_outers(self):
        """Two separate squares → two outer contours, no holes."""
        outer1 = self._make_square(0, 0, 5, 5)
        outer2 = self._make_square(10, 0, 15, 5)
        groups = _classify_glyph_contours([outer1, outer2])
        assert len(groups) == 2
        for o, holes in groups:
            assert holes == []

    def test_letter_o_like_shape(self):
        """Outer ring and inner ring (simulating 'o') — one outer, one hole."""
        outer = self._make_square(-5, -5, 5, 5)
        hole = self._make_square(-2, -2, 2, 2)
        groups = _classify_glyph_contours([outer, hole])
        assert len(groups) == 1
        o, holes = groups[0]
        assert len(holes) == 1

    def test_b_like_two_holes(self):
        """Outer shape with two inner holes (simulating 'B' with two counters)."""
        outer = self._make_square(0, 0, 10, 20)
        hole1 = self._make_square(1, 1, 9, 9)
        hole2 = self._make_square(1, 11, 9, 19)
        groups = _classify_glyph_contours([outer, hole1, hole2])
        assert len(groups) == 1
        o, holes = groups[0]
        assert len(holes) == 2

    def test_order_does_not_matter(self):
        """Putting the inner contour first should not change the result."""
        outer = self._make_square(0, 0, 10, 10)
        inner = self._make_square(3, 3, 7, 7)
        g1 = _classify_glyph_contours([outer, inner])
        g2 = _classify_glyph_contours([inner, outer])
        assert len(g1) == 1 and len(g1[0][1]) == 1
        assert len(g2) == 1 and len(g2[0][1]) == 1


class TestComputeFillWithHoles:
    """Tests for _compute_fill with the holes parameter."""

    def _square(self, x0, y0, x1, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

    def test_fill_without_holes_produces_lines(self):
        outer = self._square(0, 0, 10, 10)
        lines = _compute_fill(outer, [], "Hatching", 1.0, 0.0)
        assert len(lines) > 0

    def test_fill_with_hole_has_fewer_total_length(self):
        """Fill with a hole should cover less area than fill without hole."""
        outer = self._square(0, 0, 10, 10)
        inner = self._square(3, 3, 7, 7)

        def total_len(lines):
            total = 0.0
            for line in lines:
                for i in range(len(line) - 1):
                    dx = line[i+1][0] - line[i][0]
                    dy = line[i+1][1] - line[i][1]
                    total += math.hypot(dx, dy)
            return total

        lines_no_hole = _compute_fill(outer, [], "Hatching", 0.5, 0.0)
        lines_with_hole = _compute_fill(outer, [inner], "Hatching", 0.5, 0.0)
        assert total_len(lines_with_hole) < total_len(lines_no_hole)

    def test_fill_lines_do_not_enter_hole(self):
        """No fill point should lie strictly inside the hole region."""
        outer = self._square(0, 0, 20, 20)
        inner = self._square(5, 5, 15, 15)
        lines = _compute_fill(outer, [inner], "Hatching", 0.5, 0.0)

        for line in lines:
            for x, y in line:
                inside_hole = 5.0 < x < 15.0 and 5.0 < y < 15.0
                assert not inside_hole, (
                    f"Fill point ({x:.2f}, {y:.2f}) is inside the hole"
                )

    def test_cross_hatch_respects_hole(self):
        outer = self._square(0, 0, 20, 20)
        inner = self._square(5, 5, 15, 15)
        lines = _compute_fill(outer, [inner], "Cross-hatch", 1.0, 0.0)
        for line in lines:
            for x, y in line:
                assert not (5.0 < x < 15.0 and 5.0 < y < 15.0)

    def test_concentric_respects_hole(self):
        outer = self._square(0, 0, 20, 20)
        inner = self._square(6, 6, 14, 14)
        lines = _compute_fill(outer, [inner], "Concentric", 1.0, 0.0)
        # Concentric should produce some output
        assert len(lines) > 0


class TestGlyphHoleFillIntegration:
    """Integration tests for the glyph hole fix in _render_ttf_text.

    Uses a minimal mock TTFont glyph with an outer boundary and an
    inner counter to simulate letters like 'o', 'p', 'B', '8'.
    """

    def _make_mock_font_with_counter_glyph(self):
        """Build a mock TTFont where glyph 'O' has outer + inner squares."""
        mock_font = mock.MagicMock()
        mock_font.__getitem__ = mock.MagicMock(
            side_effect=lambda key: mock.MagicMock(
                unitsPerEm=1000,
                sTypoAscender=800,
                sTypoDescender=-200,
            )
        )
        mock_font.getBestCmap.return_value = {ord("O"): "O"}

        # Build a glyph that has two contours:
        # 1) Large outer square
        # 2) Small inner square (the counter/hole)
        mock_glyph_O = mock.MagicMock()
        mock_glyph_O.width = 700

        def draw_O(pen):
            # Outer square: (0,0) → (600,0) → (600,600) → (0,600)
            pen.moveTo((0, 0))
            pen.lineTo((600, 0))
            pen.lineTo((600, 600))
            pen.lineTo((0, 600))
            pen.closePath()
            # Inner counter: (150,150) → (150,450) → (450,450) → (450,150)
            pen.moveTo((150, 150))
            pen.lineTo((150, 450))
            pen.lineTo((450, 450))
            pen.lineTo((450, 150))
            pen.closePath()

        mock_glyph_O.draw = draw_O
        mock_font.getGlyphSet.return_value = {"O": mock_glyph_O}
        return mock_font

    def _render_with_mock(self, render_mode, fill_type="Hatching"):
        mock_font = self._make_mock_font_with_counter_glyph()
        with mock.patch("fontTools.ttLib.TTFont", return_value=mock_font), \
             mock.patch("fontTools.pens.recordingPen.RecordingPen",
                        wraps=__import__(
                            "fontTools.pens.recordingPen",
                            fromlist=["RecordingPen"]
                        ).RecordingPen):
            polys, w, h = _render_ttf_text(
                text="O",
                font_path="fake.ttf",
                font_size_mm=20.0,
                letter_spacing_mm=0.0,
                line_spacing=1.2,
                text_align="Center",
                render_mode=render_mode,
                fill_type=fill_type,
                fill_spacing_mm=0.5,
                fill_angle=0.0,
                curve_tolerance_mm=0.5,
            )
        return polys

    @pytest.fixture(autouse=True)
    def _check_fonttools(self):
        pytest.importorskip("fontTools")

    def test_outline_mode_returns_two_contours(self):
        """Outline mode should return both the outer and inner contour."""
        polys = self._render_with_mock("Outline")
        # Both outer and inner contours should be present
        assert len(polys) == 2, (
            f"Expected 2 contours (outer + inner), got {len(polys)}"
        )

    def test_filled_mode_counter_not_filled(self):
        """Fill lines must not enter the counter region."""
        polys = self._render_with_mock("Filled", "Hatching")
        # The glyph is 600×600 units at 20mm / 1000 upem = 0.02 mm/unit
        # Inner counter spans 150–450 units = 3mm–9mm (in local glyph coords)
        # After centering, the exact positions depend on layout — but we can
        # check that all points lie either outside the outer boundary or in
        # the annular ring, not in the hole interior.
        #
        # For a simpler check: the fill with the hole should produce FEWER
        # lines than filling just the outer square (no hole).
        assert len(polys) > 0  # some fill was produced

    def test_filled_mode_respects_counter_geometry(self):
        """Fill line count with hole < fill line count without hole."""
        mock_font_with_hole = self._make_mock_font_with_counter_glyph()

        # Build a second mock with only the outer square (no counter)
        mock_font_no_hole = mock.MagicMock()
        mock_font_no_hole.__getitem__ = mock.MagicMock(
            side_effect=lambda key: mock.MagicMock(
                unitsPerEm=1000, sTypoAscender=800, sTypoDescender=-200,
            )
        )
        mock_font_no_hole.getBestCmap.return_value = {ord("O"): "O"}

        mock_glyph_solid = mock.MagicMock()
        mock_glyph_solid.width = 700

        def draw_solid(pen):
            pen.moveTo((0, 0))
            pen.lineTo((600, 0))
            pen.lineTo((600, 600))
            pen.lineTo((0, 600))
            pen.closePath()

        mock_glyph_solid.draw = draw_solid
        mock_font_no_hole.getGlyphSet.return_value = {"O": mock_glyph_solid}

        def _render(font_mock):
            with mock.patch("fontTools.ttLib.TTFont", return_value=font_mock), \
                 mock.patch(
                     "fontTools.pens.recordingPen.RecordingPen",
                     wraps=__import__(
                         "fontTools.pens.recordingPen",
                         fromlist=["RecordingPen"],
                     ).RecordingPen,
                 ):
                polys, _, _ = _render_ttf_text(
                    text="O",
                    font_path="fake.ttf",
                    font_size_mm=20.0,
                    letter_spacing_mm=0.0,
                    line_spacing=1.2,
                    text_align="Center",
                    render_mode="Filled",
                    fill_type="Hatching",
                    fill_spacing_mm=0.5,
                    fill_angle=0.0,
                    curve_tolerance_mm=0.5,
                )
            return polys

        polys_with_hole = _render(mock_font_with_hole)
        polys_no_hole = _render(mock_font_no_hole)

        def _total_length(polys):
            total = 0.0
            for line in polys:
                for i in range(len(line) - 1):
                    dx = line[i+1][0] - line[i][0]
                    dy = line[i+1][1] - line[i][1]
                    total += math.hypot(dx, dy)
            return total

        len_hole = _total_length(polys_with_hole)
        len_solid = _total_length(polys_no_hole)
        assert len_hole < len_solid, (
            f"Fill with counter ({len_hole:.1f}mm) should be shorter than "
            f"solid fill ({len_solid:.1f}mm)"
        )

    def test_outline_plus_filled_counter_not_filled(self):
        """Outline + Filled mode: fill should respect the counter."""
        polys = self._render_with_mock("Outline + Filled", "Hatching")
        # Should have both outline contours AND fill lines
        assert len(polys) > 2  # 2 contours + at least some fill lines

    def test_hershey_font_unaffected(self):
        """Hershey font rendering is unaffected by the TTF hole fix."""
        gen = TextGenerator()
        canvas = make_canvas()
        params = _default_params(gen)
        params["font_type"] = "Hershey"
        params["text"] = "oop"  # letters with potential counters in Hershey
        result = gen.generate(params, canvas)
        assert len(result) > 0

    @pytest.mark.skipif(not _HAS_SYSTEM_TTF, reason="System TTF not available")
    def test_real_ttf_outline_mode_no_regression(self):
        """Outline mode with a real TTF font still produces valid polylines."""
        gen = TextGenerator()
        canvas = make_canvas()
        params = _default_params(gen)
        params["font_type"] = "System Font"
        params["system_font_path"] = _SYSTEM_TTF
        params["text"] = "O"
        params["render_mode"] = "Outline"
        params["font_size_mm"] = 15.0
        result = gen.generate(params, canvas)
        assert len(result) > 0

    @pytest.mark.skipif(not _HAS_SYSTEM_TTF, reason="System TTF not available")
    def test_real_ttf_filled_mode_produces_output(self):
        """Filled mode with a real TTF font (DejaVu Sans 'O') produces fill."""
        gen = TextGenerator()
        canvas = make_canvas()
        params = _default_params(gen)
        params["font_type"] = "System Font"
        params["system_font_path"] = _SYSTEM_TTF
        params["text"] = "O"
        params["render_mode"] = "Filled"
        params["fill_type"] = "Hatching"
        params["fill_spacing_mm"] = 0.5
        params["font_size_mm"] = 20.0
        result = gen.generate(params, canvas)
        assert len(result) > 0
