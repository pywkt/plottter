"""Tests for ExportDialog — focusing on the extension auto-append behaviour."""

from __future__ import annotations

import pytest

from plottter.gui.dialogs.export import ExportDialog


# ---------------------------------------------------------------------------
# _ensure_extension unit tests (static method — no Qt widgets needed)
# ---------------------------------------------------------------------------

class TestEnsureExtension:
    """Verify ExportDialog._ensure_extension behaviour for every format."""

    # ── SVG ─────────────────────────────────────────────────────────────────

    def test_svg_no_extension_appends_svg(self) -> None:
        assert ExportDialog._ensure_extension("my_drawing", "SVG") == "my_drawing.svg"

    def test_svg_correct_extension_unchanged(self) -> None:
        assert ExportDialog._ensure_extension("my_drawing.svg", "SVG") == "my_drawing.svg"

    def test_svg_correct_extension_uppercase_unchanged(self) -> None:
        assert ExportDialog._ensure_extension("my_drawing.SVG", "SVG") == "my_drawing.SVG"

    def test_svg_wrong_extension_left_alone(self) -> None:
        """A path with a non-SVG extension must NOT be silently overwritten."""
        result = ExportDialog._ensure_extension("my_drawing.png", "SVG")
        assert result == "my_drawing.png"

    # ── HPGL ────────────────────────────────────────────────────────────────

    def test_hpgl_no_extension_appends_plt(self) -> None:
        assert ExportDialog._ensure_extension("output", "HPGL") == "output.plt"

    def test_hpgl_plt_unchanged(self) -> None:
        assert ExportDialog._ensure_extension("output.plt", "HPGL") == "output.plt"

    def test_hpgl_hpgl_unchanged(self) -> None:
        """'.hpgl' is a valid alternate extension and must be preserved."""
        assert ExportDialog._ensure_extension("output.hpgl", "HPGL") == "output.hpgl"

    def test_hpgl_wrong_extension_left_alone(self) -> None:
        result = ExportDialog._ensure_extension("output.svg", "HPGL")
        assert result == "output.svg"

    # ── G-code ──────────────────────────────────────────────────────────────

    def test_gcode_no_extension_appends_gcode(self) -> None:
        assert ExportDialog._ensure_extension("plot", "G-code") == "plot.gcode"

    def test_gcode_gcode_unchanged(self) -> None:
        assert ExportDialog._ensure_extension("plot.gcode", "G-code") == "plot.gcode"

    def test_gcode_nc_unchanged(self) -> None:
        """'.nc' is a valid alternate extension and must be preserved."""
        assert ExportDialog._ensure_extension("plot.nc", "G-code") == "plot.nc"

    def test_gcode_wrong_extension_left_alone(self) -> None:
        result = ExportDialog._ensure_extension("plot.txt", "G-code")
        assert result == "plot.txt"

    # ── Mural ────────────────────────────────────────────────────────────────

    def test_mural_no_extension_appends_mural(self) -> None:
        assert ExportDialog._ensure_extension("wall_art", "Mural") == "wall_art.mural"

    def test_mural_correct_extension_unchanged(self) -> None:
        assert ExportDialog._ensure_extension("wall_art.mural", "Mural") == "wall_art.mural"

    def test_mural_wrong_extension_left_alone(self) -> None:
        result = ExportDialog._ensure_extension("wall_art.txt", "Mural")
        assert result == "wall_art.txt"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_path_with_dots_in_directory_no_file_extension(self) -> None:
        """A path like '/my.dir/output' has no file extension — should append."""
        result = ExportDialog._ensure_extension("/my.dir/output", "SVG")
        assert result == "/my.dir/output.svg"

    def test_empty_path_appends_extension(self) -> None:
        result = ExportDialog._ensure_extension("", "SVG")
        assert result == ".svg"

    def test_unknown_format_no_extension_no_default(self) -> None:
        """Unknown format: no extension known, path returned unchanged."""
        result = ExportDialog._ensure_extension("file", "Unknown")
        assert result == "file"
