"""Tests for plottter.fonts.google_fonts (task 18.3).

All network calls are mocked so these tests run without internet access
and without modifying the system or user's font cache.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plottter.fonts.google_fonts import (
    GoogleFontInfo,
    _style_to_weight,
    _TTF_URL_RE,
    download_google_font,
    get_google_fonts_catalog,
    search_google_fonts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(entries: list[dict]) -> str:
    """Serialise a list of catalog entries to JSON."""
    return json.dumps(entries)


def _sample_entry(
    family: str = "Test Font",
    category: str = "sans-serif",
    styles: list[str] | None = None,
) -> dict:
    if styles is None:
        styles = ["regular", "bold"]
    return {
        "family": family,
        "category": category,
        "styles": styles,
        "download_url": (
            f"https://fonts.googleapis.com/css2"
            f"?family={family.replace(' ', '+')}:wght@400&display=swap"
        ),
    }


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

class TestGetGoogleFontsCatalog:
    """Tests for get_google_fonts_catalog()."""

    def test_returns_non_empty_list(self):
        """The bundled catalog must have at least one entry."""
        catalog = get_google_fonts_catalog()
        assert isinstance(catalog, list)
        assert len(catalog) > 0

    def test_entries_are_google_font_info(self):
        """All entries must be GoogleFontInfo instances."""
        catalog = get_google_fonts_catalog()
        for item in catalog[:20]:
            assert isinstance(item, GoogleFontInfo)

    def test_entries_have_family(self):
        """Every entry must have a non-empty family name."""
        catalog = get_google_fonts_catalog()
        for item in catalog[:50]:
            assert isinstance(item.family, str)
            assert item.family.strip() != ""

    def test_entries_have_category(self):
        """Every entry must have a non-empty category."""
        catalog = get_google_fonts_catalog()
        valid_categories = {"serif", "sans-serif", "display", "handwriting", "monospace"}
        for item in catalog[:50]:
            assert item.category in valid_categories, (
                f"{item.family!r} has unexpected category {item.category!r}"
            )

    def test_entries_have_styles(self):
        """Every entry must have at least one style."""
        catalog = get_google_fonts_catalog()
        for item in catalog[:50]:
            assert isinstance(item.styles, list)
            assert len(item.styles) >= 1

    def test_catalog_contains_roboto(self):
        """Roboto is a well-known Google Font that should be in the catalog."""
        catalog = get_google_fonts_catalog()
        families = {f.family for f in catalog}
        assert "Roboto" in families

    def test_catalog_sorted_alphabetically(self):
        """Catalog should be sorted by family name (case-insensitive)."""
        catalog = get_google_fonts_catalog()
        names = [f.family.lower() for f in catalog]
        assert names == sorted(names), "Catalog is not sorted alphabetically"

    def test_in_memory_cache(self):
        """Second call returns the same list object (cached)."""
        import plottter.fonts.google_fonts as gf_module
        gf_module._catalog = None  # clear cache
        first = get_google_fonts_catalog()
        second = get_google_fonts_catalog()
        assert first is second

    def test_load_from_custom_json(self, tmp_path: Path):
        """Loading from a custom catalog JSON produces correct GoogleFontInfo objects."""
        import plottter.fonts.google_fonts as gf_module

        entries = [
            _sample_entry("Alpha", "serif"),
            _sample_entry("Beta", "monospace", ["regular"]),
        ]
        catalog_json = json.dumps(entries)

        original_path = gf_module._CATALOG_PATH
        fake_path = tmp_path / "catalog.json"
        fake_path.write_text(catalog_json, encoding="utf-8")

        original_catalog = gf_module._catalog
        gf_module._catalog = None
        gf_module._CATALOG_PATH = fake_path
        try:
            result = get_google_fonts_catalog()
            assert len(result) == 2
            assert result[0].family == "Alpha"
            assert result[0].category == "serif"
            assert result[1].family == "Beta"
            assert result[1].category == "monospace"
        finally:
            gf_module._CATALOG_PATH = original_path
            gf_module._catalog = original_catalog


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearchGoogleFonts:
    """Tests for search_google_fonts()."""

    def test_search_finds_roboto(self):
        """Searching for 'roboto' should return Roboto."""
        results = search_google_fonts("roboto")
        families = [f.family for f in results]
        assert "Roboto" in families

    def test_search_case_insensitive(self):
        """Search is case-insensitive."""
        results_lower = search_google_fonts("roboto")
        results_upper = search_google_fonts("ROBOTO")
        results_mixed = search_google_fonts("Roboto")
        assert {f.family for f in results_lower} == {f.family for f in results_upper}
        assert {f.family for f in results_lower} == {f.family for f in results_mixed}

    def test_search_returns_empty_for_nonexistent(self):
        """Searching for a nonsense string returns an empty list."""
        results = search_google_fonts("zzzzNotAFontAtAll12345")
        assert results == []

    def test_search_empty_query_returns_all(self):
        """Empty query returns the full catalog."""
        all_fonts = search_google_fonts("")
        catalog = get_google_fonts_catalog()
        assert len(all_fonts) == len(catalog)

    def test_category_filter_monospace(self):
        """Category filter restricts results to the given category."""
        results = search_google_fonts("", category="monospace")
        assert len(results) > 0
        for f in results:
            assert f.category == "monospace"

    def test_category_filter_case_insensitive(self):
        """Category filter is case-insensitive."""
        lower = search_google_fonts("", category="monospace")
        upper = search_google_fonts("", category="MONOSPACE")
        assert {f.family for f in lower} == {f.family for f in upper}

    def test_category_filter_with_query(self):
        """Combining query and category filter works correctly."""
        results = search_google_fonts("a", category="monospace")
        for f in results:
            assert "a" in f.family.lower()
            assert f.category == "monospace"

    def test_search_with_mock_catalog(self):
        """Search works on a small hand-crafted catalog."""
        import plottter.fonts.google_fonts as gf_module

        fake_catalog = [
            GoogleFontInfo("Alpha Slab", "serif", ["regular"], "https://..."),
            GoogleFontInfo("Beta Sans", "sans-serif", ["regular", "bold"], "https://..."),
            GoogleFontInfo("Gamma Mono", "monospace", ["regular"], "https://..."),
        ]
        original = gf_module._catalog
        gf_module._catalog = fake_catalog
        try:
            assert len(search_google_fonts("alpha")) == 1
            assert search_google_fonts("alpha")[0].family == "Alpha Slab"
            assert len(search_google_fonts("", category="monospace")) == 1
            assert len(search_google_fonts("beta", category="sans-serif")) == 1
            assert len(search_google_fonts("beta", category="serif")) == 0
        finally:
            gf_module._catalog = original


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# Minimal TTF header bytes (just enough for file type detection)
_FAKE_TTF_BYTES = b"\x00\x01\x00\x00" + b"\x00" * 100


def _make_mock_css_response(ttf_url: str) -> bytes:
    """Return fake CSS that contains a single TTF URL."""
    css = textwrap.dedent(f"""\
        @font-face {{
          font-family: 'TestFont';
          font-style: normal;
          font-weight: 400;
          src: url({ttf_url});
        }}
    """)
    return css.encode("utf-8")


def _make_urlopen_mock(css_bytes: bytes, font_bytes: bytes):
    """Return a mock for urllib.request.urlopen.

    The first call (CSS API) returns css_bytes.
    The second call (TTF download) returns font_bytes.
    """
    css_resp = MagicMock()
    css_resp.__enter__ = MagicMock(return_value=css_resp)
    css_resp.__exit__ = MagicMock(return_value=False)
    css_resp.read.return_value = css_bytes
    css_resp.headers = {"Content-Type": "text/css"}

    font_resp = MagicMock()
    font_resp.__enter__ = MagicMock(return_value=font_resp)
    font_resp.__exit__ = MagicMock(return_value=False)
    # Simulate chunked reads
    font_resp.headers = {"Content-Length": str(len(font_bytes))}
    _remaining = [font_bytes]

    def _read(size: int) -> bytes:
        chunk = _remaining[0][:size]
        _remaining[0] = _remaining[0][size:]
        return chunk

    font_resp.read.side_effect = _read

    mock_open = MagicMock(side_effect=[css_resp, font_resp])
    return mock_open


class TestDownloadGoogleFont:
    """Tests for download_google_font()."""

    _FAKE_TTF_URL = "https://fonts.gstatic.com/s/testfont/v1/TestFont-Regular.ttf"

    def _mock_urlopen(self, css_bytes: bytes, font_bytes: bytes):
        return _make_urlopen_mock(css_bytes, font_bytes)

    def test_saves_ttf_to_cache_dir(self, tmp_path: Path):
        """download_google_font saves a TTF file to the cache directory."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        with patch("urllib.request.urlopen", mock_open):
            path = download_google_font("TestFont", cache_dir=tmp_path)

        assert Path(path).exists()
        assert Path(path).stat().st_size == len(_FAKE_TTF_BYTES)
        assert Path(path).read_bytes() == _FAKE_TTF_BYTES

    def test_returned_path_is_absolute(self, tmp_path: Path):
        """The returned path string is an absolute path."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        with patch("urllib.request.urlopen", mock_open):
            path = download_google_font("TestFont", cache_dir=tmp_path)

        assert Path(path).is_absolute()

    def test_filename_uses_family_and_style(self, tmp_path: Path):
        """The saved file is named {family}-{style}.ttf."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        with patch("urllib.request.urlopen", mock_open):
            path = download_google_font("Test Font", "regular", cache_dir=tmp_path)

        assert Path(path).name == "Test-Font-regular.ttf"

    def test_cache_hit_skips_download(self, tmp_path: Path):
        """Second call with the same family/style returns cached path without network calls."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        with patch("urllib.request.urlopen", mock_open):
            path1 = download_google_font("TestFont", cache_dir=tmp_path)

        # Second call — no network activity
        with patch("urllib.request.urlopen") as mock_no_net:
            path2 = download_google_font("TestFont", cache_dir=tmp_path)
            mock_no_net.assert_not_called()

        assert path1 == path2

    def test_different_family_causes_new_download(self, tmp_path: Path):
        """Different family names result in separate downloads and files."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        font_bytes_a = _FAKE_TTF_BYTES
        font_bytes_b = b"\x00\x01\x00\x00" + b"\xAB" * 100

        with patch("urllib.request.urlopen") as mock_open:
            # Set up two separate response sequences
            mock_open.side_effect = [
                # First download (FamilyA)
                *_make_urlopen_mock(css_bytes, font_bytes_a).side_effect,
                # Second download (FamilyB)
                *_make_urlopen_mock(css_bytes, font_bytes_b).side_effect,
            ]
            path_a = download_google_font("FamilyA", cache_dir=tmp_path)
            path_b = download_google_font("FamilyB", cache_dir=tmp_path)

        assert path_a != path_b
        assert Path(path_a).read_bytes() == font_bytes_a
        assert Path(path_b).read_bytes() == font_bytes_b

    def test_no_cache_dir_uses_default(self, tmp_path: Path):
        """When cache_dir is None, the default ~/.plottter/fonts/google is used."""
        import plottter.fonts.google_fonts as gf_module

        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        fake_default = tmp_path / "default_cache"

        original_cache_dir = gf_module._CACHE_DIR
        gf_module._CACHE_DIR = fake_default
        path = None
        try:
            with patch("urllib.request.urlopen", mock_open):
                path = download_google_font("TestFont")
            assert Path(path).parent == fake_default
        finally:
            gf_module._CACHE_DIR = original_cache_dir
            # Clean up
            if path and Path(path).exists():
                Path(path).unlink()

    def test_progress_callback_called(self, tmp_path: Path):
        """progress_callback receives (bytes_downloaded, total_bytes) calls."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        calls: list[tuple[int, int]] = []

        def _cb(downloaded: int, total: int) -> None:
            calls.append((downloaded, total))

        with patch("urllib.request.urlopen", mock_open):
            download_google_font(
                "TestFont", cache_dir=tmp_path, progress_callback=_cb
            )

        assert len(calls) > 0
        # Final call should report all bytes downloaded
        final_downloaded, total = calls[-1]
        assert final_downloaded == len(_FAKE_TTF_BYTES)

    def test_css_api_failure_raises_runtime_error(self, tmp_path: Path):
        """A network failure on the CSS API raises RuntimeError."""
        with patch(
            "urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch CSS"):
                download_google_font("NoFont", cache_dir=tmp_path)

    def test_no_ttf_url_in_css_raises_value_error(self, tmp_path: Path):
        """If the CSS response contains no TTF URL, ValueError is raised."""
        # Return CSS with only WOFF2 URL (no .ttf)
        bad_css = b"@font-face { src: url(https://fonts.gstatic.com/s/f.woff2); }"

        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = bad_css
        resp.headers = {}

        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(ValueError, match="Could not find a TTF URL"):
                download_google_font("BadFont", cache_dir=tmp_path)

    def test_ttf_download_failure_raises_runtime_error(self, tmp_path: Path):
        """A network failure during the TTF download raises RuntimeError."""
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)

        css_resp = MagicMock()
        css_resp.__enter__ = MagicMock(return_value=css_resp)
        css_resp.__exit__ = MagicMock(return_value=False)
        css_resp.read.return_value = css_bytes
        css_resp.headers = {}

        with patch(
            "urllib.request.urlopen",
            side_effect=[css_resp, OSError("download failed")],
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_google_font("TestFont", cache_dir=tmp_path)

    def test_cache_dir_created_if_missing(self, tmp_path: Path):
        """The cache directory is created automatically if it doesn't exist."""
        nested = tmp_path / "a" / "b" / "c"
        css_bytes = _make_mock_css_response(self._FAKE_TTF_URL)
        mock_open = self._mock_urlopen(css_bytes, _FAKE_TTF_BYTES)

        with patch("urllib.request.urlopen", mock_open):
            download_google_font("TestFont", cache_dir=nested)

        assert nested.exists()


# ---------------------------------------------------------------------------
# _style_to_weight helper
# ---------------------------------------------------------------------------

class TestStyleToWeight:
    """Tests for the internal _style_to_weight() helper."""

    def test_regular_maps_to_400(self):
        assert _style_to_weight("regular") == "400"

    def test_bold_maps_to_700(self):
        assert _style_to_weight("bold") == "700"

    def test_italic_maps_to_400(self):
        assert _style_to_weight("italic") == "400"

    def test_numeric_passthrough(self):
        assert _style_to_weight("500") == "500"
        assert _style_to_weight("300") == "300"
        assert _style_to_weight("900") == "900"

    def test_numeric_italic_strips_italic(self):
        assert _style_to_weight("700italic") == "700"
        assert _style_to_weight("500italic") == "500"

    def test_unknown_defaults_to_400(self):
        assert _style_to_weight("xxxx") == "400"


# ---------------------------------------------------------------------------
# TTF URL regex
# ---------------------------------------------------------------------------

class TestTtfUrlRegex:
    """Tests for the _TTF_URL_RE regex pattern."""

    def test_matches_direct_ttf_url(self):
        css = "url(https://fonts.gstatic.com/s/roboto/v47/KFO.ttf)"
        match = _TTF_URL_RE.search(css)
        assert match is not None
        assert match.group(1).endswith(".ttf")

    def test_does_not_match_woff2(self):
        css = "url(https://fonts.gstatic.com/s/roboto/v47/KFO.woff2)"
        assert _TTF_URL_RE.search(css) is None

    def test_does_not_match_non_gstatic(self):
        css = "url(https://example.com/font.ttf)"
        assert _TTF_URL_RE.search(css) is None


# ---------------------------------------------------------------------------
# Path traversal protection in download_google_font
# ---------------------------------------------------------------------------

class TestDownloadPathTraversal:
    """Ensure that path traversal characters in family/style are rejected."""

    def test_slash_in_family_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("../attack", cache_dir=tmp_path)

    def test_backslash_in_family_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("..\\attack", cache_dir=tmp_path)

    def test_dotdot_only_family_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("..", cache_dir=tmp_path)

    def test_slash_in_style_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("TestFont", style="../etc/passwd", cache_dir=tmp_path)

    def test_backslash_in_style_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("TestFont", style="..\\attack", cache_dir=tmp_path)

    def test_dotdot_in_style_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            download_google_font("TestFont", style="..", cache_dir=tmp_path)
