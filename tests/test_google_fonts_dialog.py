"""Tests for GoogleFontsDialog (task 18.5).

Covers:
(a) dialog opens without error
(b) search filters the font list
(c) category filter works
(d) download with mock urlopen adds font to cache and shows success
(e) already-downloaded fonts show checkmark badge
(f) dialog returns selected font path on accept
"""

from __future__ import annotations

import os
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# QApplication setup (headless-safe)
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    """Return (or create) a QApplication for GUI tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_TTF_URL = "https://fonts.gstatic.com/s/testfont/v1/TestFont-Regular.ttf"
_FAKE_TTF_BYTES = b"\x00\x01\x00\x00" + b"\x00" * 100


def _fake_catalog():
    """Return a small list of GoogleFontInfo objects for testing."""
    from plottter.fonts.google_fonts import GoogleFontInfo

    return [
        GoogleFontInfo("Alfa Slab One", "display", ["regular"], "https://fonts.google.com"),
        GoogleFontInfo("Beta Sans", "sans-serif", ["regular", "bold"], "https://fonts.google.com"),
        GoogleFontInfo("Gamma Serif", "serif", ["regular"], "https://fonts.google.com"),
        GoogleFontInfo("Delta Mono", "monospace", ["regular"], "https://fonts.google.com"),
        GoogleFontInfo("Epsilon Script", "handwriting", ["regular"], "https://fonts.google.com"),
    ]


def _make_mock_css_response(ttf_url: str) -> bytes:
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
    css_resp = MagicMock()
    css_resp.__enter__ = MagicMock(return_value=css_resp)
    css_resp.__exit__ = MagicMock(return_value=False)
    css_resp.read.return_value = css_bytes
    css_resp.headers = {"Content-Type": "text/css"}

    font_resp = MagicMock()
    font_resp.__enter__ = MagicMock(return_value=font_resp)
    font_resp.__exit__ = MagicMock(return_value=False)
    font_resp.headers = {"Content-Length": str(len(font_bytes))}
    remaining = [font_bytes]

    def _read(size: int) -> bytes:
        chunk = remaining[0][:size]
        remaining[0] = remaining[0][size:]
        return chunk

    font_resp.read.side_effect = _read
    return MagicMock(side_effect=[css_resp, font_resp])


@contextmanager
def _dialog_ctx(catalog=None, cached_families=None):
    """Context manager that yields a GoogleFontsDialog with mocked catalog/cache.

    The patch for _is_font_cached stays active for the entire duration of the
    ``with`` block so that selection events see the correct cache state.
    """
    from plottter.gui.dialogs.google_fonts import GoogleFontsDialog
    import plottter.fonts.google_fonts as gf_module

    if catalog is None:
        catalog = _fake_catalog()
    cached = set(cached_families or [])

    original = gf_module._catalog
    gf_module._catalog = catalog
    try:
        with patch(
            "plottter.gui.dialogs.google_fonts._is_font_cached",
            side_effect=lambda f: f in cached,
        ):
            dialog = GoogleFontsDialog()
            yield dialog
    finally:
        gf_module._catalog = original


# ---------------------------------------------------------------------------
# (a) Dialog opens without error
# ---------------------------------------------------------------------------

class TestDialogOpens:
    def test_dialog_instantiates(self, qapp):
        """GoogleFontsDialog can be created without raising."""
        with _dialog_ctx() as dialog:
            assert dialog is not None

    def test_dialog_has_search_field(self, qapp):
        """Dialog has a search QLineEdit."""
        with _dialog_ctx() as dialog:
            assert hasattr(dialog, "_search_edit")

    def test_dialog_has_category_combo(self, qapp):
        """Dialog has a category filter QComboBox with expected options."""
        with _dialog_ctx() as dialog:
            assert hasattr(dialog, "_category_combo")
            items = [
                dialog._category_combo.itemText(i)
                for i in range(dialog._category_combo.count())
            ]
            assert "All categories" in items
            assert "Monospace" in items
            assert "Serif" in items
            assert "Sans-Serif" in items

    def test_dialog_populates_list_from_catalog(self, qapp):
        """The font list is populated from the catalog."""
        catalog = _fake_catalog()
        with _dialog_ctx(catalog=catalog) as dialog:
            assert dialog._model.rowCount() == len(catalog)

    def test_selected_path_initially_empty(self, qapp):
        """selected_font_path() returns '' before any download."""
        with _dialog_ctx() as dialog:
            assert dialog.selected_font_path() == ""


# ---------------------------------------------------------------------------
# (b) Search filters the font list
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_filters_rows_via_proxy(self, qapp):
        """Typing in the search bar filters the proxy model."""
        with _dialog_ctx() as dialog:
            before = dialog._proxy.rowCount()
            assert before == len(_fake_catalog())

            # Search for "alfa" — should match only "Alfa Slab One"
            dialog._search_edit.setText("alfa")
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == 1

    def test_search_empty_shows_all(self, qapp):
        """Clearing the search shows all fonts."""
        with _dialog_ctx() as dialog:
            dialog._search_edit.setText("alfa")
            dialog._apply_filter()

            dialog._search_edit.setText("")
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == len(_fake_catalog())

    def test_search_case_insensitive(self, qapp):
        """Search is case-insensitive."""
        with _dialog_ctx() as dialog:
            dialog._search_edit.setText("ALFA")
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible >= 1  # "Alfa Slab One" should match

    def test_search_no_match_shows_zero(self, qapp):
        """Searching for nonsense shows zero fonts."""
        with _dialog_ctx() as dialog:
            dialog._search_edit.setText("zzz_no_match_xyz")
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == 0


# ---------------------------------------------------------------------------
# (c) Category filter works
# ---------------------------------------------------------------------------

class TestCategoryFilter:
    def test_monospace_filter_shows_only_monospace(self, qapp):
        """Selecting Monospace shows only monospace fonts."""
        with _dialog_ctx() as dialog:
            mono_idx = dialog._category_combo.findText("Monospace")
            assert mono_idx >= 0
            dialog._category_combo.setCurrentIndex(mono_idx)
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == 1  # only "Delta Mono"

    def test_serif_filter(self, qapp):
        """Serif category shows only serif fonts (not sans-serif)."""
        with _dialog_ctx() as dialog:
            serif_idx = dialog._category_combo.findText("Serif")
            assert serif_idx >= 0
            dialog._category_combo.setCurrentIndex(serif_idx)
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == 1  # only "Gamma Serif"

    def test_all_categories_shows_all(self, qapp):
        """Selecting 'All categories' shows every font."""
        with _dialog_ctx() as dialog:
            mono_idx = dialog._category_combo.findText("Monospace")
            dialog._category_combo.setCurrentIndex(mono_idx)
            dialog._apply_filter()

            all_idx = dialog._category_combo.findText("All categories")
            dialog._category_combo.setCurrentIndex(all_idx)
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == len(_fake_catalog())

    def test_category_and_search_combined(self, qapp):
        """Category filter and search query can be combined."""
        with _dialog_ctx() as dialog:
            ss_idx = dialog._category_combo.findText("Sans-Serif")
            dialog._category_combo.setCurrentIndex(ss_idx)
            dialog._search_edit.setText("beta")
            dialog._apply_filter()

            visible = sum(
                1
                for row in range(dialog._proxy.rowCount())
                if not dialog._list_view.isRowHidden(row)
            )
            assert visible == 1  # only "Beta Sans"


# ---------------------------------------------------------------------------
# (e) Already-downloaded fonts show checkmark badge
# ---------------------------------------------------------------------------

class TestCachedBadge:
    def test_cached_font_shows_checkmark(self, qapp):
        """A font already on disk gets a ✓ badge in column 2."""
        from plottter.gui.dialogs.google_fonts import _COL_DOWNLOADED, _COL_NAME, _ROLE_FAMILY

        with _dialog_ctx(cached_families={"Alfa Slab One"}) as dialog:
            target_row = None
            for row in range(dialog._model.rowCount()):
                item = dialog._model.item(row, _COL_NAME)
                if item and item.data(_ROLE_FAMILY) == "Alfa Slab One":
                    target_row = row
                    break

            assert target_row is not None
            badge_item = dialog._model.item(target_row, _COL_DOWNLOADED)
            assert badge_item is not None
            assert badge_item.text() == "✓"

    def test_uncached_font_has_no_checkmark(self, qapp):
        """A font not on disk has an empty badge cell."""
        from plottter.gui.dialogs.google_fonts import _COL_DOWNLOADED, _COL_NAME, _ROLE_FAMILY

        with _dialog_ctx(cached_families=set()) as dialog:
            target_row = None
            for row in range(dialog._model.rowCount()):
                item = dialog._model.item(row, _COL_NAME)
                if item and item.data(_ROLE_FAMILY) == "Beta Sans":
                    target_row = row
                    break

            assert target_row is not None
            badge_item = dialog._model.item(target_row, _COL_DOWNLOADED)
            assert badge_item is not None
            assert badge_item.text() == ""

    def test_cached_font_enables_ok_button_on_selection(self, qapp):
        """Selecting an already-cached font enables the OK button."""
        with _dialog_ctx(cached_families={"Alfa Slab One"}) as dialog:
            first_idx = dialog._proxy.index(0, 0)
            dialog._list_view.setCurrentIndex(first_idx)
            assert dialog._ok_btn.isEnabled()

    def test_uncached_font_disables_ok_button(self, qapp):
        """Selecting an uncached font disables the OK button."""
        with _dialog_ctx(cached_families=set()) as dialog:
            second_idx = dialog._proxy.index(1, 0)
            dialog._list_view.setCurrentIndex(second_idx)
            assert not dialog._ok_btn.isEnabled()

    def test_download_btn_disabled_for_cached(self, qapp):
        """Download button is disabled when the selected font is already cached."""
        with _dialog_ctx(cached_families={"Alfa Slab One"}) as dialog:
            first_idx = dialog._proxy.index(0, 0)
            dialog._list_view.setCurrentIndex(first_idx)
            assert not dialog._download_btn.isEnabled()

    def test_download_btn_enabled_for_uncached(self, qapp):
        """Download button is enabled when the selected font is NOT cached."""
        with _dialog_ctx(cached_families=set()) as dialog:
            first_idx = dialog._proxy.index(0, 0)
            dialog._list_view.setCurrentIndex(first_idx)
            assert dialog._download_btn.isEnabled()


# ---------------------------------------------------------------------------
# (d) Download with mock urlopen adds font to cache and shows success
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_worker_emits_ok(self, qapp, tmp_path):
        """_DownloadWorker emits finished_ok with the file path on success."""
        from PyQt6.QtWidgets import QApplication
        from plottter.gui.dialogs.google_fonts import _DownloadWorker

        css_bytes = _make_mock_css_response(_FAKE_TTF_URL)
        mock_open = _make_urlopen_mock(css_bytes, _FAKE_TTF_BYTES)

        received_paths = []
        received_errors = []

        with patch("urllib.request.urlopen", mock_open):
            import plottter.fonts.google_fonts as gf_module
            orig = gf_module._CACHE_DIR
            gf_module._CACHE_DIR = tmp_path
            try:
                worker = _DownloadWorker("TestFont", "regular")
                worker.finished_ok.connect(received_paths.append)
                worker.finished_err.connect(received_errors.append)
                worker.start()
                worker.wait(5000)
                QApplication.processEvents()  # deliver queued signals
            finally:
                gf_module._CACHE_DIR = orig

        assert received_errors == [], f"Unexpected errors: {received_errors}"
        assert len(received_paths) == 1
        assert Path(received_paths[0]).exists()

    def test_download_worker_emits_error_on_failure(self, qapp):
        """_DownloadWorker emits finished_err on network failure."""
        from PyQt6.QtWidgets import QApplication
        from plottter.gui.dialogs.google_fonts import _DownloadWorker

        received_errors = []

        with patch(
            "urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            worker = _DownloadWorker("TestFont", "regular")
            worker.finished_err.connect(received_errors.append)
            worker.start()
            worker.wait(5000)
            QApplication.processEvents()

        assert len(received_errors) == 1
        assert "Failed to fetch CSS" in received_errors[0] or "Connection refused" in received_errors[0]

    def test_download_ok_updates_selected_path(self, qapp, tmp_path):
        """After a successful download, selected_font_path() returns the file path."""
        fake_path = str(tmp_path / "TestFont-regular.ttf")
        (tmp_path / "TestFont-regular.ttf").write_bytes(_FAKE_TTF_BYTES)

        with _dialog_ctx() as dialog:
            dialog._on_download_ok(fake_path)
            assert dialog.selected_font_path() == fake_path
            assert dialog._ok_btn.isEnabled()

    def test_download_ok_updates_badge(self, qapp, tmp_path):
        """After _on_download_ok, the row badge for the selected font is updated to ✓."""
        from plottter.gui.dialogs.google_fonts import _COL_DOWNLOADED, _COL_NAME, _ROLE_FAMILY

        fake_path = str(tmp_path / "AlfaSlabOne-regular.ttf")
        (tmp_path / "AlfaSlabOne-regular.ttf").write_bytes(_FAKE_TTF_BYTES)

        with _dialog_ctx() as dialog:
            # Select "Alfa Slab One"
            first_idx = dialog._proxy.index(0, 0)
            dialog._list_view.setCurrentIndex(first_idx)

            dialog._on_download_ok(fake_path)

            # Find the row and check badge
            target_row = None
            for row in range(dialog._model.rowCount()):
                item = dialog._model.item(row, _COL_NAME)
                if item and item.data(_ROLE_FAMILY) == "Alfa Slab One":
                    target_row = row
                    break

            assert target_row is not None
            badge = dialog._model.item(target_row, _COL_DOWNLOADED)
            assert badge is not None
            assert badge.text() == "✓"


# ---------------------------------------------------------------------------
# (f) Dialog returns selected font path on accept
# ---------------------------------------------------------------------------

class TestAccept:
    def test_accept_returns_selected_path(self, qapp, tmp_path):
        """After _on_accept, selected_font_path() returns the pre-set path."""
        fake_path = str(tmp_path / "Alfa-Slab-One-regular.ttf")
        (tmp_path / "Alfa-Slab-One-regular.ttf").write_bytes(_FAKE_TTF_BYTES)

        with _dialog_ctx(cached_families={"Alfa Slab One"}) as dialog:
            with patch(
                "plottter.gui.dialogs.google_fonts._GOOGLE_FONT_CACHE_DIR",
                tmp_path,
            ):
                dialog._selected_path = fake_path
                dialog._on_accept()

            assert dialog.selected_font_path() == fake_path

    def test_cancel_gives_empty_path(self, qapp):
        """After reject(), selected_font_path() returns ''."""
        with _dialog_ctx() as dialog:
            dialog.reject()
            assert dialog.selected_font_path() == ""

    def test_on_accept_with_cached_font_infers_path(self, qapp, tmp_path):
        """_on_accept infers cache path when _selected_path is empty but font is cached."""
        from plottter.gui.dialogs.google_fonts import _GOOGLE_FONT_CACHE_DIR

        # Create the cached file at the expected path
        safe_name = "Alfa-Slab-One"
        cached_file = tmp_path / f"{safe_name}-regular.ttf"
        cached_file.write_bytes(_FAKE_TTF_BYTES)

        with _dialog_ctx(cached_families={"Alfa Slab One"}) as dialog:
            with patch(
                "plottter.gui.dialogs.google_fonts._GOOGLE_FONT_CACHE_DIR",
                tmp_path,
            ):
                # Select the cached font
                first_idx = dialog._proxy.index(0, 0)
                dialog._list_view.setCurrentIndex(first_idx)
                # Don't set _selected_path — let _on_accept infer it
                dialog._selected_path = ""
                dialog._on_accept()

            assert dialog.selected_font_path() == str(cached_file)
