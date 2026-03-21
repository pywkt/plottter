"""Tests for task 53.3 — Auto-load cached AI results when source image is loaded.

Covers:
(a) Load image, run BG removal, load a different image, re-load original image —
    BG removal result is restored from cache without API call.
(b) "(cached)" indicator appears next to the AI BG checkbox when a cached result exists.
(c) Clearing the cache and reloading the image removes the cached indicator.
(d) Cached results are NOT auto-applied — the checkbox stays unchecked; only pre-loaded.
"""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="CacheTest", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _make_rgb_image(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_rgba_image(h: int = 8, w: int = 8) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return arr


def _write_bg_cache(cache_dir: str, image: np.ndarray, rgba: np.ndarray) -> str:
    """Write a fake BG-removal PNG to the cache and return its path."""
    from PIL import Image as _PIL_Image
    img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
    bg_dir = os.path.join(cache_dir, "bg_removal")
    os.makedirs(bg_dir, exist_ok=True)
    cache_path = os.path.join(bg_dir, f"{img_hash}.png")
    pil = _PIL_Image.fromarray(rgba, mode="RGBA")
    pil.save(cache_path)
    return cache_path


def _write_depth_cache(cache_dir: str, image: np.ndarray, depth: np.ndarray) -> str:
    """Write a fake depth-map PNG (16-bit) to the cache and return its path."""
    import cv2
    img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
    depth_dir = os.path.join(cache_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    cache_path = os.path.join(depth_dir, f"{img_hash}.png")
    uint16_arr = (depth * 65535).astype(np.uint16)
    cv2.imwrite(cache_path, uint16_arr)
    return cache_path


# ---------------------------------------------------------------------------
# (a) & (b): BG removal result restored from cache; "(cached)" indicator shown
# ---------------------------------------------------------------------------


class TestBgCachePreload:
    """BG-removal disk cache is loaded when a matching image is opened."""

    def test_bg_rgba_populated_from_cache(self, settings_panel, tmp_path) -> None:
        """_ai_bg_rgba is populated from disk cache when the image is loaded."""
        image = _make_rgb_image()
        rgba = _make_rgba_image()
        cache_path = _write_bg_cache(str(tmp_path), image, rgba)
        assert os.path.exists(cache_path)

        settings_panel._raw_image = image
        settings_panel._image_source_path = "/fake/image.png"
        settings_panel._ai_bg_rgba = None

        # Patch _get_cache_dir so the panel uses our tmp_path
        settings_panel._get_cache_dir = lambda: str(tmp_path)

        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._check_ai_cache_for_image(image, "/fake/image.png")

        assert settings_panel._ai_bg_rgba is not None, (
            "_ai_bg_rgba should be populated from disk cache"
        )
        np.testing.assert_array_equal(settings_panel._ai_bg_rgba, rgba)

    def test_cached_label_shown_after_cache_preload(self, settings_panel, tmp_path) -> None:
        """(cached) label is visible after a cache hit."""
        image = _make_rgb_image()
        rgba = _make_rgba_image()
        _write_bg_cache(str(tmp_path), image, rgba)

        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._check_ai_cache_for_image(image, "/fake/img.png")

        assert settings_panel._ai_bg_cached_label.isVisible(), (
            "(cached) label should be visible when a cached BG result is loaded"
        )

    def test_cached_label_hidden_when_no_cache(self, settings_panel, tmp_path) -> None:
        """(cached) label stays hidden when no matching cache file exists."""
        image = _make_rgb_image()

        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._check_ai_cache_for_image(image, "/fake/no_cache.png")

        assert not settings_panel._ai_bg_cached_label.isVisible(), (
            "(cached) label should be hidden when no cache file exists"
        )

    def test_ai_bg_rgba_none_when_no_cache(self, settings_panel, tmp_path) -> None:
        """_ai_bg_rgba stays None when no cache file exists."""
        image = _make_rgb_image()

        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._ai_bg_rgba = None
        settings_panel._check_ai_cache_for_image(image, "/fake/no_cache.png")

        assert settings_panel._ai_bg_rgba is None


# ---------------------------------------------------------------------------
# (c) Clearing cache and reloading removes the cached indicator
# ---------------------------------------------------------------------------


class TestCacheClearBehavior:
    """After clearing the disk cache, reloading the image hides the indicator."""

    def test_cached_label_hidden_after_cache_cleared(
        self, settings_panel, tmp_path
    ) -> None:
        """Delete the cache file; re-calling check finds no cache → label hidden."""
        image = _make_rgb_image()
        rgba = _make_rgba_image()
        cache_path = _write_bg_cache(str(tmp_path), image, rgba)

        settings_panel._get_cache_dir = lambda: str(tmp_path)

        # First load — cache hit
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._check_ai_cache_for_image(image, "/fake/img.png")
        assert settings_panel._ai_bg_cached_label.isVisible()

        # Clear the cache file
        os.remove(cache_path)

        # Reload — simulate what _on_load_image does
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._check_ai_cache_for_image(image, "/fake/img.png")

        assert not settings_panel._ai_bg_cached_label.isVisible(), (
            "(cached) label should be hidden after cache is cleared and image reloaded"
        )
        assert settings_panel._ai_bg_rgba is None


# ---------------------------------------------------------------------------
# (d) Cached results NOT auto-applied — checkbox stays unchecked
# ---------------------------------------------------------------------------


class TestNoCachedAutoApply:
    """Cached results are pre-loaded but NOT auto-applied (checkbox stays unchecked)."""

    def test_ai_bg_check_stays_unchecked_when_cache_exists(
        self, settings_panel, tmp_path
    ) -> None:
        """The AI BG checkbox must remain unchecked even when a cached result is loaded."""
        image = _make_rgb_image()
        rgba = _make_rgba_image()
        _write_bg_cache(str(tmp_path), image, rgba)

        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)

        # Simulate _on_load_image: uncheck first, then load cache
        settings_panel._ai_bg_check.blockSignals(True)
        settings_panel._ai_bg_check.setChecked(False)
        settings_panel._ai_bg_check.blockSignals(False)

        settings_panel._check_ai_cache_for_image(image, "/fake/img.png")

        assert not settings_panel._ai_bg_check.isChecked(), (
            "AI BG checkbox must NOT be auto-checked when a cached result is pre-loaded"
        )

    def test_ai_bg_check_reset_on_new_image_load(
        self, settings_panel, tmp_path
    ) -> None:
        """Loading a new image always unchecks the AI BG checkbox.

        Even if the previous image had BG removal checked, the checkbox must reset
        so the cached result is not auto-applied to the new image.
        """
        image = _make_rgb_image()
        rgba = _make_rgba_image()
        _write_bg_cache(str(tmp_path), image, rgba)

        settings_panel._get_cache_dir = lambda: str(tmp_path)

        # Simulate: user had checkbox checked on previous image
        settings_panel._ai_bg_check.blockSignals(True)
        settings_panel._ai_bg_check.setChecked(True)
        settings_panel._ai_bg_check.blockSignals(False)

        # Simulate _on_load_image reset sequence
        settings_panel._ai_bg_rgba = None
        settings_panel._ai_bg_cached_label.setVisible(False)
        settings_panel._ai_bg_check.blockSignals(True)
        settings_panel._ai_bg_check.setChecked(False)
        settings_panel._ai_bg_check.blockSignals(False)

        # Now load cache (as _on_load_image would)
        settings_panel._check_ai_cache_for_image(image, "/fake/img.png")

        # Even though cache was found, checkbox must remain unchecked
        assert not settings_panel._ai_bg_check.isChecked(), (
            "AI BG checkbox must be reset to unchecked when a new image is loaded, "
            "even if a cached result is found"
        )
        # But the result is pre-loaded
        assert settings_panel._ai_bg_rgba is not None


# ---------------------------------------------------------------------------
# Depth map cache pre-load
# ---------------------------------------------------------------------------


class TestDepthCachePreload:
    """Depth map disk cache is loaded into _depth_map_cache when image is opened."""

    def test_depth_map_preloaded_from_cache(self, settings_panel, tmp_path) -> None:
        """_depth_map_cache[path] is populated from disk cache without an API call."""
        image = _make_rgb_image()
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        _write_depth_cache(str(tmp_path), image, depth)

        path = "/fake/image.png"
        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._depth_map_cache.pop(path, None)

        settings_panel._check_ai_cache_for_image(image, path)

        assert path in settings_panel._depth_map_cache, (
            "_depth_map_cache should be populated from disk cache"
        )
        cached = settings_panel._depth_map_cache[path]
        np.testing.assert_allclose(cached, depth, atol=1.0 / 65535)

    def test_depth_status_label_updated_when_cache_hit(
        self, settings_panel, tmp_path
    ) -> None:
        """Depth status label shows '(cached)' when a cached depth map is found."""
        image = _make_rgb_image()
        depth = np.full((8, 8), 0.3, dtype=np.float32)
        _write_depth_cache(str(tmp_path), image, depth)

        path = "/fake/depth_test.png"
        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._depth_status_label.setText("No depth map generated")
        settings_panel._depth_map_cache.pop(path, None)

        settings_panel._check_ai_cache_for_image(image, path)

        assert "cached" in settings_panel._depth_status_label.text().lower(), (
            "Depth status label should indicate cached state"
        )

    def test_depth_status_label_unchanged_when_no_cache(
        self, settings_panel, tmp_path
    ) -> None:
        """Depth status label stays 'No depth map generated' when no cache exists."""
        image = _make_rgb_image()
        path = "/fake/nocache.png"

        settings_panel._get_cache_dir = lambda: str(tmp_path)
        settings_panel._depth_status_label.setText("No depth map generated")
        settings_panel._depth_map_cache.pop(path, None)

        settings_panel._check_ai_cache_for_image(image, path)

        assert settings_panel._depth_status_label.text() == "No depth map generated"


# ---------------------------------------------------------------------------
# _get_cache_dir helper
# ---------------------------------------------------------------------------


class TestGetCacheDir:
    """_get_cache_dir() always returns a non-empty string."""

    def test_returns_default_when_not_configured(self, settings_panel) -> None:
        """Returns a path under ~/.plottter/ai_cache when QSettings has nothing."""
        from unittest.mock import patch
        from PyQt6.QtCore import QSettings

        with patch.object(QSettings, "value", return_value=""):
            cache_dir = settings_panel._get_cache_dir()

        assert isinstance(cache_dir, str)
        assert len(cache_dir) > 0
        assert "ai_cache" in cache_dir or ".plottter" in cache_dir

    def test_returns_configured_path(self, settings_panel, tmp_path) -> None:
        """Returns the configured path when ai/cache_dir is set."""
        from unittest.mock import patch
        from PyQt6.QtCore import QSettings

        expected = str(tmp_path / "custom_cache")

        def mock_value(key, default=""):
            if key == "ai/cache_dir":
                return expected
            return default

        with patch.object(QSettings, "value", side_effect=mock_value):
            cache_dir = settings_panel._get_cache_dir()

        assert cache_dir == expected


# ---------------------------------------------------------------------------
# _AiBgWorker cache_dir plumbing
# ---------------------------------------------------------------------------


class TestAiBgWorkerCacheDir:
    """_AiBgWorker passes cache_dir to ReplicateClient so disk caching works."""

    def test_worker_passes_cache_dir_to_client(self, tmp_path) -> None:
        """_AiBgWorker uses the given cache_dir when constructing ReplicateClient."""
        from unittest.mock import MagicMock, patch
        from plottter.gui.settings_panel import _AiBgWorker
        import plottter.ai.replicate_client as rc_mod

        h, w = 4, 4
        image = np.zeros((h, w, 3), dtype=np.uint8)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/out.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgba", return_value=rgba):
                worker = _AiBgWorker(
                    api_key="r8_test",
                    image=image,
                    cache_dir=str(tmp_path),
                )
                worker.run()  # synchronous call in test

        # A PNG should have been written to the bg_removal/ subdir
        pngs = list((tmp_path / "bg_removal").glob("*.png"))
        assert len(pngs) == 1, (
            f"Expected 1 cached PNG in bg_removal/; got {len(pngs)}"
        )
