"""Tests for _MapMixin sub-panel + fetch flow — task 146.3.

Covers (spec §10.2):
- Clicking Fetch with a mocked worker stores _map_data and updates status label.
- A cache hit populates _map_data without starting a worker.
- The error path shows error text in status label and re-enables the button.
- get_params() injects _map_data only in Map mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_map_data(n_features: int = 3) -> object:
    """Return a minimal MapData instance."""
    from plottter.osm.types import MapData, MapFeature

    features = {
        "roads_major": [
            MapFeature(
                tags={"highway": "primary"},
                coords=[(35.01, 135.76), (35.02, 135.77)],
                is_area=False,
            )
            for _ in range(n_features)
        ]
    }
    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(34.9616, 135.7181, 35.0616, 135.8181),
        features=features,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project():
    canvas = Canvas.from_preset("A4", margin=10.0)
    p = Project(name="MapTest", canvas=canvas)
    p.add_layer(Layer(name="Layer 1", color="#000000"))
    return p


@pytest.fixture
def controller(project, qapp):
    from plottter.gui.project_controller import ProjectController

    return ProjectController(project)


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel

    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    # Switch to Map mode so the map group is visible and wired
    panel.on_mode_changed("Map")
    return panel


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestMapMixinStructure:
    """_MapMixin builds the expected widgets."""

    def test_map_group_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_group"), "_map_group must be created"

    def test_location_edit_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_location_edit")

    def test_fetch_btn_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_fetch_btn")

    def test_status_label_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_status_label")

    def test_extent_label_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_extent_label")

    def test_map_group_visible_in_map_mode(self, settings_panel):
        assert settings_panel._map_group.isVisible(), (
            "_map_group must be visible after on_mode_changed('Map')"
        )

    def test_map_group_hidden_in_math_mode(self, settings_panel):
        settings_panel.on_mode_changed("Math Art")
        assert not settings_panel._map_group.isVisible()

    def test_fetch_btn_initially_enabled(self, settings_panel):
        assert settings_panel._map_fetch_btn.isEnabled()

    def test_initial_status_idle(self, settings_panel):
        assert "Idle" in settings_panel._map_status_label.text() or \
               settings_panel._map_status_label.text() != "", \
               "status label should be non-empty on init"


# ---------------------------------------------------------------------------
# Empty location guard
# ---------------------------------------------------------------------------


class TestFetchEmptyLocation:
    def test_empty_location_shows_prompt(self, settings_panel):
        settings_panel._map_location_edit.setText("")
        settings_panel._on_fetch_map_clicked()
        assert "location" in settings_panel._map_status_label.text().lower()

    def test_empty_location_does_not_start_worker(self, settings_panel):
        settings_panel._map_location_edit.setText("")
        settings_panel._on_fetch_map_clicked()
        assert settings_panel._map_fetch_worker is None


# ---------------------------------------------------------------------------
# Cache-hit path
# ---------------------------------------------------------------------------


class TestFetchCacheHit:
    """When osm.cache.load returns a MapData, _map_data is populated without worker."""

    def test_cache_hit_stores_map_data(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="abc123"),
            patch("plottter.osm.cache.load", return_value=fake),
        ):
            settings_panel._on_fetch_map_clicked()

        assert settings_panel._map_data is fake

    def test_cache_hit_updates_status(self, settings_panel):
        fake = _make_fake_map_data(n_features=2)
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="abc123"),
            patch("plottter.osm.cache.load", return_value=fake),
        ):
            settings_panel._on_fetch_map_clicked()

        status = settings_panel._map_status_label.text()
        # Should mention the feature count and "cached"
        assert "cached" in status.lower() or "feature" in status.lower(), (
            f"status label should mention cache/features; got: {status!r}"
        )

    def test_cache_hit_does_not_start_worker(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="abc123"),
            patch("plottter.osm.cache.load", return_value=fake),
        ):
            settings_panel._on_fetch_map_clicked()

        assert settings_panel._map_fetch_worker is None

    def test_cache_hit_button_stays_enabled(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="abc123"),
            patch("plottter.osm.cache.load", return_value=fake),
        ):
            settings_panel._on_fetch_map_clicked()

        assert settings_panel._map_fetch_btn.isEnabled()


# ---------------------------------------------------------------------------
# Worker success path (mocked worker)
# ---------------------------------------------------------------------------


class TestFetchWorkerSuccess:
    """Mocked _MapFetchWorker: finished signal → _map_data stored, status updated."""

    def _click_fetch_with_mock_worker(self, panel, fake_data):
        """Click Fetch with cache miss and a mock worker that fires finished immediately."""
        panel._map_location_edit.setText("Kyoto, Japan")

        mock_worker = MagicMock()
        # Store connected callbacks so we can fire them
        callbacks: dict = {}

        def connect_finished(cb):
            callbacks["finished"] = cb

        def connect_error(cb):
            callbacks["error"] = cb

        def connect_progress(cb):
            callbacks["progress"] = cb

        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = connect_finished
        mock_worker.error = MagicMock()
        mock_worker.error.connect = connect_error
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = connect_progress
        mock_worker.start = MagicMock()

        with (
            patch("plottter.osm.cache.cache_key", return_value="testkey"),
            patch("plottter.osm.cache.load", return_value=None),  # cache miss
            patch(
                "plottter.gui.settings_panel.workers._MapFetchWorker",
                return_value=mock_worker,
            ),
        ):
            panel._on_fetch_map_clicked()
            # Simulate the worker firing the finished signal
            if "finished" in callbacks:
                callbacks["finished"](fake_data)

        return callbacks

    def test_worker_started_on_cache_miss(self, settings_panel):
        mock_worker = MagicMock()
        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = MagicMock()
        mock_worker.error = MagicMock()
        mock_worker.error.connect = MagicMock()
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.start = MagicMock()

        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="key1"),
            patch("plottter.osm.cache.load", return_value=None),
            patch(
                "plottter.gui.settings_panel.workers._MapFetchWorker",
                return_value=mock_worker,
            ),
        ):
            settings_panel._on_fetch_map_clicked()

        mock_worker.start.assert_called_once()

    def test_button_disabled_while_fetching(self, settings_panel):
        mock_worker = MagicMock()
        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = MagicMock()
        mock_worker.error = MagicMock()
        mock_worker.error.connect = MagicMock()
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.start = MagicMock()

        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="key2"),
            patch("plottter.osm.cache.load", return_value=None),
            patch(
                "plottter.gui.settings_panel.workers._MapFetchWorker",
                return_value=mock_worker,
            ),
        ):
            settings_panel._on_fetch_map_clicked()
            # Button should be disabled while worker is running
            assert not settings_panel._map_fetch_btn.isEnabled()

    def test_finished_stores_map_data(self, settings_panel):
        fake = _make_fake_map_data()
        self._click_fetch_with_mock_worker(settings_panel, fake)
        assert settings_panel._map_data is fake

    def test_finished_updates_status(self, settings_panel):
        fake = _make_fake_map_data(n_features=5)
        self._click_fetch_with_mock_worker(settings_panel, fake)
        status = settings_panel._map_status_label.text()
        assert "feature" in status.lower() or "loaded" in status.lower(), (
            f"status should mention features after success; got: {status!r}"
        )

    def test_finished_re_enables_button(self, settings_panel):
        fake = _make_fake_map_data()
        self._click_fetch_with_mock_worker(settings_panel, fake)
        assert settings_panel._map_fetch_btn.isEnabled()

    def test_finished_writes_cache(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        mock_worker = MagicMock()
        callbacks: dict = {}

        def connect_finished(cb):
            callbacks["finished"] = cb

        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = connect_finished
        mock_worker.error = MagicMock()
        mock_worker.error.connect = MagicMock()
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.start = MagicMock()

        with (
            patch("plottter.osm.cache.cache_key", return_value="writekey"),
            patch("plottter.osm.cache.load", return_value=None),
            patch(
                "plottter.gui.settings_panel.workers._MapFetchWorker",
                return_value=mock_worker,
            ),
            patch("plottter.osm.cache.store") as mock_store,
        ):
            settings_panel._on_fetch_map_clicked()
            settings_panel._map_fetch_cache_key = "writekey"  # ensure key is set
            if "finished" in callbacks:
                callbacks["finished"](fake)

        mock_store.assert_called_once()

    def test_finished_sets_project_attribution(self, settings_panel, controller):
        fake = _make_fake_map_data()
        self._click_fetch_with_mock_worker(settings_panel, fake)
        project = controller.current_project
        assert "map_attribution" in project.metadata, (
            "project.metadata should contain 'map_attribution' after fetch"
        )


# ---------------------------------------------------------------------------
# Worker error path
# ---------------------------------------------------------------------------


class TestFetchWorkerError:
    """Error signal: shows error text (red), re-enables button, never crashes."""

    def test_error_updates_status(self, settings_panel):
        settings_panel._on_map_fetch_error("network timeout")
        assert "network timeout" in settings_panel._map_status_label.text()

    def test_error_status_is_red(self, settings_panel):
        settings_panel._on_map_fetch_error("some error")
        # stylesheet should indicate red
        style = settings_panel._map_status_label.styleSheet()
        assert "red" in style.lower()

    def test_error_re_enables_button(self, settings_panel):
        settings_panel._map_fetch_btn.setEnabled(False)
        settings_panel._on_map_fetch_error("timeout")
        assert settings_panel._map_fetch_btn.isEnabled()

    def test_error_clears_worker_ref(self, settings_panel):
        settings_panel._map_fetch_worker = MagicMock()
        settings_panel._on_map_fetch_error("fail")
        assert settings_panel._map_fetch_worker is None

    def test_error_does_not_raise(self, settings_panel):
        """_on_map_fetch_error must never propagate an exception."""
        try:
            settings_panel._on_map_fetch_error("boom")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_on_map_fetch_error raised: {exc!r}")

    def test_error_via_mocked_worker(self, settings_panel):
        """Simulate a worker emitting error: button re-enabled, status shows error."""
        settings_panel._map_location_edit.setText("Nowhere")

        mock_worker = MagicMock()
        error_callbacks: list = []

        def connect_error(cb):
            error_callbacks.append(cb)

        mock_worker.finished = MagicMock()
        mock_worker.finished.connect = MagicMock()
        mock_worker.error = MagicMock()
        mock_worker.error.connect = connect_error
        mock_worker.progress = MagicMock()
        mock_worker.progress.connect = MagicMock()
        mock_worker.start = MagicMock()

        with (
            patch("plottter.osm.cache.cache_key", return_value="k"),
            patch("plottter.osm.cache.load", return_value=None),
            patch(
                "plottter.gui.settings_panel.workers._MapFetchWorker",
                return_value=mock_worker,
            ),
        ):
            settings_panel._on_fetch_map_clicked()
            for cb in error_callbacks:
                cb("Could not geocode location: 'Nowhere'")

        assert not settings_panel._map_fetch_btn.isEnabled() is False  # button re-enabled
        assert settings_panel._map_fetch_btn.isEnabled()
        assert "geocode" in settings_panel._map_status_label.text().lower() or \
               "Nowhere" in settings_panel._map_status_label.text()


# ---------------------------------------------------------------------------
# get_params() injection
# ---------------------------------------------------------------------------


class TestGetParamsInjection:
    """get_params() includes _map_data only in Map mode when data is loaded."""

    def test_map_data_injected_in_map_mode(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_data = fake
        settings_panel._current_mode = "Map"
        params = settings_panel.get_params()
        assert "_map_data" in params
        assert params["_map_data"] is fake

    def test_map_data_not_injected_in_math_mode(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_data = fake
        settings_panel._current_mode = "Math Art"
        params = settings_panel.get_params()
        assert "_map_data" not in params

    def test_map_data_not_injected_when_none(self, settings_panel):
        settings_panel._map_data = None
        settings_panel._current_mode = "Map"
        params = settings_panel.get_params()
        assert "_map_data" not in params
