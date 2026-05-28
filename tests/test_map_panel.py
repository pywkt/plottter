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


# ---------------------------------------------------------------------------
# Positioning controls — task 151.1
# ---------------------------------------------------------------------------


class TestMapPositioningControls:
    """'Position Map' and 'Reset to fit' button existence, enable/disable, and behaviour."""

    # -- widget existence --

    def test_position_map_btn_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_position_btn"), (
            "_map_position_btn must be created by _build_map_group"
        )

    def test_reset_to_fit_btn_exists(self, settings_panel):
        assert hasattr(settings_panel, "_map_reset_btn"), (
            "_map_reset_btn must be created by _build_map_group"
        )

    def test_position_map_btn_is_checkable(self, settings_panel):
        assert settings_panel._map_position_btn.isCheckable()

    # -- disabled before data loaded --

    def test_position_map_btn_disabled_before_fetch(self, settings_panel):
        settings_panel._map_data = None
        assert not settings_panel._map_position_btn.isEnabled(), (
            "Position Map must be disabled when no map data is loaded"
        )

    def test_reset_to_fit_btn_disabled_before_fetch(self, settings_panel):
        settings_panel._map_data = None
        assert not settings_panel._map_reset_btn.isEnabled(), (
            "Reset to fit must be disabled when no map data is loaded"
        )

    # -- enabled after data loaded via _on_map_fetch_finished --

    def test_positioning_buttons_enabled_after_fetch_finished(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)
        assert settings_panel._map_position_btn.isEnabled(), (
            "Position Map must be enabled after _on_map_fetch_finished"
        )
        assert settings_panel._map_reset_btn.isEnabled(), (
            "Reset to fit must be enabled after _on_map_fetch_finished"
        )

    def test_map_view_initialised_after_fetch_finished(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)
        assert settings_panel._map_view is not None, (
            "_map_view must be set after _on_map_fetch_finished"
        )
        view = settings_panel._map_view
        assert "center_lat" in view
        assert "center_lon" in view
        assert "scale" in view

    # -- enabled after cache-hit path --

    def test_positioning_buttons_enabled_after_cache_hit(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._map_location_edit.setText("Kyoto, Japan")

        with (
            patch("plottter.osm.cache.cache_key", return_value="abc123"),
            patch("plottter.osm.cache.load", return_value=fake),
        ):
            settings_panel._on_fetch_map_clicked()

        assert settings_panel._map_position_btn.isEnabled()
        assert settings_panel._map_reset_btn.isEnabled()

    # -- "Position Map" toggling canvas mode --

    def test_position_map_calls_set_map_position_active(self, settings_panel):
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)

        mock_canvas = MagicMock()
        settings_panel._canvas_ref = mock_canvas

        settings_panel._map_position_btn.setChecked(True)

        mock_canvas.set_map_position_active.assert_called_with(True)

    def test_position_map_pushes_preview_data(self, settings_panel):
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)

        mock_canvas = MagicMock()
        settings_panel._canvas_ref = mock_canvas

        settings_panel._map_position_btn.setChecked(True)

        mock_canvas.set_map_preview_data.assert_called_once()
        # First positional arg should be the MapData
        args, _ = mock_canvas.set_map_preview_data.call_args
        assert args[0] is fake

    def test_position_map_pushes_current_view(self, settings_panel):
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)
        assert settings_panel._map_view is not None

        mock_canvas = MagicMock()
        settings_panel._canvas_ref = mock_canvas

        settings_panel._map_position_btn.setChecked(True)

        mock_canvas.update_map_view.assert_called_once_with(settings_panel._map_view)

    def test_position_map_unchecked_deactivates_canvas(self, settings_panel):
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)

        mock_canvas = MagicMock()
        settings_panel._canvas_ref = mock_canvas

        settings_panel._map_position_btn.setChecked(True)
        settings_panel._map_position_btn.setChecked(False)

        # Last call should be False
        last_call = mock_canvas.set_map_position_active.call_args
        assert last_call[0][0] is False

    # -- "Reset to fit" restores default view --

    def test_reset_to_fit_updates_map_view(self, settings_panel):
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)

        # Overwrite _map_view with a bogus value
        settings_panel._map_view = {"center_lat": 0.0, "center_lon": 0.0, "scale": 0.001}

        mock_canvas = MagicMock()
        settings_panel._canvas_ref = mock_canvas

        settings_panel._on_reset_to_fit_clicked()

        # Should have called update_map_view with the recomputed default view
        mock_canvas.update_map_view.assert_called_once()
        restored_view = mock_canvas.update_map_view.call_args[0][0]
        assert "center_lat" in restored_view
        assert "center_lon" in restored_view
        assert "scale" in restored_view
        # Scale should not be 0.001 (bogus value was replaced)
        assert restored_view["scale"] != 0.001

    def test_reset_to_fit_stores_new_view(self, settings_panel):
        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)

        original_view = dict(settings_panel._map_view)
        # Corrupt the view
        settings_panel._map_view = {"center_lat": 99.0, "center_lon": 99.0, "scale": 0.001}

        settings_panel._on_reset_to_fit_clicked()

        # _map_view should be restored close to original
        assert settings_panel._map_view is not None
        assert abs(settings_panel._map_view["scale"] - original_view["scale"]) < 1e-6

    def test_reset_to_fit_no_crash_without_canvas(self, settings_panel):
        """Clicking Reset to fit without a canvas_ref should not raise."""
        fake = _make_fake_map_data()
        settings_panel._on_map_fetch_finished(fake)
        settings_panel._canvas_ref = None
        try:
            settings_panel._on_reset_to_fit_clicked()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_on_reset_to_fit_clicked raised: {exc!r}")


# ---------------------------------------------------------------------------
# Canvas → panel sync and _map_view param injection — task 151.2
# ---------------------------------------------------------------------------


class TestCanvasMapViewSync:
    """_on_canvas_map_view_changed updates _map_view; get_params() injects it."""

    def test_canvas_signal_updates_map_view(self, settings_panel):
        """Calling _on_canvas_map_view_changed stores the view dict."""
        settings_panel._on_canvas_map_view_changed(35.01, 135.76, 0.5)
        view = settings_panel._map_view
        assert view is not None
        assert view["center_lat"] == pytest.approx(35.01)
        assert view["center_lon"] == pytest.approx(135.76)
        assert view["scale"] == pytest.approx(0.5)

    def test_canvas_signal_overwrites_previous_view(self, settings_panel):
        """A second emission replaces the previous view."""
        settings_panel._on_canvas_map_view_changed(10.0, 20.0, 1.0)
        settings_panel._on_canvas_map_view_changed(35.01, 135.76, 2.5)
        view = settings_panel._map_view
        assert view["center_lat"] == pytest.approx(35.01)
        assert view["scale"] == pytest.approx(2.5)

    def test_canvas_signal_persists_to_project_metadata(self, settings_panel, controller):
        """_on_canvas_map_view_changed writes map_view to project.metadata."""
        settings_panel._on_canvas_map_view_changed(35.01, 135.76, 0.5)
        project = controller.current_project
        assert "map_view" in project.metadata, (
            "project.metadata should contain 'map_view' after canvas sync"
        )
        stored = project.metadata["map_view"]
        assert stored["center_lat"] == pytest.approx(35.01)
        assert stored["center_lon"] == pytest.approx(135.76)
        assert stored["scale"] == pytest.approx(0.5)

    def test_canvas_signal_no_crash_without_controller(self, settings_panel):
        """Should not raise even if _controller is None."""
        settings_panel._controller = None
        try:
            settings_panel._on_canvas_map_view_changed(1.0, 2.0, 3.0)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"_on_canvas_map_view_changed raised: {exc!r}")
        # View should still be stored
        assert settings_panel._map_view is not None

    # -- get_params() injection --

    def test_get_params_injects_map_view_in_map_mode(self, settings_panel):
        """get_params() includes _map_view when in Map mode with a view set."""
        settings_panel._current_mode = "Map"
        settings_panel._map_view = {"center_lat": 35.0, "center_lon": 135.0, "scale": 1.0}
        params = settings_panel.get_params()
        assert "_map_view" in params
        assert params["_map_view"]["center_lat"] == pytest.approx(35.0)
        assert params["_map_view"]["center_lon"] == pytest.approx(135.0)
        assert params["_map_view"]["scale"] == pytest.approx(1.0)

    def test_get_params_map_view_is_a_copy(self, settings_panel):
        """get_params() returns a copy of _map_view, not the same object."""
        settings_panel._current_mode = "Map"
        view = {"center_lat": 35.0, "center_lon": 135.0, "scale": 1.0}
        settings_panel._map_view = view
        params = settings_panel.get_params()
        assert params["_map_view"] is not view

    def test_get_params_omits_map_view_in_non_map_mode(self, settings_panel):
        """get_params() must NOT inject _map_view outside Map mode."""
        settings_panel._current_mode = "Math Art"
        settings_panel._map_view = {"center_lat": 35.0, "center_lon": 135.0, "scale": 1.0}
        params = settings_panel.get_params()
        assert "_map_view" not in params

    def test_get_params_omits_map_view_when_none(self, settings_panel):
        """get_params() must NOT inject _map_view when _map_view is None."""
        settings_panel._current_mode = "Map"
        settings_panel._map_view = None
        params = settings_panel.get_params()
        assert "_map_view" not in params


class TestMapViewPersistence:
    """Persistence of map_view in project.metadata (spec §6.3)."""

    def test_view_writes_to_metadata(self, settings_panel, controller):
        """Setting a view via _on_canvas_map_view_changed writes metadata["map_view"]."""
        settings_panel._on_canvas_map_view_changed(35.01, 135.76, 2.5)
        project = controller.current_project
        assert "map_view" in project.metadata
        stored = project.metadata["map_view"]
        assert stored["center_lat"] == pytest.approx(35.01)
        assert stored["center_lon"] == pytest.approx(135.76)
        assert stored["scale"] == pytest.approx(2.5)

    def test_reload_restores_view_from_metadata(self, settings_panel, controller):
        """Round-trip: set a view, then simulate reload — _map_view is restored from metadata."""
        fake = _make_fake_map_data()
        # First load to initialise default view
        settings_panel._on_map_fetch_finished(fake)
        # Set a custom view (simulates the user panning/zooming)
        settings_panel._on_canvas_map_view_changed(35.01, 135.76, 2.5)

        # Sanity check: metadata is written
        project = controller.current_project
        assert project.metadata.get("map_view") is not None

        # Simulate a "reload" by resetting _map_view and re-calling the init
        settings_panel._map_view = None
        settings_panel._init_map_view_from_data(fake)

        assert settings_panel._map_view is not None
        view = settings_panel._map_view
        assert view["center_lat"] == pytest.approx(35.01)
        assert view["center_lon"] == pytest.approx(135.76)
        assert view["scale"] == pytest.approx(2.5)

    def test_fallback_to_default_when_no_metadata(self, settings_panel, controller):
        """When project.metadata has no "map_view", _init_map_view_from_data uses default_map_view."""
        fake = _make_fake_map_data()
        # Ensure no prior metadata
        project = controller.current_project
        project.metadata.pop("map_view", None)

        settings_panel._map_view = None
        settings_panel._init_map_view_from_data(fake)

        assert settings_panel._map_view is not None
        view = settings_panel._map_view
        # Default view must have all required keys
        assert "center_lat" in view
        assert "center_lon" in view
        assert "scale" in view
        # Scale must be positive (fit-to-canvas result, not a bogus value)
        assert view["scale"] > 0

    def test_fallback_to_default_when_metadata_malformed(self, settings_panel, controller):
        """Malformed metadata["map_view"] falls back to default_map_view instead of crashing."""
        fake = _make_fake_map_data()
        project = controller.current_project
        # Store something that is missing required keys
        project.metadata["map_view"] = {"bad_key": 99}

        settings_panel._map_view = None
        settings_panel._init_map_view_from_data(fake)

        assert settings_panel._map_view is not None
        view = settings_panel._map_view
        assert "center_lat" in view
        assert "center_lon" in view
        assert "scale" in view

    def test_metadata_stored_as_copy(self, settings_panel, controller):
        """project.metadata["map_view"] must be an independent copy, not the live dict."""
        settings_panel._on_canvas_map_view_changed(10.0, 20.0, 1.0)
        project = controller.current_project
        stored = project.metadata["map_view"]
        # Mutating _map_view must not affect the stored copy
        settings_panel._map_view["scale"] = 999.0
        assert project.metadata["map_view"]["scale"] == pytest.approx(1.0), (
            "metadata[map_view] must be an independent copy"
        )


class TestMultiLayerRunNotClobbered:
    """Regression: browsing/selecting layers of a multi-layer run (Map, Pixel Art)
    while the panel shows a single-layer generator must NOT overwrite the run
    members' generator_info. Clobbering destroyed the _generator_run_id link,
    causing re-generation to duplicate the map instead of replacing it."""

    def _project_with_map_run(self):
        canvas = Canvas.from_preset("A4", margin=10.0)
        p = Project(name="t", canvas=canvas)
        m1 = Layer(name="Roads (major)", color="#000000", paths=[[(10.0, 10.0), (50.0, 50.0)]],
                   generator_info={"_generator_name": "Map", "_generator_run_id": "R1"})
        m2 = Layer(name="Water", color="#1E6FD0", paths=[[(20.0, 20.0), (60.0, 60.0)]],
                   generator_info={"_generator_name": "Map", "_generator_run_id": "R1"})
        txt = Layer(name="My Text", color="#000000", paths=[[(5.0, 5.0), (9.0, 9.0)]])
        for l in (m1, m2, txt):
            p.add_layer(l)
        return p, m1, m2, txt

    def _panel(self, project, qtbot):
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel
        ctrl = ProjectController(project)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.on_mode_changed("Math Art")  # single-layer generator showing
        return ctrl, panel

    def test_browsing_map_layers_preserves_run_id(self, qtbot):
        project, m1, m2, txt = self._project_with_map_run()
        ctrl, panel = self._panel(project, qtbot)

        # Simulate the user clicking through every map layer, then a text layer.
        panel._on_active_layer_changed(m1.id)
        panel._on_active_layer_changed(m2.id)
        panel._on_active_layer_changed(txt.id)

        still_tagged = [
            l.id for l in project.layers
            if isinstance(l.generator_info, dict)
            and l.generator_info.get("_generator_run_id") == "R1"
        ]
        assert set(still_tagged) == {m1.id, m2.id}, (
            "both map-run layers must keep their _generator_run_id"
        )

    def test_single_layer_snapshot_still_saved(self, qtbot):
        """The guard must not break ordinary single-layer settings memory."""
        project, m1, m2, txt = self._project_with_map_run()
        ctrl, panel = self._panel(project, qtbot)

        # Start on the text layer, then switch away → its snapshot should save.
        panel._on_active_layer_changed(txt.id)
        panel._on_active_layer_changed(m1.id)

        info = project.get_layer(txt.id).generator_info
        assert isinstance(info, dict)
        assert info.get("mode") == "Math Art"
        assert "generator_name" in info


class TestMultiLayerRunRestoresOnSelect:
    """Selecting a map-run layer must restore the Map generator + the exact
    settings used to produce that run (so the user can edit the existing map
    instead of seeing the panel stuck on whatever single-layer generator was
    showing). Mirrors the 3D Scene / single-layer settings-memory behaviour."""

    def _build(self, qtbot, qapp):
        from plottter.osm.types import MapData, MapFeature
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        md = MapData(
            location="detroit, mi",
            center=(42.33, -83.05),
            bbox=(42.32, -83.06, 42.34, -83.04),
            features={
                "roads_major": [MapFeature(tags={"highway": "primary"},
                                           coords=[(42.32, -83.06), (42.34, -83.04)],
                                           is_area=False)],
                "water": [MapFeature(tags={"natural": "water"},
                                     coords=[(42.325, -83.05), (42.335, -83.05),
                                             (42.335, -83.045), (42.325, -83.045),
                                             (42.325, -83.05)],
                                     is_area=True)],
            },
        )
        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        # Mirror MainWindow's wiring so mode_change_requested actually
        # triggers on_mode_changed in this test.
        panel.mode_change_requested.connect(panel.on_mode_changed)
        return proj, ctrl, panel, md

    def _generate_map_run(self, panel, proj, md):
        """Set tweaked Map params, prime the cache to match the resulting key,
        then dispatch a fake multi-layer generation finish."""
        from plottter.generators.base import LayerSpec
        from plottter.generators.map_generator import MapGenerator
        from plottter.osm.cache import cache_key, store

        panel.on_mode_changed("Map")
        panel._map_location_edit.setText("detroit, mi")
        # Restrict to roads + water (matches the user's scenario; keeps the
        # cache key small and predictable).
        for k in ("include_rail", "include_waterways", "include_parks",
                  "include_buildings", "include_coastline"):
            panel._param_widgets[k].setChecked(False)
        # Tweak the params we expect the restore to bring back.
        panel._param_widgets["radius_km"].setValue(2.5)
        panel._param_widgets["area_fill"].setCurrentText("hatch")
        panel._param_widgets["fill_spacing_mm"].setValue(3.7)

        # Compute the cache key the same way the panel will on restore,
        # and prime the cache so reload succeeds.
        params = panel.get_params()
        cats = panel._get_enabled_map_categories(params)
        store(
            cache_key("detroit, mi", float(params["radius_km"]),
                      str(params["extent_mode"]), cats),
            md,
        )

        idx = panel._layer_combo.findData(proj.layers[0].id)
        panel._layer_combo.setCurrentIndex(idx)
        panel._generator = MapGenerator()
        panel._pending_multilayer_regen_run_id = None
        # Mirror _on_generate: settings are captured pre-removal (in the real
        # flow this happens in _on_generate, not in _on_multilayer_generation_finished).
        panel._pending_multilayer_run_settings = panel._capture_multilayer_run_settings()
        panel._on_multilayer_generation_finished([
            LayerSpec(name="Roads (major)", color="#000000", paths=[[(10.0, 10.0), (50.0, 50.0)]]),
            LayerSpec(name="Water", color="#1E6FD0", paths=[[(20.0, 20.0), (60.0, 60.0)]]),
        ])
        return [l for l in proj.layers
                if isinstance(l.generator_info, dict)
                and l.generator_info.get("_generator_name") == "Map"]

    def test_user_workflow_restores_full_settings(self, qtbot, qapp):
        """Reproduces the user's 5-step workflow: generate map → switch to a
        single-layer generator → click back on a map layer → expect Map + its
        exact settings restored."""
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_map_run(panel, proj, md)
        proj.add_layer(Layer(name="Layer 3"))
        panel.on_mode_changed("Math Art")
        assert panel._current_mode == "Math Art"  # in single-layer mode

        # Click back on a map layer.
        panel._on_active_layer_changed(map_layers[0].id)

        assert panel._current_mode == "Map"
        assert panel._generator_type_combo.currentText() == "Map"
        # The exact settings that produced the run, not the generator defaults.
        assert panel._param_widgets["radius_km"].value() == pytest.approx(2.5)
        assert panel._param_widgets["area_fill"].currentText() == "hatch"
        assert panel._param_widgets["fill_spacing_mm"].value() == pytest.approx(3.7)
        # Include flags that were turned off must stay off after restore.
        assert panel._param_widgets["include_buildings"].isChecked() is False
        assert panel._param_widgets["include_rail"].isChecked() is False
        assert panel._map_location_edit.text() == "detroit, mi"
        # _map_data reloaded from disk cache so a subsequent Generate works
        # offline (and replaces the run, since the run id is preserved).
        assert panel._map_data is not None

    def test_other_run_layer_restores_identically(self, qtbot, qapp):
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_map_run(panel, proj, md)
        panel.on_mode_changed("Math Art")
        # Selecting the second run layer must restore the same settings.
        panel._on_active_layer_changed(map_layers[1].id)
        assert panel._current_mode == "Map"
        assert panel._param_widgets["radius_km"].value() == pytest.approx(2.5)
        assert panel._param_widgets["area_fill"].currentText() == "hatch"

    def test_run_layer_carries_generator_settings(self, qtbot, qapp):
        """Each layer of a run carries the settings dict, so the user can
        select any of them to restore."""
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_map_run(panel, proj, md)
        for layer in map_layers:
            info = layer.generator_info
            assert isinstance(info, dict)
            assert "_generator_settings" in info
            settings = info["_generator_settings"]
            assert settings["mode"] == "Map"
            assert settings["generator_name"] == "Map"
            assert settings["location"] == "detroit, mi"
            assert settings["params"]["radius_km"] == pytest.approx(2.5)


class TestMultiLayerRegenerateKeepsRunSettings:
    """Regression: after regenerating a multi-layer run (e.g. enabling rails
    and clicking Generate while on a map layer), the panel must stay on the
    multi-layer generator AND the new run layers must carry the new run's
    settings — not whatever the layer panel auto-snapped to during the
    remove-old-layers step of the macro."""

    def _build(self, qtbot, qapp):
        from plottter.osm.types import MapData, MapFeature
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        md = MapData(
            location="detroit, mi",
            center=(42.33, -83.05),
            bbox=(42.32, -83.06, 42.34, -83.04),
            features={
                "roads_major": [MapFeature(tags={"highway": "primary"},
                                           coords=[(42.32, -83.06), (42.34, -83.04)],
                                           is_area=False)],
                "water": [MapFeature(tags={"natural": "water"},
                                     coords=[(42.325, -83.05), (42.335, -83.05),
                                             (42.335, -83.045), (42.325, -83.045),
                                             (42.325, -83.05)],
                                     is_area=True)],
            },
        )
        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.mode_change_requested.connect(panel.on_mode_changed)
        return proj, ctrl, panel, md

    def _store_cache(self, panel, md):
        from plottter.osm.cache import cache_key, store
        params = panel.get_params()
        cats = panel._get_enabled_map_categories(params)
        store(
            cache_key("detroit, mi", float(params["radius_km"]),
                      str(params["extent_mode"]), cats),
            md,
        )

    def _dispatch_run(self, panel, proj, specs):
        """Mirror _on_generate's pre-emit work, then dispatch the finish."""
        from plottter.generators.map_generator import MapGenerator

        # Mirror the regen-detection in _on_generate
        prior_run_id = None
        for layer in proj.layers:
            info = layer.generator_info
            if (isinstance(info, dict)
                and info.get("_generator_name") == "Map"
                and info.get("_generator_run_id")):
                prior_run_id = info["_generator_run_id"]
        panel._pending_multilayer_regen_run_id = prior_run_id
        # Capture settings BEFORE removal can side-effect the panel (the fix
        # this test guards): must happen in _on_generate, not the finish handler.
        panel._pending_multilayer_run_settings = panel._capture_multilayer_run_settings()
        panel._generator = MapGenerator()
        panel._on_multilayer_generation_finished(specs)

    def test_regenerate_stays_on_map_and_stores_correct_settings(self, qtbot, qapp):
        from plottter.generators.base import LayerSpec

        proj, ctrl, panel, md = self._build(qtbot, qapp)

        # 1. Generate map with streets + water (no rails)
        panel.on_mode_changed("Map")
        panel._map_location_edit.setText("detroit, mi")
        for k in ("include_rail", "include_waterways", "include_parks",
                  "include_buildings", "include_coastline"):
            panel._param_widgets[k].setChecked(False)
        panel._param_widgets["radius_km"].setValue(2.5)
        self._store_cache(panel, md)
        idx = panel._layer_combo.findData(proj.layers[0].id)
        panel._layer_combo.setCurrentIndex(idx)
        self._dispatch_run(panel, proj, [
            LayerSpec(name="Roads (major)", color="#000000", paths=[[(10.0, 10.0), (50.0, 50.0)]]),
            LayerSpec(name="Water", color="#1E6FD0", paths=[[(20.0, 20.0), (60.0, 60.0)]]),
        ])

        # 2. Create new layer, generate parametric (simulate via Math Art mode)
        panel.on_mode_changed("Math Art")
        new_layer = Layer(name="Layer 3")
        proj.add_layer(new_layer)
        snap = panel._get_settings_snapshot()
        if snap:
            ctrl.set_layer_generator_info(new_layer.id, snap)
        ctrl.set_active_layer(new_layer.id)
        panel._on_active_layer_changed(new_layer.id)
        assert panel._current_mode == "Math Art"

        # 3. Click a map layer — restores Map.
        map_layers = [l for l in proj.layers
                      if isinstance(l.generator_info, dict)
                      and l.generator_info.get("_generator_name") == "Map"]
        ctrl.set_active_layer(map_layers[0].id)
        panel._on_active_layer_changed(map_layers[0].id)
        assert panel._current_mode == "Map"
        assert panel._param_widgets["include_rail"].isChecked() is False

        # 4. Add Rails, regenerate. Panel must STAY on Map afterwards, and the
        #    new run layers must carry settings with include_rail=True.
        panel._param_widgets["include_rail"].setChecked(True)
        self._store_cache(panel, md)
        self._dispatch_run(panel, proj, [
            LayerSpec(name="Roads (major)", color="#000000", paths=[[(10.0, 10.0), (50.0, 50.0)]]),
            LayerSpec(name="Water", color="#1E6FD0", paths=[[(20.0, 20.0), (60.0, 60.0)]]),
            LayerSpec(name="Rail", color="#7A4A2B", paths=[[(15.0, 15.0), (45.0, 45.0)]]),
        ])

        # Panel must stay on Map (regression: previously snapped to Parametric).
        assert panel._current_mode == "Map", (
            f"Panel snapped away from Map after regenerate: {panel._current_mode}"
        )
        assert panel._param_widgets["include_rail"].isChecked() is True

        # And the new run layers must carry the CURRENT (Map+Rails) settings,
        # not whatever the panel briefly snapped to during layer removal.
        new_map_layers = [l for l in proj.layers
                          if isinstance(l.generator_info, dict)
                          and l.generator_info.get("_generator_name") == "Map"]
        assert len(new_map_layers) == 3
        for layer in new_map_layers:
            gs = layer.generator_info.get("_generator_settings")
            assert isinstance(gs, dict)
            assert gs.get("mode") == "Map", (
                f"_generator_settings.mode is {gs.get('mode')}, expected 'Map' "
                f"(capture happened too late and recorded the wrong generator)"
            )
            assert gs.get("generator_name") == "Map"
            assert gs.get("params", {}).get("include_rail") is True

        # 5. After regen, selecting any map layer again must restore the new
        #    settings (include_rail=True, not the pre-regen False).
        panel.on_mode_changed("Math Art")
        panel._on_active_layer_changed(new_map_layers[0].id)
        assert panel._current_mode == "Map"
        assert panel._param_widgets["include_rail"].isChecked() is True


# ---------------------------------------------------------------------------
# _get_enabled_map_categories — places category — task 155.3
# ---------------------------------------------------------------------------


class TestGetEnabledMapCategoriesPlaces:
    """_get_enabled_map_categories appends 'places' iff include_place_labels is True."""

    def test_places_included_when_true(self, settings_panel):
        cats = settings_panel._get_enabled_map_categories({"include_place_labels": True})
        assert "places" in cats

    def test_places_included_by_default(self, settings_panel):
        """Default value is True, so omitting the key must still include 'places'."""
        cats = settings_panel._get_enabled_map_categories({})
        assert "places" in cats

    def test_places_excluded_when_false(self, settings_panel):
        cats = settings_panel._get_enabled_map_categories({"include_place_labels": False})
        assert "places" not in cats

    def test_toggling_changes_cache_key(self, settings_panel):
        """Different include_place_labels values must produce different cache keys."""
        from plottter.osm.cache import cache_key

        cats_on = settings_panel._get_enabled_map_categories({"include_place_labels": True})
        cats_off = settings_panel._get_enabled_map_categories({"include_place_labels": False})

        key_on = cache_key("TestCity", 1.5, "radius", cats_on)
        key_off = cache_key("TestCity", 1.5, "radius", cats_off)

        assert key_on != key_off, (
            "cache key must differ when include_place_labels toggles"
        )


class TestLabelParamRestoreOnSelect:
    """Label params (include_*_labels, label_font_size_mm, label_min_feature_mm,
    label_language) live in ordinary ``_param_widgets`` entries and must be
    captured by ``_capture_multilayer_run_settings`` / restored by
    ``_apply_multilayer_run_settings`` just like any other map param.

    Regression guard for task 156.1: setting label params away from their
    defaults, generating a run, switching modes, then selecting a run layer
    must restore every label param to the tweaked value.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build(self, qtbot, qapp):
        from plottter.osm.types import MapData, MapFeature
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        md = MapData(
            location="detroit, mi",
            center=(42.33, -83.05),
            bbox=(42.32, -83.06, 42.34, -83.04),
            features={
                "roads_major": [MapFeature(tags={"highway": "primary"},
                                           coords=[(42.32, -83.06), (42.34, -83.04)],
                                           is_area=False)],
                "water": [MapFeature(tags={"natural": "water"},
                                     coords=[(42.325, -83.05), (42.335, -83.05),
                                             (42.335, -83.045), (42.325, -83.045),
                                             (42.325, -83.05)],
                                     is_area=True)],
            },
        )
        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.mode_change_requested.connect(panel.on_mode_changed)
        return proj, ctrl, panel, md

    def _generate_label_run(self, panel, proj, md):
        """Set label params away from their defaults, prime the cache, dispatch
        a fake multi-layer generation finish, and return the resulting run layers."""
        from plottter.generators.base import LayerSpec
        from plottter.generators.map_generator import MapGenerator
        from plottter.osm.cache import cache_key, store

        panel.on_mode_changed("Map")
        panel._map_location_edit.setText("detroit, mi")

        # Tweak every label param away from its default value so we can
        # detect any param that is silently dropped from the snapshot loop.
        panel._param_widgets["include_water_labels"].setChecked(False)    # default True
        panel._param_widgets["include_park_labels"].setChecked(False)     # default True
        panel._param_widgets["include_waterway_labels"].setChecked(True)  # default False
        panel._param_widgets["include_place_labels"].setChecked(False)    # default True
        panel._param_widgets["include_road_labels"].setChecked(True)      # default False
        panel._param_widgets["label_font_size_mm"].setValue(5.0)          # default 3.5
        panel._param_widgets["label_min_feature_mm"].setValue(12.0)       # default 8.0
        panel._param_widgets["label_language"].setText("en")              # default ""

        # Compute cache key from the tweaked panel state and prime the cache.
        params = panel.get_params()
        cats = panel._get_enabled_map_categories(params)
        store(
            cache_key("detroit, mi", float(params["radius_km"]),
                      str(params["extent_mode"]), cats),
            md,
        )

        idx = panel._layer_combo.findData(proj.layers[0].id)
        panel._layer_combo.setCurrentIndex(idx)
        panel._generator = MapGenerator()
        panel._pending_multilayer_regen_run_id = None
        panel._pending_multilayer_run_settings = panel._capture_multilayer_run_settings()
        panel._on_multilayer_generation_finished([
            LayerSpec(name="Roads (major)", color="#000000",
                      paths=[[(10.0, 10.0), (50.0, 50.0)]]),
            LayerSpec(name="Water", color="#1E6FD0",
                      paths=[[(20.0, 20.0), (60.0, 60.0)]]),
        ])
        return [
            layer for layer in proj.layers
            if isinstance(layer.generator_info, dict)
            and layer.generator_info.get("_generator_name") == "Map"
        ]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_label_params_restored_after_mode_switch(self, qtbot, qapp):
        """Generate a map run with non-default label params → switch mode →
        select a run layer → every label param must be restored to the tweaked
        value (not the generator default)."""
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_label_run(panel, proj, md)

        panel.on_mode_changed("Math Art")
        assert panel._current_mode == "Math Art"

        panel._on_active_layer_changed(map_layers[0].id)

        assert panel._current_mode == "Map"
        assert panel._param_widgets["include_water_labels"].isChecked() is False
        assert panel._param_widgets["include_park_labels"].isChecked() is False
        assert panel._param_widgets["include_waterway_labels"].isChecked() is True
        assert panel._param_widgets["include_place_labels"].isChecked() is False
        assert panel._param_widgets["include_road_labels"].isChecked() is True
        assert panel._param_widgets["label_font_size_mm"].value() == pytest.approx(5.0)
        assert panel._param_widgets["label_min_feature_mm"].value() == pytest.approx(12.0)
        assert panel._param_widgets["label_language"].text() == "en"

    def test_label_params_in_settings_snapshot(self, qtbot, qapp):
        """Every run layer's ``_generator_settings`` must carry the tweaked label
        param values so any layer of the run can restore the full configuration."""
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_label_run(panel, proj, md)

        for layer in map_layers:
            settings = layer.generator_info.get("_generator_settings", {})
            params = settings.get("params", {})
            assert params.get("include_water_labels") is False
            assert params.get("include_park_labels") is False
            assert params.get("include_waterway_labels") is True
            assert params.get("include_place_labels") is False
            assert params.get("include_road_labels") is True
            assert params.get("label_font_size_mm") == pytest.approx(5.0)
            assert params.get("label_min_feature_mm") == pytest.approx(12.0)
            assert params.get("label_language") == "en"

    def test_second_run_layer_restores_label_params(self, qtbot, qapp):
        """Selecting the *second* run layer (not just the first) must also
        restore every label param — all run layers carry the same snapshot."""
        proj, ctrl, panel, md = self._build(qtbot, qapp)
        map_layers = self._generate_label_run(panel, proj, md)
        assert len(map_layers) >= 2, "need ≥2 run layers for this test"

        panel.on_mode_changed("Math Art")
        panel._on_active_layer_changed(map_layers[1].id)

        assert panel._current_mode == "Map"
        assert panel._param_widgets["include_waterway_labels"].isChecked() is True
        assert panel._param_widgets["include_road_labels"].isChecked() is True
        assert panel._param_widgets["label_font_size_mm"].value() == pytest.approx(5.0)
        assert panel._param_widgets["label_language"].text() == "en"


class TestFetchWorkerLifetime:
    """Regression: the worker's run() emits `finished` as its last action, so
    when the queued slot fires on the main thread the worker thread is still
    wrapping up. Dropping the QThread reference there used to abort with
    'QThread: Destroyed while thread is still running' + SIGABRT. The fix is
    to call worker.wait() before clearing the reference."""

    def test_finished_handler_waits_before_clearing_ref(self, qtbot, qapp):
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.on_mode_changed("Map")

        mock_worker = MagicMock()
        mock_worker.wait = MagicMock(return_value=True)
        panel._map_fetch_worker = mock_worker

        fake_map_data = _make_fake_map_data()
        panel._on_map_fetch_finished(fake_map_data)

        mock_worker.wait.assert_called_once()
        # Ref cleared only AFTER wait() — and the only acceptable order is
        # capture local, clear attr, wait. So after the slot returns the
        # attribute is None and wait was called exactly once on the original.
        assert panel._map_fetch_worker is None

    def test_error_handler_waits_before_clearing_ref(self, qtbot, qapp):
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.on_mode_changed("Map")

        mock_worker = MagicMock()
        mock_worker.wait = MagicMock(return_value=True)
        panel._map_fetch_worker = mock_worker

        panel._on_map_fetch_error("boom")

        mock_worker.wait.assert_called_once()
        assert panel._map_fetch_worker is None

    def test_handlers_tolerate_already_cleared_worker(self, qtbot, qapp):
        """Edge case: handler called when _map_fetch_worker is already None.
        Must not raise (no wait() call possible, but nothing should crash)."""
        from plottter.gui.project_controller import ProjectController
        from plottter.gui.settings_panel import SettingsPanel

        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="t", canvas=canvas)
        proj.add_layer(Layer(name="Layer 1"))
        ctrl = ProjectController(proj)
        panel = SettingsPanel(ctrl)
        qtbot.addWidget(panel)
        panel.on_mode_changed("Map")

        panel._map_fetch_worker = None
        # Should not raise
        panel._on_map_fetch_finished(_make_fake_map_data())
        panel._on_map_fetch_error("boom")
