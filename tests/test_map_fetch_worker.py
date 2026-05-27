"""Tests for _MapFetchWorker — task 146.2.

Covers:
- Patched success: finished signal emits a MapData instance.
- Patched failure: error signal emits a string; no crash.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_map_data():
    """Return a minimal MapData instance for use as a fake network result."""
    from plottter.osm.types import MapData

    return MapData(
        location="Kyoto, Japan",
        center=(35.0116, 135.7681),
        bbox=(34.9616, 135.7181, 35.0616, 135.8181),
        features={},
    )


def _make_worker(**kwargs):
    """Construct a _MapFetchWorker with sensible defaults."""
    from plottter.gui.settings_panel.workers import _MapFetchWorker

    defaults = dict(
        location="Kyoto, Japan",
        radius_km=1.5,
        extent_mode="center_radius",
        selectors=["roads_major", "water"],
        endpoint="https://overpass-api.de/api/interpreter",
        cache_dir=None,
    )
    defaults.update(kwargs)
    return _MapFetchWorker(**defaults)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestMapFetchWorkerSuccess:
    """Patched fetch_map_data returning MapData → finished emitted with it."""

    def test_finished_emitted(self, qtbot) -> None:
        """finished signal carries the MapData returned by fetch_map_data."""
        import plottter.osm as osm_mod

        fake = _make_fake_map_data()
        worker = _make_worker()

        received = []
        worker.finished.connect(received.append)

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            worker.run()

        assert len(received) == 1, "finished should fire exactly once"

    def test_finished_payload_is_map_data(self, qtbot) -> None:
        """Payload emitted by finished is a MapData instance."""
        from plottter.osm.types import MapData
        import plottter.osm as osm_mod

        fake = _make_fake_map_data()
        worker = _make_worker()

        received = []
        worker.finished.connect(received.append)

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            worker.run()

        assert isinstance(received[0], MapData)

    def test_finished_payload_has_correct_location(self, qtbot) -> None:
        """MapData emitted by finished has the location the worker was given."""
        import plottter.osm as osm_mod

        fake = _make_fake_map_data()
        worker = _make_worker(location="Kyoto, Japan")

        received = []
        worker.finished.connect(received.append)

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            worker.run()

        assert received[0].location == "Kyoto, Japan"

    def test_error_not_emitted_on_success(self, qtbot) -> None:
        """error signal must not fire when fetch succeeds."""
        import plottter.osm as osm_mod

        fake = _make_fake_map_data()
        worker = _make_worker()

        errors = []
        worker.error.connect(errors.append)

        with patch.object(osm_mod, "fetch_map_data", return_value=fake):
            worker.run()

        assert errors == [], f"error should not fire on success; got {errors}"

    def test_fetch_map_data_called_with_correct_args(self, qtbot) -> None:
        """fetch_map_data receives the location, radius, extent_mode, and endpoint."""
        import plottter.osm as osm_mod
        from unittest.mock import MagicMock

        fake = _make_fake_map_data()
        mock_fn = MagicMock(return_value=fake)
        worker = _make_worker(
            location="Brooklyn, NY",
            radius_km=2.0,
            extent_mode="place_bbox",
            selectors=["roads_major"],
            endpoint="https://overpass.kumi.systems/api/interpreter",
        )

        with patch.object(osm_mod, "fetch_map_data", mock_fn):
            worker.run()

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        # positional: location
        assert call_kwargs.args[0] == "Brooklyn, NY"
        assert call_kwargs.kwargs["radius_km"] == 2.0
        assert call_kwargs.kwargs["extent_mode"] == "place_bbox"
        assert call_kwargs.kwargs["enabled_categories"] == ["roads_major"]
        assert call_kwargs.kwargs["endpoint"] == "https://overpass.kumi.systems/api/interpreter"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestMapFetchWorkerError:
    """Patched fetch_map_data raising → error emitted; finished must not fire."""

    def test_error_emitted_on_exception(self, qtbot) -> None:
        """error signal fires when fetch_map_data raises."""
        import plottter.osm as osm_mod

        worker = _make_worker()

        errors = []
        worker.error.connect(errors.append)

        with patch.object(
            osm_mod, "fetch_map_data", side_effect=RuntimeError("network down")
        ):
            worker.run()  # must not raise

        assert len(errors) == 1, "error should fire exactly once on exception"

    def test_error_message_contains_exception_text(self, qtbot) -> None:
        """The error string includes the original exception message."""
        import plottter.osm as osm_mod

        worker = _make_worker()

        errors = []
        worker.error.connect(errors.append)

        with patch.object(
            osm_mod,
            "fetch_map_data",
            side_effect=ValueError("Could not geocode location: 'Nowhere'"),
        ):
            worker.run()

        assert "geocode" in errors[0].lower() or "Nowhere" in errors[0]

    def test_finished_not_emitted_on_error(self, qtbot) -> None:
        """finished signal must not fire when an exception is raised."""
        import plottter.osm as osm_mod

        worker = _make_worker()

        finished = []
        worker.finished.connect(finished.append)

        with patch.object(
            osm_mod, "fetch_map_data", side_effect=OSError("connection refused")
        ):
            worker.run()

        assert finished == [], "finished must not emit when fetch raises"

    def test_no_crash_on_exception(self, qtbot) -> None:
        """worker.run() completes without raising even when fetch fails."""
        import plottter.osm as osm_mod

        worker = _make_worker()

        # run() should not propagate the exception — it catches all Exception
        try:
            with patch.object(
                osm_mod, "fetch_map_data", side_effect=Exception("boom")
            ):
                worker.run()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"worker.run() should not raise; got {exc!r}")


# ---------------------------------------------------------------------------
# Signal interface
# ---------------------------------------------------------------------------


class TestMapFetchWorkerInterface:
    """_MapFetchWorker has the expected QThread interface."""

    def test_is_qthread_subclass(self, qtbot) -> None:
        from PyQt6.QtCore import QThread
        from plottter.gui.settings_panel.workers import _MapFetchWorker

        assert issubclass(_MapFetchWorker, QThread)

    def test_has_progress_signal(self, qtbot) -> None:
        from plottter.gui.settings_panel.workers import _MapFetchWorker

        worker = _make_worker()
        # Connecting to an int signal should succeed without error.
        worker.progress.connect(lambda pct: None)

    def test_has_finished_signal(self, qtbot) -> None:
        from plottter.gui.settings_panel.workers import _MapFetchWorker

        worker = _make_worker()
        worker.finished.connect(lambda obj: None)

    def test_has_error_signal(self, qtbot) -> None:
        from plottter.gui.settings_panel.workers import _MapFetchWorker

        worker = _make_worker()
        worker.error.connect(lambda msg: None)

    def test_no_parent_by_default(self, qtbot) -> None:
        """Worker is constructed without a Qt parent (prevents double-free)."""
        from plottter.gui.settings_panel.workers import _MapFetchWorker

        worker = _MapFetchWorker(
            location="Paris",
            radius_km=1.0,
            extent_mode="center_radius",
            selectors=[],
            endpoint="https://overpass-api.de/api/interpreter",
        )
        assert worker.parent() is None
