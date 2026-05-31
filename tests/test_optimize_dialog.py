"""Tests for OptimizeSettingsDialog — run_join / join_threshold wiring.

Covers:
(a) Toggling the "Join paths at junctions" checkbox round-trips through QSettings.
(b) _OptimizeWorker accepts and uses run_join / join_threshold flags.
(c) With both Merge and Join enabled, Join runs after Merge (verified via mock).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def dlg(qapp, tmp_path, monkeypatch):
    """OptimizeSettingsDialog backed by a temporary QSettings store."""
    from PyQt6.QtCore import QSettings

    # Redirect QSettings to a temp INI file so tests don't touch real prefs.
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = QSettings(str(tmp_path / "test.ini"), QSettings.Format.IniFormat)

    def _make_settings(*args, **kwargs):
        return settings

    with patch("plottter.gui.dialogs.optimize_dialog.QSettings", side_effect=_make_settings):
        from plottter.gui.dialogs.optimize_dialog import OptimizeSettingsDialog

        d = OptimizeSettingsDialog()
        yield d
        d.close()


# ---------------------------------------------------------------------------
# (a) QSettings round-trip for run_join / join_threshold
# ---------------------------------------------------------------------------


class TestJoinCheckboxQSettingsRoundTrip:
    def test_join_checkbox_off_by_default(self, dlg):
        """run_join must default to False."""
        assert not dlg._join_check.isChecked()

    def test_join_threshold_default_value(self, dlg):
        """join_threshold must default to 0.1 mm."""
        assert dlg._join_thresh_spin.value() == pytest.approx(0.1, abs=1e-6)

    def test_get_settings_includes_run_join(self, dlg):
        """get_settings() must return run_join key."""
        settings = dlg.get_settings()
        assert "run_join" in settings

    def test_get_settings_includes_join_threshold(self, dlg):
        """get_settings() must return join_threshold key."""
        settings = dlg.get_settings()
        assert "join_threshold" in settings

    def test_get_settings_run_join_reflects_checkbox(self, dlg):
        """get_settings()['run_join'] must match checkbox state."""
        dlg._join_check.setChecked(True)
        assert dlg.get_settings()["run_join"] is True
        dlg._join_check.setChecked(False)
        assert dlg.get_settings()["run_join"] is False

    def test_get_settings_join_threshold_reflects_spinbox(self, dlg):
        """get_settings()['join_threshold'] must match spinbox value."""
        dlg._join_thresh_spin.setValue(0.5)
        assert dlg.get_settings()["join_threshold"] == pytest.approx(0.5, abs=1e-6)

    def test_restore_defaults_resets_join_checkbox(self, dlg):
        """_restore_defaults() must reset join checkbox to False."""
        dlg._join_check.setChecked(True)
        dlg._restore_defaults()
        assert not dlg._join_check.isChecked()

    def test_restore_defaults_resets_join_threshold(self, dlg):
        """_restore_defaults() must reset join threshold to 0.1."""
        dlg._join_thresh_spin.setValue(2.0)
        dlg._restore_defaults()
        assert dlg._join_thresh_spin.value() == pytest.approx(0.1, abs=1e-6)

    def test_join_spinbox_disabled_when_unchecked(self, dlg):
        """join_threshold spinbox must be disabled when checkbox is unchecked."""
        dlg._join_check.setChecked(False)
        assert not dlg._join_thresh_spin.isEnabled()

    def test_join_spinbox_enabled_when_checked(self, dlg):
        """join_threshold spinbox must be enabled when checkbox is checked."""
        dlg._join_check.setChecked(True)
        assert dlg._join_thresh_spin.isEnabled()


# ---------------------------------------------------------------------------
# (b) _OptimizeWorker accepts and uses run_join / join_threshold
# ---------------------------------------------------------------------------


class TestOptimizeWorkerJoinParams:
    def test_worker_accepts_run_join_false(self, qapp):
        """_OptimizeWorker must accept run_join=False without error."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        worker = _OptimizeWorker(
            paths=[[(0.0, 0.0), (1.0, 0.0)]],
            run_join=False,
            join_threshold=0.1,
        )
        assert worker._run_join is False
        assert worker._join_threshold == pytest.approx(0.1)

    def test_worker_accepts_run_join_true(self, qapp):
        """_OptimizeWorker must accept run_join=True and store it."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        worker = _OptimizeWorker(
            paths=[[(0.0, 0.0), (1.0, 0.0)]],
            run_join=True,
            join_threshold=0.25,
        )
        assert worker._run_join is True
        assert worker._join_threshold == pytest.approx(0.25)

    def test_worker_defaults_run_join_to_false(self, qapp):
        """_OptimizeWorker.run_join must default to False."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        worker = _OptimizeWorker(paths=[[(0.0, 0.0), (1.0, 0.0)]])
        assert worker._run_join is False

    def test_worker_join_called_when_enabled(self, qapp):
        """When run_join=True, join_at_junctions must be called during run()."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        paths = [[(0.0, 0.0), (1.0, 0.0)]]
        worker = _OptimizeWorker(
            paths=paths,
            run_join=True,
            join_threshold=0.1,
            run_simplify=False,
            run_filter=False,
            run_clip=False,
            run_merge=False,
            run_2opt=False,
            run_3opt=False,
            run_or_opt=False,
        )

        called_with = {}

        def fake_join(p, threshold_mm=0.1):
            called_with["threshold_mm"] = threshold_mm
            return p

        with patch("plottter.processing.join_at_junctions", side_effect=fake_join):
            worker.run()

        assert "threshold_mm" in called_with
        assert called_with["threshold_mm"] == pytest.approx(0.1)

    def test_worker_join_not_called_when_disabled(self, qapp):
        """When run_join=False, join_at_junctions must NOT be called."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        paths = [[(0.0, 0.0), (1.0, 0.0)]]
        worker = _OptimizeWorker(
            paths=paths,
            run_join=False,
            run_simplify=False,
            run_filter=False,
            run_clip=False,
            run_merge=False,
            run_2opt=False,
            run_3opt=False,
            run_or_opt=False,
        )

        join_mock = MagicMock(return_value=paths)
        with patch("plottter.processing.join_at_junctions", join_mock):
            worker.run()

        join_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (c) Ordering: Join runs AFTER Merge (verified via mock call order)
# ---------------------------------------------------------------------------


class TestJoinRunsAfterMerge:
    def test_join_called_after_merge(self, qapp):
        """When both run_merge=True and run_join=True, merge must be called before join."""
        from plottter.gui.main_window.workers import _OptimizeWorker

        paths = [[(0.0, 0.0), (1.0, 0.0)]]
        worker = _OptimizeWorker(
            paths=paths,
            run_merge=True,
            merge_threshold=0.5,
            run_join=True,
            join_threshold=0.1,
            run_simplify=False,
            run_filter=False,
            run_clip=False,
            run_2opt=False,
            run_3opt=False,
            run_or_opt=False,
        )

        call_order: list[str] = []

        def fake_merge(p, threshold_mm=0.5):
            call_order.append("merge")
            return p

        def fake_join(p, threshold_mm=0.1):
            call_order.append("join")
            return p

        with patch("plottter.processing.merge_nearby_paths", side_effect=fake_merge), \
             patch("plottter.processing.join_at_junctions", side_effect=fake_join):
            worker.run()

        assert "merge" in call_order
        assert "join" in call_order
        assert call_order.index("merge") < call_order.index("join"), (
            f"Expected merge before join, got order: {call_order}"
        )
