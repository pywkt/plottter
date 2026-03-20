"""Tests for 3-opt integration: pipeline wiring, dialog checkbox, and distance reduction."""

from __future__ import annotations

import math
import random
import sys

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    """Ensure a QApplication exists for tests that create Qt objects."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _travel(paths: list) -> float:
    """Total pen-up travel distance from origin through all paths."""
    if not paths:
        return 0.0
    dist = math.dist((0.0, 0.0), paths[0][0])
    for i in range(len(paths) - 1):
        dist += math.dist(paths[i][-1], paths[i + 1][0])
    dist += math.dist(paths[-1][-1], (0.0, 0.0))
    return dist


def _make_random_paths(n: int, seed: int = 42) -> list:
    """Return n single-point 'paths' arranged in a shuffled order."""
    rng = random.Random(seed)
    pts = [(float(i % 20) * 5, float(i // 20) * 5) for i in range(n)]
    rng.shuffle(pts)
    return [[(x, y), (x + 0.1, y + 0.1)] for x, y in pts]


# ---------------------------------------------------------------------------
# (a) 3-opt reduces travel distance on random points
# ---------------------------------------------------------------------------

def test_3opt_reduces_travel():
    """optimize_3opt should not increase total travel distance."""
    from plottter.processing import reorder_paths, optimize_3opt

    paths = _make_random_paths(60, seed=7)
    ordered = reorder_paths(paths, num_starts=1)
    before = _travel(ordered)

    result = optimize_3opt(ordered)
    after = _travel(result)

    assert after <= before + 1e-6, (
        f"3-opt worsened travel: {before:.2f} -> {after:.2f}"
    )
    assert len(result) == len(ordered)


def test_3opt_improves_bad_ordering():
    """3-opt should improve a deliberately bad zigzag ordering."""
    from plottter.processing import optimize_3opt

    paths = []
    for i in range(30):
        x = 0.0 if i % 2 == 0 else 100.0
        y = float(i) * 2
        paths.append([(x, y), (x + 0.5, y + 0.5)])

    before = _travel(paths)
    result = optimize_3opt(paths)
    after = _travel(result)

    assert after < before, (
        f"3-opt should improve zigzag: {before:.2f} -> {after:.2f}"
    )


# ---------------------------------------------------------------------------
# (b) Optimize dialog includes the 3-opt checkbox
# ---------------------------------------------------------------------------

def test_dialog_has_3opt_checkbox(qapp):
    """OptimizeSettingsDialog must expose a run_3opt setting."""
    from plottter.gui.dialogs.optimize_dialog import OptimizeSettingsDialog

    dlg = OptimizeSettingsDialog()
    settings = dlg.get_settings()

    assert "run_3opt" in settings, "get_settings() must include 'run_3opt'"
    assert hasattr(dlg, "_opt3_check"), "Dialog must have _opt3_check attribute"


def test_dialog_3opt_default_false():
    """run_3opt default value must be False."""
    from plottter.gui.dialogs.optimize_dialog import _DEFAULTS

    assert _DEFAULTS["run_3opt"] is False


def test_dialog_3opt_tooltip(qapp):
    """The 3-opt checkbox tooltip must match the spec."""
    from plottter.gui.dialogs.optimize_dialog import OptimizeSettingsDialog

    dlg = OptimizeSettingsDialog()
    expected = (
        "3-opt — finds improvements 2-opt misses, slower. "
        "For stipple/dot art with 1000+ paths."
    )
    assert dlg._opt3_check.toolTip() == expected


# ---------------------------------------------------------------------------
# (c) Disabling 3-opt skips it in the worker pipeline
# ---------------------------------------------------------------------------

def test_3opt_skipped_when_disabled(qapp, monkeypatch):
    """When run_3opt=False, optimize_3opt should never be called."""
    import plottter.processing as proc

    called = []
    original = proc.optimize_3opt

    def spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(proc, "optimize_3opt", spy)

    from plottter.gui.main_window import _OptimizeWorker

    worker = _OptimizeWorker(
        paths=_make_random_paths(20, seed=1),
        run_weld=False,
        run_simplify=False,
        run_filter=False,
        run_clip=False,
        run_merge=False,
        run_2opt=False,
        run_3opt=False,
        run_or_opt=False,
        num_starts=1,
    )
    worker.run()

    assert called == [], "optimize_3opt should not be called when run_3opt=False"


def test_3opt_called_when_enabled(qapp, monkeypatch):
    """When run_3opt=True, optimize_3opt must be called in the pipeline."""
    import plottter.processing as proc

    called = []
    original = proc.optimize_3opt

    def spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(proc, "optimize_3opt", spy)

    from plottter.gui.main_window import _OptimizeWorker

    worker = _OptimizeWorker(
        paths=_make_random_paths(20, seed=2),
        run_weld=False,
        run_simplify=False,
        run_filter=False,
        run_clip=False,
        run_merge=False,
        run_2opt=False,
        run_3opt=True,
        run_or_opt=False,
        num_starts=1,
    )
    worker.run()

    assert called, "optimize_3opt should be called when run_3opt=True"
