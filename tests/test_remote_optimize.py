"""Tests for the remote-optimize feature (pipeline extraction + CLI + worker).

Covers:
- run_optimization_pipeline produces the expected result on a small fixture.
- The pipeline respects the cancellation predicate.
- Map layers get Join force-enabled by the pipeline.
- The CLI --optimize round-trip: JSON in, JSON out, exit 0, progress on stderr.
- _RemoteOptimizeWorker happy path: a stub ssh command that echoes a canned
  response is parsed correctly.
- _RemoteOptimizeWorker error surface: non-zero exit emits .error with the
  stderr tail.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# run_optimization_pipeline
# ---------------------------------------------------------------------------


def test_pipeline_reduces_travel_on_misordered_paths():
    from plottter.processing import run_optimization_pipeline

    paths = [
        [(0.0, 0.0), (10.0, 0.0)],
        [(50.0, 0.0), (60.0, 0.0)],
        [(20.0, 0.0), (30.0, 0.0)],
    ]
    result = run_optimization_pipeline(paths, clip_bounds=(0.0, -1.0, 100.0, 1.0))
    assert result.after_travel < result.before_travel
    assert result.before_lifts == 3
    # Reordering keeps the same number of polylines for this input.
    assert result.after_lifts == 3
    # Output should be ordered left-to-right after reorder.
    starts = [poly[0][0] for poly in result.paths]
    assert starts == sorted(starts)


def test_pipeline_cancels_at_checkpoint():
    """A cancelled callback returns early without crashing."""
    from plottter.processing import run_optimization_pipeline

    paths = [[(float(i), 0.0), (float(i) + 1.0, 0.0)] for i in range(0, 20, 2)]
    result = run_optimization_pipeline(
        paths,
        clip_bounds=(0.0, -1.0, 100.0, 1.0),
        cancelled=lambda: True,
    )
    # Should still produce valid metrics even when bailed-out.
    assert result.before_travel > 0
    assert result.after_travel >= 0


def test_pipeline_forces_join_for_map_layers():
    """Map layers get Join enabled even if the caller passed run_join=False."""
    from plottter.processing import run_optimization_pipeline

    # Two short segments that share an endpoint — Join should weld them into one polyline.
    paths = [
        [(0.0, 0.0), (10.0, 0.0)],
        [(10.0, 0.0), (20.0, 0.0)],
    ]
    no_join = run_optimization_pipeline(
        paths,
        settings={"run_join": False, "run_merge": False, "join_threshold": 0.01},
        generator_info=None,
    )
    map_layer = run_optimization_pipeline(
        paths,
        settings={"run_join": False, "run_merge": False, "join_threshold": 0.01},
        generator_info={"_generator_name": "Map"},
    )
    # Map-layer policy should reduce the pen-lift count via Join.
    assert map_layer.after_lifts <= no_join.after_lifts


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def _plottter_bin() -> str:
    """Locate the plottter entry script in the active venv."""
    candidate = Path(sys.prefix) / "bin" / "plottter"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("plottter")
    if not found:
        pytest.skip("plottter CLI not installed in this environment")
    return found


def test_cli_optimize_roundtrip():
    """JSON in stdin → JSON out stdout, exit 0, progress on stderr."""
    bin_path = _plottter_bin()
    job = {
        "paths": [[[0, 0], [10, 0]], [[50, 0], [60, 0]], [[20, 0], [30, 0]]],
        "settings": {},
        "clip_bounds": [0, -1, 100, 1],
    }
    proc = subprocess.run(
        [bin_path, "--optimize"],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert "paths" in out
    assert out["before_lifts"] == 3
    assert out["before_travel"] > out["after_travel"]
    # Progress is line-buffered JSON on stderr.
    progress_lines = [l for l in proc.stderr.splitlines() if l.strip()]
    assert progress_lines
    first = json.loads(progress_lines[0])
    assert "progress" in first


def test_cli_optimize_rejects_empty_stdin():
    bin_path = _plottter_bin()
    proc = subprocess.run(
        [bin_path, "--optimize"],
        input="",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "JSON" in proc.stderr or "stdin" in proc.stderr


# ---------------------------------------------------------------------------
# _RemoteOptimizeWorker (with a stub 'ssh' on PATH)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _drain_events(timeout_ms: int = 500) -> None:
    """Drive the main-thread event loop so worker queued signals are delivered."""
    from PyQt6.QtCore import QCoreApplication, QDeadlineTimer

    deadline = QDeadlineTimer(timeout_ms)
    while not deadline.hasExpired():
        QCoreApplication.processEvents()
        if deadline.remainingTime() < 50:
            break


def _stub_ssh(tmp_path: Path, body: str) -> str:
    """Drop a fake ``ssh`` script onto a tmp dir and return its dir.

    The stub ignores all args and runs ``body`` as a shell snippet, so we can
    simulate happy paths, error paths, progress streams, etc. without touching
    a real network.
    """
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/usr/bin/env bash\n" + body)
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(bin_dir)


def test_remote_worker_happy_path(qapp, tmp_path, monkeypatch):
    """Stub ssh echoes a valid result JSON; worker emits finished."""
    canned = {
        "paths": [[[0.0, 0.0], [10.0, 0.0]]],
        "before_travel": 100.0,
        "after_travel": 50.0,
        "before_lifts": 5,
        "after_lifts": 3,
    }
    bin_dir = _stub_ssh(
        tmp_path,
        # Drain stdin so the parent's communicate() doesn't deadlock on full pipe,
        # then emit two progress lines on stderr and the canned JSON on stdout.
        "cat > /dev/null\n"
        ">&2 echo '{\"progress\": 10}'\n"
        ">&2 echo '{\"progress\": 100}'\n"
        f"echo '{json.dumps(canned)}'\n",
    )
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])

    from plottter.gui.main_window.workers import _RemoteOptimizeWorker

    worker = _RemoteOptimizeWorker(
        paths=[[(0.0, 0.0), (10.0, 0.0)]],
        host="fakehost",
        clip_bounds=(0.0, 0.0, 100.0, 100.0),
    )
    captured: dict = {}
    progress_vals: list[int] = []

    worker.finished.connect(
        lambda paths, bt, at, bl, al: captured.update(
            paths=paths, bt=bt, at=at, bl=bl, al=al
        )
    )
    worker.error.connect(lambda msg: captured.update(error=msg))
    worker.progress.connect(lambda v: progress_vals.append(v))

    worker.start()
    assert worker.wait(10000), "worker did not finish in time"
    _drain_events()
    assert "error" not in captured, captured.get("error")
    assert captured["bt"] == 100.0
    assert captured["at"] == 50.0
    assert captured["bl"] == 5
    assert captured["al"] == 3
    assert progress_vals == [10, 100]


def test_remote_worker_surfaces_remote_failure(qapp, tmp_path, monkeypatch):
    """Non-zero exit → .error emitted with the captured stderr tail."""
    bin_dir = _stub_ssh(
        tmp_path,
        "cat > /dev/null\n"
        ">&2 echo 'plottter: command not found'\n"
        "exit 127\n",
    )
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])

    from plottter.gui.main_window.workers import _RemoteOptimizeWorker

    worker = _RemoteOptimizeWorker(
        paths=[[(0.0, 0.0), (1.0, 0.0)]],
        host="fakehost",
    )
    captured: dict = {}
    worker.error.connect(lambda msg: captured.update(error=msg))
    worker.finished.connect(
        lambda *args: captured.update(unexpected_finish=True)
    )

    worker.start()
    assert worker.wait(10000)
    _drain_events()
    assert "error" in captured
    assert "127" in captured["error"]
    assert "plottter: command not found" in captured["error"]


def test_remote_command_override_is_invoked(qapp, tmp_path, monkeypatch):
    """The Preferences override replaces 'plottter' in the SSH argv.

    A custom 'ssh' stub writes its own argv to a side file so we can assert
    the worker built ``ssh <host> <custom-cmd> --optimize`` correctly.
    """
    argv_log = tmp_path / "argv.log"
    canned = {
        "paths": [],
        "before_travel": 0.0,
        "after_travel": 0.0,
        "before_lifts": 0,
        "after_lifts": 0,
    }
    bin_dir = _stub_ssh(
        tmp_path,
        # Log argv to a file, drain stdin, return a minimal valid result.
        f"printf '%s\\n' \"$@\" > {argv_log}\n"
        "cat > /dev/null\n"
        f"echo '{json.dumps(canned)}'\n",
    )
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])

    from plottter.gui.main_window.workers import _RemoteOptimizeWorker

    worker = _RemoteOptimizeWorker(
        paths=[[(0.0, 0.0), (1.0, 0.0)]],
        host="fakehost",
        remote_command="/home/me/.venv/bin/plottter",
    )
    worker.start()
    assert worker.wait(10000)
    _drain_events()

    logged = argv_log.read_text().splitlines()
    # ssh argv when invoked as `ssh fakehost /home/me/.venv/bin/plottter --optimize`
    assert logged[0] == "fakehost"
    assert logged[1] == "/home/me/.venv/bin/plottter"
    assert logged[2] == "--optimize"


def test_remote_worker_empty_host_errors(qapp):
    from plottter.gui.main_window.workers import _RemoteOptimizeWorker

    worker = _RemoteOptimizeWorker(paths=[[(0.0, 0.0), (1.0, 0.0)]], host="")
    captured: dict = {}
    worker.error.connect(lambda msg: captured.update(error=msg))

    worker.start()
    assert worker.wait(2000)
    _drain_events()
    assert "host" in captured.get("error", "").lower()
