"""End-to-end tests for NetworkTransport against the real fake daemon.

Boots scripts/fake_plot_daemon.py in-process on an OS-assigned port and drives
NetworkTransport against it — exercising the actual HTTP contract (no mocks).
"""

from __future__ import annotations

import importlib.util
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from plottter.export.transport import (
    ConnectionStatus,
    NetworkTransport,
    RemotePlotterError,
)

_FAKE = Path(__file__).resolve().parent / "fakes" / "fake_plot_daemon.py"
SVG1 = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'


def _load_daemon_module():
    spec = importlib.util.spec_from_file_location("fake_plot_daemon", _FAKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def daemon():
    """Boot the fake daemon in-process. Yields (base_url, token)."""
    mod = _load_daemon_module()
    token = "tok123"
    mod._Handler.state = mod._State(layer_seconds=0.4)  # fast plots
    mod._Handler.token = token
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod._Handler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}", token
    finally:
        server.shutdown()
        server.server_close()


def test_health_connected(daemon):
    url, token = daemon
    status = NetworkTransport(url, token).health()
    assert isinstance(status, ConnectionStatus)
    assert status.connected is True
    assert status.source == "network"
    assert "FAKE" in status.device

    # health works WITHOUT a token (open endpoint)
    assert NetworkTransport(url, token=None).health().connected is True


def test_health_unreachable():
    status = NetworkTransport("http://127.0.0.1:1").health()  # nothing listening
    assert status.connected is False
    assert status.source == "network"


def test_run_manual(daemon):
    url, token = daemon
    NetworkTransport(url, token).run_manual("lower_pen", {"pen_pos_down": 30})  # no raise


def test_run_manual_bad_token(daemon):
    url, _ = daemon
    with pytest.raises(RemotePlotterError):
        NetworkTransport(url, token="wrong").run_manual("lower_pen", {})


def test_plot_completes_with_progress(daemon):
    url, token = daemon
    t = NetworkTransport(url, token, poll_interval=0.1)
    seen: list[float] = []
    outcome = t.plot_svg(SVG1, {}, progress_callback=seen.append)
    assert outcome.paused is False
    assert outcome.resume_svg is None
    assert seen and seen[-1] == 100.0
    assert seen == sorted(seen)  # monotonic non-decreasing


def test_busy_rejects_second_job(daemon):
    url, token = daemon
    t = NetworkTransport(url, token, poll_interval=0.1)
    result: dict = {}
    th = threading.Thread(target=lambda: result.setdefault("o", t.plot_svg(SVG1, {})))
    th.start()
    try:
        # wait until the daemon reports busy
        for _ in range(50):
            if t.health().busy:
                break
            time.sleep(0.05)
        with pytest.raises(RemotePlotterError):
            t.plot_svg(SVG1, {})       # second job -> 409
        with pytest.raises(RemotePlotterError):
            t.run_manual("raise_pen", {})  # manual while busy -> 409
    finally:
        th.join(timeout=5)


def test_pause_then_resume_round_trip(daemon):
    url, token = daemon
    t = NetworkTransport(url, token, poll_interval=0.1)

    handle_box: dict = {}

    def grab(handle):
        handle_box["h"] = handle

    # Start a plot; pause it shortly after via the duck-typed handle.
    def pause_soon():
        for _ in range(50):
            if "h" in handle_box:
                time.sleep(0.15)
                handle_box["h"].transmit_pause_request()
                return
            time.sleep(0.02)

    pause_thread = threading.Thread(target=pause_soon)
    pause_thread.start()
    outcome = t.plot_svg(SVG1, {}, on_ready=grab)
    pause_thread.join(timeout=5)

    assert outcome.paused is True
    assert outcome.resume_svg  # the job-id resume token (non-empty)

    # Resume from the token -> should run to completion.
    resumed = t.plot_svg(outcome.resume_svg, {}, resume=True)
    assert resumed.paused is False
    assert resumed.resume_svg is None
