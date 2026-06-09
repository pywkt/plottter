"""Remote-plotter wiring (specs/remote-plotter.md §5.2/§5.3).

The remote device is *configured* in Preferences → Remote Plotter and *consumed*
by the AxiDraw dialog, which selects its transport from the saved settings and
shows a read-only connection indicator. These tests cover both halves plus the
fake daemon for the "connected" case.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings

from plottter.models import Canvas, Layer, Project

_FAKE = Path(__file__).resolve().parent / "fakes" / "fake_plot_daemon.py"

_ENABLED_KEY = "remote_plotter/enabled"
_URL_KEY = "remote_plotter/url"
_TOKEN_KEY = "remote_plotter/token"


def _make_project() -> Project:
    proj = Project(name="Remote", canvas=Canvas.from_preset("A4"))
    layer = Layer(name="L0", color="#000000")
    layer.paths = [[(0.0, 0.0), (10.0, 10.0)]]
    proj.add_layer(layer)
    return proj


@pytest.fixture(autouse=True)
def _clear_remote_settings():
    """Start each test from no configured remote device (sandbox is session-wide)."""
    s = QSettings("Plottter", "Plottter")
    for k in (_ENABLED_KEY, _URL_KEY, _TOKEN_KEY):
        s.remove(k)
    s.sync()
    yield
    for k in (_ENABLED_KEY, _URL_KEY, _TOKEN_KEY):
        s.remove(k)
    s.sync()


@pytest.fixture
def daemon():
    spec = importlib.util.spec_from_file_location("fake_plot_daemon", _FAKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod._Handler.state = mod._State(layer_seconds=0.4)
    mod._Handler.token = "tok"
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod._Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}", "tok"
    finally:
        server.shutdown()
        server.server_close()


def _set_remote(url: str, token: str) -> None:
    s = QSettings("Plottter", "Plottter")
    s.setValue(_ENABLED_KEY, True)
    s.setValue(_URL_KEY, url)
    s.setValue(_TOKEN_KEY, token)
    s.sync()


# ---------------------------------------------------------------------------
# AxiDraw dialog: selects transport from saved settings + shows status
# ---------------------------------------------------------------------------

def test_defaults_to_usb_transport(qtbot):
    from plottter.export.transport import LocalUsbTransport
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    assert isinstance(dlg._transport, LocalUsbTransport)


def test_reads_network_transport_from_settings(qtbot, daemon):
    from plottter.export.transport import NetworkTransport
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    url, token = daemon
    _set_remote(url, token)

    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)

    assert isinstance(dlg._transport, NetworkTransport)
    assert dlg._transport.base_url == url
    # Read-only indicator reports the reachable network device.
    assert "via network" in dlg._status_label.text()
    assert "Remote Plotter" in dlg._remote_hint_label.text()


def test_refresh_picks_up_settings_change(qtbot, daemon):
    """Editing Preferences then hitting Refresh swaps the transport live."""
    from plottter.export.transport import LocalUsbTransport, NetworkTransport
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    url, token = daemon
    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    assert isinstance(dlg._transport, LocalUsbTransport)

    _set_remote(url, token)
    dlg._on_refresh_clicked()
    assert isinstance(dlg._transport, NetworkTransport)
    assert dlg._transport.base_url == url

    # Disabling again returns to USB on the next refresh.
    s = QSettings("Plottter", "Plottter")
    s.setValue(_ENABLED_KEY, False)
    s.sync()
    dlg._on_refresh_clicked()
    assert isinstance(dlg._transport, LocalUsbTransport)


# ---------------------------------------------------------------------------
# Preferences dialog: persists + restores the remote-plotter config
# ---------------------------------------------------------------------------

def test_preferences_persists_remote_plotter(qtbot):
    from plottter.gui.dialogs.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg._remote_plotter_url.setText("http://example.local:8080")
    dlg._remote_plotter_token.setText("secret")
    dlg._remote_plotter_enabled.setChecked(True)
    dlg._save_settings()

    s = QSettings("Plottter", "Plottter")
    assert s.value(_ENABLED_KEY, type=bool) is True
    assert s.value(_URL_KEY) == "http://example.local:8080"
    assert s.value(_TOKEN_KEY) == "secret"

    # A freshly-opened Preferences dialog restores the saved values.
    dlg2 = PreferencesDialog()
    qtbot.addWidget(dlg2)
    assert dlg2._remote_plotter_enabled.isChecked() is True
    assert dlg2._remote_plotter_url.text() == "http://example.local:8080"
    assert dlg2._remote_plotter_token.text() == "secret"


def test_preferences_test_connection_reports_device(qtbot, daemon):
    from plottter.gui.dialogs.preferences import PreferencesDialog

    url, token = daemon
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg._remote_plotter_url.setText(url)
    dlg._remote_plotter_token.setText(token)

    dlg._on_test_remote_plotter()
    assert dlg._remote_plotter_status.text().startswith("✓")
    assert "network" in dlg._remote_plotter_status.text()


# ---------------------------------------------------------------------------
# Reconnect / recovery: a paused job on the device must be adoptable + stoppable
# ---------------------------------------------------------------------------

def _pause_a_remote_job(url, token):
    """Submit a job to the daemon and pause it; return its resume token (job id)."""
    from plottter.export.transport import NetworkTransport

    t = NetworkTransport(url, token, poll_interval=0.1)
    box = {}

    def grab(handle):
        box["h"] = handle

    def pause_soon():
        for _ in range(100):
            if "h" in box:
                time.sleep(0.15)
                box["h"].transmit_pause_request()
                return
            time.sleep(0.02)

    th = threading.Thread(target=pause_soon)
    th.start()
    outcome = t.plot_svg('<svg xmlns="http://www.w3.org/2000/svg"></svg>', {}, on_ready=grab)
    th.join(timeout=5)
    assert outcome.paused and outcome.resume_svg
    return outcome.resume_svg


def test_dialog_adopts_and_stops_paused_remote_job(qtbot, daemon):
    """Opening the dialog with a paused job on the device adopts it; Stop frees it.

    This is the fix for the 'plotter busy' lockup: a job left paused (e.g. after
    a dropped Wi-Fi connection) is discovered on open and can be cancelled
    without restarting the daemon.
    """
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    url, token = daemon
    job_id = _pause_a_remote_job(url, token)
    _set_remote(url, token)

    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    # Reconcile on open adopted the paused job.
    assert dlg._resume_svg == job_id
    assert dlg._stop_btn.isEnabled() is True

    # Stop it -> daemon clears the job, dialog returns to idle.
    dlg._on_stop()
    for _ in range(60):
        if dlg._transport.active_job() is None:
            break
        time.sleep(0.05)
    assert dlg._transport.active_job() is None
    assert dlg._resume_svg is None
    assert dlg._plot_btn.isEnabled() is True
