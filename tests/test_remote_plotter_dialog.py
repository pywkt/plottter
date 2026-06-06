"""AxiDrawDialog remote-device wiring (specs/remote-plotter.md §5.2/§5.3).

Verifies the dialog selects the right transport from the remote-device settings
and persists them. Boots the fake daemon in-process for the "connected" case.
"""

from __future__ import annotations

import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from plottter.models import Canvas, Layer, Project

_FAKE = Path(__file__).resolve().parent / "fakes" / "fake_plot_daemon.py"


def _make_project() -> Project:
    proj = Project(name="Remote", canvas=Canvas.from_preset("A4"))
    layer = Layer(name="L0", color="#000000")
    layer.paths = [[(0.0, 0.0), (10.0, 10.0)]]
    proj.add_layer(layer)
    return proj


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


def test_defaults_to_usb_transport(qtbot):
    from plottter.export.transport import LocalUsbTransport
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    assert isinstance(dlg._transport, LocalUsbTransport)
    assert dlg._remote_enabled_check.isChecked() is False


def test_enabling_remote_selects_network_transport(qtbot, daemon):
    from plottter.export.transport import LocalUsbTransport, NetworkTransport
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    url, token = daemon
    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    assert isinstance(dlg._transport, LocalUsbTransport)

    dlg._remote_url_edit.setText(url)
    dlg._remote_token_edit.setText(token)
    dlg._remote_enabled_check.setChecked(True)  # toggled -> _on_remote_settings_changed

    assert isinstance(dlg._transport, NetworkTransport)
    assert dlg._transport.base_url == url
    # status indicator reflects the reachable network device
    assert "(network)" in dlg._status_label.text()

    # turning it back off returns to USB
    dlg._remote_enabled_check.setChecked(False)
    assert isinstance(dlg._transport, LocalUsbTransport)


def test_remote_settings_persisted(qtbot):
    from PyQt6.QtCore import QSettings
    from plottter.gui.dialogs.axidraw_dialog import AxiDrawDialog

    dlg = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg)
    dlg._remote_url_edit.setText("http://example.local:8080")
    dlg._remote_token_edit.setText("secret")
    dlg._remote_enabled_check.setChecked(True)

    s = QSettings("Plottter", "Plottter")
    assert s.value(AxiDrawDialog._REMOTE_ENABLED_KEY, type=bool) is True
    assert s.value(AxiDrawDialog._REMOTE_URL_KEY) == "http://example.local:8080"
    assert s.value(AxiDrawDialog._REMOTE_TOKEN_KEY) == "secret"

    # a freshly-opened dialog restores them
    dlg2 = AxiDrawDialog(_make_project())
    qtbot.addWidget(dlg2)
    assert dlg2._remote_enabled_check.isChecked() is True
    assert dlg2._remote_url_edit.text() == "http://example.local:8080"
