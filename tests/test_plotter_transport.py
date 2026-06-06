"""Tests for the PlotterTransport abstraction (specs/remote-plotter.md §5)."""

from __future__ import annotations

import pytest

from plottter.export.transport import (
    ConnectionStatus,
    LocalUsbTransport,
    PlotterTransport,
)


def test_transport_is_abstract():
    with pytest.raises(TypeError):
        PlotterTransport()  # type: ignore[abstract]


class TestLocalUsbTransport:
    def test_is_available_forwards(self, monkeypatch):
        import plottter.export.axidraw as ax
        monkeypatch.setattr(ax, "check_axidraw_available", lambda: True)
        assert LocalUsbTransport().is_available() is True
        monkeypatch.setattr(ax, "check_axidraw_available", lambda: False)
        assert LocalUsbTransport().is_available() is False

    def test_health_connected(self, monkeypatch):
        import plottter.export.axidraw as ax
        monkeypatch.setattr(ax, "check_axidraw_available", lambda: True)
        status = LocalUsbTransport().health()
        assert isinstance(status, ConnectionStatus)
        assert status.connected is True
        assert status.source == "usb"
        assert "pyaxidraw is installed" in status.detail

    def test_health_disconnected(self, monkeypatch):
        import plottter.export.axidraw as ax
        monkeypatch.setattr(ax, "check_axidraw_available", lambda: False)
        status = LocalUsbTransport().health()
        assert status.connected is False
        assert status.source == "usb"
        assert "pip install" in status.detail

    def test_plot_svg_forwards(self, monkeypatch):
        captured = {}

        def fake_plot(svg, settings, progress=None, on_ready=None, resume=False):
            captured.update(
                svg=svg, settings=settings, progress=progress,
                on_ready=on_ready, resume=resume,
            )
            return "OUTCOME"

        import plottter.export.axidraw as ax
        monkeypatch.setattr(ax, "plot_svg_string", fake_plot)

        cb = lambda p: None
        ready = lambda ad: None
        out = LocalUsbTransport().plot_svg(
            "<svg/>", {"model": 6}, cb, on_ready=ready, resume=True
        )
        assert out == "OUTCOME"
        assert captured["svg"] == "<svg/>"
        assert captured["settings"] == {"model": 6}
        assert captured["progress"] is cb
        assert captured["on_ready"] is ready   # pause-handle contract preserved
        assert captured["resume"] is True

    def test_run_manual_forwards(self, monkeypatch):
        captured = {}

        def fake_manual(command, settings):
            captured.update(command=command, settings=settings)

        import plottter.export.axidraw as ax
        monkeypatch.setattr(ax, "run_manual_command", fake_manual)

        LocalUsbTransport().run_manual("lower_pen", {"pen_pos_down": 30})
        assert captured["command"] == "lower_pen"
        assert captured["settings"] == {"pen_pos_down": 30}
