"""Plotter transports — abstract the *where* of the plotter from the dialog.

The "Plot with AxiDraw" dialog drives the plotter through a small interface so the
same dialog code works whether the plotter is connected to this machine via USB or
lives on a networked device (a Raspberry Pi running the plot daemon). See
``specs/remote-plotter.md``.

Two backends:

* :class:`LocalUsbTransport` — the current behaviour: drives the USB-connected
  plotter via ``pyaxidraw`` (``export/axidraw.py``).
* ``NetworkTransport`` — added later; talks HTTP to the plot daemon.

The interface deliberately mirrors ``plot_svg_string`` / ``run_manual_command`` so the
USB path is a thin pass-through with **no behaviour change**.

**Pause contract (forward-compatible):** ``plot_svg`` takes an ``on_ready`` callback
that is invoked with a *driver handle* once plotting is configured. The handle exposes
``transmit_pause_request()``. For USB this handle is the live ``pyaxidraw.AxiDraw``
object; for the future network transport it will be a small object whose
``transmit_pause_request()`` POSTs a pause control to the daemon. Because callers only
ever call ``transmit_pause_request()`` on the handle, the GUI workers need no changes
to support the network transport.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ConnectionStatus:
    """Snapshot of a transport's connection, for the dialog's status indicator."""

    connected: bool
    device: str = ""
    detail: str = ""           # human-readable line shown under the status label
    source: str = "none"       # "usb" | "network" | "none"
    busy: bool = False         # a plot is currently running (network only for now)


@dataclass
class ActiveJob:
    """A job already in progress on the device, for reconnect / reconcile.

    Surfaced by :meth:`PlotterTransport.active_job` so the dialog can adopt a
    plot that's already running (or left paused) on a networked device instead
    of starting a fresh one that would be rejected as busy.
    """

    job_id: str
    state: str  # "plotting" | "paused"


class PlotterTransport(ABC):
    """Where the plotter lives + how to drive it. Implemented by USB and network."""

    @abstractmethod
    def is_available(self) -> bool:
        """True if this transport can currently reach a plotter."""

    @abstractmethod
    def health(self) -> ConnectionStatus:
        """Connection snapshot for the status indicator."""

    @abstractmethod
    def plot_svg(
        self,
        svg_data: str,
        settings: dict,
        progress_callback: Optional[Callable[[float], None]] = None,
        on_ready: Optional[Callable[[Any], None]] = None,
        resume: bool = False,
    ) -> Any:
        """Plot an SVG. Returns a ``PlotOutcome`` (paused, resume_svg).

        ``on_ready`` is called with a driver handle exposing
        ``transmit_pause_request()`` once plotting is configured (see module
        docstring's pause contract).
        """

    @abstractmethod
    def run_manual(self, command: str, settings: dict) -> None:
        """Run a one-off manual command (raise_pen / lower_pen / disable_xy / …)."""

    def cancel_job(self, job_token: Optional[str]) -> None:
        """Abandon a paused / in-flight job so the device frees up.

        No-op for USB (a paused USB plot is just a local resume SVG that the
        caller drops). The network transport tells the daemon to stop the job so
        it doesn't stay busy until a restart.
        """
        return None

    def active_job(self) -> "Optional[ActiveJob]":
        """Return a job already running on the device, or ``None``.

        Only meaningful for the network transport; USB has no out-of-band job
        the dialog could reconnect to, so it always returns ``None``.
        """
        return None


class LocalUsbTransport(PlotterTransport):
    """Drive a USB-connected plotter via ``pyaxidraw`` — today's behaviour, unchanged."""

    # Imports are lazy so importing this module never requires pyaxidraw.

    def is_available(self) -> bool:
        from plottter.export.axidraw import check_axidraw_available

        return check_axidraw_available()

    def health(self) -> ConnectionStatus:
        if self.is_available():
            return ConnectionStatus(
                connected=True,
                device="USB plotter",
                source="usb",
                detail=(
                    "pyaxidraw is installed. Connect your AxiDraw via USB and "
                    "click Plot Now."
                ),
            )
        return ConnectionStatus(
            connected=False,
            source="usb",
            detail=(
                "pyaxidraw is NOT installed.\n"
                "Install it with:\n"
                "  pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip\n"
                "You can still use Preview mode to test settings without a device."
            ),
        )

    def plot_svg(
        self,
        svg_data: str,
        settings: dict,
        progress_callback: Optional[Callable[[float], None]] = None,
        on_ready: Optional[Callable[[Any], None]] = None,
        resume: bool = False,
    ) -> Any:
        from plottter.export.axidraw import plot_svg_string

        return plot_svg_string(
            svg_data,
            settings,
            progress_callback,
            on_ready=on_ready,
            resume=resume,
        )

    def run_manual(self, command: str, settings: dict) -> None:
        from plottter.export.axidraw import run_manual_command

        run_manual_command(command, settings)


# ---------------------------------------------------------------------------
# Network transport — drives a plotter via the remote plot daemon (spec §6)
# ---------------------------------------------------------------------------


class RemotePlotterError(RuntimeError):
    """Raised for daemon-side / network failures, with a user-facing message."""


class _NetworkPlotHandle:
    """Duck-typed pause handle handed to ``on_ready`` for a network plot.

    Mirrors the one method the GUI workers call on the USB ``AxiDraw`` object,
    so the workers need no changes: ``pause()`` POSTs a pause control to the
    daemon for this job.
    """

    def __init__(self, transport: "NetworkTransport", job_id: str) -> None:
        self._transport = transport
        self._job_id = job_id

    def transmit_pause_request(self) -> None:
        try:
            self._transport._control(self._job_id, "pause")
        except Exception:
            pass


class NetworkTransport(PlotterTransport):
    """Drive a plotter on a remote host running the plot daemon (specs/remote-plotter.md).

    A plot is a single daemon job: ``plot_svg`` POSTs the job, then **polls**
    status (feeding ``progress_callback``) until the daemon reports done / paused
    / error, so it slots into the existing blocking worker model unchanged.

    Resume model: on pause the daemon keeps the job alive, so ``plot_svg``
    returns ``PlotOutcome(paused=True, resume_svg=<job_id>)`` — the job id is the
    opaque resume token. A subsequent ``plot_svg(..., resume=True)`` receives that
    token as ``svg_data`` and POSTs a *resume* control to the same job instead of
    starting a new one. (USB uses the resume SVG; network uses the job id — the
    dialog treats the token opaquely either way.)
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        poll_interval: float = 0.5,
        timeout: float = 10.0,
        reconnect_grace: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or None
        self.poll_interval = poll_interval
        self.timeout = timeout
        # How long to tolerate an unreachable daemon mid-plot before treating the
        # outage as a recoverable pause (the daemon keeps plotting on its own).
        self.reconnect_grace = reconnect_grace

    # -- low-level HTTP --

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _request(self, method: str, path: str, body: Optional[dict] = None, auth: bool = True):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if auth and self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {}
            return exc.code, payload
        except urllib.error.URLError as exc:
            raise RemotePlotterError(
                f"Could not reach the remote plotter at {self.base_url}: {exc.reason}"
            ) from exc

    def _control(self, job_id: str, action: str) -> None:
        status, data = self._request("POST", f"/jobs/{job_id}/control", {"action": action})
        if status != 200:
            raise RemotePlotterError(data.get("error", f"control '{action}' failed ({status})"))

    # -- PlotterTransport interface --

    def is_available(self) -> bool:
        return self.health().connected

    def health(self) -> ConnectionStatus:
        try:
            status, data = self._request("GET", "/health", auth=False)
        except Exception:
            return ConnectionStatus(
                connected=False,
                source="network",
                detail=f"No plotter daemon reachable at {self.base_url}",
            )
        if status == 200 and data.get("ok"):
            device = data.get("device", "remote plotter")
            busy = data.get("state") in ("plotting", "paused")
            return ConnectionStatus(
                connected=True,
                device=device,
                source="network",
                busy=busy,
                detail=f"Connected: {device} (network) — {self.base_url}",
            )
        return ConnectionStatus(
            connected=False,
            source="network",
            detail=f"Plotter daemon at {self.base_url} is not ready",
        )

    def run_manual(self, command: str, settings: dict) -> None:
        status, data = self._request("POST", "/manual", {"command": command, "settings": settings})
        if status == 409:
            raise RemotePlotterError("The remote plotter is busy with a plot.")
        if status != 200:
            raise RemotePlotterError(data.get("error", f"manual command failed ({status})"))

    def cancel_job(self, job_token: Optional[str]) -> None:
        """Stop a job on the daemon so the device is freed (best-effort)."""
        if not job_token:
            return
        try:
            self._control(job_token, "stop")
        except Exception:
            # Cancelling is best-effort: if the daemon is unreachable or the job
            # is already gone there's nothing left to free.
            pass

    def active_job(self) -> Optional[ActiveJob]:
        """Report a plot already running / paused on the device, via /health."""
        try:
            status, data = self._request("GET", "/health", auth=False)
        except Exception:
            return None
        if status == 200 and data.get("ok"):
            state = data.get("state")
            job_id = data.get("current_job")
            if state in ("plotting", "paused") and job_id:
                return ActiveJob(job_id=str(job_id), state=str(state))
        return None

    def plot_svg(
        self,
        svg_data: str,
        settings: dict,
        progress_callback: Optional[Callable[[float], None]] = None,
        on_ready: Optional[Callable[[Any], None]] = None,
        resume: bool = False,
    ) -> Any:
        from plottter.export.axidraw import PlotOutcome

        if resume:
            # svg_data is the resume token (= job id from a previous pause).
            job_id = svg_data
            self._control(job_id, "resume")
        else:
            status, data = self._request("POST", "/jobs", {"svg": svg_data, "settings": settings})
            if status == 409:
                raise RemotePlotterError("The remote plotter is busy with another job.")
            if status != 200:
                raise RemotePlotterError(data.get("error", f"plot request failed ({status})"))
            job_id = data["job_id"]

        if on_ready is not None:
            on_ready(_NetworkPlotHandle(self, job_id))

        # Poll until terminal. When resuming, ignore the transient pre-resume
        # "paused" state until we've seen the job actually running again.
        saw_active = not resume
        grace_deadline = time.monotonic() + 3.0 if resume else 0.0
        # The daemon owns the job and keeps plotting even if this client briefly
        # loses the network (the whole point of remote plotting). So a failed
        # status poll is tolerated for ``reconnect_grace`` seconds rather than
        # aborting the plot view; only a sustained outage gives up.
        unreachable_since: Optional[float] = None

        while True:
            try:
                status, data = self._request("GET", f"/jobs/{job_id}")
            except RemotePlotterError:
                now = time.monotonic()
                if unreachable_since is None:
                    unreachable_since = now
                if now - unreachable_since >= self.reconnect_grace:
                    # Surface the outage as a recoverable pause carrying the job
                    # token, so the dialog offers Resume (re-attach) / Stop
                    # instead of dead-ending and stranding the job.
                    return PlotOutcome(paused=True, resume_svg=job_id)
                time.sleep(self.poll_interval)
                continue
            unreachable_since = None

            if status != 200:
                raise RemotePlotterError(data.get("error", f"status query failed ({status})"))

            state = data.get("state")
            percent = data.get("percent")
            if progress_callback is not None and percent is not None:
                progress_callback(float(percent))

            if state == "plotting":
                saw_active = True
            elif state == "done":
                return PlotOutcome(paused=False, resume_svg=None)
            elif state == "error":
                raise RemotePlotterError(data.get("error") or "remote plot error")
            elif state == "stopped":
                return PlotOutcome(paused=True, resume_svg=None)
            elif state == "paused":
                if saw_active or time.monotonic() >= grace_deadline:
                    return PlotOutcome(paused=True, resume_svg=job_id)
                # else: transient pre-resume paused — keep waiting

            time.sleep(self.poll_interval)
